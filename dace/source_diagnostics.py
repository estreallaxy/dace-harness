"""Durable native diagnosis -> one budgeted source Coding Agent request.

This stage emits hypotheses and frozen code artifacts. It never deploys a
proposal or substitutes optimizer output for task verification.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from .benchmarks.fingerprint import task_fingerprint
from .core import Artifact, Ledger, digest
from .durable import PersistentBudget
from .runtime_lock import verify_runtime_lock
from .source_execution import SourceExecution
from .source_proposals import SourceCodingAgent,decode_proposal,source_graph
from .source_usage import summarize_source_usage
from .task_registry import TaskRegistry
from .coding_request import validate_coding_request


def make_diagnostic_plan(parent,tasks,dataset,command,model,study,runtime_lock,
                         mechanism_cap=4,action_cap=4,optimizer_model=None,history=None,coding_max_tokens=12000,
                         level=None,state_context=None,coding_request=None):
    if not isinstance(command,list):
        raise ValueError('Diagnostic runner command must be an argv list')
    tasks = list(tasks)
    plan = {'schema':'dace-source-diagnostics/v1','parent':asdict(parent),'tasks':tasks,
        'dataset':str(Path(dataset).resolve()),'command':list(command),'model':model,
        'study':study,'runtime_lock':runtime_lock,'optimizer_model':optimizer_model or model,
        'mechanism_cap':mechanism_cap,'action_cap':action_cap,'history':[] if history is None else history,
        'coding_max_tokens':coding_max_tokens,
        'total_budget':len(tasks)+10,
        'task_fingerprints':{t:task_fingerprint(Path(dataset)/t) for t in tasks}}
    if level is not None:
        from .components import AblationLevel
        plan.update(schema='dace-source-diagnostics/v2',level=AblationLevel(level).value,
                    state_context={} if state_context is None else state_context)
    if coding_request is not None:
        plan['coding_request'] = validate_coding_request(coding_request,coding_max_tokens)
    validate_diagnostic_plan(plan)
    return plan


def validate_diagnostic_plan(plan):
    if plan.get('schema') not in ('dace-source-diagnostics/v1','dace-source-diagnostics/v2'):
        raise ValueError('Unknown diagnostic/proposal protocol')
    if plan['schema'].endswith('/v2'):
        from .components import AblationLevel
        AblationLevel(plan['level'])
        if not isinstance(plan['state_context'],dict):
            raise ValueError('Native controller state must be a frozen projection')
    parent = Artifact(**plan['parent'])
    source_graph(parent)
    if (not plan['tasks'] or len(set(plan['tasks']))!=len(plan['tasks'])
            or set(plan['task_fingerprints'])!=set(plan['tasks'])
            or type(plan['total_budget']) is not int or plan['total_budget']!=len(plan['tasks'])+10
            or any(type(plan[x]) is not int or plan[x]<=0 for x in ('action_cap','mechanism_cap'))):
        raise ValueError('Freeze exact diagnostic tasks, search caps and request budget')
    if type(plan['coding_max_tokens']) is not int or not 1<=plan['coding_max_tokens']<=65536:
        raise ValueError('Coding output-token cap must be frozen and bounded')
    validate_coding_request(plan.get('coding_request'),plan['coding_max_tokens'])
    if (not isinstance(plan['command'],list) or not plan['command']
            or any(not isinstance(v,str) or not v for v in plan['command'])
            or not plan['model'] or not plan['optimizer_model'] or not isinstance(plan['history'],list)):
        raise ValueError('Diagnostic and optimizer runtimes must be explicit')
    study = plan['study']
    if (not isinstance(study,dict) or set(study)!={'registry','id','lineage'}
            or any(not isinstance(v,str) or not v for v in study.values())
            or not Path(study['registry']).is_absolute()):
        raise ValueError('Diagnostic data requires an explicit shared study/lineage')
    root = Path(plan['dataset']).resolve()
    for task in plan['tasks']:
        path = (root/task).resolve()
        if path.parent!=root or path.name!=task or task_fingerprint(path)!=plan['task_fingerprints'][task]:
            raise ValueError('Frozen diagnostic task inputs changed')
    verify_runtime_lock(plan['runtime_lock'],plan['command'],plan['dataset'],plan['tasks'])
    return parent


def _excerpt(result,output):
    reward = Path(result.verifier_path or '').resolve()
    if not reward.is_relative_to(Path(output).resolve()/'executions'):
        raise ValueError('Diagnostic raw trajectory must belong to this exact execution directory')
    path = reward.parent.parent/'agent/trajectory.json'
    if not path.exists():
        return ''  # Missing native text stays missing; boundaries remain measured facts.
    if not path.resolve().is_relative_to(Path(output).resolve()/'executions'):
        raise ValueError('Diagnostic trajectory escaped its execution directory')
    # Never read solutions or verifier tests. Only the agent's own trajectory.
    with path.open('rb') as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-96000))
        return stream.read().decode('utf-8',errors='replace')[-24000:]


def _agent_usage(events,coding_owner):
    charged = sum(e['kind']=='BUDGET_CHARGED' and e['payload'].get('owner')==coding_owner for e in events)
    responses = [e['payload']['usage'] for e in events if e['kind']=='SOURCE_AGENT_RESPONSE']
    def values(field):
        return [u[field] for u in responses if type(u.get(field)) in (int,float)
                and math.isfinite(u[field]) and u[field]>=0]
    complete = (len(responses)==charged and all(len(values(f))==len(responses)
                for f in ('cost','prompt_tokens','completion_tokens')))
    return {'request_attempts':charged,'responses':len(responses),'accounting_complete':complete,
        'cost_usd':sum(values('cost')) if complete else None,
        'recorded_cost_usd':sum(values('cost')),
        'recorded_prompt_tokens':sum(values('prompt_tokens')),
        'recorded_completion_tokens':sum(values('completion_tokens'))}


def run_diagnostic_proposal(plan,output,agent=None,batch=None):
    plan = json.loads(json.dumps(plan))
    parent = validate_diagnostic_plan(plan)
    if batch is not None and (type(batch) is not int or batch<=0):
        raise ValueError('Diagnostic batch must be positive')
    output = Path(output).resolve()
    identity,study = digest(plan),plan['study']
    diag_owner,coding_owner = 'diag:'+identity,'coding:'+identity
    with TaskRegistry(study['registry']) as registry:
        registry.claim(study['id'],study['lineage'],diag_owner,'DIAG',plan['tasks'],
                       plan['task_fingerprints'],str(output),identity)
    ledger = Ledger(output/'evidence.jsonl',durable=True)
    try:
        budget = PersistentBudget(plan['total_budget'],ledger)
        freezes = [e['payload']['plan'] for e in ledger.events if e['kind']=='DIAGNOSTIC_PLAN_FROZEN']
        if freezes:
            if len(freezes)!=1 or digest(freezes[0])!=identity:
                raise ValueError('Diagnostic resume must retain the exact original protocol')
        elif len(ledger.events)==1:
            ledger.append('DIAGNOSTIC_PLAN_FROZEN',plan=plan)
        else:
            raise ValueError('Unidentified journal cannot become a diagnostic session')
        results = [e['payload']['result'] for e in ledger.events if e['kind']=='DIAGNOSTIC_STAGE_RESULT']
        if len(results)>1:
            raise ValueError('Repeated final diagnostic decision')
        def publish(result):
            result = {**result,'kind':'SOURCE_DIAGNOSTIC_PROPOSAL','plan_hash':identity,
                'execution_usage':summarize_source_usage([ledger.path]),
                'optimizer_usage':_agent_usage(ledger.events,coding_owner),
                'budget':{'total':budget.total,'spent':budget.spent,'locks':dict(budget.locks)}}
            (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            return result
        if results:
            budget.release(diag_owner)
            budget.release(coding_owner)
            return publish(results[0])
        def finish(status,**data):
            result = {'status':status,**data}
            ledger.append('DIAGNOSTIC_STAGE_RESULT',result=result)
            budget.release(diag_owner)
            budget.release(coding_owner)
            return publish(result)
        reserved = {e['payload']['owner'] for e in ledger.events if e['kind']=='BUDGET_RESERVED'}
        if diag_owner not in reserved:
            budget.reserve(diag_owner,len(plan['tasks']))
        if coding_owner not in reserved:
            budget.reserve(coding_owner,10)
        observed = {}
        for e in ledger.events:
            if e['kind']=='DIAGNOSTIC_OBSERVATION':
                p = e['payload']
                if p['owner']!=diag_owner or p['task'] not in plan['tasks'] or p['task'] in observed:
                    raise ValueError('Diagnostic observations are not unique original units')
                observed[p['task']] = p['record']
        attempted = {u['task'] for u in budget.attempts.values() if u['contract_id']==diag_owner}
        if not set(observed)<=attempted or not attempted<=set(plan['tasks']):
            raise ValueError('Diagnostic charges and committed observations disagree')
        if attempted-set(observed):
            return finish('DIAGNOSTIC_UNRESOLVED',reason='INTERRUPTED_DIAGNOSTIC_WITHOUT_COMMITTED_OBSERVATION')
        remaining = [t for t in plan['tasks'] if t not in observed]
        if budget.locks.get(diag_owner,0)!=len(remaining):
            raise ValueError('Diagnostic session lost its reserved execution budget')
        executor = SourceExecution(plan,ledger,output,{'diagnostic_id':identity})
        for task in remaining[:batch] if batch is not None else remaining:
            budget.charge_confirmation(1,diag_owner,task)
            try:
                with TaskRegistry(study['registry']) as registry:
                    registry.mark_started(study['id'],study['lineage'],diag_owner,task)
                result = executor.execute(parent,task)
                record = {'task_id':task,'boundary':result.boundary,'utility':result.utility,
                          'trace_excerpt':_excerpt(result,output)}
                ledger.append('DIAGNOSTIC_OBSERVATION',owner=diag_owner,task=task,record=record)
                observed[task] = record
            except Exception as exc:
                return finish('DIAGNOSTIC_UNRESOLVED',reason=type(exc).__name__)
        if len(observed)!=len(plan['tasks']):
            return publish({'status':'PENDING_DIAGNOSTICS',
                            'remaining':[t for t in plan['tasks'] if t not in observed]})
        proposals = [e['payload'] for e in ledger.events if e['kind']=='SOURCE_AGENT_PROPOSAL']
        if proposals:
            if len(proposals)!=1:
                raise ValueError('Repeated coding proposal in one request allocation')
            item = proposals[0]
            compiled = decode_proposal(item['payload'],parent,plan['mechanism_cap'],plan['action_cap'],plan.get('level')!='C0')
            if digest(asdict(compiled))!=digest(item['proposal']):
                raise ValueError('Coding proposal no longer matches its frozen source artifacts')
            return finish('PROPOSAL_READY' if compiled.actions else 'NO_CHANGE_PROPOSED',proposal=asdict(compiled),
                          ranking=[{'action':a.id,'disagreement':n} for a,n in compiled.ranked_actions()])
        if any(e['kind']=='BUDGET_CHARGED' and e['payload'].get('owner')==coding_owner for e in ledger.events):
            return finish('PROPOSAL_UNRESOLVED',reason='REQUEST_STARTED_WITHOUT_COMMITTED_PROPOSAL')
        try:
            verify_runtime_lock(plan['runtime_lock'],plan['command'],plan['dataset'],plan['tasks'])
            if agent is None:
                from .llm import LLM
                agent = SourceCodingAgent(LLM(model=plan['optimizer_model']),plan['coding_max_tokens'])
            if agent.llm.model!=plan['optimizer_model'] or agent.max_tokens!=plan['coding_max_tokens']:
                raise ValueError('Coding model/profile differs from the frozen optimizer model')
            def disclose(tasks,fields):
                with TaskRegistry(study['registry']) as registry:
                    registry.disclose(study['id'],study['lineage'],tasks,fields)
            options = ({'level':plan['level'],'state_context':plan['state_context']}
                       if 'level' in plan else {})
            if 'coding_request' in plan:
                options['coding_request'] = plan['coding_request']
            compiled = agent.propose(parent,[observed[t] for t in plan['tasks']],plan['history'],
                mechanism_cap=plan['mechanism_cap'],action_cap=plan['action_cap'],budget=budget,
                owner=coding_owner,ledger=ledger,disclose=disclose,**options)
            verify_runtime_lock(plan['runtime_lock'],plan['command'],plan['dataset'],plan['tasks'])
        except Exception as exc:
            return finish('PROPOSAL_UNRESOLVED',reason=type(exc).__name__)
        return finish('PROPOSAL_READY' if compiled.actions else 'NO_CHANGE_PROPOSED',proposal=asdict(compiled),
                      ranking=[{'action':a.id,'disagreement':n} for a,n in compiled.ranked_actions()])
    finally:
        ledger.close()
