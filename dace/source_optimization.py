"""Journaled multi-round source optimization with one reserved global budget.

Child execution journals are authoritative. Their budget charges are mirrored,
not added as a second experimental expense. This protocol uses the declared
Terminus edit surface; it is not yet the native C0--C5 implementation.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from .agents.policy import SourcePolicy
from .benchmarks.fingerprint import task_fingerprint
from .core import Artifact, Ledger, Prediction, digest
from .components import AblationLevel
from .source_state import SourceState
from .predicate_evidence import PredicateProtocol
from .durable import PersistentBudget
from .runtime_lock import verify_runtime_lock
from .source_confirmation import make_plan, run_plan
from .source_diagnostics import _agent_usage, make_diagnostic_plan, run_diagnostic_proposal
from .source_proposals import decode_proposal, source_graph
from .source_selection import confirmation_certificate, make_selection_plan, run_selection
from .source_usage import summarize_source_usage
from .source_evaluation import make_evaluation_plan,run_evaluation
from .task_registry import TaskRegistry


def _allocations(plan):
    allocations = {}
    for i, row in enumerate(plan['rounds']):
        prefix = f'round-{i:03}'
        allocations[prefix+'/diagnostic'] = len(row['diagnostic'])+10
        for j, tasks in enumerate(row['confirmation']):
            allocations[prefix+f'/confirm-{j:03}'] = 2*len(tasks)
        allocations[prefix+'/selection'] = (plan['intervention_cap']+1)*len(row['selection'])
    if plan.get('eval_tasks'):
        allocations['evaluation/baseline'] = len(plan['eval_tasks'])
        for i in range(len(plan['rounds'])):
            allocations[f'evaluation/round-{i:03}'] = len(plan['eval_tasks'])
    return allocations


def make_optimization_plan(parent,rounds,dataset,command,model,study,runtime_lock,
                           *,mechanism_cap=4,action_cap=4,intervention_cap=2,
                           alpha=.05,delta=0.,epsilon=0.,optimizer_model=None,
                           coding_max_tokens=12000,total_budget=None,
                           predicate_protocol=PredicateProtocol(),level=None,eval_tasks=None):
    """Round rows fix diagnostic IDs, per-action confirmation slots and selection families."""
    plan = {'schema':'dace-source-optimization/v1','parent':asdict(parent),
        'rounds':rounds,'dataset':str(Path(dataset).resolve()),'command':command,
        'model':model,'study':study,'runtime_lock':runtime_lock,
        'mechanism_cap':mechanism_cap,'action_cap':action_cap,'intervention_cap':intervention_cap,
        'alpha':alpha,'delta':delta,'epsilon':epsilon,'optimizer_model':optimizer_model or model,
        'coding_max_tokens':coding_max_tokens}
    if predicate_protocol is not None:
        if not isinstance(predicate_protocol,PredicateProtocol):
            raise ValueError('Freeze a validated immutable predicate protocol')
        plan['schema'] = 'dace-source-optimization/v2'
        plan['predicate_protocol'] = asdict(predicate_protocol)
    if level is not None:
        if predicate_protocol is None:
            raise ValueError('Native ablations require the v2 statistical contract')
        plan.update(schema='dace-source-optimization/v3',level=AblationLevel(level).value)
    if eval_tasks is not None:
        if level is None or not eval_tasks:
            raise ValueError('Blind evaluation requires a v3 native plan and nonempty fixed sample')
        plan['eval_tasks'] = list(eval_tasks)
    plan = json.loads(json.dumps(plan))
    tasks = [t for row in plan['rounds'] for t in
             row['diagnostic']+sum(row['confirmation'],[])+row['selection']]+plan.get('eval_tasks',[])
    plan['task_fingerprints'] = {t:task_fingerprint(Path(dataset)/t) for t in tasks}
    plan['total_budget'] = sum(_allocations(plan).values()) if total_budget is None else total_budget
    validate_optimization_plan(plan)
    return plan


def validate_optimization_plan(plan):
    if plan.get('schema') not in ('dace-source-optimization/v1','dace-source-optimization/v2','dace-source-optimization/v3'):
        raise ValueError('Unknown source optimization protocol')
    if plan['schema'].endswith('/v3'):
        AblationLevel(plan['level'])
    elif 'level' in plan:
        raise ValueError('Legacy plan cannot acquire retrospective ablation semantics')
    if plan['schema'].endswith(('/v2','/v3')):
        PredicateProtocol(**plan['predicate_protocol'])
    elif 'predicate_protocol' in plan:
        raise ValueError('Legacy optimization cannot add retrospective predicate confidence')
    source_graph(Artifact(**plan['parent']))
    if (any(type(plan[k]) is not int or plan[k]<=0
            for k in ('mechanism_cap','action_cap','intervention_cap','coding_max_tokens'))
            or plan['intervention_cap']>plan['action_cap'] or plan['coding_max_tokens']>65536
            or not isinstance(plan['rounds'],list) or not plan['rounds']):
        raise ValueError('Freeze positive search caps and at least one round')
    for k in ('alpha','delta','epsilon'):
        if type(plan[k]) not in (int,float) or not math.isfinite(plan[k]):
            raise ValueError('Optimization thresholds must be finite')
    if not 0<plan['alpha']<1 or any(not 0<=plan[k]<=1 for k in ('delta','epsilon')):
        raise ValueError('Invalid optimization thresholds')
    study = plan['study']
    if (not isinstance(study,dict) or set(study)!={'registry','id','lineage'}
            or any(not isinstance(v,str) or not v for v in study.values())
            or not Path(study['registry']).is_absolute()
            or not isinstance(plan['command'],list) or not plan['command']
            or any(not isinstance(v,str) or not v for v in plan['command'])
            or not isinstance(plan['model'],str) or not plan['model']
            or not isinstance(plan['optimizer_model'],str) or not plan['optimizer_model']):
        raise ValueError('Explicit shared study and model runtimes are required')
    root, tasks = Path(plan['dataset']).resolve(), {}
    with TaskRegistry(study['registry']) as registry:
        protocol = registry.protocol(study['id'])
        if Path(protocol['dataset']).resolve()!=root:
            raise ValueError('Optimization dataset differs from frozen study')
        for row in plan['rounds']:
            if (not isinstance(row,dict) or set(row)!={'diagnostic','confirmation','selection','families'}
                    or not isinstance(row['confirmation'],list)
                    or len(row['confirmation'])!=plan['intervention_cap']
                    or not isinstance(row['families'],dict)
                    or not isinstance(row['selection'],list)
                    or set(row['families'])!=set(row['selection'])
                    or any(not isinstance(v,str) or not v for v in row['families'].values())):
                raise ValueError('Every round needs fixed diagnostic, confirmation slots and selection families')
            for role,ids in [('DIAG',row['diagnostic']),('SELECT',row['selection']),
                             *(('CONFIRM',ids) for ids in row['confirmation'])]:
                if not isinstance(ids,list) or not ids:
                    raise ValueError('Empty or non-list task allocation')
                for task in ids:
                    if not isinstance(task,str) or task in tasks or (root/task).resolve().parent!=root:
                        raise ValueError('All optimization task units must be distinct across rounds and stages')
                    fingerprint = task_fingerprint(root/task)
                    if protocol['tasks'].get(task)!={'role':role,'fingerprint':fingerprint}:
                        raise ValueError('Task allocation violates shared study role/fingerprint')
                    tasks[task] = fingerprint
        if 'eval_tasks' in plan:
            if (not plan['schema'].endswith('/v3') or not isinstance(plan['eval_tasks'],list)
                    or not plan['eval_tasks']):
                raise ValueError('Blind evaluation belongs to the v3 protocol')
            for task in plan['eval_tasks']:
                if task in tasks or (root/task).resolve().parent!=root:
                    raise ValueError('Evaluation must be disjoint from all optimization tasks')
                fingerprint = task_fingerprint(root/task)
                if protocol['tasks'].get(task)!={'role':'EVAL','fingerprint':fingerprint}:
                    raise ValueError('Evaluation violates shared study role/fingerprint')
                tasks[task] = fingerprint
    if tasks!=plan['task_fingerprints']:
        raise ValueError('Frozen optimization inputs changed')
    if (type(plan['total_budget']) is not int
            or plan['total_budget']<sum(_allocations(plan).values())):
        raise ValueError('Global budget must fund all frozen stage ceilings before diagnosis')
    verify_runtime_lock(plan['runtime_lock'],plan['command'],str(root),list(tasks))


def _child_events(path):
    ledger = Ledger(path)
    try:
        return ledger.events
    finally:
        ledger.close()


def run_optimization(plan,output,agent=None):
    """Resume exact child stages; never infer a result from mutable summary files."""
    plan = json.loads(json.dumps(plan))
    validate_optimization_plan(plan)
    output = Path(output).resolve()
    ledger = Ledger(output/'evidence.jsonl',durable=True)
    try:
        budget = PersistentBudget(plan['total_budget'],ledger)
        freezes = [e['payload'] for e in ledger.events if e['kind']=='OPTIMIZATION_FROZEN']
        identity = digest(plan)
        if freezes:
            if len(freezes)!=1 or freezes[0]!={'plan':plan,'output':str(output)}:
                raise ValueError('Optimization cannot be cloned or changed on resume')
        elif len(ledger.events)==1:
            ledger.append('OPTIMIZATION_FROZEN',plan=plan,output=str(output))
        else:
            raise ValueError('Unidentified journal cannot become source optimization')

        # Completed child journals must remain exactly the proof we accepted.
        receipts = {}
        for e in ledger.events:
            if e['kind']=='OPTIMIZATION_STAGE_FINISHED':
                p = e['payload']
                if p['stage'] not in _allocations(plan):
                    raise ValueError('Receipt names a stage outside the frozen protocol')
                events = _child_events(output/p['stage']/'evidence.jsonl')
                if (p['stage'] in receipts or not events or events[-1]['hash']!=p['child_hash']):
                    raise ValueError('Completed child execution evidence changed or disappeared')
                receipts[p['stage']] = p
        terminals = [e['payload']['result'] for e in ledger.events if e['kind']=='OPTIMIZATION_RESULT']
        if len(terminals)>1:
            raise ValueError('Repeated terminal optimization result')

        def publish(result):
            children = [output/p['stage']/'evidence.jsonl' for p in receipts.values()]
            optimizer_rows = []
            for stage in receipts:
                if stage.endswith('/diagnostic'):
                    events = _child_events(output/stage/'evidence.jsonl')
                    frozen = [e['payload']['plan'] for e in events if e['kind']=='DIAGNOSTIC_PLAN_FROZEN']
                    if len(frozen)!=1:
                        raise ValueError('Optimizer usage lacks its frozen request owner')
                    optimizer_rows.append(_agent_usage(events,'coding:'+digest(frozen[0])))
            complete = all(r['accounting_complete'] for r in optimizer_rows)
            optimizer_usage = {key:sum(r[key] for r in optimizer_rows) for key in (
                'request_attempts','responses','recorded_cost_usd',
                'recorded_prompt_tokens','recorded_completion_tokens')}
            optimizer_usage.update(accounting_complete=complete,
                cost_usd=optimizer_usage['recorded_cost_usd'] if complete else None)
            result = {**result,'kind':'NATIVE_SOURCE_OPTIMIZATION','plan_hash':identity,
                'budget':{'total':budget.total,'spent':budget.spent,'locks':dict(budget.locks)},
                'execution_usage':summarize_source_usage(children) if children else None,
                'optimizer_usage':optimizer_usage,
                'child_journals':[str(p) for p in children],
                'limitations':['declared Terminus source slots, not full native C0--C5',
                    'selection feedback is optimization data, never held-out benchmark evaluation',
                    'budget units include 10 per coding request; they are not token/USD costs']}
            result['evaluation_budget_spent'] = sum(e['payload']['amount'] for e in ledger.events
                if e['kind']=='BUDGET_CHARGED' and e['payload']['owner'].startswith('evaluation/'))
            result['optimization_budget_spent'] = budget.spent-result['evaluation_budget_spent']
            (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            return result

        if terminals:
            for owner in list(budget.locks):
                budget.release(owner)
            return publish(terminals[0])
        reserved = {e['payload']['owner'] for e in ledger.events if e['kind']=='BUDGET_RESERVED'}
        for stage,amount in _allocations(plan).items():
            if stage not in reserved:
                budget.reserve(stage,amount)

        def mirror(stage):
            path = output/stage/'evidence.jsonl'
            if not path.exists():
                return
            for e in _child_events(path):
                if e['kind']=='BUDGET_CHARGED':
                    unit = {'contract_id':stage,'task':e['hash']}
                    if digest(unit) not in budget.attempts:
                        budget.charge_confirmation(e['payload']['amount'],stage,e['hash'])

        def stage_call(stage,child_plan,run):
            child_plan = json.loads(json.dumps(child_plan))  # Same tuples/lists on disk and recovery.
            if child_plan['total_budget']>_allocations(plan)[stage]:
                raise ValueError('Child budget exceeds its pre-reserved global ceiling')
            frozen = [e['payload'] for e in ledger.events if e['kind']=='OPTIMIZATION_STAGE_STARTED'
                      and e['payload']['stage']==stage]
            if frozen:
                if len(frozen)!=1 or frozen[0]['plan']!=child_plan:
                    raise ValueError('Child protocol changed after allocation')
            else:
                ledger.append('OPTIMIZATION_STAGE_STARTED',stage=stage,plan=child_plan)
            if stage in receipts:
                return receipts[stage]['result']
            # Catch up a crash after a child charge, before its outer mirror.
            mirror(stage)
            try:
                result = run(child_plan,output/stage)
            finally:
                mirror(stage)
            events = _child_events(output/stage/'evidence.jsonl')
            receipt = {'stage':stage,'child_hash':events[-1]['hash'],'result':result}
            ledger.append('OPTIMIZATION_STAGE_FINISHED',**receipt)
            receipts[stage] = receipt
            budget.release(stage)
            return result

        parent,history,rounds = Artifact(**plan['parent']),[],[]
        state = SourceState(parent,plan['mechanism_cap'],plan['level']) if 'level' in plan else None
        evaluations = []
        # All fixed looks, including unexecuted ones, share one familywise alpha.
        alpha = plan['alpha']/(len(plan['rounds'])*(plan['intervention_cap']+1))
        def finish(status,**data):
            validate_optimization_plan(plan)  # Do not seal a decision over drifted inputs/runtime.
            result = {'status':status,'deployed_artifact':asdict(parent),'rounds':rounds,**data}
            if state is not None:
                result['native_state'] = state.snapshot()
            if plan.get('eval_tasks'):
                result['evaluations'] = evaluations
            ledger.append('OPTIMIZATION_RESULT',result=result)
            for owner in list(budget.locks):
                budget.release(owner)
            return publish(result)

        def evaluate_checkpoint(stage,artifact):
            child = make_evaluation_plan(artifact,plan['eval_tasks'],plan['dataset'],plan['command'],
                plan['model'],plan['study'],plan['runtime_lock'],checkpoint=stage)
            result = stage_call(stage,child,run_evaluation)
            evaluations.append({'stage':stage,'optimization_budget':budget.spent-sum(
                e['payload']['amount'] for e in ledger.events if e['kind']=='BUDGET_CHARGED'
                and e['payload']['owner'].startswith('evaluation/')),'result':result})
            return result['status']=='EVALUATION_COMPLETE'

        if plan.get('eval_tasks') and not evaluate_checkpoint('evaluation/baseline',parent):
            return finish('OPTIMIZATION_UNRESOLVED',failed_stage='evaluation/baseline')

        for i,row in enumerate(plan['rounds']):
            prefix = f'round-{i:03}'
            options = {}
            if state is not None:
                view = state.snapshot()
                # Raw certificates are not optimizer context; only masked,
                # versioned claims and their proposed next targets are exposed.
                view.pop('proofs')
                options = {'level':plan['level'],'state_context':view}
            round_proofs = set(state.proofs) if state is not None else set()
            diag_plan = make_diagnostic_plan(parent,row['diagnostic'],plan['dataset'],plan['command'],
                plan['model'],plan['study'],plan['runtime_lock'],plan['mechanism_cap'],plan['action_cap'],
                plan['optimizer_model'],history,plan['coding_max_tokens'],**options)
            diagnosis = stage_call(prefix+'/diagnostic',diag_plan,
                                   lambda p,o:run_diagnostic_proposal(p,o,agent))
            if diagnosis['status'] not in ('PROPOSAL_READY','NO_CHANGE_PROPOSED'):
                return finish('OPTIMIZATION_UNRESOLVED',failed_stage=prefix+'/diagnostic')
            # Decode original model payload; a summary cannot invent an action.
            events = _child_events(output/prefix/'diagnostic/evidence.jsonl')
            payloads = [e['payload']['payload'] for e in events if e['kind']=='SOURCE_AGENT_PROPOSAL']
            if len(payloads)!=1:
                raise ValueError('Diagnostic result lacks its own committed proposal')
            batch = decode_proposal(payloads[0],parent,plan['mechanism_cap'],plan['action_cap'],plan.get('level')!='C0')
            batch_id = state.register(batch) if state is not None else None
            ranked = (state.scheduled_actions(batch_id) if state is not None else
                      [(None,a,score) for a,score in batch.ranked_actions()])
            eligible,feedback = [],[]
            for j,(action_batch_id,action,score) in enumerate(ranked[:plan['intervention_cap']]):
                stage = prefix+f'/confirm-{j:03}'
                action_batch = state.batches[action_batch_id]['batch'] if state is not None else batch
                predictions = action_batch.predictions_for(action.id)
                if plan.get('level')=='C0':
                    # Operational control only: not a proposed root mechanism,
                    # and none of its attribution fields reach the C0 optimizer.
                    predictions = (Prediction('activation-control','response_parse_valid',('parser',),
                        (('response_parse_valid',True),),(('response_parse_valid',True),),
                        (('command_batch_valid',True),),'UNRESOLVED',action_repairs_source=False),)
                child = make_plan(parent,SourcePolicy(**json.loads(action.artifact.content)),
                    predictions,row['confirmation'][j],plan['dataset'],
                    plan['command'],plan['model'],alpha=alpha,study=plan['study'],
                    runtime_lock=plan['runtime_lock'],predicate_protocol=(
                        PredicateProtocol(**plan['predicate_protocol'])
                        if 'predicate_protocol' in plan else None))
                result = stage_call(stage,child,run_plan)
                if result['evidence'].get('status')=='TERMINATED_UNRESOLVED':
                    return finish('OPTIMIZATION_UNRESOLVED',failed_stage=stage)
                claims = result['evidence'].get('claims',[])
                if state is not None:
                    projected = state.observe(action_batch_id,action.id,output/stage/'evidence.jsonl')
                    if projected['claims']:
                        feedback.append(projected)
                elif claims:
                    feedback.append({'source_tasks':row['confirmation'][j],'claims':claims,
                        'artifact':action.artifact.version,'scope':child['contract']['scope']})
                if result['evidence'].get('minimal')=='MINIMAL_OBSERVED':
                    proof = output/stage/'evidence.jsonl'
                    confirmation_certificate(proof)  # Invalid proof is an error, not a loser.
                    eligible.append(str(proof))
            selection = None
            if eligible:
                child = make_selection_plan(eligible,row['selection'],row['families'],plan['dataset'],
                    plan['command'],plan['model'],plan['study'],plan['runtime_lock'],
                    alpha=alpha,delta=plan['delta'],epsilon=plan['epsilon'])
                selection = stage_call(prefix+'/selection',child,run_selection)
                if selection['status']!='SELECTION_COMPLETE':
                    return finish('OPTIMIZATION_UNRESOLVED',failed_stage=prefix+'/selection')
                parent = Artifact(**selection['deployed_artifact'])
                if state is not None:
                    state.deploy(parent)
                # The next optimizer sees the deployed artifact. Register its source data.
                with TaskRegistry(plan['study']['registry']) as registry:
                    registry.disclose(plan['study']['id'],plan['study']['lineage'],row['selection'],
                                      ['selected_artifact','selection_decision'])
            if state is not None:
                # Re-project after deployment so old-parent targets are STALE
                # in the next Coding request, not copied as actionable questions.
                feedback = [state.feedback_for(key) for key in state.proofs if key not in round_proofs]
                feedback = [row for row in feedback if row['claims']]
            history.extend(feedback)
            if state is not None:
                snapshot = state.snapshot()
                checkpoints = [e['payload']['snapshot'] for e in ledger.events
                    if e['kind']=='NATIVE_STATE_CHECKPOINT' and e['payload']['round']==i]
                if checkpoints:
                    if len(checkpoints)!=1 or checkpoints[0]!=snapshot:
                        raise ValueError('Native CMVS reconstruction differs from its checkpoint')
                else:
                    ledger.append('NATIVE_STATE_CHECKPOINT',round=i,snapshot=snapshot)
            rounds.append({'round':i,'proposal_status':diagnosis['status'],
                'confirmed_actions':min(len(ranked),plan['intervention_cap']),
                'eligible_candidates':len(eligible),'selection':selection,
                'deployed_artifact':asdict(parent)})
            if plan.get('eval_tasks') and not evaluate_checkpoint(f'evaluation/round-{i:03}',parent):
                return finish('OPTIMIZATION_UNRESOLVED',failed_stage=f'evaluation/round-{i:03}')
        return finish('OPTIMIZATION_COMPLETE')
    finally:
        ledger.close()
