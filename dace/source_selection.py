"""Independent fixed-look selection of exact real source artifacts.

Confirmation eligibility is recomputed from immutable evidence, not a summary
flag. Selection units belong to SELECT, never the confirmation or EVAL pools.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from .agents.policy import SLOTS, SourcePolicy
from .benchmarks.fingerprint import task_fingerprint
from .benchmarks.live import read_reward
from .benchmarks.base import TaskResult
from .core import Artifact, Budget, Controller, Exposure, Ledger, Observation, digest, promotion
from .durable import PersistentBudget
from .source_confirmation import validate_plan as validate_confirmation
from .source_execution import SourceExecution
from .task_registry import TaskRegistry
from .source_usage import summarize_source_usage


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confirmation_feedback(path):
    """Validate closed fixed-look evidence, including inactive/unknown edits.

    Selection applies an additional minimal-activation gate; source evidence
    must remain available even when a response branch was never activated.
    """
    path = Path(path).resolve()
    ledger = Ledger(path)
    sessions = [e['payload']['plan'] for e in ledger.events if e['kind']=='SOURCE_SESSION_FROZEN']
    if len(sessions)!=1:
        raise ValueError('Candidate needs its own source confirmation session')
    plan = sessions[0]
    baseline,candidate,c = validate_confirmation(plan)
    if not plan.get('study') or not plan.get('runtime_lock'):
        raise ValueError('Selection requires study-bound, runtime-locked confirmation evidence')
    freezes = [e['payload'] for e in ledger.events if e['kind']=='CONTRACT_FROZEN']
    if (len(freezes)!=1 or freezes[0]['contract_id'] != c.id
            or freezes[0]['artifact'] != asdict(candidate)
            or digest(freezes[0]['contract']) != digest(asdict(c))):
        raise ValueError('Candidate confirmation artifact/contract mismatch')
    terminal = [e for e in ledger.events if e['kind']=='TERMINATED_UNRESOLVED'
                and e['payload'].get('contract_id')==c.id]
    if terminal or any(e['kind']=='SOURCE_EXECUTION_INVALIDATED' for e in ledger.events):
        raise ValueError('Unresolved or invalidated confirmation cannot enter selection')
    records = [e['payload'] for e in ledger.events if e['kind']=='OBSERVATION']
    observations = {}
    for record in records:
        obs = Observation(**record['observation'])
        if (record['contract_id'] != c.id or record['artifact'] != candidate.version
                or record['provenance'] != 'CONFIRMATORY' or obs.task not in c.tasks
                or obs.task in observations):
            raise ValueError('Confirmation observations are not exact independent frozen units')
        observations[obs.task] = obs
    if set(observations)!=set(c.tasks):
        raise ValueError('Candidate confirmation fixed look is incomplete')
    budget = PersistentBudget(plan['total_budget'],ledger)
    charges = [e['payload'] for e in ledger.events if e['kind']=='BUDGET_CHARGED'
               and e['payload'].get('owner')==c.id]
    if (len(charges)!=len(c.tasks) or any(p['amount']!=c.cost_per_task for p in charges)
            or {p.get('unit',{}).get('task') for p in charges}!=set(c.tasks)
            or any(p.get('unit',{}).get('contract_id')!=c.id for p in charges)
            or c.id in budget.locks):
        raise ValueError('Confirmation lacks its exact completed owner-charged sample plan')
    controller = Controller(ledger,Budget(plan['total_budget']),Exposure(ledger))
    controller.contracts[c.id],controller.observations[c.id] = c,observations
    computed = controller.evaluate(c.id)
    results = [e['payload']['result'] for e in ledger.events if e['kind']=='EVIDENCE_RESULT'
               and e['payload'].get('contract_id')==c.id]
    if len(results)!=1 or results[0] != computed:
        raise ValueError('Candidate lacks its own recomputed fixed-look feedback')
    executions = {}
    for e in ledger.events:
        if e['kind']!='SOURCE_EXECUTION_RESULT':
            continue
        p = e['payload']
        key = (p['task'],p['artifact'])
        if p.get('contract_id')!=c.id or key in executions:
            raise ValueError('Duplicate or foreign raw confirmation execution')
        result = p['result']
        TaskResult(**result)
        provenance = result.get('provenance') or {}
        artifact = candidate if p['artifact']==candidate.version else baseline
        if (provenance.get('execution_kind')!='harbor-agent' or provenance.get('oracle_access') is not False
                or provenance.get('benchmark_score_eligible') is not True
                or provenance.get('reward_source')!='official-verifier'
                or provenance.get('harness_artifact')!=p['artifact']
                or provenance.get('policy_version')!=SourcePolicy(**json.loads(artifact.content)).version
                or sum(result.get('usage',{}).get(k,0) for k in ('n_input_tokens','n_output_tokens'))<=0):
            raise ValueError('Confirmation raw execution has no genuine source-agent provenance')
        reward = Path(result.get('verifier_path') or '').resolve()
        if not reward.is_relative_to(path.parent/'executions') or read_reward(reward)!=result['utility']:
            raise ValueError('Confirmation official reward is missing or disagrees')
        executions[key] = result
    if set(executions)!={(t,a.version) for t in c.tasks for a in (baseline,candidate)}:
        raise ValueError('Confirmation must record exactly one baseline and candidate per unit')
    for task,obs in observations.items():
        before,after = executions[task,baseline.version],executions[task,candidate.version]
        if (before['utility']!=obs.before or after['utility']!=obs.after
                or before['boundary']!=obs.baseline or after['boundary']!=obs.intervention
                or after['activated']!=obs.activated):
            raise ValueError('Confirmation observation differs from its raw execution')
    certificate = {'ledger':str(path),'sha256':_hash(path),'contract_id':c.id,
        'baseline':asdict(baseline),'candidate':asdict(candidate),
        'study':plan['study'],'model':plan['model'],'command':plan['command'],
        'runtime_lock_hash':digest(plan['runtime_lock']),
        'confirmation_tasks':list(c.tasks),
        'attribution':[x for x in computed['claims'] if x['action_repairs_source']]}
    return {'certificate':certificate,'plan':plan,'feedback':computed,
            'observations':[asdict(observations[t]) for t in c.tasks]}


def confirmation_certificate(path):
    """Additional selection gate, without changing historical certificate shape."""
    result = confirmation_feedback(path)
    if result['feedback'].get('minimal')!='MINIMAL_OBSERVED':
        raise ValueError('Candidate lacks its own recomputed minimal activation feedback')
    return result['certificate']


def make_selection_plan(confirmations,tasks,families,dataset,command,model,study,runtime_lock,
                        alpha=.05,delta=0.,epsilon=0.,total_budget=None):
    certificates = sorted((confirmation_certificate(p) for p in confirmations),
                          key=lambda x:x['candidate']['version'])
    if not certificates:
        raise ValueError('Independent selection needs eligible candidates')
    tasks = list(tasks)
    plan = {'schema':'dace-source-selection/v1','baseline':certificates[0]['baseline'],
        'certificates':certificates,'tasks':tasks,'families':dict(families),
        'dataset':str(Path(dataset).resolve()),'command':command,'model':model,'study':study,
        'runtime_lock':runtime_lock,'alpha':alpha,'delta':delta,'epsilon':epsilon,
        'total_budget':len(tasks)*(1+len(certificates)) if total_budget is None else total_budget,
        'task_fingerprints':{t:task_fingerprint(Path(dataset)/t) for t in tasks}}
    validate_selection_plan(plan)
    return plan


def validate_selection_plan(plan):
    if plan.get('schema')!='dace-source-selection/v1':
        raise ValueError('Unknown source selection plan')
    for value in ('alpha','delta','epsilon'):
        if type(plan[value]) not in (int,float) or not math.isfinite(plan[value]):
            raise ValueError('Selection thresholds must be finite')
    if not 0<plan['alpha']<1 or not 0<=plan['delta']<=1 or not 0<=plan['epsilon']<=1:
        raise ValueError('Invalid frozen selection thresholds')
    tasks = plan['tasks']
    if (not tasks or len(set(tasks))!=len(tasks) or set(plan['families'])!=set(tasks)
            or any(not isinstance(v,str) or not v for v in plan['families'].values())
            or set(plan['task_fingerprints'])!=set(tasks)):
        raise ValueError('Selection requires unique tasks and explicit family labels')
    root = Path(plan['dataset']).resolve()
    for task in tasks:
        path = (root/task).resolve()
        if path.parent!=root or path.name!=task or task_fingerprint(path)!=plan['task_fingerprints'][task]:
            raise ValueError('Frozen selection task inputs changed')
    baseline = Artifact(**plan['baseline'])
    candidates = {}
    for certificate in plan['certificates']:
        if confirmation_certificate(certificate['ledger'])!=certificate:
            raise ValueError('Frozen confirmation certificate changed')
        artifact = Artifact(**certificate['candidate'])
        if (certificate['baseline']!=plan['baseline'] or artifact.parent!=baseline.version
                or certificate['study']!=plan['study'] or certificate['model']!=plan['model']
                or certificate['command']!=plan['command']
                or certificate['runtime_lock_hash']!=digest(plan['runtime_lock'])
                or set(certificate['confirmation_tasks']) & set(tasks)
                or artifact.version in candidates):
            raise ValueError('Selection must share the exact parent, protocol and independent task roles')
        candidates[artifact.version] = artifact
    if (not candidates or type(plan['total_budget']) is not int
            or plan['total_budget'] < len(tasks)*(len(candidates)+1)):
        raise ValueError('Independent selection budget cannot fund its whole fixed look')
    from .runtime_lock import verify_runtime_lock
    verify_runtime_lock(plan['runtime_lock'],plan['command'],plan['dataset'],tasks)
    return baseline,candidates


def run_selection(plan,output,batch=None):
    plan = json.loads(json.dumps(plan))
    baseline,candidates = validate_selection_plan(plan)
    if batch is not None and (type(batch) is not int or batch<=0):
        raise ValueError('Selection batch must contain a positive number of complete task units')
    output = Path(output).resolve()
    identity = digest(plan)
    owner,study = 'selection:'+identity,plan['study']
    with TaskRegistry(study['registry']) as registry:
        registry.claim(study['id'],study['lineage'],owner,'SELECT',plan['tasks'],
                       plan['task_fingerprints'],str(output),identity)
    ledger = Ledger(output/'evidence.jsonl',durable=True)
    try:
        budget = PersistentBudget(plan['total_budget'],ledger)
        frozen = [e['payload']['plan'] for e in ledger.events if e['kind']=='SELECTION_FROZEN']
        if frozen:
            if len(frozen)!=1 or digest(frozen[0])!=identity:
                raise ValueError('Selection resume must preserve the complete frozen plan')
        elif len(ledger.events)==1:
            ledger.append('SELECTION_FROZEN',plan=plan)
        else:
            raise ValueError('Unidentified journal cannot become a selection run')
        units = {}
        terminal = None
        for event in ledger.events:
            p = event['payload']
            if event['kind']=='SELECTION_UNIT':
                if (p['task'] not in plan['tasks'] or p['task'] in units
                        or set(p['after'])!=set(candidates)
                        or any(type(v) not in (int,float) or not 0<=v<=1
                               for v in [p['before'],*p['after'].values()])):
                    raise ValueError('Selection journal contains invalid independent units')
                units[p['task']] = p
            elif event['kind']=='SELECTION_RESULT':
                if terminal is not None:
                    raise ValueError('Repeated terminal selection decision')
                terminal = p['result']
        def publish(result):
            result = {**result,'kind':'REAL_SOURCE_SELECTION','plan_hash':identity,
                'execution_usage':summarize_source_usage([ledger.path]),
                'budget':{'total':budget.total,'spent':budget.spent,'locks':dict(budget.locks)}}
            (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            return result
        if terminal is not None:
            budget.release(owner)
            return publish(terminal)
        attempted = {u['task'] for u in budget.attempts.values() if u['contract_id']==owner}
        if not set(units)<=attempted or not attempted<=set(plan['tasks']):
            raise ValueError('Selection observations and owner-charged attempts disagree')
        def terminate(reason):
            result = {'status':'TERMINATED_UNRESOLVED','winner':None,
                      'deployed_artifact':asdict(baseline),'reason':reason}
            ledger.append('SELECTION_RESULT',result=result)
            budget.release(owner)
            return publish(result)
        if attempted-set(units):
            return terminate('INTERRUPTED_TASK_UNIT_WITHOUT_COMMITTED_SELECTION_OBSERVATION')
        remaining = [t for t in plan['tasks'] if t not in units]
        cost_per_unit = len(candidates)+1
        reserved = any(e['kind']=='BUDGET_RESERVED' and e['payload'].get('owner')==owner
                       for e in ledger.events)
        if not reserved and not attempted:
            budget.reserve(owner,len(plan['tasks'])*cost_per_unit)
        elif budget.locks.get(owner,0)!=len(remaining)*cost_per_unit:
            raise ValueError('Selection lost its exclusive remaining budget lock')
        executor = SourceExecution(plan,ledger,output,{'selection_id':identity})
        before_policy = SourcePolicy(**json.loads(baseline.content))
        keys = {v:sorted(k for k in SLOTS if getattr(before_policy,k)!=getattr(
            SourcePolicy(**json.loads(a.content)),k)) for v,a in candidates.items()}
        for task in remaining[:batch] if batch is not None else remaining:
            budget.charge_confirmation(cost_per_unit,owner,task)
            try:
                with TaskRegistry(study['registry']) as registry:
                    registry.mark_started(study['id'],study['lineage'],owner,task)
                before = executor.execute(baseline,task)
                after = {v:executor.execute(a,task,keys[v],identity).utility for v,a in candidates.items()}
                unit = {'task':task,'before':before.utility,'after':after}
                ledger.append('SELECTION_UNIT',**unit)
                units[task] = unit
            except Exception as exc:
                return terminate(type(exc).__name__)
        if len(units)!=len(plan['tasks']):
            return publish({'status':'PENDING_SELECTION','winner':None,
                'remaining':[t for t in plan['tasks'] if t not in units]})
        try:
            validate_selection_plan(plan)  # No decision from stale upstream proof or runtime.
        except Exception as exc:
            return terminate(type(exc).__name__)
        attributed_cost = len(plan['tasks'])*(1+1/len(candidates))
        data = {v:{'pairs':[(t,plan['families'][t],units[t]['before'],units[t]['after'][v])
                           for t in plan['tasks']], 'cost':attributed_cost,'size':len(a.content)}
                for v,a in candidates.items()}
        decision = promotion(data,sorted(set(plan['families'].values())),
                             alpha=plan['alpha'],delta=plan['delta'],epsilon=plan['epsilon'])
        winner = decision['winner']
        result = {'status':'SELECTION_COMPLETE','winner':winner,'decision':decision,
            'deployed_artifact':asdict(candidates[winner] if winner else baseline),
            'attribution':next((c['attribution'] for c in plan['certificates']
                               if c['candidate']['version']==winner),[]),
            'selection_tasks':list(plan['tasks']),
            'limitations':['fixed baseline then candidates order','selection is not held-out benchmark evaluation',
                          'no automatic diagnosis/proposal loop in this entry point']}
        ledger.append('SELECTION_RESULT',result=result)
        budget.release(owner)
        return publish(result)
    finally:
        ledger.close()
