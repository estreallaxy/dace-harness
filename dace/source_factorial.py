"""Frozen native four-configuration interaction evidence, not promotion.

All four policies execute on each distinct CONFIRM task. The measured
contrast is Y11-Y10-Y01+Y00, not a path-specific mediated effect. A missing
arm terminates unresolved; no other outcome substitutes for that control.
"""
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from .agents.policy import SLOTS,SourcePolicy
from .benchmarks.base import TaskResult
from .benchmarks.fingerprint import task_fingerprint
from .benchmarks.live import read_reward
from .core import Artifact,Ledger,digest
from .durable import PersistentBudget
from .runtime_lock import verify_runtime_lock
from .source_execution import SourceExecution
from .source_proposals import source_graph
from .source_usage import summarize_source_usage
from .task_registry import TaskRegistry

ARMS = ('00','10','01','11')


def _artifacts(parent,first,second):
    source_graph(parent)
    base = asdict(SourcePolicy(**json.loads(parent.content)))
    for edits in (first,second):
        if (not isinstance(edits,dict) or not edits or not set(edits)<=SLOTS
                or any(base[key]==value for key,value in edits.items())):
            raise ValueError('Each factorial edit must change declared source slots')
    if set(first)&set(second):
        raise ValueError('Factorial edits must affect disjoint source slots')
    return {'00':parent,**{arm:Artifact.create(SourcePolicy(**{**base,**edits}).content,parent.version)
        for arm,edits in (('10',first),('01',second),('11',{**first,**second}))}}


def make_factorial_plan(parent,first,second,tasks,dataset,command,model,study,runtime_lock,*,alpha=.05):
    arms = _artifacts(parent,first,second)
    tasks = list(tasks)
    plan = {'schema':'dace-source-factorial/v1','baseline':asdict(parent),
        'first_edits':first,'second_edits':second,'arms':{key:asdict(a) for key,a in arms.items()},
        'tasks':tasks,'dataset':str(Path(dataset).resolve()),'command':command,'model':model,
        'study':study,'runtime_lock':runtime_lock,'alpha':alpha,'total_budget':4*len(tasks),
        'scope':'terminalbench-trial','arm_order':list(ARMS),'provenance':'CONFIRMATORY',
        'task_fingerprints':{task:task_fingerprint(Path(dataset)/task) for task in tasks}}
    plan = json.loads(json.dumps(plan))
    validate_factorial_plan(plan)
    return plan


def validate_factorial_plan(plan):
    if (plan.get('schema')!='dace-source-factorial/v1' or plan['arm_order']!=list(ARMS)
            or plan['provenance']!='CONFIRMATORY' or plan['scope']!='terminalbench-trial'
            or not isinstance(plan['tasks'],list) or not plan['tasks']
            or any(not isinstance(t,str) or not t for t in plan['tasks'])
            or len(set(plan['tasks']))!=len(plan['tasks'])
            or type(plan['total_budget']) is not int or plan['total_budget']!=4*len(plan['tasks'])
            or type(plan['alpha']) not in (int,float) or not 0<plan['alpha']<1
            or not isinstance(plan['command'],list) or not plan['command']
            or any(not isinstance(v,str) or not v for v in plan['command'])
            or not isinstance(plan['model'],str) or not plan['model']):
        raise ValueError('Freeze all four controls, complete sample, runtime and statistical look')
    parent = Artifact(**plan['baseline'])
    arms = _artifacts(parent,plan['first_edits'],plan['second_edits'])
    if plan['arms']!={key:asdict(a) for key,a in arms.items()}:
        raise ValueError('Factorial controls differ from their exact independently compiled edits')
    study = plan['study']
    if (not isinstance(study,dict) or set(study)!={'registry','id','lineage'}
            or any(not isinstance(v,str) or not v for v in study.values())
            or not Path(study['registry']).is_absolute()):
        raise ValueError('Factorial evidence requires the shared study/lineage')
    root = Path(plan['dataset']).resolve()
    if set(plan['task_fingerprints'])!=set(plan['tasks']):
        raise ValueError('Frozen factorial task sample changed')
    with TaskRegistry(study['registry']) as registry:
        protocol = registry.protocol(study['id'])
        if Path(protocol['dataset']).resolve()!=root:
            raise ValueError('Factorial dataset differs from shared study')
        for task in plan['tasks']:
            if (root/task).resolve().parent!=root:
                raise ValueError('Factorial task must be one dataset directory')
            fingerprint = task_fingerprint(root/task)
            if (plan['task_fingerprints'][task]!=fingerprint
                    or protocol['tasks'].get(task)!={'role':'CONFIRM','fingerprint':fingerprint}):
                raise ValueError('Factorial task role or fingerprint changed')
    verify_runtime_lock(plan['runtime_lock'],plan['command'],plan['dataset'],plan['tasks'])
    return arms


def interaction_evidence(units,alpha):
    """Two-sided fixed-look Hoeffding interval for task-level range [-2,2]."""
    if not units or type(alpha) not in (int,float) or not 0<alpha<1:
        raise ValueError('A fixed nonempty sample and valid alpha are required')
    values = []
    activated = True
    for unit in units:
        rewards,activation = unit['rewards'],unit['activated']
        if (set(rewards)!=set(ARMS) or set(activation)!=set(ARMS[1:])
                or any(type(v) not in (int,float) or v not in (0,1) for v in rewards.values())
                or any(v is not None and type(v) is not bool for v in activation.values())):
            raise ValueError('Interaction needs four binary official rewards and actual activation records')
        values.append(rewards['11']-rewards['10']-rewards['01']+rewards['00'])
        activated = activated and all(value is True for value in activation.values())
    mean = sum(values)/len(values)
    radius = math.sqrt(8*math.log(2/alpha)/len(values))
    lower,upper = max(-2.,mean-radius),min(2.,mean+radius)
    assignment = 'POSITIVE' if lower>0 else 'NEGATIVE' if upper<0 else 'UNRESOLVED'
    return {'contrast':'Y11-Y10-Y01+Y00','n_tasks':len(values),'mean':mean,
        'lower':lower,'upper':upper,'alpha':alpha,'method':'fixed-look-task-Hoeffding-range4',
        'assignment_interaction':assignment,'activation_complete':activated,
        'interaction_status':assignment if activated else 'ACTIVATION_UNRESOLVED',
        'causal_path_identified':False,'promotion_eligible':False,
        'assumptions':['independent task units','four frozen configurations and comparable runtimes',
                       'fixed full sample look','fixed declared execution order, not order-effect identification']}


def _verified_units(events,plan,arms,output):
    identity = digest(plan)
    raw,units = {},{}
    for event in events:
        p = event['payload']
        if event['kind']=='SOURCE_EXECUTION_RESULT':
            arm,task = p.get('arm'),p['task']
            if (p.get('factorial_id')!=identity or arm not in arms or task not in plan['tasks']
                    or p['artifact']!=arms[arm].version or (task,arm) in raw):
                raise ValueError('Foreign or duplicate factorial execution')
            result = p['result']
            TaskResult(**result)
            provenance = result.get('provenance') or {}
            if (provenance.get('execution_kind')!='harbor-agent' or provenance.get('oracle_access') is not False
                    or provenance.get('benchmark_score_eligible') is not True
                    or provenance.get('reward_source')!='official-verifier'
                    or provenance.get('harness_artifact')!=arms[arm].version
                    or provenance.get('policy_version')!=SourcePolicy(**json.loads(arms[arm].content)).version
                    or sum(result.get('usage',{}).get(k,0) for k in ('n_input_tokens','n_output_tokens'))<=0):
                raise ValueError('Factorial evidence lacks genuine native policy provenance')
            reward = Path(result.get('verifier_path') or '').resolve()
            if (not reward.is_relative_to(Path(output).resolve()/'executions')
                    or result['utility'] not in (0,1) or read_reward(reward)!=result['utility']):
                raise ValueError('Factorial reward is missing, nonbinary or disagrees with its raw verifier')
            raw[task,arm] = result
        elif event['kind']=='FACTORIAL_OBSERVATION':
            if p['task'] not in plan['tasks'] or p['task'] in units or p['owner']!='factorial:'+identity:
                raise ValueError('Duplicate or foreign factorial task unit')
            units[p['task']] = p['unit']
    for task,unit in units.items():
        if not all((task,arm) in raw for arm in ARMS):
            raise ValueError('Factorial unit has missing raw controls')
        expected = {'rewards':{arm:raw[task,arm]['utility'] for arm in ARMS},
            'activated':{arm:raw[task,arm]['activated'] for arm in ARMS[1:]}}
        if unit!=expected:
            raise ValueError('Factorial observation differs from its raw four-arm evidence')
    return units,raw


def run_factorial(plan,output,batch=None):
    plan = json.loads(json.dumps(plan))
    arms = validate_factorial_plan(plan)
    if batch is not None and (type(batch) is not int or batch<=0):
        raise ValueError('Batch must be a positive number of complete factorial units')
    output = Path(output).resolve()
    identity,study = digest(plan),plan['study']
    owner = 'factorial:'+identity
    with TaskRegistry(study['registry']) as registry:
        registry.claim(study['id'],study['lineage'],owner,'CONFIRM',plan['tasks'],
                       plan['task_fingerprints'],str(output),identity)
    ledger = Ledger(output/'evidence.jsonl',durable=True)
    try:
        budget = PersistentBudget(plan['total_budget'],ledger)
        frozen = [e['payload']['plan'] for e in ledger.events if e['kind']=='FACTORIAL_FROZEN']
        if frozen:
            if len(frozen)!=1 or frozen[0]!=plan:
                raise ValueError('Factorial protocol cannot change on resume')
        elif len(ledger.events)==1:
            ledger.append('FACTORIAL_FROZEN',plan=plan)
        else:
            raise ValueError('Unidentified journal cannot become factorial evidence')
        units,raw = _verified_units(ledger.events,plan,arms,output)
        attempts = {p['task'] for p in budget.attempts.values() if p['contract_id']==owner}
        charges = [e['payload'] for e in ledger.events if e['kind']=='BUDGET_CHARGED']
        if (not set(units)<=attempts or not attempts<=set(plan['tasks'])
                or any(p.get('owner')!=owner or p['amount']!=4 or not p.get('unit') for p in charges)
                or len(charges)!=len(attempts) or budget.spent!=4*len(attempts)
                or any(task not in attempts for task,arm in raw)):
            raise ValueError('Factorial task receipts and reserved budget disagree')
        def publish(result):
            value = {**result,'kind':'NATIVE_FOUR_CONFIGURATION_INTERACTION','plan_hash':identity,
                'candidate':plan['arms']['11'],'baseline':plan['baseline'],'source_tasks':plan['tasks'],
                'budget':{'total':budget.total,'spent':budget.spent,'locks':dict(budget.locks)},
                'execution_usage':summarize_source_usage([ledger.path]),
                'limitations':['not independent promotion','not a unique-root or mediated-path proof',
                               'fixed control order; hosted model randomness is not paired by seed']}
            (output/'summary.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
            return value
        def finish(status,**data):
            result = {'status':status,**data}
            ledger.append('FACTORIAL_RESULT',result=result)
            budget.release(owner)
            return publish(result)
        terminal = [e['payload']['result'] for e in ledger.events if e['kind']=='FACTORIAL_RESULT']
        if terminal:
            if len(terminal)!=1:
                raise ValueError('Repeated terminal factorial result')
            result = terminal[0]
            if result['status']=='FACTORIAL_COMPLETE':
                if set(units)!=set(plan['tasks']) or len(raw)!=4*len(plan['tasks']):
                    raise ValueError('Terminal factorial evidence lacks its complete sample')
                expected = interaction_evidence([units[t] for t in plan['tasks']],plan['alpha'])
                if result.get('interaction')!=expected:
                    raise ValueError('Terminal factorial evidence differs from its complete fixed look')
            budget.release(owner)
            return publish(result)
        if any(e['kind'] in ('SOURCE_EXECUTION_FAILED','SOURCE_EXECUTION_INVALIDATED') for e in ledger.events):
            return finish('FACTORIAL_UNRESOLVED',reason='FAILED_OR_INVALIDATED_EXECUTION')
        if attempts-set(units):
            return finish('FACTORIAL_UNRESOLVED',reason='INTERRUPTED_FOUR_ARM_UNIT')
        if not any(e['kind']=='BUDGET_RESERVED' for e in ledger.events):
            budget.reserve(owner,plan['total_budget'])
        remaining = [t for t in plan['tasks'] if t not in units]
        if budget.locks.get(owner,0)!=4*len(remaining):
            raise ValueError('Factorial evidence lost its remaining reserved sample budget')
        for task in remaining[:batch] if batch is not None else remaining:
            budget.charge_confirmation(4,owner,task)
            try:
                with TaskRegistry(study['registry']) as registry:
                    registry.mark_started(study['id'],study['lineage'],owner,task)
                results = {}
                for arm in ARMS:
                    keys = None if arm=='00' else sorted(plan['first_edits'] if arm=='10' else
                        plan['second_edits'] if arm=='01' else set(plan['first_edits'])|set(plan['second_edits']))
                    executor = SourceExecution(plan,ledger,output,{'factorial_id':identity,'arm':arm})
                    results[arm] = executor.execute(arms[arm],task,keys,owner if keys else None)
                unit = {'rewards':{arm:results[arm].utility for arm in ARMS},
                    'activated':{arm:results[arm].activated for arm in ARMS[1:]}}
                interaction_evidence([unit],plan['alpha'])
                _verified_units(ledger.events,plan,arms,output)
                ledger.append('FACTORIAL_OBSERVATION',owner=owner,task=task,unit=unit)
                units,_ = _verified_units(ledger.events,plan,arms,output)
            except Exception as exc:
                return finish('FACTORIAL_UNRESOLVED',reason=type(exc).__name__)
        if len(units)!=len(plan['tasks']):
            return publish({'status':'PENDING_FACTORIAL','remaining':[t for t in plan['tasks'] if t not in units]})
        validate_factorial_plan(plan)
        return finish('FACTORIAL_COMPLETE',interaction=interaction_evidence(
            [units[t] for t in plan['tasks']],plan['alpha']))
    finally:
        ledger.close()


def factorial_feedback(path):
    """Read-only certificate: recompute complete controls, never trust summary.json."""
    path = Path(path).resolve()
    ledger = Ledger(path)
    try:
        frozen = [e['payload']['plan'] for e in ledger.events if e['kind']=='FACTORIAL_FROZEN']
        terminal = [e['payload']['result'] for e in ledger.events if e['kind']=='FACTORIAL_RESULT']
        if len(frozen)!=1 or len(terminal)!=1 or terminal[0]['status']!='FACTORIAL_COMPLETE':
            raise ValueError('Four-configuration evidence is not complete')
        plan = frozen[0]
        arms = validate_factorial_plan(plan)
        if any(e['kind'] in ('SOURCE_EXECUTION_FAILED','SOURCE_EXECUTION_INVALIDATED') for e in ledger.events):
            raise ValueError('Failed factorial executions cannot certify an interaction')
        units,raw = _verified_units(ledger.events,plan,arms,path.parent)
        if set(units)!=set(plan['tasks']) or len(raw)!=4*len(units):
            raise ValueError('Factorial certificate lacks all frozen controls')
        computed = interaction_evidence([units[t] for t in plan['tasks']],plan['alpha'])
        if terminal[0].get('interaction')!=computed:
            raise ValueError('Factorial certificate differs from recomputed evidence')
        budget = PersistentBudget(plan['total_budget'],ledger)
        attempts = {p['task'] for p in budget.attempts.values() if p['contract_id']=='factorial:'+digest(plan)}
        charges = [e['payload'] for e in ledger.events if e['kind']=='BUDGET_CHARGED']
        if (attempts!=set(plan['tasks']) or budget.spent!=4*len(units) or budget.locks
                or len(charges)!=len(attempts) or any(p['amount']!=4
                    or p.get('owner')!='factorial:'+digest(plan) or not p.get('unit') for p in charges)):
            raise ValueError('Factorial certificate lacks its own closed task charges')
        return {'plan':plan,'interaction':computed,'candidate':plan['arms']['11'],
            'source_tasks':plan['tasks'],'scope':plan['scope'],'ledger':str(path),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    finally:
        ledger.close()
