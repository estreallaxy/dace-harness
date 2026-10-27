"""Blind native checkpoint evaluation: never an optimizer feedback source."""
import json
from dataclasses import asdict
from pathlib import Path

from .core import Artifact,Ledger,digest
from .durable import PersistentBudget
from .source_execution import SourceExecution
from .source_usage import summarize_source_usage
from .task_registry import TaskRegistry
from .benchmarks.fingerprint import task_fingerprint


def make_evaluation_plan(artifact,tasks,dataset,command,model,study,runtime_lock,checkpoint='standalone'):
    tasks = list(tasks)
    plan = {'schema':'dace-source-evaluation/v1','artifact':asdict(artifact),'tasks':tasks,
        'dataset':str(Path(dataset).resolve()),'command':list(command),'model':model,
        'study':study,'runtime_lock':runtime_lock,'total_budget':len(tasks),'checkpoint':checkpoint,
        'task_fingerprints':{t:task_fingerprint(Path(dataset)/t) for t in tasks}}
    validate_evaluation_plan(plan)
    return plan


def validate_evaluation_plan(plan):
    from .runtime_lock import verify_runtime_lock
    artifact = Artifact(**plan['artifact'])
    from .source_proposals import source_graph
    source_graph(artifact)
    if (plan.get('schema')!='dace-source-evaluation/v1'
            or artifact!=Artifact.create(artifact.content,artifact.parent,artifact.kind)
            or not plan['tasks'] or len(set(plan['tasks']))!=len(plan['tasks'])
            or set(plan['task_fingerprints'])!=set(plan['tasks'])
            or type(plan['total_budget']) is not int or plan['total_budget']!=len(plan['tasks'])
            or not isinstance(plan['command'],list) or not plan['command'] or not plan['model']
            or not isinstance(plan['checkpoint'],str) or not plan['checkpoint']):
        raise ValueError('Evaluation requires an exact artifact and complete frozen task sample')
    study = plan['study']
    if (not isinstance(study,dict) or set(study)!={'registry','id','lineage'}
            or any(not isinstance(v,str) or not v for v in study.values())
            or not Path(study['registry']).is_absolute()):
        raise ValueError('Evaluation requires shared study/lineage isolation')
    root = Path(plan['dataset']).resolve()
    for task in plan['tasks']:
        path = (root/task).resolve()
        if path.parent!=root or path.name!=task or task_fingerprint(path)!=plan['task_fingerprints'][task]:
            raise ValueError('Frozen evaluation task inputs changed')
    verify_runtime_lock(plan['runtime_lock'],plan['command'],plan['dataset'],plan['tasks'])
    return artifact


def run_evaluation(plan,output):
    plan = json.loads(json.dumps(plan))
    artifact = validate_evaluation_plan(plan)
    output = Path(output).resolve()
    identity,study = digest(plan),plan['study']
    owner = 'evaluation:'+identity
    with TaskRegistry(study['registry']) as registry:
        registry.claim(study['id'],study['lineage'],owner,'EVAL',plan['tasks'],
                       plan['task_fingerprints'],str(output),identity)
    ledger = Ledger(output/'evidence.jsonl',durable=True)
    try:
        budget = PersistentBudget(plan['total_budget'],ledger)
        frozen = [e['payload']['plan'] for e in ledger.events if e['kind']=='EVALUATION_FROZEN']
        if frozen:
            if len(frozen)!=1 or frozen[0]!=plan:
                raise ValueError('Evaluation cannot change on resume')
        elif len(ledger.events)==1:
            ledger.append('EVALUATION_FROZEN',plan=plan)
        else:
            raise ValueError('Unidentified evaluation journal')
        terminal = [e['payload']['result'] for e in ledger.events if e['kind']=='EVALUATION_RESULT']
        units = {}
        for e in ledger.events:
            if e['kind']=='EVALUATION_UNIT':
                p = e['payload']
                if p['task'] not in plan['tasks'] or p['task'] in units or type(p['utility']) not in (int,float) or p['utility'] not in (0,1):
                    raise ValueError('Evaluation contains duplicate/foreign/nonbinary observations')
                units[p['task']] = p['utility']
        def publish(result):
            result = {**result,'plan_hash':identity,'artifact':artifact.version,'feedback_eligible':False,
                'budget':{'total':budget.total,'spent':budget.spent,'locks':dict(budget.locks)},
                'execution_usage':summarize_source_usage([ledger.path])}
            (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            return result
        if terminal:
            if len(terminal)!=1:
                raise ValueError('Repeated evaluation decision')
            if terminal[0]['status']=='EVALUATION_COMPLETE' and (set(units)!=set(plan['tasks'])
                    or terminal[0]['pass_at_1']!=sum(units.values())/len(plan['tasks'])
                    or budget.spent!=len(plan['tasks'])):
                raise ValueError('Terminal evaluation lacks its fixed sample and exact budget')
            budget.release(owner)
            return publish(terminal[0])
        if not any(e['kind']=='BUDGET_RESERVED' for e in ledger.events):
            budget.reserve(owner,len(plan['tasks']))
        attempted = {p['task'] for p in budget.attempts.values() if p['contract_id']==owner}
        if not set(units)<=attempted or not attempted<=set(plan['tasks']):
            raise ValueError('Evaluation charges and observations disagree')
        def finish(status,**data):
            result = {'status':status,**data}
            ledger.append('EVALUATION_RESULT',result=result)
            budget.release(owner)
            return publish(result)
        if attempted-set(units):
            return finish('EVALUATION_UNRESOLVED',pass_at_1=None,reason='INTERRUPTED_TASK_WITHOUT_OBSERVATION')
        executor = SourceExecution(plan,ledger,output,{'evaluation_id':identity})
        for task in plan['tasks']:
            if task in units:
                continue
            budget.charge_confirmation(1,owner,task)
            try:
                with TaskRegistry(study['registry']) as registry:
                    registry.mark_started(study['id'],study['lineage'],owner,task)
                result = executor.execute(artifact,task)
                ledger.append('EVALUATION_UNIT',task=task,utility=result.utility)
                units[task] = result.utility
            except Exception as exc:
                return finish('EVALUATION_UNRESOLVED',pass_at_1=None,reason=type(exc).__name__)
        validate_evaluation_plan(plan)
        return finish('EVALUATION_COMPLETE',pass_at_1=sum(units.values())/len(plan['tasks']),
                      tasks=len(plan['tasks']),rewards=units)
    finally:
        ledger.close()
