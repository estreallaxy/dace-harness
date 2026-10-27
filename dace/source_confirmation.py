"""Durable paired confirmation for actual, versioned Terminus policies.

This is a real execution primitive, not the synthetic catalog's optimizer.
Only measured trial boundaries may appear in a prediction. A failed run ends
the frozen obligation unresolved, without retrying or replacing its task.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .agents.policy import SLOTS, SourcePolicy
from .benchmarks.base import Task
from .benchmarks.live import LiveExecutor
from .benchmarks.fingerprint import task_fingerprint
from .core import Artifact, Contract, Controller, Exposure, Ledger, Observation, StatisticalContract, digest
from .predicate_evidence import PredicateProtocol
from .durable import PersistentBudget, contract_from_dict
from .source_usage import summarize_source_usage


BOUNDARIES = frozenset({'raw_response_parse_valid', 'response_parse_valid',
                       'command_batch_valid', 'terminal_dispatch_completed'})


def make_plan(baseline, candidate, predictions, tasks, dataset, command, model,
              total_budget=None, alpha=.05, study=None, runtime_lock=None,predicate_protocol=None):
    """Create a reviewable plan before allocating a single confirmatory call."""
    if not isinstance(baseline,(SourcePolicy,Artifact)) or not isinstance(candidate, SourcePolicy):
        raise ValueError('Real confirmations require source policies')
    parent = baseline if isinstance(baseline,Artifact) else Artifact.create(baseline.content, kind='BASELINE')
    baseline_policy = SourcePolicy(**json.loads(parent.content))
    artifact = Artifact.create(candidate.content, parent.version)
    changed = sorted(k for k in SLOTS if getattr(baseline_policy,k) != getattr(candidate,k))
    if not changed:
        raise ValueError('A confirmation requires an actual source edit')
    dataset = Path(dataset).resolve()
    tasks = tuple(tasks)
    cls = StatisticalContract if predicate_protocol is not None else Contract
    protocol = {'predicate_protocol':predicate_protocol} if predicate_protocol is not None else {}
    c = cls(artifact.version,parent.version,'terminalbench-trial',tasks,
                 tuple(predictions),cost_per_task=2,alpha=alpha,**protocol)
    plan = {'schema':'dace-source-confirmation/v2' if protocol else 'dace-source-confirmation/v1', 'baseline':asdict(parent),
        'candidate':asdict(artifact), 'contract':asdict(c), 'patch_keys':changed,
        'dataset':str(dataset), 'command':command, 'model':model,
        'total_budget':2*len(tasks) if total_budget is None else total_budget,
        'task_fingerprints':{t:task_fingerprint(dataset/t) for t in tasks}}
    if study is not None:
        plan['study'] = study
    if runtime_lock is not None:
        plan['runtime_lock'] = runtime_lock
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan.get('schema') not in ('dace-source-confirmation/v1','dace-source-confirmation/v2'):
        raise ValueError('Unsupported source confirmation plan')
    if 'study' in plan:
        study = plan['study']
        if (not isinstance(study,dict) or set(study) != {'registry','id','lineage'}
                or any(not isinstance(v,str) or not v for v in study.values())
                or not Path(study['registry']).is_absolute()):
            raise ValueError('Study binding requires absolute registry path, ID and lineage')
    baseline, candidate = Artifact(**plan['baseline']), Artifact(**plan['candidate'])
    for artifact in (baseline,candidate):
        if artifact != Artifact.create(artifact.content,artifact.parent,artifact.kind):
            raise ValueError('Frozen artifact hash mismatch')
    if (baseline.kind not in ('BASELINE','PATCH','JOINT_PATCH') or candidate.kind != 'PATCH'
            or candidate.parent != baseline.version):
        raise ValueError('Expected an exact baseline and direct source patch')
    before, after = (SourcePolicy(**json.loads(a.content)) for a in (baseline,candidate))
    keys = sorted(k for k in SLOTS if getattr(before,k) != getattr(after,k))
    if not keys or plan['patch_keys'] != keys:
        raise ValueError('Patch keys must equal the actual source diff')
    c = contract_from_dict(plan['contract'])
    if isinstance(c,StatisticalContract)!=(plan['schema']=='dace-source-confirmation/v2'):
        raise ValueError('Confirmation protocol and frozen contract schema differ')
    if (c.artifact != candidate.version or c.parent != baseline.version
            or c.cost_per_task != 2 or c.provenance != 'CONFIRMATORY'):
        raise ValueError('Paired source confirmation contract mismatch')
    for p in c.predictions:
        for fields in (p.baseline_boundary,p.intervention_boundary,p.mediator):
            if any(k not in BOUNDARIES or type(v) is not bool for k,v in fields):
                raise ValueError('Real predictions must name actual boolean trial hooks')
        if p.source not in BOUNDARIES:
            raise ValueError('Source must identify a measured trial boundary')
    if (not isinstance(plan['command'],list) or not plan['command']
            or any(not isinstance(v,str) for v in plan['command']) or not plan['model']):
        raise ValueError('Explicit runner argv and model are required')
    if type(plan['total_budget']) is not int or plan['total_budget'] < 2*len(c.tasks):
        raise ValueError('Frozen budget cannot fund the paired sample plan')
    root = Path(plan['dataset']).resolve()
    if set(plan['task_fingerprints']) != set(c.tasks):
        raise ValueError('Frozen task manifest mismatch')
    for task in c.tasks:
        path = (root/task).resolve()
        if path.parent != root or path.name != task:
            raise ValueError('Task identity must be one dataset directory')
        if task_fingerprint(path) != plan['task_fingerprints'][task]:
            raise ValueError('Frozen task inputs changed')
    if 'runtime_lock' in plan:
        from .runtime_lock import verify_runtime_lock
        verify_runtime_lock(plan['runtime_lock'],plan['command'],plan['dataset'],c.tasks)
    return baseline,candidate,c


def run_plan(plan, output, batch=None):
    plan = json.loads(json.dumps(plan))  # Freeze caller-owned nested data for this call.
    baseline,candidate,contract = validate_plan(plan)
    if batch is not None and (type(batch) is not int or batch <= 0):
        raise ValueError('Batch must be a positive number of paired units')
    output = Path(output).resolve()
    if plan.get('study'):
        from .task_registry import TaskRegistry
        study = plan['study']
        with TaskRegistry(study['registry']) as registry:
            registry.claim(study['id'],study['lineage'],digest(plan),'CONFIRM',
                contract.tasks,plan['task_fingerprints'],str(output),digest(plan))
    output.mkdir(parents=True,exist_ok=True)
    path = output/'evidence.jsonl'
    if path.exists():
        controller = Controller.restore(path,plan['total_budget'])
    else:
        ledger = Ledger(path,durable=True)
        controller = Controller(ledger,PersistentBudget(plan['total_budget'],ledger),Exposure(ledger))
    try:
        events = [e for e in controller.ledger.events if e['kind']=='SOURCE_SESSION_FROZEN']
        if events:
            if len(events)!=1 or digest(events[0]['payload']['plan']) != digest(plan):
                raise ValueError('Resume must use the exact frozen source session plan')
        else:
            if len(controller.ledger.events) != 1:
                raise ValueError('Unidentified ledger cannot become a source session')
            controller.ledger.append('SOURCE_SESSION_FROZEN',plan=plan)
        controller.freeze(candidate,contract)
        from .source_execution import SourceExecution
        executor = SourceExecution(plan,controller.ledger,output,{'contract_id':contract.id})

        def pair(task):
            if plan.get('study'):
                with TaskRegistry(study['registry']) as registry:
                    registry.mark_started(study['id'],study['lineage'],digest(plan),task)
            before = executor.execute(baseline,task)
            after = executor.execute(candidate,task,plan['patch_keys'],contract.id)
            return Observation(task,before.boundary,after.boundary,
                               before.utility,after.utility,after.activated)

        evidence = controller.resume(contract.id,pair,batch=batch)
        result = {'kind':'REAL_SOURCE_CONFIRMATION','contract_id':contract.id,
            'task_isolation':'study-lineage' if plan.get('study') else 'session-only',
            'runtime_lock_hash':digest(plan['runtime_lock']) if 'runtime_lock' in plan else None,
            'plan_hash':digest(plan),'evidence':evidence,
            'execution_usage':summarize_source_usage([controller.ledger.path]),
            'observations':[asdict(x) for x in controller.observations[contract.id].values()],
            'budget':{'total':controller.budget.total,'spent':controller.budget.spent,
                      'locks':controller.budget.locks},
            'limitations':['fixed baseline-then-candidate order; no efficacy claim from one task',
                'no outer evolution or independent promotion in this entry point',
                'unknown boundaries stay unresolved; official verifier supplies all utilities']}
        (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    finally:
        controller.ledger.close()
