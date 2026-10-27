"""Prepare native E2/E3 studies without executing any task or model call.

Conditions with identical semantics are reused across experiment labels.
All replicates/conditions share one frozen role partition and blind EVAL set.
Replicate IDs vary within-role task ordering, not hosted-model random weights.
"""
from __future__ import annotations

import json
import random
import tomllib
from dataclasses import asdict,dataclass
from pathlib import Path

from .core import digest
from .runtime_lock import verify_runtime_lock
from .source_optimization import make_optimization_plan,run_optimization,validate_optimization_plan
from .task_registry import TaskRegistry
from .coding_request import validate_coding_request


@dataclass(frozen=True)
class NativeProfile:
    rounds: int = 2
    diagnostic_tasks: int = 1
    confirmation_tasks: int = 2
    selection_tasks: int = 8
    evaluation_tasks: int = 8
    action_cap: int = 4
    coding_max_tokens: int = 12000
    partition_seed: int = 0
    family_mode: str = 'metadata-category'
    coding_request: dict | None = None
    factorial_tasks: int = 0

    def __post_init__(self):
        for name in ('rounds','diagnostic_tasks','confirmation_tasks','selection_tasks',
                     'evaluation_tasks','action_cap','coding_max_tokens'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:
                raise ValueError('Native experiment profile counts must be positive integers')
        if type(self.partition_seed) is not int or self.coding_max_tokens>65536:
            raise ValueError('Freeze the partition seed and bounded coding profile')
        if self.family_mode not in ('metadata-category','aggregate'):
            raise ValueError('Freeze metadata-category or explicit aggregate family grouping')
        validate_coding_request(self.coding_request,self.coding_max_tokens)
        if type(self.factorial_tasks) is not int or self.factorial_tasks<0:
            raise ValueError('Factorial task count must be a nonnegative frozen integer')


def native_conditions(experiments=('e2','e3'),interventions=(1,2,4),mechanisms=(2,4,8)):
    if (not experiments or len(set(experiments))!=len(experiments)
            or set(experiments)-{'e2','e3'}):
        raise ValueError('Native suite currently prepares E2/E3 only; other experiments are not relabelled')
    for levels in (interventions,mechanisms):
        if len(levels)!=3 or any(type(v) is not int or v<=0 for v in levels) or not levels[0]<levels[1]<levels[2]:
            raise ValueError('Each independent sensitivity knob requires three ordered positive levels')
    result = {}
    def add(label,level='C5',intervention=interventions[1],mechanism=mechanisms[1]):
        specification = {'level':level,'intervention_cap':intervention,'mechanism_cap':mechanism}
        key = digest(specification)[:16]
        item = result.setdefault(key,{'id':key,'specification':specification,'labels':[]})
        item['labels'].append(label)
    if 'e2' in experiments:
        for level in ('C0','C1','C2','C3','C4','C5','CONTRIBUTION_ONLY'):
            add('E2:'+level,level)
        # Failure-only and Uncoupled Dual are aliases, not additional runs.
        add('E2:Failure-only','C2')
        add('E2:Uncoupled-Dual','C3')
        add('E2:Full-DACE','C5')
    if 'e3' in experiments:
        for value in interventions:
            add('E3:intervention='+str(value),intervention=value)
        for value in mechanisms:
            add('E3:mechanism='+str(value),mechanism=value)
    return list(result.values())


def suite_design(dataset,profile=NativeProfile(),experiments=('e2','e3'),
                 interventions=(1,2,4),mechanisms=(2,4,8)):
    root = Path(dataset).resolve()
    conditions = native_conditions(experiments,interventions,mechanisms)
    maximum = max(c['specification']['intervention_cap'] for c in conditions)
    if maximum>profile.action_cap:
        raise ValueError('Frozen action cap cannot fund the highest intervention setting')
    tasks = sorted(p.name for p in root.iterdir() if p.is_dir() and (p/'task.toml').is_file())
    families = {}
    for task in tasks:
        metadata = tomllib.loads((root/task/'task.toml').read_text(encoding='utf-8')).get('metadata',{})
        category = metadata.get('category')
        if profile.family_mode=='aggregate':
            category = 'terminalbench2'
        if not isinstance(category,str) or not category:
            raise ValueError('Missing metadata.category; choose explicit aggregate mode for non-category fixtures')
        families[task] = category
    needed = profile.rounds*(profile.diagnostic_tasks+maximum*(profile.confirmation_tasks+profile.factorial_tasks)+
                             profile.selection_tasks)+profile.evaluation_tasks
    if len(tasks)<needed:
        raise ValueError(f'INSUFFICIENT_DISTINCT_TASKS: need {needed}, found {len(tasks)}; no pool recycling')
    random.Random(profile.partition_seed).shuffle(tasks)
    offset = 0
    def take(n):
        nonlocal offset
        values = tasks[offset:offset+n]
        offset += n
        return values
    allocation = {'DIAG':take(profile.rounds*profile.diagnostic_tasks),
        'CONFIRM':take(profile.rounds*maximum*(profile.confirmation_tasks+profile.factorial_tasks)),
        'SELECT':take(profile.rounds*profile.selection_tasks)}
    evaluation = take(profile.evaluation_tasks)
    # Remaining task IDs are held out too; they never become spare optimization units.
    allocation['EVAL'] = evaluation+tasks[offset:]
    return {'schema':'dace-native-suite-design/v1','dataset':str(root),'profile':asdict(profile),
        'conditions':conditions,'allocation':allocation,'eval_tasks':evaluation,
        'required_tasks':needed,'runtime_tasks':sum((allocation[r] for r in ('DIAG','CONFIRM','SELECT')),[])+evaluation,
        'max_interventions':maximum,'benchmark':'terminalbench2','families':families}


def _write_frozen(path,value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=json.loads(json.dumps(value)):
            raise ValueError('Prepared file already exists with a different frozen protocol: '+path.name)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2)


def prepare_native_suite(parent,dataset,command,model,runtime_lock,output,*,
                         profile=NativeProfile(),replicates=(0,1,2),study_id='native-e2-e3',
                         experiments=('e2','e3'),interventions=(1,2,4),mechanisms=(2,4,8)):
    design = suite_design(dataset,profile,experiments,interventions,mechanisms)
    if (not replicates or len(set(replicates))!=len(replicates)
            or any(type(r) is not int or r<0 for r in replicates)):
        raise ValueError('Replicates must be distinct nonnegative integer IDs')
    verify_runtime_lock(runtime_lock,command,design['dataset'],design['runtime_tasks'])
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    registry_path = output/'study.sqlite3'
    entries = []
    with TaskRegistry(registry_path) as registry:
        study_hash = registry.create_study(study_id,design['dataset'],design['allocation'])
        for replicate in replicates:
            pools = {r:list(design['allocation'][r]) for r in ('DIAG','CONFIRM','SELECT')}
            for index,role in enumerate(pools):
                random.Random(replicate*17+index).shuffle(pools[role])
            for condition in design['conditions']:
                key,spec = condition['id'],condition['specification']
                lineage = f'{key}-replicate-{replicate}'
                registry.create_lineage(study_id,lineage)
                study = {'registry':str(registry_path),'id':study_id,'lineage':lineage}
                rows,factorial = [],[]
                for i in range(profile.rounds):
                    selection = pools['SELECT'][i*profile.selection_tasks:(i+1)*profile.selection_tasks]
                    start = i*design['max_interventions']*(profile.confirmation_tasks+profile.factorial_tasks)
                    slots = [pools['CONFIRM'][start+j*profile.confirmation_tasks:start+(j+1)*profile.confirmation_tasks]
                             for j in range(spec['intervention_cap'])]
                    if profile.factorial_tasks:
                        offset = start+design['max_interventions']*profile.confirmation_tasks
                        factorial.append([pools['CONFIRM'][offset+j*profile.factorial_tasks:
                            offset+(j+1)*profile.factorial_tasks] for j in range(spec['intervention_cap'])])
                    rows.append({'diagnostic':pools['DIAG'][i*profile.diagnostic_tasks:(i+1)*profile.diagnostic_tasks],
                        'confirmation':slots,'selection':selection,
                        'families':{task:design['families'][task] for task in selection}})
                plan = make_optimization_plan(parent,rows,design['dataset'],command,model,study,runtime_lock,
                    mechanism_cap=spec['mechanism_cap'],intervention_cap=spec['intervention_cap'],
                    action_cap=profile.action_cap,coding_max_tokens=profile.coding_max_tokens,
                    level=spec['level'],eval_tasks=design['eval_tasks'],coding_request=profile.coding_request,
                    factorial_tasks=factorial if profile.factorial_tasks else None)
                # IDs affect task ordering only; no claim about hosted-model seeds.
                plan['replicate_id'] = replicate
                plan['suite_condition'] = key
                relative = f'plans/{lineage}.json'
                _write_frozen(output/relative,plan)
                entries.append({'condition':key,'replicate':replicate,'labels':condition['labels'],
                    'plan':relative,'plan_hash':digest(plan),'run':'runs/'+lineage,
                    'total_budget':plan['total_budget']})
    manifest = {'schema':'dace-native-suite/v1','study':study_id,'study_hash':study_hash,
        'design':design,'entries':entries,'model':model,'formal_benchmark_result':False,
        'total_reserved_run_units':sum(e['total_budget'] for e in entries),
        'notes':['Preparation made no model/task calls.','Equivalent conditions are reused, not rerun.',
                 'Replicate IDs control within-role task ordering only.',
                 'Regressions cover the explicitly frozen selection families; absent families have no guarantee.']}
    _write_frozen(output/'suite.json',manifest)
    return manifest


def run_native_suite(path,agent=None):
    """Explicit sequential dispatch; validate ALL frozen plans before the first call."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema')!='dace-native-suite/v1':
        raise ValueError('Unknown native suite')
    specifications = {c['id']:c['specification'] for c in manifest['design']['conditions']}
    if any(digest(spec)[:16]!=key for key,spec in specifications.items()):
        raise ValueError('Suite condition fingerprint changed')
    loaded,seen = [],set()
    for entry in manifest['entries']:
        plan_path = (path.parent/entry['plan']).resolve()
        output = (path.parent/entry['run']).resolve()
        if (not plan_path.is_relative_to(path.parent) or not output.is_relative_to(path.parent)
                or output in seen):
            raise ValueError('Suite paths must be unique and stay within the suite directory')
        seen.add(output)
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        spec = specifications.get(entry['condition'])
        if (digest(plan)!=entry['plan_hash'] or plan['suite_condition']!=entry['condition']
                or plan['replicate_id']!=entry['replicate'] or plan['model']!=manifest['model']
                or plan['study']['id']!=manifest['study'] or spec is None
                or any(plan[k]!=v for k,v in spec.items())):
            raise ValueError('Frozen suite plan changed')
        validate_optimization_plan(plan)
        with TaskRegistry(plan['study']['registry']) as registry:
            registry._ancestry(plan['study']['id'],plan['study']['lineage'])
        loaded.append((plan,output,entry))
    results = []
    for plan,output,entry in loaded:
        result = run_optimization(plan,output,agent=agent)
        results.append({'condition':entry['condition'],'replicate':entry['replicate'],'status':result['status']})
    return results
