"""Offline native performance/budget exports from immutable child journals.

No model/task execution. SELECT scores never become EVAL Pass@1, and outer
mirrored units never become a second monetary charge.
"""
import csv
import json
from pathlib import Path

from .core import Ledger
from .source_usage import summarize_source_usage
from .source_diagnostics import _agent_usage
from .core import digest
from .benchmarks.live import read_reward


def _events(path):
    ledger = Ledger(path)
    try:
        return ledger.events
    finally:
        ledger.close()


def native_report(run):
    root = Path(run).resolve()
    events = _events(root/'evidence.jsonl')
    frozen = [e['payload']['plan'] for e in events if e['kind']=='OPTIMIZATION_FROZEN']
    terminal = [e['payload']['result'] for e in events if e['kind']=='OPTIMIZATION_RESULT']
    if len(frozen)!=1 or len(terminal)!=1:
        raise ValueError('Report requires a closed, identified optimization run')
    plan = frozen[0]
    from .source_optimization import _allocations
    allowed = _allocations(plan)
    child_paths,stages,curve = [],[],[]
    optimization,evaluation = 0,0
    seen = set()
    for event in events:
        p = event['payload']
        if event['kind']=='BUDGET_CHARGED':
            if p['owner'] not in allowed:
                raise ValueError('Unidentified global charge')
            if p['owner'].startswith('evaluation/'):
                evaluation += p['amount']
            else:
                optimization += p['amount']
        if event['kind']!='OPTIMIZATION_STAGE_FINISHED':
            continue
        stage = p['stage']
        if stage not in allowed or stage in seen:
            raise ValueError('Duplicate or out-of-protocol child receipt')
        seen.add(stage)
        path = root/stage/'evidence.jsonl'
        if not path.resolve().is_relative_to(root):
            raise ValueError('Child evidence escaped the run directory')
        child = _events(path)
        if not child or child[-1]['hash']!=p['child_hash']:
            raise ValueError('Child evidence changed after its receipt')
        child_paths.append(path)
        raw = {}
        for entry in child:
            if entry['kind']=='SOURCE_EXECUTION_RESULT':
                p_raw = entry['payload']
                result = p_raw['result']
                provenance = result.get('provenance') or {}
                reward = Path(result.get('verifier_path') or '').resolve()
                key = (p_raw['task'],p_raw['artifact'])
                if (key in raw or not reward.is_relative_to(path.parent/'executions')
                        or provenance.get('execution_kind')!='harbor-agent'
                        or provenance.get('oracle_access') is not False
                        or provenance.get('reward_source')!='official-verifier'
                        or read_reward(reward)!=result['utility']):
                    raise ValueError('Raw execution is duplicate, foreign or disagrees with its official reward')
                raw[key] = result['utility']
        usage = summarize_source_usage([path])
        coding = {'request_attempts':0,'cost_usd':0.}
        if stage.endswith('/diagnostic'):
            diag = [e['payload']['plan'] for e in child if e['kind']=='DIAGNOSTIC_PLAN_FROZEN']
            if len(diag)!=1:
                raise ValueError('Diagnostic owner is not identified')
            coding = _agent_usage(child,'coding:'+digest(diag[0]))
        stages.append({'stage':stage,'budget_units':sum(e['payload']['amount'] for e in child
            if e['kind']=='BUDGET_CHARGED'),'task_results':usage['task_result_records'],
            'task_cost_usd':usage['cost_usd'],'recorded_task_cost_usd':usage['recorded_cost_usd'],
            'input_tokens':usage['recorded_input_tokens'],'output_tokens':usage['recorded_output_tokens'],
            'coding_requests':coding['request_attempts'],'coding_cost_usd':coding['cost_usd']})
        if stage.startswith('evaluation/'):
            plans = [e['payload']['plan'] for e in child if e['kind']=='EVALUATION_FROZEN']
            decisions = [e['payload']['result'] for e in child if e['kind']=='EVALUATION_RESULT']
            if len(plans)!=1 or len(decisions)!=1:
                raise ValueError('Checkpoint lacks its exact evaluation proof')
            rows = [e['payload'] for e in child if e['kind']=='EVALUATION_UNIT']
            tasks = plans[0]['tasks']
            decision = decisions[0]
            if decision['status']=='EVALUATION_COMPLETE':
                if (len(rows)!=len(tasks) or {r['task'] for r in rows}!=set(tasks)
                        or any(type(r['utility']) not in (int,float) or r['utility'] not in (0,1) for r in rows)):
                    raise ValueError('Incomplete/nonbinary checkpoint sample')
                score = sum(r['utility'] for r in rows)/len(tasks)
                if score!=decision['pass_at_1']:
                    raise ValueError('Checkpoint score disagrees with its task observations')
                if any(raw.get((r['task'],plans[0]['artifact']['version']))!=r['utility'] for r in rows):
                    raise ValueError('Checkpoint unit differs from its raw native execution')
                curve.append({'checkpoint':stage,'optimization_budget':optimization,
                    'evaluation_budget':evaluation,'pass_at_1':score,'n_tasks':len(tasks),
                    'artifact':plans[0]['artifact']['version']})
    child_units = sum(s['budget_units'] for s in stages)
    if child_units!=optimization+evaluation:
        raise ValueError('Unsettled/unreceipted child spending; no complete report')
    accepted = {r['selection']['winner'] for r in terminal[0].get('rounds',[])
                if r.get('selection') and r['selection'].get('winner')}
    return {'schema':'dace-native-report/v1','status':terminal[0]['status'],
        'mode':'live','model':plan['model'],'level':plan.get('level','legacy-native'),
        'formal_benchmark_result':False,'n_blind_checkpoints':len(curve),'curve':curve,'stages':stages,
        'optimization_budget':optimization,'evaluation_budget':evaluation,
        'accepted_unique_artifacts':len(accepted),
        'units_per_accepted_patch':optimization/len(accepted) if accepted else None,
        'execution_usage':summarize_source_usage(child_paths),
        'notes':['All curves use EVAL only; repeated checkpoints are not new independent task samples.',
                 'Harbor task costs are recorded values, not independently reconciled provider bills.',
                 'No model/task calls were made by this report.']}


def write_native_report(run,output):
    report = native_report(run)
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,rows,fields in (
        ('curve',report['curve'],['checkpoint','optimization_budget','evaluation_budget','pass_at_1','n_tasks','artifact']),
        ('stages',report['stages'],['stage','budget_units','task_results','task_cost_usd','recorded_task_cost_usd',
            'input_tokens','output_tokens','coding_requests','coding_cost_usd'])):
        with (output/(name+'.csv')).open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return report
