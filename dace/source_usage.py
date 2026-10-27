"""Conservative accounting for source-harness execution evidence.

Recorded sums are lower bounds on observed usage, never a claimed total when
an attempted execution is unresolved or a provider omitted billing fields.
"""
import math
from pathlib import Path

from .core import Ledger


def summarize_source_usage(paths):
    paths = [Path(p).resolve() for p in paths]
    if len(paths)!=len(set(paths)):
        raise ValueError('A usage ledger cannot be counted twice')
    records,failures,invalidations,unsettled = [],0,0,0
    for path in paths:
        if not path.is_file():
            raise ValueError('Usage evidence ledger is missing')
        events = Ledger(path).events
        charged,observed,selected,evaluated = set(),set(),set(),set()
        for e in events:
            p = e['payload']
            if e['kind']=='SOURCE_EXECUTION_RESULT':
                records.append(p['result'].get('usage') or {})
            elif e['kind']=='SOURCE_EXECUTION_FAILED':
                failures += 1
            elif e['kind']=='SOURCE_EXECUTION_INVALIDATED':
                invalidations += 1
            elif e['kind']=='BUDGET_CHARGED' and p.get('unit'):
                charged.add((p['unit']['contract_id'],p['unit']['task']))
            elif e['kind']=='OBSERVATION':
                observed.add((p['contract_id'],p['observation']['task']))
            elif e['kind']=='DIAGNOSTIC_OBSERVATION':
                observed.add((p['owner'],p['task']))
            elif e['kind']=='SELECTION_UNIT':
                selected.add(p['task'])
            elif e['kind']=='EVALUATION_UNIT':
                evaluated.add(p['task'])
        unsettled += sum((owner,task) not in observed and
                         not (owner.startswith('selection:') and task in selected) and
                         not (owner.startswith('evaluation:') and task in evaluated)
                         for owner,task in charged)
    def known(field):
        return [u[field] for u in records if type(u.get(field)) in (int,float)
                and math.isfinite(u[field]) and u[field]>=0]
    complete = (bool(records) and not failures and not invalidations and not unsettled
                and all(len(known(f))==len(records) for f in
                        ('n_input_tokens','n_output_tokens','cost_usd')))
    return {'task_result_records':len(records),'failed_execution_events':failures,
        'invalidated_execution_events':invalidations,'unsettled_task_units':unsettled,
        'accounting_complete':complete,
        'cost_usd':sum(known('cost_usd')) if complete else None,
        'recorded_cost_usd':sum(known('cost_usd')),
        'recorded_input_tokens':sum(known('n_input_tokens')),
        'recorded_output_tokens':sum(known('n_output_tokens')),
        'notes':'Recorded sums cover recorded provider fields only; incomplete totals remain unknown.'}
