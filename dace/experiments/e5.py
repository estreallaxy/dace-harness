"""E5: optimization efficiency -- reuses run ledgers/checkpoints, no new runs.

Reads checkpoint records produced by E1/E2/E4 cells under the same output
root and emits performance-budget curves and cost decomposition.

Conditions are identified by the FULL frozen run identity: (method,
benchmark, scenario, mode, model, sample-size/budget fingerprint). Curves
from runs that differ on any of these are never concatenated into one line.

Multiple seeds of the same condition are compared on a COMMON cumulative
budget grid: at grid budget b, each seed contributes the utility of its most
recent COMPLETED checkpoint at spend <= b (no interpolation, never a future
checkpoint). Aggregation reports the mean, the cross-seed variation and the
number of seeds actually reaching b.

Unit cost is budget spent per EFFECTIVE promoted patch event, counted from
the accepted selections' unique winner artifacts (a joint patch is one
event, not one per modified key). E5 never launches new runs: without
reusable ledgers it fails with an explicit instruction instead of silently
duplicating experiments.
"""
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

from ..core import Ledger

from .config import ExperimentConfig
from .report import svg_line_chart

#: methods E5 reports; DACE-C3 (E2's ablation cell) is the same configuration
#: as UncoupledDual (E1/E4's control) and is folded onto that curve.
E5_METHODS = ('LocalityHeuristic', 'UncoupledDual', 'FullDACE')
ALIASES = {'DACE-C3': 'UncoupledDual'}


def _method_of(summary: dict) -> str | None:
    name = ALIASES.get(summary.get('method'), summary.get('method'))
    return name if name in E5_METHODS else None


def _fingerprint(summary: dict) -> str:
    """Sample-size / budget profile of the frozen method spec."""
    spec = summary.get('spec') or {}
    if not spec:
        return 'spec-unknown'
    # Display names can alias the same configuration; every semantic setting,
    # including evidence policy, alpha and candidate cap, must remain distinct.
    semantic = {k:v for k,v in spec.items() if k != 'name'}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()[:16]


def _condition_of(summary: dict) -> tuple:
    """Experimental condition: full frozen run identity.

    (method, benchmark, scenario, mode, model, fingerprint): runs that differ
    on the execution mode, the model, or any sample-size/budget knob are
    different conditions and are never aggregated into one curve.
    """
    return (_method_of(summary), summary.get('benchmark'), summary.get('scenario'),
            summary.get('mode', 'mock'), summary.get('model'), _fingerprint(summary))


def _condition_key(cond: tuple) -> str:
    method, bench, scenario, mode, model, fp = cond
    return f'{method}|{bench}|{scenario or "default"}|{mode}|{model or "-"}|{fp}'


def _iter_summary_items(output_root: Path):
    for summary_path in sorted(Path(output_root).glob('**/summary.json')):
        try:
            s = json.loads(summary_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
        if s.get('mode') == 'live' and s.get('result_kind') != 'benchmark-agent':
            continue  # Historical oracle/unknown provenance is not live evidence.
        if _method_of(s) and s.get('checkpoints'):
            yield summary_path, s


def _iter_summaries(output_root: Path):
    for _, summary in _iter_summary_items(output_root):
        yield summary


def _last_completed(checkpoints, budget):
    """Most recent checkpoint with spend <= budget (no interpolation)."""
    past = [c for c in checkpoints if c[0] <= budget]
    return past[-1] if past else None


def _collect_curves(output_root: Path):
    """Aggregate seeds of each condition on the common cumulative budget grid.

    Returns ({condition_key: [(budget, mean_utility), ...]},
             {condition_key: [{'budget','mean','std','n_seeds'}, ...]}).
    At grid budget b a seed contributes only checkpoints it has COMPLETED at
    or before b; points before a seed's first checkpoint simply do not count
    that seed (n_seeds shrinks), and no future or interpolated point is ever
    used.
    """
    groups = {}
    for s in _iter_summaries(output_root):
        groups.setdefault(_condition_of(s), []).append(s['checkpoints'])
    series, stats = {}, {}
    for cond, runs in groups.items():
        grid = sorted({c[0] for run in runs for c in run})
        pts, detail = [], []
        for b in grid:
            utilities = []
            for run in runs:
                last = _last_completed(run, b)
                if last is not None:
                    utilities.append(last[1])
            if not utilities:
                continue
            mean = sum(utilities) / len(utilities)
            std = 0.0
            if len(utilities) > 1:
                std = math.sqrt(sum((u - mean) ** 2 for u in utilities)
                                / (len(utilities) - 1))
            pts.append((b, round(mean, 4)))
            detail.append({'budget': b, 'mean': round(mean, 4),
                           'std': round(std, 4), 'n_seeds': len(utilities)})
        if pts:
            key = _condition_key(cond)
            series[key] = pts
            stats[key] = detail
    return series, stats


def _promotion_events(summary: dict) -> int:
    """Effective promoted-patch events: unique winner artifacts actually
    accepted by independent selection (joint patches count once)."""
    arts = {a for r in summary.get('rounds', [])
            for a in (r.get('promoted_artifacts') or [])}
    if arts:
        return len(arts)
    # summaries written before artifact-level event bookkeeping: fall back to
    # counting accepted rounds (still events, not modified keys)
    return sum(1 for r in summary.get('rounds', []) if r.get('accepted'))


def _cost_decomposition(output_root: Path):
    """Activity counts for exactly the eligible runs used by the curves.

    These counts are not dollars or per-stage budget allocations. Missing
    associated ledgers stay explicit; corrupt evidence is rejected.
    """
    totals = {'ledgers_available':0, 'ledgers_missing':0, 'by_condition':{},
              'count_basis':'eligible runs only; activity counts, not token/USD cost'}
    mapping = {'AGENT_PROPOSAL':'agent_proposals', 'CONTRACT_FROZEN':'confirm_contracts',
               'PROMOTION':'selections'}
    for summary_path, summary in _iter_summary_items(output_root):
        key = _condition_key(_condition_of(summary))
        condition = totals['by_condition'].setdefault(key,
            {'ledgers_available':0, 'ledgers_missing':0})
        ledger_path = summary_path.parent/'evidence.jsonl'
        if not ledger_path.is_file():
            totals['ledgers_missing'] += 1
            condition['ledgers_missing'] += 1
            continue
        events = Ledger(ledger_path).events
        totals['ledgers_available'] += 1
        condition['ledgers_available'] += 1
        for event in events:
            field = mapping.get(event['kind'])
            if field:
                totals[field] = totals.get(field,0)+1
                condition[field] = condition.get(field,0)+1
    return totals


def run_e5(cfg: ExperimentConfig, llm_factory=None):
    root = Path(cfg.output)
    series, stats = _collect_curves(root)
    if not any(series.values()):
        raise ValueError('E5_NO_REUSABLE_RUNS: no prior E1/E2/E4 checkpoints under '
                         f'{root}. E5 reuses existing ledgers and never duplicates '
                         'experiments; run E1 (and E4 for the UncoupledDual control) '
                         'with the same output root first.')
    decomposition = _cost_decomposition(root)
    out = root / 'e5'
    out.mkdir(parents=True, exist_ok=True)
    svg_line_chart([(m, pts) for m, pts in series.items() if pts],
                   'cumulative budget spent (cost units)', 'eval utility (mock)',
                   'E5: performance at equal cumulative budget (grouped by full condition)',
                   out / 'e5_efficiency_curve.svg')

    # unit cost per CONDITION (never pooled across different run conditions):
    # cumulative budget spent / effective promoted patch events, where an
    # event is one accepted unique winner artifact (a joint patch is ONE)
    unit_costs = {}
    for s in _iter_summaries(root):
        cond_key = _condition_key(_condition_of(s))
        events = _promotion_events(s)
        spent = s.get('budget_spent', 0)
        if events > 0:
            unit_costs.setdefault(cond_key, []).append(spent / events)
    unit_costs = {k: round(sum(v) / len(v), 2) for k, v in unit_costs.items()}

    meta = {'curves': series,
            'curve_stats': stats,
            'conditions': {'identity': '(method, benchmark, scenario, mode, model, '
                                       'sample-size/budget fingerprint)',
                           'seeds_per_curve': {k: max(d['n_seeds'] for d in v)
                                               for k, v in stats.items()}},
            'aggregation': 'common cumulative budget grid; each seed contributes its '
                           'most recent COMPLETED checkpoint at spend <= grid point '
                           '(no interpolation, no future checkpoints); mean/std/n '
                           'reported per point',
            'unit_costs': unit_costs,
            'unit_cost_definition': 'cumulative budget spent / effective promoted patch '
                                    'events (accepted unique winner artifacts; joint '
                                    'patches count as one event)',
            'cost_decomposition': decomposition,
            'notes': 'Reuses E1/E2/E4 ledgers and checkpoints; no duplicate runs; curves '
                     'grouped by the full frozen run identity (method, benchmark, '
                     'scenario, mode, model, spec fingerprint) and aggregated over '
                     'seeds on the common budget grid.'}
    (out / 'e5_summary.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    return meta
