"""E1: overall performance -- LocalityHeuristic baseline vs Full DACE per benchmark."""
from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig, e1_methods
from .metrics import e1_row, llm_cost
from .report import write_table
from .runner import run_experiment_cells

E1_COLUMNS = ['benchmark', 'method', 'n_runs', 'pass@1_mean', 'improvement_mean',
              'cross_run_variance', 'budget_spent_mean', 'regressions',
              'regressions_observed_decline', 'regressions_noninferiority_unproven',
              'localization_accuracy']


def run_e1(cfg: ExperimentConfig, benchmarks=None, llm_factory=None):
    benchmarks = benchmarks or [cfg.benchmark]
    rows, all_summaries = [], []
    for bench in benchmarks:
        bcfg = ExperimentConfig(**{**cfg.__dict__, 'exp': 'e1'})
        summaries = run_experiment_cells(e1_methods(bcfg), bcfg, benchmark_name=bench,
                                         llm_factory=llm_factory)
        all_summaries.extend(summaries)
        by_method = {}
        for s in summaries:
            by_method.setdefault(s.method, []).append(s)
        for method, ms in by_method.items():
            rows.append({'benchmark': bench, **e1_row(method, ms)})
    out = Path(cfg.output) / 'e1'
    write_table(rows, E1_COLUMNS, out / 'e1_main_table.md', out / 'e1_main_table.csv')
    meta = {'rows': rows, 'llm': llm_cost(all_summaries),
            'benchmarks': benchmarks, 'notes':
            'Mock execution mode: pipeline verification only, not benchmark scores. '
            'Attach token/call accounting from run ledgers for the paper table.'}
    (out / 'e1_summary.json').write_text(__json(meta), encoding='utf-8')
    return meta


def __json(obj):
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
