"""E2: component ablation -- cumulative C0..C5 plus the Contribution-only control."""
from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig, e2_methods
from .metrics import e2_row
from .report import write_table
from .runner import run_experiment_cells

E2_COLUMNS = ['benchmark', 'method', 'n_runs', 'pass@1_mean', 'improvement_mean',
              'cross_run_variance', 'localization_accuracy', 'repair/compensation',
              'effective_patch_rate', 'interventions_per_run', 'budget_spent_mean',
              'regressions', 'regressions_observed_decline',
              'regressions_noninferiority_unproven']

EMPHASIS = [('C2->C3', 'failure-only vs uncoupled dual: contribution side added'),
            ('C3->C4', 'uncoupled dual vs bridged joint update: compensation relocalization'),
            ('C4->C5', 'bridged vs discriminative intervention scheduling')]


def run_e2(cfg: ExperimentConfig, llm_factory=None):
    bcfg = ExperimentConfig(**{**cfg.__dict__, 'exp': 'e2'})
    summaries = run_experiment_cells(e2_methods(bcfg), bcfg, llm_factory=llm_factory)
    by_method = {}
    for s in summaries:
        by_method.setdefault(s.method, []).append(s)
    rows = [{'benchmark': cfg.benchmark, **e2_row(method, ms)} for method, ms in by_method.items()]
    out = Path(cfg.output) / 'e2'
    write_table(rows, E2_COLUMNS, out / 'e2_ablation_table.md', out / 'e2_ablation_table.csv')
    meta = {'rows': rows, 'emphasis_pairs': [{'pair': p, 'tests': t} for p, t in EMPHASIS],
            'attribution_controls': ['C2 Failure-only', 'C3 Uncoupled Dual',
                                     'CONTRIBUTION_ONLY', 'C5 Full DACE']}
    (out / 'e2_summary.json').write_text(
        __import__('json').dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    return meta
