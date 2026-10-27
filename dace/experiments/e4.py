"""E4: mechanism correctness on controlled fault scenarios (C3 vs C5)."""
from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig, e4_methods, E4_SCENARIOS
from .metrics import e4_row
from .report import write_table
from .runner import run_experiment_cells

E4_COLUMNS = ['scenario', 'method', 'localization_accuracy', 'repair_identified',
              'compensation_identified', 'compensation_mislabeled_repair',
              'corrective_interventions']


def run_e4(cfg: ExperimentConfig, llm_factory=None):
    rows = []
    case_study = None
    for scenario in E4_SCENARIOS:
        bcfg = ExperimentConfig(**{**cfg.__dict__, 'exp': 'e4'})
        summaries = run_experiment_cells(e4_methods(bcfg), bcfg, benchmark_name=cfg.benchmark,
                                         scenario=scenario, seeds=cfg.seeds,
                                         llm_factory=llm_factory)
        by_method = {}
        for s in summaries:
            by_method.setdefault(s.method, []).append(s)
        for method, ms in by_method.items():
            rows.append({'scenario': scenario, **e4_row(method, scenario, ms)})
        if scenario == 'mislocalization' and case_study is None:
            full = next((s for s in summaries if s.method == 'FullDACE'), None)
            if full:
                case_study = _case_study(full)
    out = Path(cfg.output) / 'e4'
    write_table(rows, E4_COLUMNS, out / 'e4_correctness_table.md', out / 'e4_correctness_table.csv')
    if case_study:
        (out / 'e4_case_study.md').write_text(case_study, encoding='utf-8')
    meta = {'rows': rows, 'scenarios': list(E4_SCENARIOS)}
    (out / 'e4_summary.json').write_text(
        __import__('json').dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    return meta


def _case_study(summary) -> str:
    """Trace -> intervention -> joint update -> next patch case study (one run)."""
    lines = ['# E4 Case Study: mislocalization scenario, Full DACE',
             '',
             f'Ground truth faults: {summary.ground_truth}',
             f'Baseline eval utility: {summary.baseline_eval}, final: {summary.final_eval}',
             '']
    for rec in summary.rounds:
        lines.append(f'## Round {rec.round_idx}')
        lines.append(f'- contracts frozen: {rec.contracts_frozen} (probes: {rec.probes})')
        lines.append(f'- bridges: {rec.bridges}')
        lines.append(f'- promoted keys: {rec.promoted_keys}, eval utility: {rec.eval_utility}')
        lines.append('')
    lines.append('## Final mechanism states')
    for m in summary.mechanism_states:
        lines.append(f'- {m}')
    lines.append('')
    lines.append(f'Localization: {summary.localization}')
    return '\n'.join(lines) + '\n'
