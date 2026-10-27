"""E3: one-factor-at-a-time sensitivity with one shared middle condition."""
from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig, e3_grid, INTERVENTION_BUDGET_LEVELS, MECHANISM_CAP_LEVELS
from .metrics import e3_row
from .report import svg_line_chart, write_table
from .runner import run_experiment_cells

E3_COLUMNS = ['intervention_budget', 'mechanism_cap', 'pass@1_mean',
              'discrimination_rate', 'budget_spent_mean']


def run_e3(cfg: ExperimentConfig, llm_factory=None):
    bcfg = ExperimentConfig(**{**cfg.__dict__, 'exp': 'e3'})
    grid = e3_grid(bcfg)
    summaries = run_experiment_cells([m for _, _, m in grid], bcfg, seeds=bcfg.seeds,
                                     llm_factory=llm_factory)
    by_name = {}
    for s in summaries:
        by_name[s.method] = by_name.get(s.method, []) + [s]
    rows = []
    for b_label, k_label, spec in grid:
        rows.append(e3_row(b_label, k_label, by_name.get(spec.name, [])))
    out = Path(cfg.output) / 'e3'
    write_table(rows, E3_COLUMNS, out / 'e3_sensitivity_table.md', out / 'e3_sensitivity_table.csv')

    # Both OFAT sweeps share the one frozen midpoint; never connect unrelated cells.
    series = []
    for label,field in [('vary budget (mechanisms=mid)','mechanism_cap'),
                        ('vary mechanisms (budget=mid)','intervention_budget')]:
        middle = 'k=mid' if field=='mechanism_cap' else 'b=mid'
        pts = sorted((r['budget_spent_mean'],r['pass@1_mean']) for r in rows if r[field]==middle)
        series.append((label,pts))
    svg_line_chart(series, 'mean budget spent (cost units)', 'Pass@1 (mock)',
                   'E3: performance vs cost, one parameter varied', out / 'e3_cost_curve.svg')

    # sensitivity curves: pass@1 vs each knob with the other fixed at mid
    pts = sorted({(MECHANISM_CAP_LEVELS[r['mechanism_cap'].removeprefix('k=')], r['pass@1_mean'])
                  for r in rows if r['intervention_budget'] == 'b=mid'})
    svg_line_chart([('mechanism cap', pts)], 'mechanism cap K_active', 'Pass@1 (mock)',
                   'E3: sensitivity to active mechanism cap (budget=mid)',
                   out / 'e3_mechanism_cap.svg')
    pts = sorted({(INTERVENTION_BUDGET_LEVELS[r['intervention_budget'].removeprefix('b=')], r['pass@1_mean'])
                  for r in rows if r['mechanism_cap'] == 'k=mid'})
    svg_line_chart([('intervention budget', pts)], 'intervention budget cap', 'Pass@1 (mock)',
                   'E3: sensitivity to intervention budget (cap=mid)',
                   out / 'e3_intervention_budget.svg')
    meta = {'rows': rows,
            'design':'one-factor-at-a-time; shared mid/mid run reused',
            'seeds':list(bcfg.seeds),
            'grid': {'intervention_budget': INTERVENTION_BUDGET_LEVELS,
                     'mechanism_cap': MECHANISM_CAP_LEVELS}}
    (out / 'e3_summary.json').write_text(
        __import__('json').dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    return meta
