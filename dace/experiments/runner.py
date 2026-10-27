"""Unified experiment cell runner: (method, benchmark, seed) -> RunSummary."""
from __future__ import annotations

from pathlib import Path

from ..benchmarks import make_benchmark
from ..evolution import EvolutionLoop
from ..llm import LLM


def run_cell(method_spec, benchmark_name, seed, output_root, cfg, llm=None) -> object:
    out = Path(output_root) / f'{benchmark_name}-{method_spec.name}-seed{seed}'
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Run directory not empty: {out}; use a fresh output root')
    benchmark = make_benchmark(benchmark_name, scenario_name=cfg.scenario,
                               dataset_path=cfg.dataset_path, mode=cfg.mode,
                               command_template=cfg.command_template,
                               work_dir=cfg.work_dir, model=cfg.model)
    _preflight_capacity(benchmark, method_spec)
    loop = EvolutionLoop(benchmark, method_spec, seed, out, llm=llm)
    return loop.run()


def _preflight_capacity(benchmark, method_spec):
    """Upfront real-dataset capacity audit: the whole run's task demand must be
    coverable by unused real tasks before anything executes, instead of failing
    mid-experiment when a pool runs dry."""
    if not getattr(benchmark, 'real_dataset', False):
        return
    contracts_per_round = max(1, method_spec.intervention_budget //
                              max(1, method_spec.confirm_tasks * method_spec.cost_per_task))
    needed = (method_spec.eval_tasks
              + method_spec.rounds * (method_spec.diag_tasks
                                      + contracts_per_round * method_spec.confirm_tasks
                                      + method_spec.selection_tasks))
    available = len(benchmark._dataset_ids)
    if needed > available:
        raise ValueError(f'INSUFFICIENT_TASK_CAPACITY: run needs {needed} disjoint real tasks '
                         f'but only {available} exist; reduce sample sizes or enlarge the dataset')


def run_experiment_cells(methods, cfg, benchmark_name=None, scenario=None,
                         seeds=None, llm_factory=None):
    """Run a grid of cells; returns list of RunSummary."""
    benchmark_name = benchmark_name or cfg.benchmark
    seeds = seeds if seeds is not None else cfg.seeds
    if scenario is not None:
        cfg.scenario = scenario
    root = Path(cfg.output) / (f'{cfg.exp}-{benchmark_name}' + (f'-{scenario}' if scenario else ''))
    summaries = []
    for method in methods:
        for seed in seeds:
            llm = llm_factory() if llm_factory else None
            summaries.append(run_cell(method, benchmark_name, seed, root, cfg, llm=llm))
    return summaries
