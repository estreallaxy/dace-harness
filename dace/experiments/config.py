"""Experiment configurations for E1-E5 (frozen design from the experiment table)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..components import AblationLevel
from ..evolution import MethodSpec

# E3 parameter sensitivity: three predetermined levels per knob, others fixed.
# Levels are expressed in confirm contracts per round so they stay meaningful
# at any sample-size profile (contract cost = confirm_tasks * cost_per_task).
INTERVENTION_BUDGET_LEVELS = {'low': 1, 'mid': 4, 'high': 16}
MECHANISM_CAP_LEVELS = {'low': 2, 'mid': 4, 'high': 8}


@dataclass
class ExperimentConfig:
    exp: str
    benchmark: str = 'terminalbench2'
    scenario: str | None = None
    seeds: tuple = (0, 1, 2)
    rounds: int = 3
    output: str = 'experiments_output'
    dataset_path: str | None = None
    mode: str = 'mock'
    model: str = 'deepseek/deepseek-v4.1-flash'
    # live-mode execution plumbing: without ALL THREE of dataset_path,
    # command_template and work_dir, live experiments fail closed -- there is
    # no silent fallback to the mock simulator.
    command_template: str | None = None
    work_dir: str | None = None
    # sample sizes chosen so fixed-look Hoeffding bounds resolve the mock
    # scenarios' utility deltas (>=0.25) at alpha=0.05
    diag_tasks: int = 8
    confirm_tasks: int = 64
    selection_tasks: int = 256
    eval_tasks: int = 64

    def __post_init__(self):
        if self.mode == 'live' and not (self.dataset_path and self.command_template
                                        and self.work_dir):
            missing = [n for n, v in (('dataset_path', self.dataset_path),
                                      ('command_template', self.command_template),
                                      ('work_dir', self.work_dir)) if not v]
            raise ValueError('LIVE_EXPERIMENT_MISSING_PARAMETERS: ' + ', '.join(missing)
                             + '; refusing to fall back to the mock simulator')

    def method_spec(self, **kwargs) -> MethodSpec:
        base = dict(rounds=self.rounds, diag_tasks=self.diag_tasks,
                    confirm_tasks=self.confirm_tasks, selection_tasks=self.selection_tasks,
                    eval_tasks=self.eval_tasks)
        base.update(kwargs)
        return MethodSpec(**base)


def e1_methods(cfg: ExperimentConfig):
    # LocalityHeuristic is the locality-first heuristic baseline; it is NOT
    # the official AutoSaddler optimizer (that integration is pending).
    return [cfg.method_spec(name='LocalityHeuristic', kind='locality'),
            cfg.method_spec(name='FullDACE', kind='dace', level=AblationLevel.C5)]


def e2_methods(cfg: ExperimentConfig):
    out = [cfg.method_spec(name=f'DACE-{lvl.value}', kind='dace', level=lvl)
           for lvl in (AblationLevel.C0, AblationLevel.C1, AblationLevel.C2, AblationLevel.C3,
                       AblationLevel.C4, AblationLevel.C5)]
    out.append(cfg.method_spec(name='DACE-CONTRIBUTION_ONLY', kind='dace',
                                level=AblationLevel.CONTRIBUTION_ONLY))
    return out


def e3_grid(cfg: ExperimentConfig):
    # one confirm contract costs confirm_tasks * pair_cost (parent + candidate)
    contract_cost = max(1, cfg.confirm_tasks * 2)
    cells = []
    # One knob at a time. The common mid/mid condition is run once and used
    # in both sensitivity curves, rather than launching a 3x3 factorial grid.
    pairs = [(b,'mid') for b in INTERVENTION_BUDGET_LEVELS]
    pairs += [('mid',k) for k in MECHANISM_CAP_LEVELS if k != 'mid']
    for b_name,k_name in pairs:
        cells.append((f'b={b_name}', f'k={k_name}',
                      cfg.method_spec(name=f'C5-b{b_name}-k{k_name}', kind='dace',
                          level=AblationLevel.C5,
                          intervention_budget=INTERVENTION_BUDGET_LEVELS[b_name]*contract_cost,
                          mechanism_cap=MECHANISM_CAP_LEVELS[k_name])))
    return cells


E4_SCENARIOS = ('source_error', 'downstream_compensation', 'compound_fault', 'mislocalization')


def e4_methods(cfg: ExperimentConfig):
    return [cfg.method_spec(name='UncoupledDual', kind='dace', level=AblationLevel.C3),
            cfg.method_spec(name='FullDACE', kind='dace', level=AblationLevel.C5)]


EXPERIMENTS = ('e1', 'e2', 'e3', 'e4', 'e5')
