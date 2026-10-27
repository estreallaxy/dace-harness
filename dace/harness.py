"""Simulated multi-component harness with injectable faults (ground truth for attribution).

The environment models a Planning -> Adapter -> Tool -> Verifier execution chain
with a boolean configuration surface. Each fault is a mechanism with a known
source component, a propagation path, and the exact configuration key that
repairs it at the source. Downstream configuration keys may compensate for an
upstream fault without resolving it, which is precisely the distinction DACE's
dual attribution is designed to expose.

Boundary fields reported per run (component-level contract states):
    planner_valid    planner output satisfies its interface contract
    cache_fresh      planner state/entry is current, not stale
    adapter_valid    adapter passes a correct value downstream
    tool_success     tool call completed within contract
    verifier_agrees  verifier accepted the final result

Utility is per-task binary (pass/fail), so paired Hoeffding bounds over task
units remain valid. Partial fixes move the *mean* utility, not per-task values.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable

CONFIG_KEYS = ('planner_cast', 'adapter_cast', 'planner_refresh', 'tool_strict', 'verifier_recheck')

BASELINE_CONFIG = {k: False for k in CONFIG_KEYS}

BOUNDARY_FIELDS = ('planner_valid', 'cache_fresh', 'adapter_valid', 'tool_success', 'verifier_agrees')

COMPONENTS = ('planner', 'adapter', 'tool', 'verifier')

#: keys of the config surface that repair each fault at its own component
FAULT_KINDS = {
    'PLANNER_TYPE_ERROR': 'planner_cast',
    'STALE_PLANNER_CACHE': 'planner_refresh',
    'ADAPTER_GARBLING': 'adapter_cast',
    'TOOL_OFF_BY_ONE': 'tool_strict',
    'VERIFIER_DRIFT': 'verifier_recheck',
}


def _stable_hash(value) -> int:
    return int(hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16], 16)


@dataclass(frozen=True)
class Fault:
    kind: str
    component: str
    repair_key: str
    #: fraction of tasks whose conditions satisfy the fault's applicability premise
    scope: float = 1.0

    def applies(self, task) -> bool:
        return (task.payload['magnitude'] % 1000) / 1000.0 < self.scope


@dataclass
class Task:
    task_id: str
    family: str
    payload: dict = field(default_factory=dict)


class SimulatedEnvironment:
    """Deterministic harness simulator; faults provide attribution ground truth."""

    def __init__(self, faults: Iterable[Fault] = ()):
        self.faults = tuple(faults)
        for f in self.faults:
            if FAULT_KINDS.get(f.kind) != f.repair_key:
                raise ValueError(f'Fault {f.kind} must be repaired by {FAULT_KINDS.get(f.kind)}')

    @property
    def root_components(self) -> set:
        return {f.component for f in self.faults}

    def execute(self, config: dict, task: Task):
        """Return (boundary dict, utility in {0,1}) for one task run."""
        cfg = dict(BASELINE_CONFIG)
        cfg.update({k: v for k, v in config.items() if k in CONFIG_KEYS})
        mag = task.payload['magnitude']
        cacheable = task.payload['cacheable']

        planner_valid, cache_fresh = True, True
        for f in self.faults:
            if not f.applies(task):
                continue
            if f.kind == 'PLANNER_TYPE_ERROR':
                planner_valid = cfg['planner_cast']
            elif f.kind == 'STALE_PLANNER_CACHE' and cacheable:
                cache_fresh = cfg['planner_refresh']

        # upstream value entering the adapter is correct iff planner output and
        # its cached entry are both sound; a stale entry poisons the value even
        # though the planner's own contract (fresh computations) looks valid.
        upstream_ok = planner_valid and cache_fresh

        # adapter: garbles values only under its own fault; with adapter_cast it
        # re-derives the value from raw task input, which also compensates any
        # upstream defect (this is the compensation channel).
        own_garble = any(f.kind == 'ADAPTER_GARBLING' and f.applies(task) for f in self.faults)
        if own_garble or not upstream_ok:
            adapter_valid = cfg['adapter_cast']
        else:
            adapter_valid = True

        tool_ok = not any(f.kind == 'TOOL_OFF_BY_ONE' and f.applies(task) for f in self.faults) or cfg['tool_strict']
        tool_success = adapter_valid and tool_ok

        verifier_drift = any(f.kind == 'VERIFIER_DRIFT' and f.applies(task) for f in self.faults)
        verifier_agrees = tool_success and (cfg['verifier_recheck'] or not verifier_drift)

        boundary = {
            'planner_valid': planner_valid,
            'cache_fresh': cache_fresh,
            'adapter_valid': adapter_valid,
            'tool_success': tool_success,
            'verifier_agrees': verifier_agrees,
        }
        # task-level utility: the downstream outcome, not component contracts.
        # A compensation can restore the outcome while a source contract stays
        # violated -- exactly the distinction dual attribution exposes.
        utility = 1.0 if tool_success and verifier_agrees else 0.0
        return boundary, utility

    def activation(self, config: dict, task: Task, keys) -> bool:
        """Execution fact: was any of `keys`' fault-conditioned branch actually
        reached while running `config` on `task`?

        An edit whose branch never runs (the fault it guards against does not
        apply to this task) is NOT activated, however much the configuration
        differs from the parent: a written config key is not evidence that
        the code path was executed. This is the simulator's activation hook;
        the live executor reports the same fact from its trace hook.
        """
        exercised = set()
        planner_valid, cache_fresh = True, True
        for f in self.faults:
            if not f.applies(task):
                continue
            if f.kind == 'PLANNER_TYPE_ERROR':
                planner_valid = config.get('planner_cast', False)
                exercised.add('planner_cast')
            elif f.kind == 'STALE_PLANNER_CACHE' and task.payload['cacheable']:
                cache_fresh = config.get('planner_refresh', False)
                exercised.add('planner_refresh')
        upstream_ok = planner_valid and cache_fresh
        own_garble = any(f.kind == 'ADAPTER_GARBLING' and f.applies(task) for f in self.faults)
        if own_garble or not upstream_ok:
            exercised.add('adapter_cast')
        if any(f.kind == 'TOOL_OFF_BY_ONE' and f.applies(task) for f in self.faults):
            exercised.add('tool_strict')
        if any(f.kind == 'VERIFIER_DRIFT' and f.applies(task) for f in self.faults):
            exercised.add('verifier_recheck')
        return any(k in exercised for k in keys)

    def summarize_traces(self, traces: list) -> dict:
        """Aggregate diagnostic traces into symptom statistics (observable view).

        Boundary fields a trace does not report are UNOBSERVED, not False:
        violation rates are computed over the rows that actually observed the
        field, and a separate coverage map reports the observing fraction.
        Missing observations are never imputed and never counted as
        violations. The summary carries the source task IDs it was derived
        from so the disclosure audit can register the actual provenance of
        agent-visible derived statistics, not just field names.
        """
        n = len(traces) or 1
        rates, coverage = {}, {}
        for k in BOUNDARY_FIELDS:
            observed = [t for t in traces if k in t['boundary']]
            coverage[k] = round(len(observed) / n, 4)
            if observed:
                rates[k] = sum(1 for t in observed if not t['boundary'][k]) / len(observed)
        return {
            # violation rate per field, over OBSERVED rows only; a field no
            # trace reported is simply absent (unobservable this round)
            'symptom_rates': rates,
            'coverage': coverage,
            'mean_utility': sum(t['score'] for t in traces) / n,
            # task IDs the statistics above were derived from (disclosure audit)
            'source_tasks': [t['task'] for t in traces],
            # naive locality heuristic: first failing boundary nearest the
            # observed task failure, scanning downstream-to-upstream. Baselines
            # that trust this summary mislocalize propagated faults. A field
            # no trace observed is not treated as violated.
            'naive_localization': self._naive_localization(traces),
        }

    @staticmethod
    def _naive_localization(traces: list) -> str:
        # naive locality heuristic: the adapter's output is the observable
        # intermediate artifact, so a violated adapter boundary is blamed
        # first; internal planner contracts come next; the terminal stages
        # last. Propagated upstream faults get mislocalized to the adapter.
        order = [('adapter_valid', 'adapter'), ('cache_fresh', 'planner'),
                 ('planner_valid', 'planner'), ('tool_success', 'tool'),
                 ('verifier_agrees', 'verifier')]
        agg = {k: any(t['boundary'].get(k) is False for t in traces) for k, _ in order}
        for key, comp in order:
            if agg[key]:
                return comp
        return 'none'
