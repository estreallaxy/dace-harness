"""Benchmark adapter interface.

`mock` execution runs the SimulatedEnvironment against deterministic task
payloads (verified pipeline only; not a benchmark score claim). `live`
execution shells out to the real benchmark runtime via a command template --
scaffolded for the full-scale paper runs; mock mode is what the offline
verification uses.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

FAMILIES = ('parse', 'reason', 'tool-use', 'memory', 'shell', 'web', 'code', 'planning')


def _h(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    boundary: dict
    utility: float
    activated: bool | None = None
    usage: dict = field(default_factory=dict)
    verifier_path: str | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        import math
        if type(self.utility) not in (int, float) or not math.isfinite(self.utility) or not 0 <= self.utility <= 1:
            raise ValueError('Utility must be finite and in [0,1]')
        if any(type(v) is not bool for v in self.boundary.values()):
            raise ValueError('Observed boundary values must be booleans; omit unknown fields')

    @property
    def passed(self) -> bool:
        return self.utility >= 1.0


@dataclass
class TaskPool:
    """A role-specific pool with globally unique task IDs."""
    role: str
    tasks: list

    def ids(self) -> list:
        return [t.task_id for t in self.tasks]


class Benchmark(Protocol):
    name: str
    mode: str

    def pool(self, role: str, size: int, round_idx: int = 0, seed: int = 0) -> TaskPool: ...

    def execute(self, config: dict, task: Task, patch_keys=None,
                parent_version: str | None = None,
                candidate_version: str | None = None,
                action_id: str | None = None) -> TaskResult: ...


class MockBenchmarkBase:
    """Shared mock-execution logic for adapters without a local dataset.

    Subclasses provide `name` and `prefix`; tasks derive payload from the
    stable hash of the full task ID, so pools never overlap by construction.
    """

    name = 'mock'
    prefix = 'mock'
    mode = 'mock'
    scenario_name = 'benchmark_mix'

    def __init__(self, scenario_name: str | None = None, dataset_path: str | None = None, **_):
        from ..components import scenario as make_scenario
        if scenario_name:
            self.scenario_name = scenario_name
        self.env = make_scenario(self.scenario_name)
        self.dataset_path = dataset_path
        self._dataset_ids: list | None = None

    def pool(self, role: str, size: int, round_idx: int = 0, seed: int = 0) -> TaskPool:
        ids = self._pool_ids(role, size, round_idx, seed)
        tasks = [Task(i, 'primary',
                      {'magnitude': _h(f'{i}|mag') % 1000,
                       'cacheable': _h(f'{i}|cache') % 2 == 0})
                 for i in ids]
        return TaskPool(role, tasks)

    def _pool_ids(self, role: str, size: int, round_idx: int, seed: int) -> list:
        # evaluation pool is fixed across rounds (same IDs, EVAL role only);
        # all other pools are fresh per round.
        if role == 'EVAL':
            return [f'{self.prefix}-eval-{seed}-{j}' for j in range(size)]
        return [f'{self.prefix}-{role}-{seed}-r{round_idx}-{j}' for j in range(size)]

    def execute(self, config: dict, task: Task, patch_keys=None,
                parent_version: str | None = None,
                candidate_version: str | None = None,
                action_id: str | None = None) -> TaskResult:
        """Execute one run.

        `patch_keys` names the edit under test (the keys the candidate config
        changes relative to its parent); `parent_version`/`candidate_version`/
        `action_id` carry THIS edit's identity through the execution protocol.
        When `patch_keys` is provided, the executor reports an activation
        execution fact bound to exactly that edit; when omitted, `activated`
        stays None (unknown): activation is never inferred from a
        configuration diff. The mock executor derives activation directly
        from the edit keys, so the identity fields are not needed here; the
        live executor validates them against the runner's trace.
        """
        boundary, utility = self.env.execute(config, task)
        activated = (self.env.activation(config, task, patch_keys)
                     if patch_keys is not None else None)
        return TaskResult(task.task_id, boundary, utility, activated=activated)

    def ground_truth(self) -> dict:
        return {'faults': [{'kind': f.kind, 'component': f.component, 'repair_key': f.repair_key,
                            'scope': f.scope} for f in self.env.faults],
                'root_components': sorted(self.env.root_components)}


def make_payload(task_id: str) -> dict:
    """Deterministic task payload from the full task ID."""
    return {'magnitude': _h(f'{task_id}|mag') % 1000,
            'cacheable': _h(f'{task_id}|cache') % 2 == 0}


def dataset_pool(adapter, role: str, size: int, round_idx: int = 0, seed: int = 0) -> list:
    """Exact immutable task allocations; evaluation IDs are reserved globally."""
    import random
    if size <= 0:
        raise ValueError('Pool size must be positive')
    if not hasattr(adapter, '_allocations'):
        adapter._allocations = {}
    key = (role, seed, -1 if role == 'EVAL' else round_idx)
    if key in adapter._allocations:
        allocated = adapter._allocations[key]
        if len(allocated) != size:
            raise ValueError('Cannot resize a frozen task pool')
        return list(allocated)
    available = sorted(i for i in adapter._dataset_ids if i not in adapter._consumed)
    if len(available) < size:
        raise ValueError(f'INSUFFICIENT_TASKS: {role} needs {size}, only {len(available)} unused tasks remain')
    rng = random.Random(_h(f'{adapter.name}|{seed}|{role}|{round_idx if role != "EVAL" else -1}'))
    picked = rng.sample(available, size)
    adapter._consumed.update(picked)
    adapter._allocations[key] = tuple(picked)
    if role == 'EVAL':
        adapter._eval_ids = tuple(picked)
    return picked
