"""Benchmark adapter layer for experiments E1-E3, E5.

Task pools follow the exposure protocol: diagnosis, confirmation, selection and
evaluation instances receive disjoint task IDs because DACE forbids reusing a
task across roles (Exposure). Pool payloads are derived deterministically from
(task_id, benchmark, split) so runs are reproducible; the evaluation pool is
fixed across rounds within a run (role EVAL only).
"""
from .base import Benchmark, Task, TaskPool, TaskResult
from .terminalbench import TerminalBench2
from .gaia import Gaia2
from .swebench import SwebenchPro

REGISTRY = {
    'terminalbench2': TerminalBench2,
    'gaia2': Gaia2,
    'swebench-pro': SwebenchPro,
}


def make_benchmark(name: str, **kwargs) -> Benchmark:
    if name not in REGISTRY:
        raise ValueError(f'Unknown benchmark {name!r}; available: {sorted(REGISTRY)}')
    return REGISTRY[name](**kwargs)


__all__ = ['Benchmark', 'Task', 'TaskPool', 'TaskResult', 'TerminalBench2', 'Gaia2', 'SwebenchPro', 'REGISTRY', 'make_benchmark']
