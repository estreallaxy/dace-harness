"""SWE-Bench Pro adapter (E1 range). Same interface as TerminalBench2."""
from __future__ import annotations

import json
from pathlib import Path

from .base import Task, TaskPool, TaskResult, dataset_pool, make_payload
from ..components import scenario as make_scenario


class SwebenchPro:
    name = 'swebench-pro'
    prefix = 'swebp'

    def __init__(self, scenario_name: str | None = None, dataset_path: str | None = None,
                 mode: str = 'mock', command_template: str | None = None, **_):
        self.scenario_name = scenario_name or 'benchmark_mix'
        self.env = make_scenario(self.scenario_name)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        if mode != 'mock':
            raise NotImplementedError('Live adapter for this benchmark is not implemented; mock fallback forbidden')
        self.mode = mode
        self.command_template = command_template
        self._dataset_ids = self._load_dataset()
        self._consumed = set()
        self._eval_ids = None

    def _load_dataset(self) -> list | None:
        if not self.dataset_path or not self.dataset_path.exists():
            return None
        if self.dataset_path.is_file():
            data = json.loads(self.dataset_path.read_text(encoding='utf-8'))
            if isinstance(data, list):
                return [str(item.get('instance_id', item.get('id', i))) for i, item in enumerate(data)]
        return None

    def pool(self, role: str, size: int, round_idx: int = 0, seed: int = 0) -> TaskPool:
        if self._dataset_ids:
            ids = dataset_pool(self, role, size, round_idx, seed)
        else:
            ids = (self._fixed_eval_ids(size, seed) if role == 'EVAL'
                   else [f'{self.prefix}-{role}-{seed}-r{round_idx}-{j}' for j in range(size)])
        return TaskPool(role, [Task(i, 'primary', make_payload(i)) for i in ids])

    def _fixed_eval_ids(self, size, seed):
        return [f'{self.prefix}-eval-{seed}-{j}' for j in range(size)]

    def execute(self, config: dict, task: Task, patch_keys=None,
                parent_version: str | None = None,
                candidate_version: str | None = None,
                action_id: str | None = None) -> TaskResult:
        boundary, utility = self.env.execute(config, task)
        activated = (self.env.activation(config, task, patch_keys)
                     if patch_keys is not None else None)
        return TaskResult(task.task_id, boundary, utility, activated=activated)

    def ground_truth(self) -> dict:
        return {'faults': [{'kind': f.kind, 'component': f.component, 'repair_key': f.repair_key,
                            'scope': f.scope} for f in self.env.faults],
                'root_components': sorted(self.env.root_components)}
