"""Terminal-Bench dataset adapter with explicit mock/live execution modes."""
from __future__ import annotations
from pathlib import Path
from .base import Task, TaskPool, TaskResult, dataset_pool, make_payload
from .live import LiveExecutor
from ..components import scenario as make_scenario

MANIFEST_NAMES = ('task.toml','task.yaml','task.yml','task.json','terminal-bench-task.yaml')


class TerminalBench2:
    name = 'terminalbench2'
    prefix = 'tb2'

    def __init__(self, scenario_name=None, dataset_path=None, mode='mock',
                 command_template=None, work_dir=None,
                 model='deepseek/deepseek-v4.1-flash', **_):
        if mode not in ('mock','live'):
            raise ValueError('Unknown execution mode')
        self.scenario_name = scenario_name or 'benchmark_mix'
        self.env = make_scenario(self.scenario_name)
        self.dataset_path = Path(dataset_path).resolve() if dataset_path else None
        self.mode = mode
        self.model = model
        self.command_template = command_template
        self._dataset_ids = self._load_dataset()
        self._consumed, self._eval_ids = set(), None
        self.runner = None
        if mode == 'live':
            if not self._dataset_ids or not command_template or not work_dir:
                raise ValueError('LIVE_REQUIRES_DATASET_RUNNER_AND_WORKDIR: no mock fallback')
            self.runner = LiveExecutor(command_template, work_dir, model)

    def _load_dataset(self):
        if not self.dataset_path or not self.dataset_path.is_dir():
            return None
        ids = [p.name for p in sorted(self.dataset_path.iterdir()) if p.is_dir()
               and any((p / m).is_file() for m in MANIFEST_NAMES)]
        return ids or None

    @property
    def real_dataset(self):
        return self._dataset_ids is not None

    def pool(self, role, size, round_idx=0, seed=0):
        if size <= 0:
            raise ValueError('Pool size must be positive')
        if self._dataset_ids:
            ids = dataset_pool(self,role,size,round_idx,seed)
        else:
            ids = [f'{self.prefix}-eval-{seed}-{j}' if role == 'EVAL' else
                   f'{self.prefix}-{role}-{seed}-r{round_idx}-{j}' for j in range(size)]
        return TaskPool(role,[Task(i,'primary',make_payload(i)) for i in ids])

    def execute(self, config, task, patch_keys=None,
                parent_version=None, candidate_version=None, action_id=None):
        if self.mode == 'live':
            if task.task_id not in self._dataset_ids:
                raise ValueError('Unknown real task ID')
            # live activation comes ONLY from the runner's trace hook, bound
            # to THIS edit's identity (patch keys, parent/candidate versions,
            # frozen action id); the executor validates the trace echoes the
            # same identity and reports per-key activation events
            return self.runner.execute(config, task, self.dataset_path / task.task_id,
                                       patch_keys=patch_keys,
                                       parent_version=parent_version,
                                       candidate_version=candidate_version,
                                       action_id=action_id)
        boundary, utility = self.env.execute(config, task)
        activated = (self.env.activation(config, task, patch_keys)
                     if patch_keys is not None else None)
        return TaskResult(task.task_id, boundary, utility, activated=activated)

    def ground_truth(self):
        if self.mode == 'live':
            return {'faults':[], 'root_components':[], 'known':False}
        return {'faults':[{'kind':f.kind,'component':f.component,'repair_key':f.repair_key,'scope':f.scope}
                          for f in self.env.faults], 'root_components':sorted(self.env.root_components), 'known':True}
