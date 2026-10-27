"""Shared actual source-harness execution with auditable postflight failures."""
from dataclasses import asdict
from pathlib import Path

from .benchmarks.base import Task
from .benchmarks.fingerprint import task_fingerprint
from .benchmarks.live import LiveExecutor


class SourceExecution:
    def __init__(self,plan,ledger,output,identity):
        self.plan,self.ledger,self.identity = plan,ledger,dict(identity)
        self.runner = LiveExecutor(plan['command'],Path(output)/'executions',plan['model'])

    def verify(self,task):
        if task_fingerprint(Path(self.plan['dataset'])/task) != self.plan['task_fingerprints'][task]:
            raise ValueError('FROZEN_TASK_INPUTS_CHANGED')
        if 'runtime_lock' in self.plan:
            from .runtime_lock import verify_runtime_lock
            verify_runtime_lock(self.plan['runtime_lock'],self.plan['command'],self.plan['dataset'],
                                self.plan['task_fingerprints'])

    def execute(self,artifact,task,patch_keys=None,action_id=None):
        try:
            self.verify(task)
            result = self.runner.execute({},Task(task,'primary'),Path(self.plan['dataset'])/task,
                patch_keys=patch_keys,parent_version=artifact.parent if patch_keys else None,
                candidate_version=artifact.version if patch_keys else None,
                action_id=action_id,harness_artifact=artifact)
            self.ledger.append('SOURCE_EXECUTION_RESULT',**self.identity,
                task=task,artifact=artifact.version,result=asdict(result))
            try:
                self.verify(task)
            except Exception:
                self.ledger.append('SOURCE_EXECUTION_INVALIDATED',**self.identity,
                    task=task,artifact=artifact.version,reason='POSTFLIGHT_INPUT_OR_RUNTIME_DRIFT')
                raise
            return result
        except Exception as exc:
            self.ledger.append('SOURCE_EXECUTION_FAILED',**self.identity,
                task=task,artifact=artifact.version,error_type=type(exc).__name__)
            raise
