"""Strict process bridge. No shell, no score inferred from process exit status."""
from __future__ import annotations
import json
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from .base import TaskResult
from .process import run_owned_process


def read_reward(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Missing verifier reward artifact')
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        reward = data.get('reward') if isinstance(data, dict) else None
    else:
        reward = float(path.read_text(encoding='utf-8').strip())
    if type(reward) not in (int, float) or reward not in (0, 1):
        raise ValueError('Terminal-Bench Pass@1 requires a binary verifier reward')
    return float(reward)


class LiveExecutor:
    """A runner accepts --request and --response paths, and writes a JSON response.

    Response must reference the actual Harbor reward artifact and actual boundary
    trace, never synthesize component boundaries from the reward. Missing hooks
    are omitted and handled as unresolved. Commands are JSON argv arrays.

    Intervention runs carry the edit's identity in the request (patch_keys,
    parent_version, candidate_version, action_id). The runner's trace must
    echo that SAME identity and report a per-key ``activation_events`` map
    covering every patch key; the executor accepts activation ONLY when the
    identity matches and every key's event is present -- an old or unrelated
    edit firing its hook never counts for this edit, and missing/mismatched
    identity keeps activation unresolved (None).
    """
    def __init__(self, command, work_dir, model):
        self.argv = json.loads(command) if isinstance(command, str) else command
        if not isinstance(self.argv, list) or not self.argv or any(not isinstance(x, str) for x in self.argv):
            raise ValueError('Runner command must be a JSON argv array')
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model = model

    def execute(self, config, task, task_path, patch_keys=None,
                parent_version=None, candidate_version=None, action_id=None,
                harness_artifact=None):
        source = None
        if harness_artifact is not None:
            from ..core import Artifact
            from ..agents.policy import SourcePolicy
            if harness_artifact != Artifact.create(harness_artifact.content,
                    harness_artifact.parent, harness_artifact.kind):
                raise ValueError('Frozen source artifact hash mismatch')
            policy = SourcePolicy(**json.loads(harness_artifact.content))
            if patch_keys is not None and (candidate_version != harness_artifact.version
                    or parent_version != harness_artifact.parent):
                raise ValueError('Intervention does not identify the supplied source artifact')
            source = asdict(harness_artifact)
        work = self.work_dir / uuid.uuid4().hex
        work.mkdir()
        request = work / 'request.json'
        response = work / 'response.json'
        request.write_text(json.dumps({'schema':'dace-runner/v1','task_id':task.task_id,
            'task_path':str(task_path),'config':config,'model':self.model,
            'response_path':str(response),
            'harness_artifact':source,
            'intervention': None if patch_keys is None else {
                'patch_keys': list(patch_keys),
                'parent_version': parent_version,
                'candidate_version': candidate_version,
                'action_id': action_id,
            }}, sort_keys=True),encoding='utf-8')
        try:
            proc = run_owned_process(self.argv + ['--request',str(request),'--response',str(response)],
                cwd=work,capture_output=True,text=True,timeout=3600,shell=False)
        except subprocess.TimeoutExpired as exc:
            (work/'stdout.txt').write_text(exc.stdout or '',encoding='utf-8')
            (work/'stderr.txt').write_text(exc.stderr or '',encoding='utf-8')
            raise
        (work/'stdout.txt').write_text(proc.stdout,encoding='utf-8')
        (work/'stderr.txt').write_text(proc.stderr,encoding='utf-8')
        if proc.returncode != 0 or not response.is_file():
            raise RuntimeError(f'Benchmark runner failed or produced no response: {work}')
        data = json.loads(response.read_text(encoding='utf-8'))
        if data.get('schema') != 'dace-runner/v1' or data.get('task_id') != task.task_id or data.get('model') != self.model:
            raise ValueError('Runner identity/model mismatch')
        provenance = data.get('provenance') or {}
        if (provenance.get('execution_kind') != 'harbor-agent'
                or provenance.get('oracle_access') is not False
                or provenance.get('benchmark_score_eligible') is not True
                or provenance.get('reward_source') != 'official-verifier'):
            raise ValueError('NON_BENCHMARK_RESULT: oracle, synthetic or unidentified result rejected')
        if source and (provenance.get('harness_artifact') != harness_artifact.version
                or provenance.get('policy_version') != policy.version):
            raise ValueError('Runner used a different frozen harness policy')
        reward_path = (work / data['verifier_path']).resolve()
        trace_path = (work / data['trace_path']).resolve()
        if not reward_path.is_relative_to(work) or not trace_path.is_relative_to(work):
            raise ValueError('Runner artifacts must be inside its unique run directory')
        reward = read_reward(reward_path)
        trace = json.loads(trace_path.read_text(encoding='utf-8'))
        if trace.get('task_id') != task.task_id:
            raise ValueError('Trace task identity mismatch')
        if source and trace.get('policy_version') != policy.version:
            raise ValueError('Trace belongs to a different frozen harness policy')
        boundary = trace.get('boundary', {})
        activated = self._intervention_activation(trace, patch_keys, parent_version,
                                                   candidate_version, action_id)
        if type(boundary) is not dict or activated is not None and type(activated) is not bool:
            raise ValueError('Invalid trace hook observations')
        return TaskResult(task.task_id,boundary,reward,activated,data.get('usage',{}),str(reward_path),provenance)

    @staticmethod
    def _intervention_activation(trace, patch_keys, parent_version,
                                 candidate_version, action_id):
        """Bind the trace's activation evidence to THIS edit's identity.

        An intervention run (patch_keys given) is activated only when the
        trace echoes the exact parent/candidate/action identity of this edit
        AND reports an activation event for EVERY patch key, all True. Any
        missing key, non-boolean value, or identity mismatch leaves the
        activation UNRESOLVED (None) -- an activation fact without edit
        identity is meaningless, and a hook fired by an older edit must
        never be credited to this one. Non-intervention runs report None.
        """
        if patch_keys is None:
            return None
        echo = trace.get('intervention')
        if not isinstance(echo, dict):
            return None
        if (echo.get('patch_keys') != list(patch_keys)
                or echo.get('parent_version') != parent_version
                or echo.get('candidate_version') != candidate_version
                or echo.get('action_id') != action_id):
            return None
        events = trace.get('activation_events')
        if not isinstance(events, dict):
            return None
        for key in patch_keys:
            if type(events.get(key)) is not bool:
                return None
        return all(events[key] is True for key in patch_keys)
