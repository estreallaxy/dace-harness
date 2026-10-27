"""Run a genuine model agent using Harbor's task lifecycle and verifier.

This module never reads a task solution or mounts the task tree into the
agent. Harbor owns environment creation, instruction delivery and verification.
The five synthetic configuration repairs are not implemented for Terminus;
reject them rather than pretend a configuration change altered its harness.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .live import read_reward
from ..core import Artifact
from ..agents.policy import SourcePolicy, SLOTS
from .process import run_owned_process
from .cleanup import cleanup_owned_job


def harbor_model(model: str) -> str:
    return model if model.startswith('openrouter/') else 'openrouter/' + model


def validate_request(request):
    if request.get('schema') != 'dace-runner/v1':
        raise ValueError('Unsupported request schema')
    config = request.get('config', {})
    if not isinstance(config, dict) or any(type(v) is not bool for v in config.values()):
        raise ValueError('Expected a boolean harness configuration')
    if any(config.values()):
        raise ValueError('UNSUPPORTED_HARNESS_EDIT: stock Terminus has no synthetic repair switches')
    source = request.get('harness_artifact')
    if source is not None:
        artifact = Artifact(**source)
        if (artifact != Artifact.create(artifact.content,artifact.parent,artifact.kind)
                or artifact.kind not in ('BASELINE','PATCH','JOINT_PATCH')):
            raise ValueError('Frozen source artifact identity mismatch')
        SourcePolicy(**json.loads(artifact.content))
        intervention = request.get('intervention')
        if intervention is not None:
            if (intervention.get('candidate_version') != artifact.version
                    or intervention.get('parent_version') != artifact.parent
                    or not intervention.get('action_id')
                    or not intervention.get('patch_keys')
                    or not set(intervention['patch_keys']) <= SLOTS):
                raise ValueError('Source intervention is not bound to this artifact')
    task = Path(request['task_path']).resolve()
    if task.name != request['task_id'] or not (task / 'task.toml').is_file():
        raise ValueError('Task identity/manifest mismatch')
    if not request.get('model'):
        raise ValueError('An explicit execution model is required')
    return task


def collect_trial(job: Path, request: dict):
    trials = []
    for path in job.rglob('result.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'trial_name' in data and 'agent_info' in data:
            trials.append((path, data))
    if len(trials) != 1:
        raise ValueError('Expected exactly one Harbor trial, never an aggregate score')
    path, data = trials[0]
    if data.get('exception_info') or any(s.get('exception_info') for s in data.get('step_results') or []):
        raise RuntimeError('HARBOR_TRIAL_UNRESOLVED: see trial exception artifacts')
    if data.get('task_name', '').split('/')[-1] != request['task_id']:
        raise ValueError('Harbor result task mismatch')
    info = data['agent_info']
    expected_agent = 'dace-terminus' if request.get('harness_artifact') else 'terminus-2'
    if info.get('name') != expected_agent:
        raise ValueError('Unexpected agent: oracle and nop results are forbidden')
    model_info = info.get('model_info') or {}
    name = model_info.get('name')
    if name not in (request['model'], harbor_model(request['model'])):
        raise ValueError('Harbor result model mismatch')
    context = data.get('agent_result') or {}
    token_counts = [context.get('n_input_tokens'), context.get('n_output_tokens')]
    if sum(n for n in token_counts if type(n) is int and n >= 0) <= 0:
        raise ValueError('No recorded execution-model tokens: not a validated agent run')
    verifier = path.parent / 'verifier'
    reward_path = verifier / 'reward.json'
    if not reward_path.is_file():
        reward_path = verifier / 'reward.txt'
    reward = read_reward(reward_path)
    reported = (data.get('verifier_result') or {}).get('rewards') or {}
    if reported.get('reward') != reward:
        raise ValueError('Official verifier artifact and trial reward disagree')
    return path, reward_path, context, data


def run_request(request, response_path, harbor='harbor', max_turns=8, timeout=1800):
    task = validate_request(request)
    if max_turns <= 0 or timeout <= 0:
        raise ValueError('Execution limits must be positive')
    executable = shutil.which(harbor)
    if not executable:
        raise RuntimeError('HARBOR_NOT_INSTALLED: install the pinned runtime and pass --harbor')
    output = Path(response_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    job = output.parent / 'harbor' / 'trial'
    if job.exists():
        raise ValueError('Refusing to reuse Harbor output')
    source = request.get('harness_artifact')
    agent = 'dace.agents.terminus:EditableTerminus' if source else 'terminus-2'
    cmd = [executable, 'run', '--path', str(task), '--agent', agent,
           '--model', harbor_model(request['model']), '--jobs-dir', str(job.parent),
           '--job-name', job.name, '--n-attempts', '1', '--n-concurrent', '1',
           '--max-retries', '0', '--ak', f'max_turns={max_turns}',
           '--ak', 'temperature=0', '--quiet']
    from ..llm import load_key
    env = dict(os.environ)
    env['OPENROUTER_API_KEY'] = load_key()
    env['PYTHONUTF8'] = '1'
    policy = None
    if source:
        policy = SourcePolicy(**json.loads(source['content']))
        policy_path, identity_path = output.parent/'policy.json', output.parent/'identity.json'
        policy_path.write_text(policy.content,encoding='utf-8')
        identity_path.write_text(json.dumps({'task_id':request['task_id'],
            'intervention':request.get('intervention')},indent=2),encoding='utf-8')
        cmd += ['--ak',f'dace_policy_path={policy_path}',
                '--ak',f'dace_expected_policy={policy.version}',
                '--ak',f'dace_identity_path={identity_path}']
        root = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH','')
    # Keys only enter the child environment, never its argv or request/config.
    (output.parent / 'command.json').write_text(json.dumps(cmd, indent=2), encoding='utf-8')
    with (output.parent / 'harbor.stdout.log').open('w', encoding='utf-8') as stdout, \
            (output.parent / 'harbor.stderr.log').open('w', encoding='utf-8') as stderr:
        try:
            proc = run_owned_process(cmd,env=env,stdout=stdout,stderr=stderr,timeout=timeout,shell=False)
        except BaseException:
            report = cleanup_owned_job(job,request)
            (output.parent/'cleanup.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            raise
    if proc.returncode:
        report = cleanup_owned_job(job,request)
        (output.parent/'cleanup.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise RuntimeError('HARBOR_EXECUTION_FAILED: see Harbor logs; no score fabricated')
    result_path, reward_path, usage, trial = collect_trial(job, request)
    trace = output.parent / 'trace.json'
    if source:
        native_trace = result_path.parent/'agent'/'dace-trace.json'
        observed = json.loads(native_trace.read_text(encoding='utf-8'))
        if (observed.get('task_id') != request['task_id']
                or observed.get('policy_version') != policy.version
                or observed.get('intervention') != request.get('intervention')):
            raise ValueError('Instrumented agent returned a foreign policy/trace identity')
    else:
        observed = {'task_id':request['task_id'], 'boundary':{},
            'intervention':request.get('intervention'), 'activation_events':{},
            'observability':'Stock Terminus has no DACE component hooks; unresolved'}
    observed['official_trial'] = str(result_path.relative_to(output.parent))
    trace.write_text(json.dumps(observed,indent=2),encoding='utf-8')
    response = {'schema':'dace-runner/v1', 'task_id':request['task_id'], 'model':request['model'],
        'verifier_path':str(reward_path.relative_to(output.parent)), 'trace_path':trace.name,
        'usage':usage, 'provenance':{'execution_kind':'harbor-agent',
            'agent':'dace-terminus' if source else 'terminus-2',
            'oracle_access':False, 'benchmark_score_eligible':True,
            'reward_source':'official-verifier', 'task_checksum':trial.get('task_checksum'),
            'trial_result':str(result_path.relative_to(output.parent)),
            'harness_artifact':source['version'] if source else None,
            'policy_version':policy.version if policy else None}}
    output.write_text(json.dumps(response, indent=2), encoding='utf-8')
    return response
