"""CONTROLLED ORACLE ONLY. Never a model Benchmark result.

Protocol (see dace/benchmarks/live.py): the experiment's LiveExecutor invokes
this program with --request and --response paths. The request carries the
task (a real TB2 task directory), the harness config under test, the model
and -- for intervention runs -- the EXACT identity of the edit (patch keys,
parent/candidate versions, frozen action id).

For every request this runner performs a REAL docker execution of the task:
the four-stage DACE harness (tools/live_harness_stage.py) runs inside the
task's own image, really executes the task's solution and really runs the
task's own pytest suite for the verifier reward. The response references the
real verifier reward artifact and the real boundary/activation trace; the
trace echoes the intervention identity and reports per-key activation events
derived from execution facts (never from the configuration diff).

Fault scenarios mirror dace/components.py exactly (single source of truth):
the scenario selects which faults the harness injects, exactly like the
simulated environment's scenario table.

Usage (JSON argv array for the experiment's command_template):
  [ "<python>", "<repo>/tools/tb2_live_runner.py", "--scenario",
    "source_error", "--baked-prefix", "dace-live" ]

Images must be pre-baked (tools/bake_tb2_images.py): the task image plus
pytest installed, committed as <prefix>/<task-name>:ready.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dace.benchmarks.base import make_payload  # noqa: E402
from dace.components import scenario as make_scenario  # noqa: E402


def docker(args, timeout=1500):
    return subprocess.run(['docker'] + args, capture_output=True, text=True,
                          timeout=timeout, shell=False)


def mount_path(p: Path) -> str:
    """Windows-safe mount source (forward slashes)."""
    return str(Path(p).resolve()).replace('\\', '/')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-controlled-oracle', action='store_true', required=True)
    ap.add_argument('--request', required=True)
    ap.add_argument('--response', required=True)
    ap.add_argument('--scenario', default='source_error',
                    help='fault scenario (same table as dace.components.scenario)')
    ap.add_argument('--baked-prefix', default='dace-live')
    args = ap.parse_args()

    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    task_id = request['task_id']
    task_path = Path(request['task_path'])
    out_dir = Path(args.response).resolve().parent
    tools_dir = REPO_ROOT / 'tools'

    # fault set from the SHARED scenario table (never a runner-local variant)
    env = make_scenario(args.scenario)
    faults = [{'kind': f.kind, 'scope': f.scope} for f in env.faults]

    image = f'{args.baked_prefix}/{task_id}:ready'
    inspect = docker(['image', 'inspect', image])
    if inspect.returncode != 0:
        raise RuntimeError(f'BAKED_IMAGE_MISSING: {image}; run '
                           f'tools/bake_tb2_images.py first')

    payload = make_payload(task_id)
    intervention = request.get('intervention')
    argv = ['run', '--rm',
            '-v', f'{mount_path(task_path)}:/task:ro',
            '-v', f'{mount_path(task_path / "tests")}:/tests:ro',
            '-v', f'{mount_path(tools_dir)}:/harness:ro',
            '-v', f'{mount_path(out_dir)}:/out',
            '-e', f'DACE_TASK_ID={task_id}',
            '-e', 'DACE_CONTROLLED_ORACLE=1',
            '--entrypoint', 'python3', image,
            '/harness/oracle_harness_stage.py',
            json.dumps(request['config'], sort_keys=True),
            json.dumps(faults, sort_keys=True),
            json.dumps(payload, sort_keys=True),
            json.dumps(intervention, sort_keys=True) if intervention is not None else 'null']
    proc = docker(argv)
    (out_dir / 'docker_stdout.txt').write_text(proc.stdout, encoding='utf-8', errors='replace')
    (out_dir / 'docker_stderr.txt').write_text(proc.stderr, encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        raise RuntimeError(f'docker execution failed for {task_id} '
                           f'(exit {proc.returncode}); see {out_dir}')

    response = {
        'schema': 'dace-runner/v1',
        'task_id': task_id,
        'model': request['model'],
        'verifier_path': 'verifier/reward.txt',
        'trace_path': 'trace.json',
        'usage': {'docker_exit': proc.returncode},
        'provenance': {'execution_kind':'controlled-oracle','oracle_access':True,
                       'benchmark_score_eligible':False},
    }
    Path(args.response).write_text(
        json.dumps(response, indent=2, sort_keys=True), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
