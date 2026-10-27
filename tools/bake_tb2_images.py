"""Bake real Terminal-Bench 2.0 task images for live DACE execution.

For each task directory: pull the task's docker image (from task.toml),
ensure python3 + pytest exist inside it, and commit the prepared image as
<baked-prefix>/<task-name>:ready. Live runs then execute the task's REAL
solution and REAL pytest suite with no network dependency per execution.

Usage:
  python tools/bake_tb2_images.py --dataset datasets/terminal-bench-2 \
      [--tasks fix-git,regex-log,...] [--jobs 4] [--baked-prefix dace-live]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SETUP = r'''
set -e
APT_OPTS='-o Acquire::http::Timeout=60 -o Acquire::Retries=5 -o Acquire::https::Timeout=60 -o Acquire::https::Retries=5'
if ! command -v python3 >/dev/null 2>&1; then
  (apt-get update $APT_OPTS && apt-get install -y $APT_OPTS python3 python3-pip) || \
    (apk add --no-cache --timeout 60 python3 py3-pip) || \
    (apt-get update $APT_OPTS && apt-get install -y $APT_OPTS python3 python3-pip)
fi
if python3 -c "import pytest" 2>/dev/null; then exit 0; fi
PIP_OPTS='--timeout 60 --retries 5'
pip3 install -q $PIP_OPTS pytest==8.4.1 || \
  pip3 install -q $PIP_OPTS --break-system-packages pytest==8.4.1 || \
  python3 -m pip install -q $PIP_OPTS pytest==8.4.1 || \
  python3 -m pip install -q $PIP_OPTS --break-system-packages pytest==8.4.1
python3 -c "import pytest; print('pytest', pytest.__version__)"
'''


def docker(args, timeout=1800):
    return subprocess.run(['docker'] + args, capture_output=True, text=True,
                          timeout=timeout, shell=False)


def bake(task_dir: Path, prefix: str) -> dict:
    name = task_dir.name
    toml = tomllib.loads((task_dir / 'task.toml').read_text(encoding='utf-8'))
    image = toml.get('environment', {}).get('docker_image')
    if not image:
        return {'task': name, 'ok': False, 'reason': 'no docker_image (local build)'}
    baked = f'{prefix}/{name}:ready'
    already = docker(['image', 'inspect', baked])
    if already.returncode == 0:
        return {'task': name, 'ok': True, 'reason': 'already baked'}
    pull = docker(['pull', image])
    if pull.returncode != 0:
        return {'task': name, 'ok': False, 'reason': 'pull failed'}
    setup_sh = f'{name}-setup.sh'
    Path(setup_sh).write_text(SETUP, encoding='utf-8', newline='\n')
    host_cwd = Path.cwd() / setup_sh
    try:
        proc = docker(['run', '--name', f'dace-bake-{name}',
                       '-v', f'{str(host_cwd).replace(chr(92), "/")}:/setup.sh:ro',
                       '--entrypoint', 'bash', image, '/setup.sh'])
        if proc.returncode != 0:
            return {'task': name, 'ok': False,
                    'reason': 'setup failed: ' + proc.stderr[-200:]}
        commit = docker(['commit', f'dace-bake-{name}', baked])
        if commit.returncode != 0:
            return {'task': name, 'ok': False, 'reason': 'commit failed'}
        return {'task': name, 'ok': True, 'reason': image}
    finally:
        Path(setup_sh).unlink(missing_ok=True)
        docker(['rm', '-f', f'dace-bake-{name}'])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='datasets/terminal-bench-2')
    ap.add_argument('--tasks', default=None, help='comma-separated subset')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--baked-prefix', default='dace-live')
    args = ap.parse_args()
    root = Path(args.dataset)
    task_dirs = sorted(d for d in root.iterdir()
                       if d.is_dir() and (d / 'task.toml').is_file())
    if args.tasks:
        want = set(args.tasks.split(','))
        task_dirs = [d for d in task_dirs if d.name in want]
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda d: bake(d, args.baked_prefix), task_dirs))
    ok = [r for r in results if r['ok']]
    bad = [r for r in results if not r['ok']]
    print(json.dumps({'baked': len(ok), 'failed': len(bad),
                      'ok_tasks': [r['task'] for r in ok],
                      'failures': bad}, indent=2))
    print(f'baked={len(ok)} failed={len(bad)}', file=sys.stderr)
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
