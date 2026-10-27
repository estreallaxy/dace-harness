"""Build the live-run dataset subset from successfully baked TB2 tasks.

The live runner can only execute tasks whose images were baked
(dace-live/<task>:ready); the experiment's task pools sample randomly from
the dataset, so the live dataset must contain EXACTLY the baked tasks.
Copies (not symlinks) keep the subset self-contained and docker-mount safe.

Usage:
  python tools/build_live_subset.py [--dataset datasets/terminal-bench-2]
      [--out datasets/terminal-bench-2-live] [--baked-prefix dace-live]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def docker_images(prefix: str) -> set:
    proc = subprocess.run(['docker', 'images', '--format', '{{.Repository}}'],
                          capture_output=True, text=True, timeout=60, shell=False)
    if proc.returncode != 0:
        raise RuntimeError('Docker image inventory failed; existing datasets were not changed')
    return {name for name in proc.stdout.splitlines() if name.startswith(prefix + '/')}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='datasets/terminal-bench-2')
    ap.add_argument('--out', default='datasets/terminal-bench-2-live')
    ap.add_argument('--baked-prefix', default='dace-live')
    args = ap.parse_args()
    root, out = Path(args.dataset), Path(args.out)
    baked = docker_images(args.baked_prefix)
    tasks = sorted(d.name for d in root.iterdir()
                   if d.is_dir() and (d / 'task.toml').is_file()
                   and f'{args.baked_prefix}/{d.name}' in baked)
    # Never erase a user-selected dataset directory or an existing task split.
    # Bake success is an infrastructure filter, not a representative TB2 split.
    if out.exists():
        raise ValueError('Output already exists; use a fresh subset directory')
    if not tasks:
        raise ValueError('No prepared tasks; refusing to create an empty subset')
    out.mkdir(parents=True)
    for name in tasks:
        shutil.copytree(root / name, out / name)
    manifest = {'out': str(out), 'tasks': len(tasks), 'names': tasks,
                'selection': 'infrastructure-availability',
                'representative_benchmark_split': False}
    (out / 'subset-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0 if len(tasks) >= 22 else 1


if __name__ == '__main__':
    raise SystemExit(main())
