"""Freeze local runner implementation, installed runtime versions and images.

This records/preflights local image identity; it does not attest a removed
container's actual image or freeze network responses and hosted model weights.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

from .core import digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
import hashlib,json,sys
from importlib.metadata import distributions,distribution
versions=sorted((d.metadata['Name'],d.version) for d in distributions() if d.metadata['Name'])
dist=distribution('harbor')
sources={str(p):hashlib.sha256(dist.locate_file(p).read_bytes()).hexdigest()
         for p in dist.files or [] if str(p).replace('\\','/').startswith('harbor/') and str(p).endswith('.py')}
if not sources: raise RuntimeError('No auditable Harbor source files')
print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':versions,
                  'harbor_version':dist.version,'harbor_sources':sources},sort_keys=True))
'''


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_probe(python):
    proc = subprocess.run([str(python),'-I','-c',PROBE],capture_output=True,text=True,
                          timeout=60,shell=False)
    if proc.returncode:
        raise RuntimeError('RUNTIME_PROBE_FAILED: verify the pinned Harbor environment')
    data = json.loads(proc.stdout)
    if data.get('harbor_version') != '0.25.0':
        raise ValueError('Source execution requires Harbor 0.25.0')
    return data


def _image_identity(image):
    proc = subprocess.run(['docker','image','inspect',image,'--format',
        '{{json .Id}} {{json .RepoDigests}} {{json .Os}} {{json .Architecture}}'],
        capture_output=True,text=True,timeout=30,shell=False)
    if proc.returncode:
        raise RuntimeError('PINNED_IMAGE_UNAVAILABLE: prepare images before freezing a run')
    # Decode adjacent JSON values without printing image config or credentials.
    decoder,rest,values = json.JSONDecoder(),proc.stdout.strip(),[]
    while rest:
        value,end = decoder.raw_decode(rest)
        values.append(value)
        rest = rest[end:].lstrip()
    if len(values)!=4 or not isinstance(values[0],str) or not values[0].startswith('sha256:'):
        raise ValueError('Docker image identity is unavailable')
    return dict(zip(('id','repo_digests','os','architecture'),values))


def capture_runtime_lock(command,dataset,tasks,project=None):
    tasks = tuple(tasks)
    project = Path(project or PROJECT_ROOT).resolve()
    if project != PROJECT_ROOT.resolve():
        raise ValueError('Runtime lock must cover the executing DACE implementation')
    if (not isinstance(command,list) or command.count('--harbor')!=1
            or command.index('--harbor')+1 >= len(command)):
        raise ValueError('A runtime lock requires one explicit --harbor executable')
    executable = shutil.which(command[command.index('--harbor')+1])
    if executable is None:
        raise ValueError('Pinned Harbor executable not found')
    executable = Path(executable).resolve()
    python = executable.parent/('python.exe' if executable.suffix=='.exe' else 'python')
    if not python.is_file():
        raise ValueError('Harbor must use an inspectable dedicated Python environment')
    source_hashes = {}
    for folder in ('dace','tools'):
        for path in sorted((project/folder).rglob('*.py')):
            if '__pycache__' not in path.parts:
                source_hashes[path.relative_to(project).as_posix()] = _hash(path)
    if not source_hashes:
        raise ValueError('No project implementation found to freeze')
    argv_files = {str(Path(v).resolve()):_hash(v) for v in command if Path(v).is_file()}
    resolved_command = shutil.which(command[0])
    if not resolved_command:
        raise ValueError('Runner executable not found')
    argv_files[str(Path(resolved_command).resolve())] = _hash(resolved_command)
    images = {}
    root = Path(dataset).resolve()
    for task in tasks:
        path = (root/task).resolve()
        if path.parent != root or path.name != task:
            raise ValueError('Expected one frozen dataset task directory')
        manifest = tomllib.loads((path/'task.toml').read_text(encoding='utf-8'))
        image = manifest.get('environment',{}).get('docker_image')
        if not image:
            raise ValueError('BUILD_ONLY_TASK_UNPINNED: this lock requires a prepared Docker image')
        if image not in images:
            images[image] = _image_identity(image)
    return {'schema':'dace-runtime-lock/v2','project':str(project),
        'dataset':str(root),'tasks':sorted(set(tasks)),
        'command':list(command),'source_hashes':source_hashes,'argv_files':argv_files,
        'harbor_executable':str(executable),'harbor_launcher_sha256':_hash(executable),
        'runtime':_runtime_probe(python),'images':images,
        'image_assurance':'local image identity checked before/after execution; no container attestation'}


def verify_runtime_lock(lock,command,dataset,tasks):
    if lock.get('schema') != 'dace-runtime-lock/v2':
        raise ValueError('Unknown runtime lock schema')
    if (Path(dataset).resolve() != Path(lock['dataset']).resolve()
            or not set(tasks) <= set(lock['tasks'])):
        raise ValueError('Requested tasks are outside the frozen runtime scope')
    current = capture_runtime_lock(command,lock['dataset'],lock['tasks'],lock['project'])
    if digest(current) != digest(lock):
        raise ValueError('FROZEN_RUNTIME_CHANGED: implementation, dependencies or image drifted')
    return digest(lock)
