"""Fingerprint execution inputs without reading a task's solution."""
import hashlib
from pathlib import Path

from ..core import digest


def task_fingerprint(path):
    root = Path(path).resolve()
    files = [root/'task.toml', root/'instruction.md']
    for folder in ('environment','tests'):
        base = root/folder
        if base.exists():
            files.extend(p for p in base.rglob('*') if p.is_file())
    hashes = {}
    for path in sorted(files):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Task inputs must stay inside their frozen task directory')
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(hashes)
