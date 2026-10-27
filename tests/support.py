"""Portable scratch directories; honor the local user's D: preference."""
import os
from pathlib import Path


def test_temp_dir():
    configured = os.environ.get('DACE_TEST_TEMP')
    if configured:
        root = Path(configured)
    elif os.name=='nt' and Path('D:/').exists():
        root = Path('D:/UserData/lulu/Temp')
    else:
        return None  # Standard TMPDIR/TEMP handling on other machines.
    root.mkdir(parents=True,exist_ok=True)
    return str(root)
