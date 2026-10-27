"""Explicitly run/resume an already frozen native E2/E3 suite."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.native_experiments import run_native_suite

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True)
    args = parser.parse_args()
    print(json.dumps(run_native_suite(args.suite),indent=2))
