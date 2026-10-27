"""Freeze source runtime locally, without model calls or image downloads."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.runtime_lock import capture_runtime_lock

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command',required=True,help='JSON argv array')
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--task',action='append',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    lock = capture_runtime_lock(json.loads(args.command),args.dataset,args.task)
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(lock,stream,indent=2)
    print(path)
