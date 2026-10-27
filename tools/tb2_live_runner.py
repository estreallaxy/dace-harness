"""dace-runner/v1 bridge to real Harbor + Terminus 2 + execution model.

Oracle solutions are never read by this runner. Synthetic repairs are rejected
until a real editable agent implementation provides their semantics and hooks.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.benchmarks.harbor_runner import run_request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True)
    p.add_argument('--response',required=True)
    p.add_argument('--harbor',default='harbor')
    p.add_argument('--max-turns',type=int,default=8)
    p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args()
    run_request(json.loads(Path(a.request).read_text(encoding='utf-8')),a.response,
                harbor=a.harbor,max_turns=a.max_turns,timeout=a.timeout)

if __name__=='__main__':
    main()
