"""Run/resume a frozen multi-round native source optimization plan."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.source_optimization import run_optimization

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = run_optimization(json.loads(Path(args.plan).read_text(encoding='utf-8')),args.output)
    print(json.dumps({'status':result['status'],'budget':result['budget'],
                      'deployed_version':result['deployed_artifact']['version']},indent=2))
