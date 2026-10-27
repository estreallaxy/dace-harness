"""Run/resume native diagnostic tasks and one source Coding Agent request."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.source_diagnostics import run_diagnostic_proposal

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch',type=int)
    args = parser.parse_args()
    result = run_diagnostic_proposal(json.loads(Path(args.plan).read_text(encoding='utf-8')),
                                     args.output,batch=args.batch)
    print(json.dumps({'status':result['status'],'budget':result['budget'],
                      'optimizer_usage':result['optimizer_usage']},indent=2))
