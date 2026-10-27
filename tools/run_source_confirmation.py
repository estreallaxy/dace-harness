"""Execute/resume an exact frozen real-harness confirmation plan."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.source_confirmation import run_plan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch',type=int)
    parser.add_argument('--require-study',action='store_true')
    parser.add_argument('--require-runtime-lock',action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    if args.require_study and not plan.get('study'):
        parser.error('Study binding is required for this run')
    if args.require_runtime_lock and not plan.get('runtime_lock'):
        parser.error('Frozen runtime is required for this run')
    result = run_plan(plan,args.output,args.batch)
    print(json.dumps({'contract_id':result['contract_id'],'evidence':result['evidence'],
                      'budget':result['budget']},indent=2))
