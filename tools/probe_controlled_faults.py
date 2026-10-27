"""Run the four known native E4-preparation fault scenarios; no model/task calls."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.controlled_faults import run_controlled_faults

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses',required=True,help='JSON [{id,response}] fixtures')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    report = run_controlled_faults(json.loads(Path(args.responses).read_text(encoding='utf-8')))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({'scenarios':len(report['scenarios']),'model_calls':0,'task_executions':0}))
