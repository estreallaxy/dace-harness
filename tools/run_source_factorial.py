"""Run/resume an exact native four-configuration interaction contract."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.source_factorial import run_factorial

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch',type=int)
    args = parser.parse_args()
    result = run_factorial(json.loads(Path(args.plan).read_text(encoding='utf-8')),args.output,args.batch)
    print(json.dumps(result,indent=2))
