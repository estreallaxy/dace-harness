"""Export verified native EVAL curves/cost stages without rerunning tasks."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.source_reporting import write_native_report

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = write_native_report(args.run,args.output)
    print(json.dumps({'status':result['status'],'blind_checkpoints':result['n_blind_checkpoints']}))
