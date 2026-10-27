"""Execute/resume independent selection of confirmed real source artifacts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.source_selection import run_selection

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch',type=int)
    args = parser.parse_args()
    result = run_selection(json.loads(Path(args.plan).read_text(encoding='utf-8')),
                           args.output,args.batch)
    print(json.dumps({'status':result['status'],'winner':result['winner'],
                      'budget':result['budget']},indent=2))
