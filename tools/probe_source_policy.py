"""Run controlled source-policy probes with the installed native Harbor parser."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.agents.policy import SourcePolicy
from dace.agents.probe import probe_policy

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',required=True)
    parser.add_argument('--responses',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    report = probe_policy(SourcePolicy.read(args.policy),
        json.loads(Path(args.responses).read_text(encoding='utf-8')))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({'kind':report['kind'],'rows':len(report['rows']),
                      'benchmark_score_eligible':False}))
