"""Run a single genuine TB2 baseline task; not a completed DACE experiment."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dace.benchmarks.harbor_runner import run_request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True)
    p.add_argument('--dataset',type=Path,default=ROOT/'datasets/terminal-bench-2')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='deepseek/deepseek-v4.1-flash')
    p.add_argument('--harbor',default='harbor')
    p.add_argument('--max-turns',type=int,default=8)
    p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use a fresh output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    request={'schema':'dace-runner/v1','task_id':a.task,
             'task_path':str((a.dataset/a.task).resolve()),'model':a.model,'config':{},
             'intervention':None}
    (a.output/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    response=run_request(request,a.output/'response.json',harbor=a.harbor,
                          max_turns=a.max_turns,timeout=a.timeout)
    print(json.dumps(response,indent=2))

if __name__=='__main__': main()
