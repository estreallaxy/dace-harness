"""Design/freeze native E2/E3 plans without launching model or task execution."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.agents.policy import SourcePolicy
from dace.core import Artifact
from dace.native_experiments import NativeProfile,prepare_native_suite,suite_design

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile',help='JSON NativeProfile fields')
    parser.add_argument('--design-only',action='store_true')
    parser.add_argument('--runtime-lock')
    parser.add_argument('--command',help='JSON runner argv')
    parser.add_argument('--model',default='deepseek/deepseek-v4.1-flash')
    parser.add_argument('--replicates',default='0,1,2')
    parser.add_argument('--study',default='native-e2-e3')
    args = parser.parse_args()
    profile = NativeProfile(**json.loads(Path(args.profile).read_text(encoding='utf-8'))) if args.profile else NativeProfile()
    output = Path(args.output).resolve()
    if args.design_only:
        from dace.native_experiments import _write_frozen
        design = suite_design(args.dataset,profile)
        _write_frozen(output/'design.json',design)
        print(json.dumps({'runtime_tasks':len(design['runtime_tasks']),'conditions':len(design['conditions'])}))
    else:
        if not args.runtime_lock or not args.command:
            parser.error('--runtime-lock and --command are required to freeze executable plans')
        manifest = prepare_native_suite(Artifact.create(SourcePolicy().content,kind='BASELINE'),
            args.dataset,json.loads(args.command),args.model,
            json.loads(Path(args.runtime_lock).read_text(encoding='utf-8')),output,
            profile=profile,replicates=tuple(int(x) for x in args.replicates.split(',')),study_id=args.study)
        print(json.dumps({'runs':len(manifest['entries']),'reserved_units':manifest['total_reserved_run_units']}))
