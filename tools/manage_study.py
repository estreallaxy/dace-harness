"""Freeze/audit task roles and register optimizer lineages and disclosures."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dace.core import digest
from dace.task_registry import TaskRegistry

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',required=True)
    commands = parser.add_subparsers(dest='operation',required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--id',required=True)
    freeze.add_argument('--dataset',required=True)
    freeze.add_argument('--allocation',required=True,help='JSON mapping roles to task ID lists')
    lineage = commands.add_parser('lineage')
    lineage.add_argument('--study',required=True)
    lineage.add_argument('--id',required=True)
    lineage.add_argument('--parent')
    disclose = commands.add_parser('disclose')
    disclose.add_argument('--study',required=True)
    disclose.add_argument('--lineage',required=True)
    disclose.add_argument('--task',action='append',required=True)
    disclose.add_argument('--field',action='append',required=True)
    commands.add_parser('audit')
    args = parser.parse_args()
    with TaskRegistry(args.registry) as registry:
        if args.operation=='freeze':
            identity = registry.create_study(args.id,args.dataset,
                json.loads(Path(args.allocation).read_text(encoding='utf-8')))
            print(json.dumps({'study':args.id,'protocol_hash':identity}))
        elif args.operation=='lineage':
            registry.create_lineage(args.study,args.id,args.parent)
            print(json.dumps({'lineage':args.id,'parent':args.parent}))
        elif args.operation=='disclose':
            registry.disclose(args.study,args.lineage,args.task,args.field)
            print(json.dumps({'disclosed_tasks':args.task,'fields':args.field}))
        else:
            events = registry.audit_events()
            print(json.dumps({'events':len(events),'events_hash':digest(events),
                'last_event':events[-1]['hash'] if events else None}))
