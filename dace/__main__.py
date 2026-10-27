"""CLI: legacy demo + experiment drivers E1-E5.

Examples:
    python -m dace                                   # offline deterministic demo
    python -m dace --llm                             # demo with live patch synthesis
    python -m dace --check-api                       # verify OpenRouter access
    python -m dace run-experiment --exp e1           # E1 on Terminal-Bench 2.0 (mock)
    python -m dace run-experiment --exp e2 --seeds 0 1
    python -m dace run-experiment --exp e3
    python -m dace run-experiment --exp e4
    python -m dace run-experiment --exp e5
    python -m dace run-experiment --exp e1 --live    # live LLM proposals (deepseek-v4.1-flash)
    python -m dace run-experiment --exp e1 --benchmark terminalbench2 \
        --dataset-path D:/path/to/terminal-bench-2 --mode live \
        --command-template "tb run --task {task_id} --config {config_path}"
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from .core import Artifact, Budget, Contract, Controller, Exposure, Ledger, Observation, Prediction, promotion
from .llm import LLM


def execute(config, task):
    count = str(int(task.rsplit('-', 1)[1]) % 7 + 1)
    planned = int(count) if config['planner_cast'] else count
    adapted = int(planned) if config['adapter_cast'] else planned
    return {'planner_valid': type(planned) is int, 'adapter_valid': type(adapted) is int,
            'tool_success': type(adapted) is int}, float(type(adapted) is int)


def demo(args):
    output = args.output or Path(__file__).resolve().parents[1] / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh run directory; existing evidence will not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output / 'evidence.jsonl')
    budget = Budget(10000)
    exposure = Exposure(ledger)
    baseline = {'planner_cast': False, 'adapter_cast': False}
    parent = Artifact.create(json.dumps(baseline, sort_keys=True), kind='BASELINE')
    budget.reserve('promotion', 2 * 2 * 300)
    traces = []
    for i in range(4):
        task = f'diag-{i}'
        budget.charge(1)
        exposure.run(task, 'DIAG', parent.version)
        boundary, score = execute(baseline, task)
        traces.append({'task': task, 'boundary': boundary, 'score': score})
    ledger.append('COMPONENT_GRAPH', version=parent.version,
                  nodes=['planning:execute.planned', 'adapter:execute.adapted', 'tool:execute.tool_success'],
                  edges=[['planning', 'adapter', 'data'], ['adapter', 'tool', 'data']])
    if args.llm:
        proposed = LLM(args.model, args.env_file).propose(traces, budget, exposure, parent.version)
        patch = proposed['patch']
    else:
        patch = {'planner_cast': True, 'adapter_cast': False}
        ledger.append('OFFLINE_PROPOSAL', note='Deterministic demo proposal, no LLM call')
    if patch == baseline:
        raise ValueError('No actual configuration diff')
    configs = [patch]
    compensation = {'planner_cast': False, 'adapter_cast': True}
    configs.append(compensation if patch != compensation else {'planner_cast': True, 'adapter_cast': False})
    controller = Controller(ledger, budget, exposure)
    results, artifacts = {}, {}
    for i, config in enumerate(configs):
        artifact = Artifact.create(json.dumps(config, sort_keys=True), parent.version)
        artifacts[artifact.version] = artifact
        source_repair = config['planner_cast']
        prediction = Prediction('planner-type-error', 'planner_valid', ('planning', 'adapter', 'tool'),
                                (('planner_valid', False), ('adapter_valid', False)),
                                (('planner_valid', source_repair), ('adapter_valid', True)),
                                (('tool_success', True),), 'SOURCE_REPAIR' if source_repair else 'COMPENSATION')
        contract = Contract(artifact.version, parent.version, 'positive-count',
                            tuple(f'confirm-{i}-{j}' for j in range(128)), (prediction,))
        ledger.append('PATCH_GATE', artifact=artifact.version, passed=True,
                      diff={k: v for k, v in config.items() if v != baseline[k]})
        controller.freeze(artifact, contract)

        def paired(task, config=config):
            before, u0 = execute(baseline, task)
            after, u1 = execute(config, task)
            return Observation(task, before, after, u0, u1, True)
        controller.resume(contract.id, paired, batch=8)
        results[artifact.version] = controller.resume(contract.id, paired)
    selection = tuple(f'select-{j}' for j in range(300))
    exposure.require_fresh(selection)
    ledger.append('SELECTION_FROZEN', candidates=list(artifacts), tasks=selection,
                   families=['integer-count'], alpha=.05, max_rounds=1,
                   estimator='paired Hoeffding, independent bounded task units')
    candidates = {}
    for version, artifact in artifacts.items():
        config = json.loads(artifact.content)
        pairs = []
        for task in selection:
            budget.charge(2, 'promotion')
            exposure.run(task, 'SELECT', version)
            _, u0 = execute(baseline, task)
            _, u1 = execute(config, task)
            pairs.append((task, 'integer-count', u0, u1))
        candidates[version] = {'pairs': pairs, 'cost': 1, 'size': sum(config.values())}
    selected = promotion(candidates, ['integer-count'])
    budget.release('promotion')
    ledger.append('PROMOTION', **selected)
    winner = selected['winner']
    if winner:
        (output / 'harness.json').write_text(artifacts[winner].content, encoding='utf-8')
        ledger.append('ANCHORS_STALE', previous=parent.version, current=winner,
                      note='Parent empirical claims are not inherited by promoted version')
    summary = {'mode': 'live-coding-agent' if args.llm else 'offline-demo',
               'attribution': results, 'promotion': selected,
               'budget': {'unit': 'execution units; LLM request reserves 10 units, not USD',
                          'spent': budget.spent, 'free': budget.free}, 'output': str(output)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def run_experiment(args):
    from .experiments import (ExperimentConfig, run_e1, run_e2, run_e3, run_e4, run_e5, EXPERIMENTS)
    if args.exp not in EXPERIMENTS:
        raise SystemExit(f'Unknown experiment {args.exp!r}; choose from {EXPERIMENTS}')
    try:
        cfg = ExperimentConfig(exp=args.exp, benchmark=args.benchmark,
                               scenario=args.scenario, seeds=tuple(args.seeds),
                               rounds=args.rounds, output=args.output,
                               dataset_path=args.dataset_path, mode=args.mode,
                               model=args.model,
                               command_template=args.command_template,
                               work_dir=str(args.work_dir) if args.work_dir else None,
                               diag_tasks=args.diag_tasks, confirm_tasks=args.confirm_tasks,
                               selection_tasks=args.selection_tasks, eval_tasks=args.eval_tasks)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    llm_factory = None
    if args.live:
        llm_factory = lambda: LLM(args.model, args.env_file)
    runners = {'e1': lambda: run_e1(cfg, llm_factory=llm_factory),
               'e2': lambda: run_e2(cfg, llm_factory=llm_factory),
               'e3': lambda: run_e3(cfg, llm_factory=llm_factory),
               'e4': lambda: run_e4(cfg, llm_factory=llm_factory),
               'e5': lambda: run_e5(cfg, llm_factory=llm_factory)}
    result = runners[args.exp]()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd')

    legacy = argparse.ArgumentParser(add_help=False)
    legacy.add_argument('--llm', action='store_true', help='Use live coding-agent configuration synthesis')
    legacy.add_argument('--check-api', action='store_true')
    legacy.add_argument('--model', default='deepseek/deepseek-v4.1-flash')
    legacy.add_argument('--env-file')
    legacy.add_argument('--output', type=Path)

    if _is_legacy_invocation():
        args = legacy.parse_args()
        if args.check_api:
            response, usage = LLM(args.model, args.env_file).complete(
                [{'role': 'user', 'content': 'Reply with exactly OK.'}], 16)
            print(json.dumps({'model': args.model, 'response': response, 'usage': usage}))
            return
        demo(args)
        return

    p = sub.add_parser('run-experiment', help='Run experiment E1-E5')
    p.add_argument('--exp', required=True, choices=['e1', 'e2', 'e3', 'e4', 'e5'])
    p.add_argument('--benchmark', default='terminalbench2')
    p.add_argument('--scenario', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--output', default='experiments_output')
    p.add_argument('--dataset-path', default=None)
    p.add_argument('--mode', default='mock', choices=['mock', 'live'])
    p.add_argument('--command-template', default=None,
                   help='live runner command as a JSON argv array (required for --mode live)')
    p.add_argument('--work-dir', type=Path, default=None,
                   help='scratch directory for live runner artifacts (required for --mode live)')
    p.add_argument('--live', action='store_true', help='Use live LLM mechanism proposals')
    p.add_argument('--model', default='deepseek/deepseek-v4.1-flash')
    p.add_argument('--env-file')
    p.add_argument('--diag-tasks', type=int, default=6)
    p.add_argument('--confirm-tasks', type=int, default=32)
    p.add_argument('--selection-tasks', type=int, default=48)
    p.add_argument('--eval-tasks', type=int, default=64)
    p.set_defaults(func=run_experiment)

    p2 = sub.add_parser('check-api', help='Verify OpenRouter access')
    p2.add_argument('--model', default='deepseek/deepseek-v4.1-flash')
    p2.add_argument('--env-file')
    p2.set_defaults(func=_check_api)

    args = parser.parse_args()
    if getattr(args, 'func', None):
        args.func(args)
    else:
        parser.print_help()


def _check_api(args):
    response, usage = LLM(args.model, args.env_file).complete(
        [{'role': 'user', 'content': 'Reply with exactly OK.'}], 16)
    print(json.dumps({'model': args.model, 'response': response, 'usage': usage}))


def _is_legacy_invocation():
    import sys
    argv = sys.argv[1:]
    if not argv:
        return True          # bare `python -m dace` -> offline demo
    return argv[0].startswith('-')   # legacy flags: --llm / --check-api / --model ...


if __name__ == '__main__':
    main()
