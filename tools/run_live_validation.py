"""Evolve an explicitly supplied instrumented real agent on TB2.

This entry point no longer defaults to the official-solution oracle. A custom
runner is mandatory and must implement DACE's edits, activation events and
component hooks. For the real stock Terminus baseline use run_tb2_baseline.py.
No full DACE-on-TB2 claim is made without that editable-agent integration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dace.benchmarks.terminalbench import TerminalBench2  # noqa: E402
from dace.components import AblationLevel  # noqa: E402
from dace.evolution import EvolutionLoop, MethodSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runner-command', required=True,
                    help='JSON argv for an instrumented real editable agent; stock Terminus baseline is run_tb2_baseline.py')
    ap.add_argument('--level', default='C4')
    ap.add_argument('--scenario', default='source_error')
    ap.add_argument('--llm', action='store_true',
                    help='use the real deepseek LLM for mechanism proposals')
    ap.add_argument('--output', default='runs/live')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dataset', default='datasets/terminal-bench-2')
    ap.add_argument('--model', default='deepseek/deepseek-v4.1-flash')
    ap.add_argument('--diag-tasks', type=int, default=4)
    ap.add_argument('--confirm-tasks', type=int, default=8)
    ap.add_argument('--selection-tasks', type=int, default=8)
    ap.add_argument('--eval-tasks', type=int, default=8)
    ap.add_argument('--intervention-budget', type=int, default=16)
    ap.add_argument('--rounds', type=int, default=1)
    args = ap.parse_args()

    level = AblationLevel(args.level)
    out = Path(args.output) / (f'{args.level}-{args.scenario}-seed{args.seed}'
                              + ('-llm' if args.llm else ''))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'output directory not empty: {out}; use a fresh one')

    bench = TerminalBench2(
        scenario_name=args.scenario,
        dataset_path=str(REPO_ROOT / args.dataset),
        mode='live',
        command_template=args.runner_command,
        work_dir=str(Path(args.output) / 'work'),
        model=args.model)
    print(f'real dataset tasks: {len(bench._dataset_ids)} (mode={bench.mode})')

    spec = MethodSpec(f'live-{args.level}', kind='dace', level=level,
                      rounds=args.rounds,
                      diag_tasks=args.diag_tasks,
                      confirm_tasks=args.confirm_tasks,
                      selection_tasks=args.selection_tasks,
                      eval_tasks=args.eval_tasks,
                      intervention_budget=args.intervention_budget)
    llm = None
    if args.llm:
        from dace.llm import LLM
        llm = LLM(args.model)
        print(f'live LLM: {args.model} via OpenRouter')

    from dace.experiments.runner import _preflight_capacity
    _preflight_capacity(bench, spec)
    loop = EvolutionLoop(bench, spec, args.seed, out, llm=llm)
    t0 = time.time()
    summary = loop.run()
    dt = time.time() - t0

    print(f'\n=== LIVE RUN DONE in {dt/60:.1f} min ===')
    print(f'benchmark: {summary.benchmark} (mode={summary.mode}, '
          f'{len(bench._dataset_ids)} real tasks)')
    print(f'method: {summary.method} level={summary.level} rounds={args.rounds}')
    print(f'baseline_eval={summary.baseline_eval} final_eval={summary.final_eval}')
    print(f'budget_spent={summary.budget_spent} llm_calls={summary.llm_usage["calls"]}')
    print(f'final_config: { {k: v for k, v in summary.final_config.items() if v} }')
    print(f'promotions: {json.dumps(summary.promotions, ensure_ascii=False)}')
    for m in summary.mechanism_states:
        print(f"  mechanism {m['id']}: f={m['f_status']} r={m['r_status']} "
              f"z={m['z_bridge']}")
    for r in summary.rounds:
        print(f"  round {r.round_idx}: contracts={r.contracts_frozen} "
              f"probes={r.probes} accepted={r.accepted} "
              f"promoted_keys={r.promoted_keys} promoted_as={r.promoted_as}")
    events = [json.loads(line) for line in
              Path(summary.ledger_path).read_text(encoding='utf-8').splitlines()]
    kinds = [e['kind'] for e in events]
    for k in ('SELECTION_FROZEN', 'PROMOTION', 'PROMOTION_ATTRIBUTION',
              'PROMOTION_BLOCKED_CONFIRMATION_FAILED'):
        if k in kinds:
            print(f'  ledger has {k}: {kinds.count(k)}x')
    return 0


if __name__ == '__main__':
    sys.exit(main())
