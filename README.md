# DACE — intervention-grounded harness evolution

Research implementation of dual failure/contribution attribution for agent harness evolution. Python 3.11+, with a standard-library offline kernel and an optional pinned Harbor runtime for real tasks.

**Current status:** the native diagnosis → source proposal → paired confirmation → independent selection loop is implemented, with persistent versioned mechanism state and optional blind evaluation. This repository does **not** contain a complete scientific reproduction or formal E1–E5 results. Official AutoSaddler and live GAIA2/SWE-Bench Pro integration remain open work. See [HANDOFF.md](HANDOFF.md) for exact boundaries and next steps.

## Quick start

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m dace
```

The default example is an offline simulator, not a benchmark result. Dataset-specific tests skip when the official local dataset is absent. The test suite does not require API credentials or Docker for its controlled fixtures.

For an explicit API check, set your own `OPENROUTER_API_KEY`, then run:

```sh
python -m dace --check-api
```

The default model is `deepseek/deepseek-v4.1-flash`. Credentials are read in memory from the environment, an explicit env-file, `DACE_ENV_FILE`, or an untracked `.env.local`. The optional ignored `.dace.local.json` supports a local `env_file` path. No credentials are bundled.

## Native Terminal-Bench execution

Install Harbor in a dedicated environment and make Docker available:

```sh
python -m pip install -r requirements-live.txt
```

Obtain the official Terminal-Bench 2.0 tasks separately under `datasets/terminal-bench-2` or another explicit path. Dataset files and solutions are not distributed here. The native runner expects Harbor-format task directories with `task.toml`, instructions, environment and official verifier files; optimizer paths never read `solution/`.

1. Freeze the study's complete DIAG/CONFIRM/SELECT/EVAL partition with `tools/manage_study.py`.
2. Prepare the task images, then freeze a runtime lock with `tools/freeze_source_runtime.py`. Runtime locks bind source hashes, pinned Harbor/dependencies, command and local image identities.
3. Generate a plan using `dace.source_optimization.make_optimization_plan`. Use `level='C5'` for native v3; supply `eval_tasks` for blind checkpoints. Round rows explicitly list diagnostic tasks, per-action confirmation slots, selection tasks and family labels.
4. Run or resume the exact plan:

```sh
python tools/run_source_optimization.py --plan plan.json --output runs/study
python tools/report_native_run.py --run runs/study --output runs/study-report
```

Full preparation and recovery details: [SOURCE_OPTIMIZATION.md](SOURCE_OPTIMIZATION.md), [SOURCE_CONFIRMATION.md](SOURCE_CONFIRMATION.md), [SOURCE_SELECTION.md](SOURCE_SELECTION.md).

For deduplicated native E2/E3 plans and controlled E4 parser probes, see [NATIVE_EXPERIMENTS.md](NATIVE_EXPERIMENTS.md). These preparation tools do not produce benchmark performance results.

## Native mechanisms and budgets

- The editable slots are pure response transformations, pure command transformations and an instruction suffix. They alter the actual Terminus execution path. This initial edit surface is bounded; it is not arbitrary repository editing or an adversarial-code sandbox.
- `SourceState` retains immutable mechanism revisions and action/scope-specific response records across rounds. The active cap archives overflow according to a fixed policy while retaining all history. Deployment marks old-parent state stale.
- C0 does not construct a mechanism space. C1 adds persistent competition; C2 consumes failure evidence; C3 independently consumes failure and contribution; C4 adds typed joint-update targets; C5 ranks actions by explicit frozen prediction disagreement. Contribution-only masks failure evidence. Proposed joint targets are questions, not verified hidden faults.
- Global stage ceilings are reserved before task/model calls. Child charges are mirrored once in the global budget; they are not another monetary expense. Blind evaluation has separate reported spending.
- Started but uncommitted units end unresolved; they are not silently replayed. A terminal resume makes no new task or model calls. Changing locked source/runtime invalidates an active run.
- Only EVAL checkpoints become report curves. SELECT utility is never presented as held-out Pass@1. Exporting curves/cost breakdowns reuses journals without new execution.

## Statistics and provenance

New native statistical contracts freeze task-level exact binomial intervals, violation tolerance, alpha allocation and a fixed complete-sample look. Missing observations remain unresolved. Intervals assume independent identically distributed binary task units; cluster/stratified alternatives are not yet implemented. A matching trace is not unique-root-cause or mediated-effect identification.

Legacy v1 plans retain their original observed-pattern semantics and contract IDs. They are not retrospectively upgraded by new code. Controlled oracle paths require explicit opt-in and are excluded from real benchmark scoring.

A local known-fault native smoke completed five genuine DeepSeek/TB2 executions: diagnosis 0; confirmation 0/0; independent selection 0/1. One selection task was insufficient for promotion, so no patch was deployed. This is integration evidence, not formal benchmark performance; v3 multi-round native behavior still needs fresh real-model validation.

## Offline experiment preparation

```sh
python -m dace run-experiment --exp e2
python -m dace run-experiment --exp e3
python -m dace run-experiment --exp e4
python -m dace run-experiment --exp e5
```

These experiment drivers currently use the simulator. E3 uses two three-level one-factor sweeps with the shared midpoint reused. E5 reuses existing run ledgers. The named `LocalityHeuristic` development control is not official AutoSaddler. Live modes that lack verified adapters fail closed.

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing contracts, recovery or publishing results. Raw run artifacts, private configuration, server keys and local research manuscripts are intentionally ignored.
