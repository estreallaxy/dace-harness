# Contributing

Use Python 3.11 or newer. The offline kernel has no third-party runtime dependencies.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

Tests with a locally installed official Terminal-Bench dataset are skipped when it is absent. Mock task execution remains mock even when it uses real dataset IDs. Native execution additionally requires the pinned Harbor runtime and Docker.

Do not commit `.env.local`, keys, datasets, task solutions, raw trajectories, local study databases, or run directories. Use your own OpenRouter credentials via `OPENROUTER_API_KEY` or `DACE_ENV_FILE`.

An execution plan is immutable after freezing. Do not modify runtime-locked sources during an active run. New fixes require a new runtime lock; failed or started-but-uncommitted task units must not be silently replayed or replaced. Evidence from old protocols retains its original meaning.

Any change to statistical decisions must test missing observations, one-sample behavior, independent fixed looks, multiple-testing allocation, artifact/scope identity, and interrupted recovery. Never treat parser success, oracle execution, infrastructure failure, SELECT utility, or a mocked executor as a formal benchmark score.

Keep public documentation precise about tested scope. A verified operational contract is not unique-root-cause identification; a selected patch is not a held-out evaluation result. See `HANDOFF.md` before continuing experiment work.
