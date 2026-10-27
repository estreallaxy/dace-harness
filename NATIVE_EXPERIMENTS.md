# Native experiment preparation

These entry points prepare E2/E3 using the native v3 optimizer. Preparation and controlled parser probes make no model or task calls; they are not formal results.

```sh
python tools/prepare_native_suite.py --dataset datasets/terminal-bench-2 --output runs/design --design-only
```

The default two-round profile requires 42 distinct tasks: 2 DIAG, 16 CONFIRM, 16 SELECT and 8 sampled EVAL. Any remaining tasks are also held out as EVAL, never recycled into optimization. Metadata categories define selection families; absent selection families have no regression guarantee. Set `family_mode` to `aggregate` explicitly for datasets without categories.

E2 has seven unique configurations, C0 through C5 plus Contribution-only. Failure-only aliases C2; Uncoupled Dual aliases C3. E3 changes the intervention cap through 1/2/4 and the active mechanism cap through 2/4/8, one factor at a time. Its shared midpoint is reused. Together E2/E3 need 11 unique configurations, each repeated with the same frozen role partition. Replicate IDs change within-role ordering; they do not control hosted model randomness.

After preparing all runtime task images, freeze a runtime lock covering exactly the `runtime_tasks` listed in `design.json`. Then prepare executable plans:

```sh
python tools/prepare_native_suite.py --dataset datasets/terminal-bench-2 --output runs/native-suite --runtime-lock runtime-lock.json --command '["python", "tools/tb2_live_runner.py", "--harbor", "PATH_TO_HARBOR", "--max-turns", "8"]'
python tools/run_native_suite.py --suite runs/native-suite/suite.json
```

Use absolute executable/script paths when freezing the actual runner argv. The command JSON must match the runtime lock. `--profile` accepts a JSON object of `NativeProfile` fields; `--replicates` defaults to `0,1,2`. Plan creation reserves ceilings, not provider dollars. Running the suite explicitly incurs model/task costs; the dispatcher verifies all plans first and resumes each existing run under its own durable contracts. It does not overwrite changed protocols or recycle consumed tasks.

An optional `coding_request` profile is frozen identically into every condition, for example `{"coding_request":{"reasoning":{"enabled":false},"response_format":{"type":"json_object"}}}`. Unspecified controls retain the provider default. Check provider support first. Reasoning can share the completion-token ceiling; JSON mode does not justify accepting a length-limited reply. Non-final completions consume their allocation and terminate unresolved without retry.

Set `factorial_tasks` to a positive fixed count to reserve distinct four-configuration CONFIRM samples per intervention slot. These are additional tasks, not reused ordinary-confirmation samples. Multi-slot actions execute the four controls; unused slots keep their alpha allocation and release their unused budget. See [SOURCE_FACTORIAL.md](SOURCE_FACTORIAL.md) for evidence and C3/C4/C5 projection semantics. The default zero preserves the original 42-task design.

Default sample sizes are development settings, not a power calculation. Small confirmation samples can remain statistically unresolved and selection may reject all patches. Freeze adequate sample sizes and budgets before reporting scientific conclusions.

Controlled E4 preparation uses synthetic response fixtures with the pinned Harbor 0.25.0 parser:

```sh
python tools/probe_controlled_faults.py --responses examples/controlled-responses.json --output runs/fault-probes.json
```

The four constructors expose known response corruption, downstream command compensation, combined response/command faults, and deliberately incorrect initial localization. `command_count_preserved` detects dropped/added commands without asserting semantic correctness. Probe outputs include no official task rewards and run no terminal commands. C3/C5 optimizer comparison and real utility measurements remain required for E4.
