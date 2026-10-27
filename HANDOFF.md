# Research implementation handoff

## Implemented execution paths

- Offline deterministic harness and E1–E5 experiment drivers: `dace/evolution.py`, `dace/components.py`, `dace/experiments/`.
- Genuine Harbor 0.25.0 + Terminus2 execution: `dace/benchmarks/harbor_runner.py`; official verifier artifacts are the only native utility source.
- Editable pure response/command transformations and instruction suffix: `dace/agents/policy.py`, `dace/agents/terminus.py`.
- Durable source diagnosis/proposal, paired confirmation, independent selection, global multi-round budget: `source_diagnostics`, `source_confirmation`, `source_selection`, `source_optimization`.
- Statistical contracts: immutable task-level exact binomial fixed-look intervals, alpha allocation, missing-data handling, legacy-ID preservation.
- Native v3: persistent versioned CMVS, bounded active revisions with archived history, own/foreign response separation, stale parent invalidation, projected C0–C5/Contribution-only feedback and typed proposed joint targets.
- Optional blind EVAL checkpoints: `eval_tasks` in a v3 plan. EVAL data never enters Coding Agent context; evaluation and optimization budgets are reported separately.
- Offline native report export: `tools/report_native_run.py`; exports `curve.csv`, `stages.csv`, `report.json` from verified child receipts, without task reruns or duplicate monetary accounting.

## Start here

Read `SOURCE_OPTIMIZATION.md`, `SOURCE_CONFIRMATION.md`, and `SOURCE_SELECTION.md`. Generate plans with `make_optimization_plan(..., level='C5', eval_tasks=[...])` for the new v3 behavior. Other levels are `C0`, `C1`, `C2`, `C3`, `C4`, `CONTRIBUTION_ONLY`. Omitting `level` retains the older v2 execution path for compatibility; it does not enable persistent native CMVS.

The three native edit slots are deliberately bounded. This repository does not yet edit arbitrary repository files or execute unconstrained generated Python. It does not read official task solutions for optimization. The old oracle path is explicitly marked controlled and excluded from benchmark scoring.

At every ablation level, basic paired verification outcomes remain available for reflection. C0 never constructs a mechanism space. Typed failure/contribution/bridge evidence is separately projected by level; basic verification feedback must not be confused with enabling contribution attribution.

## Evidence boundaries

Before the v3 additions, a local controlled-fault smoke completed five real DeepSeek/TB2 task executions: diagnosis 0; confirmation baseline/candidate 0/0; independent selection 0/1. The single selection task did not pass the frozen promotion bound, so no deployment occurred. No task/LLM call was repeated on terminal recovery. These observations are integration evidence, not full-benchmark Pass@1 or a formal E1 result.

Native v3 state, ablation projections, and blind evaluation are covered by controlled executor tests. They still need fresh real-model multi-round validation. Raw local trajectories, dataset files, keys, private server configuration, and full research manuscript text are intentionally outside the public repository.

## Remaining research work — do not relabel as complete

1. **E1:** official AutoSaddler integration is absent. `LocalityHeuristic` is a named development control, never AutoSaddler. Implement/pin the actual optimizer before reporting the requested comparison. No full TB2, GAIA2, or SWE-Bench Pro main-experiment table exists.
2. **Native mechanism scope:** joint targets are controller-derived proposed questions. The Coding Agent can respond with new source hypotheses and combined slot edits, but arbitrary semantic scope splitting, held-out verification of new split predicates, four-configuration interaction identification, and retained candidate-child search are not complete native features.
3. **E2/E3:** run all native levels on the same frozen study/profile; verify model-driven behavior differences, not only masks. The active CMVS cap and confirmation-slot cap are independently frozen. No formal sensitivity results are supplied.
4. **E4:** add controlled native source, compensation, compound-fault, and wrong-localization scenarios comparing C3/C5. Current parser probes and unit fixtures are preparation, not a complete correctness table or Trace→intervention→joint update→next edit case study.
5. **E5:** native export now reuses checkpoint journals; multi-run confidence bands, official AutoSaddler comparison, and paper figures still require real comparable runs. Repeated EVAL checkpoints are not new independent samples.
6. **Other datasets:** GAIA2 and SWE-Bench Pro adapters currently reject live mode. Implement and verify their official execution/evaluation adapters; do not remove the exception and substitute simulator scores.
7. **Statistics:** exact binomial predicates require independent identically distributed binary task units. Implement a predeclared stratified/cluster-aware alternative where this assumption is unsuitable. No code here identifies mediated effects merely from trace order.

## Operational safeguards

Use separate DIAG/CONFIRM/SELECT/EVAL roles in the shared SQLite registry. Reserve every round's ceiling before diagnostic/model calls. Started ambiguous units terminate unresolved. Source changes invalidate the runtime lock. Host time limits must cover environment preparation, agent time and verifier time. Cleanup targets only precisely owned Docker projects. Recorded Harbor costs are not independently reconciled provider bills.

This handoff is explicit about remaining implementation and validation work; a green test suite is not a claim of complete scientific reproduction.

## Upstream references

Official AutoSaddler: <https://github.com/microsoft/AutoSaddler>. Its scenario-integration documentation describes the adapter boundary: <https://github.com/microsoft/AutoSaddler/blob/main/docs/scenario-integration.md>. Pin an upstream commit and implement the required benchmark/harness scenario instead of relabelling the local heuristic.
