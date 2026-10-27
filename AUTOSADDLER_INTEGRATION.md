# Official AutoSaddler integration handoff

No official AutoSaddler adapter is implemented in this repository yet. The existing LocalityHeuristic is a development control and must never be used under the AutoSaddler name in E1 or E5.

Inspected upstream: [microsoft/AutoSaddler](https://github.com/microsoft/AutoSaddler/tree/e841543ce6cf6bdf690e65134b12a829ec9f66d1), commit `e841543ce6cf6bdf690e65134b12a829ec9f66d1`, package version 0.2.0, MIT, Python >=3.12,<3.15. This checkout was inspected, not vendored or installed as a required dependency. DACE's offline kernel remains Python 3.11+.

The inspected V2 extension boundary is concrete:

| Upstream interface | Integration work |
| --- | --- |
| `v2.core.ports.ScenarioComponents` | Supply harness space, evaluator, train-only evidence builder, prompt pack and disjoint train/development cases. |
| `v2.harness.component_map.ComponentMapHarnessSpace` | Represent the same three `SourcePolicy` slots as DACE; use its validator to enforce the policy restrictions before accepting edits. |
| `v2.core.ports.Evaluator.evaluate` | Materialize each candidate, execute the existing official Harbor runner, preserve missing/invalid rewards, account for each attempt and release materializations. |
| `EvaluationContext.attempt_sink` | Use completed/start/complete/fail receipts. Reject ambiguous started attempts instead of silently replaying paid work. |
| `v2.core.ports.AgentProvider.run` | Add a real DeepSeek/OpenRouter provider with usage reporting and schema validation; fake/Claude/Codex/Copilot providers do not establish the requested DeepSeek comparison. |
| `v2.prompting.models.SessionSpec` | Provide evolve, diagnose_patch and reflect contracts, grounded in the pinned upstream methodology and permitted training evidence. |
| `v2.plugins.api` and config registry | Register an independently distributed scenario plugin; preserve upstream provenance and identities. |

See the pinned [scenario integration guide](https://github.com/microsoft/AutoSaddler/blob/e841543ce6cf6bdf690e65134b12a829ec9f66d1/docs/scenario-integration.md). The native DACE E2/E3 suite generator intentionally does not emit E1 configurations before this integration exists.

Before declaring a reproduction, freeze whether the baseline targets the publication's original implementation or this later V2 engine. A successful V2 adapter would be an official-engine integration under the declared common edit surface; it would not by itself reproduce historical paper results. Record the chosen version, model, editable surface, task splits, optimization/evaluation budget and selection policy together.

Acceptance requires real-engine controlled tests, independently verified official task rewards, crash/resume tests without duplicate task/model calls, equal base harness and model, EVAL isolation, and a fresh real-model pilot before formal E1/E5 execution. A fake-provider smoke can establish API compatibility only.
