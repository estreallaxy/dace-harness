# Working on DACE

Read `HANDOFF.md` before claiming experiment readiness. Keep simulator tests,
controlled native fixtures, real integration smokes, and formal benchmark
results distinct. LocalityHeuristic is never official AutoSaddler.

Run `python -m unittest discover -s tests -v`. Tests should not need credentials
or Docker unless explicitly marked as optional native integration tests.

Preserve frozen plans, source/runtime identities, task-role isolation, child
ledger receipts, exact parent/action/scope binding, and no-retry semantics for
ambiguous started units. Never feed EVAL into optimizer context. Never promote
from SELECT summaries or replace official verifier rewards with inferred scores.

Do not edit a source tree while a native experiment has locked it. Develop in
an isolated copy, then apply changes only after the old run has ended and its
evidence has been verified. Do not upgrade historical v1 evidence to new
statistical or causal meanings.

Keep private keys, credentials, raw trajectories, datasets/solutions, local
databases, and unpublished manuscript files out of Git. `.dace.local.json` is
machine-local configuration. Do not publish `runs/`.

When a feature is only implemented or tested on the simulator, document that
limit explicitly. Missing live adapters must fail closed. Joint-update targets
are proposed questions, not facts about hidden faults or mediated effects.
