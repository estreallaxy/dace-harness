# Native four-configuration interaction evidence

`dace.source_factorial` executes the same CONFIRM task under four exact policies: baseline (00), first edit only (10), second edit only (01), and their joint edit (11). It reports the task-level contrast `Y11-Y10-Y01+Y00`, using official binary verifier rewards. This is contribution interaction evidence, not independent promotion, unique root-cause proof, or identification of a mediated path.

```python
from dace.source_factorial import make_factorial_plan, run_factorial

plan = make_factorial_plan(
    parent_artifact,
    {'response_code': 'def transform(value):\n    return value.strip()\n'},
    {'instruction_suffix': 'Use noninteractive terminal commands.'},
    fresh_confirm_task_ids, dataset_path, runner_argv, model,
    study_binding, runtime_lock, alpha=.05)
# Save plan.json before executing it; placeholders need actual frozen inputs.
result = run_factorial(plan, output_directory)
```

The two edit groups must be nonempty, disjoint, legal source-slot changes. Each single and combined policy is independently compiled before execution. No missing/illegal control is replaced with the parent or an inferred zero. Task roles, fingerprints, source/runtime identities and exact direct-parent artifacts are validated before paid work.

```sh
python tools/run_source_factorial.py --plan plan.json --output runs/factorial --batch 1
python tools/run_source_factorial.py --plan plan.json --output runs/factorial
```

Each task unit reserves and charges four execution units before any arm starts. Fixed order is 00,10,01,11. An interrupted partially committed unit terminates unresolved; no arm is replayed. A completed task is not executed again. No interaction decision is returned until the entire frozen sample is complete. A closed certificate recomputes the contrast from all raw verifier files and checks budget, provenance and activation receipts, ignoring mutable summary files.

The contrast lies in [-2,2]. A two-sided fixed-look Hoeffding interval uses radius `sqrt(8*log(2/alpha)/N)` and clips its endpoints to this range. It assumes independent task units and comparable frozen runtimes. One task is normally unresolved. Missing activation preserves the measured assignment contrast but blocks an attributed interaction label. Fixed execution order does not identify order effects; hosted-model randomness is not paired by a seed. Cluster-aware alternatives remain separate work.

## Native optimization integration

An optional `factorial_tasks` field in a v3 optimization plan freezes one independent CONFIRM task slot per round and intervention. For two rounds and one intervention per round, its shape is `[[['fresh-x0']], [['fresh-x1']]]`. These IDs must be disjoint from DIAG, ordinary CONFIRM, SELECT and EVAL IDs. The entire `4*N` ceiling for every slot is reserved before the first diagnostic call, and its alpha share is fixed alongside ordinary confirmation and selection.

When a chosen action changes at least two slots, the optimizer executes this stage after ordinary paired confirmation. The first factor is the alphabetically first changed slot; the second factor groups the remaining changed slots. The plan records this actual decomposition. Three changed slots therefore yield a two-group interaction, not a claim about all three-way effects. Single-slot actions leave the extra reservation unused; it is released without reallocating alpha.

C3 receives factorial contribution estimates without coupled source targets. C4/C5 combine the exact action's scoped failure status with the interaction status to propose a next verification/revision question. C0/C1/C2 retain operational receipts without exposing the typed interaction estimate. Four utility controls alone never set the ordinary mechanism bridge to `INTERACTION`; source/propagation linkage still requires its own evidence. Deployment marks old-parent interaction views stale.

For prepared E2/E3 suites, set the `NativeProfile.factorial_tasks` count in `--profile`. Zero keeps the original execution path. Positive counts allocate additional independent CONFIRM tasks and budget for every condition. These counts are fixed execution sample sizes, not an additional E3 sweep.
