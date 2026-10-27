"""Harness evolution loops: DACE (ablation-configurable) and baselines.

One round of the DACE loop follows the method document's outer iteration:
diagnose (record run usage) -> propose mechanisms into the CMVS -> staged
evidence with confirm locks (failure / contribution / bridge claims, read
according to the ablation level) -> joint update -> independent blind paired
selection with simultaneous lower-bound promotion -> anchor invalidation.

Ablation semantics (E2):
  C0   patches from symptom triggers; utility-only verification; no CMVS
  C1   + shared competing mechanism space (cap K_active), utility-only
  C2   + confirmatory failure-source contracts (claims label the run's
        attribution record; selection stays utility-based)
  C3   + contribution claims read independently (Uncoupled Dual: failure and
        contribution never revise each other)
  C4   + coupled predictions name the most-upstream violated source as the
        bridge witness under a COMPENSATION hypothesis; a confirmed bridge
        demotes the patch mechanism and relocalizes the search upstream
        (joint update); joint root-repair patches become constructible
  C5   + discriminative scheduling: the same legal action is frozen with the
        predictions of the top competing mechanisms attached, ranked by
        prediction disagreement D(a;M), so one confirmatory contract
        discriminates between competing hypotheses

Mechanism premises (c_m in the method's mechanism tuple): a confirmatory
contract's tasks must satisfy the mechanism's applicability premise so the
frozen boundary expectations hold uniformly. Premises are inferred from
diagnostic traces only (upstream contract states plus task payload
predicates), never from confirmation data.

Statistical protocol (frozen before the first round): the total error rate
alpha is split Bonferroni-style across rounds, across confirm contracts and
across the predictions inside a contract; the promotion budget for the whole
run is locked before any confirmatory execution; selection pools are frozen
in the ledger before any selection task is executed.

Masked-benefit retention (section 2.4): when no candidate clears the
independent utility bound but a patch produced consistent source-repair
boundary evidence, the candidate is retained in an ISOLATED candidate
lineage -- the current harness is NOT modified and the candidate must still
win independent selection before any promotion. This is the method's
"retain the source-repair evidence, hypothesize the residual fault" rule;
it never bypasses the promotion constraint.

The LocalityHeuristic baseline (kind='locality') trusts the naive locality
summary (nearest component to the observed failure), flips one repair key,
verifies utility with a simple majority rule and reflects on rejection. It
is a hand-written locality-first heuristic, NOT the official AutoSaddler
optimizer. No mechanism space, no confirmatory contracts, no family-wise
regression control.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .components import (AblationLevel, LEVEL_FEATURES, MechanismRegistry,
                         Mechanism, candidate_patches, joint_compositions,
                         locality_patch, offline_propose, prediction_disagreement)
from .core import Artifact, Budget, Contract, Controller, Exposure, Ledger, Observation, Prediction, promotion
from .harness import BASELINE_CONFIG

COMPONENT_ORDER = {'planner': 0, 'adapter': 1, 'tool': 2, 'verifier': 3}
COMPONENT_FIELDS = {'planner': ('planner_valid', 'cache_fresh'), 'adapter': ('adapter_valid',),
                    'tool': ('tool_success',), 'verifier': ('verifier_agrees',)}
DOWNSTREAM_FIELD = 'tool_success'
MIN_PREMISE_TASKS = 8
#: paired confirmatory executions (parent + candidate) per task
PAIR_COST = 2
#: joint-update marker: this mechanism is the upstream candidate the search
#: relocalized to after a downstream compensation was confirmed (section 2.3)
RELOC_NOTE = 'relocalization candidate after downstream compensation'


@dataclass
class MethodSpec:
    name: str
    kind: str = 'dace'                 # 'dace' | 'locality'
    level: AblationLevel | None = AblationLevel.C5
    rounds: int = 3
    intervention_budget: int = 256     # per-round confirm/probe cost cap (E3)
    mechanism_cap: int = 4             # CMVS active cap K_active (E3)
    diag_tasks: int = 8
    confirm_tasks: int = 64
    selection_tasks: int = 256
    eval_tasks: int = 64
    max_candidates: int = 4
    cost_per_task: int = PAIR_COST
    alpha_total: float = 0.05
    #: FROZEN promotion-evidence policy: a candidate may enter independent
    #: selection only when its OWN confirmation contract reported
    #: MINIMAL_OBSERVED activation feedback for its exact artifact version
    #: (read back from the ledger bound to the contract's identity, never
    #: from in-memory state alone and never from an older contract's
    #: evidence for the same version).
    require_minimal_feedback: bool = True
    #: explicit opt-in for utility promotion of candidates whose edit never
    #: activated (genuinely unobservable execution); every such promotion is
    #: recorded with the frozen reason. Never a silent default.
    allow_unobserved_promotion: bool = False
    unobserved_promotion_reason: str = ''

    def __post_init__(self):
        if self.kind == 'autosaddler':
            raise ValueError('kind="autosaddler" is reserved for the real AutoSaddler '
                             'integration, which is not implemented; use kind="locality" '
                             'for the locality-first heuristic baseline')
        if self.kind not in ('dace', 'locality'):
            raise ValueError(f'Unknown method kind {self.kind!r}')
        if self.allow_unobserved_promotion and not self.unobserved_promotion_reason.strip():
            raise ValueError('allow_unobserved_promotion=True requires an explicit '
                             'unobserved_promotion_reason recorded in the frozen policy '
                             'and on every such promotion event')


@dataclass
class RoundRecord:
    round_idx: int
    spent_start: int
    spent_end: int
    eval_utility: float
    mechanisms_active: int
    contracts_frozen: int = 0
    probes: int = 0
    bridges: list = field(default_factory=list)
    promoted_keys: list = field(default_factory=list)
    #: winner artifact versions actually promoted this round -- ONE entry per
    #  accepted selection event (a joint patch is one event, not N key edits)
    promoted_artifacts: list = field(default_factory=list)
    #: method-accepted attribution label under which each promotion event
    #  deployed (SOURCE_REPAIR / COMPENSATION): a compensation promotion is
    #  reported as such and never counted as a source repair
    promoted_as: list = field(default_factory=list)
    retained_keys: list = field(default_factory=list)
    accepted: bool = False


@dataclass
class RunSummary:
    method: str
    benchmark: str
    scenario: str | None
    seed: int
    kind: str
    level: str
    final_config: dict
    config_history: list
    rounds: list
    baseline_eval: float
    final_eval: float
    localization: dict
    labeling: list
    checkpoints: list                 # (cumulative_spent, eval_utility) for E5
    mechanism_states: list
    retained_candidates: list         # patches held in the isolated lineage
    ground_truth: dict
    ledger_path: str
    budget_spent: int
    llm_usage: dict
    # run identity for cross-experiment aggregation (E5 conditions): the
    # execution mode, the model, and the frozen sample-size/budget profile
    mode: str = 'mock'
    model: str | None = None
    spec: dict = field(default_factory=dict)
    #: ablation feature flags ACTUALLY in force for this run -- experiment
    #: metrics must read the method's accepted (decision-level) claims, not
    #: the raw attribution evidence the underlying controller happened to
    #: compute. A level with bridges disabled accepts no bridge conclusion,
    #: whatever the raw labels say.
    result_kind: str = "mock"
    execution_usage: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    #: per promotion event: the keys deployed, the mechanism, the
    #: applicability scope and the method-accepted attribution label the
    #: winner was promoted under. Metrics align repair/compensation
    #: predictions with ground truth through THESE records (per
    #: (mechanism, artifact, scope)), never by flattening raw labeling.
    promotions: list = field(default_factory=list)


class EvolutionLoop:
    """Drives rounds of diagnose -> attribute -> select for one method spec."""

    def __init__(self, benchmark, method: MethodSpec, seed: int, output: Path,
                 llm=None, budget_total: int = 400000):
        self.benchmark, self.method, self.seed, self.llm = benchmark, method, seed, llm
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.output / 'evidence.jsonl')
        self.budget = Budget(budget_total)
        self.exposure = Exposure(self.ledger)
        self.controller = Controller(self.ledger, self.budget, self.exposure)
        self.registry = MechanismRegistry(cap=method.mechanism_cap)
        self.current = dict(BASELINE_CONFIG)
        self.root_artifact = Artifact.create(json.dumps(self.current, sort_keys=True), kind='BASELINE')
        self.features = (LEVEL_FEATURES[method.level] if method.kind == 'dace' and method.level
                         else dict(mechanisms=False, failure=False, contribution=False,
                                   bridge=False, probe=False))
        self.records: list[RoundRecord] = []
        self.labeling: list = []
        self.retained: list = []       # isolated candidate lineage (masked benefit)
        #: per promotion event: keys/mechanism/scope/accepted attribution
        #: label (the report-side record metrics align predictions through)
        self.promotions: list = []
        self.alpha_round = 0.05
        self.alpha_contract = 0.05
        #: artifact version -> parent version, for version-scoped claim
        #: queries (standing evidence must be bound to the current parent)
        self.artifact_parents: dict[str, str | None] = {self.root_artifact.version: None}

    def _execute(self, config, task, **kwargs):
        try:
            result = self.benchmark.execute(config, task, **kwargs)
        except Exception as exc:
            self.ledger.append('EXECUTION_FAILED', task=task.task_id,
                               config=dict(config), error_type=type(exc).__name__,
                               usage_status='UNKNOWN', action_id=kwargs.get('action_id'))
            raise
        if result.task_id != task.task_id:
            raise ValueError('Execution result task identity mismatch')
        self.ledger.append('EXECUTION_RESULT', task=task.task_id,
                           config=dict(config), utility=result.utility,
                           boundary=result.boundary, activated=result.activated,
                           usage=dict(result.usage or {}),
                           verifier_path=result.verifier_path,
                           provenance=dict(getattr(result, 'provenance', {}) or {}))
        return result

    def _usage_totals(self, kind):
        records = [e['payload'].get('usage') or {} for e in self.ledger.events
                   if e['kind'] == kind]
        input_field = 'prompt_tokens' if kind == 'AGENT_PROPOSAL' else 'n_input_tokens'
        output_field = 'completion_tokens' if kind == 'AGENT_PROPOSAL' else 'n_output_tokens'
        cost_field = 'cost' if kind == 'AGENT_PROPOSAL' else 'cost_usd'
        def total(field):
            values = [u[field] for u in records if type(u.get(field)) in (int, float)]
            return sum(values) if values and len(values) == len(records) else None
        return {'calls':len(records), 'prompt_tokens':total(input_field),
                'completion_tokens':total(output_field), 'cost_usd':total(cost_field),
                'accounting_complete':bool(records) and all(
                    total(f) is not None for f in (input_field, output_field, cost_field))}

    # ---------- task execution ----------

    def _task_map(self, pool):
        return {t.task_id: t for t in pool.tasks}

    def _diagnose(self, round_idx):
        pool = self.benchmark.pool('DIAG', self.method.diag_tasks, round_idx, self.seed)
        traces = []
        for t in pool.tasks:
            self.budget.charge(1)
            self.exposure.run(t.task_id, 'DIAG', self.root_artifact.version)
            result = self._execute(self.current, t)
            traces.append({'task': t.task_id, 'boundary': result.boundary,
                           'score': result.utility, 'payload': dict(t.payload)})
        summary = self.benchmark.env.summarize_traces(traces)
        self.ledger.append('DIAGNOSE', round=round_idx, config=self.current,
                           traces=[{k: t[k] for k in ('task', 'boundary', 'score')} for t in traces],
                           summary=summary)
        return summary, traces

    def _eval_utility(self, round_idx):
        pool = self.benchmark.pool('EVAL', self.method.eval_tasks, round_idx, self.seed)
        total = 0.0
        for t in pool.tasks:
            self.budget.charge(1)
            self.exposure.run(t.task_id, 'EVAL', self.root_artifact.version)
            total += self._execute(self.current, t).utility
        return total / max(1, len(pool.tasks))

    # ---------- mechanism premises ----------

    @staticmethod
    def _upstream_fields(component):
        order = ['planner', 'adapter', 'tool', 'verifier']
        out = []
        for c in order[:COMPONENT_ORDER[component]]:
            out.extend(COMPONENT_FIELDS[c])
        return out

    def _premise(self, mechanism, traces):
        """Infer the mechanism's applicability premise from diagnostic traces.

        Returns a payload predicate dict, or None when the premise holds for
        every task. Returns 'UNSUPPORTED' only when the mechanism's source
        field is never violated across the diagnostic pool (its failure claim
        is untestable this round) -- a field no diagnostic trace OBSERVED is
        equally untestable and is never imputed as False.

        The violation set is taken directly on the mechanism's own source
        field, independent of upstream soundness: an upstream fault that
        propagates into this field still makes the mechanism testable (the
        coupled attribution then decides whether its edit is a root repair or
        a downstream compensation).
        """
        V = [t for t in traces if t['boundary'].get(mechanism.source_field) is False]
        if not V:
            return 'UNSUPPORTED'
        ok = [t for t in traces if t['boundary'].get(mechanism.source_field) is True]
        pred = {}
        vals = {t['payload'].get('cacheable') for t in V}
        if vals == {True} and any(not t['payload'].get('cacheable') for t in ok):
            pred['cacheable'] = True
        mags = [t['payload'].get('magnitude', 0) for t in V]
        if mags and max(mags) < 500 and any(t['payload'].get('magnitude', 0) >= 500 for t in ok):
            pred['magnitude_lt'] = 500
        return pred or None

    @staticmethod
    def _matches_premise(task, pred):
        if not pred:
            return True
        if 'cacheable' in pred and task.payload.get('cacheable') is not pred['cacheable']:
            return False
        if 'magnitude_lt' in pred and not task.payload.get('magnitude', 0) < pred['magnitude_lt']:
            return False
        return True

    # ---------- prediction construction ----------

    @staticmethod
    def _witness_field(component):
        # The witness is the component's *direct* downstream boundary, not the
        # terminal outcome. Repairing a source must flip its immediate output
        # even when a further downstream fault still masks the task result --
        # otherwise a masked benefit (section 2.3) would wrongly contradict the
        # source-repair claim.
        return {'planner': 'adapter_valid', 'adapter': 'tool_success',
                'tool': 'verifier_agrees', 'verifier': 'verifier_agrees'}[component]

    def _most_upstream_violated(self, symptoms, m_P=None):
        """Most-upstream violated mechanism that can explain m_P's violation.

        A candidate upstream witness must have a symptom rate at least as high
        as the patch mechanism's own rate (minus one diagnostic task of
        slack): if m_P's source field is violated on tasks where the upstream
        fault does not apply, m_P has its own genuine fault (parallel fault,
        not propagation) and the coupled witness must not fire.
        """
        rate = symptoms['symptom_rates']
        slack = 1.0 / max(1, self.method.diag_tasks)
        rate_P = rate.get(m_P.source_field, 0) if m_P else 1.0
        best = None
        for m in self.registry.active():
            if m.f_status == 'CONTRADICTED' or m.demoted_as_compensation:
                continue
            if m is m_P:
                continue
            if rate.get(m.source_field, 0) > 0 and rate.get(m.source_field, 0) >= rate_P - slack:
                if best is None or COMPONENT_ORDER[m.component] < COMPONENT_ORDER[best.component]:
                    best = m
        return best

    def _prediction(self, m_P, patch, symptoms, coupled: bool, roots=None):
        """Prediction for patch P under mechanism m_P (None = mechanism-free or joint).

        When the frozen action does NOT repair m_P's source (a foreign action
        scheduled on m_P for discriminative testing), the prediction is inert
        under m_P's own hypothesis: its source contract stays violated and its
        witness does not recover. That honest inert prediction is what makes
        the same action discriminative between competing mechanisms.

        `roots` overrides the joint prediction's root set (the exact
        scope-compatible confirmed roots a composition was built from).
        """
        if m_P is None:
            if roots is None:
                roots = [m for m in self.registry.active() if m.root_confirmed]
            if roots:
                base = tuple((m.source_field, False) for m in roots)
                post = tuple((m.source_field, True) for m in roots) + ((DOWNSTREAM_FIELD, True),)
                return Prediction('joint:' + '+'.join(sorted(m.id for m in roots)),
                                  roots[0].source_field, roots[0].propagation, base, post,
                                  ((DOWNSTREAM_FIELD, True),), 'SOURCE_REPAIR')
            # mechanism-free ablation (C0/C1): no attribution semantics, so the
            # prediction collapses to a bare downstream-utility claim that the
            # patch restores the terminal outcome without naming a source.
            return Prediction('patch:' + '+'.join(sorted(patch)),
                              DOWNSTREAM_FIELD, (),
                              ((DOWNSTREAM_FIELD, False),), ((DOWNSTREAM_FIELD, True),),
                              ((DOWNSTREAM_FIELD, True),), 'SOURCE_REPAIR')
        witness = self._witness_field(m_P.component)
        repairs_source = patch.get(m_P.repair_key, False)
        if not repairs_source:
            # inert under m_P: the action touches another mechanism's key, so
            # m_P's own hypothesis says nothing recovers on its contract pair
            base = ((m_P.source_field, False), (witness, False))
            post = ((m_P.source_field, False), (witness, False))
            return Prediction(m_P.id, m_P.source_field, m_P.propagation, base, post,
                              ((witness, False),), 'UNRESOLVED', action_repairs_source=False)
        if coupled:
            m_root = self._most_upstream_violated(symptoms, m_P)
            if m_root is not None and \
                    COMPONENT_ORDER[m_root.component] < COMPONENT_ORDER[m_P.component]:
                # coupled witness: the upstream candidate's source must be the
                # one that flips for a true repair; if it stays violated while
                # downstream recovers, the patch bridges as compensation. The
                # witness only recovers when the patch repairs m_P's own source
                # or the upstream root (a patch that repairs neither cannot
                # rescue the downstream contract either).
                s = m_root.source_field
                s_val = patch.get(m_root.repair_key, False)
                recovers = repairs_source or s_val
                base = ((s, False), (m_P.source_field, False))
                post = ((s, s_val), (m_P.source_field, recovers), (witness, recovers))
                return Prediction(m_P.id, s, m_P.propagation, base, post,
                                  ((witness, recovers),), 'COMPENSATION')
        if m_P.component == 'verifier':
            base, post = ((m_P.source_field, False),), ((m_P.source_field, repairs_source),)
        else:
            # the baseline WITNESS expectation must match what the diagnostic
            # evidence actually shows under the CURRENT config: a downstream
            # compensation promoted by an earlier round already satisfies the
            # witness boundary, and freezing a violation the deployed config
            # no longer exhibits would CONTRADICT this mechanism's own
            # failure contract and break relocalization after compensation
            # (section 2.3). A witness the diagnostics show violated stays
            # violated; a witness the diagnostics never observed is NOT
            # imputed as satisfied (it stays violated in the frozen
            # expectation and the confirmatory predicates resolve it).
            witness_observed = symptoms['coverage'].get(witness, 0) > 0
            witness_violated = symptoms['symptom_rates'].get(witness, 0) > 0
            witness_base = not witness_violated if witness_observed else False
            base = ((m_P.source_field, False), (witness, witness_base))
            post = ((m_P.source_field, repairs_source), (witness, True))
        return Prediction(m_P.id, m_P.source_field, m_P.propagation, base, post,
                          ((witness, True),), 'SOURCE_REPAIR')

    # ---------- staged evidence ----------

    @staticmethod
    def _scope_expression(premise):
        """Canonical applicability-scope expression for a contract.

        The premise predicate inferred from diagnostic traces becomes the
        contract's SCOPE: claims earned under it are bound to that condition
        ('global' when the premise is unconditional) and never represent
        tasks outside it.
        """
        return 'global' if not premise else json.dumps(premise, sort_keys=True)

    def _ledger_minimal(self, artifact_version, contract_id=None):
        """Minimal-feedback status for one EXACT artifact version, read back
        from the ledger's EVIDENCE_RESULT events (never from in-memory
        state alone): 'MINIMAL_OBSERVED', 'MINIMAL_UNRESOLVED', or None when
        no confirmatory evidence ever completed for that version.

        The lookup is bound to the confirmation contract's identity: the
        same code + parent hashes to the same artifact version, so a NEW
        contract (different tasks, predictions or applicability scope) must
        never inherit an OLDER contract's completed evidence for that
        version. With contract_id given, only that contract's own
        EVIDENCE_RESULT counts; a lookup for a contract whose confirmation
        TERMINATED or never completed returns None.
        """
        for e in reversed(self.ledger.events):
            if e['kind'] != 'EVIDENCE_RESULT':
                continue
            if contract_id is not None and e['payload'].get('contract_id') != contract_id:
                continue
            result = e['payload']['result']
            claims = result.get('claims') or []
            if any(c.get('artifact') == artifact_version for c in claims):
                return result.get('minimal')
        return None

    def _run_contract(self, patch, predictions, round_idx, contract_idx, premise=None, kind='PATCH'):
        candidate = {**self.current, **patch}
        artifact = Artifact.create(json.dumps(candidate, sort_keys=True),
                                   self.root_artifact.version, kind=kind)
        self.artifact_parents[artifact.version] = self.root_artifact.version
        # each contract in a round gets its own disjoint confirm pool,
        # filtered by the mechanism's applicability premise
        pool = self.benchmark.pool(f'CONFIRM{contract_idx}', self.method.confirm_tasks,
                                   round_idx, self.seed)
        tasks = [t for t in pool.tasks if self._matches_premise(t, premise)]
        if len(tasks) < MIN_PREMISE_TASKS:
            self.ledger.append('PREMISE_UNSUPPORTED', artifact=artifact.version,
                               mechanism=predictions[0].mechanism,
                               premise_tasks=len(tasks))
            return artifact, None, None
        contract = Contract(artifact.version, self.root_artifact.version,
                            self._scope_expression(premise),
                            tuple(t.task_id for t in tasks), tuple(predictions),
                            cost_per_task=self.method.cost_per_task,
                            alpha=self.alpha_contract)
        self.ledger.append('PATCH_GATE', artifact=artifact.version, contract_kind=kind,
                           diff={k: v for k, v in patch.items() if self.current.get(k) != v},
                           premise=premise,
                           scope=contract.scope,
                           predictions=[len(tuple(predictions))])
        self.controller.freeze(artifact, contract)
        tmap = {t.task_id: t for t in tasks}

        def paired(task_id):
            task = tmap[task_id]
            base_result = self._execute(self.current, task)
            # the execution protocol carries THIS edit's identity: the exact
            # patch keys, the direct parent version, the candidate version
            # and the frozen action/contract id. The executor's activation
            # event must be bound to exactly this identity (never to an old
            # or unrelated edit that happens to share the run).
            cand_result = self._execute(
                candidate, task, patch_keys=tuple(patch),
                parent_version=self.root_artifact.version,
                candidate_version=artifact.version,
                action_id=contract.id)
            b_boundary, b_util = base_result.boundary, base_result.utility
            i_boundary, i_util = cand_result.boundary, cand_result.utility
            # activation is an EXECUTION EVENT reported by the executor's
            # hook for exactly this edit (patch_keys): the branch the edit
            # modifies was actually reached in the intervention run. It is
            # never inferred from a configuration diff (a written key is not
            # evidence the code path ran), and a missing event stays None so
            # activation, contribution AND bridge all remain UNRESOLVED.
            return Observation(task_id, b_boundary, i_boundary, b_util, i_util,
                               cand_result.activated)

        result = self.controller.resume(contract.id, paired)
        return artifact, contract, result

    def _update_registry(self, result):
        """Consume contract claims strictly according to the ablation level.

        Claims are version-, action- AND scope-scoped. Every claim is
        recorded in the mechanism's response branch for the EXACT artifact
        and applicability scope it was earned under, so evidence from one
        condition never overwrites another condition's branch. The global
        decision slots (f/r/z, claim_version) are written ONLY by
        own-action claims earned under the unconditional 'global' scope;
        scoped claims update the mechanism's scoped view for their own
        condition (a "cacheable=true" conclusion never speaks for
        non-cacheable tasks). A newer artifact with unresolved evidence
        never inherits an older artifact's SOURCE_REPAIR/COMPENSATION
        conclusion within the same scope. Foreign-action (discriminative
        partner) claims stay in the response branches for audit and are
        never allowed to rewrite the mechanism's own view.
        """
        if not result or not result.get('claims'):
            return
        f, r, z = self.features['failure'], self.features['contribution'], self.features['bridge']
        by_id = {m.id: m for m in self.registry.active()}
        for claim in result['claims']:
            m_P = by_id.get(claim['mechanism'])
            if m_P is None:
                continue
            own = claim.get('action_repairs_source', True)
            artifact = claim.get('artifact', '')
            scope = claim.get('scope') or 'global'
            m_P.responses.setdefault(artifact, {})[scope] = {
                'failure': claim['failure'],
                'contribution': claim['contribution'],
                'bridge': claim['bridge'],
                'action_repairs_source': own,
            }
            # RAW evidence audit trail: the full unfiltered claim the
            # controller produced, tagged with whether it is the
            # mechanism's own-action claim or a foreign-action
            # discrimination record. Experiment metrics never read this as
            # the method's decision -- they read the feature-filtered view
            # (features / claim_view / promotions); levels with a view
            # disabled accept no conclusion from it.
            self.labeling.append({'mechanism': m_P.id, 'bridge': claim['bridge'],
                                  'repair_key': m_P.repair_key, 'artifact': artifact,
                                  'scope': scope, 'own': own})
            if not own:
                # discrimination record only: no decision slot may move
                continue
            # feature-filtered decision view: the SAME consumption entry as
            # the global slots below. An attribution view the level disables
            # (failure / contribution / bridge) reads UNRESOLVED here, so a
            # scoped claim can never switch on machinery the ablation turned
            # off -- C2 cannot read contribution or bridge conclusions, C3
            # cannot read bridge conclusions, whatever the raw claim said.
            # The response branch recorded above keeps the full unfiltered
            # evidence as the audit trail.
            view = {'failure': claim['failure'] if f else 'UNRESOLVED',
                    'contribution': claim['contribution'] if r else 'UNRESOLVED',
                    'bridge': claim['bridge'] if z else 'UNRESOLVED',
                    'artifact': artifact}
            if scope != 'global':
                # conditional evidence: bound to its applicability scope
                m_P.scoped_views[scope] = view
                continue
            if f:
                m_P.f_status = {'CONSISTENT': 'CONSISTENT', 'CONTRADICTED': 'CONTRADICTED'}.get(
                    claim['failure'], 'UNRESOLVED')
            if r:
                m_P.r_status = 'POSITIVE' if claim['contribution'] == 'POSITIVE' else 'UNRESOLVED'
            m_P.claim_version = artifact
            if z:
                # unconditional overwrite within the global scope: UNRESOLVED
                # evidence resets a stale bridge instead of inheriting it
                m_P.z_bridge = claim['bridge']
                if claim['bridge'] == 'COMPENSATION':
                    m_P.joint_note = 'demoted: contribution bridges as compensation, not root repair'
                elif claim['bridge'] == 'SOURCE_REPAIR' and m_P.f_status == 'CONSISTENT':
                    m_P.joint_note = 'root cause confirmed by bridged dual attribution'
        # joint update: relocalize the upstream candidate after a compensation
        if z:
            demoted = [by_id[c['mechanism']] for c in result['claims']
                       if c.get('action_repairs_source', True)
                       and c['bridge'] == 'COMPENSATION' and c['mechanism'] in by_id]
            if demoted:
                most_downstream = max(demoted, key=lambda x: COMPONENT_ORDER[x.component])
                for m in self.registry.active():
                    if COMPONENT_ORDER[m.component] < COMPONENT_ORDER[most_downstream.component] \
                            and m.f_status == 'PROPOSED':
                        m.joint_note = RELOC_NOTE
                        break

    # ---------- selection and promotion ----------

    def _select_and_promote(self, candidates, round_idx):
        """Independent blind paired selection with simultaneous lower bounds.

        `candidates` is a list of (patch, kind) pairs; each candidate's
        selection artifact is created with the SAME kind as its confirmatory
        contract, so the version entering selection is exactly the version
        that earned its own minimal feedback. The candidate set and task pool
        are frozen in the ledger BEFORE any selection task executes; the
        per-round alpha comes from the protocol frozen at run start (no
        ad-hoc 0.05 per round).
        """
        if not candidates:
            return None
        capped = candidates[:self.method.max_candidates]
        pool = self.benchmark.pool('SELECT', self.method.selection_tasks, round_idx, self.seed)
        families = sorted({t.family for t in pool.tasks})
        # freeze first: candidates, versions and tasks are committed before execution
        versions = {}
        by_version = {}
        for patch, kind in capped:
            candidate = {**self.current, **patch}
            artifact = Artifact.create(json.dumps(candidate, sort_keys=True),
                                       self.root_artifact.version, kind=kind)
            versions[artifact.version] = artifact
            by_version[artifact.version] = (patch, kind)
            self.artifact_parents[artifact.version] = self.root_artifact.version
        self.ledger.append('SELECTION_FROZEN', candidates=list(versions),
                           tasks=pool.ids(), families=families, round=round_idx,
                           alpha=self.alpha_round,
                           estimator='paired Hoeffding; alpha split across rounds '
                                     '(protocol frozen at run start)')
        owner = 'promotion'
        entries = {}
        for version, artifact in versions.items():
            patch, _kind = by_version[version]
            candidate = {**self.current, **patch}
            pairs = []
            for t in pool.tasks:
                self.budget.charge(PAIR_COST, owner)
                self.exposure.run(t.task_id, 'SELECT', version)
                u0 = self._execute(self.current, t).utility
                u1 = self._execute(candidate, t).utility
                pairs.append((t.task_id, t.family, u0, u1))
            entries[version] = {'pairs': pairs, 'cost': len(patch), 'size': sum(candidate.values())}
        selected = promotion(entries, families, alpha=self.alpha_round)
        # full results (including per-family regression bounds) go to the ledger
        self.ledger.append('PROMOTION', round=round_idx, **selected)
        selected = dict(selected)
        selected['winner_artifact'] = versions.get(selected.get('winner'))
        return selected

    # ---------- main loop ----------

    def run(self) -> RunSummary:
        m = self.method
        self._freeze_protocol()
        baseline_eval = self._eval_utility(-1)
        config_history = [dict(self.current)]
        checkpoints = [(self.budget.spent, baseline_eval)]
        llm_usage = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0}

        for round_idx in range(m.rounds):
            rec = RoundRecord(round_idx, self.budget.spent, 0, 0.0, 0)
            symptoms, traces = self._diagnose(round_idx)

            if m.kind == 'locality':
                self._locality_round(symptoms, round_idx, rec)
            else:
                self._dace_round(symptoms, traces, round_idx, rec, llm_usage)

            rec.spent_end = self.budget.spent
            rec.eval_utility = self._eval_utility(round_idx)
            rec.mechanisms_active = len(self.registry.active())
            checkpoints.append((self.budget.spent, rec.eval_utility))
            config_history.append(dict(self.current))
            self.records.append(rec)

        self.budget.release('promotion')
        final_eval = self.records[-1].eval_utility if self.records else baseline_eval
        summary = RunSummary(
            method=m.name, benchmark=self.benchmark.name,
            scenario=getattr(self.benchmark, 'scenario_name', None),
            seed=self.seed, kind=m.kind,
            level=m.level.value if m.level else 'locality-heuristic',
            final_config=self.current, config_history=config_history, rounds=self.records,
            baseline_eval=baseline_eval, final_eval=final_eval,
            localization=self._localization(), labeling=self.labeling,
            checkpoints=checkpoints, mechanism_states=self._mechanism_states(),
            retained_candidates=[dict(r['patch']) for r in self.retained],
            ground_truth=self.benchmark.ground_truth(),
            ledger_path=str(self.ledger.path), budget_spent=self.budget.spent,
            llm_usage=self._usage_totals("AGENT_PROPOSAL"),
            execution_usage=self._usage_totals("EXECUTION_RESULT"),
            result_kind="benchmark-agent" if self.benchmark.mode == "live" else "mock",
            # run identity: execution mode, model and the frozen sample/budget
            # profile, so cross-experiment aggregation (E5) never mixes
            # conditions that differ on any of them
            mode=getattr(self.benchmark, 'mode', 'mock'),
            model=getattr(self.benchmark, 'model', None),
            spec={**asdict(m), 'budget_total': self.budget.total},
            # ablation flags actually in force + per-promotion attribution
            # records: the DECISION-side data experiment metrics consume (a
            # disabled view accepts no conclusion, whatever the raw evidence)
            features=dict(self.features), promotions=list(self.promotions))
        (self.output / 'summary.json').write_text(
            json.dumps(_jsonable(summary), indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        (self.output / 'harness.json').write_text(
            json.dumps(self.current, indent=2), encoding='utf-8')
        return summary

    def _freeze_protocol(self):
        """Freeze the statistical and budget protocol before the first round."""
        m = self.method
        contract_cap = m.intervention_budget // max(1, m.confirm_tasks * m.cost_per_task)
        self.alpha_round = m.alpha_total / max(1, m.rounds)
        self.alpha_contract = self.alpha_round / max(1, contract_cap)
        # the whole run's promotion budget is locked before any confirmatory
        # execution; confirm contracts can only use what remains
        self.budget.reserve('promotion', m.rounds * m.max_candidates * m.selection_tasks * PAIR_COST)
        self.ledger.append('PROTOCOL_FROZEN', rounds=m.rounds, alpha_total=m.alpha_total,
                           alpha_per_round=self.alpha_round, alpha_per_contract=self.alpha_contract,
                           diag_tasks=m.diag_tasks, confirm_tasks=m.confirm_tasks,
                           selection_tasks=m.selection_tasks, eval_tasks=m.eval_tasks,
                           intervention_budget=m.intervention_budget, mechanism_cap=m.mechanism_cap,
                           max_candidates=m.max_candidates, cost_per_task=m.cost_per_task,
                           promotion_reserved=m.rounds * m.max_candidates * m.selection_tasks * PAIR_COST,
                           estimator='paired Hoeffding lower bounds; Bonferroni across '
                                     'rounds, contracts and predictions')
        # promotion-evidence policy, frozen BEFORE any round: what minimal
        # feedback a candidate's own confirmation contract must have earned
        # before it may be promoted, whether genuinely unobservable
        # executions may ever ride utility selection (only with an explicit
        # recorded reason), and the identity the evidence is bound to
        self.ledger.append('PROMOTION_EVIDENCE_POLICY_FROZEN',
                           require_minimal_feedback=m.require_minimal_feedback,
                           allow_unobserved_promotion=m.allow_unobserved_promotion,
                           unobserved_promotion_reason=m.unobserved_promotion_reason or None,
                           evidence_source='ledger EVIDENCE_RESULT minimal status bound to '
                                           "the candidate's exact (artifact, contract, "
                                           'scope) identity',
                           basis='a candidate is selectable only on the minimal feedback '
                                 'of its OWN confirmation contract for its exact artifact '
                                 'version; a confirmation that terminated or never '
                                 'completed this round is rejected outright under every '
                                 'policy, and a MINIMAL_UNRESOLVED candidate is blocked '
                                 'unless the policy explicitly allows it with a '
                                 'recorded reason')

    def _dace_round(self, symptoms, traces, round_idx, rec, llm_usage):
        m = self.method
        # 1. propose into the CMVS
        if self.features['mechanisms']:
            if self.llm is not None:
                proposed = self.llm.propose_mechanisms(symptoms, self.registry, self.budget,
                                                       self.exposure, self.root_artifact.version)
                llm_usage['calls'] += 1
                from .components import MECHANISM_CATALOG, PROPAGATION
                for item in proposed.get('mechanisms', []):
                    mid = item['id']
                    if mid in PROPAGATION:
                        for cid, comp, src, _, rep in MECHANISM_CATALOG:
                            if cid == mid:
                                self.registry.propose(Mechanism(mid, comp, src,
                                                                PROPAGATION[mid], rep))
                                break
            else:
                offline_propose(symptoms, self.registry)

        # 2. candidate patches (single-key from the CMVS; joint root-repair
        #    compositions from pairwise-scope-compatible confirmed roots)
        joint_plans = []
        if self.features['mechanisms']:
            patches = candidate_patches(self.registry, joint=False)
            if self.features['bridge']:
                joint_plans = joint_compositions(self.registry)
        else:
            from .components import MECHANISM_CATALOG
            patches = [{repair: True} for _, _, _, trig, repair in MECHANISM_CATALOG
                       if symptoms['symptom_rates'].get(trig, 0) > 0]
            seen, uniq = set(), []
            for p in patches:
                k = json.dumps(p, sort_keys=True)
                if k not in seen:
                    seen.add(k)
                    uniq.append(p)
            patches = uniq

        # 3. staged evidence with confirm locks, capped by intervention budget.
        #    Every planned item carries its applicability premise and the
        #    scope expression its contract will run under, so claims are
        #    consumed through the scope-matched query (claim_view) instead of
        #    being flattened into unconditional state.
        contract_cap = m.intervention_budget // max(1, m.confirm_tasks * m.cost_per_task)
        planned = []
        for patch in patches:
            # already-applied edits are inert: re-confirming the parent against
            # itself earns nothing and burns confirm budget
            if all(self.current.get(k) == v for k, v in patch.items()):
                continue
            mech = None
            for x in self.registry.active():
                if x.repair_key in patch:
                    mech = x
                    break
            premise = self._premise(mech, traces) if mech else None
            planned.append({'patch': patch, 'mech': mech, 'kind': 'PATCH',
                            'partner': None, 'disagreement': 0.0,
                            'premise': premise,
                            'scope': (self._scope_expression(premise)
                                      if premise != 'UNSUPPORTED' else None)})
        for comp in joint_plans:
            if all(self.current.get(k) == v for k, v in comp['patch'].items()):
                continue
            planned.append({'patch': comp['patch'], 'mech': None, 'kind': 'JOINT_PATCH',
                            'partner': None, 'disagreement': 0.0,
                            'premise': comp['premise'],
                            'scope': self._scope_expression(comp['premise']),
                            'roots': comp['roots']})

        def relocalization_priority(item) -> int:
            """0 when the joint update's search focus sits on this action's
            mechanism: a downstream compensation was confirmed (whether or
            not it was later promoted -- a deployed compensation leaves the
            root cause in place), so the search relocalized upstream
            (section 2.3). Derived from registry state at SCHEDULING time:
            a joint_note marker can be overwritten by the same round's own
            upstream confirmation and reset by a promotion, but the demoted
            mechanism's label (active or archived) is durable evidence that
            the search focus belongs upstream."""
            mech = item['mech']
            if mech is None:
                return 1
            demoted = [m for m in list(self.registry.active())
                       + list(self.registry.archived.values())
                       if m.z_bridge == 'COMPENSATION']
            if not demoted:
                return 1
            most_downstream = max(demoted, key=lambda x: COMPONENT_ORDER[x.component])
            if (COMPONENT_ORDER[mech.component] < COMPONENT_ORDER[most_downstream.component]
                    and mech.f_status == 'PROPOSED'):
                return 0
            return 1

        if self.features['contribution'] and not self.features['probe']:
            # contribution-driven scheduling (C3/C4, Uncoupled Dual): an action
            # whose mechanism already holds a POSITIVE own-action contribution
            # claim -- read under the scope THIS round's contract for it will
            # run in (claim_view: the scoped view for that exact condition,
            # else the unconditional slots) -- earned against a child of the
            # CURRENT parent has standing evidence; the confirm budget goes to
            # actions without it first. Without the contribution view (C2) the
            # same evidence is recorded but never read, so planning is
            # unchanged -- this is the decision path that separates C3 from C2.
            def has_standing_evidence(item):
                mech = item['mech']
                if mech is None:
                    return False
                view = mech.claim_view(item['scope'] or 'global')
                if not view or view.get('contribution') != 'POSITIVE':
                    return False
                version = view.get('artifact')
                return bool(version) and \
                    self.artifact_parents.get(version) == self.root_artifact.version

            def schedule_key(item):
                return (relocalization_priority(item),
                        1 if has_standing_evidence(item) else 0,
                        json.dumps(item['patch'], sort_keys=True))
            planned.sort(key=schedule_key)
        if self.features['probe']:
            # C5 discriminative scheduling: for each action, freeze the top
            # competing mechanism's prediction on the SAME action, ranked by
            # D(a;M) computed from the two frozen predictions on their common
            # observable fields (an inert action keeps baseline values). One
            # confirmatory contract then discriminates between the hypotheses.
            coupled = self.features['bridge']
            for item in planned:
                if item['mech'] is None:
                    continue
                pred_own = self._prediction(item['mech'], item['patch'], symptoms, coupled=coupled)
                best_partner, best_d = None, 0.0
                for other in self.registry.active():
                    if other is item['mech'] or other.f_status == 'CONTRADICTED' \
                            or other.demoted_as_compensation:
                        continue
                    d = prediction_disagreement(
                        pred_own, self._prediction(other, item['patch'], symptoms, coupled=coupled))
                    if d > best_d:
                        best_partner, best_d = other, d
                item['partner'], item['disagreement'] = best_partner, best_d
            # highest-disagreement actions are scheduled first within the cap,
            # AFTER the joint update's relocalization candidate: a confirmed
            # downstream compensation moves the search upstream (section 2.3),
            # and the relocalized candidate must not be starved of confirmatory
            # budget by inert actions that merely discriminate between live
            # hypotheses (the demoted mechanism is archived, so the
            # relocalization candidate has lost its disagreeing partner)
            planned.sort(key=lambda it: (relocalization_priority(it),
                                         -it['disagreement'],
                                         json.dumps(it['patch'], sort_keys=True)))
        planned = planned[:max(0, contract_cap)]

        bridges = []
        executed = []
        for item in planned:
            coupled = self.features['bridge']
            if item['kind'] == 'JOINT_PATCH':
                predictions = (self._prediction(None, item['patch'], symptoms, coupled=coupled,
                                                roots=item.get('roots')),)
            else:
                predictions = [self._prediction(item['mech'], item['patch'], symptoms, coupled=coupled)]
                if item['partner'] is not None:
                    # the SAME legal action, frozen with the competing mechanism's
                    # prediction attached: one contract, two discriminable claims
                    predictions.append(self._prediction(item['partner'], item['patch'],
                                                        symptoms, coupled=coupled))
            premise = item['premise']
            if premise == 'UNSUPPORTED':
                executed.append({**item, 'result': None})
                continue
            artifact, contract, result = self._run_contract(
                item['patch'], tuple(predictions), round_idx, len(executed),
                premise=premise, kind=item['kind'])
            executed.append({**item, 'result': result, 'artifact': artifact.version,
                             'contract_id': contract.id if contract else None,
                             'scope': contract.scope if contract else item['scope']})
            if contract is None:
                continue
            rec.contracts_frozen += 1
            if len(predictions) >= 2:
                rec.probes += 1
            if result and result.get('claims'):
                bridges.append({'mechanism': predictions[0].mechanism,
                                'bridge': result['claims'][0]['bridge'],
                                'failure': result['claims'][0]['failure'],
                                'contribution': result['claims'][0]['contribution'],
                                'disagreement': round(item['disagreement'], 4),
                                'discriminated': [c['mechanism'] for c in result['claims'][1:]]})
            self._update_registry(result)
        rec.bridges = bridges

        # 4. joint root-repair patches constructible from THIS round's newly
        #    confirmed roots must earn their OWN minimal feedback before they
        #    may enter selection: two single repairs being confirmed says
        #    nothing about their exact composition (the edits may interact).
        #    Compositions respect scope compatibility -- roots confirmed under
        #    mutually exclusive conditions never merge, and a composition's
        #    contract runs under its merged applicability premise. The joint
        #    artifact goes through the SAME freeze/lock/evidence entrance as
        #    every patch; with no confirmatory slot left this round it is
        #    explicitly deferred -- it is never allowed to ride into selection
        #    on the constituents' inherited claims.
        if self.features['bridge']:
            already_executed = {(json.dumps(it['patch'], sort_keys=True), it.get('scope'))
                                for it in executed}
            slots_left = contract_cap - rec.contracts_frozen
            for comp in joint_compositions(self.registry):
                p = comp['patch']
                if len(p) < 2:
                    continue
                scope_expr = self._scope_expression(comp['premise'])
                if (json.dumps(p, sort_keys=True), scope_expr) in already_executed:
                    continue
                if all(self.current.get(k) == v for k, v in p.items()):
                    continue
                if slots_left <= 0:
                    self.ledger.append('JOINT_PATCH_DEFERRED', round=round_idx, patch=p,
                                       premise=comp['premise'], scope=scope_expr,
                                       reason='no confirmatory budget slot left this round',
                                       note='a joint patch never enters selection without '
                                            'its own minimal feedback on its exact artifact')
                    continue
                prediction = self._prediction(None, p, symptoms, coupled=True,
                                              roots=comp['roots'])
                artifact, contract, result = self._run_contract(
                    p, (prediction,), round_idx, len(executed),
                    premise=comp['premise'], kind='JOINT_PATCH')
                executed.append({'patch': p, 'mech': None, 'kind': 'JOINT_PATCH',
                                 'partner': None, 'disagreement': 0.0,
                                 'premise': comp['premise'], 'scope': scope_expr,
                                 'roots': comp['roots'],
                                 'result': result, 'artifact': artifact.version,
                                 'contract_id': contract.id if contract else None})
                if contract is not None:
                    rec.contracts_frozen += 1
                    slots_left -= 1

        # 5. eligibility gate per level, then independent selection.
        #    FROZEN promotion-evidence policy first: a candidate may enter
        #    selection ONLY on the minimal feedback of its OWN confirmation
        #    contract, read back from the ledger and bound to the exact
        #    (artifact, contract, scope) identity. A confirmation that
        #    TERMINATED or never completed this round is explicitly rejected
        #    under EVERY policy -- it is a failed confirmation, not an
        #    unobservable execution, and may never ride an older contract's
        #    completed evidence for the same artifact version. A completed
        #    contract with MINIMAL_UNRESOLVED feedback is blocked unless the
        #    frozen policy explicitly allows unobserved utility promotion
        #    with its recorded reason -- never a silent default.
        #    Attribution queries are scope-matched (claim_view under the
        #    contract's own scope): a mechanism demoted under one condition
        #    stays selectable in the scope where its contract earned
        #    source-repair evidence, and contribution evidence earned under
        #    another condition never confirms this action.
        #    A COMPENSATION conclusion demotes the ROOT-CAUSE claim only: the
        #    edit is not the source repair, but that says nothing against its
        #    own deployable benefit. The candidate keeps its earned minimal
        #    feedback, contribution and regression evidence and enters
        #    independent selection under the same gates as every other
        #    candidate (method section 2.4: retain valid compensations and
        #    let independent tasks decide promotion). When such a candidate
        #    wins, the promotion is REPORTED as a compensation deployment and
        #    never counted as a source repair -- the root cause persists.
        eligible, unconfirmed, unobserved, blocked, failed = [], [], [], [], []
        #: patch key -> method-accepted attribution record for the promotion
        #: report: the mechanism, the contract's applicability scope and the
        #: accepted bridge label (ablation-filtered: without the bridge view
        #: a method cannot demote a patch, so its accepted claim defaults to
        #: source repair -- the honest measurement of the uncoupled ablation)
        promotion_claims = {}

        def claim_record(patch, mech, scope, bridge_label):
            return {'patch': dict(patch),
                    'mechanism': mech.id if mech is not None else None,
                    'scope': scope or 'global',
                    'bridge': bridge_label}

        for item in executed:
            if item['result'] is None:
                continue
            if item['result'].get('minimal') is None:
                # TERMINATED_UNRESOLVED or a still-pending confirmation: this
                # round's own confirmation did not complete. Hard rejection.
                failed.append(item.get('artifact')
                              or json.dumps(item['patch'], sort_keys=True))
                continue
            minimal = self._ledger_minimal(item.get('artifact', ''), item.get('contract_id'))
            if minimal != 'MINIMAL_OBSERVED':
                if self.method.allow_unobserved_promotion:
                    # a never-activated edit supports no bridge conclusion:
                    # the promotion report records an unresolved label
                    promotion_claims[json.dumps(item['patch'], sort_keys=True)] = claim_record(
                        item['patch'], item.get('mech'), item.get('scope'), 'UNRESOLVED')
                    unobserved.append((item['patch'], item['kind']))
                else:
                    blocked.append(item.get('artifact')
                                   or json.dumps(item['patch'], sort_keys=True))
                continue
            if item['kind'] == 'JOINT_PATCH':
                # a joint patch is only selectable on ITS OWN minimal feedback;
                # composing confirmed repairs does not confirm the composition.
                # Report only the exact joint artifact's own bridge evidence.
                joint_bridges = {c.get('bridge') for c in item['result'].get('claims', [])
                                 if c.get('artifact') == item.get('artifact')}
                joint_label = (next(iter(joint_bridges)) if len(joint_bridges) == 1
                               and joint_bridges <= {'SOURCE_REPAIR', 'COMPENSATION'}
                               else 'UNRESOLVED')
                promotion_claims[json.dumps(item['patch'], sort_keys=True)] = claim_record(
                    item['patch'], None, item.get('scope'), joint_label)
                eligible.append((item['patch'], 'JOINT_PATCH'))
                continue
            if item['kind'] != 'PATCH':
                continue
            mech = item['mech']
            view = mech.claim_view(item.get('scope') or 'global') if mech is not None else None
            if self.features['bridge'] and view is not None:
                # method-accepted bridge label under THIS contract's scope
                # (the decision views are already ablation-filtered)
                label = view.get('bridge')
                if label not in ('SOURCE_REPAIR', 'COMPENSATION', 'UNRESOLVED'):
                    label = 'UNRESOLVED'
            else:
                # no bridge view (or a mechanism-free level): the method
                # cannot demote the patch, so its accepted attribution claim
                # defaults to source repair
                label = 'SOURCE_REPAIR'
            promotion_claims[json.dumps(item['patch'], sort_keys=True)] = claim_record(
                item['patch'], mech, item.get('scope'), label)
            # a COMPENSATION label does NOT exclude the candidate: it only
            # demotes the root-cause claim. The candidate proceeds through
            # the same contribution/utility gates below.
            if (self.method.level == AblationLevel.CONTRIBUTION_ONLY and mech is not None
                    and (view is None or view.get('contribution') != 'POSITIVE')):
                # the contribution-only control selects strictly on the
                # contribution view (that is its defining semantics)
                continue
            if self.features['contribution'] and mech is not None:
                if view is not None and view.get('contribution') == 'POSITIVE':
                    eligible.append((item['patch'], 'PATCH'))
                else:
                    unconfirmed.append((item['patch'], 'PATCH'))
                continue
            eligible.append((item['patch'], 'PATCH'))
        if failed:
            self.ledger.append('PROMOTION_BLOCKED_CONFIRMATION_FAILED', round=round_idx,
                               blocked=failed,
                               note="this round's own confirmation contract did not "
                                    'complete (TERMINATED or pending): the candidate is '
                                    'rejected outright and may never ride an older '
                                    "contract's evidence for the same artifact version")
        if blocked:
            self.ledger.append('PROMOTION_BLOCKED_UNOBSERVED', round=round_idx,
                               blocked=blocked,
                               policy='PROMOTION_EVIDENCE_POLICY_FROZEN',
                               note='candidates whose exact artifact version never earned '
                                    'MINIMAL_OBSERVED activation feedback are not '
                                    'selectable under the frozen policy')
        if unobserved:
            self.ledger.append('PROMOTION_UNOBSERVED_ALLOWED', round=round_idx,
                               candidates=[p for p, _ in unobserved],
                               reason=self.method.unobserved_promotion_reason,
                               note='frozen policy explicitly allows utility promotion of '
                                    'unactivated edits; every such promotion is recorded')
            eligible.extend(unobserved)
        # contribution-evidenced actions first; unconfirmed ones keep their
        # utility-only selectability but fill the shortlist after them
        eligible = eligible + unconfirmed
        selected = self._select_and_promote(eligible, round_idx)
        if selected and selected.get('winner'):
            winner_artifact = selected.get('winner_artifact')
            for patch, kind in eligible:
                candidate = {**self.current, **patch}
                if Artifact.create(json.dumps(candidate, sort_keys=True),
                                   self.root_artifact.version,
                                   kind=kind).version == selected['winner']:
                    claim = promotion_claims.get(
                        json.dumps(patch, sort_keys=True),
                        {'patch': dict(patch), 'mechanism': None, 'scope': 'global',
                         'bridge': 'SOURCE_REPAIR'})
                    self._apply_promotion(patch, rec, round_idx, winner_artifact, claim=claim)
                    break

        # 6. Masked-benefit retention (section 2.4): when no candidate clears
        #    the independent utility bound but a patch produced consistent
        #    source-repair boundary evidence, retain it in an ISOLATED
        #    candidate lineage. The current harness is NOT modified, the
        #    candidate does NOT count as accepted, and it must still win
        #    independent selection in a later round before any promotion.
        if not rec.accepted and self.features['failure']:
            retained_keys = {json.dumps(r['patch'], sort_keys=True) for r in self.retained}
            for item in executed:
                if item['kind'] != 'PATCH' or item['result'] is None \
                        or not item['result'].get('claims'):
                    continue
                claim = item['result']['claims'][0]
                evidence = (claim['failure'] == 'CONSISTENT'
                            and claim['boundary'] == 'CONSISTENT'
                            and claim['mediator'] == 'CONSISTENT')
                if self.features['bridge']:
                    evidence = evidence and claim['bridge'] == 'SOURCE_REPAIR'
                patch = item['patch']
                patch_key = json.dumps(patch, sort_keys=True)
                if not evidence or patch_key in retained_keys \
                        or all(self.current.get(k) == v for k, v in patch.items()):
                    continue
                cand_artifact = Artifact.create(
                    json.dumps({**self.current, **patch}, sort_keys=True),
                    self.root_artifact.version, kind='PATCH')
                self.retained.append({'patch': dict(patch), 'mechanism': claim['mechanism'],
                                      'artifact': cand_artifact.version})
                rec.retained_keys = sorted(patch)
                self.ledger.append('MASKED_BENEFIT_RETAINED', round=round_idx, patch=patch,
                                   mechanism=claim['mechanism'],
                                   candidate_artifact=cand_artifact.version,
                                   parent_artifact=self.root_artifact.version,
                                   note='source-repair evidence retained in an isolated '
                                        'candidate lineage; current harness unchanged; the '
                                        'candidate must still win independent selection '
                                        'before promotion')
                break

    def _apply_promotion(self, patch, rec, round_idx, winner_artifact, claim=None):
        """Apply the selected winner and invalidate stale mechanism claims.

        The winning artifact's exact identity is preserved (the promoted root
        keeps the version it was selected under). Mechanism claims are bound
        to the version they were earned against: mechanisms whose repair was
        promoted are archived with their labels; every other active mechanism
        is reset to PROPOSED -- claims are NOT inherited by the new version.

        `claim` is the method-accepted attribution record the winner was
        promoted under (mechanism, applicability scope, bridge label). A
        COMPENSATION promotion deploys a verified beneficial offset while
        the root cause persists: it is recorded and reported as a
        compensation deployment and never counted as a source repair.
        """
        applied = {k: v for k, v in patch.items() if self.current.get(k) != v}
        if not applied:
            return
        claim = claim or {'patch': dict(patch), 'mechanism': None, 'scope': 'global',
                          'bridge': 'SOURCE_REPAIR'}
        self.current.update(applied)
        rec.promoted_keys = sorted(applied)
        rec.accepted = True
        self.ledger.append('ANCHORS_STALE', previous=self.root_artifact.version,
                           current=winner_artifact.version if winner_artifact else None,
                           basis='INDEPENDENT_SELECTION',
                           note='mechanism claims are not inherited by the promoted version')
        # preserve the winner's exact artifact identity as the new root
        self.root_artifact = (winner_artifact if winner_artifact is not None else
                              Artifact.create(json.dumps(self.current, sort_keys=True),
                                              kind='BASELINE'))
        self.artifact_parents[self.root_artifact.version] = None
        # ONE promotion event per accepted selection (a joint patch counts as
        # one effective patch event, not one per modified key)
        rec.promoted_artifacts = [self.root_artifact.version]
        # report-side attribution record: the method-accepted label under
        # which THIS deployment happened (scope-matched, ablation-filtered)
        rec.promoted_as = [claim['bridge']]
        self.promotions.append({'round': round_idx, 'keys': sorted(applied),
                                'mechanism': claim['mechanism'], 'scope': claim['scope'],
                                'bridge': claim['bridge'], 'artifact': self.root_artifact.version})
        if claim['bridge'] == 'COMPENSATION':
            self.ledger.append('PROMOTION_ATTRIBUTION', round=round_idx, patch=applied,
                               attribution='COMPENSATION', mechanism=claim['mechanism'],
                               scope=claim['scope'], artifact=self.root_artifact.version,
                               note='the deployed edit is a verified compensation: it '
                                    'offsets a persisting upstream root cause, and this '
                                    'promotion is reported as a compensation deployment -- '
                                    'never as a source repair')
        else:
            self.ledger.append('PROMOTION_ATTRIBUTION', round=round_idx, patch=applied,
                               attribution=claim['bridge'], mechanism=claim['mechanism'],
                               scope=claim['scope'], artifact=self.root_artifact.version,
                               note='method-accepted attribution label (scope-matched, '
                                    'ablation-filtered) under which the winner was promoted')
        # stale-remap on the ACTUAL modified key set: archive the mechanisms
        # whose repair was promoted (labels kept as the historical record),
        # reset the rest (claims and scoped views are not inherited across
        # versions; response branches stay as the audit trail)
        for mech in list(self.registry.active()):
            if mech.repair_key in applied:
                self.registry.archived[mech.id] = self.registry.mechanisms.pop(mech.id)
            else:
                mech.f_status, mech.r_status = 'PROPOSED', 'PROPOSED'
                mech.z_bridge = 'UNBRIDGED'
                mech.claim_version = ''
                mech.scoped_views = {}
                # claims are NOT inherited by the promoted version -- but the
                # joint update's relocalization directive SURVIVES the
                # promotion: a deployed compensation leaves the upstream root
                # cause in place, so the upstream search focus stays valid on
                # the new parent (section 2.3)
                mech.joint_note = (RELOC_NOTE if mech.joint_note == RELOC_NOTE
                                   else 'claims reset: parent version promoted, anchors stale')
        # retained candidates whose keys were promoted leave the lineage
        self.retained = [r for r in self.retained
                         if not all(k in applied for k in r['patch'])]

    def _locality_round(self, symptoms, round_idx, rec):
        """Locality-first heuristic baseline: diagnose -> single-key patch ->
        majority verify -> reflect. NOT the official AutoSaddler optimizer."""
        tried = set(k for k, v in self.current.items() if v)
        attempts = 0
        while attempts < 3:
            patch = locality_patch(symptoms, tried)
            if patch is None:
                break
            key = next(iter(patch))
            candidate = {**self.current, **patch}
            pool = self.benchmark.pool('VERIFY', 16, round_idx * 10 + attempts, self.seed)
            delta = 0.0
            for t in pool.tasks:
                # paired parent+candidate execution is charged as two runs
                self.budget.charge(PAIR_COST)
                self.exposure.run(t.task_id, 'VERIFY', self.root_artifact.version)
                u0 = self._execute(self.current, t).utility
                u1 = self._execute(candidate, t).utility
                delta += u1 - u0
            self.ledger.append('LOCALITY_VERIFY', round=round_idx, attempt=attempts,
                               patch=patch, mean_delta=delta / max(1, len(pool.tasks)))
            attempts += 1
            if delta > 0:
                self.current.update(patch)
                rec.promoted_keys = sorted(patch)
                rec.accepted = True
                rec.contracts_frozen = attempts
                self.root_artifact = Artifact.create(json.dumps(self.current, sort_keys=True),
                                                     kind='BASELINE')
                break
            tried.add(key)
        rec.contracts_frozen = attempts

    # ---------- end-of-run attribution bookkeeping ----------

    def _mechanism_states(self):
        out = []
        for m in list(self.registry.active()) + list(self.registry.archived.values()):
            out.append({'id': m.id, 'component': m.component, 'source_field': m.source_field,
                        'repair_key': m.repair_key, 'f_status': m.f_status,
                        'r_status': m.r_status, 'z_bridge': m.z_bridge, 'joint_note': m.joint_note,
                        'claim_version': m.claim_version,
                        # per-(mechanism, action, artifact, scope) response
                        # branches: the full audit trail of every claim, keyed
                        # by the exact artifact AND applicability scope it was
                        # earned under
                        'response_branches': {a: dict(s) for a, s in m.responses.items()},
                        # latest own-action claim per applicability scope
                        # (conditional attribution views)
                        'scoped_views': dict(m.scoped_views)})
        return out

    def _localization(self):
        """Predicted root components vs environment ground truth."""
        from .components import MECHANISM_CATALOG
        gt = self.benchmark.ground_truth()
        if gt.get('known') is False:
            return {'truth': None, 'predicted': None, 'accuracy': None,
                    'correct': None, 'spurious': None, 'missed': None,
                    'status': 'GROUND_TRUTH_UNAVAILABLE'}
        truth = set(gt['root_components'])
        if self.method.kind == 'locality' or self.method.level in (AblationLevel.C0,
                                                                    AblationLevel.C1):
            key_to_comp = {repair: comp for _, comp, _, _, repair in MECHANISM_CATALOG}
            predicted = {key_to_comp[k] for k, v in self.current.items()
                         if v and k in key_to_comp}
        else:
            all_m = list(self.registry.active()) + list(self.registry.archived.values())
            # a component counts as a predicted root when it holds a
            # root-repair confirmation under ANY applicability scope (the
            # decision views are ablation-filtered, so levels without the
            # bridge view never report scoped root confirmations either)
            roots = [m for m in all_m if m.root_confirmations()]
            predicted = ({m.component for m in roots} if roots else
                         {m.component for m in all_m if m.f_status == 'CONSISTENT'})
        inter = truth & predicted
        union = truth | predicted
        return {'truth': sorted(truth), 'predicted': sorted(predicted),
                'accuracy': len(inter) / len(union) if union else 1.0,
                'correct': sorted(inter), 'spurious': sorted(predicted - truth),
                'missed': sorted(truth - predicted)}


def _jsonable(obj):
    from dataclasses import fields, is_dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
