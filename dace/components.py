"""Coupled Mechanism Version Space (CMVS) and ablation-level components.

Implements the shared mechanism space of DACE section 2.1: each mechanism m
carries a source component s, an applicability premise c, a propagation path p,
and intervention predictions O_m(a,x) = (B, A, Z, Y). Mechanism states track
the two attribution views plus the bridge between them:

    F_m          failure-source claim      (baseline boundary violation)
    R_{m,a}      edit-contribution claim   (activation/mediator/utility)
    Z_{m,a}      bridge status             (couples F and R; section 2.3)

Ablation levels (experiment E2):
    C0  basic diagnose-modify-verify loop (no mechanism space)
    C1  + shared competing mechanism space (CMVS), no confirmatory attribution
    C2  + failure-source attribution (frozen confirmatory failure contracts)
    C3  + independent contribution attribution  -> Uncoupled Dual
    C4  + dual-attribution bridging and joint update
    C5  + attribution-driven discriminative intervention -> Full DACE
    CONTRIBUTION_ONLY  control: contribution attribution without failure side
"""
from __future__ import annotations

import json

from dataclasses import dataclass, field
from enum import Enum

from .harness import BOUNDARY_FIELDS, CONFIG_KEYS, Fault, SimulatedEnvironment
from .core import Prediction

PROPOSED, CONSISTENT, CONTRADICTED, UNRESOLVED = 'PROPOSED', 'CONSISTENT', 'CONTRADICTED', 'UNRESOLVED'


class AblationLevel(str, Enum):
    C0 = 'C0'
    C1 = 'C1'
    C2 = 'C2'
    C3 = 'C3'
    C4 = 'C4'
    C5 = 'C5'
    CONTRIBUTION_ONLY = 'CONTRIBUTION_ONLY'


#: which attribution machinery each level activates
LEVEL_FEATURES = {
    AblationLevel.C0: dict(mechanisms=False, failure=False, contribution=False, bridge=False, probe=False),
    AblationLevel.C1: dict(mechanisms=True, failure=False, contribution=False, bridge=False, probe=False),
    AblationLevel.C2: dict(mechanisms=True, failure=True, contribution=False, bridge=False, probe=False),
    AblationLevel.C3: dict(mechanisms=True, failure=True, contribution=True, bridge=False, probe=False),
    AblationLevel.C4: dict(mechanisms=True, failure=True, contribution=True, bridge=True, probe=False),
    AblationLevel.C5: dict(mechanisms=True, failure=True, contribution=True, bridge=True, probe=True),
    AblationLevel.CONTRIBUTION_ONLY: dict(mechanisms=True, failure=False, contribution=True, bridge=False, probe=False),
}


@dataclass
class Mechanism:
    """CMVS entry: m = (s, c, p, g, A_m, O_hat, r)."""
    id: str
    component: str                      # s in L_t
    source_field: str                   # boundary predicate naming the source contract
    propagation: tuple                  # p: component path from source to symptom
    repair_key: str                     # the action that repairs s at the source
    premise: str = 'global'
    #: claim states (F, R, Z) -- updated only by confirmatory evidence
    f_status: str = PROPOSED            # failure-source claim
    r_status: str = PROPOSED            # edit-contribution claim
    z_bridge: str = 'UNBRIDGED'         # SOURCE_REPAIR / COMPENSATION / INTERACTION / UNRESOLVED / UNBRIDGED
    joint_note: str = ''                # joint-update revision note (section 2.3)
    #: artifact version the current claims were earned against (stale-anchor
    #: audit: claims are bound to the exact parent version they were frozen on)
    claim_version: str = ''
    #: response branches keyed by the EXACT artifact each claim was earned on,
    #: then by the contract's applicability scope expression:
    #: {artifact: {scope: {failure, contribution, bridge,
    #:                     action_repairs_source}}}.
    #: Claims are per (mechanism, action, artifact, scope): evidence earned
    #: under one applicability condition never overwrites the branch of
    #: another. A newer artifact's unresolved response never inherits an
    #: older artifact's bridge, and a foreign-action (discrimination) claim
    #: never overwrites the mechanism's own decision slots.
    responses: dict = field(default_factory=dict)
    #: latest OWN-ACTION claim per applicability scope expression
    #: ({scope: {failure, contribution, bridge, artifact}}). The global
    #: decision slots above speak only for the unconditional ('global')
    #: scope; a scoped claim (e.g. "cacheable=true") is recorded here and
    #: never taken to represent tasks outside its condition.
    scoped_views: dict = field(default_factory=dict)

    def response_for(self, artifact: str, scope: str = 'global') -> dict | None:
        """The claim this mechanism earned against one exact artifact version
        under one applicability scope."""
        return self.responses.get(artifact, {}).get(scope)

    def _global_view(self) -> dict:
        return {'failure': self.f_status, 'contribution': self.r_status,
                'bridge': self.z_bridge, 'artifact': self.claim_version}

    def claim_view(self, scope: str = 'global') -> dict | None:
        """Own-action decision view under ONE applicability scope.

        'global' reads the decision slots (the unconditional view). A scoped
        query reads the scoped view earned under exactly that condition;
        when no scoped view exists for it, the GLOBAL slots apply
        (unconditional evidence covers every subpopulation). A conclusion
        earned under a DIFFERENT condition is unknown here and never leaks
        into this query: scope mismatch stays unknown.
        """
        if scope == 'global':
            return self._global_view()
        view = self.scoped_views.get(scope)
        if view is not None:
            return view
        return self._global_view()

    def root_confirmations(self) -> list:
        """[(scope, view)] for every applicability scope where the latest
        own-action claim reads as a confirmed root repair.

        The global slots speak only for the unconditional scope; each scoped
        view speaks only for its own condition -- never for tasks outside
        it. Composing conditional roots requires pairwise-compatible scopes
        (see joint_compositions).
        """
        out = []
        if self.f_status == CONSISTENT and self.z_bridge == 'SOURCE_REPAIR':
            out.append(('global', self._global_view()))
        for scope in sorted(self.scoped_views):
            v = self.scoped_views[scope]
            if v.get('failure') == 'CONSISTENT' and v.get('bridge') == 'SOURCE_REPAIR':
                out.append((scope, v))
        return out

    @property
    def root_confirmed(self) -> bool:
        """Root-repair confirmation under the GLOBAL (unconditional) scope
        only. Conditional conclusions are queried via claim_view(scope) /
        root_confirmations() and never become unconditional state."""
        return self.f_status == CONSISTENT and self.z_bridge == 'SOURCE_REPAIR'

    @property
    def demoted_as_compensation(self) -> bool:
        """Compensation demotion under the GLOBAL scope only. A mechanism
        demoted under one condition stays selectable in scopes where its
        contract earned source-repair evidence."""
        return self.z_bridge == 'COMPENSATION'

    @property
    def contribution_positive(self) -> bool:
        """POSITIVE edit-contribution evidence in the GLOBAL view only."""
        return self.r_status == 'POSITIVE'


class MechanismRegistry:
    """CMVS with active-configuration cap K_active (frozen per round)."""

    def __init__(self, cap: int = 4):
        self.cap = cap
        self.mechanisms: dict[str, Mechanism] = {}
        self.archived: dict[str, Mechanism] = {}

    def propose(self, m: Mechanism) -> Mechanism:
        if m.id in self.mechanisms or m.id in self.archived:
            return self.mechanisms.get(m.id) or self.archived[m.id]
        if len(self.mechanisms) >= self.cap:
            # archive the stalest proposed-but-undistinguished entry
            candidates = [x for x in self.mechanisms.values()
                          if x.f_status == PROPOSED and not x.root_confirmed]
            if candidates:
                victim = sorted(candidates, key=lambda x: x.id)[0]
                self.archived[victim.id] = self.mechanisms.pop(victim.id)
            elif len(self.mechanisms) >= self.cap:
                raise ValueError('Mechanism cap reached with no archivable entry')
        self.mechanisms[m.id] = m
        return m

    def active(self) -> list:
        return list(self.mechanisms.values())

    def competing_pairs(self):
        ms = [m for m in self.mechanisms.values()
              if m.f_status != CONTRADICTED and not m.demoted_as_compensation]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if ms[i].component != ms[j].component or ms[i].source_field != ms[j].source_field:
                    yield ms[i], ms[j]


#: catalog of candidate mechanisms matched against observed symptoms (section 2.1:
#: trace-constrained competing hypotheses). Offline proposals come from here; the
#: live LLM path must emit the same schema.
MECHANISM_CATALOG = (
    # id, component, source_field, trigger symptom, repair key
    ('m_planner_type', 'planner', 'planner_valid', 'planner_valid', 'planner_cast'),
    ('m_planner_cache', 'planner', 'cache_fresh', 'cache_fresh', 'planner_refresh'),
    ('m_adapter_garble', 'adapter', 'adapter_valid', 'adapter_valid', 'adapter_cast'),
    ('m_tool_offbyone', 'tool', 'tool_success', 'tool_success', 'tool_strict'),
    ('m_verifier_drift', 'verifier', 'verifier_agrees', 'verifier_agrees', 'verifier_recheck'),
)

PROPAGATION = {
    'm_planner_type': ('planner', 'adapter', 'tool'),
    'm_planner_cache': ('planner', 'adapter', 'tool'),
    'm_adapter_garble': ('adapter', 'tool'),
    'm_tool_offbyone': ('tool',),
    'm_verifier_drift': ('verifier',),
}


def offline_propose(symptoms: dict, registry: MechanismRegistry) -> list:
    """Deterministic mechanism proposals from observed symptom rates."""
    out = []
    for mid, comp, src, trigger, repair in MECHANISM_CATALOG:
        if symptoms['symptom_rates'].get(trigger, 0) > 0:
            m = Mechanism(mid, comp, src, PROPAGATION[mid], repair)
            out.append(registry.propose(m))
    return out


def _compatible(pred_a, pred_b) -> bool:
    """Two applicability predicates are compatible when they assign no
    contradictory value to a shared key (None = unconditional)."""
    if pred_a is None or pred_b is None:
        return True
    return all(pred_a[k] == pred_b[k] for k in set(pred_a) & set(pred_b))


def joint_compositions(registry: MechanismRegistry) -> list:
    """Joint root-repair compositions from CONFIRMED roots (section 2.3).

    Roots confirmed under different applicability scopes compose ONLY when
    their scopes are pairwise compatible; the composition's applicability
    premise is the merged predicate and its confirmatory contract runs
    under exactly that scope. A root confirmed under some condition is
    never merged into an unconditional (or a mutually exclusive)
    composition -- "holds under some condition" is not "holds for this
    action". Each mechanism contributes at most one confirmation (its
    global one when present, else its first scoped one).
    """
    entries = []
    for m in registry.active():
        for scope, _view in m.root_confirmations():
            pred = None if scope == 'global' else json.loads(scope)
            entries.append((m, pred))
    compositions, assigned = [], set()
    for m, pred in sorted(entries, key=lambda e: e[0].id):
        if m.id in assigned:
            continue
        group, merged = [(m, pred)], (dict(pred) if pred else {})
        assigned.add(m.id)
        for m2, pred2 in sorted(entries, key=lambda e: e[0].id):
            if m2.id in assigned:
                continue
            if all(_compatible(pred2, p) for _, p in group):
                group.append((m2, pred2))
                assigned.add(m2.id)
                if pred2:
                    merged.update(pred2)
        if len(group) >= 2:
            patch = {gm.repair_key: True for gm, _ in group}
            compositions.append({'patch': patch,
                                 'premise': merged or None,
                                 'roots': [gm for gm, _ in group],
                                 'key': 'joint:' + '+'.join(sorted(patch))})
    return compositions


def candidate_patches(registry: MechanismRegistry, joint: bool = False) -> list:
    """Patch construction from the mechanism space (section 2.2).

    Single-key repairs for every active mechanism; with joint=True the
    joint root-repair compositions from pairwise-scope-compatible confirmed
    roots (dual-attribution-guided patch construction) are appended. The
    compensation competitors remain in the pool.
    """
    patches = {}
    for m in registry.active():
        patches.setdefault(m.repair_key, {m.repair_key: True})
    out = list(patches.values())
    if joint:
        for comp in joint_compositions(registry):
            out.append(dict(comp['patch']))
    return out


def prediction_disagreement(pred_i: Prediction, pred_j: Prediction) -> float:
    """D(a; M) for ONE frozen action under two competing predictions.

    Compared ONLY on fields BOTH predictions explicitly freeze in their
    intervention boundaries: those are the expectations each mechanism
    actually committed to for this action, so one run of the action can
    refute exactly one of them where they differ. A field a prediction does
    NOT name is UNKNOWN for that mechanism -- it is never backfilled from a
    baseline value, from the other prediction, or from an implicit "this
    action changes nothing" assumption (an inert expectation must be
    written into the frozen prediction itself to be scored). With no
    commonly named field the action cannot separate the hypotheses and the
    score is exactly 0.0.

    Mechanism identity, component or source-field disjointness are never
    scored as disagreement -- only differing committed expectations are.
    """
    int_i = dict(pred_i.intervention_boundary)
    int_j = dict(pred_j.intervention_boundary)
    common = sorted(set(int_i) & set(int_j))
    if not common:
        return 0.0
    return sum(1 for k in common if int_i[k] != int_j[k]) / len(common)


def locality_patch(symptoms: dict, tried: set) -> dict | None:
    """Locality-first heuristic baseline (E1 comparison control).

    This is NOT the official AutoSaddler optimizer: it is a hand-written
    locality-first heuristic that trusts the naive downstream-to-upstream
    localization summary, flips the repair key of the nearest failing
    component, and reflects (retries the next candidate) on rejection. No
    mechanism space, no attribution, no LLM. Results under this baseline
    must be reported under the name "LocalityHeuristic", never "AutoSaddler";
    wiring up the real AutoSaddler implementation is a separate, pending
    integration that must freeze its version and share the base harness,
    execution model, tasks and budget.
    """
    comp = symptoms['naive_localization']
    for _, c, _, _, repair in MECHANISM_CATALOG:
        if c == comp and repair not in tried:
            return {repair: True}
    # reflection: fall through to any untried key in catalog order
    for _, _, _, _, repair in MECHANISM_CATALOG:
        if repair not in tried:
            return {repair: True}
    return None


def scenario(name: str) -> SimulatedEnvironment:
    """Controlled fault scenarios for experiment E4.

    Scope is chosen so per-patch utility deltas stay large enough for the
    fixed-look Hoeffding bounds at the default sample sizes.
    """
    table = {
        # known source error at the planner; adapter_cast only compensates
        'source_error': (Fault('PLANNER_TYPE_ERROR', 'planner', 'planner_cast'),),
        # stale planner cache: symptom surfaces at the adapter boundary;
        # adapter_cast re-derives values and fully compensates downstream
        'downstream_compensation': (Fault('STALE_PLANNER_CACHE', 'planner', 'planner_refresh'),),
        # two independent faults; single patches only partially repair
        'compound_fault': (Fault('PLANNER_TYPE_ERROR', 'planner', 'planner_cast'),
                           Fault('TOOL_OFF_BY_ONE', 'tool', 'tool_strict', scope=0.5)),
        # same stale-cache fault, but the naive locator blames the adapter
        # (mislocalization): only bridged/probing attribution relocalizes
        'mislocalization': (Fault('STALE_PLANNER_CACHE', 'planner', 'planner_refresh'),),
        # verifier-side drift for completeness of the component space
        'verifier_drift': (Fault('VERIFIER_DRIFT', 'verifier', 'verifier_recheck', scope=0.5),),
        # benchmark mix: several concurrent faults across components (E1-E3)
        'benchmark_mix': (Fault('PLANNER_TYPE_ERROR', 'planner', 'planner_cast'),
                          Fault('STALE_PLANNER_CACHE', 'planner', 'planner_refresh', scope=0.5),
                          Fault('TOOL_OFF_BY_ONE', 'tool', 'tool_strict', scope=0.5)),
    }
    return SimulatedEnvironment(table[name])
