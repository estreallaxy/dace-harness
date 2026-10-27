from __future__ import annotations
import hashlib
import json
import math
import threading
import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from .predicate_evidence import PredicateProtocol, evaluate_predicates


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Artifact:
    version: str
    parent: str | None
    content: str
    kind: str = 'PATCH'

    @classmethod
    def create(cls, content, parent=None, kind='PATCH'):
        return cls(digest([content, parent, kind]), parent, content, kind)


@dataclass(frozen=True)
class Prediction:
    mechanism: str
    source: str
    propagation: tuple[str, ...]
    baseline_boundary: tuple[tuple[str, object], ...]
    intervention_boundary: tuple[tuple[str, object], ...]
    mediator: tuple[tuple[str, object], ...]
    bridge: str
    #: whether the frozen action repairs THIS prediction's named source at
    #: the source (own action) or is a foreign action scheduled on this
    #: mechanism for discrimination (inert under this mechanism's hypothesis).
    #: Claim consumption is version- and action-scoped: only own-action
    #: claims may update a mechanism's decision slots.
    action_repairs_source: bool = True
    # Evidence predicates must name observable fields; absent fields stay unresolved.
    def __post_init__(self):
        for fields in (self.baseline_boundary, self.intervention_boundary, self.mediator):
            if type(fields) is not tuple or any(type(pair) is not tuple or len(pair) != 2
                    or not isinstance(pair[0], str) or type(pair[1]) not in (str, int, float, bool, type(None))
                    for pair in fields):
                raise ValueError('Predicates must be immutable tuples of scalar expectations')
        if self.bridge not in ('SOURCE_REPAIR', 'COMPENSATION', 'INTERACTION', 'UNRESOLVED'):
            raise ValueError('Invalid bridge type')


@dataclass(frozen=True)
class Contract:
    artifact: str
    parent: str
    scope: str
    tasks: tuple[str, ...]
    predictions: tuple[Prediction, ...]
    provenance: str = 'CONFIRMATORY'
    cost_per_task: int = 2
    alpha: float = .05

    def __post_init__(self):
        if type(self.tasks) is not tuple or type(self.predictions) is not tuple or not self.predictions:
            raise ValueError('Frozen tasks and predictions must be nonempty immutable tuples')
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError('Frozen tasks must be nonempty and unique')
        if self.provenance not in ('EXPLORATORY', 'CONFIRMATORY'):
            raise ValueError('Invalid provenance')
        if self.cost_per_task <= 0 or not 0 < self.alpha < 1:
            raise ValueError('Invalid cost or error rate')

    @property
    def id(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class StatisticalContract(Contract):
    """Versioned protocol; legacy Contract IDs and journal schemas stay unchanged."""
    predicate_protocol: PredicateProtocol = field(default_factory=PredicateProtocol)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.predicate_protocol,PredicateProtocol):
            raise ValueError('Statistical contract requires an immutable predicate protocol')
        for prediction in self.predictions:
            for predicates_ in (prediction.baseline_boundary,prediction.intervention_boundary,prediction.mediator):
                if (not predicates_ or len(dict(predicates_))!=len(predicates_)
                        or any(type(value) is not bool for _,value in predicates_)):
                    raise ValueError('Statistical predicates require unique, nonempty boolean fields')


class Budget:
    """Mutually exclusive owner locks and free funds, denominated in cost units."""
    def __init__(self, total):
        if type(total) not in (int, float) or not math.isfinite(total) or total < 0:
            raise ValueError('Negative budget')
        self.total, self.spent, self.locks = total, 0, {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_amount(amount):
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise ValueError('Budget amount must be finite and nonnegative')

    @property
    def free(self):
        return self.total - self.spent - sum(self.locks.values())

    def reserve(self, owner, amount):
        with self._lock:
            self._validate_amount(amount)
            if owner in self.locks:
                return  # Idempotent continuation, never top up a contract.
            if amount < 0 or amount > self.free:
                raise ValueError('Insufficient unreserved budget')
            self.locks[owner] = amount

    def charge(self, amount, owner=None):
        with self._lock:
            self._validate_amount(amount)
            available = self.free if owner is None else self.locks.get(owner, 0)
            if amount < 0 or amount > available:
                raise ValueError('Owner budget exhausted')
            if owner is not None:
                self.locks[owner] -= amount
            self.spent += amount

    def release(self, owner):
        with self._lock:
            return self.locks.pop(owner, 0)


class Ledger:
    """Append-only JSONL with hash chaining; raw observations are never relabelled."""
    def __init__(self, path, durable=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.durable = durable
        self._lock = threading.RLock()
        self._lease = None
        self._closed = False
        if durable:
            self._acquire_writer_lease()
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                event = json.loads(line)
                expected = digest({k: v for k, v in event.items() if k != 'hash'})
                if (event['hash'] != expected or event['index'] != len(self.events)
                        or event['previous'] != (self.events[-1]['hash'] if self.events else None)):
                    raise ValueError('Ledger integrity failure')
                self.events.append(event)

    def append(self, kind, **payload):
        with self._lock:
            if self._closed:
                raise ValueError('Ledger is closed')
            # Snapshot caller-owned data: later config/summary edits must never
            # rewrite an in-memory event after its hash and disk row were frozen.
            snapshot = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
            event = dict(index=len(self.events), previous=self.events[-1]['hash'] if self.events else None,
                         kind=kind, payload=snapshot)
            event['hash'] = digest(event)
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
                f.flush()
                if self.durable:
                    os.fsync(f.fileno())
            self.events.append(event)
            return copy.deepcopy(event)

    def _acquire_writer_lease(self):
        lease = self.path.with_suffix(self.path.suffix + '.lock').open('a+b')
        try:
            lease.seek(0, os.SEEK_END)
            if lease.tell() == 0:
                lease.write(b'0')
                lease.flush()
            lease.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            lease.close()
            raise RuntimeError('LEDGER_ALREADY_OWNED: another durable writer is active') from None
        self._lease = lease

    def close(self):
        with self._lock:
            if self._lease is not None:
                self._lease.close()  # OS releases the byte/flock lease, including on crash.
                self._lease = None
            self._closed = True

    def __del__(self):
        lease = getattr(self, '_lease', None)
        if lease is not None:
            lease.close()


class Exposure:
    def __init__(self, ledger):
        self.ledger = ledger
        self.used = {}  # task -> roles. Conservative global exclusion includes ancestors.
        self.disclosed = set()
        for e in ledger.events:
            p = e['payload']
            if e['kind'] == 'RUN_USED':
                self.used.setdefault(p['task'], set()).add(p['role'])
            elif e['kind'] == 'AGENT_DISCLOSED':
                self.disclosed.update(p['tasks'])

    def run(self, task, role, artifact):
        roles = self.used.get(task, set())
        if roles and roles != {role}:
            raise ValueError('Task roles must remain disjoint')
        self.ledger.append('RUN_USED', task=task, role=role, artifact=artifact)
        self.used.setdefault(task, set()).add(role)

    def disclose(self, tasks, fields, lineage):
        self.ledger.append('AGENT_DISCLOSED', tasks=list(tasks), fields=list(fields), lineage=lineage)
        self.disclosed.update(tasks)

    def require_fresh(self, tasks):
        if any(t in self.used or t in self.disclosed for t in tasks):
            raise ValueError('Task or derived evidence has already been used/disclosed')


@dataclass(frozen=True)
class Observation:
    task: str
    baseline: dict
    intervention: dict
    before: float
    after: float
    activated: bool | None

    def __post_init__(self):
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (self.before, self.after)):
            raise ValueError('Utilities must be bounded in [0, 1]')


def lower_bound(deltas, alpha):
    """One-sided Hoeffding for independent task-level paired differences in [-1,1]."""
    if not deltas or not 0 < alpha < 1:
        return None
    if any(not math.isfinite(x) or not -1 <= x <= 1 for x in deltas):
        raise ValueError('Invalid paired differences')
    return sum(deltas) / len(deltas) - math.sqrt(2 * math.log(1 / alpha) / len(deltas))


def upper_bound(deltas, alpha):
    """One-sided Hoeffding upper bound (direction-matched companion of
    lower_bound). A regression is statistically CONFIRMED only when this
    bound sits below zero; a negative LOWER bound alone merely means the
    run cannot yet rule a regression out."""
    if not deltas or not 0 < alpha < 1:
        return None
    if any(not math.isfinite(x) or not -1 <= x <= 1 for x in deltas):
        raise ValueError('Invalid paired differences')
    return sum(deltas) / len(deltas) + math.sqrt(2 * math.log(1 / alpha) / len(deltas))


def predicates(rows, expected):
    if not expected or any(any(k not in r for k, _ in expected) for r in rows):
        return 'UNRESOLVED'
    return 'CONSISTENT' if all(all(r[k] == v for k, v in expected) for r in rows) else 'CONTRADICTED'


class Controller:
    def __init__(self, ledger, budget, exposure):
        self.ledger, self.budget, self.exposure = ledger, budget, exposure
        self.contracts, self.observations, self.finished = {}, {}, set()
        self.terminal_results = {}

    @classmethod
    def restore(cls, path, total_budget):
        """Recover journaled confirmation obligations, never restart a run blindly.

        Only a ledger originally using PersistentBudget can be resumed; legacy
        memory-only budget histories cannot be reconstructed without inventing
        costs. The outer optimization loop has its own lifecycle and is not
        implicitly resumed by this API.
        """
        from .durable import restore_controller
        return restore_controller(path, total_budget)

    def freeze(self, artifact, contract):
        if contract.artifact != artifact.version or contract.parent != artifact.parent:
            raise ValueError('Exact artifact and direct parent must match')
        if artifact.kind not in ('PATCH', 'JOINT_PATCH', 'PROBE'):
            raise ValueError('Unsupported artifact kind')
        if artifact != Artifact.create(artifact.content, artifact.parent, artifact.kind):
            raise ValueError('Artifact content hash mismatch')
        if contract.id in self.contracts:
            return
        self.exposure.require_fresh(contract.tasks)
        # Reserve all stages atomically before any confirmatory execution.
        self.budget.reserve(contract.id, len(contract.tasks) * contract.cost_per_task)
        self.ledger.append('CONTRACT_FROZEN', contract=asdict(contract), contract_id=contract.id,
                           artifact=asdict(artifact))
        self.contracts[contract.id] = contract
        self.observations[contract.id] = {}

    def resume(self, contract_id, execute: Callable, batch=None):
        c = self.contracts[contract_id]
        if contract_id in self.finished:
            return self.evaluate(contract_id)
        done = self.observations[contract_id]
        remaining = [t for t in c.tasks if t not in done]
        for task in remaining[:batch] if batch is not None else remaining:
            # Charge attempted runs too; errors terminate without replacing samples.
            if hasattr(self.budget, 'charge_confirmation'):
                self.budget.charge_confirmation(c.cost_per_task, c.id, task)
            else:
                self.budget.charge(c.cost_per_task, c.id)
            self.exposure.run(task, 'PROBE', c.artifact)
            try:
                obs = execute(task)
                if obs.task != task:
                    raise ValueError('Statistical unit mismatch')
            except Exception as exc:
                self.finished.add(c.id)
                self.budget.release(c.id)
                self.ledger.append('TERMINATED_UNRESOLVED', contract_id=c.id, reason=type(exc).__name__)
                result = {'status': 'TERMINATED_UNRESOLVED'}
                self.terminal_results[c.id] = result
                return copy.deepcopy(result)
            self.ledger.append('OBSERVATION', contract_id=c.id, artifact=c.artifact,
                               provenance=c.provenance, observation=asdict(obs))
            done[task] = obs
        if len(done) != len(c.tasks):
            self.ledger.append('PENDING_CONFIRM', contract_id=c.id,
                               remaining=[t for t in c.tasks if t not in done],
                               reserved_cost=self.budget.locks[c.id])
            return {'status': 'EVIDENCE_INCOMPLETE', 'claims': []}
        self.finished.add(c.id)
        self.budget.release(c.id)
        result = self.evaluate(c.id)
        self.ledger.append('EVIDENCE_RESULT', contract_id=c.id, result=result)
        self.terminal_results[c.id] = copy.deepcopy(result)
        return result

    def evaluate(self, contract_id):
        if contract_id in self.terminal_results:
            return copy.deepcopy(self.terminal_results[contract_id])
        c, rows = self.contracts[contract_id], list(self.observations[contract_id].values())
        if len(rows) != len(c.tasks):
            return {'status': 'EVIDENCE_INCOMPLETE', 'claims': []}
        if c.provenance == 'EXPLORATORY':
            return {'status': 'EXPLORATORY_ONLY', 'claims': []}
        claims = []
        alpha = c.alpha / max(1, len(c.predictions))
        for p in c.predictions:
            empirical = None
            if isinstance(c,StatisticalContract):
                # Split this prediction's alpha across all three predicate
                # families and its utility contrast before looking at any data.
                empirical = {}
                for name,expected,side in (
                        ('failure',p.baseline_boundary,'baseline'),
                        ('boundary',p.intervention_boundary,'intervention'),
                        ('mediator',p.mediator,'intervention')):
                    empirical[name] = evaluate_predicates(
                        [{'task':r.task,'boundary':getattr(r,side)} for r in rows],dict(expected),
                        alpha=alpha/4,max_violation_rate=c.predicate_protocol.max_violation_rate)
                failure,boundary,mediator = (empirical[k]['status'] for k in ('failure','boundary','mediator'))
            else:
                failure = predicates([r.baseline for r in rows], p.baseline_boundary)
                boundary = predicates([r.intervention for r in rows], p.intervention_boundary)
                mediator = predicates([r.intervention for r in rows], p.mediator)
            activation = all(r.activated is True for r in rows)
            utility = lower_bound([r.after - r.before for r in rows], alpha/4 if empirical else alpha)
            contribution = 'POSITIVE' if activation and mediator == 'CONSISTENT' and utility > 0 else 'UNRESOLVED'
            bridge = 'UNRESOLVED'
            # The bridge witness is decided on source-boundary evidence alone:
            # a source repair flips the named source field (with its mediator
            # downstream), while a compensation keeps the source violated while
            # the downstream recovers. Task utility is deliberately NOT required
            # here -- a masked benefit (section 2.3: the source is repaired but a
            # further fault still suppresses the terminal utility) must retain its
            # source-repair evidence rather than be contradicted into UNRESOLVED.
            # An execution that never activated the edit (hook reports False or
            # is missing entirely) can support NO bridge conclusion at all:
            # the edit was never exercised, so neither repair nor compensation
            # was observed.
            if activation and failure == boundary == mediator == 'CONSISTENT':
                base = dict(p.baseline_boundary)
                post = dict(p.intervention_boundary)
                if base.get(p.source) is False:
                    if post.get(p.source) is True and p.bridge == 'SOURCE_REPAIR':
                        bridge = 'SOURCE_REPAIR'
                    elif post.get(p.source) is False and p.bridge == 'COMPENSATION':
                        bridge = 'COMPENSATION'
                # INTERACTION stays unresolved without four legal factorial contrasts.
            claim = dict(mechanism=p.mechanism, failure=failure, boundary=boundary,
                               mediator=mediator, contribution=contribution, utility_lower=utility,
                               bridge=bridge, artifact=c.artifact, scope=c.scope,
                               action_repairs_source=p.action_repairs_source)
            if empirical is not None:
                claim['predicate_evidence'] = empirical
                claim['utility_alpha'] = alpha/4
                claim['causal_identification'] = False
            claims.append(claim)
        minimal = all(r.activated is not None and bool(r.intervention) for r in rows)
        complete = minimal and all(r.activated is True for r in rows) and all(x['failure'] != 'UNRESOLVED' and x['boundary'] != 'UNRESOLVED'
                                   and x['mediator'] != 'UNRESOLVED' for x in claims)
        return {'status': 'EVIDENCE_COMPLETE' if complete else 'EVIDENCE_INCOMPLETE',
                'minimal': 'MINIMAL_OBSERVED' if minimal and all(r.activated for r in rows) else 'MINIMAL_UNRESOLVED',
                'claims': claims}


def promotion(candidates, families, alpha=.05, delta=0., epsilon=0., max_cost=math.inf):
    """Frozen candidate/family collection, simultaneous bounds for ALL claims.

    candidates maps version -> {pairs: [(task_id, family, before, after)], cost, size}.
    Tasks must be independent experimental units sampled under the frozen protocol.
    """
    if not candidates or not 0 < alpha < 1:
        return {'winner': None, 'results': {}}
    per_claim = alpha / (len(candidates) * (1 + len(families)))
    results = {}
    for version, candidate in candidates.items():
        pairs = candidate['pairs']
        if len({r[0] for r in pairs}) != len(pairs):
            raise ValueError('Duplicate statistical units')
        deltas = [r[3] - r[2] for r in pairs]
        overall = lower_bound(deltas, per_claim)
        regressions = {f: lower_bound([r[3] - r[2] for r in pairs if r[1] == f], per_claim) for f in families}
        # per-family mean and BOTH one-sided bounds: deployment regression
        # reporting must separate "statistically confirmed regression"
        # (direction-matched upper bound below zero) from "non-inferiority
        # not yet established" (lower bound below zero). A negative lower
        # bound is NOT an observed regression: small samples, zero change
        # and even positive true effects can all produce one.
        family_details = {}
        for f in families:
            fd = [r[3] - r[2] for r in pairs if r[1] == f]
            if fd:
                family_details[f] = {'mean': sum(fd) / len(fd),
                                     'lower': lower_bound(fd, per_claim),
                                     'upper': upper_bound(fd, per_claim)}
            else:
                family_details[f] = {'mean': None, 'lower': None, 'upper': None}
        eligible = candidate['cost'] <= max_cost and overall is not None and overall > delta and all(v is not None and v >= -epsilon for v in regressions.values())
        results[version] = dict(eligible=eligible, utility_lower=overall, regressions=regressions,
                                family_details=family_details)
    eligible = [v for v in candidates if results[v]['eligible']]
    winner = min(eligible, key=lambda v: (-results[v]['utility_lower'], candidates[v]['cost'], candidates[v]['size'], v)) if eligible else None
    return {'winner': winner, 'results': results, 'alpha_per_claim': per_claim}
