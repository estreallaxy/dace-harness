"""Journaled budget ownership and exact fixed-look confirmation recovery.

Single writer per ledger. A started execution without a committed observation
is ambiguous after a crash and terminates unresolved: never silently rerun or
replace its frozen task. Safe partial batches resume only their unstarted units.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

from .core import (Artifact, Budget, Contract, Controller, Exposure, Ledger,
                   Observation, Prediction, StatisticalContract, digest)
from .predicate_evidence import PredicateProtocol


class PersistentBudget(Budget):
    def __init__(self, total, ledger):
        super().__init__(total)
        self.ledger = ledger
        self.attempts = {}
        budget_events = [e for e in ledger.events if e['kind'].startswith('BUDGET_')]
        if not budget_events:
            if ledger.events:
                raise ValueError('LEGACY_BUDGET_NOT_RECOVERABLE: no journaled initial allocation')
            ledger.append('BUDGET_INITIALIZED', total=total, schema='dace-budget/v1')
            return
        initial = budget_events[0]
        if (initial['kind'] != 'BUDGET_INITIALIZED' or initial['payload'].get('total') != total
                or initial['payload'].get('schema') != 'dace-budget/v1'):
            raise ValueError('Frozen budget identity mismatch')
        for event in budget_events[1:]:
            p = event['payload']
            if event['kind'] == 'BUDGET_RESERVED':
                Budget.reserve(self, p['owner'], p['amount'])
            elif event['kind'] == 'BUDGET_CHARGED':
                unit = p.get('unit')
                if unit:
                    key = digest(unit)
                    if key in self.attempts:
                        raise ValueError('Duplicate charged confirmation unit')
                    self.attempts[key] = copy.deepcopy(unit)
                Budget.charge(self, p['amount'], p.get('owner'))
            elif event['kind'] == 'BUDGET_RELEASED':
                if Budget.release(self, p['owner']) != p['amount']:
                    raise ValueError('Budget release journal mismatch')
            else:
                raise ValueError('Unknown durable budget event')

    def _transaction(self, method, kind, *args, unit=None, **kwargs):
        with self._lock:
            preview = Budget(self.total)
            preview.spent, preview.locks = self.spent, dict(self.locks)
            result = getattr(Budget, method)(preview, *args, **kwargs)
            if unit and digest(unit) in self.attempts:
                raise ValueError('Confirmation task already attempted; no implicit retry')
            if method == 'reserve':
                if args[0] in self.locks:
                    return result  # Idempotent continuation, no second allocation.
                payload = {'owner':args[0], 'amount':args[1]}
            elif method == 'charge':
                payload = {'amount':args[0], 'owner':kwargs.get('owner'), 'unit':unit}
            else:
                payload = {'owner':args[0], 'amount':result}
            self.ledger.append(kind, **payload)
            self.spent, self.locks = preview.spent, preview.locks
            if unit:
                self.attempts[digest(unit)] = copy.deepcopy(unit)
            return result

    def reserve(self, owner, amount):
        return self._transaction('reserve', 'BUDGET_RESERVED', owner, amount)

    def charge(self, amount, owner=None):
        return self._transaction('charge', 'BUDGET_CHARGED', amount, owner=owner)

    def charge_confirmation(self, amount, contract_id, task):
        return self._transaction('charge', 'BUDGET_CHARGED', amount, owner=contract_id,
                                 unit={'contract_id':contract_id, 'task':task})

    def release(self, owner):
        if owner not in self.locks:
            return 0
        return self._transaction('release', 'BUDGET_RELEASED', owner)


def contract_from_dict(data):
    data = copy.deepcopy(data)
    predictions = []
    for raw in data['predictions']:
        raw['propagation'] = tuple(raw['propagation'])
        for field in ('baseline_boundary', 'intervention_boundary', 'mediator'):
            raw[field] = tuple(tuple(pair) for pair in raw[field])
        predictions.append(Prediction(**raw))
    data['predictions'], data['tasks'] = tuple(predictions), tuple(data['tasks'])
    if 'predicate_protocol' in data:
        data['predicate_protocol'] = PredicateProtocol(**data['predicate_protocol'])
        return StatisticalContract(**data)
    return Contract(**data)


def restore_controller(path: Path | str, total_budget):
    ledger = Ledger(path, durable=True)
    try:
        return _restore_controller(ledger, total_budget)
    except BaseException:
        ledger.close()
        raise


def _restore_controller(ledger, total_budget):
    budget = PersistentBudget(total_budget, ledger)
    controller = Controller(ledger, budget, Exposure(ledger))
    artifacts = {}
    for event in list(ledger.events):
        p = event['payload']
        kind = event['kind']
        if kind == 'CONTRACT_FROZEN':
            contract = contract_from_dict(p['contract'])
            artifact = Artifact(**p['artifact'])
            if (contract.id != p['contract_id'] or artifact.version != contract.artifact
                    or artifact.parent != contract.parent
                    or artifact != Artifact.create(artifact.content, artifact.parent, artifact.kind)):
                raise ValueError('Frozen contract/artifact identity mismatch')
            if contract.id in controller.contracts:
                raise ValueError('Repeated contract freeze in recovery journal')
            controller.contracts[contract.id] = contract
            controller.observations[contract.id] = {}
            artifacts[artifact.version] = artifact
        elif kind == 'OBSERVATION':
            c = controller.contracts[p['contract_id']]
            obs = Observation(**p['observation'])
            if (p['artifact'] != c.artifact or p['provenance'] != c.provenance
                    or obs.task not in c.tasks or obs.task in controller.observations[c.id]):
                raise ValueError('Observation identity or independent-unit mismatch')
            if digest({'contract_id':c.id, 'task':obs.task}) not in budget.attempts:
                raise ValueError('Observation lacks an owner-charged execution')
            controller.observations[c.id][obs.task] = obs
        elif kind in ('EVIDENCE_RESULT', 'TERMINATED_UNRESOLVED'):
            c = controller.contracts[p['contract_id']]
            controller.finished.add(c.id)
            controller.terminal_results[c.id] = (copy.deepcopy(p['result'])
                if kind == 'EVIDENCE_RESULT' else {'status':'TERMINATED_UNRESOLVED'})
    controller.artifacts = artifacts

    for c in controller.contracts.values():
        done = controller.observations[c.id]
        attempted = {u['task'] for u in budget.attempts.values() if u['contract_id'] == c.id}
        if not attempted <= set(c.tasks):
            raise ValueError('Charged execution outside the frozen sample plan')
        if c.id in controller.finished:
            budget.release(c.id)  # Also settles a crash during terminal bookkeeping.
            continue
        interrupted = attempted - set(done)
        if interrupted:
            ledger.append('TERMINATED_UNRESOLVED', contract_id=c.id,
                          reason='INTERRUPTED_ATTEMPT_WITHOUT_COMMITTED_OBSERVATION',
                          tasks=sorted(interrupted))
            controller.finished.add(c.id)
            controller.terminal_results[c.id] = {'status':'TERMINATED_UNRESOLVED'}
            budget.release(c.id)
        elif len(done) == len(c.tasks):
            # The fixed look was reached before a bookkeeping crash. It is
            # evaluated once without executing/charging any new statistical unit.
            result = controller.evaluate(c.id)
            ledger.append('EVIDENCE_RESULT', contract_id=c.id, result=result)
            controller.terminal_results[c.id] = copy.deepcopy(result)
            controller.finished.add(c.id)
            budget.release(c.id)
        else:
            if budget.locks.get(c.id) != (len(c.tasks) - len(done)) * c.cost_per_task:
                raise ValueError('Pending confirmation lost its exclusive frozen cost lock')
    # A crash between reservation and CONTRACT_FROZEN has no runnable obligation.
    # Release only unmistakable unbound confirmation locks; promotion stays owned.
    for owner in list(budget.locks):
        if len(owner) == 64 and owner not in controller.contracts:
            budget.release(owner)
            ledger.append('ORPHAN_CONFIRM_LOCK_RELEASED', owner=owner)
    return controller
