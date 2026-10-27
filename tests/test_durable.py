from tests.support import test_temp_dir
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from dace.core import Artifact, Budget, Contract, Controller, Exposure, Ledger, Observation, Prediction
from dace.durable import PersistentBudget


class DurableConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'evidence.jsonl'
        self.artifact = Artifact.create('patch', 'parent')
        self.prediction = Prediction('m', 'valid', ('planner', 'adapter'),
            (('valid',False),), (('valid',True),), (('mediated',True),), 'SOURCE_REPAIR')

    def create(self, total=1000):
        ledger = Ledger(self.path, durable=True)
        self.addCleanup(ledger.close)
        budget = PersistentBudget(total, ledger)
        return Controller(ledger, budget, Exposure(ledger))

    def recover(self, total=1000):
        controller = Controller.restore(self.path, total)
        self.addCleanup(controller.ledger.close)
        return controller

    def contract(self, n=6):
        return Contract(self.artifact.version, self.artifact.parent, 'global',
                        tuple(f'task-{i}' for i in range(n)), (self.prediction,))

    @staticmethod
    def obs(task):
        return Observation(task, {'valid':False}, {'valid':True,'mediated':True}, 0,1,True)

    def test_restart_resumes_exact_remaining_units_and_original_budget_lock(self):
        c = self.contract()
        first = self.create()
        first.budget.reserve('promotion', 300)
        first.budget.charge(20)
        first.freeze(self.artifact, c)
        initial = asdict(c)
        self.assertEqual(first.resume(c.id, self.obs, batch=2)['status'], 'EVIDENCE_INCOMPLETE')
        first.ledger.close()
        second = self.recover()
        self.assertEqual(asdict(second.contracts[c.id]), initial)
        self.assertEqual((second.budget.spent, second.budget.locks[c.id]), (24,8))
        seen = []
        def execute(task):
            seen.append(task)
            return self.obs(task)
        result = second.resume(c.id, execute)
        self.assertEqual(seen, list(c.tasks[2:]))
        self.assertEqual(result['status'], 'EVIDENCE_COMPLETE')
        self.assertEqual(second.budget.spent, 32)
        self.assertEqual(second.budget.locks, {'promotion':300})
        self.assertEqual(second.budget.free, 668)
        second.resume(c.id, lambda t: self.fail('Completed contract executed twice'))
        self.assertEqual(second.budget.spent, 32)
        self.assertEqual(sum(e['kind']=='CONTRACT_FROZEN' for e in second.ledger.events), 1)

    def test_locked_obligation_runs_when_unreserved_budget_is_zero(self):
        c = self.contract()
        first = self.create(total=12)
        first.freeze(self.artifact,c)
        first.resume(c.id,self.obs,batch=1)
        self.assertEqual(first.budget.free, 0)
        first.ledger.close()
        second = self.recover(total=12)
        self.assertEqual(second.budget.free, 0)
        self.assertEqual(second.resume(c.id,self.obs)['status'], 'EVIDENCE_COMPLETE')
        self.assertEqual(second.budget.spent, 12)

    def test_interrupted_attempt_terminates_without_sample_replacement(self):
        c = self.contract()
        first = self.create()
        first.freeze(self.artifact,c)
        def interrupted(task):
            raise KeyboardInterrupt('simulated process stop after charge')
        with self.assertRaises(KeyboardInterrupt):
            first.resume(c.id,interrupted)
        first.ledger.close()
        second = self.recover()
        self.assertEqual(second.budget.spent, 2)
        self.assertNotIn(c.id,second.budget.locks)
        result = second.resume(c.id,lambda t:self.fail('Ambiguous unit rerun after crash'))
        self.assertEqual(result, {'status':'TERMINATED_UNRESOLVED'})
        event = next(e for e in second.ledger.events if e['kind']=='TERMINATED_UNRESOLVED')
        self.assertEqual(event['payload']['tasks'], ['task-0'])

    def test_fixed_look_committed_before_bookkeeping_crash_is_evaluated_once(self):
        c = self.contract()
        first = self.create()
        first.freeze(self.artifact,c)
        append = first.ledger.append
        def crash(kind,**payload):
            if kind == 'EVIDENCE_RESULT':
                raise KeyboardInterrupt('fixed look reached, result not yet committed')
            return append(kind,**payload)
        with patch.object(first.ledger,'append',side_effect=crash):
            with self.assertRaises(KeyboardInterrupt):
                first.resume(c.id,self.obs)
        first.ledger.close()
        second = self.recover()
        self.assertEqual(second.budget.spent, 12)
        self.assertEqual(second.resume(c.id,lambda t:self.fail('Frozen look rerun'))['status'], 'EVIDENCE_COMPLETE')
        self.assertEqual(sum(e['kind']=='EVIDENCE_RESULT' for e in second.ledger.events),1)
        second.ledger.close()
        third = self.recover()
        self.assertEqual(sum(e['kind']=='EVIDENCE_RESULT' for e in third.ledger.events),1)

    def test_second_durable_writer_is_rejected(self):
        self.create()
        with self.assertRaisesRegex(RuntimeError,'LEDGER_ALREADY_OWNED'):
            Ledger(self.path,durable=True)

    def test_frozen_total_cannot_be_changed_on_restart(self):
        first = self.create()
        first.ledger.close()
        with self.assertRaises(ValueError):
            self.recover(total=1001)
        # A failed restore must not retain its OS writer lease via traceback.
        correct = self.recover(total=1000)
        self.assertEqual(correct.budget.total,1000)

    def test_legacy_memory_only_budget_is_not_invented_during_recovery(self):
        ledger = Ledger(self.path)
        ledger.append('CONTRACT_FROZEN', incomplete_legacy_event=True)
        with self.assertRaisesRegex(ValueError,'LEGACY_BUDGET_NOT_RECOVERABLE'):
            self.recover()

    def test_ledger_snapshots_mutable_payloads(self):
        ledger = Ledger(self.path)
        config = {'repair':False}
        returned = ledger.append('CONFIG',config=config)
        config['repair'] = True
        returned['payload']['config']['repair'] = True
        self.assertFalse(ledger.events[0]['payload']['config']['repair'])
        self.assertEqual(ledger.events, Ledger(self.path).events)

    def test_nonfinite_budget_cannot_corrupt_owner_partitions(self):
        for value in (float('nan'),float('inf'),-1,True):
            with self.assertRaises(ValueError):
                Budget(value)
        budget = Budget(10)
        for method in (lambda:budget.reserve('x',float('nan')),
                       lambda:budget.charge(float('inf'))):
            with self.assertRaises(ValueError):
                method()
        self.assertEqual((budget.free,budget.spent,budget.locks),(10,0,{}))

    def test_real_process_restart_replays_committed_batch(self):
        # Separate interpreter: proves recovery does not depend on Python objects
        # surviving in this test process or on an in-memory task map.
        script = '''
import sys
from dace.core import *
from dace.durable import PersistentBudget
l=Ledger(sys.argv[1],durable=True)
b=PersistentBudget(100,l)
c=Controller(l,b,Exposure(l))
a=Artifact.create('patch','parent')
p=Prediction('m','valid',('planner',),(('valid',False),),(('valid',True),),(('mediated',True),),'SOURCE_REPAIR')
k=Contract(a.version,a.parent,'global',tuple('task-'+str(i) for i in range(6)),(p,))
c.freeze(a,k)
c.resume(k.id,lambda t:Observation(t,{'valid':False},{'valid':True,'mediated':True},0,1,True),batch=2)
'''
        proc = subprocess.run([sys.executable,'-c',script,str(self.path)],
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(proc.returncode,0,proc.stderr)
        second = self.recover(total=100)
        c = next(iter(second.contracts.values()))
        self.assertEqual(second.budget.spent,4)
        second.resume(c.id,self.obs)
        self.assertEqual(second.budget.spent,12)


if __name__ == '__main__':
    unittest.main()
