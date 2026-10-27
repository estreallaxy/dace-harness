from tests.support import test_temp_dir
import tempfile
import unittest
from pathlib import Path
from dace.core import *


class DaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.ledger = Ledger(Path(self.tmp.name) / 'ledger.jsonl')
        self.budget = Budget(1000)
        self.exposure = Exposure(self.ledger)
        self.ctrl = Controller(self.ledger, self.budget, self.exposure)
        self.parent = Artifact.create('baseline')
        self.artifact = Artifact.create('patch', self.parent.version)
        self.p = Prediction('m', 'source', ('source','adapter'), (('source',False),),
                            (('source',True),), (('ok',True),), 'SOURCE_REPAIR')

    def contract(self, n=128, provenance='CONFIRMATORY'):
        return Contract(self.artifact.version, self.parent.version, 'scope',
                        tuple(f't{i}' for i in range(n)), (self.p,), provenance)

    def obs(self, task):
        return Observation(task, {'source':False}, {'source':True,'ok':True}, 0,1,True)

    def test_budget_partitions(self):
        b = Budget(20)
        b.reserve('promotion',10)
        b.reserve('confirm',8)
        with self.assertRaises(ValueError): b.charge(3)
        b.charge(5,'confirm')
        self.assertEqual((b.free,b.spent,b.locks['confirm']), (2,5,3))
        b.reserve('confirm',8)
        self.assertEqual(b.locks['confirm'],3)
        with self.assertRaises(ValueError): b.charge(4,'confirm')

    def test_fixed_look_and_idempotent_resume(self):
        c = self.contract()
        self.ctrl.freeze(self.artifact,c)
        r = self.ctrl.resume(c.id,self.obs,batch=5)
        self.assertEqual(r['claims'],[])
        self.assertEqual(self.budget.locks[c.id],246)
        r = self.ctrl.resume(c.id,self.obs)
        self.assertEqual(r['claims'][0]['bridge'],'SOURCE_REPAIR')
        spent = self.budget.spent
        self.ctrl.resume(c.id,self.obs)
        self.assertEqual(spent,self.budget.spent)

    def test_exploration_never_confirms(self):
        c = self.contract(provenance='EXPLORATORY')
        self.ctrl.freeze(self.artifact,c)
        r = self.ctrl.resume(c.id,self.obs)
        self.assertEqual(r,{'status':'EXPLORATORY_ONLY','claims':[]})

    def test_compensation_is_not_source_repair(self):
        self.p = Prediction('m','source',('source','adapter'),(('source',False),),
                            (('source',False),), (('ok',True),), 'COMPENSATION')
        c = self.contract()
        self.ctrl.freeze(self.artifact,c)
        r = self.ctrl.resume(c.id, lambda t: Observation(t,{'source':False},{'source':False,'ok':True},0,1,True))
        self.assertEqual(r['claims'][0]['bridge'],'COMPENSATION')

    def test_missing_hook_unresolved(self):
        c = self.contract()
        self.ctrl.freeze(self.artifact,c)
        r = self.ctrl.resume(c.id, lambda t: Observation(t,{'source':False},{'ok':True},0,1,None))
        self.assertEqual(r['claims'][0]['bridge'],'UNRESOLVED')
        self.assertEqual(r['minimal'],'MINIMAL_UNRESOLVED')

    def test_no_activation_no_contribution(self):
        c = self.contract()
        self.ctrl.freeze(self.artifact,c)
        r = self.ctrl.resume(c.id, lambda t: Observation(t,{'source':False},{'source':True,'ok':True},0,1,False))
        self.assertEqual(r['claims'][0]['contribution'],'UNRESOLVED')

    def test_no_activation_no_bridge(self):
        # review round-2 reproduction: the executor hook reports the edit was
        # never activated (activated=False) even though every frozen boundary
        # predicate matches -- NO bridge conclusion may be drawn from a diff
        # that never executed, so SOURCE_REPAIR stays UNRESOLVED
        c = self.contract()
        self.ctrl.freeze(self.artifact, c)
        r = self.ctrl.resume(c.id, lambda t: Observation(t, {'source': False},
                                                         {'source': True, 'ok': True}, 0, 1, False))
        self.assertEqual(r['claims'][0]['bridge'], 'UNRESOLVED')
        self.assertEqual(r['minimal'], 'MINIMAL_UNRESOLVED')

    def test_claim_carries_action_scope(self):
        # claims are action-scoped: a foreign-action (discrimination) claim
        # must be distinguishable from an own-action claim at consumption time
        c = self.contract()
        self.ctrl.freeze(self.artifact, c)
        r = self.ctrl.resume(c.id, self.obs)
        self.assertTrue(r['claims'][0]['action_repairs_source'])

    def test_no_confirmation_after_disclosure(self):
        c = self.contract(2)
        self.exposure.disclose(['t0'],['derived-summary'],self.artifact.version)
        with self.assertRaises(ValueError): self.ctrl.freeze(self.artifact,c)
        self.assertEqual(self.budget.spent,0)

    def test_no_contract_without_full_lock(self):
        c = self.contract(600)
        with self.assertRaises(ValueError): self.ctrl.freeze(self.artifact,c)
        self.assertEqual(self.ctrl.contracts,{})

    def test_exact_artifact_version(self):
        c = self.contract(2)
        other = Artifact.create('different', self.parent.version)
        with self.assertRaises(ValueError): self.ctrl.freeze(other,c)

    def test_no_promotion_with_missing_family(self):
        candidates={'v':{'pairs':[(str(i),'a',0,1) for i in range(128)],'cost':1,'size':1}}
        self.assertIsNone(promotion(candidates,['a','missing'])['winner'])

    def test_duplicate_units_rejected(self):
        candidates={'v':{'pairs':[('same','a',0,1)]*128,'cost':1,'size':1}}
        with self.assertRaises(ValueError): promotion(candidates,['a'])

    def test_ledger_integrity(self):
        self.ledger.append('X',test=True)
        path=self.ledger.path
        path.write_text(path.read_text().replace('true','false'))
        with self.assertRaises(ValueError): Ledger(path)

    def test_failures_charged_and_terminate(self):
        c=self.contract(2)
        self.ctrl.freeze(self.artifact,c)
        def fail(t): raise RuntimeError('execution failure')
        self.assertEqual(self.ctrl.resume(c.id,fail)['status'],'TERMINATED_UNRESOLVED')
        self.assertEqual(self.budget.spent,2)
        self.assertNotIn(c.id,self.budget.locks)

if __name__ == '__main__': unittest.main()
