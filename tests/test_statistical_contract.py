from tests.support import test_temp_dir
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from dace.core import Artifact,Budget,Contract,Controller,Exposure,Ledger,Observation,Prediction,StatisticalContract
from dace.durable import PersistentBudget,contract_from_dict
from dace.predicate_evidence import PredicateProtocol


class StatisticalContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'evidence.jsonl'
        self.parent = Artifact.create('parent',kind='BASELINE')
        self.candidate = Artifact.create('candidate',self.parent.version)
        self.prediction = Prediction('m','valid',('source','sink'),(('valid',False),),
                                    (('valid',True),),(('mediator',True),),'SOURCE_REPAIR')

    def contract(self,n=1):
        return StatisticalContract(self.candidate.version,self.parent.version,'frozen-scope',
            tuple(str(i) for i in range(n)),(self.prediction,),predicate_protocol=PredicateProtocol(.1))

    def evaluate(self,n=1,missing=False):
        c = self.contract(n)
        ledger = Ledger(self.path)
        controller = Controller(ledger,Budget(2*n),Exposure(ledger))
        controller.contracts[c.id] = c
        controller.observations[c.id] = {t:Observation(t,{'valid':False},
            {'valid':True,**({} if missing and t=='0' else {'mediator':True})},0,1,True) for t in c.tasks}
        return controller.evaluate(c.id)

    def test_single_perfect_pair_keeps_random_claims_and_bridge_unresolved(self):
        result = self.evaluate()
        claim = result['claims'][0]
        self.assertEqual(result['minimal'],'MINIMAL_OBSERVED')
        self.assertEqual(result['status'],'EVIDENCE_INCOMPLETE')
        self.assertEqual((claim['failure'],claim['boundary'],claim['mediator']),('UNRESOLVED',)*3)
        self.assertEqual(claim['bridge'],'UNRESOLVED')
        self.assertFalse(claim['causal_identification'])

    def test_sufficient_fixed_sample_supports_contract_not_unique_root(self):
        result = self.evaluate(1000)
        claim = result['claims'][0]
        self.assertEqual(result['status'],'EVIDENCE_COMPLETE')
        self.assertEqual(claim['bridge'],'SOURCE_REPAIR')
        self.assertEqual(claim['contribution'],'POSITIVE')
        self.assertFalse(claim['causal_identification'])

    def test_three_predicate_families_and_utility_share_alpha(self):
        claim = self.evaluate(1000)['claims'][0]
        allocations = [v['alpha'] for v in claim['predicate_evidence'].values()]
        self.assertAlmostEqual(sum(allocations)+claim['utility_alpha'],.05)

    def test_missing_mediator_retains_its_own_unresolved_predicate(self):
        claim = self.evaluate(1000,missing=True)['claims'][0]
        self.assertEqual(claim['failure'],'CONSISTENT')
        self.assertEqual(claim['boundary'],'CONSISTENT')
        self.assertEqual(claim['mediator'],'UNRESOLVED')
        self.assertEqual(claim['bridge'],'UNRESOLVED')
        self.assertEqual(claim['predicate_evidence']['mediator']['predicates']['mediator']['n_missing'],1)

    def test_protocol_changes_contract_id_and_cannot_reuse_legacy_claim(self):
        c = self.contract()
        legacy = Contract(c.artifact,c.parent,c.scope,c.tasks,c.predictions)
        self.assertNotEqual(c.id,legacy.id)
        self.assertNotIn('predicate_protocol',asdict(legacy))
        self.assertEqual(contract_from_dict(json.loads(json.dumps(asdict(legacy)))).id,legacy.id)

    def test_statistical_contract_roundtrips_exactly_through_json(self):
        c = self.contract()
        restored = contract_from_dict(json.loads(json.dumps(asdict(c))))
        self.assertIsInstance(restored,StatisticalContract)
        self.assertEqual(restored,c)
        self.assertEqual(restored.id,c.id)

    def test_durable_recovery_preserves_protocol_and_does_not_reexecute(self):
        c = self.contract()
        ledger = Ledger(self.path,durable=True)
        controller = Controller(ledger,PersistentBudget(2,ledger),Exposure(ledger))
        controller.freeze(self.candidate,c)
        result = controller.resume(c.id,lambda t:Observation(t,{'valid':False},
            {'valid':True,'mediator':True},0,1,True))
        ledger.close()
        before = self.path.read_bytes()
        recovered = Controller.restore(self.path,2)
        try:
            self.assertEqual(recovered.contracts[c.id],c)
            def forbidden(task):
                self.fail('Completed statistical unit was replayed')
            self.assertEqual(recovered.resume(c.id,forbidden),result)
        finally:
            recovered.ledger.close()
        self.assertEqual(self.path.read_bytes(),before)


class NativeStatisticalConfirmationTests(unittest.TestCase):
    def setUp(self):
        from tests.test_source_selection import SourceSelectionTests
        self.fixture = SourceSelectionTests(methodName='runTest')
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def make(self):
        from dace.agents.policy import SourcePolicy
        from dace.source_confirmation import make_plan
        f = self.fixture
        return make_plan(SourcePolicy(),SourcePolicy(instruction_suffix='statistical-repair'),
            [f.prediction],['confirm-b'],f.dataset,f.command,f.model,study=f.study,
            runtime_lock=f.runtime,predicate_protocol=PredicateProtocol(.1))

    def test_native_v2_keeps_single_pair_unresolved_but_records_actual_activation(self):
        from dace.source_confirmation import run_plan
        from dace.source_selection import confirmation_certificate
        f,plan = self.fixture,self.make()
        f.execution_output = f.root/'statistical-confirmation'
        result = run_plan(plan,f.execution_output)
        self.assertEqual(plan['schema'],'dace-source-confirmation/v2')
        self.assertEqual(result['evidence']['minimal'],'MINIMAL_OBSERVED')
        self.assertEqual(result['evidence']['claims'][0]['bridge'],'UNRESOLVED')
        certificate = confirmation_certificate(f.execution_output/'evidence.jsonl')
        self.assertFalse(certificate['attribution'][0]['causal_identification'])

    def test_new_protocol_cannot_be_relabelled_as_legacy(self):
        from dace.source_confirmation import validate_plan
        plan = self.make()
        plan['schema'] = 'dace-source-confirmation/v1'
        with self.assertRaisesRegex(ValueError,'schema differ'):
            validate_plan(plan)


if __name__=='__main__':
    unittest.main()
