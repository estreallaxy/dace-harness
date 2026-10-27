"""Reducer tests with controlled closed-feedback fixtures, no benchmark scores."""
import copy
import unittest
from dataclasses import asdict
from unittest.mock import patch

from dace.agents.policy import SourcePolicy
from dace.core import Artifact,Contract
from dace.source_proposals import decode_proposal
from dace.source_state import SourceState
from tests.test_source_proposals import proposal


class NativeStateTests(unittest.TestCase):
    def setUp(self):
        self.parent = Artifact.create(SourcePolicy().content,kind='BASELINE')
        self.batch = decode_proposal(proposal(),self.parent)

    def proof(self,action='strip',**overrides):
        artifact = next(a.artifact for a in self.batch.actions if a.id==action)
        contract = Contract(artifact.version,self.parent.version,'scope',('task',),self.batch.predictions_for(action))
        claims = [dict(mechanism=p.mechanism,failure='CONSISTENT',boundary='CONSISTENT',
            mediator='CONSISTENT',contribution='POSITIVE',utility_lower=.2,bridge='SOURCE_REPAIR',
            artifact=artifact.version,scope='scope',action_repairs_source=p.action_repairs_source,
            **{}) for p in contract.predictions]
        for claim in claims:
            claim.update(overrides)
        cert = {'baseline':asdict(self.parent),'candidate':asdict(artifact),
            'ledger':'fixture.jsonl','sha256':'f'*64,'confirmation_tasks':['task']}
        return {'certificate':cert,'plan':{'contract':asdict(contract)},'feedback':{'claims':claims},
                'observations':[{'task':'task','before':0,'after':1}]}

    def observe(self,state,batch_id,**overrides):
        with patch('dace.source_state.confirmation_feedback',return_value=self.proof(**overrides)):
            return state.observe(batch_id,'strip','fixture.jsonl')

    def test_cap_preserves_archived_hypotheses_instead_of_clearing_history(self):
        state = SourceState(self.parent,cap=2)
        state.register(self.batch)
        payload = proposal()
        for m in payload['mechanisms']:
            m['explanation'] += ' new revision'
        state.register(decode_proposal(payload,self.parent))
        snapshot = state.snapshot()
        self.assertEqual(len(snapshot['records']),4)
        self.assertEqual(sum(r['status']=='ACTIVE' for r in snapshot['records'].values()),2)
        self.assertEqual(sum(r['status']=='ARCHIVED' for r in snapshot['records'].values()),2)

    def test_identical_proposal_is_idempotent(self):
        state = SourceState(self.parent)
        first = state.register(self.batch)
        before = state.snapshot()
        self.assertEqual(state.register(self.batch),first)
        self.assertEqual(state.snapshot(),before)

    def test_compatibility_alone_does_not_raise_mechanism_priority(self):
        state = SourceState(self.parent,cap=2)
        self.observe(state,state.register(self.batch))
        old = set(state.records)
        payload = proposal()
        for m in payload['mechanisms']:
            m['explanation'] += ' independently proposed revision'
        state.register(decode_proposal(payload,self.parent))
        self.assertTrue(all(state.records[key]['status']=='ARCHIVED' for key in old))

    def test_response_conflict_does_not_delete_source_compatibility(self):
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        self.observe(state,identity,boundary='CONTRADICTED',bridge='UNRESOLVED')
        own = next(r for r in state.records.values() if r['hypothesis']['id']=='format')
        self.assertEqual(own['source_views']['scope']['failure'],'CONSISTENT')
        self.assertEqual(own['status'],'ACTIVE')
        self.assertIn('REVISE_RESPONSE_BRANCH',[t['kind'] for t in state.targets])
        self.assertNotIn('REVISE_SOURCE_PREDICATE',[t['kind'] for t in state.targets])

    def test_compensation_preserves_failure_and_generates_remaining_source_question(self):
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        feedback = self.observe(state,identity,bridge='COMPENSATION')
        self.assertEqual(feedback['claims'][0]['failure'],'CONSISTENT')
        self.assertEqual(feedback['claims'][0]['bridge'],'COMPENSATION')
        self.assertIn('TEST_REMAINING_SOURCE_RISK',[t['kind'] for t in state.targets])
        self.assertTrue(all(t['status']=='PROPOSED' for t in state.targets))

    def test_masked_repair_is_retained_without_inventing_a_residual_fault(self):
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        feedback = self.observe(state,identity,contribution='UNRESOLVED')
        self.assertEqual(feedback['claims'][0]['bridge'],'SOURCE_REPAIR')
        self.assertIn('TEST_RESIDUAL_FAULT_OR_UNDERPOWERED_EFFECT',[t['kind'] for t in state.targets])
        self.assertTrue(all(t['causal_identification'] is False for t in state.targets))

    def test_uncoupled_dual_does_not_expose_bridge_or_joint_targets(self):
        state = SourceState(self.parent,level='C3')
        identity = state.register(self.batch)
        feedback = self.observe(state,identity,bridge='COMPENSATION')
        claim = feedback['claims'][0]
        self.assertEqual(claim['failure'],'CONSISTENT')
        self.assertEqual(claim['contribution'],'POSITIVE')
        self.assertEqual(claim['bridge'],'UNBRIDGED')
        self.assertNotIn('next_targets',claim)
        self.assertEqual(state.targets,[])

    def test_failure_only_hides_contribution_and_its_effect_estimate(self):
        state = SourceState(self.parent,level='C2')
        feedback = self.observe(state,state.register(self.batch))
        self.assertEqual(feedback['claims'][0]['failure'],'CONSISTENT')
        self.assertEqual(feedback['claims'][0]['contribution'],'UNRESOLVED')
        self.assertNotIn('utility_lower',feedback['claims'][0])

    def test_contribution_only_hides_failure_and_bridge(self):
        state = SourceState(self.parent,level='CONTRIBUTION_ONLY')
        feedback = self.observe(state,state.register(self.batch))
        self.assertEqual(feedback['claims'][0]['failure'],'UNRESOLVED')
        self.assertEqual(feedback['claims'][0]['contribution'],'POSITIVE')
        self.assertEqual(feedback['claims'][0]['bridge'],'UNBRIDGED')

    def test_foreign_action_does_not_overwrite_own_source_view(self):
        state = SourceState(self.parent)
        self.observe(state,state.register(self.batch))
        foreign = next(r for r in state.records.values() if r['hypothesis']['id']=='dispatch')
        self.assertEqual(foreign['source_views'],{})
        self.assertEqual(len(foreign['responses']),1)

    def test_evidence_cannot_be_rebound_to_another_candidate(self):
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        proof = self.proof()
        proof['certificate']['candidate']['version'] = 'different'
        with patch('dace.source_state.confirmation_feedback',return_value=proof):
            with self.assertRaisesRegex(ValueError,'Foreign artifact'):
                state.observe(identity,'strip','fixture.jsonl')

    def test_deployment_invalidates_old_parent_state_without_deleting_it(self):
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        self.observe(state,identity)
        state.deploy(self.batch.actions[0].artifact)
        self.assertTrue(all(r['status']=='STALE' for r in state.records.values()))
        self.assertTrue(all(t['status']=='STALE' for t in state.targets))
        with self.assertRaisesRegex(ValueError,'Stale'):
            self.observe(state,identity)

    def test_c5_uses_disagreement_while_c4_uses_fixed_action_order(self):
        state = SourceState(self.parent,level='C4')
        ranked = state.ranked_actions(state.register(self.batch))
        self.assertEqual([n for _,n in ranked],[0,0])
        full = SourceState(self.parent,level='C5')
        self.assertEqual([n for _,n in full.ranked_actions(full.register(self.batch))],[1,0])

    def test_c0_has_no_mechanism_records(self):
        state = SourceState(self.parent,level='C0')
        payload = proposal()
        payload['mechanisms'] = []
        batch = decode_proposal(payload,self.parent,mechanisms_required=False)
        identity = state.register(batch)
        self.assertEqual(state.records,{})
        self.assertEqual(len(state.ranked_actions(identity)),2)

    def test_snapshot_mutation_does_not_rewrite_controller_state(self):
        state = SourceState(self.parent)
        state.register(self.batch)
        snapshot = state.snapshot()
        snapshot['records'].clear()
        self.assertEqual(len(state.records),2)


if __name__=='__main__':
    unittest.main()
