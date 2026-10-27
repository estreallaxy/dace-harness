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

    def test_same_named_predictions_for_different_edits_are_distinct_revisions(self):
        state = SourceState(self.parent,cap=4)
        original = state.register(self.batch)
        self.observe(state,original)
        payload = proposal()
        payload['actions'][0]['edits']['response_code'] = 'def transform(value):\n    return value.lstrip()\n'
        changed = state.register(decode_proposal(payload,self.parent))
        self.assertFalse(set(state.batches[original]['mechanisms'].values()) &
                         set(state.batches[changed]['mechanisms'].values()))
        for revision in state.batches[changed]['mechanisms'].values():
            self.assertEqual(state.records[revision]['responses'],{})
            self.assertEqual(state.records[revision]['source_views'],{})
        self.assertEqual(len(state.records),4)

    def test_active_prior_batch_remains_schedulable_when_new_round_has_no_proposal(self):
        state = SourceState(self.parent)
        original = state.register(self.batch)
        empty = state.register(decode_proposal({'actions':[],'mechanisms':[]},self.parent))
        scheduled = state.scheduled_actions(empty)
        self.assertEqual([(b,a.id,n) for b,a,n in scheduled],[(original,'strip',1),(original,'suffix',0)])
        self.assertEqual(state.batches[scheduled[0][0]]['batch'].predictions_for('strip'),
                         self.batch.predictions_for('strip'))

    def test_global_scheduler_deduplicates_artifacts_without_merging_hypotheses(self):
        state = SourceState(self.parent,cap=4)
        original = state.register(self.batch)
        payload = proposal()
        for mechanism in payload['mechanisms']:
            mechanism['explanation'] += ' alternative hypothesis revision'
        latest = state.register(decode_proposal(payload,self.parent))
        scheduled = state.scheduled_actions(latest)
        self.assertEqual(len(scheduled),2)
        self.assertEqual({b for b,_,_ in scheduled},{latest})
        self.assertNotEqual(state.batches[original]['mechanisms'],state.batches[latest]['mechanisms'])

    def test_archived_and_stale_proposals_cannot_consume_new_interventions(self):
        state = SourceState(self.parent,cap=2)
        state.register(self.batch)
        payload = proposal()
        for mechanism in payload['mechanisms']:
            mechanism['explanation'] += ' replacement'
        latest = state.register(decode_proposal(payload,self.parent))
        self.assertEqual({b for b,_,_ in state.scheduled_actions(latest)},{latest})
        state.deploy(self.batch.actions[0].artifact)
        with self.assertRaisesRegex(ValueError,'Stale'):
            state.scheduled_actions(latest)

    def test_c0_never_retains_actions_through_a_mechanism_pool(self):
        state = SourceState(self.parent,level='C0')
        payload = proposal()
        payload['mechanisms'] = []
        state.register(decode_proposal(payload,self.parent,mechanisms_required=False))
        empty = state.register(decode_proposal({'actions':[],'mechanisms':[]},self.parent,mechanisms_required=False))
        self.assertEqual(state.scheduled_actions(empty),[])

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

    def mediator_proof(self):
        payload = proposal()
        payload['mechanisms'][1]['outcomes']['strip']['mediator']['command_batch_valid'] = False
        self.batch = decode_proposal(payload,self.parent)
        proof = self.proof()
        for claim in proof['feedback']['claims']:
            status = 'CONSISTENT' if claim['mechanism']=='format' else 'CONTRADICTED'
            claim['predicate_evidence'] = {'mediator':{'predicates':{
                'command_batch_valid':{'status':status}}}}
        return proof

    def test_c5_retains_discrimination_in_frozen_mediator_predictions(self):
        proof = self.mediator_proof()
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        with patch('dace.source_state.confirmation_feedback',return_value=proof):
            state.observe(identity,'strip','fixture.jsonl')
        rows = [row for record in state.records.values() for row in record['discriminations']]
        self.assertEqual(len(rows),2)
        self.assertEqual({row['family'] for row in rows},{'mediator'})
        self.assertEqual({row['field'] for row in rows},{'command_batch_valid'})
        self.assertEqual({row['status'] for row in rows},{'PREDICTION_SUPPORTED','PREDICTION_CONFLICT'})

    def test_discrimination_cannot_borrow_status_from_another_predicate_family(self):
        proof = self.mediator_proof()
        for claim in proof['feedback']['claims']:
            claim['predicate_evidence'] = {'boundary':claim['predicate_evidence']['mediator']}
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        with patch('dace.source_state.confirmation_feedback',return_value=proof):
            state.observe(identity,'strip','fixture.jsonl')
        self.assertTrue(all(not record['discriminations'] for record in state.records.values()))

    def test_discriminator_field_order_is_stable_for_checkpoint_reconstruction(self):
        payload = proposal()
        for mechanism in payload['mechanisms']:
            value = mechanism['id']=='format'
            mechanism['outcomes']['strip']['mediator'] = {
                'command_count_preserved':value,'command_batch_valid':value}
        self.batch = decode_proposal(payload,self.parent)
        proof = self.proof()
        for claim in proof['feedback']['claims']:
            status = 'CONSISTENT' if claim['mechanism']=='format' else 'CONTRADICTED'
            claim['predicate_evidence'] = {'mediator':{'predicates':{
                field:{'status':status} for field in ('command_count_preserved','command_batch_valid')}}}
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        with patch('dace.source_state.confirmation_feedback',return_value=proof):
            state.observe(identity,'strip','fixture.jsonl')
        for record in state.records.values():
            self.assertEqual([row['field'] for row in record['discriminations']],
                             ['command_batch_valid','command_count_preserved'])

    def test_factorial_effect_and_conflicting_failure_drive_a_joint_revision_question(self):
        payload = proposal()
        payload['actions'][0]['edits']['instruction_suffix'] = 'Controlled joint edit.'
        self.batch = decode_proposal(payload,self.parent)
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        self.observe(state,identity,failure='CONTRADICTED',bridge='UNRESOLVED')
        action = next(a for a in self.batch.actions if a.id=='strip')
        proof = {'plan':{'baseline':asdict(self.parent)},'candidate':asdict(action.artifact),
            'interaction':{'interaction_status':'POSITIVE','activation_complete':True},
            'source_tasks':['factorial-task'],'scope':'scope','ledger':'fixture-factorial.jsonl','sha256':'a'*64}
        with patch('dace.source_factorial.factorial_feedback',return_value=proof):
            state.observe_factorial(identity,'strip','fixture-factorial.jsonl')
            before = state.snapshot()
            state.observe_factorial(identity,'strip','fixture-factorial.jsonl')
            self.assertEqual(state.snapshot(),before)
        target = next(t for t in state.targets if t['kind']=='REVISE_SOURCE_FOR_FACTORIAL_EFFECT')
        self.assertEqual(target['failure_status'],'CONTRADICTED')
        self.assertEqual(target['interaction_status'],'POSITIVE')
        self.assertFalse(target['causal_identification'])
        state.deploy(action.artifact)
        self.assertTrue(all(entry['parent_status']=='STALE' for entry in state.interactions.values()))

    def test_factorial_response_cannot_be_transferred_to_a_different_action_artifact(self):
        state = SourceState(self.parent)
        identity = state.register(self.batch)
        proof = {'plan':{'baseline':asdict(self.parent)},'candidate':asdict(self.batch.actions[1].artifact)}
        with patch('dace.source_factorial.factorial_feedback',return_value=proof):
            with self.assertRaisesRegex(ValueError,'another exact parent or action'):
                state.observe_factorial(identity,'strip','fixture.jsonl')
        self.assertEqual(state.interactions,{})


if __name__=='__main__':
    unittest.main()
