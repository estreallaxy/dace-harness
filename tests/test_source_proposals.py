from tests.support import test_temp_dir
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from dace.agents.policy import SourcePolicy
from dace.core import Artifact, Budget, Ledger
from dace.source_proposals import SourceCodingAgent,decode_proposal,source_graph
from dace.task_registry import TaskRegistry
from dace.benchmarks.fingerprint import task_fingerprint


def proposal():
    return {'actions':[
        {'id':'strip','edits':{'response_code':'def transform(value):\n    return value.strip()\n'}},
        {'id':'suffix','edits':{'instruction_suffix':'Use noninteractive commands.'}}],
        'mechanisms':[
            {'id':'format','component':'response_transform','source':'response_parse_valid',
             'propagation':['response_transform','parser','command_transform','terminal'],
             'baseline':{'response_parse_valid':False},'own_action':'strip',
             'outcomes':{
                 'strip':{'intervention':{'response_parse_valid':True},'mediator':{'command_batch_valid':True},'bridge':'SOURCE_REPAIR'},
                 'suffix':{'intervention':{'response_parse_valid':False},'mediator':{'command_batch_valid':True},'bridge':'UNRESOLVED'}},
             'explanation':'Response conversion may violate the parser contract.'},
            {'id':'dispatch','component':'terminal','source':'terminal_dispatch_completed',
             'propagation':['terminal'],'baseline':{'terminal_dispatch_completed':False},'own_action':'suffix',
             'outcomes':{
                 'strip':{'intervention':{'terminal_dispatch_completed':False,'response_parse_valid':False},'mediator':{'command_batch_valid':True},'bridge':'UNRESOLVED'},
                 'suffix':{'intervention':{'terminal_dispatch_completed':True,'response_parse_valid':False},'mediator':{'command_batch_valid':True},'bridge':'SOURCE_REPAIR'}},
             'explanation':'An interactive terminal command may prevent completion.'}]}


class SourceProposalTests(unittest.TestCase):
    def setUp(self):
        self.parent = Artifact.create(SourcePolicy().content,kind='BASELINE')

    def test_source_graph_binds_real_symbols_and_exact_policy_version(self):
        graph = source_graph(self.parent)
        self.assertEqual(graph['artifact'],self.parent.version)
        self.assertEqual(len(graph['integration_sha256']),64)
        self.assertGreater(graph['slots']['response_code']['source_line'],0)
        changed = Artifact.create(SourcePolicy(instruction_suffix='changed').content,self.parent.version)
        self.assertNotEqual(source_graph(changed)['artifact'],graph['artifact'])

    def test_proposals_compile_actual_code_edits_and_foreign_predictions(self):
        batch = decode_proposal(proposal(),self.parent)
        self.assertEqual(batch.parent,self.parent.version)
        strip = next(a for a in batch.actions if a.id=='strip')
        self.assertEqual(strip.artifact.parent,self.parent.version)
        self.assertIn('value.strip()',strip.artifact.content)
        predictions = batch.predictions_for('strip')
        self.assertEqual([p.action_repairs_source for p in predictions],[True,False])
        self.assertFalse(hasattr(batch.mechanisms[0],'confirmed'))

    def test_disagreement_scheduler_uses_frozen_predictions_only(self):
        batch = decode_proposal(proposal(),self.parent)
        ranked = batch.ranked_actions()
        self.assertEqual([(a.id,n) for a,n in ranked],[('strip',1),('suffix',0)])

    def test_equivalent_edits_are_reused_instead_of_run_twice(self):
        payload = proposal()
        payload['actions'].append({'id':'strip-copy','edits':copy.deepcopy(payload['actions'][0]['edits'])})
        for m in payload['mechanisms']:
            m['outcomes']['strip-copy'] = copy.deepcopy(m['outcomes']['strip'])
        batch = decode_proposal(payload,self.parent)
        self.assertEqual(len(batch.actions),2)
        self.assertEqual(batch.aliases,(('strip-copy','strip'),))

    def test_equivalent_edits_cannot_have_contradictory_same_mechanism_predictions(self):
        payload = proposal()
        payload['actions'].append({'id':'copy','edits':copy.deepcopy(payload['actions'][0]['edits'])})
        for m in payload['mechanisms']:
            m['outcomes']['copy'] = copy.deepcopy(m['outcomes']['strip'])
        payload['mechanisms'][0]['outcomes']['copy']['intervention']['response_parse_valid'] = False
        with self.assertRaisesRegex(ValueError,'Equivalent edits'):
            decode_proposal(payload,self.parent)

    def test_unknown_or_semantic_truth_fields_are_not_observations(self):
        payload = proposal()
        payload['mechanisms'][0]['baseline'] = {'root_cause_correct':False}
        with self.assertRaisesRegex(ValueError,'measured boolean'):
            decode_proposal(payload,self.parent)

    def test_code_slots_are_not_measured_source_or_intervention_fields(self):
        payload = proposal()
        payload['mechanisms'][0]['source'] = 'response_code'
        with self.assertRaisesRegex(ValueError,'not anchored'):
            decode_proposal(payload,self.parent)
        payload = proposal()
        payload['mechanisms'][0]['outcomes']['strip']['intervention'] = {'response_code':True}
        with self.assertRaisesRegex(ValueError,'measured boolean'):
            decode_proposal(payload,self.parent)

    def test_unfrozen_caps_and_unanchored_propagation_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'schema/caps'):
            decode_proposal(proposal(),self.parent,mechanism_cap=1)
        payload = proposal()
        payload['mechanisms'][0]['propagation'] = ['terminal','parser']
        with self.assertRaisesRegex(ValueError,'not anchored'):
            decode_proposal(payload,self.parent)

    def test_parser_defect_can_compete_with_model_format_explanation(self):
        payload = proposal()
        m = payload['mechanisms'][0]
        m['component'] = 'parser'
        m['source'] = 'raw_response_parse_valid'
        m['propagation'] = ['parser','command_transform','terminal']
        m['baseline'] = {'raw_response_parse_valid':False}
        for outcome in m['outcomes'].values():
            outcome['intervention']['raw_response_parse_valid'] = False
            outcome['bridge'] = 'COMPENSATION'
        batch = decode_proposal(payload,self.parent)
        self.assertEqual(batch.mechanisms[0].component,'parser')

    def test_open_files_cannot_be_smuggled_into_source_policy(self):
        payload = proposal()
        payload['actions'][0]['edits']['response_code'] = 'def transform(value):\n    return open("solution")\n'
        with self.assertRaises(ValueError):
            decode_proposal(payload,self.parent)

    def test_incomplete_competing_predictions_and_extra_claims_are_rejected(self):
        payload = proposal()
        del payload['mechanisms'][0]['outcomes']['suffix']
        with self.assertRaisesRegex(ValueError,'Every action'):
            decode_proposal(payload,self.parent)
        payload = proposal()
        payload['mechanisms'][0]['confirmed'] = True
        with self.assertRaisesRegex(ValueError,'hypotheses'):
            decode_proposal(payload,self.parent)

    def test_no_evidence_can_return_an_explicit_no_change_batch(self):
        batch = decode_proposal({'actions':[],'mechanisms':[]},self.parent)
        self.assertEqual(batch.actions,())
        self.assertEqual(batch.ranked_actions(),[])


class CodingRequestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = Artifact.create(SourcePolicy().content,kind='BASELINE')
        self.ledger = Ledger(self.root/'evidence.jsonl')
        self.budget = Budget(100)
        self.budget.reserve('coding',10)
        self.llm = Mock(model='controlled-model-fixture')
        self.llm.complete.return_value = (json.dumps(proposal()),{'prompt_tokens':10,'completion_tokens':20,'cost':.001})
        self.agent = SourceCodingAgent(self.llm)
        self.disclose = Mock()
        self.diagnostics = [{'task_id':'diag','boundary':{'response_parse_valid':False},'utility':0}]

    def request(self,history=None):
        return self.agent.propose(self.parent,self.diagnostics,history or [],
            mechanism_cap=4,action_cap=4,budget=self.budget,owner='coding',
            ledger=self.ledger,disclose=self.disclose)

    def test_one_request_is_charged_disclosed_and_only_proposed(self):
        batch = self.request()
        self.assertEqual(self.budget.spent,10)
        self.llm.complete.assert_called_once()
        self.disclose.assert_called_once()
        self.assertEqual(self.disclose.call_args.args[0],['diag'])
        self.assertEqual(len(batch.actions),2)
        self.assertEqual(self.ledger.events[-1]['kind'],'SOURCE_AGENT_PROPOSAL')
        self.assertEqual(self.ledger.events[-1]['payload']['status'],'PROPOSED')

    def test_bad_response_keeps_usage_and_never_silently_retries(self):
        self.llm.complete.return_value = ('not json',{'cost':.01})
        with self.assertRaises(ValueError):
            self.request()
        self.assertEqual(self.budget.spent,10)
        self.llm.complete.assert_called_once()
        responses = [e for e in self.ledger.events if e['kind']=='SOURCE_AGENT_RESPONSE']
        self.assertEqual(responses[0]['payload']['usage']['cost'],.01)
        self.assertEqual(self.ledger.events[-1]['kind'],'SOURCE_AGENT_FAILED')

    def test_length_limited_response_is_rejected_even_if_its_json_is_parseable(self):
        self.llm.complete.return_value = (json.dumps(proposal()),
            {'cost':.01,'_completion':{'finish_reason':'length'}})
        from dace.coding_request import IncompleteCodingResponse
        with self.assertRaises(IncompleteCodingResponse):
            self.request()
        self.llm.complete.assert_called_once()
        self.assertEqual(self.budget.spent,10)
        self.assertFalse(any(e['kind']=='SOURCE_AGENT_PROPOSAL' for e in self.ledger.events))
        response = next(e['payload'] for e in self.ledger.events if e['kind']=='SOURCE_AGENT_RESPONSE')
        self.assertEqual(response['usage']['cost'],.01)

    def test_invalid_request_controls_are_rejected_before_charge_or_disclosure(self):
        with self.assertRaises(ValueError):
            self.agent.propose(self.parent,self.diagnostics,[],mechanism_cap=4,action_cap=4,
                budget=self.budget,owner='coding',ledger=self.ledger,disclose=self.disclose,
                coding_request={'max_tokens':999999})
        self.assertEqual(self.budget.spent,0)
        self.llm.complete.assert_not_called()
        self.disclose.assert_not_called()

    def test_native_state_task_sources_are_disclosed_before_any_model_request(self):
        def guard(tasks,fields):
            if 'held-out' in tasks:
                raise ValueError('Held-out evaluation evidence cannot reach the optimizer')
        with self.assertRaisesRegex(ValueError,'Held-out'):
            self.agent.propose(self.parent,self.diagnostics,[],mechanism_cap=4,action_cap=4,
                budget=self.budget,owner='coding',ledger=self.ledger,disclose=guard,level='C5',
                state_context={'records':{'revision':{'source_tasks':['held-out']}}})
        self.llm.complete.assert_not_called()

    def test_null_text_response_keeps_its_actual_provider_price(self):
        self.llm.complete.return_value = (None,{'cost':.02,'prompt_tokens':3,'completion_tokens':6000,
                                               '_completion':{'finish_reason':'length'}})
        with self.assertRaisesRegex(ValueError,'no bounded text'):
            self.request()
        response = next(e['payload'] for e in self.ledger.events if e['kind']=='SOURCE_AGENT_RESPONSE')
        self.assertEqual(response['usage']['cost'],.02)
        self.assertEqual(response['content_status'],'NON_TEXT')
        self.llm.complete.assert_called_once()

    def test_transport_exception_text_does_not_enter_error_events(self):
        self.llm.complete.side_effect = RuntimeError('sensitive transport body')
        with self.assertRaises(RuntimeError):
            self.request()
        self.assertNotIn('sensitive',self.ledger.path.read_text(encoding='utf-8'))

    def test_history_cannot_drop_its_source_tasks_or_change_claim_scope(self):
        with self.assertRaisesRegex(ValueError,'Feedback must retain'):
            self.request([{'source_tasks':[],'claims':[],'artifact':'a','scope':'s'}])
        self.llm.complete.assert_not_called()
        self.assertEqual(self.budget.spent,0)

    def test_eval_feedback_cannot_hide_inside_history(self):
        dataset = self.root/'dataset'
        for name in ('diag','held-out'):
            path = dataset/name
            path.mkdir(parents=True)
            (path/'task.toml').write_text('schema_version="1.1"',encoding='utf-8')
            (path/'instruction.md').write_text(name,encoding='utf-8')
        registry_path = self.root/'registry.sqlite3'
        with TaskRegistry(registry_path) as registry:
            registry.create_study('s',dataset,{'DIAG':['diag'],'EVAL':['held-out']})
            registry.create_lineage('s','run')
            for task,role in [('diag','DIAG'),('held-out','EVAL')]:
                registry.claim('s','run',task,role,[task],{task:task_fingerprint(dataset/task)},
                               str(self.root/task),task)
                registry.mark_started('s','run',task,task)
            self.disclose = lambda tasks,fields:registry.disclose('s','run',tasks,fields)
            with self.assertRaisesRegex(ValueError,'Held-out'):
                self.request([{'source_tasks':['held-out'],'claims':[],
                               'artifact':self.parent.version,'scope':'trial'}])
        self.llm.complete.assert_not_called()


if __name__=='__main__':
    unittest.main()
