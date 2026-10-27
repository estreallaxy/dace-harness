from tests.support import test_temp_dir
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

from dace.agents.policy import SourcePolicy
from dace.benchmarks.base import TaskResult
from dace.core import Artifact,Ledger
from dace.source_diagnostics import make_diagnostic_plan,run_diagnostic_proposal
from dace.source_proposals import SourceCodingAgent
from dace.task_registry import TaskRegistry
from tests.test_source_proposals import proposal


class SourceDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset'
        for name in ('d0','d1','d2','eval'):
            path = self.dataset/name
            path.mkdir(parents=True)
            (path/'task.toml').write_text('schema_version="1.1"',encoding='utf-8')
            (path/'instruction.md').write_text(name,encoding='utf-8')
        self.registry = self.root/'registry.sqlite3'
        with TaskRegistry(self.registry) as registry:
            registry.create_study('s',self.dataset,{'DIAG':['d0','d1','d2'],'EVAL':['eval']})
            registry.create_lineage('s','run')
        runtime = patch('dace.source_diagnostics.verify_runtime_lock',return_value='fixture')
        runtime.start()
        self.addCleanup(runtime.stop)
        runtime2 = patch('dace.runtime_lock.verify_runtime_lock',return_value='fixture')
        runtime2.start()
        self.addCleanup(runtime2.stop)
        self.parent = Artifact.create(SourcePolicy().content,kind='BASELINE')
        self.plan = make_diagnostic_plan(self.parent,['d0','d1','d2'],self.dataset,['fixture-runner'],
            'model',{'registry':str(self.registry),'id':'s','lineage':'run'},
            {'schema':'controlled-fixture'})
        self.output = self.root/'session'
        self.llm = Mock(model='model')
        self.llm.complete.return_value = (json.dumps(proposal()),
            {'cost':.001,'prompt_tokens':50,'completion_tokens':100})
        self.agent = SourceCodingAgent(self.llm)
        runner = patch('dace.benchmarks.live.LiveExecutor.execute',side_effect=self.execute)
        self.runner = runner.start()
        self.addCleanup(runner.stop)

    def execute(self,config,task,path,**kwargs):
        reward = self.output/'executions'/task.task_id/'verifier/reward.txt'
        reward.parent.mkdir(parents=True,exist_ok=True)
        reward.write_text('0',encoding='utf-8')
        return TaskResult(task.task_id,{'response_parse_valid':False},0,
            usage={'n_input_tokens':10,'n_output_tokens':5,'cost_usd':.01},
            verifier_path=str(reward))

    def run_stage(self,batch=None):
        return run_diagnostic_proposal(self.plan,self.output,self.agent,batch=batch)

    def test_batches_preserve_remaining_tasks_and_do_not_request_coding_early(self):
        first = self.run_stage(batch=1)
        self.assertEqual(first['remaining'],['d1','d2'])
        self.assertEqual(first['budget']['spent'],1)
        self.llm.complete.assert_not_called()
        second = self.run_stage(batch=1)
        self.assertEqual(second['remaining'],['d2'])
        self.llm.complete.assert_not_called()
        final = self.run_stage()
        self.assertEqual(final['status'],'PROPOSAL_READY')
        self.assertEqual(final['budget']['spent'],13)
        self.assertEqual(final['budget']['locks'],{})
        self.assertEqual(self.runner.call_count,3)
        self.llm.complete.assert_called_once()
        self.assertTrue(final['execution_usage']['accounting_complete'])
        self.assertEqual(final['execution_usage']['cost_usd'],.03)
        self.assertTrue(final['optimizer_usage']['accounting_complete'])

    def test_completed_stage_can_resume_without_calls_or_journal_edits(self):
        self.run_stage()
        before = hashlib.sha256((self.output/'evidence.jsonl').read_bytes()).hexdigest()
        again = self.run_stage()
        self.assertEqual(again['status'],'PROPOSAL_READY')
        self.assertEqual(self.runner.call_count,3)
        self.llm.complete.assert_called_once()
        self.assertEqual(hashlib.sha256((self.output/'evidence.jsonl').read_bytes()).hexdigest(),before)

    def test_interrupted_task_is_not_replaced_or_reexecuted(self):
        self.runner.side_effect = KeyboardInterrupt('controlled process interruption')
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage()
        self.runner.side_effect = self.execute
        result = self.run_stage()
        self.assertEqual(result['status'],'DIAGNOSTIC_UNRESOLVED')
        self.assertEqual(self.runner.call_count,1)
        self.llm.complete.assert_not_called()

    def test_started_coding_request_without_response_is_not_retried(self):
        self.llm.complete.side_effect = KeyboardInterrupt('controlled lost API response')
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage()
        result = self.run_stage()
        self.assertEqual(result['status'],'PROPOSAL_UNRESOLVED')
        self.llm.complete.assert_called_once()
        self.assertEqual(result['budget']['spent'],13)
        self.assertIsNone(result['optimizer_usage']['cost_usd'])

    def test_committed_proposal_survives_crash_before_final_summary(self):
        append = Ledger.append
        def interrupt(ledger,kind,**payload):
            if kind=='DIAGNOSTIC_STAGE_RESULT':
                raise KeyboardInterrupt('controlled bookkeeping interruption')
            return append(ledger,kind,**payload)
        with patch.object(Ledger,'append',new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_stage()
        result = self.run_stage()
        self.assertEqual(result['status'],'PROPOSAL_READY')
        self.llm.complete.assert_called_once()
        self.assertEqual(self.runner.call_count,3)

    def test_bad_model_json_keeps_price_but_does_not_create_a_candidate(self):
        self.llm.complete.return_value = ('invalid json',{'cost':.01,'prompt_tokens':1,'completion_tokens':2})
        result = self.run_stage()
        self.assertEqual(result['status'],'PROPOSAL_UNRESOLVED')
        self.assertNotIn('proposal',result)
        self.assertEqual(result['optimizer_usage']['cost_usd'],.01)

    def test_disclosure_records_all_actual_diagnostic_sources(self):
        self.run_stage()
        with TaskRegistry(self.registry) as registry:
            disclosures = [e['payload'] for e in registry.audit_events() if e['kind']=='AGENT_DISCLOSED']
        self.assertEqual(disclosures[-1]['tasks'],['d0','d1','d2'])
        self.assertNotIn('eval',disclosures[-1]['tasks'])

    def test_eval_tasks_are_rejected_before_execution(self):
        self.plan['tasks'] = ['eval']
        from dace.benchmarks.fingerprint import task_fingerprint
        self.plan['task_fingerprints'] = {'eval':task_fingerprint(self.dataset/'eval')}
        self.plan['total_budget'] = 11
        with self.assertRaisesRegex(ValueError,'violates frozen study'):
            self.run_stage()
        self.runner.assert_not_called()

    def test_wrong_optimizer_model_cannot_start_request(self):
        self.llm.model = 'different-model'
        result = self.run_stage()
        self.assertEqual(result['status'],'PROPOSAL_UNRESOLVED')
        self.llm.complete.assert_not_called()

    def test_explicit_no_change_is_not_mislabeled_as_an_actionable_candidate(self):
        self.llm.complete.return_value = ('{"actions":[],"mechanisms":[]}',
            {'cost':.001,'prompt_tokens':10,'completion_tokens':5})
        result = self.run_stage()
        self.assertEqual(result['status'],'NO_CHANGE_PROPOSED')
        self.assertFalse(result['proposal']['actions'])
        self.assertEqual(result['budget']['spent'],13)
        self.llm.complete.assert_called_once()

    def test_output_token_cap_mismatch_cannot_silently_change_request_profile(self):
        self.agent.max_tokens = 6000
        result = self.run_stage()
        self.assertEqual(result['status'],'PROPOSAL_UNRESOLVED')
        self.llm.complete.assert_not_called()


if __name__=='__main__':
    unittest.main()
