from tests.support import test_temp_dir
"""Controlled execution fixtures; these are not model/benchmark result runs."""
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from dace.agents.policy import SourcePolicy
from dace.benchmarks.base import TaskResult
from dace.core import Artifact, Ledger, Prediction, digest
from dace.source_confirmation import make_plan, run_plan
from dace.source_selection import (confirmation_certificate, make_selection_plan,
                                   run_selection, validate_selection_plan)
from dace.task_registry import TaskRegistry


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset'
        self.tasks = [f'select-{i:02}' for i in range(32)]
        for name in ['confirm-a','confirm-b','diag','eval',*self.tasks]:
            path = self.dataset/name
            path.mkdir(parents=True)
            (path/'task.toml').write_text('schema_version="1.1"',encoding='utf-8')
            (path/'instruction.md').write_text('Unique unit '+name,encoding='utf-8')
        self.registry = self.root/'registry.sqlite3'
        with TaskRegistry(self.registry) as registry:
            registry.create_study('s',self.dataset,{'CONFIRM':['confirm-a','confirm-b'],
                'SELECT':self.tasks,'DIAG':['diag'],'EVAL':['eval']})
            registry.create_lineage('s','run')
        self.study = {'registry':str(self.registry),'id':'s','lineage':'run'}
        self.runtime = {'schema':'controlled-test-fixture'}
        self.command = ['fixture-runner']
        self.model = 'deepseek/deepseek-v4.1-flash'
        self.prediction = Prediction('m','raw_response_parse_valid',('response','parser'),
            (('raw_response_parse_valid',False),),(('raw_response_parse_valid',True),),
            (('command_batch_valid',True),),'SOURCE_REPAIR')
        self.counter = 0
        self.equal_selection = False
        runtime = patch('dace.runtime_lock.verify_runtime_lock',return_value='controlled-fixture')
        runtime.start()
        self.addCleanup(runtime.stop)
        runner = patch('dace.benchmarks.live.LiveExecutor.execute',side_effect=self.execute)
        self.runner = runner.start()
        self.addCleanup(runner.stop)
        self.confirm_a = self.confirm('confirm-a','repair-a')
        self.plan = self.selection_plan([self.confirm_a])
        self.output = self.root/'selection'

    def execute(self,config,task,path,**kwargs):
        self.counter += 1
        artifact = kwargs['harness_artifact']
        policy = SourcePolicy(**json.loads(artifact.content))
        changed = bool(policy.instruction_suffix)
        utility = 1 if changed or (self.equal_selection and task.task_id in self.tasks) else 0
        reward = self.execution_output/'executions'/f'fixture-{self.counter}'/'reward.txt'
        reward.parent.mkdir(parents=True,exist_ok=True)
        reward.write_text(str(utility),encoding='utf-8')
        return TaskResult(task.task_id,{'raw_response_parse_valid':changed,
            'response_parse_valid':True,'command_batch_valid':True},utility,
            True if kwargs['patch_keys'] else None,
            usage={'n_input_tokens':12,'n_output_tokens':4,'cost_usd':.001},
            verifier_path=str(reward),provenance={'execution_kind':'harbor-agent',
                'oracle_access':False,'benchmark_score_eligible':True,
                'reward_source':'official-verifier','harness_artifact':artifact.version,
                'policy_version':policy.version})

    def confirm(self,task,suffix):
        plan = make_plan(SourcePolicy(),SourcePolicy(instruction_suffix=suffix),
            [self.prediction],[task],self.dataset,self.command,self.model,
            study=self.study,runtime_lock=self.runtime)
        self.execution_output = self.root/task
        result = run_plan(plan,self.execution_output)
        self.assertEqual(result['evidence']['minimal'],'MINIMAL_OBSERVED')
        return self.execution_output/'evidence.jsonl'

    def selection_plan(self,certificates,tasks=None):
        tasks = self.tasks if tasks is None else tasks
        return make_selection_plan(certificates,tasks,{t:'primary' for t in tasks},
            self.dataset,self.command,self.model,self.study,self.runtime)

    def select(self,plan=None,batch=None):
        self.execution_output = self.output
        return run_selection(plan or self.plan,self.output,batch)

    def test_own_confirmation_is_recomputed_from_journal_not_summary(self):
        (self.confirm_a.parent/'summary.json').write_text('{"evidence":"fake summary"}',encoding='utf-8')
        certificate = confirmation_certificate(self.confirm_a)
        self.assertEqual(certificate['attribution'][0]['bridge'],'SOURCE_REPAIR')

    def test_foreign_discrimination_claim_is_not_reported_as_own_deployment_attribution(self):
        self.prediction = replace(self.prediction,action_repairs_source=False)
        other = self.confirm('confirm-b','foreign-probe')
        self.assertEqual(confirmation_certificate(other)['attribution'],[])

    def test_fixed_look_resume_uses_remaining_tasks_then_preserves_exact_winner(self):
        start = self.counter
        first = self.select(batch=1)
        self.assertEqual(first['status'],'PENDING_SELECTION')
        self.assertEqual(first['budget']['spent'],2)
        self.assertEqual(self.counter-start,2)
        self.assertEqual(list(first['budget']['locks'].values()),[62])
        result = self.select()
        artifact = self.plan['certificates'][0]['candidate']
        self.assertEqual(result['status'],'SELECTION_COMPLETE')
        self.assertEqual(result['winner'],artifact['version'])
        self.assertEqual(result['deployed_artifact'],artifact)
        self.assertEqual(result['attribution'][0]['bridge'],'SOURCE_REPAIR')
        self.assertEqual(result['budget']['spent'],64)
        self.assertEqual(self.counter-start,64)
        before = (self.output/'evidence.jsonl').read_bytes()
        self.select()
        self.assertEqual(self.counter-start,64)
        self.assertEqual((self.output/'evidence.jsonl').read_bytes(),before)

    def test_one_task_cannot_promote_despite_perfect_observed_improvement(self):
        result = self.select(self.selection_plan([self.confirm_a],self.tasks[:1]))
        self.assertIsNone(result['winner'])
        self.assertEqual(result['deployed_artifact'],self.plan['baseline'])

    def test_no_utility_gain_does_not_promote_confirmed_source_repair(self):
        self.equal_selection = True
        result = self.select()
        self.assertIsNone(result['winner'])
        self.assertEqual(result['deployed_artifact'],self.plan['baseline'])

    def test_multiple_candidates_share_one_baseline_execution_per_task(self):
        other = self.confirm('confirm-b','repair-b')
        plan = self.selection_plan([self.confirm_a,other],self.tasks[:1])
        start = self.counter
        result = self.select(plan)
        self.assertEqual(self.counter-start,3)
        self.assertEqual(result['budget']['spent'],3)

    def test_failed_candidate_terminates_without_replacement_or_baseline_replay(self):
        def fail(config,task,path,**kwargs):
            if kwargs['patch_keys']:
                raise RuntimeError('controlled infrastructure failure')
            return self.execute(config,task,path,**kwargs)
        with patch('dace.benchmarks.live.LiveExecutor.execute',side_effect=fail) as runner:
            result = self.select()
            self.assertEqual(runner.call_count,2)
        self.assertEqual(result['status'],'TERMINATED_UNRESOLVED')
        self.assertIsNone(result['winner'])
        self.assertEqual(result['budget']['spent'],2)
        count = self.counter
        self.select()
        self.assertEqual(self.counter,count)

    def test_process_interrupt_between_baseline_and_candidate_never_replays_unit(self):
        def interrupt(config,task,path,**kwargs):
            if kwargs['patch_keys']:
                raise KeyboardInterrupt('controlled process interruption')
            return self.execute(config,task,path,**kwargs)
        with patch('dace.benchmarks.live.LiveExecutor.execute',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.select()
        count = self.counter
        result = self.select()
        self.assertEqual(self.counter,count)
        self.assertEqual(result['status'],'TERMINATED_UNRESOLVED')
        self.assertEqual(result['budget']['spent'],2)

    def test_bookkeeping_interrupt_after_fixed_look_does_not_execute_again(self):
        append = Ledger.append
        def interrupt(ledger,kind,**payload):
            if kind=='SELECTION_RESULT':
                raise KeyboardInterrupt('result bookkeeping interruption')
            return append(ledger,kind,**payload)
        with patch.object(Ledger,'append',new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.select()
        count = self.counter
        result = self.select()
        self.assertEqual(self.counter,count)
        self.assertIsNotNone(result['winner'])
        ledger = Ledger(self.output/'evidence.jsonl')
        self.assertEqual(sum(e['kind']=='SELECTION_RESULT' for e in ledger.events),1)

    def test_interrupt_between_protocol_freeze_and_reservation_can_resume_safely(self):
        plan = self.selection_plan([self.confirm_a],self.tasks[:1])
        append = Ledger.append
        def interrupt(ledger,kind,**payload):
            if kind=='BUDGET_RESERVED':
                raise KeyboardInterrupt('reservation not yet committed')
            return append(ledger,kind,**payload)
        with patch.object(Ledger,'append',new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.select(plan)
        count = self.counter
        result = self.select(plan)
        self.assertEqual(result['status'],'SELECTION_COMPLETE')
        self.assertEqual(result['budget']['spent'],2)
        self.assertEqual(self.counter-count,2)

    def test_changed_certificate_cannot_enter_frozen_selection(self):
        Ledger(self.confirm_a).append('AUDIT_NOTE',note='new event changes frozen evidence identity')
        with self.assertRaisesRegex(ValueError,'certificate changed'):
            self.select()
        self.assertFalse(self.output.exists())

    def test_invalidated_confirmation_is_rejected_even_with_cached_success(self):
        Ledger(self.confirm_a).append('SOURCE_EXECUTION_INVALIDATED',reason='controlled drift')
        with self.assertRaisesRegex(ValueError,'invalidated'):
            self.selection_plan([self.confirm_a])

    def test_candidate_artifact_cannot_be_replaced_in_selection_plan(self):
        self.plan['certificates'][0]['candidate']['version'] = 'different-candidate'
        with self.assertRaisesRegex(ValueError,'certificate changed'):
            self.select()

    def test_confirmation_and_selection_samples_cannot_overlap(self):
        with self.assertRaisesRegex(ValueError,'independent task roles'):
            self.selection_plan([self.confirm_a],['confirm-a'])

    def test_eval_pool_cannot_be_used_for_selection(self):
        plan = self.selection_plan([self.confirm_a],['eval'])
        start = self.counter
        with self.assertRaisesRegex(ValueError,'violates frozen study'):
            self.select(plan)
        self.assertEqual(self.counter,start)

    def test_deployed_patch_keeps_its_version_as_next_rounds_parent(self):
        parent = Artifact(**self.plan['certificates'][0]['candidate'])
        plan = make_plan(parent,SourcePolicy(instruction_suffix='next repair'),[self.prediction],
            ['confirm-b'],self.dataset,self.command,self.model,study=self.study,runtime_lock=self.runtime)
        self.assertEqual(plan['baseline'],asdict(parent))
        self.assertEqual(plan['candidate']['parent'],parent.version)


if __name__=='__main__':
    unittest.main()
