"""Controlled four-arm executors and real verifier-file fixtures; no benchmark result."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dace.agents.policy import SourcePolicy
from dace.benchmarks.base import TaskResult
from dace.benchmarks.live import LiveExecutor
from dace.core import Artifact
from dace.source_factorial import interaction_evidence,make_factorial_plan,run_factorial,factorial_feedback
from dace.task_registry import TaskRegistry
from tests.support import test_temp_dir


class FactorialStatisticsTests(unittest.TestCase):
    def unit(self,rewards=None,activated=True):
        return {'rewards':rewards or {'00':0,'10':0,'01':0,'11':1},
                'activated':dict.fromkeys(('10','01','11'),activated)}

    def test_one_task_is_not_a_confirmed_interaction(self):
        evidence = interaction_evidence([self.unit()],.05)
        self.assertEqual(evidence['mean'],1)
        self.assertEqual(evidence['interaction_status'],'UNRESOLVED')
        self.assertFalse(evidence['promotion_eligible'])
        self.assertFalse(evidence['causal_path_identified'])

    def test_four_cell_contrast_has_a_range_four_fixed_look_interval(self):
        evidence = interaction_evidence([self.unit() for _ in range(128)],.05)
        self.assertGreater(evidence['lower'],0)
        self.assertEqual(evidence['interaction_status'],'POSITIVE')
        negative = interaction_evidence([self.unit({'00':0,'10':1,'01':1,'11':1}) for _ in range(128)],.05)
        self.assertEqual(negative['interaction_status'],'NEGATIVE')

    def test_missing_activation_does_not_become_an_attributed_joint_effect(self):
        evidence = interaction_evidence([self.unit(activated=None) for _ in range(128)],.05)
        self.assertEqual(evidence['assignment_interaction'],'POSITIVE')
        self.assertEqual(evidence['interaction_status'],'ACTIVATION_UNRESOLVED')

    def test_missing_or_fractional_controls_are_not_substituted(self):
        for rewards in ({'00':0,'10':0,'11':1},{'00':0,'10':0,'01':.5,'11':1}):
            with self.subTest(rewards=rewards),self.assertRaises(ValueError):
                interaction_evidence([self.unit(rewards)],.05)


class NativeFactorialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset'
        for task in ('a','b','held-out'):
            path = self.dataset/task
            path.mkdir(parents=True)
            (path/'task.toml').write_text('schema_version="1.1"',encoding='utf-8')
            (path/'instruction.md').write_text('Unique task '+task,encoding='utf-8')
        registry = self.root/'study.sqlite3'
        with TaskRegistry(registry) as db:
            db.create_study('s',self.dataset,{'CONFIRM':['a','b'],'EVAL':['held-out']})
            db.create_lineage('s','root')
        self.study = {'registry':str(registry),'id':'s','lineage':'root'}
        for name in ('dace.source_factorial.verify_runtime_lock','dace.runtime_lock.verify_runtime_lock'):
            mocked = patch(name,return_value='fixture')
            mocked.start()
            self.addCleanup(mocked.stop)
        self.parent = Artifact.create(SourcePolicy().content,kind='BASELINE')
        self.first = {'response_code':'def transform(value):\n    return value.strip()\n'}
        self.second = {'instruction_suffix':'Controlled instruction intervention.'}
        self.plan = make_factorial_plan(self.parent,self.first,self.second,['a','b'],self.dataset,
            ['controlled-runner'],'controlled-model',self.study,{'schema':'fixture'})
        self.output = self.root/'run'
        self.calls = []
        self.rewards = {'00':0,'10':0,'01':0,'11':1}
        mocked = patch.object(LiveExecutor,'execute',autospec=True,side_effect=self.execute)
        self.runner = mocked.start()
        self.addCleanup(mocked.stop)

    def execute(self,runner,config,task,path,**kwargs):
        artifact = kwargs['harness_artifact']
        arm = next(key for key,value in self.plan['arms'].items() if value['version']==artifact.version)
        self.calls.append((task.task_id,arm))
        reward = runner.work_dir/f'{task.task_id}-{arm}'/'verifier/reward.txt'
        reward.parent.mkdir(parents=True,exist_ok=True)
        reward.write_text(str(self.rewards[arm]),encoding='utf-8')
        return TaskResult(task.task_id,{'response_parse_valid':True},self.rewards[arm],
            activated=True if arm!='00' else None,
            usage={'n_input_tokens':10,'n_output_tokens':5,'cost_usd':.01},verifier_path=str(reward),
            provenance={'execution_kind':'harbor-agent','oracle_access':False,'benchmark_score_eligible':True,
                'reward_source':'official-verifier','harness_artifact':artifact.version,
                'policy_version':SourcePolicy(**json.loads(artifact.content)).version})

    def run_stage(self,batch=None):
        return run_factorial(self.plan,self.output,batch)

    def test_all_four_native_configurations_are_measured_per_task_and_costed_once(self):
        result = self.run_stage()
        self.assertEqual(result['status'],'FACTORIAL_COMPLETE')
        self.assertEqual(self.calls,[(task,arm) for task in ('a','b') for arm in ('00','10','01','11')])
        self.assertEqual(result['budget'],{'total':8,'spent':8,'locks':{}})
        self.assertEqual(result['interaction']['mean'],1)
        self.assertEqual(result['interaction']['interaction_status'],'UNRESOLVED')
        self.assertAlmostEqual(result['execution_usage']['cost_usd'],.08)
        self.assertEqual(result['execution_usage']['unsettled_task_units'],0)
        self.assertEqual(factorial_feedback(self.output/'evidence.jsonl')['candidate'],self.plan['arms']['11'])

    def test_partial_batches_recover_only_remaining_complete_units(self):
        result = self.run_stage(batch=1)
        self.assertEqual(result['status'],'PENDING_FACTORIAL')
        self.assertNotIn('interaction',result)
        self.assertEqual(result['budget']['spent'],4)
        self.assertEqual(self.run_stage()['status'],'FACTORIAL_COMPLETE')
        self.assertEqual(len(self.calls),8)
        self.assertEqual(len(set(self.calls)),8)

    def test_terminal_resume_and_certificate_ignore_mutable_summary(self):
        expected = self.run_stage()
        before = (self.output/'evidence.jsonl').read_bytes()
        (self.output/'summary.json').write_text('{"interaction":"invented"}',encoding='utf-8')
        self.assertEqual(self.run_stage(),expected)
        self.assertEqual((self.output/'evidence.jsonl').read_bytes(),before)
        self.assertEqual(len(self.calls),8)

    def test_interrupted_arm_terminates_without_replaying_any_control(self):
        original = self.runner.side_effect
        def interrupt(*args,**kwargs):
            if len(self.calls)==1:
                raise KeyboardInterrupt('controlled interruption')
            return original(*args,**kwargs)
        self.runner.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage()
        self.runner.side_effect = original
        result = self.run_stage()
        self.assertEqual(result['status'],'FACTORIAL_UNRESOLVED')
        self.assertEqual(result['reason'],'INTERRUPTED_FOUR_ARM_UNIT')
        self.assertEqual(len(self.calls),1)
        self.assertEqual(result['budget']['spent'],4)
        self.assertIsNone(result['execution_usage']['cost_usd'])
        with self.assertRaises(ValueError):
            factorial_feedback(self.output/'evidence.jsonl')

    def test_missing_runtime_control_does_not_fall_back_to_parent_or_score_zero(self):
        original = self.runner.side_effect
        def fail(*args,**kwargs):
            if len(self.calls)==2:
                raise RuntimeError('controlled missing third arm')
            return original(*args,**kwargs)
        self.runner.side_effect = fail
        result = self.run_stage()
        self.assertEqual(result['status'],'FACTORIAL_UNRESOLVED')
        self.assertNotIn('interaction',result)
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.run_stage(),result)
        self.assertEqual(len(self.calls),2)

    def test_invalid_role_or_overlap_is_rejected_before_any_execution(self):
        with self.assertRaisesRegex(ValueError,'role'):
            make_factorial_plan(self.parent,self.first,self.second,['held-out'],self.dataset,
                ['controlled-runner'],'controlled-model',self.study,{'schema':'fixture'})
        with self.assertRaisesRegex(ValueError,'disjoint'):
            make_factorial_plan(self.parent,self.second,{'instruction_suffix':'Another edit'},['a'],
                self.dataset,['controlled-runner'],'controlled-model',self.study,{'schema':'fixture'})
        self.runner.assert_not_called()

    def test_noop_and_illegal_source_edits_cannot_create_four_controls(self):
        for edits in ({'response_code':SourcePolicy().response_code},
                      {'response_code':'def transform(value):\n    return open("private")\n'}):
            with self.subTest(edits=edits),self.assertRaises(ValueError):
                make_factorial_plan(self.parent,edits,self.second,['a'],self.dataset,
                    ['controlled-runner'],'controlled-model',self.study,{'schema':'fixture'})
        self.runner.assert_not_called()

    def test_raw_reward_changes_invalidate_read_only_certificate_and_resume(self):
        self.run_stage()
        (self.output/'executions/a-11/verifier/reward.txt').write_text('0',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'reward'):
            factorial_feedback(self.output/'evidence.jsonl')
        with self.assertRaisesRegex(ValueError,'reward'):
            self.run_stage()
        self.assertEqual(len(self.calls),8)

    def test_changed_joint_artifact_and_incomplete_certificate_fail_closed(self):
        plan = copy.deepcopy(self.plan)
        plan['arms']['11']['version'] = plan['arms']['10']['version']
        with self.assertRaisesRegex(ValueError,'controls'):
            run_factorial(plan,self.output)
        self.run_stage(batch=1)
        with self.assertRaisesRegex(ValueError,'not complete'):
            factorial_feedback(self.output/'evidence.jsonl')
