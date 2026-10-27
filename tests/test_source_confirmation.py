from tests.support import test_temp_dir
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from dace.agents.policy import IDENTITY, PolicyRuntime, SourcePolicy, compile_transform
from dace.benchmarks.base import Task, TaskResult
from dace.benchmarks.live import LiveExecutor
from dace.core import Artifact, Prediction
from dace.source_confirmation import make_plan, run_plan, validate_plan


class PolicyTests(unittest.TestCase):
    def test_response_transform_really_changes_parser_input(self):
        runtime = PolicyRuntime(SourcePolicy(response_code=
            'def transform(value):\n    return value.strip()\n'))
        self.assertEqual(runtime.response_transform('  response\n'),'response')
        self.assertTrue(runtime.events[-1]['changed'])

    def test_command_code_can_edit_actual_command_batches(self):
        runtime = PolicyRuntime(SourcePolicy(command_code='def transform(value):\n'
            '    return [{"keystrokes": c["keystrokes"] + "\\n", '
            '"duration_sec": c["duration_sec"]} for c in value]\n'))
        before = [{'keystrokes':'pwd','duration_sec':1}]
        self.assertEqual(runtime.command_transform(before)[0]['keystrokes'],'pwd\n')
        self.assertEqual(before[0]['keystrokes'],'pwd')
        self.assertTrue(runtime.events[-1]['changed'])

    def test_identity_is_executed_but_not_reported_as_changed(self):
        runtime = PolicyRuntime(SourcePolicy())
        runtime.response_transform('hello')
        self.assertEqual(runtime.events,[{'hook':'response_code','executed':True,'changed':False}])

    def test_external_effects_and_private_runtime_are_outside_edit_surface(self):
        for code in ('def transform(value):\n    import os\n    return value',
                     'def transform(value):\n    return open("secret")',
                     'def transform(value):\n    return value.__class__',
                     'def transform(value):\n    while True:\n        pass',
                     'def transform(value):\n    value[0] = "x"\n    return value'):
            with self.subTest(code=code), self.assertRaises(ValueError):
                compile_transform(code)

    def test_invalid_output_cannot_reach_terminal(self):
        for code in ('def transform(value):\n    return "oops"',
                     'def transform(value):\n    return [{"keystrokes":"pwd","duration_sec":True}]',
                     'def transform(value):\n    return [{"keystrokes":"pwd","duration_sec":61}]'):
            with self.subTest(code=code),self.assertRaises(ValueError):
                PolicyRuntime(SourcePolicy(command_code=code)).command_transform([])

    def test_every_policy_edit_changes_content_address(self):
        baseline = SourcePolicy()
        self.assertNotEqual(baseline.version,replace(baseline,instruction_suffix='new').version)
        self.assertEqual(baseline.response_code,IDENTITY)


class SourceSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset'
        for name in ('a','b'):
            task = self.dataset/name
            task.mkdir(parents=True)
            (task/'task.toml').write_text('schema_version="1.1"',encoding='utf-8')
            (task/'instruction.md').write_text('solve task',encoding='utf-8')
        self.prediction = Prediction('parser normalization','raw_response_parse_valid',
            ('model','parser'),(('raw_response_parse_valid',False),),
            (('raw_response_parse_valid',False),('response_parse_valid',True)),
            (('command_batch_valid',True),),'COMPENSATION')
        self.plan = make_plan(SourcePolicy(),SourcePolicy(response_code=
            'def transform(value):\n    return value.strip()\n'),[self.prediction],['a','b'],
            self.dataset,['verified-runner'],'deepseek/deepseek-v4.1-flash')
        self.output = self.root/'session'

    @staticmethod
    def execute(config,task,task_path,**kwargs):
        changed = kwargs['patch_keys'] is not None
        return TaskResult(task.task_id,{'raw_response_parse_valid':False,
            'response_parse_valid':changed,'command_batch_valid':True},float(changed),
            activated=True if changed else None)

    def test_paired_source_contract_resumes_without_replaying_completed_task(self):
        with patch('dace.source_confirmation.LiveExecutor.execute',side_effect=self.execute) as execute:
            first = run_plan(self.plan,self.output,batch=1)
            self.assertEqual(execute.call_count,2)
            self.assertEqual(first['evidence']['status'],'EVIDENCE_INCOMPLETE')
            self.assertEqual(first['budget']['spent'],2)
            self.assertEqual(list(first['budget']['locks'].values()),[2])
        with patch('dace.source_confirmation.LiveExecutor.execute',side_effect=self.execute) as execute:
            second = run_plan(self.plan,self.output)
            self.assertEqual([c.args[1].task_id for c in execute.call_args_list],['b','b'])
            self.assertEqual(second['evidence']['status'],'EVIDENCE_COMPLETE')
            self.assertEqual(second['evidence']['claims'][0]['bridge'],'COMPENSATION')
            # Two pairs cannot establish a positive bounded-utility contribution.
            self.assertEqual(second['evidence']['claims'][0]['contribution'],'UNRESOLVED')
        with patch('dace.source_confirmation.LiveExecutor.execute') as execute:
            self.assertEqual(run_plan(self.plan,self.output)['budget']['spent'],4)
            execute.assert_not_called()

    def test_failure_after_baseline_does_not_retry_candidate_or_replace_task(self):
        def fail(config,task,task_path,**kwargs):
            if kwargs['patch_keys']:
                raise RuntimeError('infrastructure problem')
            return self.execute(config,task,task_path,**kwargs)
        with patch('dace.source_confirmation.LiveExecutor.execute',side_effect=fail) as execute:
            result = run_plan(self.plan,self.output)
            self.assertEqual(execute.call_count,2)
        self.assertEqual(result['evidence']['status'],'TERMINATED_UNRESOLVED')
        self.assertEqual(result['observations'],[])
        with patch('dace.source_confirmation.LiveExecutor.execute') as execute:
            run_plan(self.plan,self.output)
            execute.assert_not_called()

    def test_frozen_runner_cannot_change_between_batches(self):
        with patch('dace.source_confirmation.LiveExecutor.execute',side_effect=self.execute):
            run_plan(self.plan,self.output,batch=1)
        self.plan['command'] = ['different-runner']
        with self.assertRaisesRegex(ValueError,'exact frozen'):
            run_plan(self.plan,self.output)

    def test_modified_task_manifest_is_rejected_before_running(self):
        (self.dataset/'a'/'instruction.md').write_text('different task',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'task inputs changed'):
            run_plan(self.plan,self.output)
        self.assertFalse(self.output.exists())

    def test_boolean_catalog_predicates_cannot_claim_real_observability(self):
        self.plan['contract']['predictions'][0]['baseline_boundary'] = [['planner_source_ok',False]]
        with self.assertRaisesRegex(ValueError,'actual boolean trial hooks'):
            validate_plan(self.plan)

    def test_changed_slot_cannot_be_omitted_from_activation_identity(self):
        self.plan['patch_keys'] = ['instruction_suffix']
        with self.assertRaisesRegex(ValueError,'actual source diff'):
            validate_plan(self.plan)

    def test_input_drift_during_run_ends_unresolved_but_preserves_usage(self):
        def execute(config,task,path,**kwargs):
            result = self.execute(config,task,path,**kwargs)
            (self.dataset/task.task_id/'instruction.md').write_text('changed during run',encoding='utf-8')
            return result
        with patch('dace.source_confirmation.LiveExecutor.execute',side_effect=execute) as runner:
            result = run_plan(self.plan,self.output)
            self.assertEqual(runner.call_count,1)
        self.assertEqual(result['evidence']['status'],'TERMINATED_UNRESOLVED')
        self.assertEqual(result['observations'],[])
        from dace.core import Ledger
        kinds = [e['kind'] for e in Ledger(self.output/'evidence.jsonl').events]
        self.assertIn('SOURCE_EXECUTION_RESULT',kinds)
        self.assertIn('SOURCE_EXECUTION_INVALIDATED',kinds)


class SourceBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy = SourcePolicy(instruction_suffix='edit')
        self.artifact = Artifact.create(self.policy.content,'parent')
        self.runner = LiveExecutor(['runner'],self.root,'model')

    def process(self,argv,**kwargs):
        request = json.loads(Path(argv[-3]).read_text(encoding='utf-8'))
        self.assertEqual(request['harness_artifact'],asdict(self.artifact))
        work = Path(argv[-1]).parent
        (work/'reward.txt').write_text('1',encoding='utf-8')
        (work/'trace.json').write_text(json.dumps({'task_id':'a','policy_version':self.policy.version,
            'boundary':{'command_batch_valid':True},'intervention':request['intervention'],
            'activation_events':{'instruction_suffix':True}}),encoding='utf-8')
        provenance = {'execution_kind':'harbor-agent','oracle_access':False,
            'benchmark_score_eligible':True,'reward_source':'official-verifier',
            'harness_artifact':self.artifact.version,'policy_version':self.policy.version}
        Path(argv[-1]).write_text(json.dumps({'schema':'dace-runner/v1','task_id':'a',
            'model':'model','provenance':provenance,'verifier_path':'reward.txt',
            'trace_path':'trace.json'}),encoding='utf-8')
        return SimpleNamespace(returncode=0,stdout='',stderr='')

    def execute(self):
        return self.runner.execute({},Task('a','primary'),self.root/'a',
            patch_keys=['instruction_suffix'],parent_version='parent',
            candidate_version=self.artifact.version,action_id='frozen-action',
            harness_artifact=self.artifact)

    def test_actual_source_and_activation_identity_cross_process_protocol(self):
        with patch('dace.benchmarks.live.run_owned_process',side_effect=self.process):
            result = self.execute()
        self.assertTrue(result.activated)
        self.assertEqual(result.utility,1)

    def test_foreign_policy_provenance_cannot_be_accepted(self):
        def process(argv,**kwargs):
            result = self.process(argv,**kwargs)
            path = Path(argv[-1])
            data = json.loads(path.read_text(encoding='utf-8'))
            data['provenance']['policy_version'] = 'unrelated-policy'
            path.write_text(json.dumps(data),encoding='utf-8')
            return result
        with patch('dace.benchmarks.live.run_owned_process',side_effect=process):
            with self.assertRaisesRegex(ValueError,'different frozen harness'):
                self.execute()

    def test_foreign_policy_trace_cannot_borrow_valid_provenance(self):
        def process(argv,**kwargs):
            result = self.process(argv,**kwargs)
            path = Path(argv[-1]).parent/'trace.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['policy_version'] = 'foreign-policy'
            path.write_text(json.dumps(data),encoding='utf-8')
            return result
        with patch('dace.benchmarks.live.run_owned_process',side_effect=process):
            with self.assertRaisesRegex(ValueError,'different frozen harness'):
                self.execute()


if __name__ == '__main__':
    unittest.main()
