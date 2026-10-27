from tests.support import test_temp_dir
"""Whole pipeline fixtures: no real API, Docker, or benchmark claims."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from dace.agents.policy import SourcePolicy
from dace.benchmarks.base import TaskResult
from dace.benchmarks.live import LiveExecutor
from dace.core import Artifact,Ledger,digest
from dace.source_optimization import make_optimization_plan,run_optimization,validate_optimization_plan
from dace.source_proposals import SourceCodingAgent
from dace.task_registry import TaskRegistry


def candidate(suffix):
    return {'actions':[{'id':'repair','edits':{'instruction_suffix':suffix}}],
        'mechanisms':[{'id':'dispatch','component':'terminal','source':'terminal_dispatch_completed',
            'propagation':['terminal'],'baseline':{'terminal_dispatch_completed':False},
            'own_action':'repair','outcomes':{'repair':{
                'intervention':{'terminal_dispatch_completed':True},
                'mediator':{'command_batch_valid':True},'bridge':'SOURCE_REPAIR'}},
            'explanation':'Controlled fixture of a terminal instruction repair.'}]}


class SourceOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset'
        self.rounds = [{'diagnostic':[f'd{i}'],'confirmation':[[f'c{i}']],
            'selection':[f's{i}-{j}' for j in range(32)],
            'families':{f's{i}-{j}':'primary' for j in range(32)}} for i in range(2)]
        roles = {'DIAG':['d0','d1'],'CONFIRM':['c0','c1'],
                 'SELECT':sum([r['selection'] for r in self.rounds],[]),'EVAL':['eval']}
        for name in sum(roles.values(),[]):
            path = self.dataset/name
            path.mkdir(parents=True)
            (path/'task.toml').write_text('schema_version="1.1"',encoding='utf-8')
            (path/'instruction.md').write_text('Unique task '+name,encoding='utf-8')
        registry_path = self.root/'study.sqlite3'
        with TaskRegistry(registry_path) as registry:
            registry.create_study('s',self.dataset,roles)
            registry.create_lineage('s','run')
        self.study = {'registry':str(registry_path),'id':'s','lineage':'run'}
        for name in ('dace.runtime_lock.verify_runtime_lock',
                     'dace.source_optimization.verify_runtime_lock',
                     'dace.source_diagnostics.verify_runtime_lock'):
            mock = patch(name,return_value='controlled-fixture')
            mock.start()
            self.addCleanup(mock.stop)
        self.parent = Artifact.create(SourcePolicy().content,kind='BASELINE')
        self.plan = make_optimization_plan(self.parent,self.rounds,self.dataset,['fixture-runner'],
            'fixture-model',self.study,{'schema':'fixture'},intervention_cap=1,action_cap=1)
        self.output = self.root/'optimization'
        self.calls = []
        self.llm = Mock(model='fixture-model')
        self.llm.complete.side_effect = self.propose
        self.agent = SourceCodingAgent(self.llm)
        mock = patch.object(LiveExecutor,'execute',autospec=True,side_effect=self.execute)
        self.runner = mock.start()
        self.addCleanup(mock.stop)

    def propose(self,messages,**kwargs):
        index = self.llm.complete.call_count
        return json.dumps(candidate('repair-'+str(index))),{
            'cost':.001,'prompt_tokens':10,'completion_tokens':20}

    def execute(self,runner,config,task,path,**kwargs):
        artifact = kwargs['harness_artifact']
        self.calls.append((task.task_id,artifact.version))
        changed = bool(kwargs.get('patch_keys'))
        reward = runner.work_dir/f'fixture-{len(self.calls)}'/'verifier/reward.txt'
        reward.parent.mkdir(parents=True,exist_ok=True)
        reward.write_text('1' if changed else '0',encoding='utf-8')
        return TaskResult(task.task_id,{'terminal_dispatch_completed':changed,'command_batch_valid':True},
            float(changed),True if changed else None,
            usage={'n_input_tokens':10,'n_output_tokens':5,'cost_usd':.01},
            verifier_path=str(reward),provenance={'execution_kind':'harbor-agent','oracle_access':False,
                'benchmark_score_eligible':True,'reward_source':'official-verifier',
                'harness_artifact':artifact.version,
                'policy_version':SourcePolicy(**json.loads(artifact.content)).version})

    def run_pipeline(self):
        return run_optimization(self.plan,self.output,self.agent)

    def test_two_rounds_preserve_winner_parent_and_global_charges(self):
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(len(result['rounds']),2)
        first,second = [r['deployed_artifact'] for r in result['rounds']]
        self.assertEqual(first['parent'],self.parent.version)
        self.assertEqual(second['parent'],first['version'])
        self.assertNotEqual(first['version'],second['version'])
        self.assertEqual(result['budget']['locks'],{})
        self.assertEqual(result['budget']['spent'],2*(11+2+64))
        self.assertEqual(len(self.calls),2*(1+2+64))
        self.assertEqual(len(set(self.calls)),len(self.calls))
        self.llm.complete.assert_called()
        self.assertEqual(self.llm.complete.call_count,2)
        self.assertAlmostEqual(result['execution_usage']['cost_usd'],len(self.calls)*.01)
        self.assertAlmostEqual(result['optimizer_usage']['cost_usd'],.002)
        self.assertEqual(result['optimizer_usage']['request_attempts'],2)
        # Second request contains the first confirmed claim, with original artifact scope.
        second_prompt = self.llm.complete.call_args.args[0][0]['content']
        self.assertIn(first['version'],second_prompt)
        self.assertIn('SOURCE_REPAIR',second_prompt)

    def test_terminal_resume_ignores_summary_and_makes_no_calls(self):
        expected = self.run_pipeline()
        (self.output/'summary.json').write_text('{"status":"fake"}',encoding='utf-8')
        before = (self.output/'evidence.jsonl').read_bytes()
        self.assertEqual(self.run_pipeline(),expected)
        self.assertEqual((self.output/'evidence.jsonl').read_bytes(),before)
        self.assertEqual(self.llm.complete.call_count,2)

    def test_all_stage_ceilings_reserved_before_first_execution(self):
        original = self.runner.side_effect
        def check(*args,**kwargs):
            events = Ledger(self.output/'evidence.jsonl').events
            if not self.calls:
                reserves = [e for e in events if e['kind']=='BUDGET_RESERVED']
                self.assertEqual(len(reserves),6)
                self.assertEqual(sum(e['payload']['amount'] for e in reserves),self.plan['total_budget'])
            return original(*args,**kwargs)
        self.runner.side_effect = check
        self.run_pipeline()

    def test_insufficient_global_budget_rejected_before_calls(self):
        self.plan['total_budget'] -= 1
        with self.assertRaisesRegex(ValueError,'Global budget'):
            self.run_pipeline()
        self.runner.assert_not_called()
        self.llm.complete.assert_not_called()

    def test_role_overlap_and_eval_leak_rejected_before_calls(self):
        for role_error in ('overlap','eval'):
            with self.subTest(role_error=role_error):
                plan = copy.deepcopy(self.plan)
                plan['rounds'][1]['diagnostic'] = ['d0' if role_error=='overlap' else 'eval']
                with self.assertRaises(ValueError):
                    validate_optimization_plan(plan)
        self.runner.assert_not_called()

    def test_interrupted_task_is_not_reexecuted_and_global_charge_settles(self):
        self.runner.side_effect = KeyboardInterrupt('controlled task interruption')
        with self.assertRaises(KeyboardInterrupt):
            self.run_pipeline()
        self.runner.side_effect = self.execute
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_UNRESOLVED')
        self.assertEqual(result['budget']['spent'],1)
        self.assertEqual(result['budget']['locks'],{})
        self.assertEqual(self.runner.call_count,1)
        self.llm.complete.assert_not_called()

    def test_crash_after_child_completion_recovers_without_repeating_requests(self):
        append = Ledger.append
        def interrupt(ledger,kind,**payload):
            if kind=='OPTIMIZATION_STAGE_FINISHED' and payload['stage']=='round-000/diagnostic':
                raise KeyboardInterrupt('controlled outer bookkeeping crash')
            return append(ledger,kind,**payload)
        with patch.object(Ledger,'append',new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pipeline()
        self.assertEqual(self.llm.complete.call_count,1)
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(self.llm.complete.call_count,2)
        self.assertEqual(result['budget']['spent'],154)

    def test_crash_before_outer_mirror_reconciles_child_charge_exactly_once(self):
        charge = __import__('dace.durable',fromlist=['PersistentBudget']).PersistentBudget.charge_confirmation
        def interrupt(budget,amount,owner,task):
            if owner=='round-000/diagnostic':
                raise KeyboardInterrupt('controlled outer mirror crash')
            return charge(budget,amount,owner,task)
        with patch('dace.durable.PersistentBudget.charge_confirmation',new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pipeline()
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(result['budget']['spent'],154)
        self.assertEqual(self.llm.complete.call_count,2)

    def test_completed_confirmation_recovery_retains_serialized_contract_identity(self):
        append = Ledger.append
        def interrupt(ledger,kind,**payload):
            if kind=='OPTIMIZATION_STAGE_FINISHED' and payload['stage']=='round-000/confirm-000':
                raise KeyboardInterrupt('controlled confirmation receipt interruption')
            return append(ledger,kind,**payload)
        with patch.object(Ledger,'append',new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pipeline()
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(len(self.calls),134)
        self.assertEqual(result['budget']['spent'],154)

    def test_completed_child_tampering_is_not_hidden_by_outer_success(self):
        self.run_pipeline()
        path = self.output/'round-000/diagnostic/evidence.jsonl'
        child = Ledger(path,durable=True)
        child.append('UNEXPECTED_AFTER_RECEIPT')
        child.close()
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            self.run_pipeline()

    def test_malformed_model_output_stops_without_confirmation_or_retry(self):
        self.llm.complete.side_effect = None
        self.llm.complete.return_value = ('malformed',{'cost':.001})
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_UNRESOLVED')
        self.assertEqual(result['budget']['spent'],11)
        self.assertEqual(len(self.calls),1)
        self.run_pipeline()
        self.llm.complete.assert_called_once()

    def test_no_change_is_recorded_without_synthetic_confirmation(self):
        self.llm.complete.side_effect = None
        self.llm.complete.return_value = ('{"actions":[],"mechanisms":[]}',{'cost':.001})
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(result['deployed_artifact'],self.plan['parent'])
        self.assertEqual(result['budget']['spent'],22)
        self.assertEqual(len(self.calls),2)
        self.assertTrue(all(r['selection'] is None for r in result['rounds']))

    def test_global_alpha_is_split_over_all_prespecified_looks(self):
        self.run_pipeline()
        events = Ledger(self.output/'evidence.jsonl').events
        child_plans = [e['payload']['plan'] for e in events if e['kind']=='OPTIMIZATION_STAGE_STARTED']
        alphas = [p['contract']['alpha'] if 'contract' in p else p['alpha']
                  for p in child_plans if 'contract' in p or 'alpha' in p]
        self.assertEqual(len(alphas),4)
        self.assertAlmostEqual(sum(alphas),self.plan['alpha'])

    def test_confirmation_interruption_stops_outer_run_without_retry(self):
        original = self.execute
        def interrupt(runner,config,task,path,**kwargs):
            if task.task_id=='c0' and kwargs.get('patch_keys'):
                raise KeyboardInterrupt('controlled confirmation interruption')
            return original(runner,config,task,path,**kwargs)
        self.runner.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_pipeline()
        count = self.runner.call_count
        self.runner.side_effect = original
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_UNRESOLVED')
        self.assertEqual(result['failed_stage'],'round-000/confirm-000')
        self.assertEqual(self.runner.call_count,count)
        self.assertEqual(result['budget']['spent'],13)
        self.assertIsNone(result['execution_usage']['cost_usd'])
        self.llm.complete.assert_called_once()

    def test_selection_interruption_stops_without_new_round_or_task_replay(self):
        original = self.execute
        def interrupt(runner,config,task,path,**kwargs):
            if task.task_id=='s0-0' and kwargs.get('patch_keys'):
                raise KeyboardInterrupt('controlled selection interruption')
            return original(runner,config,task,path,**kwargs)
        self.runner.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_pipeline()
        count = self.runner.call_count
        self.runner.side_effect = original
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_UNRESOLVED')
        self.assertEqual(result['failed_stage'],'round-000/selection')
        self.assertEqual(self.runner.call_count,count)
        self.assertEqual(result['budget']['spent'],15)
        self.assertEqual(result['deployed_artifact'],self.plan['parent'])
        self.assertIsNone(result['execution_usage']['cost_usd'])

    def test_optimizer_missing_price_does_not_hide_behind_complete_task_usage(self):
        self.llm.complete.side_effect = None
        self.llm.complete.return_value = ('{"actions":[],"mechanisms":[]}',{})
        result = self.run_pipeline()
        self.assertTrue(result['execution_usage']['accounting_complete'])
        self.assertFalse(result['optimizer_usage']['accounting_complete'])
        self.assertIsNone(result['optimizer_usage']['cost_usd'])

    def test_native_c5_reconstructs_persistent_state_and_stale_parent_targets(self):
        self.plan.update(schema='dace-source-optimization/v3',level='C5')
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(result['native_state']['level'],'C5')
        self.assertEqual(len(result['native_state']['records']),2)
        self.assertTrue(all(r['status']=='STALE' for r in result['native_state']['records'].values()))
        prompt = self.llm.complete.call_args.args[0][0]['content']
        context = json.loads(prompt.split('Context:\n',1)[1])
        self.assertEqual(context['accepted_history'][0]['claims'][0]['mechanism_status'],'STALE')
        self.assertTrue(all(t['status']=='STALE' for t in context['accepted_history'][0]['claims'][0]['next_targets']))
        before = (self.output/'evidence.jsonl').read_bytes()
        self.run_pipeline()
        self.assertEqual((self.output/'evidence.jsonl').read_bytes(),before)
        self.assertEqual(self.llm.complete.call_count,2)

    def test_native_c3_feedback_omits_joint_targets_and_bridge(self):
        self.plan.update(schema='dace-source-optimization/v3',level='C3')
        result = self.run_pipeline()
        self.assertEqual(result['native_state']['targets'],[])
        context = json.loads(self.llm.complete.call_args.args[0][0]['content'].split('Context:\n',1)[1])
        claim = context['accepted_history'][0]['claims'][0]
        self.assertEqual(claim['bridge'],'UNBRIDGED')
        self.assertNotIn('next_targets',claim)

    def test_optimizer_request_profile_reaches_each_frozen_diagnostic_child(self):
        self.plan['coding_request'] = {'reasoning':{'enabled':False},'response_format':{'type':'json_object'}}
        self.llm.complete.side_effect = None
        self.llm.complete.return_value = ('{"actions":[],"mechanisms":[]}',
            {'cost':.001,'prompt_tokens':3,'completion_tokens':20,'_completion':{'finish_reason':'stop'}})
        self.assertEqual(self.run_pipeline()['status'],'OPTIMIZATION_COMPLETE')
        for call in self.llm.complete.call_args_list:
            self.assertEqual(call.kwargs['request_options'],self.plan['coding_request'])
        self.plan['coding_request']['reasoning']['enabled'] = True
        with self.assertRaisesRegex(ValueError,'changed on resume'):
            self.run_pipeline()
        self.assertEqual(self.llm.complete.call_count,2)

    def test_native_retained_action_survives_empty_proposal_and_mid_round_recovery(self):
        rows = copy.deepcopy(self.rounds)
        for row in rows:
            row['selection'] = row['selection'][:1]
            row['families'] = {row['selection'][0]:'primary'}
        self.plan = make_optimization_plan(self.parent,rows,self.dataset,['fixture-runner'],
            'fixture-model',self.study,{'schema':'fixture'},intervention_cap=1,action_cap=1,level='C5')
        usage = {'cost':.001,'prompt_tokens':10,'completion_tokens':20}
        self.llm.complete.side_effect = [
            (json.dumps(candidate('retained-edit')),usage),
            (json.dumps({'actions':[],'mechanisms':[]}),usage)]
        append = Ledger.append
        def interrupt(ledger,kind,**payload):
            if kind=='OPTIMIZATION_STAGE_FINISHED' and payload['stage']=='round-001/diagnostic':
                raise KeyboardInterrupt('controlled recovery before retained-action scheduling')
            return append(ledger,kind,**payload)
        with patch.object(Ledger,'append',new=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pipeline()
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(result['rounds'][1]['proposal_status'],'NO_CHANGE_PROPOSED')
        self.assertEqual([r['confirmed_actions'] for r in result['rounds']],[1,1])
        self.assertEqual(result['deployed_artifact']['version'],self.parent.version)
        records = list(result['native_state']['records'].values())
        self.assertEqual(len(records),1)
        self.assertEqual(len(records[0]['responses']),2)
        first = {version for task,version in self.calls if task=='c0'}
        second = {version for task,version in self.calls if task=='c1'}
        self.assertEqual(first,second)
        self.assertEqual(len(first),2)
        self.assertEqual(len(self.calls),10)
        self.assertEqual(self.llm.complete.call_count,2)
        call_count = len(self.calls)
        self.run_pipeline()
        self.assertEqual(len(self.calls),call_count)

    def test_native_c0_executes_without_building_mechanism_space(self):
        self.plan.update(schema='dace-source-optimization/v3',level='C0')
        original = self.propose
        def c0(messages,**kwargs):
            text,usage = original(messages,**kwargs)
            payload = json.loads(text)
            payload['mechanisms'] = []
            return json.dumps(payload),usage
        self.llm.complete.side_effect = c0
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(result['native_state']['records'],{})
        context = json.loads(self.llm.complete.call_args.args[0][0]['content'].split('Context:\n',1)[1])
        self.assertTrue(context['accepted_history'])
        self.assertTrue(all(c['feedback_type']=='UTILITY_VERIFICATION'
            for row in context['accepted_history'] for c in row['claims']))
        self.assertTrue(all('mechanism' not in c
            for row in context['accepted_history'] for c in row['claims']))

    def enable_evaluation(self):
        from dace.benchmarks.fingerprint import task_fingerprint
        self.plan.update(schema='dace-source-optimization/v3',level='C5',eval_tasks=['eval'])
        self.plan['task_fingerprints']['eval'] = task_fingerprint(self.dataset/'eval')
        self.plan['total_budget'] += 3

    def test_blind_evaluation_has_separate_budget_and_never_reaches_optimizer(self):
        self.enable_evaluation()
        result = self.run_pipeline()
        self.assertEqual(len(result['evaluations']),3)
        self.assertEqual(result['evaluation_budget_spent'],3)
        self.assertEqual(result['optimization_budget_spent'],154)
        self.assertEqual(result['budget']['spent'],157)
        self.assertTrue(result['execution_usage']['accounting_complete'])
        for args in self.llm.complete.call_args_list:
            self.assertNotIn('"eval"',args.args[0][0]['content'])
        with TaskRegistry(self.study['registry']) as registry:
            with self.assertRaisesRegex(ValueError,'Held-out'):
                registry.disclose('s','run',['eval'],['pass_at_1'])
        calls = len(self.calls)
        self.run_pipeline()
        self.assertEqual(len(self.calls),calls)

    def test_same_parent_can_be_evaluated_at_distinct_fixed_checkpoints(self):
        self.enable_evaluation()
        self.llm.complete.side_effect = None
        self.llm.complete.return_value = ('{"actions":[],"mechanisms":[]}',{})
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_COMPLETE')
        self.assertEqual(len(result['evaluations']),3)
        self.assertEqual(result['evaluation_budget_spent'],3)
        self.assertEqual(len(self.calls),5)

    def test_interrupted_blind_evaluation_does_not_start_optimization_or_retry(self):
        self.enable_evaluation()
        self.runner.side_effect = KeyboardInterrupt('controlled evaluation interruption')
        with self.assertRaises(KeyboardInterrupt):
            self.run_pipeline()
        self.runner.side_effect = self.execute
        result = self.run_pipeline()
        self.assertEqual(result['status'],'OPTIMIZATION_UNRESOLVED')
        self.assertEqual(result['failed_stage'],'evaluation/baseline')
        self.assertEqual(result['budget']['spent'],1)
        self.llm.complete.assert_not_called()
        self.assertEqual(self.runner.call_count,1)

    def test_native_export_reuses_blind_checkpoints_without_calls_or_double_cost(self):
        from dace.source_reporting import write_native_report,native_report
        self.enable_evaluation()
        self.run_pipeline()
        before = {str(p):p.read_bytes() for p in self.output.rglob('evidence.jsonl')}
        calls = len(self.calls)
        report = write_native_report(self.output,self.root/'export')
        self.assertEqual(report['n_blind_checkpoints'],3)
        self.assertEqual([r['optimization_budget'] for r in report['curve']],[0,77,154])
        self.assertEqual(report['evaluation_budget'],3)
        self.assertAlmostEqual(report['execution_usage']['cost_usd'],len(self.calls)*.01)
        self.assertEqual(len(self.calls),calls)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.output.rglob('evidence.jsonl')})
        # Raw verifier drift cannot be hidden behind the checkpoint's summary.
        reward = next((self.output/'evaluation/baseline').rglob('reward.txt'))
        reward.write_text('1',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'official reward'):
            native_report(self.output)


if __name__=='__main__':
    unittest.main()
