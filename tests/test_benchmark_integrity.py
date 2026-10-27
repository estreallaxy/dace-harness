from tests.support import test_temp_dir
"""Scientific result provenance and real-agent bridge regression checks."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from dace.benchmarks.harbor_runner import collect_trial, validate_request
from dace.benchmarks.live import LiveExecutor
from dace.experiments.e5 import _collect_curves, _fingerprint
from dace.experiments.metrics import localization_accuracy, repair_compensation_confusion


class HarborResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = self.root / 'task-a'
        self.task.mkdir()
        (self.task / 'task.toml').write_text('schema_version="1.1"', encoding='utf-8')
        self.request = {'schema':'dace-runner/v1', 'task_id':'task-a',
                        'task_path':str(self.task), 'model':'deepseek/deepseek-v4.1-flash',
                        'config':{}}
        self.job = self.root / 'job'
        self.trial = self.job / 'task-a__trial'
        (self.trial / 'verifier').mkdir(parents=True)
        (self.trial / 'verifier' / 'reward.txt').write_text('1\n', encoding='utf-8')
        self.data = {'task_name':'task-a', 'trial_name':'task-a__trial',
            'agent_info':{'name':'terminus-2', 'model_info':{'name':self.request['model']}},
            'agent_result':{'n_input_tokens':12, 'n_output_tokens':4, 'cost_usd':0.001},
            'verifier_result':{'rewards':{'reward':1}}, 'exception_info':None}

    def write_result(self):
        (self.trial / 'result.json').write_text(json.dumps(self.data), encoding='utf-8')

    def test_official_reward_and_model_tokens_are_required(self):
        self.write_result()
        _, reward, usage, _ = collect_trial(self.job, self.request)
        self.assertEqual(reward.read_text().strip(), '1')
        self.assertEqual(usage['n_input_tokens'], 12)

    def test_oracle_cannot_be_accepted_as_model_agent(self):
        self.data['agent_info']['name'] = 'oracle'
        self.write_result()
        with self.assertRaises(ValueError):
            collect_trial(self.job, self.request)

    def test_infrastructure_exception_is_unresolved_not_zero_or_one(self):
        self.data['exception_info'] = {'exception_type':'EnvironmentError'}
        self.write_result()
        with self.assertRaises(RuntimeError):
            collect_trial(self.job, self.request)

    def test_missing_model_usage_is_not_a_verified_model_run(self):
        self.data['agent_result'] = {}
        self.write_result()
        with self.assertRaises(ValueError):
            collect_trial(self.job, self.request)

    def test_reward_must_match_trial_report(self):
        self.data['verifier_result']['rewards']['reward'] = 0
        self.write_result()
        with self.assertRaises(ValueError):
            collect_trial(self.job, self.request)

    def test_synthetic_switch_not_silently_ignored_by_real_agent(self):
        validate_request(self.request)
        self.request['config'] = {'planner_cast':True}
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_HARNESS_EDIT'):
            validate_request(self.request)

    def test_missing_official_reward_never_inferred_from_exit_status(self):
        self.write_result()
        (self.trial / 'verifier' / 'reward.txt').unlink()
        with self.assertRaises(ValueError):
            collect_trial(self.job, self.request)

    def test_historical_live_oracle_is_excluded_from_efficiency_results(self):
        old = {'method':'FullDACE', 'benchmark':'terminalbench2', 'mode':'live',
               'checkpoints':[[1,1.0]], 'spec':{'rounds':1}}
        (self.root / 'summary.json').write_text(json.dumps(old), encoding='utf-8')
        self.assertEqual(_collect_curves(self.root), ({}, {}))
        old['result_kind'] = 'controlled-oracle'
        (self.root / 'summary.json').write_text(json.dumps(old), encoding='utf-8')
        self.assertEqual(_collect_curves(self.root), ({}, {}))

    def test_different_promotion_policies_cannot_share_efficiency_curve(self):
        a = {'spec':{'rounds':1, 'allow_unobserved_promotion':False}}
        b = {'spec':{'rounds':1, 'allow_unobserved_promotion':True}}
        self.assertNotEqual(_fingerprint(a), _fingerprint(b))

    def test_real_task_without_ground_truth_never_scores_mechanism_accuracy(self):
        summary = SimpleNamespace(ground_truth={'known':False, 'faults':[]},
            localization={'accuracy':1.0}, final_config={'planner_cast':True})
        self.assertIsNone(localization_accuracy(summary))
        confusion = repair_compensation_confusion([summary])
        self.assertEqual(confusion['ground_truth_unavailable'], 1)
        self.assertEqual(confusion['repair_identified'], 0)
        self.assertEqual(confusion['compensation_mislabeled_repair'], 0)


if __name__ == '__main__':
    unittest.main()
