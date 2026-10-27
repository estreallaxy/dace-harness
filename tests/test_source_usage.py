from tests.support import test_temp_dir
import tempfile
import unittest
from pathlib import Path

from dace.core import Ledger
from dace.source_usage import summarize_source_usage


class SourceUsageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'evidence.jsonl'
        self.ledger = Ledger(self.path)

    def result(self,cost=.01):
        usage = {'n_input_tokens':100,'n_output_tokens':10}
        if cost is not None:
            usage['cost_usd'] = cost
        self.ledger.append('SOURCE_EXECUTION_RESULT',result={'usage':usage})

    def test_completed_recorded_calls_have_a_known_total(self):
        self.result()
        result = summarize_source_usage([self.path])
        self.assertTrue(result['accounting_complete'])
        self.assertEqual(result['cost_usd'],.01)

    def test_failed_candidate_cannot_hide_behind_known_baseline_cost(self):
        self.result()
        self.ledger.append('SOURCE_EXECUTION_FAILED',error_type='TimeoutExpired')
        result = summarize_source_usage([self.path])
        self.assertFalse(result['accounting_complete'])
        self.assertIsNone(result['cost_usd'])
        self.assertEqual(result['recorded_cost_usd'],.01)

    def test_missing_provider_price_stays_unknown(self):
        self.result(cost=None)
        result = summarize_source_usage([self.path])
        self.assertIsNone(result['cost_usd'])
        self.assertEqual(result['recorded_input_tokens'],100)

    def test_interrupted_charged_unit_without_failure_event_stays_unknown(self):
        self.ledger.append('BUDGET_CHARGED',amount=2,owner='c',unit={'contract_id':'c','task':'t'})
        self.result()
        result = summarize_source_usage([self.path])
        self.assertIsNone(result['cost_usd'])
        self.assertEqual(result['unsettled_task_units'],1)

    def test_committed_observation_settles_its_exact_charged_unit(self):
        self.ledger.append('BUDGET_CHARGED',amount=2,owner='c',unit={'contract_id':'c','task':'t'})
        self.result()
        self.result()
        self.ledger.append('OBSERVATION',contract_id='c',observation={'task':'t'})
        result = summarize_source_usage([self.path])
        self.assertTrue(result['accounting_complete'])
        self.assertEqual(result['cost_usd'],.02)

    def test_same_ledger_cannot_double_the_reported_cost(self):
        self.result()
        with self.assertRaises(ValueError):
            summarize_source_usage([self.path,self.path])

    def test_invalidated_execution_keeps_recorded_cost_but_not_total_claim(self):
        self.result()
        self.ledger.append('SOURCE_EXECUTION_INVALIDATED',reason='runtime drift')
        result = summarize_source_usage([self.path])
        self.assertIsNone(result['cost_usd'])
        self.assertEqual(result['recorded_cost_usd'],.01)


if __name__=='__main__':
    unittest.main()
