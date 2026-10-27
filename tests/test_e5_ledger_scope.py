from tests.support import test_temp_dir
"""E5 overheads must use the same scientifically eligible runs as curves."""
import json
import tempfile
import unittest
from pathlib import Path

from dace.core import Ledger
from dace.experiments.e5 import _cost_decomposition


class E5LedgerScopeTests(unittest.TestCase):
    def test_oracle_and_unassociated_ledgers_cannot_contaminate_overheads(self):
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            root = Path(tmp)
            for name,summary in [
                ('accepted',{'method':'FullDACE','mode':'mock','checkpoints':[[1,.5]]}),
                ('oracle',{'method':'FullDACE','mode':'live','result_kind':'controlled-oracle',
                           'checkpoints':[[1,1.0]]}),
                ('unassociated',None)]:
                path = root/name
                ledger = Ledger(path/'evidence.jsonl')
                ledger.append('AGENT_PROPOSAL',usage={'cost':1})
                ledger.append('CONTRACT_FROZEN',fixture=True)
                if summary is not None:
                    (path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
            totals = _cost_decomposition(root)
            self.assertEqual(totals.get('agent_proposals'),1)
            self.assertEqual(totals.get('confirm_contracts'),1)

    def test_missing_associated_ledger_is_explicit_not_zero_overhead(self):
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            root = Path(tmp)
            (root/'summary.json').write_text(json.dumps({'method':'FullDACE','mode':'mock',
                'checkpoints':[[1,.5]]}),encoding='utf-8')
            totals = _cost_decomposition(root)
            self.assertEqual(totals['ledgers_missing'],1)
            self.assertEqual(totals['ledgers_available'],0)

    def test_corrupt_eligible_ledger_cannot_silently_contribute_to_report(self):
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            root = Path(tmp)
            (root/'summary.json').write_text(json.dumps({'method':'FullDACE','mode':'mock',
                'checkpoints':[[1,.5]]}),encoding='utf-8')
            ledger = Ledger(root/'evidence.jsonl')
            ledger.append('AGENT_PROPOSAL',usage={'cost':1})
            event = json.loads(ledger.path.read_text(encoding='utf-8'))
            event['payload']['usage']['cost'] = 999
            ledger.path.write_text(json.dumps(event)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'integrity'):
                _cost_decomposition(root)


if __name__=='__main__':
    unittest.main()
