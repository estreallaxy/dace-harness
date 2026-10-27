from tests.support import test_temp_dir
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from dace.agents.policy import SourcePolicy
from dace.benchmarks.base import TaskResult
from dace.benchmarks.fingerprint import task_fingerprint
from dace.core import Prediction
from dace.source_confirmation import make_plan, run_plan
from dace.task_registry import TaskRegistry


class TaskRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset'
        for name in ('a','b','c','d','e'):
            path = self.dataset/name
            path.mkdir(parents=True)
            (path/'task.toml').write_text('schema_version="1.1"',encoding='utf-8')
            (path/'instruction.md').write_text('Distinct task '+name,encoding='utf-8')
        self.path = self.root/'study.sqlite3'
        self.allocation = {'CONFIRM':['a','b'],'DIAG':['c'],'SELECT':['d'],'EVAL':['e']}
        with self.registry() as r:
            r.create_study('s',self.dataset,self.allocation)
            r.create_lineage('s','run-1')

    def registry(self):
        return TaskRegistry(self.path)

    def claim(self,r,tasks=('a',),owner='session-1',lineage='run-1',role='CONFIRM',output=None):
        r.claim('s',lineage,owner,role,tasks,
            {t:task_fingerprint(self.dataset/t) for t in tasks},
            str(output or self.root/owner),'hash-'+owner)

    def test_study_partition_cannot_change_after_freezing(self):
        swapped = dict(self.allocation,CONFIRM=['b','e'],EVAL=['a'])
        with self.registry() as r,self.assertRaisesRegex(ValueError,'immutable'):
            r.create_study('s',self.dataset,swapped)

    def test_new_session_cannot_reuse_reserved_or_failed_task(self):
        with self.registry() as r:
            self.claim(r)
            with self.assertRaisesRegex(ValueError,'already reserved'):
                self.claim(r,owner='another-contract')
            r.mark_started('s','run-1','session-1','a')
            with self.assertRaisesRegex(ValueError,'already reserved'):
                self.claim(r,owner='after-failure')

    def test_exact_resume_is_idempotent_but_output_clone_is_rejected(self):
        with self.registry() as r:
            self.claim(r)
            original = r.audit_events()
            self.claim(r)
            self.assertEqual(original,r.audit_events())
            with self.assertRaisesRegex(ValueError,'cloned'):
                self.claim(r,output=self.root/'cloned-output')

    def test_losing_local_evidence_does_not_allow_same_session_retry(self):
        with self.registry() as r:
            self.claim(r)
            r.mark_started('s','run-1','session-1','a')
            self.claim(r)  # Same owner can recover the plan, but cannot rerun its task.
            with self.assertRaisesRegex(ValueError,'already started'):
                r.mark_started('s','run-1','session-1','a')

    def test_eval_role_cannot_be_used_for_confirmation(self):
        with self.registry() as r,self.assertRaisesRegex(ValueError,'violates frozen study'):
            self.claim(r,tasks=('e',))

    def test_eval_checkpoints_may_repeat_but_never_reach_optimizer(self):
        with self.registry() as r:
            self.claim(r,tasks=('e',),owner='eval-1',role='EVAL')
            r.mark_started('s','run-1','eval-1','e')
            self.claim(r,tasks=('e',),owner='eval-2',role='EVAL')
            with self.assertRaisesRegex(ValueError,'Held-out'):
                r.disclose('s','run-1',['e'],['mean_reward'])

    def test_disclosure_requires_actual_started_source_units(self):
        with self.registry() as r:
            self.claim(r,tasks=('c',),role='DIAG')
            with self.assertRaisesRegex(ValueError,'no registered execution'):
                r.disclose('s','run-1',['c'],['symptom_rates'])
            r.mark_started('s','run-1','session-1','c')
            r.disclose('s','run-1',['c'],['symptom_rates'])
            self.assertEqual(r.audit_events()[-1]['payload']['tasks'],['c'])

    def test_fork_inherits_parent_exposure_and_reservations(self):
        with self.registry() as r:
            self.claim(r)
            r.mark_started('s','run-1','session-1','a')
            r.disclose('s','run-1',['a'],['mechanism_summary'])
            r.create_lineage('s','fork','run-1')
            with self.assertRaisesRegex(ValueError,'already disclosed'):
                self.claim(r,owner='fork-session',lineage='fork')
            self.claim(r,tasks=('b',),owner='fresh-fork-session',lineage='fork')

    def test_independent_replicates_can_share_frozen_task_partition(self):
        with self.registry() as r:
            self.claim(r)
            r.create_lineage('s','independent-run-2')
            self.claim(r,owner='replicate-session',lineage='independent-run-2')

    def test_new_parent_disclosure_invalidates_childs_pending_reservation(self):
        with self.registry() as r:
            r.create_lineage('s','fork','run-1')
            self.claim(r,lineage='fork',owner='child-session')
            self.claim(r,owner='parent-session')
            r.mark_started('s','run-1','parent-session','a')
            r.disclose('s','run-1',['a'],['feedback'])
            with self.assertRaisesRegex(ValueError,'disclosed after'):
                r.mark_started('s','fork','child-session','a')

    def test_concurrent_sessions_cannot_both_acquire_same_unit(self):
        barrier = threading.Barrier(2)
        def reserve(owner):
            with self.registry() as r:
                barrier.wait(timeout=10)
                try:
                    self.claim(r,owner=owner)
                    return 'accepted'
                except ValueError:
                    return 'rejected'
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(reserve,owner) for owner in ('race-1','race-2')]
            self.assertEqual(sorted(f.result(timeout=20) for f in futures),['accepted','rejected'])

    def test_renaming_identical_task_does_not_make_independent_unit(self):
        (self.dataset/'b'/'instruction.md').write_bytes((self.dataset/'a'/'instruction.md').read_bytes())
        with self.registry() as r,self.assertRaisesRegex(ValueError,'Duplicate task content'):
            r.create_study('copied',self.dataset,self.allocation)

    def test_task_input_drift_violates_study_before_execution(self):
        (self.dataset/'a'/'instruction.md').write_text('changed',encoding='utf-8')
        with self.registry() as r,self.assertRaisesRegex(ValueError,'fingerprint'):
            self.claim(r)

    def test_event_chain_is_auditable(self):
        with self.registry() as r:
            self.claim(r)
            self.assertEqual(len(r.audit_events()),3)
            r.db.execute('UPDATE events SET payload=? WHERE idx=0',('{}',))
            with self.assertRaisesRegex(ValueError,'integrity'):
                r.audit_events()

    def test_source_session_integrates_registry_before_any_model_execution(self):
        p = Prediction('normalization','raw_response_parse_valid',('model','parser'),
            (('raw_response_parse_valid',False),),(('response_parse_valid',True),),
            (('command_batch_valid',True),),'UNRESOLVED')
        plan = make_plan(SourcePolicy(),SourcePolicy(instruction_suffix='edit'),[p],['a','b'],
            self.dataset,['runner'],'model',study={'registry':str(self.path),'id':'s','lineage':'run-1'})
        def execute(config,task,path,**kwargs):
            return TaskResult(task.task_id,{'raw_response_parse_valid':True,
                'response_parse_valid':True,'command_batch_valid':True},1,
                True if kwargs['patch_keys'] else None)
        with patch('dace.source_confirmation.LiveExecutor.execute',side_effect=execute) as runner:
            first = run_plan(plan,self.root/'real-session',batch=1)
            self.assertEqual(first['task_isolation'],'study-lineage')
            self.assertEqual(runner.call_count,2)
            second = run_plan(plan,self.root/'real-session')
            self.assertEqual(runner.call_count,4)
            self.assertEqual(second['budget']['spent'],4)
            run_plan(plan,self.root/'real-session')
            self.assertEqual(runner.call_count,4)
        with self.registry() as r:
            started = [e for e in r.audit_events() if e['kind']=='EXECUTION_STARTED']
            self.assertEqual([e['payload']['task'] for e in started],['a','b'])


if __name__ == '__main__':
    unittest.main()
