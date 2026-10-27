import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dace.agents.policy import SourcePolicy
from dace.core import Artifact
from dace.native_experiments import NativeProfile,native_conditions,prepare_native_suite,run_native_suite,suite_design
from dace.task_registry import TaskRegistry
from tests.support import test_temp_dir


class NativeExperimentPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'dataset'
        for i in range(12):
            path = self.dataset/str(i)
            path.mkdir(parents=True)
            (path/'task.toml').write_text('schema_version="1.1"\n[metadata]\ncategory="fixture"\n',encoding='utf-8')
            (path/'instruction.md').write_text('Distinct fixture '+str(i),encoding='utf-8')
        self.profile = NativeProfile(rounds=1,diagnostic_tasks=1,confirmation_tasks=1,
            selection_tasks=2,evaluation_tasks=2,action_cap=4)
        self.parent = Artifact.create(SourcePolicy().content,kind='BASELINE')
        for name in ('dace.native_experiments.verify_runtime_lock',
                     'dace.source_optimization.verify_runtime_lock',
                     'dace.runtime_lock.verify_runtime_lock'):
            mock = patch(name,return_value='fixture')
            mock.start()
            self.addCleanup(mock.stop)
        self.out = self.root/'suite'

    def prepare(self,**kwargs):
        return prepare_native_suite(self.parent,self.dataset,['fixture-runner'],'fixture-model',
            {'schema':'fixture'},self.out,profile=self.profile,**kwargs)

    def test_e3_is_five_unique_conditions_with_shared_middle(self):
        conditions = native_conditions(('e3',))
        self.assertEqual(len(conditions),5)
        center = next(c for c in conditions if c['specification']['intervention_cap']==2
                      and c['specification']['mechanism_cap']==4)
        self.assertEqual(len(center['labels']),2)
        for condition in conditions:
            spec = condition['specification']
            self.assertTrue(spec['intervention_cap']==2 or spec['mechanism_cap']==4)

    def test_e2_and_e3_reuse_equivalent_levels_and_attribution_aliases(self):
        conditions = native_conditions()
        self.assertEqual(len(conditions),11)
        c2 = next(c for c in conditions if c['specification']['level']=='C2')
        self.assertIn('E2:Failure-only',c2['labels'])
        c5 = next(c for c in conditions if c['specification']=={
            'level':'C5','intervention_cap':2,'mechanism_cap':4})
        self.assertIn('E2:C5',c5['labels'])
        self.assertIn('E3:intervention=2',c5['labels'])

    def test_freezes_unique_runs_complete_roles_and_common_eval(self):
        manifest = self.prepare(replicates=(0,1))
        self.assertEqual(len(manifest['entries']),22)
        outputs = set()
        eval_sets = set()
        for entry in manifest['entries']:
            plan = json.loads((self.out/entry['plan']).read_text(encoding='utf-8'))
            self.assertEqual(plan['schema'],'dace-source-optimization/v3')
            self.assertEqual(plan['replicate_id'],entry['replicate'])
            outputs.add(entry['run'])
            eval_sets.add(tuple(plan['eval_tasks']))
            self.assertTrue(all(v=='fixture' for row in plan['rounds'] for v in row['families'].values()))
        self.assertEqual(len(outputs),22)
        self.assertEqual(len(eval_sets),1)
        with TaskRegistry(self.out/'study.sqlite3') as registry:
            self.assertEqual(len(registry.protocol('native-e2-e3')['tasks']),12)
            self.assertFalse(any(e['kind']=='EXECUTION_STARTED' for e in registry.audit_events()))

    def test_preparation_is_idempotent_without_touching_plan_files(self):
        expected = self.prepare(replicates=(0,))
        before = {p.name:p.read_bytes() for p in (self.out/'plans').iterdir()}
        self.assertEqual(self.prepare(replicates=(0,)),expected)
        self.assertEqual(before,{p.name:p.read_bytes() for p in (self.out/'plans').iterdir()})

    def test_suite_freezes_the_same_optimizer_request_controls_in_every_condition(self):
        self.profile = replace(self.profile,coding_request={'reasoning':{'enabled':False},
            'response_format':{'type':'json_object'}})
        manifest = self.prepare(replicates=(0,))
        for entry in manifest['entries']:
            plan = json.loads((self.out/entry['plan']).read_text(encoding='utf-8'))
            self.assertEqual(plan['coding_request'],self.profile.coding_request)

    def test_capacity_failure_occurs_before_output_or_registry_creation(self):
        self.profile = replace(self.profile,rounds=10)
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT_DISTINCT_TASKS'):
            self.prepare(replicates=(0,))
        self.assertFalse(self.out.exists())

    def test_changed_existing_protocol_is_not_overwritten(self):
        self.prepare(replicates=(0,))
        self.profile = replace(self.profile,coding_max_tokens=9000)
        with self.assertRaisesRegex(ValueError,'different frozen protocol'):
            self.prepare(replicates=(0,))

    def test_all_plans_are_validated_before_any_dispatch(self):
        manifest = self.prepare(replicates=(0,))
        last = self.out/manifest['entries'][-1]['plan']
        value = json.loads(last.read_text(encoding='utf-8'))
        value['model'] = 'tampered'
        last.write_text(json.dumps(value),encoding='utf-8')
        with patch('dace.native_experiments.run_optimization') as run:
            with self.assertRaisesRegex(ValueError,'Frozen suite plan changed'):
                run_native_suite(self.out/'suite.json')
            run.assert_not_called()

    def test_only_explicit_runner_dispatches_prepared_cells(self):
        manifest = self.prepare(replicates=(0,),experiments=('e3',))
        with patch('dace.native_experiments.run_optimization',return_value={'status':'fixture'}) as run:
            results = run_native_suite(self.out/'suite.json')
            self.assertEqual(len(results),5)
            self.assertEqual(run.call_count,5)

    def test_manifest_model_mismatch_cannot_dispatch_mixed_model_comparison(self):
        manifest = self.prepare(replicates=(0,),experiments=('e3',))
        manifest['model'] = 'different-model'
        (self.out/'suite.json').write_text(json.dumps(manifest),encoding='utf-8')
        with patch('dace.native_experiments.run_optimization') as run:
            with self.assertRaisesRegex(ValueError,'Frozen suite plan changed'):
                run_native_suite(self.out/'suite.json')
            run.assert_not_called()

    def test_mismatched_action_capacity_and_duplicate_replicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'action cap'):
            suite_design(self.dataset,replace(self.profile,action_cap=1))
        with self.assertRaisesRegex(ValueError,'Replicates'):
            self.prepare(replicates=(0,0))

    def test_absent_category_is_not_silently_grouped_as_primary(self):
        (self.dataset/'0/task.toml').write_text('schema_version="1.1"',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'metadata.category'):
            suite_design(self.dataset,self.profile)
        aggregate = suite_design(self.dataset,replace(self.profile,family_mode='aggregate'))
        self.assertEqual(set(aggregate['families'].values()),{'terminalbench2'})


if __name__=='__main__':
    unittest.main()
