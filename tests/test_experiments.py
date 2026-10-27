from tests.support import test_temp_dir
import json
import os
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

from dace.benchmarks import make_benchmark
from dace.components import AblationLevel
from dace.core import Ledger
from dace.evolution import EvolutionLoop, MethodSpec
from dace.experiments import (ExperimentConfig, run_e1, run_e2, run_e3, run_e4, run_e5)
from dace.experiments.e5 import _collect_curves, _condition_key, _condition_of
from dace.experiments.config import e3_grid
from dace.experiments.metrics import (discrimination_rate, e1_row,
                                      regression_count, regression_stats,
                                      repair_compensation_confusion)


def cfg(exp, output):
    return ExperimentConfig(exp=exp, seeds=(0,), rounds=2, output=str(output),
                            diag_tasks=6, confirm_tasks=32, selection_tasks=64, eval_tasks=16)


def _walk_remove(root):
    # best-effort cleanup: the environment's bulk-delete guard may raise
    # SystemExit once the per-turn deleted-file count crosses its threshold;
    # cleanup failures must never fail an otherwise passing test
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            try:
                os.remove(os.path.join(dirpath, name))
            except OSError:
                pass
            except BaseException:
                return
        for name in dirnames:
            try:
                os.rmdir(os.path.join(dirpath, name))
            except OSError:
                pass
            except BaseException:
                return
    try:
        os.rmdir(root)
    except OSError:
        pass
    except BaseException:
        pass


class ExperimentTests(unittest.TestCase):
    setUp_cls = None

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(dir=test_temp_dir())
        cls.output = Path(cls.tmpdir) / 'experiments_output'

    @classmethod
    def tearDownClass(cls):
        _walk_remove(cls.tmpdir)

    def test_e1_main_table(self):
        meta = run_e1(cfg('e1', self.output), benchmarks=['terminalbench2'])
        self.assertEqual(len(meta['rows']), 2)
        methods = {r['method'] for r in meta['rows']}
        self.assertEqual(methods, {'LocalityHeuristic', 'FullDACE'})
        for col in ('pass@1_mean', 'improvement_mean', 'cross_run_variance',
                    'regressions', 'localization_accuracy'):
            self.assertIn(col, meta['rows'][0])
        self.assertTrue((self.output / 'e1' / 'e1_main_table.md').exists())
        self.assertTrue((self.output / 'e1' / 'e1_main_table.csv').exists())

    def test_e1_live_mode_fails_closed_without_runner(self):
        c = cfg('e1live', self.output)
        c.mode = 'live'
        with self.assertRaises(ValueError):
            ExperimentConfig(exp='e1live', benchmark=c.benchmark, seeds=c.seeds,
                             rounds=c.rounds, output=str(self.output), mode='live',
                             dataset_path=None, command_template=None, work_dir=None)

    def test_e2_ablation_table(self):
        meta = run_e2(cfg('e2', self.output))
        methods = {r['method'] for r in meta['rows']}
        self.assertEqual(methods, {f'DACE-{l}' for l in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5')} |
                         {'DACE-CONTRIBUTION_ONLY'})
        self.assertTrue((self.output / 'e2' / 'e2_ablation_table.md').exists())
        self.assertTrue((self.output / 'e2' / 'e2_ablation_table.csv').exists())

    def test_e3_sensitivity_grid(self):
        meta = run_e3(cfg('e3', self.output))
        self.assertEqual(len(meta['rows']), 5)
        pairs = {(r['intervention_budget'],r['mechanism_cap']) for r in meta['rows']}
        self.assertEqual(pairs,{('b=low','k=mid'),('b=mid','k=mid'),('b=high','k=mid'),
                                ('b=mid','k=low'),('b=mid','k=high')})
        budgets = {r['intervention_budget'] for r in meta['rows']}
        self.assertEqual(budgets, {'b=low', 'b=mid', 'b=high'})
        self.assertTrue((self.output / 'e3' / 'e3_sensitivity_table.csv').exists())
        self.assertTrue((self.output / 'e3' / 'e3_cost_curve.svg').exists())
        self.assertTrue((self.output / 'e3' / 'e3_mechanism_cap.svg').exists())
        self.assertTrue((self.output / 'e3' / 'e3_intervention_budget.svg').exists())

    def test_e3_varies_only_one_semantic_parameter_per_condition(self):
        grid = e3_grid(cfg('e3-design',self.output))
        middle = next(asdict(spec) for b,k,spec in grid if (b,k)==('b=mid','k=mid'))
        for b,k,spec in grid:
            changed = {key for key,value in asdict(spec).items()
                       if key!='name' and value != middle[key]}
            if (b,k)==('b=mid','k=mid'):
                self.assertEqual(changed,set())
            else:
                self.assertEqual(changed,{'intervention_budget' if k=='k=mid' else 'mechanism_cap'})

    def test_sensitivity_and_correctness_respect_all_configured_seeds(self):
        for name,driver in [('e3',run_e3),('e4',run_e4)]:
            c = cfg(name+'-seeds',self.output)
            c.seeds = (11,23)
            with patch('dace.experiments.'+name+'.run_experiment_cells',
                       side_effect=RuntimeError('stop before execution')) as runner:
                with self.assertRaisesRegex(RuntimeError,'stop before execution'):
                    driver(c)
                self.assertEqual(runner.call_args.kwargs['seeds'],(11,23))

    def test_e4_correctness_table_and_case_study(self):
        meta = run_e4(cfg('e4', self.output))
        scenarios = {r['scenario'] for r in meta['rows']}
        self.assertEqual(scenarios, {'source_error', 'downstream_compensation',
                                     'compound_fault', 'mislocalization'})
        methods = {r['method'] for r in meta['rows']}
        self.assertEqual(methods, {'UncoupledDual', 'FullDACE'})
        self.assertTrue((self.output / 'e4' / 'e4_correctness_table.md').exists())
        case = self.output / 'e4' / 'e4_case_study.md'
        self.assertTrue(case.exists())
        text = case.read_text(encoding='utf-8')
        self.assertIn('Round 0', text)
        self.assertIn('mislocalization', text)

    def test_e5_reuses_checkpoints_and_groups_by_condition(self):
        # Standalone fixtures: do not depend on earlier E1/E2 tests having
        # populated this class's shared output directory with a C3 run.
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            root = Path(tmp)
            for name,method in [('full','FullDACE'),('c3','DACE-C3')]:
                self._write_summary(root,name,method=method,checkpoints=[[10,.5],[20,.75]],
                    spec=self.SPEC,budget_spent=20,
                    rounds=[{'promoted_artifacts':['fixture-'+name],'accepted':True}])
                Ledger(root/name/'evidence.jsonl').append('AGENT_PROPOSAL',fixture=True)
            before = {p:p.read_bytes() for p in root.glob('*/*') if p.is_file()}
            def forbid_new_run():
                self.fail('E5 must reuse evidence without requesting a new model run')
            meta = run_e5(cfg('e5',root),llm_factory=forbid_new_run)
            keys = list(meta['curves'])
            self.assertTrue(keys)
            self.assertTrue(all('|' in k for k in keys))
            self.assertTrue(any('FullDACE' in k for k in keys))
            self.assertTrue(any('UncoupledDual' in k for k in keys))
            self.assertTrue((root/'e5/e5_efficiency_curve.svg').exists())
            self.assertTrue((root/'e5/e5_summary.json').exists())
            self.assertIn('unit_cost_definition',meta)
            for path,content in before.items():
                self.assertEqual(path.read_bytes(),content)

    def test_e5_fails_closed_without_reusable_runs(self):
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as empty:
            with self.assertRaises(ValueError):
                run_e5(cfg('e5empty', empty))

    # ---------- round-2 review: equal-budget comparison ----------

    @staticmethod
    def _write_summary(root, name, *, method='FullDACE', benchmark='terminalbench2',
                       scenario=None, mode='mock', model=None, spec=None,
                       checkpoints=None, rounds=None, budget_spent=0):
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / 'summary.json').write_text(json.dumps({
            'method': method, 'benchmark': benchmark, 'scenario': scenario,
            'mode': mode, 'model': model, 'spec': spec or {},
            'result_kind': 'benchmark-agent' if mode == 'live' else 'mock',
            'checkpoints': checkpoints or [], 'rounds': rounds or [],
            'budget_spent': budget_spent}), encoding='utf-8')

    SPEC = {'rounds': 2, 'diag_tasks': 8, 'confirm_tasks': 64, 'selection_tasks': 256,
            'eval_tasks': 64, 'intervention_budget': 256, 'mechanism_cap': 4}

    def test_e5_grid_uses_last_completed_checkpoint_no_interpolation(self):
        # [P2] seeds are compared on the common cumulative budget grid using
        # each seed's most recent COMPLETED checkpoint at spend <= grid point;
        # no interpolation and no future checkpoints
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            self._write_summary(tmp, 'seedA', spec=self.SPEC,
                                checkpoints=[[10, 0.1], [30, 0.5]])
            self._write_summary(tmp, 'seedB', spec=self.SPEC,
                                checkpoints=[[20, 0.3]])
            meta = run_e5(cfg('e5grid', tmp))
            self.assertEqual(len(meta['curves']), 1)
            key = next(iter(meta['curves']))
            self.assertEqual(meta['curves'][key], [(10, 0.1), (20, 0.2), (30, 0.4)])
            stats = meta['curve_stats'][key]
            self.assertEqual([s['n_seeds'] for s in stats], [1, 2, 2])
            self.assertEqual([s['budget'] for s in stats], [10, 20, 30])
            # variation is reported, not hidden
            self.assertGreater(stats[2]['std'], 0.0)

    def test_e5_unit_cost_counts_artifact_events_not_keys(self):
        # [P2] unit cost = budget / accepted unique winner artifacts: a joint
        # patch promoting 3 keys in ONE event costs one event, not three
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            self._write_summary(
                tmp, 'events', spec=self.SPEC, scenario='s1', budget_spent=100,
                checkpoints=[[10, 0.1]],
                rounds=[{'accepted': True, 'promoted_keys': ['a', 'b'],
                         'promoted_artifacts': ['art1']},
                        {'accepted': True, 'promoted_keys': ['c'],
                         'promoted_artifacts': ['art2']}])
            self._write_summary(
                tmp, 'keys', spec=self.SPEC, scenario='s2', budget_spent=30,
                checkpoints=[[10, 0.1]],
                rounds=[{'accepted': True, 'promoted_keys': ['a', 'b', 'c'],
                         'promoted_artifacts': ['artA']}])
            meta = run_e5(cfg('e5unit', tmp))
            costs = meta['unit_costs']
            self.assertEqual(len(costs), 2)
            values = sorted(costs.values())
            self.assertEqual(values, [30.0, 50.0])
            self.assertIn('joint patches count as one event',
                          meta['unit_cost_definition'])

    def test_e5_groups_by_full_run_identity(self):
        # [P2] mode, model and the frozen sample-size/budget fingerprint are
        # part of the condition: runs differing on any of them never share
        # a curve, and summaries record the identity
        with tempfile.TemporaryDirectory(dir=test_temp_dir()) as tmp:
            self._write_summary(tmp, 'a', spec=self.SPEC, mode='mock',
                                model='m1', checkpoints=[[10, 0.1]])
            self._write_summary(tmp, 'b', spec=self.SPEC, mode='live',
                                model='m1', checkpoints=[[10, 0.2]])
            other_spec = dict(self.SPEC, confirm_tasks=32)
            self._write_summary(tmp, 'c', spec=other_spec, mode='mock',
                                model='m1', checkpoints=[[10, 0.3]])
            self._write_summary(tmp, 'd', spec=self.SPEC, mode='mock',
                                model='m2', checkpoints=[[10, 0.4]])
            series, _ = _collect_curves(Path(tmp))
            self.assertEqual(len(series), 4)
            keys = list(series)
            conds = [_condition_of({'method': 'FullDACE', 'benchmark': 'terminalbench2',
                                    'scenario': None, 'mode': m, 'model': mo,
                                    'spec': s})
                     for m, mo, s in (('mock', 'm1', self.SPEC), ('live', 'm1', self.SPEC),
                                      ('mock', 'm1', other_spec), ('mock', 'm2', self.SPEC))]
            self.assertEqual(sorted(keys), sorted(_condition_key(c) for c in conds))

    # ---------- round-5 review: experiment metrics read DECISION claims ----------

    def _scratch_dir(self):
        """Best-effort temp dir: TemporaryDirectory's strict cleanup trips the
        environment's bulk-delete guard during long suite runs, so cleanup is
        best-effort and must never fail an otherwise passing test."""
        d = tempfile.mkdtemp(dir=test_temp_dir())
        self.addCleanup(_walk_remove, d)
        return Path(d)

    def _run_cell_summary(self, tmp, name, level, scenario, **spec_kw):
        base = dict(kind='dace', level=level, rounds=1, diag_tasks=8,
                    confirm_tasks=64, selection_tasks=256, eval_tasks=8,
                    intervention_budget=128)
        base.update(spec_kw)
        spec = MethodSpec(name, **base)
        bench = make_benchmark('terminalbench2', scenario_name=scenario)
        return EvolutionLoop(bench, spec, 0, Path(tmp) / name).run()

    def test_discrimination_rate_scores_the_ablated_method_not_raw_labels(self):
        # [P1-2] review round-5 reproduction: C2 (bridge view DISABLED) on
        # source_error produces raw labeling entries with SOURCE_REPAIR
        # labels, but the METHOD accepted no bridge conclusion -- its
        # decision view reads UNRESOLVED, so the discrimination metric must
        # score 0.0. The same scenario under C5 (bridge enabled) resolves
        # its own claims and scores above 0. The underlying controller's
        # raw attribution must never credit an ablated method.
        tmp = self._scratch_dir()
        if True:
            c2 = self._run_cell_summary(tmp, 'c2', AblationLevel.C2, 'source_error')
            self.assertTrue(c2.labeling)
            self.assertTrue(any(e['bridge'] in ('SOURCE_REPAIR', 'COMPENSATION')
                                for e in c2.labeling))
            self.assertEqual(c2.features.get('bridge'), False)
            self.assertEqual(discrimination_rate([c2]), 0.0)
            c5 = self._run_cell_summary(tmp, 'c5', AblationLevel.C5, 'source_error')
            self.assertEqual(c5.features.get('bridge'), True)
            self.assertGreater(discrimination_rate([c5]), 0.0)
            # foreign-action discrimination records are not the method's
            # own claim and never enter the discrimination denominator
            for e in c5.labeling:
                self.assertIn('own', e)

    def test_confusion_aligns_predictions_with_promotion_records(self):
        # [P1-2] the repair/compensation confusion aligns predictions with
        # ground truth through the per-promotion attribution records
        # (mechanism/scope-matched, ablation-filtered), never by flattening
        # raw labeling across artifacts and scopes. On source_error with a
        # confirm budget that covers only the adapter_cast contract, the
        # bridged C4 method promotes it AND identifies it as a compensation;
        # the uncoupled C3 method cannot demote it and honestly mislabels
        # the deployed compensation as a repair.
        tmp = self._scratch_dir()
        if True:
            c4 = self._run_cell_summary(tmp, 'conf4', AblationLevel.C4, 'source_error')
            conf4 = repair_compensation_confusion([c4])
            self.assertEqual(conf4['compensation_identified'], 1)
            self.assertEqual(conf4['compensation_mislabeled_repair'], 0)
            c3 = self._run_cell_summary(tmp, 'conf3', AblationLevel.C3, 'source_error')
            conf3 = repair_compensation_confusion([c3])
            # the uncoupled view cannot tell repair from compensation: the
            # honest ablation measurement is a mislabeled compensation
            self.assertEqual(conf3['compensation_mislabeled_repair'], 1)
            self.assertEqual(conf3['compensation_identified'], 0)

    def test_regression_stats_split_by_evidence_and_winner_only(self):
        # [P2-3] review round-5 reproduction semantics: a negative paired
        # LOWER bound is "non-inferiority not yet established", never an
        # observed regression (small samples, zero change and even positive
        # true effects can produce one); only a direction-matched UPPER
        # bound below zero confirms a regression, and only DEPLOYED winners
        # are ever counted -- rejected candidates constrained nothing
        tmp = self._scratch_dir()
        if True:
            ledger = Ledger(Path(tmp) / 'evidence.jsonl')
            winner, loser = 'W-version', 'L-version'
            ledger.append('PROMOTION', round=0, winner=winner, results={
                winner: {'eligible': True, 'utility_lower': 0.4, 'regressions': {
                    'f_confirmed': -0.6, 'f_decline': -0.5, 'f_unproven': -0.3},
                    'family_details': {
                        # upper < 0: statistically CONFIRMED regression
                        'f_confirmed': {'mean': -0.4, 'lower': -0.6, 'upper': -0.2},
                        # mean < 0 but upper >= 0: observed decline only
                        'f_decline': {'mean': -0.1, 'lower': -0.5, 'upper': 0.3},
                        # lb < 0 only: non-inferiority not established
                        'f_unproven': {'mean': 0.05, 'lower': -0.3, 'upper': 0.4}}},
                loser: {'eligible': False, 'utility_lower': -0.2, 'regressions': {
                    'f_rejected': -0.9},
                    # a rejected candidate's confirmed-looking bound must
                    # never count: nothing of it was deployed
                    'family_details': {
                        'f_rejected': {'mean': -0.8, 'lower': -0.9, 'upper': -0.7}}}})
            # a promotion event with no winner contributes nothing
            ledger.append('PROMOTION', round=1, winner=None, results={
                loser: {'eligible': False, 'utility_lower': -0.1, 'regressions': {
                    'f_orphan': -0.5},
                    'family_details': {
                        'f_orphan': {'mean': -0.3, 'lower': -0.5, 'upper': -0.1}}}})
            import types
            fake = types.SimpleNamespace(ledger_path=str(ledger.path), final_eval=0.5,
                                         baseline_eval=0.5, budget_spent=100,
                                         localization={'accuracy': 1.0}, rounds=[])
            stats = regression_stats([fake])
            self.assertEqual(stats, {'confirmed': 1, 'observed_decline': 1,
                                     'noninferiority_unproven': 1})
            self.assertEqual(regression_count([fake]), 1)
            row = e1_row('M', [fake])
            self.assertEqual(row['regressions'], 1)
            self.assertEqual(row['regressions_observed_decline'], 1)
            self.assertEqual(row['regressions_noninferiority_unproven'], 1)

    def test_no_promotion_run_reports_zero_regressions(self):
        # [P2-3] review round-5 reproduction: a C2 run with no promotion, no
        # deployment and no performance change must report ZERO regressions
        # (previously reported 2 from negative lower bounds of rejected
        # candidates). selection_tasks=1 keeps the paired bound uninformative
        # so nothing clears the promotion gate.
        tmp = self._scratch_dir()
        if True:
            s = self._run_cell_summary(tmp, 'noreg', AblationLevel.C2, 'source_error',
                                       selection_tasks=1)
            self.assertFalse(any(r.accepted for r in s.rounds))
            self.assertEqual(s.baseline_eval, s.final_eval)
            stats = regression_stats([s])
            self.assertEqual(stats, {'confirmed': 0, 'observed_decline': 0,
                                     'noninferiority_unproven': 0})
            self.assertEqual(e1_row('C2', [s])['regressions'], 0)

    def test_e2_table_reports_metric_split_for_ablations(self):
        # final-table assertion (not just view fields): the E2 table carries
        # the regression evidence split, and every confusion cell reflects
        # the method-accepted attribution labels. Fresh output root: run
        # directories must be empty at start (repeated suite runs).
        out = self._scratch_dir()
        meta = run_e2(cfg('e2r5', out))
        row = meta['rows'][0]
        for col in ('regressions', 'regressions_observed_decline',
                    'regressions_noninferiority_unproven'):
            self.assertIn(col, row)
        text = (out / 'e2' / 'e2_ablation_table.csv').read_text(encoding='utf-8')
        self.assertIn('regressions_noninferiority_unproven', text)

    def test_e1_table_carries_regression_evidence_split(self):
        out = self._scratch_dir()
        run_e1(cfg('e1r5', out), benchmarks=['terminalbench2'])
        text = (out / 'e1' / 'e1_main_table.csv').read_text(encoding='utf-8')
        self.assertIn('regressions_observed_decline', text)
        self.assertIn('regressions_noninferiority_unproven', text)


if __name__ == '__main__':
    unittest.main()
