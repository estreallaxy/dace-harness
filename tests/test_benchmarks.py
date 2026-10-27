from tests.support import test_temp_dir
import tempfile
import unittest
from pathlib import Path

from dace.benchmarks import make_benchmark
from dace.benchmarks.terminalbench import TerminalBench2


class BenchmarkPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        self.bench = make_benchmark('terminalbench2')

    def test_pools_disjoint_across_roles(self):
        seen = set()
        for role in ('DIAG', 'CONFIRM0', 'CONFIRM1', 'SELECT', 'EVAL', 'VERIFY'):
            pool = self.bench.pool(role, 10, 0, 42)
            ids = set(pool.ids())
            self.assertFalse(ids & seen, f'{role} overlaps previous pools')
            seen |= ids

    def test_confirm_pools_disjoint_across_contracts_and_rounds(self):
        seen = set()
        for r in range(2):
            for c in range(3):
                ids = set(self.bench.pool(f'CONFIRM{c}', 8, r, 42).ids())
                self.assertFalse(ids & seen)
                seen |= ids

    def test_eval_pool_fixed_across_rounds(self):
        e0 = self.bench.pool('EVAL', 12, 0, 7).ids()
        e1 = self.bench.pool('EVAL', 12, 3, 7).ids()
        self.assertEqual(e0, e1)

    def test_execute_returns_bounded_result(self):
        pool = self.bench.pool('DIAG', 4, 0, 0)
        res = self.bench.execute({'planner_cast': True}, pool.tasks[0])
        self.assertIn(res.utility, (0.0, 1.0))
        self.assertIn('tool_success', res.boundary)

    def test_mock_activation_is_an_execution_fact(self):
        # review round-2 reproduction: activation is reported by the
        # executor for the exact edit (patch_keys), never inferred from a
        # configuration diff. An edit whose fault-conditioned branch never
        # runs on the task is NOT activated.
        pool = self.bench.pool('DIAG', 8, 0, 0)
        task = pool.tasks[0]
        # planner fault applies to every task in benchmark_mix
        res = self.bench.execute({'planner_cast': True}, task, patch_keys=('planner_cast',))
        self.assertIs(res.activated, True)
        # a key whose branch never runs (no verifier drift on this task's
        # magnitude, or no edit exercised) is not activated
        res_inert = self.bench.execute({'verifier_recheck': True}, task,
                                       patch_keys=('verifier_recheck',))
        if not any(f.kind == 'VERIFIER_DRIFT' and f.applies(task) for f in self.bench.env.faults):
            self.assertIs(res_inert.activated, False)
        # no patch_keys -> unknown, stays None
        res_unknown = self.bench.execute({'planner_cast': True}, task)
        self.assertIsNone(res_unknown.activated)

    def test_missing_boundary_stays_unobservable(self):
        # review round-2 reproduction: a legal TaskResult with an EMPTY
        # boundary (nothing observed) must not crash the diagnostic summary;
        # unobserved fields carry no violation rate instead of being forced
        # to False
        pool = self.bench.pool('DIAG', 4, 0, 0)
        traces = [{'task': t.task_id, 'boundary': {}, 'score': 0.0} for t in pool.tasks]
        summary = self.bench.env.summarize_traces(traces)
        self.assertEqual(summary['symptom_rates'], {})
        self.assertTrue(all(c == 0.0 for c in summary['coverage'].values()))
        # partial observation: rates only over the rows that observed the field
        traces = [{'task': t.task_id, 'boundary': {'planner_valid': False},
                   'score': 0.0} for t in pool.tasks[:2]]
        traces += [{'task': t.task_id, 'boundary': {}, 'score': 0.0}
                   for t in pool.tasks[2:]]
        summary = self.bench.env.summarize_traces(traces)
        self.assertEqual(sorted(summary['symptom_rates']), ['planner_valid'])
        self.assertEqual(summary['symptom_rates']['planner_valid'], 1.0)
        self.assertEqual(summary['coverage']['planner_valid'], 0.5)

    def test_registry_unknown_benchmark(self):
        with self.assertRaises(ValueError):
            make_benchmark('nope')

    def test_dataset_loader(self):
        root = Path(self.tmp.name) / 'loader'
        root.mkdir(parents=True)
        for name in ('task-a', 'task-b', 'task-c', 'task-d', 'task-e', 'task-f'):
            (root / name).mkdir()
            (root / name / 'task.yaml').write_text('id: ' + name, encoding='utf-8')
        (root / 'notatask').mkdir()
        bench = TerminalBench2(dataset_path=str(root))
        self.assertTrue(bench.real_dataset)
        ids = set(bench.pool('DIAG', 3, 0, 1).ids())
        self.assertTrue(ids <= {'task-a', 'task-b', 'task-c', 'task-d', 'task-e', 'task-f'})
        # disjoint sampling across roles
        ids2 = set(bench.pool('SELECT', 3, 0, 1).ids())
        self.assertFalse(ids & ids2)

    def test_dataset_eval_pool_excluded_from_other_roles(self):
        # review reproduction: evaluation tasks must be consumed globally so no
        # later DIAG/CONFIRM/SELECT pool can pick them (no test leakage)
        root = Path(self.tmp.name) / 'evaldisjoint'
        root.mkdir(parents=True)
        names = [f'task-{i:02d}' for i in range(30)]
        for name in names:
            (root / name).mkdir()
            (root / name / 'task.yaml').write_text('id: ' + name, encoding='utf-8')
        bench = TerminalBench2(dataset_path=str(root))
        eval_ids = set(bench.pool('EVAL', 4, 0, 7).ids())
        self.assertEqual(len(eval_ids), 4)
        for role in ('DIAG', 'CONFIRM0', 'SELECT'):
            other = set(bench.pool(role, 8, 0, 7).ids())
            self.assertFalse(eval_ids & other, f'{role} leaked evaluation tasks')
        # remaining capacity is enforced, not silently short
        with self.assertRaises(ValueError):
            bench.pool('DIAG', 12, 1, 7)  # only 10 unused tasks remain

    def test_live_mode_requires_all_runner_parameters(self):
        # fail closed: no silent fallback to the mock simulator in live mode
        root = Path(self.tmp.name) / 'livecheck'
        (root / 'task-a').mkdir(parents=True)
        (root / 'task-a' / 'task.yaml').write_text('id: task-a', encoding='utf-8')
        with self.assertRaises(ValueError):
            TerminalBench2(mode='live')
        with self.assertRaises(ValueError):
            TerminalBench2(mode='live', dataset_path=str(root), command_template='["tb"]')

    def test_ground_truth(self):
        gt = self.bench.ground_truth()
        self.assertIn('root_components', gt)
        self.assertTrue(gt['root_components'])


import json
import sys
import tempfile
import unittest
from pathlib import Path

from dace.benchmarks import make_benchmark
from dace.benchmarks.terminalbench import TerminalBench2

FAKE_RUNNER = r'''
import json, sys
from pathlib import Path

argv = sys.argv[1:]
req_path = Path(argv[argv.index("--request") + 1])
req = json.loads(req_path.read_text(encoding="utf-8"))
resp_path = Path(argv[argv.index("--response") + 1])
work = req_path.parent
policy = json.loads((Path(__file__).parent / "policy.json").read_text(encoding="utf-8"))
(work / "reward.json").write_text(json.dumps({"reward": policy.get("reward", 1)}), encoding="utf-8")
trace = {"task_id": req["task_id"],
         "boundary": {"planner_valid": True, "cache_fresh": True, "adapter_valid": True,
                      "tool_success": True, "verifier_agrees": True}}
intervention = req.get("intervention")
if intervention is not None:
    if policy.get("echo_identity", True):
        trace["intervention"] = intervention
    if policy.get("omit_events"):
        pass  # no activation_events at all in the trace
    else:
        events = policy.get("activation_events")
        if events is None:
            # default: every patch key of THIS edit fired its hook
            events = {k: True for k in intervention["patch_keys"]}
        trace["activation_events"] = events
(work / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
resp_path.write_text(json.dumps({"schema": "dace-runner/v1", "task_id": req["task_id"],
                                 "model": req["model"], "verifier_path": "reward.json",
                                 "trace_path": "trace.json",
                                 "provenance": policy.get("provenance", {
                                     "execution_kind":"harbor-agent", "oracle_access":False,
                                     "benchmark_score_eligible":True, "reward_source":"official-verifier"})}), encoding="utf-8")
'''


class LiveActivationIdentityTests(unittest.TestCase):
    """[P1] review round-3: live activation events must be bound to THIS
    edit's identity (patch keys, parent/candidate versions, frozen action
    id), validated against the trace's echo -- an old edit's hook firing is
    never credited to a new edit, and missing/mismatched identity stays
    unresolved."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=test_temp_dir())
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.runner_dir = root / 'runner'
        self.runner_dir.mkdir()
        (self.runner_dir / 'fake_runner.py').write_text(FAKE_RUNNER, encoding='utf-8')
        self.dataset = root / 'dataset'
        for name in ('task-a', 'task-b'):
            (self.dataset / name).mkdir(parents=True)
            (self.dataset / name / 'task.yaml').write_text('id: ' + name, encoding='utf-8')
        self.work = root / 'work'
        self.bench = TerminalBench2(
            dataset_path=str(self.dataset), mode='live',
            command_template=json.dumps([sys.executable,
                                         str(self.runner_dir / 'fake_runner.py')]),
            work_dir=str(self.work))

    def _policy(self, **policy):
        (self.runner_dir / 'policy.json').write_text(json.dumps(policy), encoding='utf-8')

    def _task(self):
        return self.bench.pool('DIAG', 1, 0, 0).tasks[0]

    def test_activation_true_only_with_full_identity_echo(self):
        self._policy()
        res = self.bench.execute({'planner_refresh': True}, self._task(),
                                 patch_keys=('planner_refresh',),
                                 parent_version='parent-v', candidate_version='cand-v',
                                 action_id='action-1')
        self.assertIs(res.activated, True)

    def test_oracle_result_rejected_even_when_reward_and_identity_match(self):
        self._policy(provenance={'execution_kind':'controlled-oracle',
                                 'oracle_access':True, 'benchmark_score_eligible':False,
                                 'reward_source':'official-verifier'})
        with self.assertRaisesRegex(ValueError, 'NON_BENCHMARK_RESULT'):
            self.bench.execute({}, self._task())

    def test_unidentified_result_rejected(self):
        self._policy(provenance={})
        with self.assertRaisesRegex(ValueError, 'NON_BENCHMARK_RESULT'):
            self.bench.execute({}, self._task())

    def test_old_edit_firing_is_not_new_edit_activation(self):
        # the trace's hooks report an OLD edit (planner_cast) firing while
        # the edit under test (planner_refresh) never did: activation is
        # False -- the old edit's True is never credited to this edit
        self._policy(activation_events={'planner_cast': True, 'planner_refresh': False})
        res = self.bench.execute({'planner_refresh': True}, self._task(),
                                 patch_keys=('planner_refresh',),
                                 parent_version='parent-v', candidate_version='cand-v',
                                 action_id='action-1')
        self.assertIs(res.activated, False)

    def test_missing_activation_events_stay_unresolved(self):
        # the trace carries no activation_events at all -> unresolved
        self._policy(omit_events=True)
        res = self.bench.execute({'planner_refresh': True}, self._task(),
                                 patch_keys=('planner_refresh',),
                                 parent_version='parent-v', candidate_version='cand-v',
                                 action_id='action-1')
        self.assertIsNone(res.activated)
        # an event map missing the patch key is equally incomplete: unresolved
        self._policy(activation_events={})
        res = self.bench.execute({'planner_refresh': True}, self._task(),
                                 patch_keys=('planner_refresh',),
                                 parent_version='parent-v', candidate_version='cand-v',
                                 action_id='action-1')
        self.assertIsNone(res.activated)

    def test_identity_mismatch_stays_unresolved(self):
        # the trace does not echo THIS edit's identity (stale/foreign echo):
        # activation cannot be attributed and stays unresolved
        self._policy(echo_identity=False)
        res = self.bench.execute({'planner_refresh': True}, self._task(),
                                 patch_keys=('planner_refresh',),
                                 parent_version='parent-v', candidate_version='cand-v',
                                 action_id='action-1')
        self.assertIsNone(res.activated)

    def test_non_intervention_run_reports_no_activation(self):
        # diagnostic/eval runs carry no edit identity: no activation fact
        self._policy()
        res = self.bench.execute({'planner_refresh': True}, self._task())
        self.assertIsNone(res.activated)

    def test_joint_patch_needs_every_key_event(self):
        # a joint edit must activate on EVERY patch key; one missing key
        # leaves the whole edit's activation unresolved
        self._policy(activation_events={'planner_cast': True})
        res = self.bench.execute({'planner_cast': True, 'tool_strict': True}, self._task(),
                                 patch_keys=('planner_cast', 'tool_strict'),
                                 parent_version='parent-v', candidate_version='cand-v',
                                 action_id='action-2')
        self.assertIsNone(res.activated)


if __name__ == '__main__':
    unittest.main()
