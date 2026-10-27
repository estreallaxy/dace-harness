import unittest

from dace.harness import BASELINE_CONFIG, Fault, SimulatedEnvironment, Task
from dace.components import scenario


def task(tid, magnitude, cacheable):
    return Task(tid, 'primary', {'magnitude': magnitude, 'cacheable': cacheable})


class HarnessSemanticsTests(unittest.TestCase):
    def test_planner_type_error_breaks_everything(self):
        env = scenario('source_error')
        t = task('t', 3, False)
        boundary, utility = env.execute(BASELINE_CONFIG, t)
        self.assertFalse(boundary['planner_valid'])
        self.assertFalse(boundary['adapter_valid'])
        self.assertEqual(utility, 0.0)

    def test_adapter_cast_compensates_planner_defect(self):
        env = scenario('source_error')
        t = task('t', 3, False)
        boundary, utility = env.execute({'adapter_cast': True}, t)
        # downstream recovered but the planner defect persists -> compensation
        self.assertFalse(boundary['planner_valid'])
        self.assertTrue(boundary['tool_success'])
        self.assertEqual(utility, 1.0)

    def test_planner_cast_is_source_repair(self):
        env = scenario('source_error')
        t = task('t', 3, False)
        boundary, utility = env.execute({'planner_cast': True}, t)
        self.assertTrue(boundary['planner_valid'])
        self.assertEqual(utility, 1.0)

    def test_stale_cache_poisons_only_cacheable_tasks(self):
        env = scenario('mislocalization')
        affected = task('a', 3, True)     # applies (scope 1.0) and cacheable
        clean = task('b', 3, False)
        b1, u1 = env.execute(BASELINE_CONFIG, affected)
        self.assertFalse(b1['cache_fresh'])
        self.assertFalse(b1['adapter_valid'])
        self.assertEqual(u1, 0.0)
        b2, u2 = env.execute(BASELINE_CONFIG, clean)
        self.assertEqual(u2, 1.0)

    def test_adapter_cast_compensates_stale_cache(self):
        env = scenario('mislocalization')
        affected = task('a', 3, True)
        boundary, utility = env.execute({'adapter_cast': True}, affected)
        self.assertFalse(boundary['cache_fresh'])   # root cause unresolved
        self.assertTrue(boundary['adapter_valid'])
        self.assertEqual(utility, 1.0)

    def test_compound_fault_needs_both_repairs(self):
        env = scenario('compound_fault')
        t = task('t', 3, False)  # tool fault applies to half the magnitudes
        _, u_planner = env.execute({'planner_cast': True}, t)
        _, u_both = env.execute({'planner_cast': True, 'tool_strict': True}, t)
        self.assertLess(u_planner, 1.0)
        self.assertEqual(u_both, 1.0)

    def test_naive_localization_blames_adapter_for_propagated_fault(self):
        env = scenario('mislocalization')
        traces = []
        for i in range(8):
            t = task(f't{i}', i, True)
            boundary, utility = env.execute(BASELINE_CONFIG, t)
            traces.append({'task': t.task_id, 'boundary': boundary, 'score': utility})
        summary = env.summarize_traces(traces)
        self.assertEqual(summary['naive_localization'], 'adapter')

    def test_fault_validation(self):
        with self.assertRaises(ValueError):
            SimulatedEnvironment([Fault('PLANNER_TYPE_ERROR', 'planner', 'adapter_cast')])


if __name__ == '__main__':
    unittest.main()
