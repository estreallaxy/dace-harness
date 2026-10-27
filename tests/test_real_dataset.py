from tests.support import test_temp_dir
"""Real Terminal-Bench 2.0 dataset validation (dataset mode, mock execution).

Skipped automatically when the real dataset directory is absent, so the suite
stays green on machines without the dataset. What is validated here:
  - the adapter loads the REAL dataset (89 task directories with task.toml)
  - pools draw disjoint REAL task IDs with the global EVAL reservation
  - the capacity preflight rejects sample profiles the dataset cannot cover
  - a full evolution cell runs end-to-end against real task IDs (execution
    semantics remain the controlled mock until a live runner is attached;
    the task-ID layer is fully real)
"""
import json
import unittest
from pathlib import Path

from dace.benchmarks import make_benchmark
from dace.benchmarks.base import TaskPool
from dace.components import AblationLevel
from dace.evolution import EvolutionLoop, MethodSpec
from dace.experiments.runner import _preflight_capacity

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / 'datasets' / 'terminal-bench-2'
HAVE_DATASET = DATASET.is_dir() and any(DATASET.glob('*/task.toml'))


@unittest.skipUnless(HAVE_DATASET, 'real Terminal-Bench 2.0 dataset not present')
class RealDatasetTests(unittest.TestCase):
    def test_real_dataset_loads_and_pools_are_disjoint(self):
        bench = make_benchmark('terminalbench2', dataset_path=str(DATASET))
        self.assertTrue(bench.real_dataset)
        self.assertGreaterEqual(len(bench._dataset_ids), 80)
        ids = set(bench._dataset_ids)
        seen = set()
        for role, size, rnd in [('DIAG', 4, 0), ('CONFIRM0', 8, 0),
                                ('SELECT', 8, 0), ('EVAL', 8, 0)]:
            pool = bench.pool(role, size, rnd, 0)
            self.assertIsInstance(pool, TaskPool)
            for tid in pool.ids():
                self.assertIn(tid, ids, 'pool IDs must be REAL dataset tasks')
                self.assertNotIn(tid, seen, 'task roles must be disjoint')
                seen.add(tid)
        # the EVAL pool is reserved globally: no other role may sample it
        ev = bench.pool('EVAL', 8, 0, 0)
        for tid in ev.ids():
            self.assertNotIn(tid, {t for p in [bench.pool('DIAG', 4, 1, 0)]
                                   for t in p.ids()})

    def test_capacity_preflight_rejects_profiles_the_dataset_cannot_cover(self):
        bench = make_benchmark('terminalbench2', dataset_path=str(DATASET))
        huge = MethodSpec('huge', kind='dace', level=AblationLevel.C5, rounds=3,
                          diag_tasks=8, confirm_tasks=8, selection_tasks=8,
                          eval_tasks=8, intervention_budget=160)
        with self.assertRaises(ValueError) as ctx:
            _preflight_capacity(bench, huge)
        self.assertIn('INSUFFICIENT_TASK_CAPACITY', str(ctx.exception))
        small = MethodSpec('small', kind='dace', level=AblationLevel.C5, rounds=1,
                           diag_tasks=4, confirm_tasks=8, selection_tasks=8,
                           eval_tasks=8, intervention_budget=16)
        _preflight_capacity(bench, small)  # must pass

    def test_evolution_cell_runs_on_real_task_ids(self):
        import tempfile
        tmp = tempfile.mkdtemp(dir=test_temp_dir())
        self.addCleanup(_best_effort_remove, tmp)
        bench = make_benchmark('terminalbench2', dataset_path=str(DATASET),
                               scenario_name='source_error')
        spec = MethodSpec('ds-cell', kind='dace', level=AblationLevel.C4, rounds=1,
                          diag_tasks=4, confirm_tasks=8, selection_tasks=8,
                          eval_tasks=8, intervention_budget=16)
        loop = EvolutionLoop(bench, spec, 0, Path(tmp) / 'cell')
        summary = loop.run()
        # the ledger's task IDs are real TB2 task names, and the run completed
        events = [json.loads(line) for line in
                  Path(summary.ledger_path).read_text(encoding='utf-8').splitlines()]
        used = [e['payload']['task'] for e in events if e['kind'] == 'RUN_USED']
        self.assertTrue(used)
        self.assertTrue(all(t in set(bench._dataset_ids) for t in used))
        self.assertEqual(summary.mode, 'mock')  # dataset mode = real IDs, mock exec
        self.assertEqual(len(summary.promotions), 1)
        self.assertEqual(summary.promotions[0]['bridge'], 'COMPENSATION')


def _best_effort_remove(root):
    import os
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


if __name__ == '__main__':
    unittest.main()
