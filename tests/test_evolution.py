from tests.support import test_temp_dir
import json
import tempfile
import unittest
from pathlib import Path

from dace.benchmarks import make_benchmark
from dace.benchmarks.base import TaskResult
from dace.benchmarks.terminalbench import TerminalBench2
from dace.components import AblationLevel, Mechanism
from dace.core import Artifact
from dace.evolution import EvolutionLoop, MethodSpec, RoundRecord


def spec(name, **kw):
    base = dict(rounds=2, diag_tasks=8, confirm_tasks=64, selection_tasks=256,
                eval_tasks=24, intervention_budget=256)
    base.update(kw)
    return MethodSpec(name, **base)


class _ForcedActivationBenchmark(TerminalBench2):
    """Executor stub: reports a forced activation fact regardless of config."""
    forced = False

    def execute(self, config, task, patch_keys=None, **_identity):
        res = super().execute(config, task, patch_keys=patch_keys)
        return TaskResult(res.task_id, res.boundary, res.utility, activated=self.forced)


class _FailingCandidateBenchmark(TerminalBench2):
    """Executor stub: candidate executions (patch_keys given) raise once
    `failing` is set; baseline executions are unaffected."""
    failing = False

    def execute(self, config, task, patch_keys=None, **_identity):
        if patch_keys is not None and self.failing:
            raise RuntimeError('confirmatory execution failed')
        return super().execute(config, task, patch_keys=patch_keys, **_identity)


class _NoBoundaryBenchmark(TerminalBench2):
    """Executor stub: legally observes NO boundary field at all."""

    def execute(self, config, task, patch_keys=None, **_identity):
        res = super().execute(config, task, patch_keys=patch_keys)
        return TaskResult(res.task_id, {}, res.utility, activated=res.activated)


class EvolutionLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(dir=test_temp_dir())
        self.addCleanup(self._cleanup_tmp)

    def _cleanup_tmp(self):
        # best-effort cleanup: the environment's bulk-delete guard may raise
        # SystemExit once the per-turn deleted-file count crosses its
        # threshold; cleanup failures must never fail an otherwise passing
        # test (leftover dirs are removed out-of-band)
        import os
        root = self.tmpdir
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

    def loop(self, spec, scenario=None, seed=0):
        bench = make_benchmark('terminalbench2', scenario_name=scenario)
        return EvolutionLoop(bench, spec, seed, Path(self.tmpdir) / f'{spec.name}-{seed}').run()

    def loop_only(self, spec, scenario=None, seed=0):
        bench = make_benchmark('terminalbench2', scenario_name=scenario)
        return EvolutionLoop(bench, spec, seed, Path(self.tmpdir) / f'{spec.name}-{seed}')

    def events(self, summary):
        text = Path(summary.ledger_path).read_text(encoding='utf-8')
        return [json.loads(line) for line in text.splitlines()]

    @staticmethod
    def read_events(path):
        text = Path(path).read_text(encoding='utf-8')
        return [json.loads(line) for line in text.splitlines()]

    def test_c5_repairs_source_error(self):
        s = self.loop(spec('c5', kind='dace', level=AblationLevel.C5), 'source_error')
        self.assertEqual(s.final_eval, 1.0)
        self.assertEqual(s.localization['truth'], ['planner'])
        self.assertEqual(s.localization['predicted'], ['planner'])
        bridges = [b['bridge'] for r in s.rounds for b in r.bridges]
        self.assertIn('SOURCE_REPAIR', bridges)

    def test_c5_demotes_compensation_and_relocalizes(self):
        # intervention_budget=512 -> all four candidate actions earn contracts
        # this round, so the demotion assertions do not depend on the
        # discriminative scheduling order. Round 0's blind selection cannot
        # distinguish the root repair from the compensation (identical paired
        # utility), so the frozen tie-break may deploy the COMPENSATION --
        # reported as such, never as a source repair. Round 1 then
        # relocalizes: the upstream planner root is confirmed (scoped view)
        # and the final report names the true root while the root repair
        # itself earns no redundant deployment over the compensation.
        s = self.loop(spec('c5', kind='dace', level=AblationLevel.C5,
                           intervention_budget=512), 'mislocalization')
        self.assertEqual(s.localization['truth'], ['planner'])
        self.assertEqual(s.localization['predicted'], ['planner'])
        bridges = [b['bridge'] for r in s.rounds for b in r.bridges]
        self.assertIn('COMPENSATION', bridges)
        # whatever won the round-0 tie, the promotion attribution is honest:
        # a compensation is reported as COMPENSATION, a repair as SOURCE_REPAIR
        self.assertTrue(s.promotions)
        self.assertIn(s.promotions[0]['bridge'], ('SOURCE_REPAIR', 'COMPENSATION'))
        if s.final_config.get('adapter_cast'):
            self.assertEqual(s.promotions[0]['bridge'], 'COMPENSATION')
        else:
            self.assertEqual(s.final_config.get('planner_refresh'), True)
            self.assertEqual(s.promotions[0]['bridge'], 'SOURCE_REPAIR')
        # utility is fully restored either way
        self.assertEqual(s.final_eval, 1.0)

    def test_c3_uncoupled_never_demotes_adapter(self):
        s = self.loop(spec('c3', kind='dace', level=AblationLevel.C3), 'mislocalization')
        # the uncoupled dual confirms the adapter mechanism via its own failure
        # predicates within the round and cannot demote it without the bridge;
        # promotion resets unconfirmed claims, so the assertion is on the
        # within-round evidence, not on inherited states
        adapter_claims = [b for r in s.rounds for b in r.bridges
                          if b['mechanism'] == 'm_adapter_garble']
        self.assertTrue(adapter_claims)
        self.assertTrue(all(c['bridge'] != 'COMPENSATION' for c in adapter_claims))
        self.assertTrue(any(c['failure'] == 'CONSISTENT' for c in adapter_claims))

    def test_c0_reaches_utility_without_attribution(self):
        s = self.loop(spec('c0', kind='dace', level=AblationLevel.C0), 'source_error')
        self.assertEqual(s.final_eval, 1.0)
        self.assertEqual(s.labeling, [])

    def test_contribution_only_control_runs(self):
        s = self.loop(spec('contrib', kind='dace', level=AblationLevel.CONTRIBUTION_ONLY),
                      'source_error')
        self.assertEqual(s.final_eval, 1.0)

    def test_locality_baseline_mislocalizes(self):
        s = self.loop(spec('lh', kind='locality'), 'mislocalization')
        # locality-first: the baseline patches the adapter (compensation) and
        # still reaches full utility without ever repairing the root cause
        self.assertEqual(s.final_eval, 1.0)
        self.assertEqual(s.localization['predicted'], ['adapter'])

    def test_locality_progresses_on_mix(self):
        s = self.loop(spec('lhmix', kind='locality', rounds=3), 'benchmark_mix')
        self.assertGreaterEqual(s.final_eval, s.baseline_eval)

    def test_autosaddler_kind_is_reserved_and_fails_closed(self):
        with self.assertRaises(ValueError):
            spec('fake', kind='autosaddler')

    def test_c5_progresses_on_mix(self):
        s = self.loop(spec('c5mix', kind='dace', level=AblationLevel.C5, rounds=3),
                      'benchmark_mix')
        self.assertGreaterEqual(s.final_eval, s.baseline_eval)
        self.assertGreaterEqual(s.final_eval, 0.75)

    def test_masked_retention_never_modifies_current_harness(self):
        # review reproduction: source_error, C5, 1 round, confirm_tasks=16,
        # selection_tasks=1 -> winner=None, no candidate eligible; the retained
        # masked-benefit candidate must NOT replace the parent harness
        loop = self.loop_only(spec('masked', kind='dace', level=AblationLevel.C5,
                                   rounds=1, confirm_tasks=16, selection_tasks=1),
                              'source_error')
        before = dict(loop.current)
        s = loop.run()
        masked = [e for e in self.events(s) if e['kind'] == 'MASKED_BENEFIT_RETAINED']
        self.assertTrue(masked, 'the masked-benefit path must fire in this reproduction')
        # the harness is unchanged; the candidate lives in an isolated lineage
        self.assertEqual(s.final_config, before)
        self.assertTrue(all(not r.accepted for r in s.rounds))
        self.assertTrue(any(r.retained_keys for r in s.rounds))
        for e in masked:
            payload = e['payload']
            self.assertNotEqual(payload['candidate_artifact'], payload['parent_artifact'])
        # and the retained candidate is visible in the summary for later rounds
        self.assertTrue(any('planner_cast' in p for p in s.retained_candidates))

    def test_promotion_requires_independent_selection(self):
        # no utility signal at all: nothing may be promoted, accepted or applied
        s = self.loop(spec('nosig', kind='dace', level=AblationLevel.C5, rounds=1,
                           confirm_tasks=16, selection_tasks=1), 'benchmark_mix')
        self.assertEqual(s.final_config, {k: False for k in s.final_config})
        self.assertTrue(all(not r.accepted for r in s.rounds))
        winners = [e for e in self.events(s)
                   if e['kind'] == 'PROMOTION' and e['payload'].get('winner')]
        self.assertEqual(winners, [])

    def test_c5_probes_are_substantive(self):
        # C5 freezes the competing mechanism's prediction on the SAME action:
        # probes > 0 and the contract discriminates; C4 has none of these
        s5 = self.loop(spec('p5', kind='dace', level=AblationLevel.C5, rounds=1),
                       'source_error')
        s4 = self.loop(spec('p4', kind='dace', level=AblationLevel.C4, rounds=1),
                       'source_error')
        self.assertGreater(sum(r.probes for r in s5.rounds), 0)
        self.assertEqual(sum(r.probes for r in s4.rounds), 0)
        discriminated = [b['discriminated'] for r in s5.rounds for b in r.bridges
                         if b['discriminated']]
        self.assertTrue(discriminated)

    def test_protocol_frozen_before_first_round(self):
        s = self.loop(spec('proto', kind='dace', level=AblationLevel.C5), 'source_error')
        kinds = [e['kind'] for e in self.events(s)]
        self.assertEqual(kinds[0], 'PROTOCOL_FROZEN')
        frozen = self.events(s)[0]['payload']
        self.assertAlmostEqual(frozen['alpha_per_round'], frozen['alpha_total'] / frozen['rounds'])
        self.assertLess(frozen['alpha_per_contract'], frozen['alpha_per_round'])
        self.assertGreater(frozen['promotion_reserved'], 0)

    def test_selection_frozen_before_execution(self):
        s = self.loop(spec('sfz', kind='dace', level=AblationLevel.C5), 'source_error')
        events = self.events(s)
        sel_idx = next(i for i, e in enumerate(events) if e['kind'] == 'SELECTION_FROZEN')
        first_select_run = next(i for i, e in enumerate(events)
                                if e['kind'] == 'RUN_USED'
                                and e['payload']['role'] == 'SELECT')
        self.assertLess(sel_idx, first_select_run)

    def test_winner_artifact_identity_preserved(self):
        loop = self.loop_only(spec('ident', kind='dace', level=AblationLevel.C5, rounds=1),
                              'source_error')
        s = loop.run()
        stale = [e for e in self.events(s) if e['kind'] == 'ANCHORS_STALE']
        self.assertTrue(stale)
        self.assertTrue(all(e['payload']['current'] for e in stale))
        # the promoted root keeps the winner's exact version (lineage intact)
        self.assertEqual(loop.root_artifact.version, stale[-1]['payload']['current'])

    def test_claims_reset_and_archive_follow_modified_keys(self):
        loop = self.loop_only(spec('remap', kind='dace', level=AblationLevel.C5, rounds=1),
                              'source_error')
        s = loop.run()
        applied = {k for r in s.rounds for k in r.promoted_keys}
        for m in s.mechanism_states:
            if m['repair_key'] in applied:
                continue  # archived with labels as the historical record
            # claims are NOT inherited by the promoted version
            self.assertEqual(m['f_status'], 'PROPOSED', m['id'])
            self.assertEqual(m['z_bridge'], 'UNBRIDGED', m['id'])

    def test_intervention_budget_caps_contracts(self):
        s = self.loop(spec('lowb', kind='dace', level=AblationLevel.C5,
                           intervention_budget=64), 'benchmark_mix')
        for r in s.rounds:
            self.assertLessEqual(r.contracts_frozen, 1)

    def test_checkpoints_monotone_spending(self):
        s = self.loop(spec('ck', kind='dace', level=AblationLevel.C5), 'source_error')
        spent = [c[0] for c in s.checkpoints]
        self.assertEqual(spent, sorted(spent))
        self.assertEqual(len(s.checkpoints), len(s.rounds) + 1)

    def test_ledger_written_and_summary_serialized(self):
        s = self.loop(spec('io', kind='dace', level=AblationLevel.C5), 'source_error')
        self.assertTrue(Path(s.ledger_path).exists())
        out = Path(self.tmpdir) / 'io-0'
        data = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(data['method'], 'io')
        self.assertIn('localization', data)
        self.assertIn('scenario', data)
        self.assertIn('retained_candidates', data)
        # run identity (E5 condition grouping): mode, model and frozen spec
        self.assertIn('mode', data)
        self.assertIn('spec', data)
        self.assertEqual(data['spec']['confirm_tasks'], 64)

    # ---------- round-2 review reproductions ----------

    def test_activation_is_executor_fact_not_config_diff(self):
        # [P1] the executor hook says the edit never activated (activated=False)
        # even though the candidate config differs on every patch key and the
        # frozen boundary predicates all match: NO confirmatory conclusion
        # (contribution, bridge, root) may be drawn from an unexecuted diff
        bench = _ForcedActivationBenchmark(scenario_name='source_error')
        bench.forced = False
        loop = EvolutionLoop(bench, spec('actf', kind='dace', level=AblationLevel.C5, rounds=1),
                             0, Path(self.tmpdir) / 'actf-0')
        s = loop.run()
        for e in self.events(s):
            if e['kind'] == 'EVIDENCE_RESULT':
                for claim in e['payload']['result']['claims']:
                    if claim.get('action_repairs_source'):
                        self.assertEqual(claim['contribution'], 'UNRESOLVED')
                        self.assertEqual(claim['bridge'], 'UNRESOLVED')
                self.assertEqual(e['payload']['result']['minimal'], 'MINIMAL_UNRESOLVED')
        for m in s.mechanism_states:
            self.assertNotEqual(m['z_bridge'], 'SOURCE_REPAIR', m['id'])
        self.assertEqual([b for r in s.rounds for b in r.bridges
                          if b['bridge'] == 'SOURCE_REPAIR'], [])

    def test_missing_activation_event_stays_unresolved(self):
        # [P1] missing activation event (activated=None) must keep every
        # confirmatory slot UNRESOLVED -- never silently treated as True
        bench = _ForcedActivationBenchmark(scenario_name='source_error')
        bench.forced = None
        loop = EvolutionLoop(bench, spec('actn', kind='dace', level=AblationLevel.C5, rounds=1),
                             0, Path(self.tmpdir) / 'actn-0')
        s = loop.run()
        own = [c for e in self.events(s) if e['kind'] == 'EVIDENCE_RESULT'
               for c in e['payload']['result']['claims'] if c.get('action_repairs_source')]
        self.assertTrue(own)
        self.assertTrue(all(c['contribution'] == 'UNRESOLVED' for c in own))
        self.assertTrue(all(c['bridge'] == 'UNRESOLVED' for c in own))

    def test_bridge_not_inherited_across_artifacts(self):
        # [P1] candidate-A earned CONSISTENT/POSITIVE/SOURCE_REPAIR; candidate-B
        # later earns CONSISTENT/UNRESOLVED/UNRESOLVED on the same mechanism:
        # B's unresolved evidence must REPLACE A's conclusion, and the response
        # branches keep both audit records separate
        loop = self.loop_only(spec('brg', kind='dace', level=AblationLevel.C5, rounds=1),
                              'source_error')
        m = loop.registry.propose(Mechanism('m_planner_type', 'planner', 'planner_valid',
                                            ('planner', 'adapter', 'tool'), 'planner_cast'))

        def claim(artifact, bridge, contribution='POSITIVE', own=True, scope='global'):
            return {'mechanism': 'm_planner_type', 'failure': 'CONSISTENT',
                    'boundary': 'CONSISTENT', 'mediator': 'CONSISTENT',
                    'contribution': contribution, 'utility_lower': 0.5,
                    'bridge': bridge, 'artifact': artifact, 'scope': scope,
                    'action_repairs_source': own}
        loop._update_registry({'claims': [claim('candidate-A', 'SOURCE_REPAIR')]})
        self.assertTrue(m.root_confirmed)
        loop._update_registry({'claims': [claim('candidate-B', 'UNRESOLVED',
                                                contribution='UNRESOLVED')]})
        self.assertEqual(m.claim_version, 'candidate-B')
        self.assertEqual(m.z_bridge, 'UNRESOLVED')
        self.assertFalse(m.root_confirmed)
        self.assertEqual(m.response_for('candidate-A')['bridge'], 'SOURCE_REPAIR')
        self.assertEqual(m.response_for('candidate-B')['bridge'], 'UNRESOLVED')
        # a foreign-action (discrimination) claim never rewrites the
        # mechanism's own decision slots
        loop._update_registry({'claims': [claim('candidate-C', 'COMPENSATION', own=False)]})
        self.assertEqual(m.z_bridge, 'UNRESOLVED')
        self.assertEqual(m.claim_version, 'candidate-B')
        self.assertFalse(m.root_confirmed)

    def test_empty_boundary_observation_is_unobservable_not_crash(self):
        # [P1] a legal TaskResult with an empty boundary dict must not crash
        # diagnosis (previously KeyError 'planner_valid'): unobserved fields
        # stay unobservable -- no violation rates, no premise, no contract
        bench = _NoBoundaryBenchmark(scenario_name='benchmark_mix')
        loop = EvolutionLoop(bench, spec('nob', kind='dace', level=AblationLevel.C5, rounds=1),
                             0, Path(self.tmpdir) / 'nob-0')
        s = loop.run()
        events = self.events(s)
        diag = next(e for e in events if e['kind'] == 'DIAGNOSE')
        self.assertEqual(diag['payload']['summary']['symptom_rates'], {})
        self.assertTrue(all(c == 0.0 for c in diag['payload']['summary']['coverage'].values()))
        self.assertEqual([e for e in events if e['kind'] == 'CONTRACT_FROZEN'], [])
        self.assertEqual([e for e in events if e['kind'] == 'PROMOTION'], [])
        self.assertEqual(s.final_config, {k: False for k in s.final_config})

    def test_c3_contribution_evidence_selects_different_action(self):
        # [P1] C2 vs C3, same seed/scenario: when one action has POSITIVE
        # contribution evidence and another is unconfirmed, the contribution
        # view reorders the frozen shortlist so the cap selects a DIFFERENT
        # next action than the utility-only order (C2)
        kw = dict(rounds=1, diag_tasks=8, confirm_tasks=48, selection_tasks=64,
                  eval_tasks=24, intervention_budget=288, mechanism_cap=8,
                  max_candidates=1)
        s2 = self.loop(MethodSpec('c2a', kind='dace', level=AblationLevel.C2, **kw),
                       'benchmark_mix')
        s3 = self.loop(MethodSpec('c3a', kind='dace', level=AblationLevel.C3, **kw),
                       'benchmark_mix')

        def frozen(s):
            return [e for e in self.events(s) if e['kind'] == 'SELECTION_FROZEN'][0]['payload']['candidates']
        f2, f3 = frozen(s2), frozen(s3)
        self.assertNotEqual(f2, f3)
        # C2 keeps the planned order (planner_cast first); C3 promotes the
        # POSITIVE-contribution action (adapter_cast) ahead of it
        self.assertTrue(s2.final_config.get('planner_cast'))
        self.assertTrue(s3.final_config.get('adapter_cast'))

    def test_joint_patch_earns_own_contract_before_selection(self):
        # [P1] a newly constructible joint patch (>=2 confirmed roots) must go
        # through the freeze/lock/evidence entrance on its OWN exact artifact
        # before it may enter selection; nothing may enter selection without
        # its own frozen contract
        loop = self.loop_only(spec('jp', kind='dace', level=AblationLevel.C5, rounds=1,
                                   confirm_tasks=32, selection_tasks=64,
                                   intervention_budget=512), 'benchmark_mix')
        # two roots confirmed against children of the current root, exactly
        # as a natural round would have earned them
        for mid, comp, src, repair in (
                ('m_planner_type', 'planner', 'planner_valid', 'planner_cast'),
                ('m_tool_offbyone', 'tool', 'tool_success', 'tool_strict')):
            m = Mechanism(mid, comp, src, (comp,), repair)
            m.f_status, m.z_bridge, m.r_status = 'CONSISTENT', 'SOURCE_REPAIR', 'POSITIVE'
            loop.registry.propose(m)
        symptoms, traces = loop._diagnose(0)
        loop._freeze_protocol()
        rec = RoundRecord(0, loop.budget.spent, 0, 0.0, 0)
        loop._dace_round(symptoms, traces, 0, rec, {'calls': 0, 'prompt_tokens': 0,
                                                    'completion_tokens': 0})
        events = self.read_events(loop.ledger.path)
        joint_gates = [e for e in events if e['kind'] == 'PATCH_GATE'
                       and e['payload']['contract_kind'] == 'JOINT_PATCH']
        self.assertTrue(joint_gates, 'the joint patch must earn its own contract')
        contracted = {e['payload']['artifact']['version'] for e in events
                      if e['kind'] == 'CONTRACT_FROZEN'}
        for e in events:
            if e['kind'] != 'SELECTION_FROZEN':
                continue
            for cand in e['payload']['candidates']:
                self.assertIn(cand, contracted,
                              'no candidate may enter selection without its own '
                              'frozen contract on its exact artifact')

    def test_joint_patch_deferred_without_budget_slot(self):
        # [P1] with no confirmatory slot at all (intervention budget below one
        # contract), the joint patch is explicitly DEFERRED and never rides
        # into selection on inherited constituent claims
        loop = self.loop_only(spec('jpd', kind='dace', level=AblationLevel.C5, rounds=1,
                                   confirm_tasks=32, selection_tasks=64,
                                   intervention_budget=32), 'benchmark_mix')
        for mid, comp, src, repair in (
                ('m_planner_type', 'planner', 'planner_valid', 'planner_cast'),
                ('m_tool_offbyone', 'tool', 'tool_success', 'tool_strict')):
            m = Mechanism(mid, comp, src, (comp,), repair)
            m.f_status, m.z_bridge, m.r_status = 'CONSISTENT', 'SOURCE_REPAIR', 'POSITIVE'
            loop.registry.propose(m)
        symptoms, traces = loop._diagnose(0)
        loop._freeze_protocol()
        rec = RoundRecord(0, loop.budget.spent, 0, 0.0, 0)
        loop._dace_round(symptoms, traces, 0, rec, {'calls': 0, 'prompt_tokens': 0,
                                                    'completion_tokens': 0})
        events = self.read_events(loop.ledger.path)
        deferred = [e for e in events if e['kind'] == 'JOINT_PATCH_DEFERRED']
        self.assertTrue(deferred)
        # no confirmatory budget slot existed, so no JOINT_PATCH artifact was
        # ever frozen and no selection happened on inherited claims
        self.assertEqual([e for e in events if e['kind'] == 'PATCH_GATE'
                          and e['payload']['contract_kind'] == 'JOINT_PATCH'], [])
        self.assertEqual([e for e in events if e['kind'] == 'SELECTION_FROZEN'], [])
        self.assertEqual(rec.promoted_keys, [])

    # ---------- round-3 review reproductions ----------

    def test_unactivated_candidate_cannot_promote(self):
        # [P1] review round-3 reproduction: every executor hook reports
        # activated=False, every EVIDENCE_RESULT is MINIMAL_UNRESOLVED -- the
        # promotion entry must REFUSE the candidate under the frozen policy.
        # The assertion is on the final winner/accepted state, not on bridge
        # strings (source_error, C5, 1 round, confirm_tasks=16,
        # selection_tasks=256, eval_tasks=8, intervention_budget=128)
        bench = _ForcedActivationBenchmark(scenario_name='source_error')
        bench.forced = False
        loop = EvolutionLoop(bench, spec('unobs', kind='dace', level=AblationLevel.C5,
                                         rounds=1, confirm_tasks=16,
                                         selection_tasks=256, eval_tasks=8,
                                         intervention_budget=128),
                             0, Path(self.tmpdir) / 'unobs-0')
        before = dict(loop.current)
        s = loop.run()
        events = self.events(s)
        results = [e for e in events if e['kind'] == 'EVIDENCE_RESULT']
        self.assertTrue(results)
        self.assertTrue(all(e['payload']['result']['minimal'] == 'MINIMAL_UNRESOLVED'
                            for e in results))
        blocked = [e for e in events if e['kind'] == 'PROMOTION_BLOCKED_UNOBSERVED']
        self.assertTrue(blocked, 'the unobserved candidate must be blocked explicitly')
        winners = [e for e in events if e['kind'] == 'PROMOTION'
                   and e['payload'].get('winner')]
        self.assertEqual(winners, [])
        self.assertTrue(all(not r.accepted for r in s.rounds))
        self.assertEqual(s.final_config, before)
        self.assertFalse(any(r.promoted_keys for r in s.rounds))

    def test_unobserved_promotion_requires_explicit_frozen_policy(self):
        # [P1] utility promotion of a never-activated edit is possible ONLY
        # through an explicitly configured frozen policy with a recorded
        # reason; the reason is echoed on every such promotion event
        bench = _ForcedActivationBenchmark(scenario_name='source_error')
        bench.forced = False
        loop = EvolutionLoop(bench, spec('unobsok', kind='dace', level=AblationLevel.C5,
                                         rounds=1, confirm_tasks=16,
                                         selection_tasks=256, eval_tasks=8,
                                         intervention_budget=128,
                                         allow_unobserved_promotion=True,
                                         unobserved_promotion_reason='executor hook '
                                         'unavailable for this component; utility '
                                         'selection proceeds per frozen policy'),
                             0, Path(self.tmpdir) / 'unobsok-0')
        s = loop.run()
        events = self.events(s)
        policy = [e for e in events if e['kind'] == 'PROMOTION_EVIDENCE_POLICY_FROZEN']
        self.assertTrue(policy)
        self.assertTrue(policy[0]['payload']['allow_unobserved_promotion'])
        self.assertTrue(policy[0]['payload']['unobserved_promotion_reason'])
        allowed = [e for e in events if e['kind'] == 'PROMOTION_UNOBSERVED_ALLOWED']
        self.assertTrue(allowed)
        self.assertTrue(all(e['payload']['reason'] for e in allowed))
        # utility still had to win independent selection for the promotion
        self.assertTrue(any(r.accepted for r in s.rounds))

    def test_unobserved_promotion_without_reason_fails_closed(self):
        # opting in without a recorded reason is a configuration error
        with self.assertRaises(ValueError):
            spec('norea', kind='dace', level=AblationLevel.C5,
                 allow_unobserved_promotion=True, unobserved_promotion_reason='  ')

    def test_promotion_evidence_policy_frozen_before_first_round(self):
        s = self.loop(spec('pep', kind='dace', level=AblationLevel.C5), 'source_error')
        kinds = [e['kind'] for e in self.events(s)]
        self.assertEqual(kinds[0], 'PROTOCOL_FROZEN')
        self.assertEqual(kinds[1], 'PROMOTION_EVIDENCE_POLICY_FROZEN')
        self.assertLess(kinds.index('PROMOTION_EVIDENCE_POLICY_FROZEN'),
                        kinds.index('DIAGNOSE'))

    def test_scoped_claims_are_kept_separate(self):
        # [P1] review round-3 reproduction: the same mechanism and artifact
        # earn a SOURCE_REPAIR under scope cacheable=true and an UNRESOLVED
        # under scope cacheable=false -- BOTH response branches survive (the
        # conditional conclusions stay queryable), and the global decision
        # slots are written only by the global-scope claim
        loop = self.loop_only(spec('sc', kind='dace', level=AblationLevel.C5, rounds=1),
                              'source_error')
        m = loop.registry.propose(Mechanism('m_planner_type', 'planner', 'planner_valid',
                                            ('planner', 'adapter', 'tool'), 'planner_cast'))

        def claim(scope, bridge):
            return {'mechanism': 'm_planner_type', 'failure': 'CONSISTENT',
                    'boundary': 'CONSISTENT', 'mediator': 'CONSISTENT',
                    'contribution': 'POSITIVE', 'utility_lower': 0.5,
                    'bridge': bridge, 'artifact': 'artifact-X', 'scope': scope,
                    'action_repairs_source': True}
        loop._update_registry({'claims': [claim('{"cacheable": true}', 'SOURCE_REPAIR')]})
        loop._update_registry({'claims': [claim('{"cacheable": false}', 'UNRESOLVED')]})
        # both conditional branches retained -- neither overwrites the other
        self.assertEqual(len(m.responses['artifact-X']), 2)
        self.assertEqual(m.response_for('artifact-X', '{"cacheable": true}')['bridge'],
                         'SOURCE_REPAIR')
        self.assertEqual(m.response_for('artifact-X', '{"cacheable": false}')['bridge'],
                         'UNRESOLVED')
        # global slots untouched by scoped claims; conditional views kept
        self.assertEqual(m.f_status, 'PROPOSED')
        self.assertEqual(m.z_bridge, 'UNBRIDGED')
        self.assertEqual(m.scoped_views['{"cacheable": false}']['bridge'], 'UNRESOLVED')
        # a later global claim does move the global slots (latest wins,
        # including UNRESOLVED, within the same scope)
        loop._update_registry({'claims': [claim('global', 'SOURCE_REPAIR')]})
        self.assertEqual(m.claim_version, 'artifact-X')
        self.assertTrue(m.root_confirmed)

    def test_c5_disagreement_computed_from_frozen_predictions(self):
        # [P2] D(a;M) values come from the frozen predictions on the fields
        # BOTH sides explicitly committed to -- computed fractions in [0,1],
        # never identity-hardcoded, and never fabricated for fields a
        # prediction did not name (unit-level no-fabrication tests live in
        # test_components.DisagreementTests)
        s5 = self.loop(spec('d5', kind='dace', level=AblationLevel.C5, rounds=1),
                       'source_error')
        ds = [b['disagreement'] for r in s5.rounds for b in r.bridges]
        self.assertTrue(ds)
        self.assertTrue(all(0.0 <= d <= 1.0 for d in ds))
        self.assertTrue(any(d > 0.0 for d in ds),
                        'the discriminative scheduling must find real '
                        'common-field disagreement between competitors')

    # ---------- round-4 review reproductions ----------

    def test_disabled_attribution_views_never_enter_decisions(self):
        # [P1-1] review round-4 reproduction: a scoped claim carrying
        # contribution=POSITIVE and bridge=SOURCE_REPAIR arrives under a
        # level that DISABLED those views (C2: failure-only). The decision
        # views are gated by the SAME feature switches as the global slots
        # -- the conditional branch can never switch on machinery the
        # ablation turned off, at any level
        scoped = {'mechanism': 'm_planner_type', 'failure': 'CONSISTENT',
                  'boundary': 'CONSISTENT', 'mediator': 'CONSISTENT',
                  'contribution': 'POSITIVE', 'utility_lower': 0.5,
                  'bridge': 'SOURCE_REPAIR', 'artifact': 'artifact-X',
                  'scope': '{"cacheable": true}', 'action_repairs_source': True}
        cases = [
            # level -> expected decision-view fields (failure, contribution, bridge)
            (AblationLevel.C2, ('CONSISTENT', 'UNRESOLVED', 'UNRESOLVED')),
            (AblationLevel.C3, ('CONSISTENT', 'POSITIVE', 'UNRESOLVED')),
            (AblationLevel.CONTRIBUTION_ONLY, ('UNRESOLVED', 'POSITIVE', 'UNRESOLVED')),
            (AblationLevel.C5, ('CONSISTENT', 'POSITIVE', 'SOURCE_REPAIR')),
        ]
        for i, (level, expected) in enumerate(cases):
            loop = self.loop_only(spec(f'ab{i}', kind='dace', level=level, rounds=1),
                                  'source_error')
            m = loop.registry.propose(Mechanism('m_planner_type', 'planner', 'planner_valid',
                                                ('planner', 'adapter', 'tool'), 'planner_cast'))
            loop._update_registry({'claims': [scoped]})
            view = m.scoped_views['{"cacheable": true}']
            self.assertEqual((view['failure'], view['contribution'], view['bridge']),
                             expected, f'{level.value} must gate its decision views')
            # the disabled conclusion never becomes a decision label
            self.assertFalse(m.root_confirmed, level.value)
            self.assertFalse(m.contribution_positive, level.value)
            self.assertFalse(m.demoted_as_compensation, level.value)
            # the raw response branch keeps the full unfiltered evidence
            self.assertEqual(
                m.response_for('artifact-X', '{"cacheable": true}')['bridge'],
                'SOURCE_REPAIR', level.value)
            # and the scope-matched query reads the FILTERED view
            self.assertEqual(m.claim_view('{"cacheable": true}')['bridge'],
                             expected[2], level.value)

    def test_compensation_under_other_scope_does_not_exclude(self):
        # [P1-2] review round-4 reproduction: the mechanism concluded
        # COMPENSATION under cacheable=false, but its contract THIS round
        # runs in the scope where it holds source-repair evidence -- the
        # candidate must stay selectable (the old any()-read excluded it
        # unconditionally with no scope matching). Since the round-5 fix a
        # scope-matched COMPENSATION label also no longer EXCLUDES a
        # candidate at all (it only demotes the root-cause claim), so the
        # assertion is on selectability: the planner candidate appears in
        # the frozen selection set and an accepted promotion happens. Which
        # candidate wins the utility tie is decided by the frozen
        # deterministic tie-break, not by attribution.
        loop = self.loop_only(spec('scx', kind='dace', level=AblationLevel.C5, rounds=1,
                                   confirm_tasks=16, selection_tasks=64,
                                   intervention_budget=256), 'source_error')
        m = loop.registry.propose(Mechanism('m_planner_type', 'planner', 'planner_valid',
                                            ('planner', 'adapter', 'tool'), 'planner_cast'))
        m.scoped_views['{"cacheable": false}'] = {
            'failure': 'CONSISTENT', 'contribution': 'UNRESOLVED',
            'bridge': 'COMPENSATION', 'artifact': 'artifact-old'}
        self.assertFalse(m.demoted_as_compensation)  # not a GLOBAL conclusion
        symptoms, traces = loop._diagnose(0)
        loop._freeze_protocol()
        parent_version = loop.root_artifact.version  # selection candidates hang off THIS parent
        base_config = dict(loop.current)             # ... and THIS pre-round config
        rec = RoundRecord(0, loop.budget.spent, 0, 0.0, 0)
        loop._dace_round(symptoms, traces, 0, rec, {'calls': 0, 'prompt_tokens': 0,
                                                    'completion_tokens': 0})
        # the source_error contract runs under the global scope: the
        # compensation concluded under ANOTHER condition never excluded the
        # candidate -- it was frozen into the selection set and the round
        # accepted a promotion
        self.assertTrue(rec.accepted, 'the candidate must stay selectable when the '
                                      'compensation was concluded under another scope')
        events = self.read_events(loop.ledger.path)
        frozen = next(e for e in events if e['kind'] == 'SELECTION_FROZEN')
        planner_candidate = {'planner_cast': True}
        import json as _json
        planner_version = Artifact.create(
            _json.dumps({**base_config, **planner_candidate}, sort_keys=True),
            parent_version, kind='PATCH').version
        self.assertIn(planner_version, frozen['payload']['candidates'])

    def test_ledger_minimal_bound_to_contract_identity(self):
        # [P1-3] review round-4 reproduction A: the same artifact version
        # (same code + parent) earned MINIMAL_OBSERVED under a GLOBAL
        # contract; a NEW scoped contract on that version terminates. The
        # minimal-feedback query is bound to the contract's identity -- the
        # old global evidence must not leak into the new scoped confirmation
        bench = _FailingCandidateBenchmark(scenario_name='source_error')
        loop = EvolutionLoop(bench, spec('cid', kind='dace', level=AblationLevel.C5,
                                         rounds=1, confirm_tasks=16, selection_tasks=64,
                                         intervention_budget=256),
                             0, Path(self.tmpdir) / 'cid-0')
        symptoms, traces = loop._diagnose(0)
        loop._freeze_protocol()
        m = loop.registry.propose(Mechanism('m_planner_type', 'planner', 'planner_valid',
                                            ('planner', 'adapter', 'tool'), 'planner_cast'))
        pred = loop._prediction(m, {'planner_cast': True}, symptoms, coupled=True)
        a1, c1, r1 = loop._run_contract({'planner_cast': True}, (pred,), 0, 0,
                                        premise=None, kind='PATCH')
        self.assertEqual(r1['minimal'], 'MINIMAL_OBSERVED')
        bench.failing = True
        a2, c2, r2 = loop._run_contract({'planner_cast': True}, (pred,), 0, 1,
                                        premise={'cacheable': True}, kind='PATCH')
        self.assertEqual(r2['status'], 'TERMINATED_UNRESOLVED')
        # same code + parent -> identical artifact version, distinct contracts
        self.assertEqual(a2.version, a1.version)
        self.assertNotEqual(c2.id, c1.id)
        # the new scoped confirmation has NO completed evidence of its own
        self.assertIsNone(loop._ledger_minimal(a2.version, c2.id))
        # the old contract's evidence stays queryable under ITS identity only
        self.assertEqual(loop._ledger_minimal(a1.version, c1.id), 'MINIMAL_OBSERVED')

    def test_failed_confirmation_cannot_ride_old_evidence(self):
        # [P1-3] review round-4 reproduction B: an older contract completed
        # MINIMAL_OBSERVED for the candidate's exact artifact version; this
        # round's own confirmation TERMINATES. The candidate must NOT ride
        # the old evidence into selection -- the gate is bound to the
        # (artifact, contract, scope) identity and failed confirmations are
        # rejected outright under every policy
        bench = _FailingCandidateBenchmark(scenario_name='source_error')
        loop = EvolutionLoop(bench, spec('old', kind='dace', level=AblationLevel.C2,
                                         rounds=1, confirm_tasks=16, selection_tasks=256,
                                         intervention_budget=64),
                             0, Path(self.tmpdir) / 'old-0')
        symptoms, traces = loop._diagnose(0)
        loop._freeze_protocol()
        m = loop.registry.propose(Mechanism('m_planner_type', 'planner', 'planner_valid',
                                            ('planner', 'adapter', 'tool'), 'planner_cast'))
        pred = loop._prediction(m, {'planner_cast': True}, symptoms, coupled=False)
        # pre-complete ONE confirmation for the planner_cast artifact
        # (contract idx 5 -> its own fresh task pool), as a prior round would
        artifact, contract, result = loop._run_contract(
            {'planner_cast': True}, (pred,), 0, 5, premise=None, kind='PATCH')
        self.assertEqual(result['minimal'], 'MINIMAL_OBSERVED')
        # now every candidate execution fails: this round's own contracts
        # terminate without writing any evidence
        bench.failing = True
        rec = RoundRecord(0, loop.budget.spent, 0, 0.0, 0)
        loop._dace_round(symptoms, traces, 0, rec, {'calls': 0, 'prompt_tokens': 0,
                                                    'completion_tokens': 0})
        events = self.read_events(loop.ledger.path)
        terminated = [e for e in events if e['kind'] == 'TERMINATED_UNRESOLVED']
        self.assertTrue(terminated)
        failed = [e for e in events if e['kind'] == 'PROMOTION_BLOCKED_CONFIRMATION_FAILED']
        self.assertTrue(failed, 'the failed confirmation must block promotion explicitly')
        # nothing reached selection and nothing was promoted
        self.assertEqual([e for e in events if e['kind'] == 'SELECTION_FROZEN'], [])
        self.assertFalse(rec.accepted)
        self.assertEqual(rec.promoted_keys, [])
        # even the explicit unobserved-promotion opt-in does NOT rescue a
        # failed confirmation (it covers unobservable executions, not failed
        # ones) -- separate spec, same reproduction
        bench2 = _FailingCandidateBenchmark(scenario_name='source_error')
        loop2 = EvolutionLoop(bench2, spec('oldok', kind='dace', level=AblationLevel.C2,
                                           rounds=1, confirm_tasks=16, selection_tasks=256,
                                           intervention_budget=64,
                                           allow_unobserved_promotion=True,
                                           unobserved_promotion_reason='executor hook '
                                           'unavailable for this component'),
                              0, Path(self.tmpdir) / 'oldok-0')
        symptoms2, traces2 = loop2._diagnose(0)
        loop2._freeze_protocol()
        m2 = loop2.registry.propose(Mechanism('m_planner_type', 'planner', 'planner_valid',
                                              ('planner', 'adapter', 'tool'), 'planner_cast'))
        pred2 = loop2._prediction(m2, {'planner_cast': True}, symptoms2, coupled=False)
        loop2._run_contract({'planner_cast': True}, (pred2,), 0, 5, premise=None, kind='PATCH')
        bench2.failing = True
        rec2 = RoundRecord(0, loop2.budget.spent, 0, 0.0, 0)
        loop2._dace_round(symptoms2, traces2, 0, rec2,
                          {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
        events2 = self.read_events(loop2.ledger.path)
        self.assertEqual([e for e in events2 if e['kind'] == 'SELECTION_FROZEN'], [])
        self.assertFalse(rec2.accepted)

    # ---------- round-5 review reproductions ----------

    def test_beneficial_compensation_wins_independent_selection(self):
        # [P1-1] review round-5 reproduction: source_error, C4, 1 round, a
        # confirm budget that covers only the adapter_cast contract. The
        # candidate earns MINIMAL_OBSERVED + COMPENSATION + POSITIVE
        # contribution. A compensation conclusion demotes the ROOT-CAUSE
        # claim only -- it does not deny the edit's own deployable benefit,
        # so the candidate must RECEIVE an independent selection chance
        # (previously it was excluded outright and no SELECTION_FROZEN ever
        # appeared)
        s = self.loop(spec('r5c', kind='dace', level=AblationLevel.C4, rounds=1,
                           selection_tasks=256, eval_tasks=8, intervention_budget=128),
                      'source_error')
        events = self.events(s)
        kinds = [e['kind'] for e in events]
        self.assertIn('SELECTION_FROZEN', kinds,
                      'the compensation candidate must reach independent selection')
        self.assertIn('PROMOTION_ATTRIBUTION', kinds)
        # the compensation candidate won and deployed
        self.assertEqual(s.final_config.get('adapter_cast'), True)
        self.assertGreater(s.final_eval, s.baseline_eval)
        # the promotion is REPORTED as a compensation deployment, never as a
        # source repair: the root cause persists
        self.assertEqual(len(s.promotions), 1)
        promo = s.promotions[0]
        self.assertEqual(promo['bridge'], 'COMPENSATION')
        self.assertEqual(promo['mechanism'], 'm_adapter_garble')
        self.assertEqual(promo['keys'], ['adapter_cast'])
        self.assertEqual([e for e in events if e['kind'] == 'PROMOTION_ATTRIBUTION']
                         [0]['payload']['attribution'], 'COMPENSATION')
        self.assertEqual(s.rounds[0].promoted_as, ['COMPENSATION'])
        # the mechanism report keeps its demoted label: adapter_cast is a
        # compensation, not a confirmed root repair
        adapter = next(m for m in s.mechanism_states if m['id'] == 'm_adapter_garble')
        self.assertEqual(adapter['z_bridge'], 'COMPENSATION')

    def test_root_cause_report_stays_after_compensation_promotion(self):
        # [P1-1] end-to-end: after a compensation is promoted (round 0), the
        # upstream root cause PERSISTS and relocalization must still confirm
        # it in round 1 -- the upstream mechanism's baseline witness
        # expectation matches the deployed compensation instead of freezing
        # a violation the deployed config no longer exhibits (which would
        # CONTRADICT its own failure contract and break relocalization)
        s = self.loop(spec('r5r', kind='dace', level=AblationLevel.C4, rounds=2,
                           selection_tasks=256, eval_tasks=8, intervention_budget=128),
                      'source_error')
        # round 0 deployed the compensation; round 1 confirmed the root
        self.assertEqual(s.promotions[0]['bridge'], 'COMPENSATION')
        planner = next(m for m in s.mechanism_states if m['id'] == 'm_planner_type')
        self.assertEqual((planner['f_status'], planner['z_bridge']),
                         ('CONSISTENT', 'SOURCE_REPAIR'))
        self.assertEqual(s.localization['truth'], ['planner'])
        self.assertEqual(s.localization['predicted'], ['planner'])
        self.assertEqual(s.final_eval, 1.0)

    def test_unresolved_promotion_label_reported_not_defaulted(self):
        # [P1-1] edge: a policy-allowed unobserved promotion records an
        # UNRESOLVED attribution label (a never-activated edit supports no
        # bridge conclusion); metrics count it separately instead of
        # silently defaulting it to source repair
        bench = _ForcedActivationBenchmark(scenario_name='source_error')
        bench.forced = False  # every activation fact reads False
        loop = EvolutionLoop(bench, spec('r5u', kind='dace', level=AblationLevel.C5,
                                         rounds=1, confirm_tasks=16, selection_tasks=256,
                                         intervention_budget=256,
                                         allow_unobserved_promotion=True,
                                         unobserved_promotion_reason='executor hook '
                                         'unavailable for this component'),
                             0, Path(self.tmpdir) / 'r5u-0')
        s = loop.run()
        self.assertTrue(s.promotions,
                       'the unobserved-promotion policy is explicitly enabled; the '
                       'utility-winning candidate must deploy')
        self.assertEqual(s.promotions[0]['bridge'], 'UNRESOLVED')
        self.assertEqual(s.rounds[0].promoted_as, ['UNRESOLVED'])


if __name__ == '__main__':
    unittest.main()
