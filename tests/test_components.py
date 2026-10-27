import unittest

from dace.core import Prediction
from dace.components import (AblationLevel, LEVEL_FEATURES, Mechanism, MechanismRegistry,
                             joint_compositions, offline_propose, candidate_patches,
                             locality_patch, prediction_disagreement)

SYMPTOMS = {
    'symptom_rates': {'planner_valid': 0.0, 'cache_fresh': 0.5, 'adapter_valid': 0.5,
                      'tool_success': 0.5, 'verifier_agrees': 0.5},
    'mean_utility': 0.5, 'naive_localization': 'adapter',
}


class RegistryTests(unittest.TestCase):
    def test_cap_archives_stalest_proposed(self):
        reg = MechanismRegistry(cap=2)
        for mid in ('m_a', 'm_b', 'm_c'):
            reg.propose(Mechanism(mid, 'planner', 'planner_valid', ('planner',), 'planner_cast'))
        self.assertEqual(sorted(reg.mechanisms), ['m_b', 'm_c'])
        self.assertIn('m_a', reg.archived)

    def test_root_confirmed_survives_cap(self):
        reg = MechanismRegistry(cap=2)
        m = reg.propose(Mechanism('m_a', 'planner', 'planner_valid', ('planner',), 'planner_cast'))
        m.f_status = 'CONSISTENT'
        m.z_bridge = 'SOURCE_REPAIR'
        reg.propose(Mechanism('m_b', 'adapter', 'adapter_valid', ('adapter',), 'adapter_cast'))
        reg.propose(Mechanism('m_c', 'tool', 'tool_success', ('tool',), 'tool_strict'))
        self.assertIn('m_a', reg.mechanisms)  # confirmed root is not archived


class DisagreementTests(unittest.TestCase):
    """D(a;M) is computed ONLY from the expectations both predictions
    explicitly freeze for the SAME action -- never from mechanism identity
    and never from values backfilled for fields a prediction did not name."""

    @staticmethod
    def pred(source, witness, int_values, repairs=True):
        base = ((source, False), (witness, False))
        post = tuple(int_values)
        return Prediction('m', source, ('source', 'adapter'), base, post,
                          ((witness, True),), 'SOURCE_REPAIR',
                          action_repairs_source=repairs)

    def test_identical_predictions_score_zero(self):
        a = self.pred('cache_fresh', 'adapter_valid', (('cache_fresh', True),
                                                       ('adapter_valid', True)))
        b = self.pred('planner_valid', 'adapter_valid', (('cache_fresh', True),
                                                         ('adapter_valid', True)))
        # same expectations on all named fields: not discriminable by this
        # action even though ids and source fields differ
        self.assertEqual(prediction_disagreement(a, b), 0.0)

    def test_shared_field_disagreement_is_a_fraction(self):
        # two predictions that both freeze an expectation for adapter_valid
        # but disagree on it: the action is discriminable on that field
        repair = self.pred('cache_fresh', 'adapter_valid', (('cache_fresh', True),
                                                            ('adapter_valid', True)))
        inert = self.pred('adapter_valid', 'tool_success', (('adapter_valid', False),
                                                            ('tool_success', False)))
        d = prediction_disagreement(repair, inert)
        self.assertEqual(d, 1.0)  # 1 of 1 commonly named field differs

    def test_disjoint_fields_score_zero_no_fabrication(self):
        # review round-3 reproduction: prediction A names only x, prediction
        # B names only y -- NEITHER committed to the other's field. The
        # missing expectations stay UNKNOWN; they must NOT be backfilled
        # from baselines or an implicit "unchanged" assumption (which
        # previously inflated the score to a fabricated 1.0)
        a = Prediction('m1', 'x', ('x',), (('x', False),), (('x', True),),
                       (('x', True),), 'SOURCE_REPAIR')
        b = Prediction('m2', 'y', ('y',), (('y', False),), (('y', True),),
                       (('y', True),), 'SOURCE_REPAIR')
        self.assertEqual(prediction_disagreement(a, b), 0.0)

    def test_different_fields_with_agreeing_predictions_score_zero(self):
        # two mechanisms on different fields whose frozen predictions AGREE
        # (both inert: every named field keeps its violated baseline) are not
        # discriminable by this action -- identity and field disjointness are
        # never scored as disagreement, only differing expectations are
        a = Prediction('m1', 's1', ('s1',), (('s1', False),), (('s1', False),),
                       (('s1', False),), 'UNRESOLVED')
        b = Prediction('m2', 's2', ('s2',), (('s2', False),), (('s2', False),),
                       (('s2', False),), 'UNRESOLVED')
        self.assertEqual(prediction_disagreement(a, b), 0.0)

    def test_partial_disagreement_over_common_fields(self):
        # two commonly named fields, one differing: a real fraction, not 0/1
        a = self.pred('cache_fresh', 'adapter_valid', (('cache_fresh', True),
                                                       ('adapter_valid', True)))
        b = self.pred('adapter_valid', 'tool_success', (('cache_fresh', True),
                                                        ('adapter_valid', False)))
        self.assertEqual(prediction_disagreement(a, b), 0.5)


class MechanismResponseTests(unittest.TestCase):
    def test_response_branches_are_per_artifact_and_scope(self):
        # review round-3 reproduction: the SAME mechanism and artifact earn
        # claims under TWO applicability conditions -- both response branches
        # are retained; one condition's conclusion never overwrites the
        # other's
        m = Mechanism('m_a', 'planner', 'planner_valid', ('planner',), 'planner_cast')
        m.responses.setdefault('artifact-A', {})['{"cacheable": true}'] = {
            'failure': 'CONSISTENT', 'contribution': 'POSITIVE',
            'bridge': 'SOURCE_REPAIR', 'action_repairs_source': True}
        m.responses.setdefault('artifact-A', {})['{"cacheable": false}'] = {
            'failure': 'UNRESOLVED', 'contribution': 'UNRESOLVED',
            'bridge': 'UNRESOLVED', 'action_repairs_source': True}
        self.assertEqual(len(m.responses['artifact-A']), 2)
        self.assertEqual(m.response_for('artifact-A', '{"cacheable": true}')['bridge'],
                         'SOURCE_REPAIR')
        self.assertEqual(m.response_for('artifact-A', '{"cacheable": false}')['bridge'],
                         'UNRESOLVED')
        self.assertIsNone(m.response_for('artifact-B'))

    def test_scoped_views_never_become_global_state(self):
        # review round-4 reproduction: a scoped claim records a CONDITIONAL
        # view; the global decision slots speak only for the unconditional
        # scope. Scoped conclusions are queried via claim_view(scope) and
        # never flatten into global labels -- "holds under some condition"
        # is not "holds unconditionally"
        m = Mechanism('m_a', 'planner', 'planner_valid', ('planner',), 'planner_cast')
        m.scoped_views['{"cacheable": true}'] = {
            'failure': 'CONSISTENT', 'contribution': 'POSITIVE',
            'bridge': 'SOURCE_REPAIR', 'artifact': 'artifact-A'}
        m.scoped_views['{"cacheable": false}'] = {
            'failure': 'CONSISTENT', 'contribution': 'UNRESOLVED',
            'bridge': 'COMPENSATION', 'artifact': 'artifact-A'}
        # global slots untouched; neither conditional conclusion leaks into
        # the unconditional state in EITHER direction
        self.assertEqual(m.f_status, 'PROPOSED')
        self.assertEqual(m.z_bridge, 'UNBRIDGED')
        self.assertFalse(m.root_confirmed)
        self.assertFalse(m.contribution_positive)
        self.assertFalse(m.demoted_as_compensation)
        # the conditional conclusions remain queryable per scope
        self.assertEqual(m.claim_view('{"cacheable": true}')['bridge'], 'SOURCE_REPAIR')
        self.assertEqual(m.claim_view('{"cacheable": false}')['bridge'], 'COMPENSATION')
        # a scope with no conditional view of its own falls back to the
        # unconditional view (never to the OTHER condition's conclusion)
        self.assertEqual(m.claim_view('{"magnitude_lt": 500}'),
                         m._global_view())
        # root confirmations enumerate per scope, never as global state
        self.assertEqual([s for s, _ in m.root_confirmations()],
                         ['{"cacheable": true}'])

    def test_global_view_covers_scoped_query(self):
        # unconditional evidence covers every subpopulation: a scoped query
        # without its own view falls back to the global slots
        m = Mechanism('m_a', 'planner', 'planner_valid', ('planner',), 'planner_cast')
        m.f_status, m.r_status, m.z_bridge, m.claim_version = \
            'CONSISTENT', 'POSITIVE', 'SOURCE_REPAIR', 'artifact-G'
        view = m.claim_view('{"cacheable": true}')
        self.assertEqual(view['bridge'], 'SOURCE_REPAIR')
        self.assertEqual(view['artifact'], 'artifact-G')


class JointCompositionTests(unittest.TestCase):
    """Joint root-repair composition respects scope compatibility: roots
    confirmed under mutually exclusive conditions never merge; compatible
    scopes compose with the merged applicability premise."""

    @staticmethod
    def confirmed(mid, comp, src, repair, scope='global'):
        m = Mechanism(mid, comp, src, (comp,), repair)
        if scope == 'global':
            m.f_status, m.z_bridge, m.r_status = 'CONSISTENT', 'SOURCE_REPAIR', 'POSITIVE'
        else:
            m.scoped_views[scope] = {'failure': 'CONSISTENT', 'contribution': 'POSITIVE',
                                     'bridge': 'SOURCE_REPAIR', 'artifact': mid}
        return m

    def test_mutually_exclusive_scopes_do_not_compose(self):
        # review round-4 reproduction: root under cacheable=true and root
        # under cacheable=false -- "exists under some condition" must not
        # merge them into one unconditional joint patch
        reg = MechanismRegistry()
        reg.propose(self.confirmed('m_a', 'planner', 'planner_valid', 'planner_cast',
                                   '{"cacheable": true}'))
        reg.propose(self.confirmed('m_b', 'tool', 'tool_success', 'tool_strict',
                                   '{"cacheable": false}'))
        self.assertEqual(joint_compositions(reg), [])

    def test_compatible_scopes_compose_with_merged_premise(self):
        reg = MechanismRegistry()
        reg.propose(self.confirmed('m_a', 'planner', 'planner_valid', 'planner_cast',
                                   '{"cacheable": true}'))
        reg.propose(self.confirmed('m_b', 'tool', 'tool_success', 'tool_strict',
                                   '{"magnitude_lt": 500}'))
        comps = joint_compositions(reg)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0]['patch'], {'planner_cast': True, 'tool_strict': True})
        self.assertEqual(comps[0]['premise'], {'cacheable': True, 'magnitude_lt': 500})

    def test_global_root_composes_with_scoped_root(self):
        reg = MechanismRegistry()
        reg.propose(self.confirmed('m_a', 'planner', 'planner_valid', 'planner_cast'))
        reg.propose(self.confirmed('m_b', 'tool', 'tool_success', 'tool_strict',
                                   '{"magnitude_lt": 500}'))
        comps = joint_compositions(reg)
        self.assertEqual(len(comps), 1)
        # unconditional + conditional compose under the conditional premise
        # (the intersection), never as an unconditional joint claim
        self.assertEqual(comps[0]['premise'], {'magnitude_lt': 500})
        self.assertEqual([m.id for m in comps[0]['roots']], ['m_a', 'm_b'])


class ProposalTests(unittest.TestCase):
    def test_offline_propose_matches_symptoms(self):
        reg = MechanismRegistry()
        proposed = offline_propose(SYMPTOMS, reg)
        ids = {m.id for m in proposed}
        self.assertIn('m_planner_cache', ids)
        self.assertIn('m_adapter_garble', ids)
        self.assertNotIn('m_planner_type', ids)

    def test_candidate_patches_include_joint(self):
        reg = MechanismRegistry()
        offline_propose(SYMPTOMS, reg)
        single = candidate_patches(reg, joint=False)
        self.assertTrue(all(len(p) == 1 for p in single))
        for m in reg.active():
            m.f_status, m.z_bridge = 'CONSISTENT', 'SOURCE_REPAIR'
        joint = candidate_patches(reg, joint=True)
        self.assertTrue(any(len(p) > 1 for p in joint))

    def test_locality_patch_follows_naive_localization_then_reflects(self):
        patch = locality_patch(SYMPTOMS, set())
        self.assertEqual(patch, {'adapter_cast': True})
        patch2 = locality_patch(SYMPTOMS, {'adapter_cast'})
        self.assertNotEqual(patch, patch2)


class AblationLevelTests(unittest.TestCase):
    def test_level_features_monotone(self):
        # cumulative levels: features never switch off going up the path
        seq = [AblationLevel.C0, AblationLevel.C1, AblationLevel.C2, AblationLevel.C3,
               AblationLevel.C4, AblationLevel.C5]
        for a, b in zip(seq, seq[1:]):
            for feat, on in LEVEL_FEATURES[a].items():
                if on:
                    self.assertTrue(LEVEL_FEATURES[b][feat],
                                    f'{a.value} enables {feat} lost at {b.value}')

    def test_contribution_only_control(self):
        f = LEVEL_FEATURES[AblationLevel.CONTRIBUTION_ONLY]
        self.assertTrue(f['contribution'])
        self.assertFalse(f['failure'])
        self.assertFalse(f['bridge'])


if __name__ == '__main__':
    unittest.main()
