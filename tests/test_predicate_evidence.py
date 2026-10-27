import math
import unittest
from dace.predicate_evidence import binomial_interval,evaluate_predicates


class PredicateEvidenceTests(unittest.TestCase):
    def evaluate(self,values,**kwargs):
        rows = [{'task':str(i),'boundary':{'valid':v} if v is not None else {}}
                for i,v in enumerate(values)]
        return evaluate_predicates(rows,{'valid':True},alpha=.05,
                                   max_violation_rate=kwargs.get('tolerance',.1))

    def test_single_matching_trace_does_not_confirm_random_predicate(self):
        result = self.evaluate([True])
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertFalse(result['causal_identification'])

    def test_single_violation_does_not_reject_population_hypothesis(self):
        self.assertEqual(self.evaluate([False])['status'],'UNRESOLVED')

    def test_sufficient_fixed_sample_can_bound_violations(self):
        result = self.evaluate([True]*1000)
        self.assertEqual(result['status'],'CONSISTENT')
        self.assertLess(result['predicates']['valid']['violation_interval'][1],.1)

    def test_repeated_violation_can_exclude_frozen_tolerance(self):
        self.assertEqual(self.evaluate([False]*1000)['status'],'CONTRADICTED')

    def test_interval_crossing_threshold_stays_unresolved(self):
        self.assertEqual(self.evaluate([False]*110+[True]*890)['status'],'UNRESOLVED')

    def test_missing_observation_cannot_change_fixed_population(self):
        result = self.evaluate([True]*1000+[None])
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertEqual(result['predicates']['valid']['n_missing'],1)
        self.assertEqual(result['predicates']['valid']['anchors']['missing'],['1000'])

    def test_every_predicate_gets_a_simultaneous_alpha_allocation(self):
        rows = [{'task':str(i),'boundary':{'a':True,'b':False}} for i in range(1000)]
        result = evaluate_predicates(rows,{'a':True,'b':False},alpha=.05,max_violation_rate=.1)
        self.assertEqual(result['status'],'CONSISTENT')
        self.assertEqual([x['alpha'] for x in result['predicates'].values()],[.025,.025])

    def test_duplicate_units_cannot_inflate_sample_size(self):
        with self.assertRaisesRegex(ValueError,'unique task'):
            evaluate_predicates([{'task':'same','boundary':{}},{'task':'same','boundary':{}}],
                                {'valid':True},alpha=.05,max_violation_rate=.1)

    def test_non_boolean_observations_are_not_silent_matches(self):
        with self.assertRaises(ValueError):
            self.evaluate([1])

    def test_zero_tolerance_cannot_be_established_from_finite_random_sample(self):
        self.assertEqual(self.evaluate([True]*1000,tolerance=0)['status'],'UNRESOLVED')

    def test_nonfinite_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predicates([{'task':'a','boundary':{}}],{'valid':True},
                                alpha=math.nan,max_violation_rate=.1)

    def test_exact_interval_matches_primary_scipy_documentation_example(self):
        # Official scipy.stats BinomTestResult.proportion_ci example, k=7/n=50.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
        low,high = binomial_interval(7,50,.05)
        self.assertAlmostEqual(low,.05819170033997342,places=11)
        self.assertAlmostEqual(high,.26739600249700846,places=11)

    def test_exact_interval_has_at_least_nominal_binomial_coverage(self):
        n,alpha = 10,.05
        intervals = [binomial_interval(k,n,alpha) for k in range(n+1)]
        for i in range(1,20):
            p = i/20
            missed = sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k,(lo,hi) in enumerate(intervals)
                         if not lo<=p<=hi)
            self.assertLessEqual(missed,alpha+1e-12)

    def test_extreme_binomial_counts_stay_in_unit_interval(self):
        for k,n in ((0,10000),(10000,10000),(1,10000),(9999,10000)):
            lo,hi = binomial_interval(k,n,1e-8)
            self.assertLessEqual(0,lo)
            self.assertLessEqual(lo,hi)
            self.assertLessEqual(hi,1)


if __name__=='__main__':
    unittest.main()
