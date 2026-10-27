"""Fixed-look task-level empirical predicate bounds, not causal identification.

Exact binomial simultaneous intervals require independent task
units; no inference from a single matching trace to a population invariant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PredicateProtocol:
    max_violation_rate: float = .1
    schema: str = 'dace-task-predicates/v1'
    unit: str = 'task'
    interval: str = 'EXACT_CLOPPER_PEARSON_BONFERRONI'
    look: str = 'FIXED_COMPLETE_SAMPLE'

    def __post_init__(self):
        if (self.schema!='dace-task-predicates/v1' or self.unit!='task'
                or self.interval!='EXACT_CLOPPER_PEARSON_BONFERRONI'
                or self.look!='FIXED_COMPLETE_SAMPLE'
                or type(self.max_violation_rate) not in (int,float)
                or not math.isfinite(self.max_violation_rate)
                or not 0<=self.max_violation_rate<=1):
            raise ValueError('Unsupported frozen task predicate protocol')


def _binomial_cdf(n,k,p):
    if k<0:
        return 0.
    if k>=n or p<=0:
        return 1.
    if p>=1:
        return 0.
    # Log-sum-exp recurrence keeps extreme tails representable without SciPy.
    term = n*math.log1p(-p)
    peak,total = term,1.
    log_odds = math.log(p)-math.log1p(-p)
    for i in range(k):
        term += math.log(n-i)-math.log(i+1)+log_odds
        if term>peak:
            total = total*math.exp(peak-term)+1
            peak = term
        else:
            total += math.exp(term-peak)
    return min(1.,math.exp(peak)*total)


def binomial_interval(k,n,alpha):
    """Two-sided Clopper--Pearson by inversion of exact binomial tails."""
    if (type(k) is not int or type(n) is not int or n<=0 or not 0<=k<=n
            or type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0<alpha<1):
        raise ValueError('Exact binomial interval requires valid counts/confidence')
    tail = alpha/2
    def invert(upper):
        low,high = 0.,1.
        for _ in range(60):
            p = (low+high)/2
            # P(X>=k) = P(n-X<=n-k); no subtraction from a near-one CDF.
            value = _binomial_cdf(n,k,p) if upper else _binomial_cdf(n,n-k,1-p)
            if (value>tail) if upper else (value<tail):
                low = p
            else:
                high = p
        return (low+high)/2
    lower = 0. if k==0 else math.exp(math.log(tail)/n) if k==n else invert(False)
    upper = 1. if k==n else -math.expm1(math.log(tail)/n) if k==0 else invert(True)
    return [lower,upper]


def evaluate_predicates(task_rows,expected,*,alpha,max_violation_rate):
    """Evaluate a fully observed frozen task list at its single planned look.

    Missing predicates remain unresolved even if all other tasks match. This
    avoids treating the reached/observable subset as the original sample.
    The caller must verify frozen sample completion and provenance first.
    """
    if (type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0<alpha<1
            or type(max_violation_rate) not in (int,float)
            or not math.isfinite(max_violation_rate) or not 0<=max_violation_rate<=1):
        raise ValueError('Freeze finite confidence and violation tolerance before execution')
    if (not isinstance(expected,dict) or not expected
            or any(not isinstance(k,str) or not k or type(v) is not bool for k,v in expected.items())):
        raise ValueError('Predicates require named boolean expectations')
    if not isinstance(task_rows,list) or not task_rows:
        raise ValueError('Fixed look requires task-level observations')
    ids = set()
    for row in task_rows:
        if (not isinstance(row,dict) or set(row)!={'task','boundary'}
                or not isinstance(row['task'],str) or not row['task'] or row['task'] in ids
                or not isinstance(row['boundary'],dict)
                or any(type(row['boundary'][k]) is not bool for k in expected if k in row['boundary'])):
            raise ValueError('Use unique task units with actual boolean or missing observations')
        ids.add(row['task'])
    field_alpha = alpha/len(expected)
    results = {}
    for field,value in sorted(expected.items()):
        ok = [r['task'] for r in task_rows if field in r['boundary'] and r['boundary'][field]==value]
        bad = [r['task'] for r in task_rows if field in r['boundary'] and r['boundary'][field]!=value]
        missing = [r['task'] for r in task_rows if field not in r['boundary']]
        n = len(ok)+len(bad)
        if n:
            interval = binomial_interval(len(bad),n,field_alpha)
        else:
            interval = [0.,1.]
        status = 'UNRESOLVED'
        if n and not missing:
            if interval[1]<=max_violation_rate:
                status = 'CONSISTENT'
            elif interval[0]>max_violation_rate:
                status = 'CONTRADICTED'
        results[field] = {'expected':value,'status':status,'n':n,'n_ok':len(ok),
            'n_bad':len(bad),'n_missing':len(missing),'violation_interval':interval,
            'alpha':field_alpha,'max_violation_rate':max_violation_rate,
            'anchors':{'ok':ok,'bad':bad,'missing':missing}}
    states = {r['status'] for r in results.values()}
    aggregate = ('CONTRADICTED' if 'CONTRADICTED' in states else
                 'CONSISTENT' if states=={'CONSISTENT'} else 'UNRESOLVED')
    return {'status':aggregate,'predicates':results,'alpha':alpha,
        'unit':'task','look':'FIXED_COMPLETE_SAMPLE','interval':'EXACT_CLOPPER_PEARSON_BONFERRONI',
        'assumptions':['independent identically distributed binary task units',
                       'task population and predicates fixed before observations'],
        'causal_identification':False}
