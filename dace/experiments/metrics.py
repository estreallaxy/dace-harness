"""Metric computation for E1-E5 from run summaries and ledgers."""
from __future__ import annotations

import json
from pathlib import Path

from ..components import MECHANISM_CATALOG

KEY_TO_COMP = {rep: comp for _, comp, _, _, rep in MECHANISM_CATALOG}


def _bridge_feature(summary) -> bool:
    """Whether the run's method actually had the bridge view enabled.

    Reads the ablation flags recorded on the summary (legacy summaries fall
    back to the level table). Raw labeling keeps whatever the underlying
    controller computed, but a level with bridges disabled ACCEPTS no bridge
    conclusion -- its decision view reads UNRESOLVED regardless of raw
    labels, and experiment metrics must score exactly that."""
    feats = getattr(summary, 'features', None)
    if feats is None:
        from ..components import AblationLevel, LEVEL_FEATURES
        try:
            return bool(LEVEL_FEATURES[AblationLevel(summary.level)].get('bridge'))
        except (KeyError, ValueError):
            return False
    return bool(feats.get('bridge'))


def pass_at_1(summary) -> float:
    return summary.final_eval


def mean_improvement(summary) -> float:
    return summary.final_eval - summary.baseline_eval


def cross_run_variance(values) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def localization_accuracy(summary) -> float:
    if getattr(summary, 'ground_truth', {}).get('known') is False:
        return None
    return summary.localization.get('accuracy', 0.0)


def _mean_known(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def repair_compensation_confusion(summaries):
    """Repair/compensation identification per promoted repair key.

    Truth: a promoted key is a true repair iff some fault is repaired by it
    at its own component; if it only offsets an upstream fault it is a
    compensation. Prediction: the method-ACCEPTED attribution label recorded
    for the promotion event that deployed the key (scope-matched and
    ablation-filtered -- see RunSummary.promotions). Alignment is per
    promotion event (mechanism, artifact, scope), never a flattening of raw
    labeling across artifacts and conditions.

    Methods without the bridge view cannot demote a patch: their accepted
    claim defaults to source repair, which is the honest measurement of the
    uncoupled ablation (they mislabel compensations as repairs). A method
    WITH the bridge view whose own claim stayed UNRESOLVED is counted in a
    separate bucket and never forced into either class.
    """
    tp = fp = tn = fn = unresolved = unavailable = 0
    for s in summaries:
        if s.ground_truth.get('known') is False:
            unavailable += 1
            continue
        gt_faults = {f['repair_key']: f for f in s.ground_truth.get('faults', [])}
        key_label = {}
        for p in getattr(s, 'promotions', None) or []:
            for k in p.get('keys', []):
                key_label[k] = p.get('bridge', 'SOURCE_REPAIR')
        for key, value in s.final_config.items():
            if not value or key not in KEY_TO_COMP:
                continue
            truth_compensation = key not in gt_faults
            label = key_label.get(key)
            if label is None:
                # no promotion record (locality baseline or legacy run): a
                # method with no accepted bridge claim defaults to repair
                label = 'SOURCE_REPAIR'
            if label == 'UNRESOLVED':
                unresolved += 1
                continue
            predicted_compensation = label == 'COMPENSATION'
            if truth_compensation and predicted_compensation:
                tn += 1
            elif truth_compensation and not predicted_compensation:
                fp += 1  # mislabeled compensation as repair
            elif not truth_compensation and predicted_compensation:
                fn += 1
            else:
                tp += 1
    return {'repair_identified': tp, 'compensation_identified': tn,
            'compensation_mislabeled_repair': fp,
            'repair_mislabeled_compensation': fn,
            'bridge_unresolved': unresolved, 'ground_truth_unavailable': unavailable}


def effective_patch_rate(summaries):
    total = promoted = 0
    for s in summaries:
        for rec in s.rounds:
            proposed = rec.contracts_frozen
            total += proposed
            if rec.accepted and rec.promoted_keys:
                promoted += 1
    return (promoted / total) if total else 0.0


def intervention_cost(summaries) -> float:
    return sum(s.budget_spent for s in summaries) / max(1, len(summaries))


def discrimination_rate(summaries):
    """Share of the method's OWN attribution events whose bridge the METHOD
    accepted (decision view, ablation-filtered) was resolved.

    Raw labeling is the audit trail of whatever the underlying controller
    computed; this metric reads DECISION claims only. Foreign-action
    discrimination records are never the mechanism's own claim, and a level
    with the bridge view disabled resolves nothing -- its accepted bridge is
    UNRESOLVED whatever the raw labels say, so an ablated method is never
    credited with bridge identification it did not perform."""
    resolved = total = 0
    for s in summaries:
        bridge_on = _bridge_feature(s)
        for entry in s.labeling:
            if not entry.get('own', True):
                continue
            total += 1
            accepted = entry['bridge'] if bridge_on else 'UNRESOLVED'
            if accepted in ('SOURCE_REPAIR', 'COMPENSATION'):
                resolved += 1
    return resolved / total if total else 0.0


def regression_stats(summaries) -> dict:
    """Deployment regression accounting, split by evidence strength.

    Only DEPLOYED winners are counted -- a rejected candidate's bounds
    constrained nothing that was ever deployed. For each deployed winner and
    task family:

      confirmed            direction-matched paired UPPER bound below zero
                           (statistically confirmed regression)
      observed_decline     paired mean negative but the upper bound does not
                           confirm it
      noninferiority_unproven  paired lower bound negative: the run cannot yet
                           RULE OUT a regression -- never itself a regression
                           (small samples, zero change and even positive true
                           effects can all produce a negative lower bound)
    """
    stats = {'confirmed': 0, 'observed_decline': 0, 'noninferiority_unproven': 0}
    for s in summaries:
        ledger = Path(s.ledger_path)
        if not ledger.exists():
            continue
        for line in ledger.read_text(encoding='utf-8').splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get('kind') != 'PROMOTION':
                continue
            payload = ev.get('payload', {})
            winner = payload.get('winner')
            if not winner:
                continue
            details = (payload.get('results', {}).get(winner, {})
                       .get('family_details') or {})
            for _family, d in details.items():
                upper, lower, mean = d.get('upper'), d.get('lower'), d.get('mean')
                if upper is not None and upper < 0:
                    stats['confirmed'] += 1
                elif mean is not None and mean < 0:
                    stats['observed_decline'] += 1
                elif lower is not None and lower < 0:
                    stats['noninferiority_unproven'] += 1
    return stats


def regression_count(summaries) -> int:
    """Statistically CONFIRMED family regressions on deployed winners only."""
    return regression_stats(summaries)['confirmed']


def budget_curve(summaries):
    """Performance-budget curve points (E5): cumulative cost vs eval utility."""
    curves = {}
    for s in summaries:
        curves.setdefault(s.method, []).append(
            [(spent, utility) for spent, utility in s.checkpoints])
    return curves


def llm_cost(summaries):
    optimizer = [s.llm_usage for s in summaries]
    executions = [getattr(s, 'execution_usage', {}) for s in summaries]
    active = [u for u in optimizer + executions if u.get('calls', 0) > 0]
    def total(field):
        values = [u[field] for u in active if type(u.get(field)) in (int, float)]
        return sum(values) if active and len(values) == len(active) else None
    return {'calls':sum(u.get('calls', 0) for u in optimizer),
            'optimizer_calls':sum(u.get('calls', 0) for u in optimizer),
            'task_executions':sum(u.get('calls', 0) for u in executions),
            'input_tokens':total('prompt_tokens'), 'output_tokens':total('completion_tokens'),
            'cost_usd':total('cost_usd'),
            'accounting_complete':bool(active) and all(u.get('accounting_complete') for u in active)}


def e1_row(method, summaries):
    pas = [pass_at_1(s) for s in summaries]
    reg = regression_stats(summaries)
    return {'method': method, 'n_runs': len(summaries),
            'pass@1_mean': round(sum(pas) / len(pas), 4),
            'improvement_mean': round(sum(mean_improvement(s) for s in summaries) / len(summaries), 4),
            'cross_run_variance': round(cross_run_variance(pas), 6),
            'budget_spent_mean': round(intervention_cost(summaries), 1),
            # deployment regression accounting split by evidence strength:
            # only statistically confirmed regressions on deployed winners
            # count as regressions; a negative lower bound alone is reported
            # separately as non-inferiority not yet established
            'regressions': reg['confirmed'],
            'regressions_observed_decline': reg['observed_decline'],
            'regressions_noninferiority_unproven': reg['noninferiority_unproven'],
            'localization_accuracy': _mean_known(localization_accuracy(s) for s in summaries)}


def e2_row(method, summaries):
    return {**e1_row(method, summaries),
            'repair/compensation': repair_compensation_confusion(summaries),
            'effective_patch_rate': round(effective_patch_rate(summaries), 4),
            'interventions_per_run': round(sum(r.contracts_frozen for s in summaries
                                               for r in s.rounds) / max(1, len(summaries)), 2)}


def e3_row(b_label, k_label, summaries):
    pas = [pass_at_1(s) for s in summaries]
    return {'intervention_budget': b_label, 'mechanism_cap': k_label,
            'pass@1_mean': round(sum(pas) / len(pas), 4),
            'discrimination_rate': round(discrimination_rate(summaries), 4),
            'budget_spent_mean': round(intervention_cost(summaries), 1)}


def e4_row(method, scenario, summaries):
    conf = repair_compensation_confusion(summaries)
    interventions = sum(r.contracts_frozen for s in summaries for r in s.rounds)
    return {'method': method, 'scenario': scenario,
            'localization_accuracy': _mean_known(localization_accuracy(s) for s in summaries),
            'repair_identified': conf['repair_identified'],
            'compensation_identified': conf['compensation_identified'],
            'compensation_mislabeled_repair': conf['compensation_mislabeled_repair'],
            'corrective_interventions': round(interventions / max(1, len(summaries)), 2)}
