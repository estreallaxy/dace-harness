"""Native, version-scoped CMVS and typed joint-update targets.

This reducer never calls a model or promotes a harness. Only validated closed
confirmation journals can supply empirical claims; new revisions start as
proposals. Archived and stale records remain in the audit history.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict

from .components import AblationLevel, LEVEL_FEATURES
from .core import Artifact,digest
from .durable import contract_from_dict
from .source_selection import confirmation_feedback


def project_claim(claim,level):
    features = LEVEL_FEATURES[AblationLevel(level)]
    result = copy.deepcopy(claim)
    if not features['failure']:
        result['failure'] = 'UNRESOLVED'
    if not features['contribution']:
        for key in ('boundary','mediator','contribution'):
            result[key] = 'UNRESOLVED'
        result.pop('utility_lower',None)
        result.pop('utility_alpha',None)
    if not features['bridge']:
        result['bridge'] = 'UNBRIDGED'
    if 'predicate_evidence' in result:
        keep = ({'failure'} if features['failure'] else set()) | (
            {'boundary','mediator'} if features['contribution'] else set())
        result['predicate_evidence'] = {k:v for k,v in result['predicate_evidence'].items() if k in keep}
    result['causal_identification'] = False
    return result


class SourceState:
    def __init__(self,parent,cap=4,level='C5'):
        if type(cap) is not int or cap<=0:
            raise ValueError('Active mechanism cap must be positive')
        if parent!=Artifact.create(parent.content,parent.parent,parent.kind):
            raise ValueError('Exact source parent required')
        self.parent,self.cap,self.level = parent.version,cap,AblationLevel(level)
        self.records,self.batches,self.proofs,self.targets = {},{}, {},[]
        self.utility = {}
        self.sequence = 0

    @property
    def features(self):
        return LEVEL_FEATURES[self.level]

    def register(self,batch):
        if batch.parent!=self.parent:
            raise ValueError('Mechanism proposals belong to the current exact parent')
        identity = digest(asdict(batch))
        if identity in self.batches:
            return identity
        self.sequence += 1
        mapping = {}
        if self.features['mechanisms']:
            for m in batch.mechanisms:
                # The full frozen hypothesis/response specification is a revision.
                # Action IDs are local labels, not immutable interventions.
                # Bind every response prediction to its exact source edit.
                revision = digest({'parent':self.parent,'hypothesis':asdict(m),
                    'actions':[asdict(a) for a in batch.actions]})
                mapping[m.id] = revision
                if revision not in self.records:
                    self.records[revision] = {'revision':revision,'parent':self.parent,
                        'hypothesis':asdict(m),'status':'ACTIVE','introduced':self.sequence,
                        'action_versions':{a.id:a.artifact.version for a in batch.actions},
                        'responses':{},'source_views':{},'discriminations':[]}
            active = [r for r in self.records.values() if r['status']=='ACTIVE']
            def priority(r):
                supported = any(v['status']=='PREDICTION_SUPPORTED' for v in r['discriminations'])
                # Frozen, outcome-independent criterion for untested proposals;
                # mere empirical compatibility cannot raise mechanism priority.
                return (-int(supported),-r['introduced'],r['revision'])
            keep = {r['revision'] for r in sorted(active,key=priority)[:self.cap]}
            for record in active:
                if record['revision'] not in keep:
                    record.update(status='ARCHIVED',archive_reason='ACTIVE_CAP_PREDECLARED_PRIORITY')
        self.batches[identity] = {'batch':batch,'mechanisms':mapping}
        return identity

    def scheduled_actions(self,batch_id):
        """Compete active same-parent proposals across rounds without rebinding.

        Each action retains its original batch's prediction contract. Missing
        cross-batch predictions are never invented. Identical artifacts are
        scheduled once, preferring disagreement, then the latest proposal.
        C0 has no persistent mechanism pool and uses the current batch only.
        """
        if self.batches[batch_id]['batch'].parent!=self.parent:
            raise ValueError('Stale action pool cannot target the current parent')
        candidates = []
        for index,(identity,item) in enumerate(self.batches.items()):
            if item['batch'].parent!=self.parent:
                continue
            if not self.features['mechanisms'] and identity!=batch_id:
                continue
            for action,score in self.ranked_actions(identity):
                candidates.append((identity,action,score,index))
        candidates.sort(key=lambda row:(-row[2],-row[3],row[1].id,row[0]))
        seen,result = set(),[]
        for identity,action,score,_ in candidates:
            if action.artifact.version in seen:
                continue
            seen.add(action.artifact.version)
            result.append((identity,action,score))
        return result

    def ranked_actions(self,batch_id):
        item = self.batches[batch_id]
        batch = item['batch']
        active_ids = {m for m,r in item['mechanisms'].items() if self.records[r]['status']=='ACTIVE'}
        actions = list(batch.actions)
        if self.features['mechanisms']:
            actions = [a for a in actions if any(m.id in active_ids and m.own_action==a.id for m in batch.mechanisms)]
        if not self.features['probe']:
            return [(a,0) for a in sorted(actions,key=lambda a:a.id)]
        # Disagreement uses only explicit frozen predictions of active revisions.
        from itertools import combinations
        scores = {}
        for action in actions:
            views = [dict(p.intervention_boundary+p.mediator) for p in batch.predictions_for(action.id)
                     if p.mechanism in active_ids]
            scores[action.id] = sum(sum(left[k]!=right[k] for k in left.keys() & right.keys())
                                    for left,right in combinations(views,2))
        return [(a,scores[a.id]) for a in sorted(actions,key=lambda a:(-scores[a.id],a.id))]

    def observe(self,batch_id,action_id,path):
        item = self.batches[batch_id]
        batch = item['batch']
        if batch.parent!=self.parent:
            raise ValueError('Stale mechanism responses cannot update the current parent')
        action = next(a for a in batch.actions if a.id==action_id)
        proof = confirmation_feedback(path)
        cert,plan,feedback = proof['certificate'],proof['plan'],proof['feedback']
        contract = contract_from_dict(plan['contract'])
        expected = batch.predictions_for(action_id)
        # C0 has an internal operational activation contract, never a mechanism.
        if (cert['candidate']!=asdict(action.artifact) or cert['baseline']['version']!=batch.parent
                or (self.features['mechanisms'] and contract.predictions!=expected)):
            raise ValueError('Foreign artifact, action or hypothesis cannot update native CMVS')
        key = contract.id
        if key in self.proofs:
            if self.proofs[key]!=cert:
                raise ValueError('Accepted evidence changed')
            return self.feedback_for(key)
        self.proofs[key] = cert
        rows = proof['observations']
        self.utility[key] = {'artifact':contract.artifact,'scope':contract.scope,
            'feedback_type':'UTILITY_VERIFICATION','n_tasks':len(rows),
            'before_mean':sum(r['before'] for r in rows)/len(rows),
            'after_mean':sum(r['after'] for r in rows)/len(rows),
            'attribution_available':False}
        if self.features['mechanisms']:
            for claim in feedback['claims']:
                revision = item['mechanisms'][claim['mechanism']]
                record = self.records[revision]
                projected = project_claim(claim,self.level)
                entry = {'claim':projected,'contract_id':key,'artifact':contract.artifact,
                    'scope':contract.scope,'source_tasks':list(contract.tasks),
                    'ledger':cert['ledger'],'ledger_sha256':cert['sha256']}
                record['responses'][key] = entry
                # Foreign-action responses never overwrite own source/response slots.
                if claim['action_repairs_source']:
                    record['source_views'][contract.scope] = projected
                if self.features['bridge']:
                    self._joint_targets(record,entry)
            if self.features['probe']:
                self._discriminate(item,action_id,feedback,key)
        return self.feedback_for(key)

    def _joint_targets(self,record,entry):
        claim = entry['claim']
        kinds = []
        if claim['failure']=='CONTRADICTED':
            kinds.append('REVISE_SOURCE_PREDICATE')
        if claim['boundary']=='CONTRADICTED' or claim['mediator']=='CONTRADICTED':
            kinds.append('REVISE_RESPONSE_BRANCH')
        if claim['bridge']=='COMPENSATION':
            kinds.append('TEST_REMAINING_SOURCE_RISK')
        if claim['bridge']=='SOURCE_REPAIR' and claim['contribution']!='POSITIVE':
            kinds.append('TEST_RESIDUAL_FAULT_OR_UNDERPOWERED_EFFECT')
        if claim['contribution']=='POSITIVE' and claim['bridge']=='UNRESOLVED':
            kinds.append('DISCRIMINATE_REPAIR_FROM_COMPENSATION')
        if not kinds:
            kinds.append('RESOLVE_FROZEN_PREDICATES')
        for kind in kinds:
            target = {'kind':kind,'revision':record['revision'],'artifact':entry['artifact'],
                'scope':entry['scope'],'contract_id':entry['contract_id'],
                'status':'PROPOSED','causal_identification':False}
            if target not in self.targets:
                self.targets.append(target)

    def _discriminate(self,item,action_id,feedback,contract_id):
        from itertools import combinations
        predictions = {p.mechanism:{'boundary':dict(p.intervention_boundary),'mediator':dict(p.mediator)}
                       for p in item['batch'].predictions_for(action_id)}
        claims = {c['mechanism']:c for c in feedback['claims']}
        for left,right in combinations(sorted(predictions),2):
            for family in ('boundary','mediator'):
                lhs,rhs = predictions[left][family],predictions[right][family]
                for field in sorted(lhs.keys() & rhs.keys()):
                    if lhs[field]==rhs[field]:
                        continue
                    states = [claims[m].get('predicate_evidence',{}).get(family,{}).get('predicates',{}).get(field,{}).get('status')
                              for m in (left,right)]
                    # Legacy matches, aggregate labels and cross-family evidence
                    # cannot identify this frozen response discriminator.
                    if set(states)!={'CONSISTENT','CONTRADICTED'}:
                        continue
                    for m,status in zip((left,right),states):
                        record = self.records[item['mechanisms'][m]]
                        record['discriminations'].append({'contract_id':contract_id,'family':family,'field':field,
                            'status':'PREDICTION_SUPPORTED' if status=='CONSISTENT' else 'PREDICTION_CONFLICT',
                            'causal_identification':False})

    def feedback_for(self,contract_id):
        cert = self.proofs[contract_id]
        claims = []
        for record in self.records.values():
            entry = record['responses'].get(contract_id)
            if entry:
                claim = copy.deepcopy(entry['claim'])
                claim['mechanism_status'] = record['status']
                if self.features['bridge']:
                    claim['next_targets'] = [copy.deepcopy(t) for t in self.targets
                        if t['revision']==record['revision'] and t['contract_id']==contract_id]
                claims.append(claim)
        # Basic verify/reflect outcomes are available at every level. This is
        # not typed contribution attribution and never includes EVAL results.
        claims.append(copy.deepcopy(self.utility[contract_id]))
        return {'source_tasks':cert['confirmation_tasks'],'claims':claims,
                'artifact':cert['candidate']['version'],
                'scope':next((c['scope'] for c in claims),'terminalbench-trial')}

    def deploy(self,artifact):
        if artifact.version==self.parent:
            return
        if artifact.parent!=self.parent or artifact!=Artifact.create(artifact.content,artifact.parent,artifact.kind):
            raise ValueError('Only a directly selected exact child may replace the parent')
        self.parent = artifact.version
        for record in self.records.values():
            if record['parent']!=self.parent:
                record['status'] = 'STALE'
        for target in self.targets:
            target['status'] = 'STALE'

    def snapshot(self):
        return json.loads(json.dumps({'schema':'dace-native-cmvs/v1','parent':self.parent,
            'cap':self.cap,'level':self.level.value,'records':self.records,'targets':self.targets,
            'proofs':self.proofs,'utility':self.utility,'sequence':self.sequence}))
