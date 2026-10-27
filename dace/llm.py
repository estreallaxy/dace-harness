"""OpenRouter transport. Keys are loaded in memory and never written to runs."""
import json
import os
import urllib.request
from pathlib import Path
from .coding_request import validate_coding_request

DEFAULT_ENV = Path('.env.local')


def load_key(env_file=None):
    if os.environ.get('OPENROUTER_API_KEY'):
        return os.environ['OPENROUTER_API_KEY']
    selected = env_file or os.environ.get('DACE_ENV_FILE')
    if not selected:
        local = Path(__file__).resolve().parents[1]/'.dace.local.json'
        if local.is_file():
            selected = json.loads(local.read_text(encoding='utf-8')).get('env_file')
    path = Path(selected) if selected else DEFAULT_ENV
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip().startswith('OPENROUTER_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    raise ValueError('OPENROUTER_API_KEY is missing')


class LLM:
    def __init__(self, model='deepseek/deepseek-v4.1-flash', env_file=None):
        self.model, self.key = model, load_key(env_file)

    def complete(self, messages, max_tokens=1800,*,request_options=None):
        body = {'model': self.model, 'messages': messages, 'temperature': 0,
                'max_tokens': max_tokens}
        body.update(validate_coding_request(request_options,max_tokens))
        req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                    data=json.dumps(body).encode(), headers={
                        'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.load(response)
        except Exception as exc:
            # No HTTP body/headers or credentials enter errors.
            raise RuntimeError('LLM transport failed: ' + type(exc).__name__) from None
        choices = data.get('choices') or []
        choice = choices[0] if choices and isinstance(choices[0],dict) else {}
        message = choice.get('message') or {}
        usage = dict(data['usage']) if isinstance(data.get('usage'),dict) else {}
        # Retain billing even when a reasoning/length-limited response has no
        # user-facing text. No HTTP body, headers or credential enter metadata.
        usage['_completion'] = {'finish_reason':choice.get('finish_reason','NO_CHOICES'),
                                'returned_model':data.get('model'),'request_id':data.get('id')}
        return message.get('content') if isinstance(message,dict) else None, usage

    def propose(self, traces, budget, exposure, lineage):
        # A fixed conservative request-unit cap; monetary accounting is not inferred.
        budget.charge(10)
        exposure.disclose([t['task'] for t in traces], ['trace', 'component_graph'], lineage)
        prompt = '''You are the DACE coding agent. Propose competing failure-source mechanisms
and a configuration patch for a harness. Planner emits count as a string instead of integer;
adapter forwards count unchanged; tool requires integer. Configuration supports exactly two
boolean fields: planner_cast and adapter_cast. Return JSON only, with keys mechanisms (list
of {id, source, explanation}), patch ({planner_cast: bool, adapter_cast: bool}), reasoning (string).
An adapter patch may compensate for a planner defect; improvement does not prove root cause.
Observed diagnostic traces: ''' + json.dumps(traces)
        content, usage = self.complete([{'role': 'user', 'content': prompt}])
        clean = content.strip()
        if clean.startswith('```'):
            clean = clean.split('\n', 1)[1].rsplit('```', 1)[0]
        result = json.loads(clean)
        patch = result.get('patch', {})
        if set(patch) != {'planner_cast', 'adapter_cast'} or any(type(v) is not bool for v in patch.values()):
            raise ValueError('Patch violates allowed configuration schema')
        if not isinstance(result.get('mechanisms'), list) or len(result['mechanisms']) > 8:
            raise ValueError('Invalid mechanism proposals')
        exposure.ledger.append('AGENT_PROPOSAL', lineage=lineage, proposal=result, usage=usage,
                               status='PROPOSED')
        return result

    def propose_mechanisms(self, symptoms, registry, budget, exposure, lineage):
        """Live CMVS proposals for the evolution loop (experiment live mode).

        The agent selects/extends mechanism hypotheses from the frozen catalog
        schema; every emitted mechanism must name a real component, boundary
        source field and configuration repair key. Proposals stay PROPOSED --
        only confirmatory contracts turn them into claims.
        """
        from .components import MECHANISM_CATALOG, PROPAGATION
        budget.charge(10)
        # disclosure audit registers the ACTUAL source task IDs the symptom
        # statistics were derived from (carried by the summary), so the
        # provenance of agent-visible derived data is traceable -- not just
        # the field names of the summary itself.
        tasks = list(symptoms.get('source_tasks', []))
        exposure.disclose(tasks, ['symptom_rates', 'naive_localization'], lineage)
        catalog_view = [{'id': mid, 'component': comp, 'source_field': src, 'repair_key': rep,
                         'trigger_symptom': trig}
                        for mid, comp, src, trig, rep in MECHANISM_CATALOG]
        prompt = ('You are the DACE coding agent. Given harness diagnostic symptoms, select the '
                  'competing failure-source mechanisms worth confirmatory testing, ordered by '
                  'plausibility. Catalog schema: ' + json.dumps(catalog_view) +
                  '\nObserved symptom rates (fraction of diagnostic tasks violating each boundary '
                  'contract): ' + json.dumps(symptoms['symptom_rates']) +
                  '\nA naive locality heuristic suggests: ' + symptoms['naive_localization'] +
                  '. Note: a downstream boundary violation may only propagate an upstream fault; '
                  'symptom location is not root-cause evidence. Return JSON only: '
                  '{"mechanisms": [{"id": "...", "component": "...", "source_field": "...", '
                  '"repair_key": "..."}], "reasoning": "..."} with mechanisms drawn from the '
                  'catalog ids.')
        content, usage = self.complete([{'role': 'user', 'content': prompt}])
        clean = content.strip()
        if clean.startswith('```'):
            clean = clean.split('\n', 1)[1].rsplit('```', 1)[0]
        result = json.loads(clean)
        valid_ids = {c[0] for c in MECHANISM_CATALOG}
        valid = []
        for item in result.get('mechanisms', []):
            if item.get('id') in valid_ids and item.get('id') in PROPAGATION:
                for mid, comp, src, _, rep in MECHANISM_CATALOG:
                    if mid == item['id']:
                        valid.append({'id': mid, 'component': comp, 'source_field': src,
                                      'repair_key': rep})
                        break
        result['mechanisms'] = valid
        exposure.ledger.append('AGENT_PROPOSAL', lineage=lineage, proposal=result, usage=usage,
                               status='PROPOSED')
        return result
