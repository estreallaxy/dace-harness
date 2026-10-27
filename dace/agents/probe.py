"""Controlled real-parser probes; no model, terminal or benchmark reward."""
from dataclasses import asdict
from importlib.metadata import version

from .policy import PolicyRuntime,SourcePolicy
from ..core import digest


def probe_policy(policy,responses):
    if version('harbor')!='0.25.0':
        raise ValueError('Controlled parser probes require Harbor 0.25.0')
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
    return _probe_with_parser(policy,responses,TerminusJSONPlainParser())


def _probe_with_parser(policy,responses,parser):
    """Shared pure probe logic; injected test parsers never become native scores."""
    if (not isinstance(responses,list) or not responses or len(responses)>512
            or any(not isinstance(r,dict) or set(r)!={'id','response'}
                   or not isinstance(r['id'],str) or not r['id']
                   or not isinstance(r['response'],str) or len(r['response'])>1_000_000 for r in responses)
            or len({r['id'] for r in responses})!=len(responses)):
        raise ValueError('Controlled probes need bounded, uniquely identified response fixtures')
    rows = []
    for row in responses:
        runtime = PolicyRuntime(policy)
        raw = parser.parse_response(row['response'])
        transformed = runtime.response_transform(row['response'])
        parsed = parser.parse_response(transformed)
        commands = [{'keystrokes':c.keystrokes,'duration_sec':min(c.duration,60)} for c in parsed.commands]
        input_count = len(commands)
        commands = runtime.command_transform(commands)
        rows.append({'id':row['id'],'boundary':{'raw_response_parse_valid':not bool(raw.error),
            'response_parse_valid':not bool(parsed.error),'command_batch_valid':True,
            'command_count_preserved':len(commands)==input_count},
            'command_count':len(commands),'command_input_count':input_count,'events':runtime.events,
            'input_hash':digest(row['response']),'transformed_hash':digest(transformed)})
    return {'kind':'CONTROLLED_NATIVE_PARSER_PROBE','oracle_access':False,
        'benchmark_score_eligible':False,'model_calls':0,'terminal_commands_executed':0,
        'policy_version':policy.version,'policy':asdict(policy),'rows':rows,
        'limitations':['synthetic response fixtures','no utility/reward or semantic task correctness measured',
                       'instruction suffix and terminal dispatch hooks are not exercised']}
