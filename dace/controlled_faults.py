"""Known native policy faults for E4 preparation, not benchmark rewards.

Fixtures exercise actual pure policy transforms and (by default) Harbor's
parser. No terminal command, model request, or task verifier is executed.
Known constructors supply ground truth; they do not prove semantic causes
on naturally occurring benchmark failures.
"""
from dataclasses import dataclass

from .agents.policy import SourcePolicy
from .agents.probe import probe_policy


INVALID_RESPONSE = 'def transform(value):\n    return "CONTROLLED_INVALID_JSON"\n'
DROP_COMMANDS = 'def transform(value):\n    return []\n'
COMPENSATE_COMMANDS = ('def transform(value):\n'
    '    return [{"keystrokes":"printf controlled_probe_only\\n","duration_sec":0.1}]\n')


@dataclass(frozen=True)
class ControlledFault:
    name: str
    baseline: SourcePolicy
    actions: tuple
    source_slots: tuple
    initial_location: str


def controlled_faults():
    return (
        ControlledFault('source_error',SourcePolicy(response_code=INVALID_RESPONSE),
            (('source-repair',SourcePolicy()),),('response_code',),'response_transform'),
        ControlledFault('downstream_compensation',SourcePolicy(response_code=INVALID_RESPONSE),
            (('compensation',SourcePolicy(response_code=INVALID_RESPONSE,command_code=COMPENSATE_COMMANDS)),
             ('source-repair',SourcePolicy())),('response_code',),'response_transform'),
        ControlledFault('compound_fault',SourcePolicy(response_code=INVALID_RESPONSE,command_code=DROP_COMMANDS),
            (('response-only',SourcePolicy(command_code=DROP_COMMANDS)),
             ('command-only',SourcePolicy(response_code=INVALID_RESPONSE)),
             ('joint-repair',SourcePolicy())),('response_code','command_code'),'response_transform'),
        ControlledFault('mislocalization',SourcePolicy(command_code=DROP_COMMANDS),
            (('command-repair',SourcePolicy()),),('command_code',),'parser'),
    )


def run_controlled_faults(responses,probe=probe_policy):
    reports = []
    for case in controlled_faults():
        before = probe(case.baseline,responses)
        interventions = []
        for name,policy in case.actions:
            after = probe(policy,responses)
            interventions.append({'action':name,'policy_version':policy.version,
                'observations':after['rows']})
        reports.append({'scenario':case.name,'known_source_slots':case.source_slots,
            'initial_location':case.initial_location,'baseline_policy':case.baseline.version,
            'baseline':before['rows'],'interventions':interventions})
    return {'schema':'dace-controlled-native-faults/v1','formal_benchmark_result':False,
        'kind':'CONTROLLED_PARSER_POLICY_FIXTURES','model_calls':0,'task_executions':0,
        'terminal_commands_executed':0,'official_rewards_measured':False,'scenarios':reports,
        'limitations':['known injected faults','synthetic response fixtures',
            'no C3/C5 optimizer comparison yet','no task success or population-causal claim']}
