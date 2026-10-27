"""Instrumented editable Terminus 2 integration pinned to Harbor 0.25.0."""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2, Terminus2Options

from .policy import PolicyRuntime, SourcePolicy


class EditableTerminusOptions(Terminus2Options):
    dace_policy_path: str | None = None
    dace_expected_policy: str | None = None
    dace_identity_path: str | None = None


class _ObservedParser:
    def __init__(self,parser,agent):
        self.parser,self.agent = parser,agent

    def __getattr__(self,name):
        return getattr(self.parser,name)

    def parse_response(self,response):
        before = self.parser.parse_response(response)
        transformed = self.agent.policy_runtime.response_transform(response)
        result = self.parser.parse_response(transformed)
        self.agent.boundaries.append({'raw_response_parse_valid':not bool(before.error),
                                       'response_parse_valid':not bool(result.error)})
        return result


class EditableTerminus(Terminus2):
    options_model = EditableTerminusOptions

    def __init__(self,logs_dir,dace_policy_path=None,dace_expected_policy=None,
                 dace_identity_path=None,**kwargs):
        if version('harbor') != '0.25.0':
            raise ValueError('EditableTerminus requires the verified Harbor 0.25.0 interface')
        if not dace_policy_path or not dace_expected_policy or not dace_identity_path:
            raise ValueError('Frozen policy and execution identity are required')
        self.policy = SourcePolicy.read(dace_policy_path)
        if self.policy.version != dace_expected_policy:
            raise ValueError('Frozen policy content changed before execution')
        self.identity = json.loads(Path(dace_identity_path).read_text(encoding='utf-8'))
        self.policy_runtime = PolicyRuntime(self.policy)
        self.boundaries = []
        self.dace_logs_dir = Path(logs_dir)
        super().__init__(logs_dir=logs_dir,dace_policy_path=dace_policy_path,
            dace_expected_policy=dace_expected_policy,dace_identity_path=dace_identity_path,**kwargs)

    @staticmethod
    def name():
        return 'dace-terminus'

    def version(self):
        return 'dace-terminus/v2:' + self.policy.version[:12]

    def _get_parser(self):
        return _ObservedParser(super()._get_parser(),self)

    async def _handle_llm_interaction(self,*args,**kwargs):
        result = list(await super()._handle_llm_interaction(*args,**kwargs))
        commands = [{'keystrokes':c.keystrokes,'duration_sec':c.duration_sec} for c in result[0]]
        adapted = self.policy_runtime.command_transform(commands)
        result[0] = [Command(**command) for command in adapted]
        self.boundaries.append({'command_batch_valid':True,
                                'command_count_preserved':len(adapted)==len(commands)})
        return tuple(result)

    async def _execute_commands(self,*args,**kwargs):
        result = await super()._execute_commands(*args,**kwargs)
        self.boundaries.append({'terminal_dispatch_completed':not result[0]})
        return result

    def _write_trace(self):
        fields = {k for row in self.boundaries for k in row}
        observed = {k:all(row[k] for row in self.boundaries if k in row) for k in fields}
        intervention = self.identity.get('intervention')
        executed = {e['hook'] for e in self.policy_runtime.events if e['executed']}
        activation = ({k:k in executed for k in intervention['patch_keys']}
                      if intervention is not None else {})
        self.dace_logs_dir.mkdir(parents=True,exist_ok=True)
        (self.dace_logs_dir/'dace-trace.json').write_text(json.dumps({
            'task_id':self.identity['task_id'],'intervention':intervention,
            'policy_version':self.policy.version,'boundary':observed,
            'activation_events':activation,'events':self.policy_runtime.events},indent=2),encoding='utf-8')

    async def run(self,instruction,environment,context):
        try:
            await super().run(self.policy_runtime.suffix(instruction),environment,context)
        finally:
            self._write_trace()
