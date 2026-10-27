"""Content-addressed, editable code slots for an instrumented terminal harness.

These pure transformation slots are a declared initial edit surface, not a
claim to support arbitrary repository modifications. No task, solution, test,
filesystem handle or provider credential is passed into a transformation.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core import digest


IDENTITY = 'def transform(value):\n    return value\n'
SLOTS = frozenset({'response_code', 'command_code', 'instruction_suffix'})
SAFE_BUILTINS = {'str':str, 'int':int, 'float':float, 'bool':bool,
                 'len':len, 'isinstance':isinstance}
SAFE_METHODS = frozenset({'strip','lstrip','rstrip','replace','split','splitlines',
                          'join','startswith','endswith','get','copy','lower','upper'})
ALLOWED_NODES = (ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,
    ast.Assign,ast.If,ast.Expr,ast.Name,ast.Load,ast.Store,ast.Constant,
    ast.Call,ast.Attribute,ast.keyword,ast.Subscript,ast.Slice,
    ast.Dict,ast.List,ast.Tuple,ast.ListComp,ast.comprehension,
    ast.IfExp,ast.Compare,ast.Eq,ast.NotEq,ast.Is,ast.IsNot,ast.In,ast.NotIn,
    ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.BoolOp,ast.And,ast.Or,
    ast.UnaryOp,ast.Not,ast.USub,ast.BinOp,ast.Add)


def compile_transform(source: str):
    if not isinstance(source,str) or len(source) > 16000:
        raise ValueError('Transform source must be a bounded Python function')
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('Exactly one transform function is allowed')
    fn = tree.body[0]
    if (fn.name != 'transform' or fn.decorator_list or fn.returns
            or len(fn.args.args) != 1 or fn.args.args[0].arg != 'value'
            or fn.args.args[0].annotation or fn.args.posonlyargs or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg or fn.args.defaults):
        raise ValueError('Required signature: def transform(value)')
    for node in ast.walk(tree):
        if not isinstance(node,ALLOWED_NODES):
            raise ValueError(f'Unsupported code-slot syntax: {type(node).__name__}')
        if isinstance(node,ast.FunctionDef) and node is not fn:
            raise ValueError('Nested functions are outside the edit surface')
        if isinstance(node,ast.Name) and node.id.startswith('_'):
            raise ValueError('Private/runtime names are outside the edit surface')
        if isinstance(node,ast.Attribute) and node.attr not in SAFE_METHODS:
            raise ValueError('Unsupported method in pure transform')
        if isinstance(node,ast.Call):
            target = node.func
            if not (isinstance(target,ast.Name) and target.id in SAFE_BUILTINS
                    or isinstance(target,ast.Attribute) and target.attr in SAFE_METHODS):
                raise ValueError('Only declared pure operations may be called')
        if isinstance(node,ast.Assign) and any(not isinstance(t,ast.Name) for t in node.targets):
            raise ValueError('Transforms may assign local variables only')
    namespace = {'__builtins__':dict(SAFE_BUILTINS)}
    exec(compile(tree,'<dace-frozen-policy>','exec'),namespace)
    return namespace['transform']


@dataclass(frozen=True)
class SourcePolicy:
    response_code: str = IDENTITY
    command_code: str = IDENTITY
    instruction_suffix: str = ''
    schema: str = 'dace-terminus-policy/v1'

    def __post_init__(self):
        if self.schema != 'dace-terminus-policy/v1':
            raise ValueError('Unknown real harness policy schema')
        if not isinstance(self.instruction_suffix,str) or len(self.instruction_suffix)>16000:
            raise ValueError('Instruction edit exceeds declared scope')
        compile_transform(self.response_code)
        compile_transform(self.command_code)

    @property
    def content(self):
        return json.dumps(asdict(self),sort_keys=True,ensure_ascii=False)

    @property
    def version(self):
        return digest(asdict(self))

    @classmethod
    def read(cls,path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


class PolicyRuntime:
    def __init__(self,policy):
        self.policy = policy
        self.response = compile_transform(policy.response_code)
        self.commands = compile_transform(policy.command_code)
        self.events = []

    def response_transform(self,value):
        if not isinstance(value,str) or len(value)>1_000_000:
            raise ValueError('Invalid response transform input')
        transformed = self.response(value)
        if not isinstance(transformed,str) or len(transformed)>1_000_000:
            raise ValueError('Response transform violated its string contract')
        self.events.append({'hook':'response_code', 'executed':True,
                            'changed':transformed != value})
        return transformed

    def command_transform(self,value):
        transformed = self.commands(value)
        if not isinstance(transformed,list) or len(transformed)>1024:
            raise ValueError('Command transform must return a bounded list')
        for command in transformed:
            if not isinstance(command,dict) or set(command)!={'keystrokes','duration_sec'}:
                raise ValueError('Command transform changed the command schema')
            duration = command['duration_sec']
            if (not isinstance(command['keystrokes'],str) or len(command['keystrokes'])>65536
                    or type(duration) not in (float,int) or not math.isfinite(duration)
                    or not 0<=duration<=60):
                raise ValueError('Command transform violated terminal command contract')
        self.events.append({'hook':'command_code', 'executed':True,
                            'changed':transformed != value})
        return transformed

    def suffix(self,instruction):
        self.events.append({'hook':'instruction_suffix','executed':True,
                            'changed':bool(self.policy.instruction_suffix)})
        return instruction + self.policy.instruction_suffix
