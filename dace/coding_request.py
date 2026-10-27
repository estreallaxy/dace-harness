"""Bounded, explicitly frozen optimizer request controls, never credentials."""
import json


class IncompleteCodingResponse(ValueError):
    """A provider declared a non-final completion; billing still applies."""


def validate_coding_request(value,max_tokens):
    if value is None:
        return {}
    if (not isinstance(value,dict) or set(value)-{'reasoning','response_format'}
            or type(max_tokens) is not int or max_tokens<=0):
        raise ValueError('Coding request only supports explicit reasoning and JSON-mode controls')
    if 'reasoning' in value:
        reasoning = value['reasoning']
        if (not isinstance(reasoning,dict) or not reasoning
                or set(reasoning)-{'enabled','max_tokens'}
                or ('enabled' in reasoning and type(reasoning['enabled']) is not bool)
                or ('max_tokens' in reasoning and (type(reasoning['max_tokens']) is not int
                    or not 0<reasoning['max_tokens']<max_tokens))
                or (reasoning.get('enabled') is False and 'max_tokens' in reasoning)):
            raise ValueError('Freeze a valid reasoning mode/budget smaller than the completion cap')
    if 'response_format' in value and value['response_format']!={'type':'json_object'}:
        raise ValueError('Only JSON-object response mode is supported')
    return json.loads(json.dumps(value))
