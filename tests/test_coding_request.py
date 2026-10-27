import io
import json
import unittest
from unittest.mock import patch

from dace.coding_request import validate_coding_request
from dace.llm import LLM


class CodingRequestProfileTests(unittest.TestCase):
    def test_control_profile_is_copied_without_hidden_defaults(self):
        self.assertEqual(validate_coding_request(None,12000),{})
        profile = {'reasoning':{'enabled':False},'response_format':{'type':'json_object'}}
        saved = validate_coding_request(profile,12000)
        profile['reasoning']['enabled'] = True
        self.assertFalse(saved['reasoning']['enabled'])

    def test_request_cannot_override_model_credentials_or_completion_budget(self):
        for key in ('model','messages','headers','api_key','max_tokens','provider'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_coding_request({key:'unauthorized override'},12000)

    def test_reasoning_budget_must_leave_output_room_and_match_its_mode(self):
        for reasoning in ({},{'enabled':'false'},{'max_tokens':True},{'max_tokens':0},
                          {'max_tokens':12000},{'enabled':False,'max_tokens':1000},
                          {'exclude':True}):
            with self.subTest(reasoning=reasoning),self.assertRaises(ValueError):
                validate_coding_request({'reasoning':reasoning},12000)
        self.assertEqual(validate_coding_request({'reasoning':{'max_tokens':2000}},12000),
                         {'reasoning':{'max_tokens':2000}})

    def test_transport_serializes_only_the_explicit_validated_profile(self):
        profile = {'reasoning':{'enabled':False},'response_format':{'type':'json_object'}}
        with patch('dace.llm.load_key',return_value='controlled-key'):
            client = LLM(model='controlled-model')
        response = {'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]}
        with patch('dace.llm.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as send:
            client.complete([{'role':'user','content':'JSON fixture'}],max_tokens=12000,request_options=profile)
        body = json.loads(send.call_args.args[0].data)
        self.assertEqual(body['reasoning'],profile['reasoning'])
        self.assertEqual(body['response_format'],profile['response_format'])
        self.assertEqual(body['max_tokens'],12000)
        self.assertEqual(body['model'],'controlled-model')

    def test_arbitrary_response_formats_are_not_silently_forwarded(self):
        for response_format in ('json',{},None,{'type':'text'},{'type':'json_object','extra':True}):
            with self.subTest(response_format=response_format),self.assertRaises(ValueError):
                validate_coding_request({'response_format':response_format},12000)
