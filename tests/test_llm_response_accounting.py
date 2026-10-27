import io
import json
import unittest
from unittest.mock import patch

from dace.llm import LLM


class LLMResponseAccountingTests(unittest.TestCase):
    def complete(self,data):
        with patch('dace.llm.load_key',return_value='controlled-key-fixture'):
            client = LLM()
        with patch('dace.llm.urllib.request.urlopen',return_value=io.BytesIO(json.dumps(data).encode())):
            return client.complete([{'role':'user','content':'controlled test'}])

    def test_reasoning_only_response_retains_billing_and_finish_reason(self):
        content,usage = self.complete({'id':'fixture-request','model':'fixture-returned-model',
            'choices':[{'message':{'content':None},'finish_reason':'length'}],
            'usage':{'cost':.02,'prompt_tokens':3,'completion_tokens':6000}})
        self.assertIsNone(content)
        self.assertEqual(usage['cost'],.02)
        self.assertEqual(usage['_completion']['finish_reason'],'length')
        self.assertNotIn('controlled-key',json.dumps(usage))

    def test_empty_choices_do_not_discard_returned_usage(self):
        content,usage = self.complete({'choices':[],'usage':{'cost':.001}})
        self.assertIsNone(content)
        self.assertEqual(usage['cost'],.001)
        self.assertEqual(usage['_completion']['finish_reason'],'NO_CHOICES')


if __name__=='__main__':
    unittest.main()
