"""Pure fixture tests; a separate optional native-parser run verifies integration."""
import json
import unittest
from types import SimpleNamespace

from dace.agents.probe import _probe_with_parser
from dace.controlled_faults import run_controlled_faults


class FixtureParser:
    def parse_response(self,value):
        try:
            data = json.loads(value)
            commands = [SimpleNamespace(keystrokes=c['keystrokes'],duration=c['duration']) for c in data['commands']]
            return SimpleNamespace(error='',commands=commands)
        except (ValueError,KeyError):
            return SimpleNamespace(error='invalid fixture',commands=[])


class ControlledFaultTests(unittest.TestCase):
    def setUp(self):
        responses = [{'id':'fixture','response':json.dumps({'analysis':'controlled','plan':'controlled',
            'commands':[{'keystrokes':'printf fixture\n','duration':.1}]})}]
        report = run_controlled_faults(responses,probe=lambda p,r:_probe_with_parser(p,r,FixtureParser()))
        self.cases = {c['scenario']:c for c in report['scenarios']}
        self.report = report

    def action(self,scenario,action):
        return next(a['observations'][0] for a in self.cases[scenario]['interventions'] if a['action']==action)

    def test_source_repair_restores_parser_contract(self):
        self.assertFalse(self.cases['source_error']['baseline'][0]['boundary']['response_parse_valid'])
        self.assertTrue(self.action('source_error','source-repair')['boundary']['response_parse_valid'])

    def test_compensation_restores_commands_without_repairing_source(self):
        row = self.action('downstream_compensation','compensation')
        self.assertFalse(row['boundary']['response_parse_valid'])
        self.assertEqual(row['command_count'],1)
        self.assertFalse(row['boundary']['command_count_preserved'])

    def test_compound_fault_masks_the_first_source_repair(self):
        partial = self.action('compound_fault','response-only')
        self.assertTrue(partial['boundary']['response_parse_valid'])
        self.assertTrue(partial['boundary']['command_batch_valid'])
        self.assertFalse(partial['boundary']['command_count_preserved'])
        self.assertEqual(partial['command_count'],0)
        full = self.action('compound_fault','joint-repair')
        self.assertTrue(full['boundary']['command_count_preserved'])
        self.assertEqual(full['command_count'],1)

    def test_wrong_initial_location_is_explicit_control_metadata(self):
        case = self.cases['mislocalization']
        self.assertEqual(case['initial_location'],'parser')
        self.assertEqual(case['known_source_slots'],('command_code',))
        self.assertTrue(case['baseline'][0]['boundary']['response_parse_valid'])
        self.assertFalse(case['baseline'][0]['boundary']['command_count_preserved'])

    def test_no_surrogate_is_presented_as_official_reward(self):
        self.assertFalse(self.report['official_rewards_measured'])
        self.assertEqual(self.report['terminal_commands_executed'],0)
        self.assertEqual(self.report['model_calls'],0)


if __name__=='__main__':
    unittest.main()
