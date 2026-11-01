"""Offline tool and loop tests; no LLM behavior is claimed here."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from agent import run_agent
from tools import calculator, current_date, notes_search


def tool_reply(name, arguments):
    return {'role': 'assistant', 'tool_calls': [{'id': 'test_call', 'type': 'function',
            'function': {'name': name, 'arguments': arguments}}]}


class LocalTests(unittest.TestCase):
    def run_quietly(self, goal, model):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_agent(goal, model_call=model)

    def test_percentage(self):
        self.assertAlmostEqual(calculator('17.5 / 100 * 2340')['result'], 409.5)

    def test_arithmetic(self):
        self.assertEqual(calculator('-(2 + 3) * 4 + 21 // 2')['result'], -10)

    def test_unsafe_expression(self):
        for expression in ["__import__('os').system('echo unsafe')", '2**100000', 'True', '[1]', '1e999']:
            with self.assertRaises(ValueError):
                calculator(expression)

    def test_zero_division(self):
        with self.assertRaises(ZeroDivisionError):
            calculator('1/0')

    def test_date(self):
        from datetime import date
        data = current_date()
        self.assertEqual(date.fromisoformat(data['date']).timetuple().tm_yday, data['day_of_year'])
        self.assertEqual(sum(data['month_lengths']), data['days_in_year'])

    def test_search(self):
        results = notes_search('GRADIENT DESCENT')['matches']
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0]['line'], 2)
        self.assertEqual(notes_search('quantum banana')['matches'], [])

    def test_validation(self):
        for value in ['', ' ' * 3, 'x' * 201, None]:
            with self.assertRaises(ValueError):
                calculator(value)
        with self.assertRaises(ValueError):
            notes_search('x' * 101)
        def must_not_call(messages):
            self.fail('Invalid user input must not reach the model')
        for value in ['', 'x' * 2001]:
            self.assertIn('Input error', self.run_quietly(value, must_not_call))

    def test_errors_return_to_model(self):
        for name, args in [('calculator', '{'), ('calculator', '{"expression":"1/0"}'),
                           ('missing', '{}'), ('current_date', '{"extra":1}'),
                           ('calculator', '[]'), ('calculator', 'x' * 2001)]:
            def model(messages):
                if messages[-1]['role'] == 'user':
                    return tool_reply(name, args)
                self.assertEqual(messages[-1]['tool_call_id'], 'test_call')
                self.assertIn('Tool error', json.loads(messages[-1]['content'])['error'])
                return {'role': 'assistant', 'content': 'Error received.'}
            self.assertEqual(self.run_quietly('test', model), 'Error received.')

    def test_missing_folder(self):
        with patch('tools.NOTES_DIR', Path('/nonexistent_study_notes_folder')):
            with self.assertRaises(FileNotFoundError):
                notes_search('gradient')

    def test_step_limit(self):
        calls = []
        def model(messages):
            calls.append(1)
            return tool_reply('calculator', '{"expression":"2+2"}')
        self.assertIn('step limit', self.run_quietly('continue', model))
        self.assertEqual(len(calls), 8)

    def test_final_answer_stops(self):
        self.assertEqual(self.run_quietly('test', lambda m: {'content': 'Done'}), 'Done')

    def test_api_failure(self):
        def failing_model(messages):
            raise ConnectionError('simulated outage')
        self.assertIn('Model error', self.run_quietly('test', failing_model))


if __name__ == '__main__':
    unittest.main(verbosity=2)
