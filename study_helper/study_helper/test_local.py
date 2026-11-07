"""Offline tests for tools, OOP workflow, validation, evaluation and deduplication."""
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch
from agent import StudyHelperAgent
from config import get_settings
from evaluate import compare, load_cases, reference_for
from offline_demo import DemoModel, final_reply, tool_reply
from request_log import RequestLogger
from session import UserSession
from tools import calculator, current_date, notes_search
from validation import ResponseValidator


class MemoryLogger:
    """Capture log events without creating files for most tests."""

    def __init__(self):
        """Start with no recorded events."""
        self.events = []

    def event(self, request_id, event, **data):
        """Save one event for assertions."""
        self.events.append({'request_id': request_id, 'event': event, **data})


class LocalTests(unittest.TestCase):
    """Check both successful requests and deliberately broken responses."""

    def make_agent(self, model_call):
        """Return an agent with deterministic model replies and an in-memory log."""
        return StudyHelperAgent(model_call=model_call, logger=MemoryLogger())

    def test_arithmetic(self):
        """The calculator respects operators and rejects dangerous input."""
        self.assertEqual(calculator('17.5 / 100 * 2340')['result'], 409.5)
        self.assertEqual(calculator('-(2+3)*4+21//2')['result'], -10)
        for expression in ["__import__('os')", '2**100000', 'True', '[1]', '1e999']:
            with self.assertRaises(ValueError):
                calculator(expression)
        with self.assertRaises(ZeroDivisionError):
            calculator('1/0')

    def test_tool_validation(self):
        """Tools reject empty, oversized and wrong-type inputs."""
        for value in ['', ' ' * 3, 'x' * 201, None]:
            with self.assertRaises(ValueError):
                calculator(value)
        with self.assertRaises(ValueError):
            notes_search('x' * 101)

    def test_date(self):
        """The tool's calendar metadata agrees with its ISO date."""
        data = current_date()
        self.assertEqual(date.fromisoformat(data['date']).timetuple().tm_yday, data['day_of_year'])
        self.assertEqual(sum(data['month_lengths']), data['days_in_year'])

    def test_notes(self):
        """Search is case-insensitive and returns empty results for absent terms."""
        self.assertEqual(len(notes_search('GRADIENT DESCENT')['matches']), 4)
        self.assertEqual(notes_search('quantum banana')['matches'], [])
        with patch('tools.NOTES_DIR', Path('/missing_study_notes')):
            with self.assertRaises(FileNotFoundError):
                notes_search('gradient')

    def test_bad_input_never_calls_model(self):
        """Invalid requests are logged but never sent to a model."""
        def forbidden(messages, model):
            self.fail('The model must not be called.')
        agent = self.make_agent(forbidden)
        for text in ['', 'x' * 2001]:
            self.assertEqual(agent.process(text).status, 'error')
        self.assertEqual([e['event'] for e in agent.logger.events],
                         ['request', 'response', 'request', 'response'])

    def test_missing_key(self):
        """Missing API configuration gives a controlled error."""
        agent = StudyHelperAgent(replace(get_settings(), api_key=''), logger=MemoryLogger())
        self.assertIn('OPENAI_API_KEY', agent.process('hello').answer)

    def test_errors_return_to_model(self):
        """All ordinary tool dispatch failures become model-visible tool errors."""
        for name, raw in [('calculator', '{'), ('calculator', '{"expression":"1/0"}'),
                          ('missing', '{}'), ('current_date', '{"extra":1}'),
                          ('calculator', '[]'), ('calculator', 'x' * 2001)]:
            def model(messages, selected):
                if messages[-1]['role'] == 'user':
                    reply = tool_reply(name, {})
                    reply['tool_calls'][0]['function']['arguments'] = raw
                    return reply
                self.assertIn('Tool error', json.loads(messages[-1]['content'])['error'])
                self.assertEqual(messages[-1]['tool_call_id'], 'demo_call')
                return final_reply('I cannot complete this request.', 'unable')
            self.assertEqual(self.make_agent(model).process('test').status, 'unable')

    def test_step_limit(self):
        """Eight model calls are allowed, with no ninth call or final retry."""
        calls = []
        def model(messages, selected):
            calls.append(1)
            return tool_reply('calculator', {'expression': '2+2'})
        result = self.make_agent(model).process('continue')
        self.assertEqual(result.status, 'stopped')
        self.assertEqual(len(calls), 8)

    def test_api_failure_and_malformed_reply(self):
        """Network and protocol failures terminate cleanly."""
        def fail(messages, selected):
            raise ConnectionError('secret server detail')
        result = self.make_agent(fail).process('test')
        self.assertNotIn('secret', result.answer)
        for response in [None, {'tool_calls': 'bad'}, {'tool_calls': [{}]}]:
            agent = self.make_agent(lambda m, s: response)
            self.assertEqual(agent.process('test').status, 'error')

    def test_invalid_response_is_blocked(self):
        """An unstructured answer is not displayed as a valid final answer."""
        result = self.make_agent(lambda m, s: {'content': '409.5'}).process('test')
        self.assertEqual(result.status, 'invalid')
        self.assertTrue(result.validation_errors)
        self.assertEqual(result.steps, 1)

    def test_fabricated_citation_and_wrong_number(self):
        """The validator rejects unsupported citations and numeric mismatches."""
        validator = ResponseValidator()
        trace = [{'tool': 'calculator', 'output': {'result': 409.5}}]
        bad = final_reply('The answer is 1409.5.', citations=[{'file': 'fake.md', 'line': 4}])
        _, errors = validator.validate(bad['content'], trace)
        self.assertEqual(len(errors), 2)
        _, errors = validator.validate(final_reply('409.50')['content'], trace)
        self.assertEqual(errors, [])

    def test_notes_need_retrieved_citations(self):
        """Note answers must cite a line that was actually retrieved."""
        trace = [{'tool': 'notes_search', 'output': notes_search('gradient descent')}]
        validator = ResponseValidator()
        _, errors = validator.validate(final_reply('Gradient descent reduces loss.')['content'], trace)
        self.assertTrue(errors)
        reply = final_reply('Gradient descent reduces loss.', citations=[{'file': 'machine_learning.md', 'line': 2}])
        self.assertEqual(validator.validate(reply['content'], trace)[1], [])

    def test_wrong_date_is_blocked(self):
        """A date-only answer must match the date returned by the tool."""
        trace = [{'tool': 'current_date', 'output': {'date': '2026-11-02'}}]
        validator = ResponseValidator()
        reply = final_reply("Today is 2026-11-03.")
        self.assertTrue(validator.validate(reply['content'], trace)[1])
        reply = final_reply("Today is 2026-11-02.")
        self.assertEqual(validator.validate(reply['content'], trace)[1], [])

    def test_duplicate_submission(self):
        """Submitting the same prompt twice runs the agent only once per session."""
        calls = []
        def model(messages, selected):
            calls.append(1)
            return final_reply('I cannot predict your score.', 'unable')
        session = UserSession(self.make_agent(model))
        first, cached = session.submit('exam score', get_settings().default_model)
        second, cached = session.submit(' exam score ', get_settings().default_model)
        self.assertTrue(cached)
        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(len(calls), 1)
        session.clear()
        session.submit('exam score', get_settings().default_model)
        self.assertEqual(len(calls), 2)

    def test_request_and_response_logging(self):
        """Each processed request gets one correlated final response event."""
        agent = self.make_agent(DemoModel())
        result = agent.process(load_cases()[0]['prompts'][0])
        events = agent.logger.events
        self.assertEqual(sum(e['event'] == 'request' for e in events), 1)
        self.assertEqual(sum(e['event'] == 'response' for e in events), 1)
        self.assertTrue(all(e['request_id'] == result.request_id for e in events))
        self.assertIn('model_response', [e['event'] for e in events])

    def test_disk_logs_redact_key(self):
        """A configured API key is not written even if it appears in a prompt."""
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(get_settings(), api_key='secret-test-value', log_file=Path(folder) / 'log.jsonl')
            logger = RequestLogger(settings)
            logger.event('id', 'request', prompt='secret-test-value')
            text = settings.log_file.read_text()
            self.assertNotIn('secret-test-value', text)
            self.assertIn('[REDACTED]', text)
            for handler in list(logger.logger.handlers):
                handler.close()
                logger.logger.removeHandler(handler)

    def test_ground_truth_dates(self):
        """Date truth handles same-day, next-year and leap-year boundaries."""
        case = next(c for c in load_cases() if c['id'] == 'combined')
        for day, expected in [(date(2026, 11, 2), 43), (date(2026, 12, 15), 0),
                              (date(2026, 12, 16), 364), (date(2028, 2, 28), 291)]:
            self.assertEqual(reference_for(case, day)['expected_number'], expected)

    def test_all_fifteen_scripted_variants(self):
        """Check the evaluation harness using clearly labeled scripted answers."""
        agent = self.make_agent(DemoModel())
        today = date.fromisoformat(current_date()['date'])
        for case in load_cases():
            for prompt in case['prompts']:
                result = agent.process(prompt)
                judgement = compare(result, reference_for(case, today))
                self.assertTrue(judgement['passed'], judgement)

    def test_evaluation_detects_wrong_answer(self):
        """A structurally valid but incorrect result fails ground-truth comparison."""
        result = self.make_agent(DemoModel()).process(load_cases()[0]['prompts'][0])
        result.answer = 'The answer is 410.'
        self.assertFalse(compare(result, load_cases()[0])['passed'])

    def test_config_cached(self):
        """All imports share one settings object without repeated environment reads."""
        self.assertIs(get_settings(), get_settings())


if __name__ == '__main__':
    unittest.main(verbosity=2)
