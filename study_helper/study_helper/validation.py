"""Check answer structure and tool evidence without making another model call."""
import json
import math
import re


def numbers_in(text):
    """Extract numeric tokens so 409.50 matches 409.5 but not 1409.5."""
    return [float(item) for item in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?',
                                              text.replace(',', ''))]


def contains_number(text, expected):
    """Check whether an answer contains the expected numeric value."""
    return any(math.isclose(value, expected, abs_tol=1e-8, rel_tol=1e-9)
               for value in numbers_in(text))


class ResponseValidator:
    """Validate format, retrieved citations and the last calculator result.

    These checks are deliberately limited: they cannot prove that every natural
    language claim is true. The evaluation suite separately checks known answers.
    """

    def __init__(self, max_chars=6000):
        """Set a maximum length for the user-facing answer."""
        self.max_chars = max_chars

    def validate(self, content, trace):
        """Return parsed response and errors; do not retry or call another LLM."""
        errors = []
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            return {}, ['The final response must be valid JSON.']
        if not isinstance(payload, dict):
            return {}, ['The final response must be a JSON object.']
        if set(payload) != {'answer', 'status', 'citations'}:
            errors.append('Expected answer, status and citations fields only.')
        answer = payload.get('answer')
        if not isinstance(answer, str) or not answer.strip():
            errors.append('The answer is empty or is not text.')
        elif len(answer) > self.max_chars:
            errors.append('The answer is too long.')
        if payload.get('status') not in ('answered', 'unable'):
            errors.append('The response status is invalid.')
        citations = payload.get('citations')
        evidence = {(m['file'], m['line']) for event in trace
                    for m in event['output'].get('matches', [])}
        if not isinstance(citations, list):
            errors.append('Citations must be a list.')
        else:
            for citation in citations:
                if (not isinstance(citation, dict)
                    or set(citation) != {'file', 'line'}
                    or not isinstance(citation.get('file'), str)
                    or type(citation.get('line')) is not int):
                    errors.append('A citation has an invalid format.')
                elif (citation['file'], citation['line']) not in evidence:
                    errors.append('A citation was not returned by notes_search.')
            if evidence and payload.get('status') == 'answered' and not citations:
                errors.append('Retrieved notes need at least one citation.')
        results = [e['output']['result'] for e in trace
                   if e['tool'] == 'calculator' and 'result' in e['output']]
        if results and payload.get('status') == 'answered' and isinstance(answer, str):
            if not contains_number(answer, results[-1]):
                errors.append('The answer does not contain the last calculator result.')
        successful = [event for event in trace if 'error' not in event['output']]
        if (successful and {event['tool'] for event in successful} == {'current_date'}
                and payload.get('status') == 'answered' and isinstance(answer, str)):
            expected_date = successful[-1]['output']['date']
            reported_dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', answer)
            if not reported_dates or any(day != expected_date for day in reported_dates):
                errors.append('The answer does not match the date tool result.')
        return payload, errors
