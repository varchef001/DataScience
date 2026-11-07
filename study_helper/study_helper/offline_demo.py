"""Demonstrate real tools with scripted decisions, never as live model evidence."""
import json
from datetime import date
from agent import StudyHelperAgent
from config import ROOT
from evaluate import evaluate, load_cases


def tool_reply(name, arguments):
    """Build a simulated model tool request in the same shape as the API."""
    return {'role': 'assistant', 'tool_calls': [{'id': 'demo_call', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def final_reply(answer, status='answered', citations=None):
    """Build a simulated structured final response for offline tests."""
    return {'role': 'assistant', 'content': json.dumps(
        {'answer': answer, 'status': status, 'citations': citations or []})}


class DemoModel:
    """Supply fixed decisions for the 15 known prompts; this is not an LLM."""

    def __call__(self, messages, model):
        """Choose the next scripted action from the request and real tool outputs."""
        prompt = messages[1]['content']
        case = next(c for c in load_cases() if prompt in c['prompts'])['id']
        outputs = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        count = len(outputs)
        if case == 'unknown':
            return final_reply('I cannot predict your exact exam score with these tools.', 'unable')
        if case == 'percentage':
            if count == 0:
                return tool_reply('calculator', {'expression': '17.5 / 100 * 2340'})
            return final_reply(f"17.5% of 2,340 is {outputs[-1]['result']}.")
        if case == 'today':
            if count == 0:
                return tool_reply('current_date', {})
            return final_reply(f"Today's date is {outputs[-1]['date']}.")
        if case == 'notes':
            if count == 0:
                return tool_reply('notes_search', {'query': 'gradient descent'})
            return final_reply('Gradient descent reduces loss by updating parameters opposite the gradient.',
                               citations=[{'file': 'machine_learning.md', 'line': 3}])
        if count == 0:
            return tool_reply('current_date', {})
        if count == 1:
            today = date.fromisoformat(outputs[0]['date'])
            target = date(today.year, 12, 15)
            if target < today:
                target = date(today.year + 1, 12, 15)
            return tool_reply('calculator', {'expression': f'{target.toordinal()} - {today.toordinal()}'})
        if count == 2:
            return tool_reply('notes_search', {'query': 'gradient descent'})
        return final_reply(f"There are {outputs[1]['result']} days until the next December 15. "
                           'Gradient descent updates parameters opposite the gradient to reduce loss.',
                           citations=[{'file': 'machine_learning.md', 'line': 3}])


if __name__ == '__main__':
    agent = StudyHelperAgent(model_call=DemoModel())
    print('SCRIPTED OFFLINE DEMO: real tools, prepared decisions and answers. No LLM was called.')
    summary = evaluate(agent, load_cases(), ROOT / 'evaluation' / 'offline_results', mode='scripted_offline')
    print(json.dumps(summary, indent=2))
