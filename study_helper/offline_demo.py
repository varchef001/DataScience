"""Scripted model stand-in. Executes real tools, NOT a live model evaluation."""
import json
from datetime import date
from agent import run_agent
from run_test_prompts import PROMPTS
from test_local import tool_reply
from tools import current_date


def scripted_model(actions):
    actions = iter(actions)
    def respond(messages):
        action = next(actions)
        if isinstance(action, tuple):
            return tool_reply(action[0], json.dumps(action[1]))
        return {'role': 'assistant', 'content': action}
    return respond


if __name__ == '__main__':
    today = date.fromisoformat(current_date()['date'])
    target = date(today.year, 12, 15)
    if target < today:
        target = date(today.year + 1, 12, 15)
    expression = f'{target.toordinal()} - {today.toordinal()}'
    scripts = [
        [('calculator', {'expression': '17.5 / 100 * 2340'}), '17.5% of 2,340 is 409.5.'],
        [('current_date', {}), f"Today's date is {today.isoformat()} (America/Chicago)."],
        [('notes_search', {'query': 'gradient descent'}),
         'The notes say gradient descent reduces loss by updating parameters opposite the gradient (machine_learning.md:2-4).'],
        [('current_date', {}), ('calculator', {'expression': expression}),
         ('notes_search', {'query': 'gradient descent'}),
         f'There are {(target-today).days} days until {target.isoformat()}, excluding today. Gradient descent updates parameters opposite the gradient (machine_learning.md:3).'],
        ['I cannot predict your exact exam score. My tools only calculate, read the current date, and search notes.'],
    ]
    print('OFFLINE DEMONSTRATION: scripted tool choices and final answers; real tool execution.')
    print('Not proof of live LLM behavior. The date difference is prepared by this test harness.')
    for number, (prompt, actions) in enumerate(zip(PROMPTS, scripts), 1):
        print(f'\n=== Test {number} ===\nPrompt: {prompt}')
        run_agent(prompt, model_call=scripted_model(actions))
