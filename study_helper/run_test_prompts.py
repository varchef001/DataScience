"""Record the five required prompts using the actual LLM API."""
import contextlib
from datetime import datetime
import os
from pathlib import Path
from agent import make_client, run_agent

PROMPTS = [
    'What is 17.5% of 2,340?',
    "What is today's date?",
    'What do my notes say about gradient descent?',
    'How many days until 15 December, and what do my notes say about gradient descent?',
    'What exact score will I get on my next exam?',
]

if __name__ == '__main__':
    client = make_client()  # Fail before creating a log if setup is missing.
    path = Path(__file__).with_name('test_log.txt')
    with path.open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        print('LIVE API TEST LOG')
        print('Run timestamp:', datetime.now().astimezone().isoformat())
        print('Model:', os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'))
        for number, prompt in enumerate(PROMPTS, 1):
            print(f'\n=== Test {number} ===\nPrompt: {prompt}')
            print('Returned answer:', run_agent(prompt, client=client))
    print(f'Saved {path.name}. Review tool choices and answers before submission.')
