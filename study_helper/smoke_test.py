"""Confirm one plain model call works before running the agent."""
import os
from agent import make_client

if __name__ == '__main__':
    client = make_client()
    result = client.chat.completions.create(
        model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
        messages=[{'role': 'user', 'content': 'Reply with: Connection works.'}])
    print(result.choices[0].message.content)
