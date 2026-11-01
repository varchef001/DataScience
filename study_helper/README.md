# Study Helper Agent

This project uses one OpenAI model and three Python tools. A chatbot normally
answers directly, while this agent can choose a tool, read its result, and call
another tool before answering. I kept the program small so the loop is easy to follow.

## Verification status — read before submitting

The implementation and offline tests are complete. No API key was available in
the preparation environment, so the plain API smoke test and five live model
runs are **not verified**. `offline_demo_log.txt` records real tool execution
with explicitly scripted model replies; it is not a live test log. Run the
commands below to create `test_log.txt`. The reflection is a 200–300 word draft
that honestly distinguishes checked behavior from unobserved model behavior.
Update its first two answers after the live runs. Those empirical requirements
remain pending until then; do not submit the offline demonstration as live evidence.

## Setup

Use Python 3.10 or newer. Open a terminal inside this folder:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in the terminal environment. Do not put the key in source
code, screenshots, or the submitted folder. On macOS/Linux, hidden entry avoids
putting the key in shell history:

```bash
read -s -p 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
echo
```

In PowerShell, use a hidden prompt:

```powershell
$secret = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
```

The default model is `gpt-4o-mini`; optionally set `OPENAI_MODEL` to another
OpenAI model supporting Chat Completions function calling. API access and billing
are required. The program does not load a `.env` file automatically.

## Run in assignment order

```bash
python smoke_test.py
python -m unittest -v test_local
python agent.py "What is 17.5% of 2,340?"
python agent.py
python run_test_prompts.py
```

The first command confirms a plain model call. The second tests tools and loop
safeguards locally. The last command records all five required prompts, including
step numbers, tool names, arguments, results, and final answers in `test_log.txt`.
Inspect that file: successful execution alone does not prove correct answers.
To reproduce the clearly labeled offline demonstration:

```bash
python offline_demo.py > offline_demo_log.txt
```

## Tools

| Tool | Input | What it does |
|---|---|---|
| calculator | expression: string | Evaluates numbers and +, -, *, /, //, %, parentheses and unary signs using an AST allowlist; never uses eval. |
| current_date | empty object | Returns today's date in America/Chicago and calendar metadata. |
| notes_search | query: string | Searches the notes folder's .txt/.md files for a case-insensitive literal phrase and returns matching lines. |

Put UTF-8 notes directly in `notes/`. The included machine learning notes are
sample notes created for this project. Search is literal, not semantic, and does
not recurse into subfolders. Results include filenames and line numbers.
Percentages should be written as division by 100; `%` means remainder.

## Loop and safeguards

`agent.py` sends the system prompt, request and three JSON tool schemas to the
model. It appends the assistant's tool call and a matching tool-result message,
then repeats. A text-only response ends the loop. There are at most eight model
calls; even if the eighth asks for a tool, no ninth model call is made. The
program reports `Stopped: step limit reached (8 iterations).`

Empty requests and requests over 2,000 characters are rejected. Expressions are
limited to 200 characters and 80 AST nodes; results must be finite and at most
10^15 in magnitude. Queries are limited to 100 characters. Unknown tools,
malformed JSON, invalid arguments, missing notes, and arithmetic errors are
returned to the model as tool errors. Notes are capped at 100 KB per file and
20 matching lines of 500 characters each. Symlink files are ignored. API failures
return a concise error; retries are disabled and requests have a timeout.

For an unspecified year, December 15 means the next occurrence (today included).
The day count excludes today. The model receives month lengths and must use the
calculator for arithmetic. On October 28, 2026, the expected difference is 48 days.
The model still needs live testing for calendar reasoning and refusal behavior.

## Required test checks

| Prompt | Expected behavior |
|---|---|
| What is 17.5% of 2,340? | Calculator only; answer 409.5. |
| What is today's date? | Date tool only; answer matches America/Chicago date. |
| What do my notes say about gradient descent? | Notes search; grounded answer citing matching lines. |
| How many days until 15 December, and what do my notes say about gradient descent? | Date, calculator and notes search; correct date difference and grounded notes summary. |
| What exact score will I get on my next exam? | Explain this cannot be determined; do not invent a score. |

## Files and requirement coverage

- `agent.py`: one API, environment key, loop, eight-step limit, logging and errors.
- `tools.py`: three functions, validation and JSON input schemas.
- `notes/machine_learning.md`: sample study notes.
- `smoke_test.py`: plain API connection check.
- `run_test_prompts.py`: live log generator for all five prompts.
- `test_cases.md`: five required prompts and additional safeguard test cases.
- `test_local.py`, `local_test_results.txt`: local tool and safeguard checks.
- `offline_demo.py`, `offline_demo_log.txt`: labeled offline demonstration.
- `reflection.md`: 200–300 word reflection draft, with empirical limitations stated.

## Reference

OpenAI function calling guide (SDK integration reference):
https://developers.openai.com/api/docs/guides/function-calling
