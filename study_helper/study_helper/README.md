# Study Helper Agent

This is a small Python agent with a Streamlit UI. Enter a question, choose a
model, and click **Ask agent**. The model can use a calculator, today's date,
and a search of local study notes before giving one final response.

## Setup on Windows / VS Code

Open a terminal inside `study_helper` and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use Python 3.10 or newer. Set the API key with a hidden prompt:

```powershell
$secret = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.
Do not put the API key in config.json, source files, screenshots or GitHub.
API access is required for live requests; an ordinary chat subscription is not
used by this application. On macOS/Linux, activate with
`source .venv/bin/activate` and set OPENAI_API_KEY in that terminal.

## Configuration in one place

`config.json` stores the default model, model choices, time zone, eight-step
limit, input/response limits, timeout, notes folder and log path. Model choices
are configuration, not a guarantee of access on your API account.

`config.py` reads this file and environment variables once using a cached
`get_settings()` function. Other files reuse the same Settings object. Only
this module accesses environment variables. OPENAI_MODEL can override the
default model; OPENAI_API_KEY supplies the secret. Restart Streamlit after
changing config.json or environment variables. No .env loader is used.

## One run per user request

The form submits only when **Ask agent** is clicked. Each browser session
keeps one UserSession and one StudyHelperAgent. The same trimmed prompt and
model reuse the saved result, including errors, until **Clear results / allow
a new attempt** is clicked. Editing a widget, opening an expander or rerendering
the page does not call the model again. A new browser session has a new cache.

One user request can still require several model/tool calls. That is the agent
loop required by the assignment, capped at eight iterations. It is not eight
independent user runs. Automatic API retries and final-answer retries are off.
The evaluation script intentionally sends 15 separate prompts and is never run
automatically by the UI.

## Class-based design

`StudyHelperAgent` in agent.py owns configuration, the reusable API client,
logging and response validation. `process(prompt, model)` handles one request
and returns an AgentResult with the answer, status, citations, steps, trace and
validation errors. `call_model()` asks the LLM for the next action; `_run_tool()`
executes that action. `_finish()` logs the final result once.

`UserSession` handles duplicate UI submissions. `RequestLogger` handles logs,
and `ResponseValidator` handles checks. Docstrings explain the classes and
methods. The tools remain simple functions so they are easy to read and test.

## Tools and safeguards

- calculator: safe AST arithmetic with +, -, *, /, //, %, parentheses and unary signs; no eval. Use /100 for percentages. Limit: 200 characters, 80 AST nodes and finite results up to 10^15 in magnitude.
- current_date: today's ISO date plus calendar metadata in the configured timezone.
- notes_search: case-insensitive literal search of UTF-8 .txt/.md files directly in notes/. Returns filenames, line numbers and text. Limit: 100-character query, 100 KB per file, 20 matches of up to 500 characters. No symlink traversal.

The loop catches tool errors and sends them back to the model. Empty or oversized
user input is rejected. Invalid model protocol responses and API failures stop
with a readable error. The model is instructed to treat notes as data and say
when its tools cannot answer. These instructions do not guarantee perfect behavior.

## Request and response logging

Each processed request receives a unique request_id. `logs/agent.jsonl` records:

- request: prompt and selected model;
- model_request and model_response: each internal step;
- tool: selected function, arguments and result/error;
- response: final answer, status and validation details.

Each event has a UTC timestamp. The log rotates at 2 MB with two backups. The
configured key is redacted, but logs contain prompts and retrieved notes, so do
not enter other secrets. Logs are excluded by .gitignore and from the ZIP.
A cached UI response does not create a new agent request or model call.

## Response validation

The model is instructed to finish with JSON containing answer, status and
citations. Before displaying the answer, validation checks:

1. Valid JSON, exact field names, a nonempty bounded answer, and a supported status.
2. Citations refer to lines actually returned by notes_search.
3. An answered notes request with retrieved matches includes a citation.
4. An answered calculator request includes the last calculator result.
5. A date-only answer matches the date returned by the tool.

A failed answer is marked invalid and withheld from the main answer panel.
Validation reasons are shown in the UI, and the raw model output remains in the
local log. The agent does not automatically retry. These checks do not prove
all claims are true: a cited line may be misinterpreted or a number may occur in
an incorrect explanation. Ground-truth evaluation adds task-specific checks.

## Three phrasings and ground truth

`evaluation/ground_truth.json` contains five question groups with three phrasings
each. For example:

1. What is 17.5% of 2,340?
2. Calculate 17.5 percent of 2340.
3. If I take 17.5 percent of 2,340, how much is that?

All three have the same expected answer: **409.5**. The other groups cover today's
date, gradient descent notes, days until December 15 plus notes, and an unknown
future exam score. The reference notes are manually authored sample notes.
Calendar answers are computed independently with Python datetime, not an LLM.
Dates are evaluated in the configured timezone, including leap years and year
rollover. On November 2, 2026, there are 43 days until December 15.

Run the actual model comparison with:

```powershell
python evaluate.py
```

This makes 15 separate agent runs. It saves each prompt, reference answer,
actual response, trace and checks in evaluation/live_results/results.json and
results.csv. The summary shows how many variants passed for each question,
whether all three were correct, and whether their status/text matched. Answers
can have different wording and still pass. The scorer checks required tool
sets, numbers, dates, note concepts, citations and refusal behavior.

This is rule-based evaluation, not a second LLM judge. It can reject legitimate
paraphrases or accept keyword overlap, so read the saved answers too. A matching
status or identical text alone is not proof of correctness. Edit the notes and
their ground truth together if you change the sample content.

## Commands

```powershell
python smoke_test.py
python agent.py "What is 17.5% of 2,340?"
python run_test_prompts.py
python evaluate.py
python -m unittest -v test_local test_ui
python offline_demo.py
```

The smoke test makes one plain API call. run_test_prompts.py records the five
original assignment questions in test_log.txt. The unittest command runs code
and Streamlit UI checks without a real model call. offline_demo.py compares all
15 variants using scripted model replies and real tools; it writes clearly
labeled results under evaluation/offline_results/.

## What was actually verified

All 22 offline tests passed, including the real Streamlit form through AppTest,
duplicate submissions, validation failures, log redaction and ground-truth
comparison. The 15 scripted demonstration variants passed their rule checks.
This verifies the code paths and evaluation harness, not LLM accuracy.

No API key was available during preparation. Live API behavior and paraphrase
consistency remain unmeasured. Run the smoke test, five assignment tests and
live evaluation before submitting; update reflection.md using actual observations.

## Files

| File / folder | Purpose |
|---|---|
| app.py | Streamlit UI |
| agent.py | StudyHelperAgent and AgentResult classes; CLI entry point |
| config.json, config.py | Shared configuration and one-time environment loading |
| session.py | One agent per user session and duplicate-request cache |
| tools.py, notes/ | Three tools and sample study notes |
| request_log.py, logs/ | JSON event logger and generated private logs |
| validation.py | Final-response checks |
| evaluate.py, evaluation/ground_truth.json | Three-phrasing evaluation and reference answers |
| evaluation/offline_results/ | Scripted demonstration comparison, not live evidence |
| smoke_test.py | Plain model connection check |
| run_test_prompts.py | Five original assignment tests |
| test_local.py, test_ui.py | Automated local and UI tests |
| local_test_results.txt | Recorded automated test results |
| offline_demo.py, offline_demo_log.txt | Scripted demonstration and summary |
| test_cases.md | Short test checklist |
| reflection.md | Reflection draft; empirical model observations still pending |

## References

- OpenAI tool calling: https://developers.openai.com/api/docs/guides/function-calling
- Streamlit forms: https://docs.streamlit.io/develop/concepts/architecture/forms
- Streamlit session state: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
