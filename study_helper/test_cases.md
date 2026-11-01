# Test Cases

These are the test cases for my Study Helper Agent. The expected answers are
listed below. The five tests with the actual model still need to be run with
an API key.

## 1. Percentage calculation

Input: What is 17.5% of 2,340?

Expected: The agent should use the calculator and return 409.5.
The calculation is 17.5 / 100 * 2340.

## 2. Today's date

Input: What is today's date?

Expected: The agent should use the date tool and return today's date in the
America/Chicago time zone.

## 3. Searching notes

Input: What do my notes say about gradient descent?

Expected: The agent should search the notes and explain that gradient descent
reduces loss by updating parameters in the opposite direction of the gradient.
It should include the filename and line numbers from the notes.

## 4. Using more than one tool

Input: How many days until 15 December, and what do my notes say about gradient descent?

Expected: The agent should use all three tools. It should get today's date,
calculate the days until December 15, and search for gradient descent.
For example, if today is November 1, 2026, the answer should be 44 days.
The count should not include today. It should also give a short summary of
the matching notes.

## 5. Question the agent cannot answer

Input: What exact score will I get on my next exam?

Expected: The agent should say it cannot determine my exam score using the
available tools. It should not make up a score.

## Extra checks

- Empty input: Show an input error without calling the model.
- Input over 2,000 characters: Reject it with an input error.
- Division by zero: Return a tool error to the model without crashing.
- Unsafe calculator input: Reject code or function calls.
- Topic not in the notes: Return no matches instead of making up notes.
- Repeated tool calls: Stop after eight iterations and show the step-limit message.

## Running the tests

Run the five main tests with:

```bash
python run_test_prompts.py
```

The output is saved in `test_log.txt`. Check the tool calls and final answers
against the expected results above.

Run the local checks without an API key with:

```bash
python -m unittest -v test_local
```

All 12 local tests passed in the recorded run. The offline demonstration uses
scripted model responses, so it does not show whether the real model picks
the right tools. The five live tests are still pending.
