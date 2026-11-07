# Test Cases

## Main questions

1. Ask for 17.5% of 2,340. The calculator should return 409.5.
2. Ask for today's date. The date tool should return the configured local date.
3. Ask about gradient descent in the notes. The answer should describe reducing loss and moving opposite the gradient, with a real citation.
4. Ask for days until December 15 and a notes summary. The agent should use all three tools and answer both parts.
5. Ask for an exact future exam score. The agent should say it cannot determine that score.

Each question has three phrasings in evaluation/ground_truth.json.
Run `python evaluate.py` to compare all 15 live responses with the expected
answers. Check results.json or results.csv under evaluation/live_results/.
These live results have not been generated here because no API key is available.

## UI checks

- Type a question and click Ask agent. The answer and tool trace should appear.
- Submit the same question and model again. It should reuse the answer without another API call.
- Open the validation or tool details. This should not start another run.
- Click Clear results, then submit again. This should allow an intentional new run.
- Submit an empty question. It should ask for a question without calling the model.

## Code checks

- Check correct arithmetic, case-insensitive search and calendar information.
- Reject unsafe expressions, empty input and oversized input.
- Return tool errors without crashing.
- Stop after eight iterations.
- Reject invalid final JSON, invented citations and mismatched calculator results.
- Log a request and its final response with the same request ID.
- Keep the configured API key out of the logs.
- Reject wrong answers when comparing with ground truth.

Run `python -m unittest -v test_local test_ui` for the automated checks.
All 22 tests passed in the recorded run. Scripted model responses are used for
offline checks. They do not prove the real LLM will behave the same way.
