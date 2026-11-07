"""Compare three phrasings per question with independent ground truth."""
import argparse
import csv
import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from agent import StudyHelperAgent
from config import ROOT, get_settings
from validation import contains_number, numbers_in


def load_cases():
    """Load the manually authored evaluation cases and reference expectations."""
    path = ROOT / 'evaluation' / 'ground_truth.json'
    return json.loads(path.read_text(encoding='utf-8'))['cases']


def reference_for(case, today):
    """Resolve date-dependent answers using Python, independently of the agent."""
    expected = dict(case)
    if case['id'] == 'today':
        expected['expected_date'] = today.isoformat()
        expected['reference_answer'] = f"Today's date is {today.isoformat()}."
    if case['id'] == 'combined':
        target = date(today.year, 12, 15)
        if target < today:
            target = date(today.year + 1, 12, 15)
        expected['expected_number'] = (target - today).days
        expected['reference_answer'] = (
            f'{(target - today).days} days until {target.isoformat()}; gradient descent '
            'updates parameters opposite the gradient to reduce loss.')
    return expected


def compare(result, expected):
    """Return transparent rule-based checks, not an LLM judge or exact-text score."""
    answer = result.answer.lower()
    used = {event['tool'] for event in result.trace}
    required = set(expected['required_tools'])
    checks = {'status': result.status == expected['expected_status'],
              'response_validation': not result.validation_errors and result.status in ('answered', 'unable'),
              'tool_selection': used == required}
    if 'expected_number' in expected:
        checks['numeric_answer'] = contains_number(answer, expected['expected_number'])
        if expected['id'] == 'combined':
            day_counts = re.findall(r'(\d+)\s+days?\b', answer)
            checks['day_count'] = bool(day_counts) and all(
                int(n) == expected['expected_number'] for n in day_counts)
        else:
            # Reject a wrong asserted answer even if the right number is quoted elsewhere.
            allowed = [17.5, 2340, 100, 0.175, expected['expected_number']]
            checks['no_unexpected_numbers'] = all(
                any(abs(n-a) < 1e-8 for a in allowed) for n in numbers_in(answer))
    if 'expected_date' in expected:
        dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', answer)
        checks['date_answer'] = bool(dates) and all(d == expected['expected_date'] for d in dates)
    if 'concept_groups' in expected:
        checks['note_concepts'] = all(any(term in answer for term in group)
                                      for group in expected['concept_groups'])
        checks['note_citations'] = any(c['file'] == 'machine_learning.md'
                                      and c['line'] in (2, 3, 4, 5) for c in result.citations)
    if expected['id'] == 'unknown':
        checks['clear_refusal'] = any(term in answer for term in
                                     ('cannot', "can't", 'unable', "don't know", 'cannot predict'))
        checks['no_invented_score'] = not numbers_in(answer)
    return {'checks': checks, 'passed': all(checks.values())}


def evaluate(agent, cases, output_dir, mode='live'):
    """Run each variant once and save full responses, checks and a summary."""
    settings = agent.settings
    rows = []
    for case in cases:
        for variant, prompt in enumerate(case['prompts'], 1):
            before = datetime.now(ZoneInfo(settings.timezone)).date()
            result = agent.process(prompt)
            after = datetime.now(ZoneInfo(settings.timezone)).date()
            expected_day = before
            # Handle a legitimate run crossing midnight without trusting an arbitrary date.
            if before != after:
                for event in result.trace:
                    if event['tool'] == 'current_date' and 'date' in event['output']:
                        observed = date.fromisoformat(event['output']['date'])
                        if observed in (before, after):
                            expected_day = observed
            expected = reference_for(case, expected_day)
            judgement = compare(result, expected)
            rows.append({'case': case['id'], 'variant': variant, 'prompt': prompt,
                         'mode': mode, 'model': settings.default_model,
                         'reference': expected['reference_answer'],
                         **result.to_dict(), **judgement})
    groups = []
    for case in cases:
        group = [row for row in rows if row['case'] == case['id']]
        groups.append({'case': case['id'], 'passed': sum(row['passed'] for row in group),
                       'total': len(group), 'all_three_correct': all(row['passed'] for row in group),
                       'same_status': len({row['status'] for row in group}) == 1,
                       'same_text': len({row['answer'] for row in group}) == 1})
    summary = {'mode': mode, 'total': len(rows), 'passed': sum(r['passed'] for r in rows),
               'groups': groups, 'note': 'Rule-based score; equivalent wording may fail and keyword overlap does not prove factual correctness. Review the answers.'}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'results.json').write_text(json.dumps({'summary': summary, 'results': rows}, indent=2), encoding='utf-8')
    with (output_dir / 'results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['case', 'variant', 'prompt', 'reference', 'answer', 'status', 'passed', 'checks'])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if key == 'checks' else row[key]
                             for key in writer.fieldnames})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate 15 live model responses')
    parser.add_argument('--output', default=str(ROOT / 'evaluation' / 'live_results'))
    args = parser.parse_args()
    settings = get_settings()
    if not settings.api_key:
        raise SystemExit('Set OPENAI_API_KEY first. No live evaluation was run.')
    summary = evaluate(StudyHelperAgent(settings), load_cases(), args.output)
    print(json.dumps(summary, indent=2))
