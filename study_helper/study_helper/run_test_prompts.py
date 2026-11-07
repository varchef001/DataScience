"""Run the five original questions once each and save their full traces."""
import json
from config import ROOT, get_settings
from agent import StudyHelperAgent
from evaluate import load_cases

PROMPTS = [case['prompts'][0] for case in load_cases()]


def main():
    """Record live results only; missing configuration does not create a log."""
    settings = get_settings()
    if not settings.api_key:
        raise SystemExit('Set OPENAI_API_KEY first. No live tests were run.')
    agent = StudyHelperAgent(settings)
    with (ROOT / 'test_log.txt').open('w', encoding='utf-8') as log:
        log.write(f'LIVE API TEST LOG\nModel: {settings.default_model}\n')
        for number, prompt in enumerate(PROMPTS, 1):
            result = agent.process(prompt)
            log.write(f'\nTest {number}\nPrompt: {prompt}\n')
            log.write(json.dumps(result.to_dict(), indent=2) + '\n')
    print('Saved test_log.txt. Review the actual responses before submission.')


if __name__ == '__main__':
    main()
