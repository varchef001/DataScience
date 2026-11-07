"""Load shared settings once; only this module reads environment variables."""
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Immutable settings shared by the UI, agent, tools and evaluation."""
    default_model: str
    models: tuple
    max_steps: int
    timeout_seconds: int
    max_input_chars: int
    max_response_chars: int
    timezone: str
    notes_folder: Path
    log_file: Path
    api_key: str = field(default='', repr=False)


@lru_cache(maxsize=1)
def get_settings():
    """Read config.json and environment overrides once per Python process."""
    data = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    environment = {name: os.environ.get(name, '')
                   for name in ('OPENAI_API_KEY', 'OPENAI_MODEL')}
    model = environment['OPENAI_MODEL'] or data['default_model']
    models = tuple(dict.fromkeys([model] + data['models']))
    if not all(isinstance(item, str) and item.strip() for item in models):
        raise ValueError('Model names must be nonempty strings.')
    if type(data['max_steps']) is not int or not 1 <= data['max_steps'] <= 8:
        raise ValueError('max_steps must be between 1 and 8.')
    for key in ('timeout_seconds', 'max_input_chars', 'max_response_chars'):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    ZoneInfo(data['timezone'])
    return Settings(model, models, data['max_steps'], data['timeout_seconds'],
                    data['max_input_chars'], data['max_response_chars'],
                    data['timezone'], ROOT / data['notes_folder'],
                    ROOT / data['log_file'], environment['OPENAI_API_KEY'])
