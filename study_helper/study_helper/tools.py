"""The three tools used by the Study Helper Agent."""
import ast
import math
import operator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from config import get_settings

SETTINGS = get_settings()
NOTES_DIR = SETTINGS.notes_folder
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
             ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def checked_text(value, maximum):
    """Reject missing or oversized text and strip outer whitespace."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Input must be a nonempty string.')
    if len(value) > maximum:
        raise ValueError(f'Input must be at most {maximum} characters.')
    return value.strip()


def calculator(expression):
    """Evaluate a bounded arithmetic expression using an AST allowlist."""
    expression = checked_text(expression, 200)
    tree = ast.parse(expression, mode='eval')
    if len(list(ast.walk(tree))) > 80:
        raise ValueError('Expression is too complex.')

    def calculate(node):
        """Recursively evaluate allowed numeric nodes only."""
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            result = OPERATORS[type(node.op)](calculate(node.left), calculate(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = calculate(node.operand)
            if isinstance(node.op, ast.USub):
                result = -result
        else:
            raise ValueError('Only numbers, parentheses and + - * / // % are allowed.')
        if abs(result) > 10**15 or not math.isfinite(result):
            raise ValueError('Result is outside the supported range.')
        return result

    return {'result': calculate(tree.body)}


def current_date():
    """Return today and calendar metadata in the configured time zone."""
    today = datetime.now(ZoneInfo(SETTINGS.timezone)).date()
    leap = today.year % 4 == 0 and (today.year % 100 != 0 or today.year % 400 == 0)
    return {'date': today.isoformat(), 'timezone': SETTINGS.timezone,
            'day_of_year': today.timetuple().tm_yday,
            'days_in_year': 366 if leap else 365,
            'month_lengths': [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]}


def notes_search(query):
    """Return matching local note lines with filenames and line numbers."""
    query = checked_text(query, 100).casefold()
    if not NOTES_DIR.is_dir():
        raise FileNotFoundError('The notes folder is missing.')
    matches = []
    # Fixed folder, no model-supplied paths, and no symlink traversal.
    for path in sorted(NOTES_DIR.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in ('.txt', '.md'):
            continue
        if path.stat().st_size > 100_000:
            raise ValueError(f'Note exceeds 100 KB: {path.name}')
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if query in line.casefold():
                matches.append({'file': path.name, 'line': number, 'text': line[:500]})
                if len(matches) == 20:
                    return {'matches': matches, 'limit_reached': True}
    return {'matches': matches, 'limit_reached': False}


TOOLS = {'calculator': calculator, 'current_date': current_date, 'notes_search': notes_search}


def describe(name, description, properties):
    """Build a strict function schema for the model tool list."""
    return {'type': 'function', 'function': {
        'name': name, 'description': description, 'strict': True,
        'parameters': {'type': 'object', 'properties': properties,
                       'required': list(properties), 'additionalProperties': False}}}


TOOL_DESCRIPTIONS = [
    describe('calculator', 'Evaluate basic arithmetic with +, -, *, /, //, %, and parentheses; write percentages as /100 and omit thousands commas.',
             {'expression': {'type': 'string'}}),
    describe('current_date', 'Return today in the configured time zone with day-of-year and month lengths for calendar calculations.', {}),
    describe('notes_search', 'Search local .txt and .md notes for a case-insensitive literal topic phrase and return matching lines with filenames and line numbers.',
             {'query': {'type': 'string'}}),
]
