import re
from . import state

def answer_prompt(query, contexts, generator):
    """Use only text that fits; save the exact shortened context for auditing."""
    tokenizer = generator.tokenizer
    question = tokenizer.decode(tokenizer.encode(query, add_special_tokens=False)[:100], skip_special_tokens=True)
    intro = f'Answer the question using only the context. If the context does not answer it, say Insufficient evidence. Give the answer first, then cite a source number such as [1].\nQuestion: {question}\nContext:\n'
    remaining = 480 - len(tokenizer.encode(intro + '\nAnswer:', add_special_tokens=False))
    used = []
    parts = []
    chosen = contexts[:3]
    for c in chosen:
        label = f'[{len(used) + 1}] '
        budget = max(1, remaining // max(1, len(chosen) - len(used)))
        label_tokens = len(tokenizer.encode(label, add_special_tokens=False))
        tokens = tokenizer.encode(c['text'], add_special_tokens=False)[:max(1, budget - label_tokens - 3)]
        excerpt = tokenizer.decode(tokens, skip_special_tokens=True)
        block = label + excerpt
        parts.append(block)
        remaining -= len(tokenizer.encode(block + '\n', add_special_tokens=False))
        used.append({'chunk_id': c['chunk_id'], 'source': c['source'], 'text': excerpt})
    return (intro + '\n'.join(parts) + '\nAnswer:', used)

def generate_answer(query, contexts, generator):
    if not query.strip():
        raise ValueError('Enter a non-empty question')
    if not contexts:
        return {'answer': 'Insufficient evidence.', 'model_text': 'Insufficient evidence.', 'sources': [], 'context': [], 'model_citations': []}
    prompt, used = answer_prompt(query, contexts, generator)
    raw = generator(prompt, max_new_tokens=48, do_sample=False)[0]['generated_text'].strip()
    first_output = raw
    retried = not re.search(r'[A-Za-z0-9]', re.sub(r'\[[^\]]*\]', '', raw))
    if retried:
        short_query = generator.tokenizer.decode(
            generator.tokenizer.encode(query, add_special_tokens=False)[:100], skip_special_tokens=True)
        retry_prompt = ('Answer using only this context. If the answer is absent, say Insufficient evidence.\n'
                        + 'Context: ' + '\n'.join(c['text'] for c in used)
                        + '\nQuestion: ' + short_query + '\nAnswer:')
        raw = generator(retry_prompt, max_new_tokens=48, do_sample=False)[0]['generated_text'].strip()
        if not re.search(r'[A-Za-z0-9]', re.sub(r'\[[^\]]*\]', '', raw)):
            raw = 'The model did not produce an answer.'
    supplied_ids = {c['chunk_id'] for c in used}
    number_to_id = {str(i + 1): c['chunk_id'] for i, c in enumerate(used)}
    raw = re.sub('\\[(\\d+)\\]', lambda m: '[' + number_to_id.get(m[1], m[1]) + ']', raw)
    citations = re.findall('\\[([^\\]]+)\\]', raw)
    valid = [s for s in citations if s in supplied_ids]
    cleaned = re.sub('\\[([^\\]]+)\\]', lambda m: m[0] if m[1] in supplied_ids else '', raw).strip()
    sources = [{'chunk_id': c['chunk_id'], 'source': c['source']} for c in used]
    answer = cleaned
    if not valid:
        answer += '\nContext sources supplied: ' + ' '.join((f"[{c['chunk_id']}]" for c in used))
    return {'answer': answer, 'model_text': raw, 'sources': sources, 'context': used,
            'model_citations': valid, 'retried_citation_only': retried, 'first_model_output': first_output}

