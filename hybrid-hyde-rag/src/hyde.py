
from . import state

def generate_hypothetical_document(query, generator):
    """Generate without looking at the corpus; this text is never answer evidence."""
    prompt = f'Write a short {state.DATASET} reference paragraph that would answer this question. Include likely technical terms.\nQuestion: ' + query + '\nParagraph:'
    return generator(prompt, max_new_tokens=72, do_sample=False)[0]['generated_text']

