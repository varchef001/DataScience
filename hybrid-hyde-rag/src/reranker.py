
from . import state

def rerank(query, candidates, top_k=10):
    if not candidates:
        return []
    scores = state.RERANKER.predict([(query, c['text']) for c in candidates], batch_size=16, show_progress_bar=False)
    rows = [{**c, 'reranker_score': float(s)} for c, s in zip(candidates, scores)]
    return sorted(rows, key=lambda c: -c['reranker_score'])[:top_k]

