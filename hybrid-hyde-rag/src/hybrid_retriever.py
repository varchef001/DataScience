import numpy as np
from .embeddings import query_embedding
from .bm25_retriever import bm25_search
from .dense_retriever import dense_search
from . import state

def normalized_scores(results):
    values = np.array([r['score'] for r in results], dtype=float)
    if len(values) == 0:
        return {}
    width = values.max() - values.min()
    normalized = (values - values.min()) / width if width > 1e-12 else np.zeros_like(values)
    return {r['chunk_id']: float(s) for r, s in zip(results, normalized)}

def fuse_results(lexical, dense, alpha):
    """Min-max per candidate list; absent candidates contribute zero."""
    b = normalized_scores(lexical)
    d = normalized_scores(dense)
    candidates = {r['chunk_id']: r for r in lexical + dense}
    fused = []
    for key, row in candidates.items():
        score = alpha * b.get(key, 0.0) + (1 - alpha) * d.get(key, 0.0)
        fused.append({**row, 'score': score, 'bm25_normalized': b.get(key, 0.0), 'dense_normalized': d.get(key, 0.0)})
    return sorted(fused, key=lambda r: (-r['score'], r['chunk_id']))

def hybrid_search(query, top_k=10, alpha=0.5, use_hyde=True):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between 0 and 1')
    vector = query_embedding(query, use_hyde)['vector']
    lexical = bm25_search(query, state.CANDIDATES)
    dense = dense_search(vector, state.CANDIDATES)
    return fuse_results(lexical, dense, alpha)[:top_k]

