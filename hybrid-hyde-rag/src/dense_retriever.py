import numpy as np
from .embeddings import query_embedding
from . import state

def dense_search(query_or_embedding, top_k=10):
    if isinstance(query_or_embedding, str):
        vector = query_embedding(query_or_embedding)['vector']
    else:
        vector = np.asarray(query_or_embedding, dtype='float32')
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
    scores, positions = state.INDEX.search(vector.reshape(1, -1), min(top_k, len(state.CHUNKS)))
    return [{**state.CHUNKS[int(i)], 'score': float(s)} for s, i in zip(scores[0], positions[0]) if i >= 0]

