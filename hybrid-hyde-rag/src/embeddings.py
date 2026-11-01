import time
import numpy as np
from .hyde import generate_hypothetical_document
from . import state

def load_models():
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import pipeline
    torch.set_num_threads(4)
    if state.EMBEDDER is None:
        state.EMBEDDER = SentenceTransformer(state.EMBED_MODEL, device='cpu')
        state.GENERATOR = pipeline('text2text-generation', model=state.GEN_MODEL, device=-1)
        state.RERANKER = CrossEncoder(state.RERANK_MODEL, device='cpu', max_length=384)

def build_dense_index(chunks, embedding_model):
    import faiss
    state.CHUNKS = chunks
    state.VECTORS = embedding_model.encode([c['text'] for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype('float32')
    state.INDEX = faiss.IndexFlatIP(state.VECTORS.shape[1])
    state.INDEX.add(state.VECTORS)
    return state.INDEX

def query_embedding(query, use_hyde=False):
    cache = state.HYDE_CACHE if use_hyde else state.QUERY_CACHE
    if query not in cache:
        start = time.perf_counter()
        hypothetical = generate_hypothetical_document(query, state.GENERATOR) if use_hyde else ''
        vector = state.EMBEDDER.encode(hypothetical if use_hyde else query, normalize_embeddings=True).astype('float32')
        cache[query] = {'vector': vector, 'hypothetical': hypothetical, 'seconds': time.perf_counter() - start}
    return cache[query]

