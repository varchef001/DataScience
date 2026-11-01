from .embeddings import load_models, build_dense_index, query_embedding
from .bm25_retriever import build_bm25_index, bm25_search
from .dense_retriever import dense_search
from .hybrid_retriever import fuse_results, hybrid_search
from .reranker import rerank
from .generator import generate_answer
from .ingestion import ingest_documents
from . import state

def demo(query, size=128, alpha=0.5):
    if not query.strip():
        raise ValueError('Enter a non-empty question')
    initialize(size)
    info = query_embedding(query, True)
    lexical = bm25_search(query, state.CANDIDATES)
    dense = dense_search(info['vector'], state.CANDIDATES)
    fused = fuse_results(lexical, dense, alpha)
    ranked = rerank(query, fused[:state.CANDIDATES], 10)
    result = generate_answer(query, ranked, state.GENERATOR)
    return {'query': query, 'hypothetical_document': info['hypothetical'], 'bm25': lexical, 'dense_hyde': dense, 'fused': fused, 'reranked': ranked, 'final_context': result['context'], 'generation': result}


def initialize(chunk_size=128):
    state.CHUNK_SIZE = chunk_size
    load_models()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(state.ROOT / "data/raw/tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    state.CHUNKS = ingest_documents([state.corpus_path()], tokenizer, chunk_size)
    if len(state.CHUNKS) < 100:
        raise ValueError("At least 100 chunks are required")
    build_dense_index(state.CHUNKS, state.EMBEDDER)
    build_bm25_index(state.CHUNKS)
    return state.CHUNKS

def __getattr__(name):
    return getattr(state, name)
