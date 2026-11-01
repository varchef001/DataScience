from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DATASET = "insurance"
CHUNK_SIZE = 128
OVERLAP = 24
CANDIDATES = 30
CHUNKS = []
INDEX = BM25 = EMBEDDER = GENERATOR = RERANKER = VECTORS = None
QUERY_CACHE = {}
HYDE_CACHE = {}

def select_dataset(dataset):
    global DATASET
    if dataset not in {"insurance", "python"}:
        raise ValueError("Choose insurance or python")
    if DATASET != dataset:
        QUERY_CACHE.clear()
        HYDE_CACHE.clear()
    DATASET = dataset

def corpus_path():
    return ROOT / "data" / "raw" / f"{DATASET}_documents.json"

def evaluation_path():
    return ROOT / "data" / ("evaluation.json" if DATASET == "insurance" else "python_evaluation.json")
