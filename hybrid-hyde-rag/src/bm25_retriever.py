import re
import numpy as np
from rank_bm25 import BM25Okapi
from . import state

def tokenize(text):
    return re.findall('[a-z0-9_]+', text.lower())

def build_bm25_index(chunks):
    state.BM25 = BM25Okapi([tokenize(c['text']) for c in chunks])
    return state.BM25

def bm25_search(query, top_k=10):
    scores = state.BM25.get_scores(tokenize(query))
    order = np.argsort(-scores, kind='stable')[:top_k]
    return [{**state.CHUNKS[int(i)], 'score': float(scores[i])} for i in order]

