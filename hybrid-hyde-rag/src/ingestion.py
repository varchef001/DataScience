import json
import re
from pathlib import Path
from . import state
from .chunking import chunk_document

def ingest_documents(paths, tokenizer=None, chunk_size=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(state.EMBED_MODEL)
    chunks, seen = [], set()
    for path in paths:
        for doc in json.loads(Path(path).read_text(encoding="utf-8")):
            doc = {**doc, "text": re.sub(r"\s+", " ", doc["text"]).strip()}
            if not doc["text"] or doc["text"] in seen:
                continue
            seen.add(doc["text"])
            chunks.extend(chunk_document(doc, tokenizer, chunk_size or state.CHUNK_SIZE))
    return chunks
