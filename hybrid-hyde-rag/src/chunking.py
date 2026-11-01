"""Overlapping token windows with character offsets for evidence labels."""
from . import state

def chunk_document(doc, tokenizer, chunk_size=128, overlap=24):
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    if hasattr(tokenizer, "encode") and not callable(tokenizer):
        offsets = tokenizer.encode(doc["text"], add_special_tokens=False).offsets
    else:
        offsets = tokenizer(doc["text"], add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False, verbose=False)["offset_mapping"]
    result = []
    for start in range(0, len(offsets), chunk_size-overlap):
        end = min(start+chunk_size, len(offsets))
        left, right = offsets[start][0], offsets[end-1][1]
        result.append({"chunk_id": f"{doc['document_id']}_{chunk_size}_{start}",
                       "document_id": doc["document_id"], "title": doc["title"],
                       "text": doc["text"][left:right], "source": doc["source"],
                       "metadata": {**doc["metadata"], "char_start": left, "char_end": right,
                                    "token_start": start, "token_end": end}})
        if end == len(offsets):
            break
    return result
