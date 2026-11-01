"""Prepare either bundled corpus and remap evidence to token chunks."""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen
from tokenizers import Tokenizer
from src import state
from src.ingestion import ingest_documents


def restore_insurance():
    """Rebuild the exact selected answer texts from the pinned upstream release."""
    path = state.ROOT / "data/raw/insurance_documents.json"
    docs = json.loads(path.read_text(encoding="utf-8"))
    commit = docs[0]["metadata"]["upstream_commit"]
    url = f"https://raw.githubusercontent.com/shuzi/insuranceQA/{commit}/V2/"
    vocab_text = urlopen(url + "vocabulary", timeout=60).read().decode()
    vocab = dict(line.split("\t", 1) for line in vocab_text.splitlines())
    raw = gzip.decompress(urlopen(url + "InsuranceQA.label2answer.raw.encoded.gz", timeout=60).read()).decode()
    answers = dict(line.split("\t", 1) for line in raw.splitlines())
    for doc in docs:
        text = " ".join(vocab.get(word, word) for word in answers[doc["metadata"]["answer_id"]].split())
        text = re.sub(r"\s+", " ", text).strip()
        if hashlib.sha256(text.encode()).hexdigest() != doc["metadata"]["sha256"]:
            raise ValueError("Upstream answer hash does not match the snapshot")
        doc["text"] = text
    path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(dataset):
    state.select_dataset(dataset)
    tokenizer = Tokenizer.from_file(str(state.ROOT / "data/raw/tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    questions = json.loads(state.evaluation_path().read_text(encoding="utf-8"))
    out = state.ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    for size in [128, 220]:
        chunks = ingest_documents([state.corpus_path()], tokenizer, size)
        if len(chunks) < 100:
            raise ValueError("The assignment requires at least 100 chunks")
        docs = {d["document_id"]:d for d in json.loads(state.corpus_path().read_text(encoding="utf-8"))}
        for q in questions:
            relevant = set()
            for e in q["evidence"]:
                if docs[e["document_id"]]["text"][e["start"]:e["end"]] != e["quote"]:
                    raise ValueError("Evidence text has changed: " + q["question_id"])
                found = [c["chunk_id"] for c in chunks if c["document_id"] == e["document_id"]
                         and c["metadata"]["char_start"] <= e["start"] and c["metadata"]["char_end"] >= e["end"]]
                if not found:
                    raise ValueError("Evidence crosses all chunk boundaries: " + q["question_id"])
                relevant.update(found)
            q["relevant_chunk_ids"][str(size)] = sorted(relevant)
        (out / f"{dataset}_chunks_{size}.json").write_text(json.dumps(chunks,indent=2), encoding="utf-8")
        print(dataset, size, len(chunks), "chunks")
    state.evaluation_path().write_text(json.dumps(questions,indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["insurance", "python", "all"], default="all")
    parser.add_argument("--restore-insurance", action="store_true")
    args = parser.parse_args()
    if args.restore_insurance:
        restore_insurance()
    for dataset in (["insurance", "python"] if args.dataset == "all" else [args.dataset]):
        prepare(dataset)
