"""Run experiments, or show the complete pipeline for one question."""
import argparse
import csv
import json
import math
import pickle
import platform
import re
import time
from collections import Counter
from pathlib import Path
import numpy as np
import psutil
from . import pipeline as rag
from . import state

ROOT = state.ROOT
SIZES = [128, 220]
ALPHAS = [0.25, 0.50, 0.75]


def peak_memory_mb():
    # /proc/self avoids PID-namespace issues in container environments.
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024
    return psutil.Process().memory_info().rss / 1024**2


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label_chunks(questions, chunks, size):
    # A chunk is relevant when it fully contains an annotated evidence span.
    for q in questions:
        relevant = set()
        for evidence in q["evidence"]:
            matches = [c["chunk_id"] for c in chunks
                       if c["document_id"] == evidence["document_id"]
                       and c["metadata"]["char_start"] <= evidence["start"]
                       and c["metadata"]["char_end"] >= evidence["end"]]
            if not matches:
                raise ValueError(f"Evidence crosses every chunk boundary: {q['question_id']}")
            relevant.update(matches)
        q["relevant_chunk_ids"][str(size)] = sorted(relevant)


def retrieval_metrics(results, relevant):
    if not relevant:
        return {"recall5": None, "recall10": None, "mrr": None, "ndcg10": None}
    ids = [r["chunk_id"] for r in results]
    relevant = set(relevant)
    hits = [int(key in relevant) for key in ids[:10]]
    first = next((i + 1 for i, hit in enumerate(hits) if hit), None)
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return {"recall5": len(set(ids[:5]) & relevant) / len(relevant),
            "recall10": len(set(ids[:10]) & relevant) / len(relevant),
            "mrr": 1 / first if first else 0.0, "ndcg10": dcg / ideal}


def words(text):
    text = re.sub(r"\[[^\]]*\]", "", text.lower())
    return [w for w in re.findall(r"\w+", text) if w not in {"a", "an", "the"}]


def token_f1(answer, expected):
    actual, gold = words(answer), words(expected)
    if not actual or not gold:
        return float(actual == gold)
    overlap = sum((Counter(actual) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(actual), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def abstained(text):
    return bool(re.search(r"insufficient evidence|not (?:provided|mentioned|specified)|cannot (?:answer|determine)|don't know", text.lower()))


def mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def retrieve(q, method, alpha):
    extra = 0.0
    if method != "E1":
        hyde = method in {"E4", "E5", "E6"}
        info = rag.query_embedding(q, hyde)
        extra = info["seconds"]
    started = time.perf_counter()
    if method == "E1":
        results = rag.bm25_search(q, 10)
    elif method in {"E2", "E4"}:
        results = rag.dense_search(info["vector"], 10)
    else:
        results = rag.hybrid_search(q, 30 if method == "E6" else 10, alpha, method in {"E5", "E6"})
        if method == "E6":
            results = rag.rerank(q, results, 10)
    # Cached query preparation is charged back so HyDE latency is not hidden.
    return results, time.perf_counter() - started + extra


def evaluate(questions, size, method, alpha, split, generate=False, answer_log=None):
    rows = []
    for q in [q for q in questions if q["split"] == split]:
        results, seconds = retrieve(q["question"], method, alpha)
        metrics = retrieval_metrics(results, q["relevant_chunk_ids"][str(size)])
        f1 = refusal = citation_rate = None
        gen_time = 0.0
        if generate:
            started = time.perf_counter()
            output = rag.generate_answer(q["question"], results, rag.GENERATOR)
            gen_time = time.perf_counter() - started
            f1 = max(token_f1(output["model_text"], a) for a in [q["expected_answer"]] + q["acceptable_answers"])
            refusal = float(abstained(output["model_text"])) if not q["evidence"] else None
            citation_rate = float(bool(output["model_citations"]))
            if answer_log:
                answer_log.write(json.dumps({"size": size, "experiment": method, "alpha": alpha,
                    "question_id": q["question_id"], "question": q["question"],
                    "expected_answer": q["expected_answer"], "hyde": rag.HYDE_CACHE.get(q["question"], {}).get("hypothetical", ""),
                    "retrieved_ids": [r["chunk_id"] for r in results], **output}, ensure_ascii=False) + "\n")
                answer_log.flush()
        rows.append({"chunk_size": size, "experiment": method, "alpha": alpha, "split": split,
                     "question_id": q["question_id"], "query_type": q["query_type"], **metrics,
                     "answer_f1": f1, "unanswerable_refusal": refusal, "model_citation_present": citation_rate,
                     "retrieval_seconds": seconds, "generation_seconds": gen_time if generate else None})
    return rows


def run_experiments():
    import faiss
    out = ROOT / "results" / state.DATASET
    out.mkdir(parents=True, exist_ok=True)
    questions = json.loads(state.evaluation_path().read_text(encoding="utf-8"))
    summaries, all_rows, memory, selected = [], [], [], {}
    rag.load_models()
    # Load and warm each model once. These startup costs are excluded from query latency.
    rag.EMBEDDER.encode(state.DATASET)
    rag.GENERATOR("Say hello.", max_new_tokens=4)
    rag.RERANKER.predict([(state.DATASET, "Reference material about " + state.DATASET)])
    print("Preparing query and HyDE embeddings...", flush=True)
    for q in questions:
        rag.query_embedding(q["question"], False)
        rag.query_embedding(q["question"], True)
    with (out / "answers.jsonl").open("w", encoding="utf-8") as answer_log:
        for size in SIZES:
            start = time.perf_counter()
            chunks = rag.initialize(size)
            label_chunks(questions, chunks, size)
            state.evaluation_path().write_text(json.dumps(questions, indent=2), encoding="utf-8")
            setup_seconds = time.perf_counter() - start
            index_mb = (len(faiss.serialize_index(rag.INDEX)) + len(pickle.dumps(rag.BM25))) / 1024**2
            memory.append({"chunk_size": size, "chunks": len(chunks), "index_build_seconds": setup_seconds,
                           "serialized_indexes_mb": index_mb, "vectors_mb": rag.VECTORS.nbytes / 1024**2})
            print(f"Size {size}: {len(chunks)} chunks", flush=True)
            best = {}
            for method in ["E3", "E5"]:
                trials = []
                for alpha in ALPHAS:
                    rows = evaluate(questions, size, method, alpha, "dev")
                    all_rows.extend(rows)
                    trials.append((mean(rows, "mrr"), alpha))
                best[method] = max(trials, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]
            selected[str(size)] = best
            configs = [("E1", None), ("E2", None)]
            configs += [("E3", a) for a in ALPHAS] + [("E4", None)]
            configs += [("E5", a) for a in ALPHAS] + [("E6", best["E5"])]
            for method, alpha in configs:
                generate = method not in {"E3", "E5"} or alpha == best[method]
                print(f"Running size={size} {method} alpha={alpha} answers={generate}", flush=True)
                rows = evaluate(questions, size, method, alpha, "test", generate, answer_log)
                all_rows.extend(rows)
                summaries.append({"chunk_size": size, "chunks": len(chunks), "experiment": method,
                    "alpha": alpha, "selected_on_dev": generate, "test_questions": len(rows),
                    "retrieval_questions": sum(r["mrr"] is not None for r in rows),
                    "recall_at_5": mean(rows, "recall5"), "recall_at_10": mean(rows, "recall10"),
                    "mrr_at_10": mean(rows, "mrr"), "ndcg_at_10": mean(rows, "ndcg10"),
                    "answer_token_f1": mean(rows, "answer_f1"),
                    "unanswerable_refusal_rate": mean(rows, "unanswerable_refusal"),
                    "model_citation_rate": mean(rows, "model_citation_present"),
                    "mean_retrieval_seconds": mean(rows, "retrieval_seconds"),
                    "p95_retrieval_seconds": float(np.percentile([r["retrieval_seconds"] for r in rows],95)),
                    "mean_generation_seconds": mean(rows, "generation_seconds"),
                    "shared_run_peak_rss_mb": peak_memory_mb(),
                    "serialized_indexes_mb": index_mb})
                write_csv(out / "experiment_summary.csv", summaries)
                write_csv(out / "retrieval_results.csv", all_rows)
    paired_ids = {q["question_id"] for q in questions if q["split"] == "test"}
    # Full runs compare every held-out question at both chunk sizes.
    comparison = []
    for size in SIZES:
        paired = [r for r in all_rows if r["chunk_size"] == size and r["experiment"] == "E1"
                  and r["split"] == "test" and r["question_id"] in paired_ids]
        comparison.append({"chunk_size": size, "method": "BM25", "paired_questions": len(paired),
                           "question_ids": ";".join(sorted(paired_ids)),
                           "recall_at_5": mean(paired, "recall5"), "recall_at_10": mean(paired, "recall10"),
                           "mrr_at_10": mean(paired, "mrr"), "ndcg_at_10": mean(paired, "ndcg10"),
                           "answer_token_f1": mean(paired, "answer_f1")})
    write_csv(out / "chunk_size_comparison.csv", comparison)
    peak = peak_memory_mb()
    for summary in summaries:
        summary["shared_run_peak_rss_mb"] = peak
    write_csv(out / "experiment_summary.csv", summaries)
    (out / "run_info.json").write_text(json.dumps({"dataset": state.DATASET, "status": "complete", "selected_alpha": selected, "index_cost": memory,
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "threads": 4, "device": "cpu",
        "embedding_model": rag.EMBED_MODEL, "generator": rag.GEN_MODEL, "reranker": rag.RERANK_MODEL,
        "latency_note": "Warm models. Cached preparation charged back once per request. Excludes downloads, model loading, indexing.",
        "shared_run_peak_rss_mb": peak,
        "memory_note": "Shared process high-water RSS includes all models and caches; not isolated per-method memory. Linux uses VmHWM; other platforms sample RSS."}, indent=2), encoding="utf-8")
    print("Finished. Results are in the results folder.", flush=True)


def run_bm25_only():
    """Real E1 retrieval measurements without downloading neural models."""
    from tokenizers import Tokenizer
    from .ingestion import ingest_documents
    tokenizer = Tokenizer.from_file(str(ROOT / "data/raw/tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    questions = json.loads(state.evaluation_path().read_text(encoding="utf-8"))
    out = ROOT / "results" / state.DATASET / "bm25_only"
    out.mkdir(parents=True, exist_ok=True)
    summary, details = [], []
    for size in SIZES:
        state.CHUNKS = ingest_documents([state.corpus_path()], tokenizer, size)
        if len(state.CHUNKS) < 100:
            raise ValueError("At least 100 chunks are required")
        label_chunks(questions, state.CHUNKS, size)
        rag.build_bm25_index(state.CHUNKS)
        rows = evaluate(questions, size, "E1", None, "test", generate=False)
        details.extend(rows)
        summary.append({"dataset": state.DATASET, "chunk_size": size, "chunks": len(state.CHUNKS),
                        "experiment": "E1", "test_questions": len(rows),
                        "retrieval_questions": sum(r["mrr"] is not None for r in rows),
                        "recall_at_5": mean(rows, "recall5"), "recall_at_10": mean(rows, "recall10"),
                        "mrr_at_10": mean(rows, "mrr"), "ndcg_at_10": mean(rows, "ndcg10"),
                        "mean_retrieval_seconds": mean(rows, "retrieval_seconds"),
                        "answer_token_f1": None, "status": "retrieval_only_no_generation"})
    write_csv(out / "experiment_summary.csv", summary)
    write_csv(out / "retrieval_results.csv", details)
    state.evaluation_path().write_text(json.dumps(questions, indent=2), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({"dataset": state.DATASET,
        "status": "E1 retrieval complete at both chunk sizes; neural experiments and generation not run",
        "python": platform.python_version(), "platform": platform.platform()}, indent=2))
    print(json.dumps(summary, indent=2))
