import argparse
import json
from src import state, pipeline as rag
from src.evaluation import run_experiments, run_bm25_only, SIZES
ROOT = state.ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dataset", choices=["insurance", "python"], default="insurance")
    parser.add_argument("--bm25-only", action="store_true", help="Evaluate BM25 without loading neural models")
    parser.add_argument("--question", type=str)
    parser.add_argument("--chunk-size", type=int, choices=SIZES, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if args.bm25_only and not args.evaluate:
        parser.error("--bm25-only requires --evaluate")
    state.select_dataset(args.dataset)
    if args.evaluate:
        if args.bm25_only:
            run_bm25_only()
        else:
            run_experiments()
        from src.results import refresh_results
        refresh_results()
    else:
        question = args.question or input(f"Enter a {args.dataset} question: ")
        result = rag.demo(question, args.chunk_size, args.alpha)
        print("\nORIGINAL QUESTION:", result["query"])
        print("\nHyDE DOCUMENT:", result["hypothetical_document"])
        for label in ["bm25", "dense_hyde", "fused", "reranked"]:
            print("\n" + label.upper())
            for row in result[label][:10]:
                print(row["chunk_id"], round(row.get("reranker_score", row["score"]), 4), row["text"][:140])
        print("\nFINAL CONTEXT:", json.dumps(result["final_context"], indent=2))
        print("\nANSWER:", result["generation"]["answer"])
        print("\nSOURCES:", json.dumps(result["generation"]["sources"], indent=2))
        out = ROOT / "results" / args.dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / "demo.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
