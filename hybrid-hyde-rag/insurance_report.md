# Insurance extension: data and evaluation report

## Scope
The assignment structure is now implemented with separate source modules and a walkthrough notebook. Insurance is the default domain, while the original Python documentation remains selectable. The three pretrained models are unchanged. The models were not trained on this subset.

## Data
InsuranceQA V2 provides the source answers. We include 500 unique answer documents, sampled reproducibly after retaining the evaluation's gold answers. The source commit and hashes are stored with the data. The answer text is historical and must not be treated as verified current insurance advice. The upstream research-use notice and citation are bundled.

The dataset yields 680 chunks at 128 tokens and 540 at 220 tokens, both with 24-token overlap. Metadata includes document ID, title, source URL, answer ID and offsets. Each of 36 test questions and 12 development questions has evidence annotations, expected answers and chunk labels at both sizes. There are six query types with six test and two development examples each. Original test/validation provenance is retained; adaptations and new questions are marked. The two-source questions combine separate facts rather than necessarily demanding chained reasoning. Ambiguous queries have an explicit assumed interpretation. The evaluation is a small constructed subset, not the published InsuranceQA benchmark.

## Implemented experiments
E1 is BM25, E2 dense retrieval, E3 weighted hybrid, E4 HyDE plus dense, E5 HyDE plus hybrid, and E6 adds a cross-encoder reranker. Both sizes and three alpha values are supported. Alpha weights BM25; 1-alpha weights dense scores. Min-max normalization is done within each candidate list, absent candidates get zero, and chunk IDs remove duplicates. Development MRR selects alpha before test answer scoring.

## Measurements actually run
| BM25 chunk size | Chunks | Test questions | Answerable questions | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 680 | 36 | 30 | 0.450 | 0.567 | 0.369 | 0.413 |
| 220 | 540 | 36 | 30 | 0.483 | 0.567 | 0.375 | 0.418 |

The larger chunks improved BM25 Recall@5 slightly on this constructed set, while Recall@10 stayed the same. These measurements do not establish the best size for dense, HyDE, reranking or generation. No-answer questions are excluded from retrieval averages. Source spans and all chunk IDs were checked at both sizes. Complete per-question records are saved under results/insurance/bm25_only/.

## Required evaluation questions: current evidence and remaining work
1. **When does BM25 outperform dense?** Not yet measured on insurance; a paired E1/E2 comparison is required.
2. **When does dense recover BM25 misses?** Not yet measured on insurance.
3. **Does HyDE help paraphrases?** Implemented but not yet measured on insurance.
4. **Does HyDE hallucinate terminology?** It can generate unsupported text; an insurance per-question error review remains necessary.
5. **Which alpha is best?** Insurance development-set sweeps have not run. No best alpha is claimed.
6. **Does reranking improve retrieval or answers?** Requires the pending E5/E6 comparison.
7. **How does chunk size matter?** E1 results above show a small Recall@5 improvement; generation effects are not measured.
8. **What about absent answers?** Six test questions ask about private records absent from the corpus. Retrieval still returns documents. The generator is prompted to refuse, but insurance refusal performance is not yet measured.
9. **What are latency and memory costs?** BM25 query timings are recorded in CSVs. Neural latency and memory for insurance are not measured. The earlier Python measurements remain separately identified.
10. **What failure modes remain?** Historical source inaccuracies, incomplete relevance labels, ambiguity, unsupported generation, clipped context, and source IDs that do not prove claim support.

## Reproduction
Follow README.md. Run `python app.py --dataset insurance --evaluate --bm25-only` for the included retrieval experiment. Run `python app.py --dataset insurance --evaluate` for the full pending sweep. Use `--dataset python` for the original domain. The original four-page PDF and Python metrics are archived under results/python/; they are not insurance results.

Reference: Feng et al., *Applying Deep Learning to Answer Selection: A Study and An Open Task*, ASRU 2015. Dataset: https://github.com/shuzi/insuranceQA .

Validation: a real insurance demo completed through embeddings, HyDE, BM25, fusion, reranking and final generation. Its output is saved in `results/insurance/demo.json`. This single-question execution is not a neural benchmark. Source restoration hashes, evidence-to-chunk labels, CLI imports, fusion edge cases and dataset cache switching were checked.

Generation now retries once if the model emits only a citation or empty text. The raw first output and retry flag are saved for auditing. If it still produces no answer, the system reports that generation failed. Historical Python benchmarks predate this change and the generalized domain prompt; rerunning the revised code can yield different answers.

Demo observation: for "Is workers compensation the same as disability insurance?", HyDE incorrectly suggested they were equivalent. Retrieval and reranking supplied the relevant source answer; the generator initially emitted only a citation. After the implemented retry with unlabeled context, the answer was "No". This illustrates both HyDE risk and the generator's limited response quality. Supplied source IDs are listed separately, not claimed as model-generated citations.
