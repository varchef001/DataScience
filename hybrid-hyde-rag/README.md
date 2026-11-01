# HyDE + Hybrid RAG

This project answers questions using BM25, dense search, HyDE, weighted fusion, an optional reranker, and a Hugging Face generator. The default dataset is insurance. The original Python documentation dataset is also included. Run one domain at a time, as required by the assignment.

## Folder structure

The project follows section 14 of the uploaded assignment. Each requested module contains its corresponding implementation.

| Location | Purpose |
|---|---|
| `data/raw/` | InsuranceQA and Python source documents, provenance and tokenizer snapshot |
| `data/processed/` | Token chunks for both datasets and both sizes |
| `data/evaluation.json` | Insurance evaluation questions and evidence labels |
| `data/python_evaluation.json` | Original Python evaluation questions |
| `src/ingestion.py`, `src/chunking.py` | Clean, deduplicate, split and retain metadata |
| `src/embeddings.py` | Model loading, embeddings and FAISS indexing |
| `src/bm25_retriever.py`, `src/dense_retriever.py` | Keyword and vector retrieval |
| `src/hyde.py`, `src/hybrid_retriever.py` | Hypothetical documents and normalized fusion |
| `src/reranker.py`, `src/generator.py` | Cross-encoder reranking and grounded generation |
| `src/evaluation.py` | Metrics, development-set tuning and experiment loop |
| `src/state.py`, `src/pipeline.py`, `src/results.py` | Shared state, pipeline coordination and combined result tables |
| `notebooks/hybrid_hyde_rag.ipynb` | Step-by-step notebook alternative |
| `results/retrieval_results.csv` | Measured per-question results, with dataset and source-file columns |
| `results/experiment_summary.csv` | Measured experiment summaries, with dataset and source-file columns |
| `results/python/` | Original Python benchmarks and original four-page report |
| `results/insurance/` | Insurance measurements and demo output |
| `app.py` | Command line demo and evaluation entry point |
| `prepare_data.py` | Reproduce chunking, evidence labels and the pinned insurance snapshot |
| `architecture.svg` | Pipeline diagram |
| `insurance_report.md` | Insurance dataset, experiment results and limitations |

## Run in VS Code on Windows

Extract this ZIP. Open the **hybrid-hyde-rag** folder itself using File > Open Folder. Open Terminal > New Terminal. Run commands from the folder containing `app.py`.

This project was checked using Python 3.12 on Linux. Use a separate Python 3.12 environment on Windows; the previous Python 3.14 installation has not been tested with this dependency set. Install Python 3.12 first if `py -3.12` is unavailable.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_data.py --dataset all
.\.venv\Scripts\python.exe app.py --dataset insurance --question "Is workers compensation the same as disability insurance?"
```

Activation is optional because these commands use the virtual environment's interpreter directly. In VS Code, choose **Python: Select Interpreter** and select `.venv`. The first neural run downloads three model files and needs an internet connection. Later runs can reuse the downloaded cache. CPU execution can be slow.

Use the original Python dataset:

```powershell
.\.venv\Scripts\python.exe app.py --dataset python --question "How do I remove duplicates from a list?"
```

Run the full insurance experiment sweep:

```powershell
.\.venv\Scripts\python.exe app.py --dataset insurance --evaluate
```

Run only the lightweight BM25 retrieval benchmark:

```powershell
.\.venv\Scripts\python.exe app.py --dataset insurance --evaluate --bm25-only
```

For a BM25-only environment, only `numpy rank-bm25 tokenizers psutil` are needed. Full neural runs need all requirements. The notebook is optional and can be opened with the VS Code Jupyter extension; install `ipykernel` in the environment if prompted.

## Dataset details

**InsuranceQA V2 subset:** 500 unique answer documents; 680 chunks at 128 tokens and 540 at 220 tokens. There are 36 test questions (6 per type) and 12 development questions (2 per type). Types are exact keywords, paraphrases, two-source/multi-hop questions, long questions, ambiguous questions and unavailable answers. The evaluation is a constructed subset, not the official full InsuranceQA benchmark. Original question/answer IDs and split provenance are preserved. Variants, two-source questions and private-information questions are explicitly marked as authored. Source answer documents contain answers, not evaluation questions.

Original test questions only feed the test subset; original validation questions feed development. Multi-hop questions combine two facts from different source answers and do not necessarily require a chain of inference. The corpus is fixed before tuning. All models are pretrained; there is no model fine-tuning. Only fusion alpha is selected on development MRR.

Gold relevance uses annotated character spans mapped to chunks at each size. Labels are not exhaustive. Ambiguous queries deliberately omit context; the gold interpretation is documented, and asking for clarification may be appropriate even though token F1 does not reward it. No-answer questions request private records absent from this public corpus.

The source is the research corpus at https://github.com/shuzi/insuranceQA. It contains historical answers, including dated law, prices, product descriptions and opinions. The demo is a research retrieval exercise, not verified current insurance advice. Upstream states research-purpose use; its README is included in `data/raw/INSURANCEQA_README.md`. Cite: Minwei Feng, Bing Xiang, Michael R. Glass, Lidan Wang, Bowen Zhou, *Applying Deep Learning to Answer Selection: A Study and An Open Task*, ASRU 2015.

`data/raw/insurance_provenance.json` records the pinned upstream commit and selection method. To reconstruct the selected document text from upstream and verify its SHA-256 hashes:

```powershell
.\.venv\Scripts\python.exe prepare_data.py --dataset insurance --restore-insurance
```

The question variants and evidence annotations are bundled author-created labels. They are not automatically generated by the preparation script. The tokenizer snapshot comes from the `all-MiniLM-L6-v2` model repository and is bundled for stable chunk boundaries.

**Python documentation:** eight official Python 3.13 pages; 616/330 chunks, 36 test and 12 development questions. Source URLs are retained and `PYTHON_DOCS_LICENSE.txt` accompanies the content. Original source pages: https://docs.python.org/3.13/ .

## Models and retrieval

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- HyDE and final generation: `google/flan-t5-base`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Lexical search: `rank_bm25.BM25Okapi`
- Dense index: FAISS `IndexFlatIP` over normalized vectors (cosine similarity)

Chunks overlap by 24 embedding-tokenizer tokens. Model loading uses the CPU and four torch threads. BM25 uses the original question; HyDE creates a possible answer without looking at the corpus, embeds that text, then searches actual documents. Hypothetical text is never final-answer evidence.

For each top-30 candidate list, min-max normalization maps scores to [0,1]. Empty lists contribute nothing; constant-score lists receive zeros. Fusion uses **alpha * normalized_BM25 + (1-alpha) * normalized_dense** with missing scores zero and chunk-ID deduplication. Alphas are 0.25, 0.50, 0.75. The cross-encoder reranks the top 30 fused candidates. The final prompt uses excerpts from the top three chunks within a 480-token input budget.

The demo prints the question, hypothetical document, lexical and dense hits, fused ranking, reranker scores, exact final context, answer and source mapping. It saves `results/<dataset>/demo.json`.

## Evaluation and saved results

E1 BM25; E2 dense; E3 hybrid; E4 HyDE+dense; E5 HyDE+hybrid; E6 HyDE+hybrid+reranking. Full runs evaluate both sizes and all three fusion weights, selecting alpha separately on development questions. Answer generation is evaluated for E1, E2, E4, E6 and the selected E3/E5 configurations.

Recall@5/10 measures the fraction of labeled relevant chunks retrieved. MRR@10 uses the first relevant rank. Binary nDCG@10 measures ranking quality. Questions without relevant chunks are excluded from retrieval averages and included in generation/refusal evaluation. Answer token F1 measures lexical overlap, not factual faithfulness. Model citation presence and no-answer refusal are reported separately. Timing excludes model loading and indexing; cached embedding/HyDE preparation time is charged back once per query. Memory is a shared-process peak, not per-method isolated memory.

**Included measurements:** the original Python run completed all six methods at 128 tokens; its 220-token BM25 comparison has only 16 paired questions. Insurance includes completed E1 retrieval tests on all 36 test questions at both sizes. Insurance E2–E6 and answer-quality benchmarks have not been run. Empty answer metric cells mean not measured. The original PDF describes the Python run only. The insurance report explicitly records the new scope. The notebook is provided without fabricated execution outputs.

Full runs write to `results/<dataset>/`; BM25-only runs write to `results/<dataset>/bm25_only/` so they do not overwrite full results. Root CSVs combine measured rows with provenance. Do not average different datasets or duplicate full/BM25-only E1 runs together. Running a full sweep overwrites that dataset's prior full-run files; copy them first if you need an archive.

## Limitations

The small generator can produce weak or unsupported answers. It is instructed to refuse insufficient evidence, but this is not guaranteed. Numeric citations are mapped to supplied chunk IDs and invented IDs are removed. If no valid citation is generated, the output lists **Context sources supplied** separately; this is traceability, not verified claim-level support. Historical InsuranceQA answers may be incorrect or outdated. Further work includes reviewing labels, checking support sentence by sentence, running all insurance experiments and assessing stronger models.

Validation: a real insurance demo completed through embeddings, HyDE, BM25, fusion, reranking and final generation. Its output is saved in `results/insurance/demo.json`. This single-question execution is not a neural benchmark. Source restoration hashes, evidence-to-chunk labels, CLI imports, fusion edge cases and dataset cache switching were checked.

Generation now retries once if the model emits only a citation or empty text. The raw first output and retry flag are saved for auditing. If it still produces no answer, the system reports that generation failed. Historical Python benchmarks predate this change and the generalized domain prompt; rerunning the revised code can yield different answers.
