# RAG Assignment: Baseline vs. Re-Ranking

Two retrieval pipelines over the **same chunks** of one PDF, compared on stress-test queries:

| | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Retrieve | bi-encoder (`all-MiniLM-L6-v2`) + FAISS cosine search, top 10 | same top 10 |
| Re-rank | none | cross-encoder (`ms-marco-MiniLM-L-6-v2`) on the 10 (query, chunk) pairs |
| Context for the LLM | top 3 by cosine | top 3 by cross-encoder score |
| LLM | Gemini (`gemini-3.7-flash`), identical prompt for both | |

**Corpus:** the *Digital Personal Data Protection Act, 2023* and the *Digital Personal Data Protection Rules, 2025*,
merged into one 39-page PDF ([data/dpdp_corpus.pdf](data/dpdp_corpus.pdf); sources in [data/SOURCES.md](data/SOURCES.md)),
split into 230 chunks of 120 words (30-word overlap).

## Results at a glance

Numbers produced by [rag_baseline_vs_reranking.ipynb](rag_baseline_vs_reranking.ipynb) (all of them are also saved in [results/](results/)).

**Stress-test pool: 112 queries scored.** Outcome of re-ranking, judged by where the chunk that holds the answer (the "gold" chunk) lands relative to the top 3 that the LLM sees:

| Type | ALREADY TOP-3 | FIXED | GOLD NOT IN TOP-10 | STILL OUTSIDE TOP-3 | WORSE | All |
|---|---|---|---|---|---|---|
| Distractor | 20 | 5 | 2 | 1 | 0 | 28 |
| Exception | 30 | 4 | 30 | 7 | 2 | 73 |
| Negation | 5 | 2 | 2 | 1 | 1 | 11 |
| All | 55 | 11 | 34 | 9 | 3 | 112 |

Gold-chunk hit rates over the same pool (adversarial queries, so not typical performance):

| Pipeline | hit@1 | hit@3 | hit@10 | MRR@10 |
|---|---|---|---|---|
| bi-encoder (Pipeline 1) | 0.393 | 0.518 | 0.696 | 0.473 |
| after cross-encoder (Pipeline 2) | 0.446 | 0.589 | 0.696 | 0.533 |

**Selected cases** (rank of the gold chunk before and after re-ranking, and what each answer did):

| Case | Type | Query | Gold rank before -> after | Answer 1 | Answer 2 |
|---|---|---|---|---|---|
| DIS-15 | Distractor | How much is the penalty for breaking a voluntary undertaking given to the Board? | 7 -> 1 | omits the key fact | contains the key fact |
| DIS-05 | Distractor | What is the monthly salary of the Chairperson of the Board? | 6 -> 2 | states a wrong fact | contains the key fact |
| NEG-10 | Negation | Are Board Members paid a sitting fee for attending meetings? | 5 -> 1 | omits the key fact | contains the key fact |
| NEG-06 | Negation | Can a Consent Manager read the personal data that passes through its platform? | 5 -> 3 | omits the key fact | contains the key fact |
| DIS-18 | Distractor | Who selects the Chairperson of the Board? | 9 -> 3 | omits the key fact | contains the key fact |
| EXC-63 | Exception | Can a court or tribunal process personal data without following the Act's duties? | 4 -> 2 | omits the key fact | contains the key fact |
| EXC-20 | Exception | Does the Act apply to an Indian outsourcing firm processing the data of foreign customers? | 1 -> 7 | contains the key fact | omits the key fact |

**Retrieval latency** (CPU, 180 timed runs over the six rescued queries; the Gemini call is excluded and reported separately in the notebook):

| Stage (CPU) | Median (ms) | p95 (ms) |
|---|---|---|
| query embedding (bi-encoder) | 12.5 | 14.3 |
| FAISS search (top 10) | 0.1 | 0.1 |
| PIPELINE 1 retrieval total | 12.6 | 14.4 |
| cross-encoder, 10 pairs | 361.5 | 384.0 |
| PIPELINE 2 retrieval total | 374.4 | 395.9 |

## Repository layout

| Path | Contents |
|---|---|
| [rag_baseline_vs_reranking.ipynb](rag_baseline_vs_reranking.ipynb) | the whole experiment, with outputs saved |
| [data/](data/) | the corpus PDF and its sources |
| [results/](results/) | tables, figures and cached Gemini answers written by the notebook |
| [requirements.txt](requirements.txt) | pinned Python dependencies |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU build is enough
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key-here" > .env                              # git-ignored; or type it when prompted
jupyter notebook rag_baseline_vs_reranking.ipynb                        # run all cells from the repo root
```

Everything except the Gemini calls runs offline on CPU; the two models are downloaded from Hugging Face on first use.
Gemini answers are cached in `results/llm_cache.json` (free-tier keys allow only about 20 requests per day and model), so
re-running the notebook does not call the API again. Delete that file to regenerate the answers. The API key is read from
`.env` (listed in `.gitignore`) and is never written into the notebook.
