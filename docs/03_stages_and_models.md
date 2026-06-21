# Stages & Models

## The "labels-as-documents" framing

Instead of treating XMTC as classification, each label is represented by its
own short text (the EuroVoc term, the Wikipedia category name, or — see
[`06_rag_labels.md`](06_rag_labels.md) — an LLM-expanded description of it).
The label space becomes a small retrieval corpus, and every stage's job is:
**given an input document, rank the labels whose text is most relevant to it.**

This is implemented through one common interface
(`src/cascade/stages/base.py`, class `Stage`):

```python
def rank_batch(self, texts, candidate_label_ids=None, top_k=None) -> list[(ids, scores)]:
    ...
```

- `candidate_label_ids=None` → score against the **full label space** (used
  for single-stage baselines and the first stage of a cascade).
- `candidate_label_ids=[...]` per doc → score only that **restricted
  candidate set** (used by every stage after the first one in a cascade).

This one interface is what lets `pipeline/cascade.py` chain any stages
together generically.

## The three stages

### 1. BM25 (`stages/bm25_stage.py`) — CPU, lexical

- Library: [`bm25s`](https://github.com/xhluca/bm25s) (`k1=1.5, b=0.75`, the
  paper's settings).
- Corpus = label texts (indexed once at init); query = the input document
  (word-truncated to `truncation.bm25_query_tokens` from the dataset config,
  default 512 words — relevant for Wiki10-31k's long articles).
- **Gotcha already fixed in this codebase:** if you pass `corpus=...` to
  `bm25s.BM25()`'s constructor, `.retrieve()` returns the corpus *items*
  (i.e. label text strings) instead of integer ids. We only pass `corpus`
  to `.index()`, not the constructor, so `.retrieve()` returns label ids
  (which align 1:1 with `label_id` since labels are 0-indexed and contiguous).

### 2. Bi-encoder (`stages/bi_encoder_stage.py`) — GPU, semantic

- Library: `sentence-transformers`. Default model:
  `sentence-transformers/all-MiniLM-L6-v2` (off-the-shelf / zero-shot —
  no fine-tuning yet, see [`Project context`](#fine-tuning-not-yet-done) below).
- All label embeddings are precomputed once at init (cheap: even 30,938
  labels × 384-dim float32 ≈ 47 MB) and indexed into a **FAISS flat index**
  (`IndexFlatIP`, since embeddings are L2-normalized so inner product =
  cosine similarity).
- **Full-label-space mode** (`candidate_label_ids=None`): FAISS search over
  all labels.
- **Cascade mode** (restricted candidates, e.g. following BM25): scores only
  the candidate subset directly via matrix-vector product — cheap since
  candidate sets are ≤ a few thousand.
- `model.max_seq_length` is set from `truncation.encoder_doc_tokens` in the
  dataset config (384 for Eurlex, 256 for Wiki10's longer docs).
- fp16 on CUDA by default (`fp16: true` in `configs/models/bi_encoder.yaml`).

### 3. Cross-encoder (`stages/cross_encoder_stage.py`) — GPU, joint re-ranking

- Library: `sentence_transformers.CrossEncoder`. Default model:
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Reads `(document, label_text)` pairs **jointly** through one transformer
  (not separate embeddings) — much more accurate, much more expensive.
- **Always restrict this to a small candidate set in practice** (≤100). A
  full-label-space pass would be `N_test × L` pairs — e.g. ~64M pairs for
  Wiki10-31k's full test×label combination, which is computationally
  infeasible. The code will still run with `candidate_label_ids=None` (it
  loops per-doc) but logs a warning; this is exactly the motivation for the
  cascade in the first place.
- fp16 on CUDA by default; `max_length=512` token budget per pair.

## Model configs

Each stage's hyperparameters live in `configs/models/<stage>.yaml`:

| File | Key params |
|---|---|
| `bm25.yaml` | `k1`, `b` |
| `bi_encoder.yaml` | `model_name`, `device`, `fp16`, `batch_size`, `normalize_embeddings` |
| `cross_encoder.yaml` | `model_name`, `device`, `fp16`, `batch_size`, `max_length` |
| `rag_label_generator.yaml` | see [`06_rag_labels.md`](06_rag_labels.md) |

Stage instances are built from these configs by a small registry in
`src/cascade/pipeline/config.py` (`STAGE_REGISTRY` / `register_stage`) — see
[`04_running_pipelines.md`](04_running_pipelines.md) for how to add a new stage type.

## Metrics

Implemented directly in numpy in `src/cascade/eval/metrics.py` (not via
`pyxclib`, which requires a Cython build that failed to install here):

- **Precision@k**, **nDCG@k** — standard ranking metrics.
- **PSP@k** (propensity-scored precision), **PS-nDCG@k** — same metrics but
  weighting each correct label by its inverse propensity (`1/p_l`), so
  correctly ranking a rare label counts more than a common one. This is
  the metric the xCoRetriev paper optimizes for.
- `recall_at_k()` — fraction of a doc's true labels present in its top-k
  predictions; used to validate first-stage cutoffs (see
  `first_stage_recall@N` in every experiment's `metrics.json`).
- `src/cascade/eval/tail_head.py` — splits labels into tail/head (Pareto
  80/20 by training frequency, as in the paper) and computes what fraction
  of top-k predictions are tail labels — used for the Fig. 6-style plot.

Sanity-checked: feeding gold labels as predictions gives P@k/nDCG@k ≈ 1.0;
shuffled/wrong predictions give 0. See the metrics module's docstring for
the exact formulas.

## Fine-tuning: not yet done

All models above are used **off-the-shelf / zero-shot** — this was an
explicit phase-1 decision (see project plan) to get the full pipeline,
metrics, and plots working before investing in fine-tuning. Results are
correspondingly weaker than the paper's fine-tuned baselines, especially on
Eurlex-4k's legal jargon (see `05_results.md` for the actual numbers and a
comparison against the paper).
