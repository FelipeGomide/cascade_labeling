# RAG-Labels: LLM-Generated Label Descriptions

## Why

Bare label text can be lexically thin or ambiguous out of context — e.g. the
Eurlex-4k EuroVoc term `"leave"` could mean parental leave, annual leave, or
leave of absence. The xCoRetriev paper's "RAG-labels" mechanism addresses
exactly this: ground each label's meaning in the documents that actually
carry it, and have an LLM write a short description from that context. This
mainly helps the **semantic** stages (bi-encoder, cross-encoder), which rely
on meaning rather than exact term overlap; it helps BM25 less (and could even
dilute it, since BM25 is lexical/term-frequency based).

This implementation **simplifies the paper's recipe**: the paper runs a
separate LLM-as-optimizer loop to iteratively refine its generation prompt
against 256 GPT-4-labeled samples. We skip that loop — it exists to
compensate for a weak/uncalibrated LLM, and isn't necessary with a clear
fixed prompt and a capable instruct model — and go straight to the
retrieval-augmented generation step.

## How it works

For each label:
1. Look up all training docs carrying that label (`label_to_docs`, built
   once from `train_df["label_ids"]`).
2. Sample up to `n_examples_per_label` (default 5, matching the paper) of
   them, each truncated to `example_doc_truncate_words` words.
3. Feed the label text + those examples into a fixed prompt template asking
   the LLM to describe what the label means based on what the examples have
   in common (`PROMPT_TEMPLATE` in `src/cascade/data/rag_labels.py`).
4. Generate greedily (`do_sample=False`) with a local instruct LLM.
5. Cache `(label_id, label_text, rag_description, augmented_text)` to
   `data/processed/<dataset>/rag_labels.parquet`, where
   `augmented_text = "{label_text}. {rag_description}"` (falls back to bare
   `label_text` if generation came back empty, which happens occasionally).

Labels with zero training examples get an empty description (and just use
the bare label text).

## Model choice

Default: **`Qwen/Qwen2.5-1.5B-Instruct`**, run locally (no API key, no
per-call cost), configured in `configs/models/rag_label_generator.yaml`:

```yaml
model_name: Qwen/Qwen2.5-1.5B-Instruct
device: cuda
fp16: true
n_examples_per_label: 5
example_doc_truncate_words: 150
max_new_tokens: 80
batch_size: 16
```

This was an explicit tradeoff decision: a small local model is free and has
no rate limits across ~35k labels total (vs. an API-based frontier model:
faster/higher quality but costs real money per call; or a quantized 8B
model: closer to the paper's choice of LLM but tight on 8GB VRAM and
slower). Swap `model_name` for any other HF `AutoModelForCausalLM`
+ chat-template-compatible instruct model if you want to try a different
tradeoff point.

## Performance characteristics on this hardware (RTX 4060 Ti, 8GB)

Generation is **batched** (left-padded, one `model.generate()` call per
batch) — this matters a lot: the naive per-label loop measured ~1.2s/label
(≈79 min for Eurlex-4k, ≈10+ hours for Wiki10-31k); batching at
`batch_size=16` brought this to **~0.31s/label** (≈22 min for Eurlex-4k,
≈2.8 hours for Wiki10-31k).

**Batch size is memory-constrained, not just throughput-constrained:**
- `batch_size=64` reliably OOMs (`CUDA out of memory`) — prompts with 5
  examples × 150 words can reach ~1000 tokens, and 64 of those in flight at
  once exceeds 8GB.
- `batch_size=32` *also* OOMs on some slices (longer-than-average prompts).
- `batch_size=24` fits, but gives no meaningful throughput improvement over
  16 (~0.33s/label vs ~0.31s/label) — the model is small enough that
  batching saturates quickly.
- **`batch_size=16` is the safe, recommended setting** — verified stable
  across multiple slices with adequate headroom, and the time cost of
  going lower than 24 is negligible.

If you change `model_name` to something larger, re-verify the batch size
fits — don't assume 16 is safe for an 8B model.

## Running it

```bash
.venv/bin/python scripts/05_generate_rag_labels.py --dataset eurlex-4k
```

Shows a live `tqdm` progress bar with ETA:

```
RAG-labels:  42%|████▏     | 104/248 [07:31<10:23, 4.21s/batch, labels=1664/3956]
```

(`248` = number of batches = `ceil(n_labels / batch_size)`; the `labels=`
postfix shows actual label count completed.)

Writes `data/processed/<dataset>/rag_labels.parquet`. Re-running overwrites
it (generation is deterministic — greedy decoding — but example sampling per
label uses `random.Random(seed)`, default `seed=42`, so re-runs are
reproducible too).

## Using RAG-labels in an experiment

Set `use_rag_labels: true` in an experiment yaml:

```yaml
exp_id: bm25_bi_1000_100_rag
dataset: eurlex-4k
use_rag_labels: true
pipeline:
  - stage: bm25
    top_k: 1000
  - stage: bi_encoder
    top_k: 100
```

`src/cascade/data/loaders.py`'s `load_label_texts()` switches between raw
`label_text` and the RAG `augmented_text` based on this flag, and raises a
clear error if `rag_labels.parquet` doesn't exist yet for that dataset. This
swaps the label text used by **every stage in the pipeline** (BM25 included)
— if you want BM25 to keep using bare label text while only the semantic
stages get the augmented version, that would require a small extension
(currently out of scope) to pass different label texts per stage rather than
one shared list.
