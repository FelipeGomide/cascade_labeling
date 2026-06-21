# Results: Storage, Aggregation & Analysis

## Per-run output (`results/runs/<exp_id>/`)

Every call to `scripts/03_run_experiment.py` writes a self-contained folder:

| File | Contents |
|---|---|
| `config.yaml` | Exact snapshot of the experiment config used — enables exact reproduction. |
| `predictions.parquet` | `doc_id, ranked_label_ids (list[int]), scores (list[float])` — the final ranking per test doc. |
| `metrics.json` | `precision@{1,5,10}`, `ndcg@{1,5,10}`, `psp@{1,5,10}`, `psndcg@{1,5,10}`, plus `first_stage_recall@N`. |
| `resources.json` | `hardware` (GPU/CPU/RAM info), `total` (wall time, CPU time, peak RSS/VRAM, mean GPU util, energy J), `per_stage` (wall time + mean candidate count per stage). |
| `per_stage.parquet` | `doc_id, stage, candidate_count, approx_latency_s` — per-doc, per-stage breakdown (latency is an average-per-doc-in-batch approximation, not measured per individual doc, to preserve GPU batching efficiency). |

## Aggregated table (`results/summary/all_runs.parquet`)

```bash
.venv/bin/python scripts/04_aggregate_results.py
```

Scans every folder in `results/runs/`, and builds **one tidy long-format
row per (run × metric)** — this is what makes plotting trivial (no pivoting
needed for most plots):

```
exp_id | dataset | pipeline | stage_cutoffs | metric_name | k | value |
total_time_s | cpu_time_s | peak_vram_mb | mean_gpu_util_pct | energy_j |
throughput_docs_per_s | timestamp
```

Notes on a couple of columns:
- `pipeline` / `stage_cutoffs` are derived from `config.yaml`'s `pipeline:`
  list, e.g. `pipeline="bm25->bi_encoder"`, `stage_cutoffs="1000>100"`.
- We report `cpu_time_s`, `peak_vram_mb`, `mean_gpu_util_pct`, `energy_j` as
  **directly measured** values rather than inventing a "gpu_time_s" split —
  we have no per-op CUDA profiling, so a precise CPU/GPU time split isn't
  measurable; the measured fields above are what's actually accurate.
- `psp` / `psndcg` etc. come from `metric_name` + `k` columns rather than a
  wide `psp@1`/`psp@5`/... column set, so filtering is just
  `df[(df.metric_name == "psndcg") & (df.k == 5)]`.

Re-run this script any time after adding new experiment runs.

## Notebooks → figures

All three notebooks read from `results/summary/all_runs.parquet` (and
`results/runs/*/predictions.parquet` for the tail/head plot) and save PNGs to
`reports/figures/`. Run with the registered `cascade-venv` Jupyter kernel
(see [`01_setup.md`](01_setup.md)).

| Notebook | Produces |
|---|---|
| `01_dataset_eda.ipynb` | `01_label_frequency_and_reward.png` (long-tail label distribution + reward curve), `02_doc_label_cardinality.png` (labels/doc and doc-length histograms). |
| `02_results_analysis.ipynb` | `03_metrics_by_pipeline.png` (grouped bar chart of every metric × k, per pipeline). |
| `03_accuracy_vs_compute.ipynb` | `04_pareto_tradeoff.png` (**the core deliverable** — compute cost (s/doc) vs PS-nDCG@5, with the Pareto frontier drawn), `05_stage_latency_breakdown.png` (stacked per-stage wall time), `06_first_stage_recall.png` (BM25 recall@N curve — justifies cascade cutoffs), `07_tail_head_proportion.png` (paper Fig. 6-style: proportion of tail labels in top-k predictions per pipeline). |

All three notebooks `os.chdir()` to the repo root in their first cell if
launched from `notebooks/`, so all paths inside them (`configs/...`,
`results/...`, `reports/...`) are repo-root-relative, not notebook-relative.

To regenerate all figures after new runs:

```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cascade-venv \
  01_dataset_eda.ipynb 02_results_analysis.ipynb 03_accuracy_vs_compute.ipynb
```

## How to read the Pareto plot

X-axis is `1 / throughput_docs_per_s` (seconds per document, log scale), Y is
PS-nDCG@5. A pipeline is **Pareto-optimal** if no other pipeline beats it on
*both* axes simultaneously. As of the runs in this repo (off-the-shelf
models, no fine-tuning), `bm25_only` dominates on Eurlex-4k (cheapest *and*
best) — the semantic stages don't yet pay for themselves because they're
zero-shot on legal jargon they've never seen. On Wiki10-31k, the
`bm25→bi_encoder` cascade is the strongest point, beating both
single-stage baselines. See the project's fine-tuning discussion (not yet
implemented) for the expected next step in closing this gap, especially for
Eurlex-4k.

## Current results vs. the paper (context, not a guarantee)

Caveat: this isn't apples-to-apples — the paper uses 5-fold CV with
fine-tuned bi-/cross-encoders and RAG-labels; this repo (so far) uses a
single test split with off-the-shelf, zero-shot models. The fair comparison
metric is **PS-nDCG@k** (normalized in both setups); our raw `psp` is
*unnormalized* (can exceed 1 for tail-heavy correct hits) and isn't on the
same scale as the paper's PS-Precision@k.

- **Eurlex-4k:** our best PS-nDCG@1 (~0.084-0.14) is roughly 3-5x below even
  the weakest paper baseline (~0.40-0.49) — expected, since legal/EuroVoc
  jargon has no domain adaptation in off-the-shelf MiniLM models.
- **Wiki10-31k:** our `bm25→bi_encoder` cascade's PS-nDCG@1 (~0.33) actually
  **exceeds** the paper's full xCoRetriev (~0.213) at k=1, and is
  competitive at k=5/10. Plausible explanation: Wikipedia category names
  tend to appear verbatim in article text, making lexical/semantic matching
  unusually easy — but treat this as a promising single-split signal, not a
  confirmed result, until cross-validation is added.
