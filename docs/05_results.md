# Results: Storage, Aggregation & Analysis

## Where everything lives

```
results/
├── runs/<exp_id>/        # one folder per experiment run (see below)
└── summary/
    └── all_runs.parquet  # every run's metrics + cost, one tidy long-format table
```

Each call to `scripts/03_run_experiment.py --config configs/experiments/<exp_id>.yaml`
writes (or overwrites) `results/runs/<exp_id>/`. Re-running the same config
overwrites that run's folder in place — there's no versioning, so if you want
to keep an old result, copy the folder out first.

## Per-run files (`results/runs/<exp_id>/`)

### `config.yaml`

An exact snapshot of the experiment config that produced this run (copied
verbatim from `configs/experiments/<exp_id>.yaml` at run time) — this is what
makes a run fully reproducible without needing to know what config file
generated it.

```yaml
exp_id: bm25_only
dataset: eurlex-4k
batch_size: 256
eval_ks: [1, 5, 10]
test_sample_size: null
pipeline:
  - stage: bm25
    top_k: 10
```

### `metrics.json`

Final effectiveness numbers, flat dict keyed `"<metric>@<k>"`:

```json
{
  "precision@1": 0.217, "precision@5": 0.103, "precision@10": 0.071,
  "ndcg@1": 0.217,      "ndcg@5": 0.133,      "ndcg@10": 0.146,
  "psp@1": 0.594,       "psp@5": 0.281,       "psp@10": 0.192,
  "psndcg@1": 0.142,    "psndcg@5": 0.115,    "psndcg@10": 0.129,
  "first_stage_recall@10": 0.142
}
```

- `precision@k` / `ndcg@k` — standard ranking metrics.
- `psp@k` — propensity-scored precision. **Unnormalized** in this codebase
  (can exceed 1 for tail-heavy correct hits) — not on the same scale as the
  xCoRetriev paper's PS-Precision@k. Don't compare it directly to the paper.
- `psndcg@k` — propensity-scored nDCG, normalized the same way as the paper
  (`PSDCG/PSIDCG`) — this is the metric that's actually comparable across
  our results and the paper's.
- `first_stage_recall@N` — fraction of each doc's true labels present in the
  *first* pipeline stage's top-N output (N = that stage's `top_k`). Sanity
  check for whether a cascade's first cutoff is too aggressive.

### `resources.json`

Cost/hardware metrics, nested:

```json
{
  "hardware": {
    "platform": "Linux-6.8.0-...", "cpu_count": 12,
    "total_ram_gb": 31.3, "gpu_name": "NVIDIA GeForce RTX 4060 Ti", "gpu_vram_gb": 8.0
  },
  "total": {
    "stage": "bm25_only",
    "wall_time_s": 1.73, "cpu_time_s": 1.75,
    "peak_rss_mb": 505.8, "peak_vram_mb": 0.0,
    "mean_gpu_util_pct": 50.4, "energy_j": 26.3
  },
  "per_stage": {
    "bm25": { "wall_time_s": 1.71, "mean_candidates": 10.0 }
  }
}
```

- `hardware` — machine info, constant across runs on the same box (sanity
  context, not something you'd plot).
- `total` — whole-run cost. `wall_time_s` is the only field used to derive
  `throughput_docs_per_s` later (see below); everything else is reported
  as-measured.
  - **No `gpu_time_s` field exists.** There's no per-op CUDA profiling in
    this codebase, so a clean CPU/GPU time split isn't measurable.
    `mean_gpu_util_pct` is the closest proxy for how GPU-bound a run was.
  - `cpu_time_s` can exceed `wall_time_s` (multi-threaded CPU work — data
    loading, tokenization — overlapping with GPU compute), which is
    expected, not a bug.
- `per_stage` — one entry per pipeline stage, with that stage's own wall
  time and average candidate-set size. Useful for finding which stage in a
  multi-stage cascade actually dominates total cost.

### `per_stage.parquet`

Per-document, per-stage breakdown — finer-grained than `resources.json`'s
`per_stage` summary:

```
doc_id  stage  candidate_count  approx_latency_s
0       bm25   10               0.000442
1       bm25   10               0.000442
```

`approx_latency_s` is an **average-per-doc-in-batch** approximation (not
measured per individual document — preserves GPU batching efficiency
instead of timing each doc separately). Use this when you need a
latency/candidate-count distribution rather than just a per-run total.

### `predictions.parquet`

The actual output — one row per test document:

```
doc_id  ranked_label_ids        scores
0       [12, 884, 31, ...]      [28.1, 20.8, 15.6, ...]
```

`ranked_label_ids` is best-first. This is what every metric in
`metrics.json` was computed from, and what the tail/head plots
(`src/cascade/eval/tail_head.py`) read directly — useful any time you want
to dig into *which* labels a pipeline actually predicted, not just the
aggregate score (e.g. checking whether a metric jump is driven by a handful
of labels dominating the predictions).

## The aggregated table (`results/summary/all_runs.parquet`)

Build/refresh it any time after adding or re-running experiments:

```bash
.venv/bin/python scripts/04_aggregate_results.py
```

This scans every folder under `results/runs/`, and for each run, **explodes
`metrics.json` into one row per metric** while carrying the run's config and
cost columns along on every row (`src/cascade/eval/aggregate.py`):

| Column | Source | Notes |
|---|---|---|
| `exp_id`, `dataset` | `config.yaml` | |
| `pipeline` | `config.yaml` | e.g. `"bm25->bi_encoder->cross_encoder"` |
| `stage_cutoffs` | `config.yaml` | e.g. `"1000>100>10"`, aligned positionally with `pipeline` |
| `metric_name`, `k` | `metrics.json` key, parsed (`"psndcg@5"` → `"psndcg"`, `5`) | `k` is `None` for keys with no `@k` (none currently) |
| `value` | `metrics.json` | the metric's value |
| `total_time_s` | `resources.json["total"]["wall_time_s"]` | |
| `cpu_time_s`, `peak_vram_mb`, `mean_gpu_util_pct`, `energy_j` | `resources.json["total"]` | reported as-measured, no derived GPU-only split |
| `throughput_docs_per_s` | `len(predictions.parquet) / wall_time_s` | computed at aggregation time, not stored per-run |
| `timestamp` | `resources.json`'s file mtime | when the run finished |

A run is **silently skipped** if `config.yaml`, `metrics.json`, or
`resources.json` is missing — so a crashed/incomplete run just won't show
up (no error raised). If a run you expect is missing from the aggregate
table, check that all three files exist in its folder before suspecting a
bug in the aggregation step.

Because every row repeats the same cost columns, `cost = df[cost_cols].drop_duplicates(subset=['exp_id'])`
is the standard pattern for getting one row per run when you only care
about cost (see recipes below).

## How to gather info — recipes

All of these assume `.venv/bin/python` and `import pandas as pd`.

**List every run with its pipeline shape:**
```python
df = pd.read_parquet('results/summary/all_runs.parquet')
df[['exp_id', 'dataset', 'pipeline', 'stage_cutoffs']].drop_duplicates()
```

**Compare one metric across runs, for one dataset:**
```python
sub = df[(df.dataset == 'wiki10-31k') & (df.metric_name == 'psndcg') & (df.k == 5)]
sub[['exp_id', 'value']].sort_values('value', ascending=False)
```

**Cost-only view (one row per run):**
```python
cost_cols = ['exp_id', 'dataset', 'pipeline', 'stage_cutoffs',
             'total_time_s', 'cpu_time_s', 'peak_vram_mb',
             'mean_gpu_util_pct', 'energy_j', 'throughput_docs_per_s']
cost = df[cost_cols].drop_duplicates(subset=['exp_id'])
```

**Join cost + a quality metric (e.g. for a Pareto-style comparison):**
```python
q = df[(df.metric_name == 'psndcg') & (df.k == 5)][['exp_id', 'value']]
merged = cost.merge(q, on='exp_id')
merged['s_per_doc'] = 1 / merged['throughput_docs_per_s']
```

**Find the Pareto frontier (cheapest-for-its-quality runs) for a dataset:**
```python
def pareto(sub):
    sub = sub.sort_values('s_per_doc')
    best, keep = -1, []
    for _, r in sub.iterrows():
        if r['value'] > best:
            keep.append(r['exp_id']); best = r['value']
    return keep

pareto(merged[merged.dataset == 'eurlex-4k'])
```

**Per-stage latency breakdown for one run** (needs the run's own
`per_stage.parquet`, not the aggregate table):
```python
ps = pd.read_parquet('results/runs/bm25_bi_cross_1000_100_10/per_stage.parquet')
ps.groupby('stage')['approx_latency_s'].sum()  # total time spent per stage across all docs
```

**Inspect what a run actually predicted** (e.g. to check label diversity /
head-label bias, as we did when the cross-encoder's precision jump looked
suspicious):
```python
preds = pd.read_parquet('results/runs/<exp_id>/predictions.parquet')
top1 = preds['ranked_label_ids'].apply(lambda x: x[0])
top1.value_counts().head(10)   # which labels dominate the top-1 predictions
```

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

## Reading the cost/quality trade-off

X-axis is `1 / throughput_docs_per_s` (seconds per document), Y is a quality
metric. A pipeline is **Pareto-optimal** if no cheaper pipeline beats its
quality, and no higher-quality pipeline is cheaper. Two things worth keeping
in mind when reading this:

- **The metric you pick changes the verdict.** Raw `precision@k` and
  propensity-scored `psndcg@k` can disagree about whether a pipeline change
  (e.g. adding a fine-tuned cross-encoder) was worth it — `precision@k`
  rewards getting *any* correct label, `psndcg@k` specifically rewards
  correctly ranking rare/tail labels and discounts common ones. A pipeline
  that leans into predicting frequent, generic labels can score much higher
  on `precision@k` while scoring *lower* on `psndcg@k`. Always state which
  metric a cost/quality claim is based on.
- **Wider candidate pools aren't free, and aren't always better.** Cascades
  with a larger `top_k` between stages (e.g. cross-encoder fed top-100
  candidates instead of top-10/15) cost substantially more (often 5-10x)
  and have sometimes scored *worse* on `psndcg@k` in this repo's runs — so
  "more candidates" should never be assumed to dominate "fewer candidates"
  on either axis without checking both.

## Current results vs. the paper (context, not a guarantee)

Caveat: this isn't apples-to-apples — the paper uses 5-fold CV with
fine-tuned bi-/cross-encoders and RAG-labels, evaluated as a two-stage
retrieve-and-fuse pipeline (no cross-encoder at all). This repo uses a
single test split, a sequential BM25→bi-encoder→cross-encoder cascade (the
cross-encoder is our own addition, not in the paper), and fine-tuning has
since been added for both the bi-encoder and cross-encoder on both datasets.
The only metric directly comparable to the paper's reported numbers is
**PS-nDCG@k** (normalized the same way in both); our raw `psp` is
unnormalized and isn't on the same scale as the paper's PS-Precision@k.

- **Eurlex-4k:** our best PS-nDCG@1 (~0.45, fully fine-tuned cascade) is
  within ~8% of the paper's xCoRetriev (~0.49), though the gap widens at
  k=5/10 (~20-30% below) — plausibly because the paper's tail/head-stratified
  fusion keeps rebalancing deeper into the ranked list, while our cascade
  just narrows top-k once per stage with no tail-aware re-ranking.
- **Wiki10-31k:** our best PS-nDCG@1 (~0.35, fine-tuned bm25→bi-encoder)
  actually **exceeds** the paper's full xCoRetriev (~0.21) at k=1, and is
  competitive at k=5/10. Plausible explanation: Wikipedia category names
  tend to appear verbatim in article text, making lexical/semantic matching
  unusually easy — treat this as a promising single-split signal, not a
  confirmed result, since we don't have cross-validation or significance
  testing like the paper does.
