# Cascade Labeling — Documentation

This project implements and evaluates a cascade ranking pipeline for Extreme
Multi-Label Text Classification (XMTC), inspired by the xCoRetriev paper
(`../XMTC.pdf`). It reframes label assignment as retrieval: each label is
represented by its own text ("labels-as-documents"), and an input document is
ranked against the label space through a cascade of increasingly expensive
stages:

```
BM25 (CPU, lexical)  -->  bi-encoder (GPU, semantic)  -->  cross-encoder (GPU, joint re-ranking)
```

Each stage narrows the candidate label set for the next one. The goal is to
study the **accuracy vs. compute tradeoff**: how much ranking quality each
stage (alone or chained) buys, and at what CPU/GPU/time/energy cost.

See `../Project.md` for the original brief and
`/home/fgomide/.claude/plans/read-the-md-file-woolly-starlight.md` for the
full development plan.

## Documentation index

1. [`01_setup.md`](01_setup.md) — environment setup, hardware assumptions.
2. [`02_datasets.md`](02_datasets.md) — which datasets, how they're fetched and prepared, known data quirks.
3. [`03_stages_and_models.md`](03_stages_and_models.md) — the three classifier stages, the labels-as-documents framing, model configs.
4. [`04_running_pipelines.md`](04_running_pipelines.md) — how to run an experiment, how to define a new one.
5. [`05_results.md`](05_results.md) — where results are stored, the aggregated table, the analysis notebooks/figures.
6. [`06_rag_labels.md`](06_rag_labels.md) — the optional RAG-labels enhancement (LLM-generated label descriptions).

## Project layout

```
cascade_labeling/
├── configs/
│   ├── datasets/      # per-dataset metadata, truncation lengths, propensity constants
│   ├── models/        # per-stage model hyperparameters
│   └── experiments/   # one yaml per pipeline configuration to run
├── data/
│   ├── raw/           # downloaded PECOS xmc-base archives, untouched
│   ├── external/      # third-party mapping files (e.g. EuroVoc descriptors)
│   └── processed/     # train/test/labels parquet + propensity.npy per dataset
├── src/cascade/
│   ├── data/          # download, prepare, load datasets; RAG-label generation
│   ├── stages/        # BM25 / bi-encoder / cross-encoder stage implementations
│   ├── pipeline/       # chains stages into a cascade; builds pipelines from config; runs experiments
│   ├── eval/           # metrics (P@k, nDCG@k, PSP@k, PS-nDCG@k), aggregation, tail/head split
│   ├── profiling/      # CPU/GPU/RAM/energy resource monitor
│   └── utils/          # io/logging/seeding helpers
├── scripts/            # thin CLI entry points (00_download_data.py ... 05_generate_rag_labels.py)
├── results/
│   ├── runs/<exp_id>/  # one folder per experiment run
│   └── summary/        # all_runs.parquet, the tidy table all plots are built from
├── notebooks/          # EDA, results analysis, accuracy-vs-compute plots
├── reports/figures/    # PNGs exported by the notebooks
└── docs/               # you are here
```
