# Cascade Labeling — XMTC Cascade Ranking Pipeline

A cascade ranking pipeline for Extreme Multi-Label Text Classification (XMTC), inspired by
the xCoRetriev approach (`../XMTC.pdf`). Reframes label assignment as retrieval: each label is
represented by its own text ("labels-as-documents"), and a text query is ranked against the
label space through a cascade of increasingly expensive stages:

```
BM25 (CPU, top 1000/500/100) -> bi-encoder (GPU, top 100/50) -> cross-encoder (GPU, final top-10)
```

Evaluated on **Eurlex-4k** and **Wiki10-31k** using propensity-scored XMTC metrics
(PSP@k, PS-nDCG@k) alongside standard Precision@k / nDCG@k, plus CPU/GPU/time/energy profiling
per stage — to study the accuracy-vs-compute tradeoff.

See `Project.md` for the original brief, and **[`docs/`](docs/00_overview.md) for full documentation**
(setup, datasets, stages/models, running & creating pipelines, results, RAG-labels).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
# see docs/01_setup.md for the full dependency install (torch, sentence-transformers, etc.)
```

(A conda alternative, `environment.yml`, is also provided.)

## Pipeline

```bash
.venv/bin/python scripts/00_download_data.py --dataset eurlex-4k
.venv/bin/python scripts/01_prepare_data.py --dataset eurlex-4k
.venv/bin/python scripts/03_run_experiment.py --config configs/experiments/bm25_only.yaml
.venv/bin/python scripts/04_aggregate_results.py
```

Results land in `results/runs/<exp_id>/` and are aggregated into `results/summary/all_runs.parquet`
for analysis in `notebooks/`. See `docs/04_running_pipelines.md` and `docs/05_results.md` for details.

## Important caveat

A true "cross-encoder only over the full label space" baseline is computationally infeasible
(N_test x L pairs). We report `crossencoder_only` restricted to a fixed cheap candidate pool
(BM25 top-100) and call this out explicitly — this infeasibility is the motivation for the cascade.
