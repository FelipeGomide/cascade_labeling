# Running & Creating Pipelines

## Quick start: run an existing experiment

```bash
# one-time per dataset
.venv/bin/python scripts/00_download_data.py --dataset eurlex-4k
.venv/bin/python scripts/01_prepare_data.py --dataset eurlex-4k

# run a pipeline
.venv/bin/python scripts/03_run_experiment.py --config configs/experiments/bm25_only.yaml
```

This prints the final metrics and writes everything to
`results/runs/<exp_id>/` (see [`05_results.md`](05_results.md)).

## Experiments already defined

| Config | Dataset | Pipeline |
|---|---|---|
| `bm25_only.yaml` | eurlex-4k | BM25 (single stage) |
| `biencoder_only.yaml` | eurlex-4k | bi-encoder (single stage, full label space) |
| `crossencoder_only.yaml` | eurlex-4k | BM25 top-100 → cross-encoder (see caveat below) |
| `bm25_bi_1000_100.yaml` | eurlex-4k | BM25 top-1000 → bi-encoder top-100 |
| `bm25_bi_cross_1000_100_10.yaml` | eurlex-4k | BM25 top-1000 → bi-encoder top-100 → cross-encoder top-10 |
| `wiki10_*.yaml` | wiki10-31k | same five pipelines on Wiki10-31k |

**Caveat on `crossencoder_only`:** a true cross-encoder-only baseline over
the *full* label space is computationally infeasible (N_test × L pairs —
e.g. ~64M for Wiki10-31k). This config restricts scoring to a fixed BM25
top-100 candidate pool as a practical stand-in "single-stage" baseline. It's
not a pure baseline; treat it as "cross-encoder, given a cheap candidate
pool" when interpreting results.

## Anatomy of an experiment config

```yaml
exp_id: bm25_bi_1000_100        # also the results/runs/<exp_id>/ folder name
dataset: eurlex-4k               # must match a configs/datasets/<name>.yaml
batch_size: 256                  # docs per outer batch fed to each stage
eval_ks: [1, 5, 10]               # k values for P@k / nDCG@k / PSP@k / PS-nDCG@k
test_sample_size: null           # null = full test set; int = random subsample (for quick dev runs)
use_rag_labels: false            # see 06_rag_labels.md — use LLM-augmented label text instead of raw label text
pipeline:
  - stage: bm25
    top_k: 1000                  # candidates this stage outputs, passed to the next stage
  - stage: bi_encoder
    top_k: 100                   # final cutoff, since this is the last stage
```

Each `pipeline` entry's `top_k` is the number of candidates that stage hands
to the *next* stage (or, for the last stage, the final number of ranked
predictions evaluated).

## Creating a new experiment

1. Copy an existing yaml in `configs/experiments/` and change `exp_id`,
   `dataset`, and the `pipeline` list (stage names + `top_k` cutoffs).
2. Run it: `.venv/bin/python scripts/03_run_experiment.py --config configs/experiments/<your_new>.yaml`
3. Re-run `scripts/04_aggregate_results.py` to fold it into
   `results/summary/all_runs.parquet` for plotting.

For quick iteration before committing to a full run, set
`test_sample_size: 100` (or similar) — useful for sanity-checking a new
pipeline shape before paying the full compute cost (cross-encoder runs in
particular: ~0.1s/doc × candidate-set-size, so a full Wiki10-31k test set
of 6,616 docs can take several minutes to ~8 minutes depending on cutoffs).

## Adding a new stage type

1. Implement a new `Stage` subclass in `src/cascade/stages/` implementing
   `rank_batch(self, texts, candidate_label_ids, top_k)` — see
   `stages/base.py` and the three existing stages for the contract.
2. Register a builder function in `src/cascade/pipeline/config.py`:

   ```python
   @register_stage("my_new_stage")
   def _build_my_new_stage(model_cfg: dict, dataset_cfg: dict, label_texts: list[str]):
       return MyNewStage(label_texts, **model_cfg.get("params", {}))
   ```
3. Add `configs/models/my_new_stage.yaml` with its hyperparameters.
4. Reference `stage: my_new_stage` in any experiment yaml's `pipeline` list.

## Adding a new dataset

1. Add `configs/datasets/<name>.yaml` (see `eurlex-4k.yaml` for the
   template: `pecos_url`, `raw_dir`, `processed_dir`, `expected:` stats,
   `propensity:` constants `A`/`B`, `truncation:` token limits). If your
   dataset's `output-items.txt` ever contains placeholder/meaningless
   numeric label text that genuinely needs resolving against an external
   vocabulary (like Eurlex's EuroVoc codes), add that resolution logic
   behind an explicit opt-in flag — see the warning in
   [`02_datasets.md`](02_datasets.md) about why this must not be applied
   unconditionally.
2. `.venv/bin/python scripts/00_download_data.py --dataset <name>`
3. `.venv/bin/python scripts/01_prepare_data.py --dataset <name>`
4. Write experiment configs pointing `dataset: <name>`.

## What happens during a run (for debugging)

`src/cascade/pipeline/runner.py` (`run_experiment`):
1. Loads the experiment + dataset config, the processed train/test/labels
   data, and resolves label texts (raw or RAG-augmented).
2. Builds the `CascadePipeline` from the `pipeline:` list
   (`pipeline/config.py`).
3. Wraps the whole run in a `ResourceMonitor` (`profiling/monitor.py`) that
   samples GPU power/util in a background thread and reports wall time, CPU
   time, peak RAM/VRAM, mean GPU util, and energy (J).
4. Computes metrics (`eval/metrics.py`) plus `first_stage_recall@N` — the
   ceiling that the first stage's cutoff imposes on everything downstream.
5. Writes `config.yaml`, `predictions.parquet`, `metrics.json`,
   `resources.json`, `per_stage.parquet` to `results/runs/<exp_id>/`.
