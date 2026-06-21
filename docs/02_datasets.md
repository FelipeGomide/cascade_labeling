# Datasets

Two standard XMC benchmarks are used: **Eurlex-4k** and **Wiki10-31k**
(originally the brief specified AmazonCat-13k as the second dataset; this was
swapped for Wiki10-31k by agreement, since both are large-label-space
problems where multi-staging is useful, and Wiki10-31k is more tractable on
an 8 GB GPU).

| Dataset | Docs (train/test) | Labels | Avg labels/doc | Notes |
|---|---|---|---|---|
| Eurlex-4k | 15,449 / 3,865 | 3,956 | ~5.3 | EU legal documents; labels = EuroVoc descriptor terms. Short, jargon-heavy labels. |
| Wiki10-31k | 14,146 / 6,616 | 30,938 | ~18.6 | Wikipedia articles; labels = Wikipedia category tags. Long documents (~2000+ words). |

These numbers are asserted in code (`src/cascade/data/prepare.py`) against
the values in `configs/datasets/*.yaml` (`expected:` block) — if PECOS ever
changes the archive contents, preparation will fail loudly rather than
silently produce a different dataset.

## Source

Both datasets come from the **PECOS `xmc-base`** archives (Amazon), mirrored
on archive.org:

- `https://archive.org/download/pecos-dataset/xmc-base/eurlex-4k.tar.gz`
- `https://archive.org/download/pecos-dataset/xmc-base/wiki10-31k.tar.gz`

Each archive contains everything needed:
- `X.trn.txt`, `X.tst.txt` — one raw document per line
- `Y.trn.npz`, `Y.tst.npz` — sparse (doc × label) label matrices
- `output-items.txt` — **one label's surface text per line** (this is what
  makes the "labels-as-documents" framing possible — every stage scores the
  input text directly against these label texts)

If the archive.org mirror ever moves, the fallback is the
[Extreme Classification Repository](https://manikvarma.org/downloads/XC/XMLRepository.html)
— download the equivalent files manually into `data/raw/<dataset>/`.

## Pipeline: download → prepare

```bash
.venv/bin/python scripts/00_download_data.py --dataset eurlex-4k
.venv/bin/python scripts/01_prepare_data.py --dataset eurlex-4k
```

(same with `--dataset wiki10-31k`)

- **`00_download_data.py`** (`src/cascade/data/download.py`) downloads the
  `.tar.gz`, extracts it, and locates the real dataset root (PECOS nests files
  under `xmc-base/<dataset>/`, handled by `_find_dataset_root`'s recursive search).
- **`01_prepare_data.py`** (`src/cascade/data/prepare.py`) reads the raw files
  and writes, under `data/processed/<dataset>/`:
  - `train.parquet`, `test.parquet` — columns `doc_id, text, label_ids (list[int])`
  - `labels.parquet` — columns `label_id, label_text`
  - `propensity.npy` — inverse propensity per label (float32 array, see below)

  It also asserts row/label counts match `expected:` in the dataset config.

## Propensity model

XMTC evaluation needs a notion of how "rewarding" a label is — rare (tail)
labels should count for more than common (head) labels. We use the standard
Jain et al. (2016) propensity model (`src/cascade/eval/propensity.py`):

```
p_l = 1 / (1 + C * exp(-A * ln(N_l + B)))
C   = (ln(N) - 1) * (B + 1)^A
```

where `N` is the number of training docs and `N_l` the number of training
docs carrying label `l`. `1/p_l` (inverse propensity) is the reward weight
used in PSP@k / PS-nDCG@k. Both datasets use `A=0.55, B=1.5`
(`configs/datasets/*.yaml`, `propensity:` block) — the standard convention
for these benchmarks.

## A data quirk worth knowing: EuroVoc numeric codes (Eurlex-4k only)

A small minority of Eurlex-4k's `output-items.txt` entries are **bare legacy
EuroVoc numeric codes** (e.g. `"2164"`, `"4067.0"`) instead of the actual
descriptor term — about 20 out of 3,956 labels. Left as-is, these would be
meaningless tokens for BM25/encoders to match against.

`prepare.py` resolves these via the `eurovoc_descriptors.json` mapping
(sourced from the [`nlpaueb/multi-eurlex`](https://github.com/nlpaueb/multi-eurlex)
repo, cached at `data/external/eurovoc_descriptors.json`, auto-downloaded on
first use): code `"2164"` → `"EC institutional body"`, etc. 18/20 resolve
cleanly; 2 are retired EuroVoc concepts with no current English term and are
left as the bare code.

**This resolution is gated by `resolve_numeric_labels_via_eurovoc: true` in
`configs/datasets/eurlex-4k.yaml` and is NOT applied to Wiki10-31k.** This
matters: Wiki10-31k's label space *legitimately* contains numeric strings as
real Wikipedia category tags (e.g. a category literally named `"007"` or
`"1000"`). An earlier version of this code applied the EuroVoc resolution
unconditionally and silently corrupted ~285 genuine Wiki10-31k labels by
matching their numeric IDs against unrelated EuroVoc legal terms (e.g. label
`"10"` was overwritten with `"domestic trade"`). If you add a new dataset
with genuinely meaningless numeric label IDs, opt it into
`resolve_numeric_labels_via_eurovoc` deliberately — don't assume it's safe by default.

## Loading prepared data in code

```python
from cascade.data.loaders import load_processed

train_df, test_df, labels_df, inv_propensity = load_processed("eurlex-4k")
```
