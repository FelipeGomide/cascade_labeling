"""Turn raw PECOS xmc-base files into analysis-friendly processed artifacts.

Produces, under <processed_dir>:
  train.parquet     - doc_id (int), text (str), label_ids (list[int])
  test.parquet      - same schema
  labels.parquet    - label_id (int), label_text (str)
  propensity.npy    - inverse propensity per label_id (float32 array, len = n_labels)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.sparse import load_npz

from cascade.eval.propensity import compute_inverse_propensity
from cascade.utils.io import ensure_dir, load_yaml
from cascade.utils.logging import get_logger

logger = get_logger(__name__)

# A small minority of PECOS xmc-base label texts are bare legacy EuroVoc numeric
# codes (e.g. Eurlex-4k has ~20/3956) instead of the descriptor term. This mapping
# (from nlpaueb/multi-eurlex) resolves code -> English term so BM25/encoders always
# see real text, not opaque numbers.
EUROVOC_DESCRIPTORS_URL = (
    "https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/"
    "eurovoc_descriptors.json"
)
EUROVOC_DESCRIPTORS_PATH = Path("data/external/eurovoc_descriptors.json")

_NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def _resolve_numeric_label_texts(label_texts: list[str], enabled: bool) -> list[str]:
    """Replace bare numeric EuroVoc codes with their English descriptor term.

    Only meaningful for EuroVoc-labelled datasets (Eurlex-4k): some other XMC
    datasets (e.g. Wiki10-31k) legitimately use numeric strings as label text
    (e.g. Wikipedia category "007"), so blindly resolving any numeric label
    against the EuroVoc thesaurus would corrupt unrelated labels.
    """
    if not enabled:
        return label_texts
    numeric_idx = [i for i, t in enumerate(label_texts) if _NUMERIC_RE.match(t)]
    if not numeric_idx:
        return label_texts

    if not EUROVOC_DESCRIPTORS_PATH.exists():
        logger.info(
            f"{len(numeric_idx)} numeric label codes found; downloading EuroVoc "
            f"descriptor mapping -> {EUROVOC_DESCRIPTORS_PATH}"
        )
        ensure_dir(EUROVOC_DESCRIPTORS_PATH.parent)
        resp = requests.get(EUROVOC_DESCRIPTORS_URL, timeout=60)
        resp.raise_for_status()
        EUROVOC_DESCRIPTORS_PATH.write_bytes(resp.content)

    with open(EUROVOC_DESCRIPTORS_PATH) as f:
        descriptors = json.load(f)

    resolved = list(label_texts)
    n_unresolved = 0
    for i in numeric_idx:
        code = label_texts[i].split(".")[0]
        en_term = descriptors.get(code, {}).get("en")
        if en_term:
            resolved[i] = en_term
        else:
            n_unresolved += 1
            logger.warning(f"Could not resolve EuroVoc code {label_texts[i]!r} to text; keeping code")
    logger.info(f"Resolved {len(numeric_idx) - n_unresolved}/{len(numeric_idx)} numeric label codes")
    return resolved


def _labels_per_doc(Y) -> list[list[int]]:
    Y = Y.tocsr()
    return [Y.indices[Y.indptr[i] : Y.indptr[i + 1]].tolist() for i in range(Y.shape[0])]


def prepare_dataset(dataset_cfg: dict, raw_root: Path) -> Path:
    processed_dir = ensure_dir(dataset_cfg["processed_dir"])
    expected = dataset_cfg.get("expected", {})

    texts_trn = _read_lines(raw_root / "X.trn.txt")
    texts_tst = _read_lines(raw_root / "X.tst.txt")
    Y_trn = load_npz(raw_root / "Y.trn.npz")
    Y_tst = load_npz(raw_root / "Y.tst.npz")
    label_texts = _read_lines(raw_root / "output-items.txt")
    label_texts = _resolve_numeric_label_texts(
        label_texts, enabled=dataset_cfg.get("resolve_numeric_labels_via_eurovoc", False)
    )

    n_labels = len(label_texts)
    assert Y_trn.shape[1] == n_labels, f"Y.trn label dim {Y_trn.shape[1]} != {n_labels} labels"
    assert Y_tst.shape[1] == n_labels, f"Y.tst label dim {Y_tst.shape[1]} != {n_labels} labels"
    assert len(texts_trn) == Y_trn.shape[0], "X.trn line count != Y.trn row count"
    assert len(texts_tst) == Y_tst.shape[0], "X.tst line count != Y.tst row count"

    if "n_train" in expected:
        assert len(texts_trn) == expected["n_train"], (
            f"n_train mismatch: got {len(texts_trn)}, expected {expected['n_train']}"
        )
    if "n_test" in expected:
        assert len(texts_tst) == expected["n_test"], (
            f"n_test mismatch: got {len(texts_tst)}, expected {expected['n_test']}"
        )
    if "n_labels" in expected:
        assert n_labels == expected["n_labels"], (
            f"n_labels mismatch: got {n_labels}, expected {expected['n_labels']}"
        )

    train_df = pd.DataFrame(
        {
            "doc_id": np.arange(len(texts_trn)),
            "text": texts_trn,
            "label_ids": _labels_per_doc(Y_trn),
        }
    )
    test_df = pd.DataFrame(
        {
            "doc_id": np.arange(len(texts_tst)),
            "text": texts_tst,
            "label_ids": _labels_per_doc(Y_tst),
        }
    )
    labels_df = pd.DataFrame(
        {"label_id": np.arange(n_labels), "label_text": label_texts}
    )

    prop_cfg = dataset_cfg.get("propensity", {"A": 0.55, "B": 1.5})
    inv_propensity = compute_inverse_propensity(Y_trn, A=prop_cfg["A"], B=prop_cfg["B"]).astype(
        np.float32
    )

    train_df.to_parquet(processed_dir / "train.parquet")
    test_df.to_parquet(processed_dir / "test.parquet")
    labels_df.to_parquet(processed_dir / "labels.parquet")
    np.save(processed_dir / "propensity.npy", inv_propensity)

    logger.info(
        f"Prepared {dataset_cfg['name']}: "
        f"{len(train_df)} train / {len(test_df)} test / {n_labels} labels -> {processed_dir}"
    )
    return processed_dir


def main():
    import argparse

    from cascade.data.download import download_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    cfg = load_yaml(f"configs/datasets/{args.dataset}.yaml")
    raw_root = download_dataset(cfg)
    prepare_dataset(cfg, raw_root)


if __name__ == "__main__":
    main()
