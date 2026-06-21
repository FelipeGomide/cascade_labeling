from pathlib import Path

import numpy as np
import pandas as pd

from cascade.utils.io import load_yaml


def load_dataset_config(dataset: str) -> dict:
    return load_yaml(f"configs/datasets/{dataset}.yaml")


def load_processed(dataset: str):
    """Return (train_df, test_df, labels_df, inv_propensity) for a prepared dataset."""
    cfg = load_dataset_config(dataset)
    processed_dir = Path(cfg["processed_dir"])

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    labels_df = pd.read_parquet(processed_dir / "labels.parquet")
    inv_propensity = np.load(processed_dir / "propensity.npy")

    return train_df, test_df, labels_df, inv_propensity


def load_label_texts(dataset: str, labels_df: pd.DataFrame, use_rag_labels: bool = False) -> list[str]:
    """Return per-label text used by the stages: either the raw label text, or
    (if use_rag_labels and rag_labels.parquet exists) the RAG-augmented text
    (label_text + ". " + LLM-generated description grounded in example docs).
    """
    if not use_rag_labels:
        return labels_df["label_text"].tolist()

    cfg = load_dataset_config(dataset)
    rag_path = Path(cfg["processed_dir"]) / "rag_labels.parquet"
    if not rag_path.exists():
        raise FileNotFoundError(
            f"use_rag_labels=true but {rag_path} not found. "
            f"Run scripts/05_generate_rag_labels.py --dataset {dataset} first."
        )
    rag_df = pd.read_parquet(rag_path).set_index("label_id")
    return [rag_df.loc[lid, "augmented_text"] for lid in labels_df["label_id"]]
