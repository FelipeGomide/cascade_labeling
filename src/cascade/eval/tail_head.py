"""Tail/head label split (Pareto convention, as in the xCoRetriev paper):
sorting labels by descending training-set frequency, the most frequent labels
making up the top 20% of the label count are "head"; the remaining 80% are "tail".
"""

import numpy as np
import pandas as pd


def compute_tail_label_ids(train_df: pd.DataFrame, n_labels: int, head_fraction: float = 0.2) -> set[int]:
    label_freq = np.zeros(n_labels, dtype=int)
    for ids in train_df["label_ids"]:
        for l in ids:
            label_freq[l] += 1

    order = np.argsort(-label_freq)  # most frequent first
    n_head = int(round(n_labels * head_fraction))
    head_ids = set(order[:n_head].tolist())
    tail_ids = set(range(n_labels)) - head_ids
    return tail_ids


def tail_proportion_at_k(
    predictions: list[list[int]], tail_label_ids: set[int], k: int
) -> float:
    """Mean fraction of a doc's top-k predicted labels that are tail labels."""
    fractions = []
    for pred in predictions:
        topk = pred[:k]
        if len(topk) == 0:
            continue
        fractions.append(sum(1 for l in topk if l in tail_label_ids) / len(topk))
    return float(np.mean(fractions)) if fractions else 0.0
