"""XMTC evaluation metrics: Precision@k, nDCG@k, and their propensity-scored
variants PSP@k / PS-nDCG@k (Jain et al., 2016), matching the metrics used in the
xCoRetriev paper. Implemented directly in numpy (no pyxclib dependency, which
requires a fragile Cython build) but following the same formulas:

  P@k      = (1/k) * sum_{i<=k} rel_i
  nDCG@k   = DCG@k / IDCG@k,        DCG@k = sum_{i<=k} rel_i / log2(i+1)
  PSP@k    = (1/k) * sum_{i<=k} rel_i * inv_propensity(label_i)
  PSnDCG@k = PSDCG@k / PSIDCG@k,    PSDCG@k weights rel_i by inv_propensity(label_i);
             PSIDCG@k is the best achievable PSDCG@k, i.e. ranking the doc's true
             labels by descending inverse propensity first.
"""

import numpy as np


def _dcg_weights(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def evaluate_predictions(
    predictions: list[list[int]],
    true_labels: list[list[int]],
    inv_propensity: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Average P@k, nDCG@k, PSP@k, PS-nDCG@k over a set of documents.

    predictions[i]  - ranked list of predicted label_ids for doc i (best first)
    true_labels[i]  - list of gold label_ids for doc i
    inv_propensity  - array indexed by label_id, 1/p_l per label
    """
    n = len(predictions)
    max_k = max(ks)
    weights = _dcg_weights(max_k)

    sums = {f"{metric}@{k}": 0.0 for metric in ("precision", "ndcg", "psp", "psndcg") for k in ks}

    for pred, true in zip(predictions, true_labels):
        true_set = set(true)
        n_true = len(true_set)

        pred_k = list(pred[:max_k])
        rel = np.zeros(max_k)
        prop = np.zeros(max_k)
        for i, lbl in enumerate(pred_k):
            if lbl in true_set:
                rel[i] = 1.0
            prop[i] = inv_propensity[lbl]

        true_inv_prop_desc = np.sort(
            np.array([inv_propensity[l] for l in true_set]) if n_true else np.array([])
        )[::-1]

        for k in ks:
            sums[f"precision@{k}"] += rel[:k].sum() / k

            dcg = (rel[:k] * weights[:k]).sum()
            idcg = weights[: min(k, n_true)].sum() if n_true > 0 else 0.0
            sums[f"ndcg@{k}"] += (dcg / idcg) if idcg > 0 else 0.0

            sums[f"psp@{k}"] += (rel[:k] * prop[:k]).sum() / k

            psdcg = (rel[:k] * prop[:k] * weights[:k]).sum()
            top_true_prop = true_inv_prop_desc[:k]
            psidcg = (top_true_prop * weights[: len(top_true_prop)]).sum()
            sums[f"psndcg@{k}"] += (psdcg / psidcg) if psidcg > 0 else 0.0

    return {key: total / n for key, total in sums.items()}


def recall_at_k(
    predictions: list[list[int]],
    true_labels: list[list[int]],
    k: int,
) -> float:
    """Mean fraction of a doc's true labels present in its top-k predictions.

    Used to validate first-stage (e.g. BM25) candidate cutoffs: it sets an upper
    bound on what later cascade stages can possibly recover.
    """
    n = len(predictions)
    total = 0.0
    for pred, true in zip(predictions, true_labels):
        true_set = set(true)
        if not true_set:
            continue
        topk = set(pred[:k])
        total += len(topk & true_set) / len(true_set)
    return total / n
