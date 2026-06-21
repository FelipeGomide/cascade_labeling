"""Fine-tunes the cross-encoder stage on a dataset's training set.

Not covered by the xCoRetriev paper (the paper has no cross-encoder; this
stage is our own addition to the cascade, beyond what the paper does — see
docs/03_stages_and_models.md). We fine-tune it as a binary relevance
classifier: BinaryCrossEntropyLoss over (doc, label_text) pairs, the same
loss family used to train the base `ms-marco-MiniLM` model we start from.

Negatives are **mixed hard negatives** from two sources, split evenly: BM25
top-N candidates (lexically similar but incorrect) and bi-encoder top-N
candidates (semantically similar but incorrect, using an already fine-tuned
bi-encoder if available). This matters: a first version trained on BM25
negatives only collapsed when fed bi-encoder-sourced candidates downstream
(PS-nDCG@1 dropped to 0.012 from the bi-encoder's standalone 0.412) — the
model had never learned to discriminate that kind of confusable candidate.
Mixing both negative sources teaches it to handle candidates from either
upstream stage. See docs/06_rag_labels.md sibling doc on fine-tuning for
the full incident writeup.

Optional `anchor_truncate_words`: each training pair stores its own copy of
the full document text, duplicated once per (positive + negative) pair for
that doc -- on Eurlex-4k's shorter documents this was never an issue, but on
Wiki10-31k (docs averaging ~2000+ words, ~56 pairs/doc) it OOM-crashed the
build step at ~29 GiB RSS before training even started (confirmed via dmesg
OOM-killer). Pass `anchor_truncate_words` to truncate the stored document
text up front -- this is a side option, off by default, so existing behavior
(and Eurlex-4k's results) are unaffected unless explicitly requested.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

from cascade.stages.bi_encoder_stage import BiEncoderStage
from cascade.stages.bm25_stage import BM25Stage
from cascade.utils.io import ensure_dir
from cascade.utils.logging import get_logger

logger = get_logger(__name__)


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def build_training_pairs(
    train_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    n_negatives_per_positive: int = 2,
    hard_negative_pool_size: int = 50,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    bi_encoder_model_name: str | None = None,
    anchor_truncate_words: int | None = None,
    seed: int = 42,
) -> Dataset:
    label_text_by_id = dict(zip(labels_df["label_id"], labels_df["label_text"]))
    n_labels = len(labels_df)
    rng = random.Random(seed)
    texts = train_df["text"].tolist()
    if anchor_truncate_words is not None:
        texts = [_truncate_words(t, anchor_truncate_words) for t in texts]

    bm25 = BM25Stage(labels_df["label_text"].tolist(), k1=bm25_k1, b=bm25_b)
    bm25_results = bm25.rank_batch(texts, top_k=hard_negative_pool_size)

    bi_results = None
    if bi_encoder_model_name:
        logger.info(f"Mining bi-encoder hard negatives using {bi_encoder_model_name}")
        bi_encoder = BiEncoderStage(labels_df["label_text"].tolist(), model_name=bi_encoder_model_name)
        bi_results = bi_encoder.rank_batch(texts, top_k=hard_negative_pool_size)

    sentence1, sentence2, labels = [], [], []
    for i, (_, row) in enumerate(train_df.iterrows()):
        true_label_ids = set(row["label_ids"])
        text = texts[i]

        for label_id in true_label_ids:
            sentence1.append(text)
            sentence2.append(label_text_by_id[label_id])
            labels.append(1.0)

        n_negatives_needed = n_negatives_per_positive * len(true_label_ids)
        bm25_wrong = [c for c in bm25_results[i][0].tolist() if c not in true_label_ids]
        rng.shuffle(bm25_wrong)

        if bi_results is not None:
            n_from_bm25 = (n_negatives_needed + 1) // 2
            n_from_bi = n_negatives_needed - n_from_bm25
            bi_wrong = [c for c in bi_results[i][0].tolist() if c not in true_label_ids]
            rng.shuffle(bi_wrong)
            negatives = list(dict.fromkeys(bm25_wrong[:n_from_bm25] + bi_wrong[:n_from_bi]))
        else:
            negatives = bm25_wrong[:n_negatives_needed]

        # Pool exhausted (rare, short docs / small candidate pools): top up with random labels.
        if len(negatives) < n_negatives_needed:
            fallback_pool = [
                lid for lid in rng.sample(range(n_labels), min(n_labels, n_negatives_needed * 3))
                if lid not in true_label_ids and lid not in negatives
            ]
            negatives += fallback_pool[: n_negatives_needed - len(negatives)]

        for label_id in negatives[:n_negatives_needed]:
            sentence1.append(text)
            sentence2.append(label_text_by_id[label_id])
            labels.append(0.0)

    logger.info(
        f"Built {len(sentence1)} training pairs "
        f"({labels.count(1.0)} positive / {labels.count(0.0)} negative) from {len(train_df)} docs"
    )
    return Dataset.from_dict({"sentence1": sentence1, "sentence2": sentence2, "label": labels})


def finetune_cross_encoder(
    dataset_cfg: dict,
    train_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    base_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    output_dir: str | None = None,
    num_train_epochs: float = 3.0,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    n_negatives_per_positive: int = 2,
    bi_encoder_model_name: str | None = None,
    anchor_truncate_words: int | None = None,
    max_steps: int = -1,
    fp16: bool = True,
    seed: int = 42,
) -> Path:
    output_dir = output_dir or f"models/finetuned/{dataset_cfg['name']}/cross_encoder"
    ensure_dir(output_dir)

    train_dataset = build_training_pairs(
        train_df,
        labels_df,
        n_negatives_per_positive=n_negatives_per_positive,
        bi_encoder_model_name=bi_encoder_model_name,
        anchor_truncate_words=anchor_truncate_words,
        seed=seed,
    )

    model = CrossEncoder(base_model_name, num_labels=1, max_length=max_length)
    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir=f"{output_dir}/_checkpoints",
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=0.1,
        fp16=fp16,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )

    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)
    trainer.train()

    model.save_pretrained(output_dir)
    logger.info(f"Saved fine-tuned cross-encoder -> {output_dir}")
    return Path(output_dir)


def main():
    import argparse

    from cascade.data.loaders import load_dataset_config, load_processed

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--n-negatives", type=int, default=2)
    parser.add_argument(
        "--bi-encoder-model-name",
        default=None,
        help="Path/name of a (typically fine-tuned) bi-encoder to mine additional hard "
        "negatives from, mixed 50/50 with BM25 hard negatives. Omit for BM25-only negatives.",
    )
    parser.add_argument(
        "--anchor-truncate-words",
        type=int,
        default=None,
        help="Truncate each document's stored text to this many words before building training "
        "pairs. Off by default (matches original behavior); needed on datasets with long "
        "documents (e.g. Wiki10-31k) where storing the full text once per pair OOMs the build "
        "step. Eurlex-4k never needed this.",
    )
    parser.add_argument("--max-steps", type=int, default=-1, help="Override for quick smoke tests")
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.dataset)
    train_df, _, labels_df, _ = load_processed(args.dataset)

    finetune_cross_encoder(
        dataset_cfg,
        train_df,
        labels_df,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_negatives_per_positive=args.n_negatives,
        bi_encoder_model_name=args.bi_encoder_model_name,
        anchor_truncate_words=args.anchor_truncate_words,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
