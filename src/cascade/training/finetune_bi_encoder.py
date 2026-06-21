"""Fine-tunes the bi-encoder stage's backbone on a dataset's training set,
following the spirit of the xCoRetriev paper's dense-retriever training
(Section 3.2.2 / 4): contrastive learning that pulls a document's embedding
toward its true labels' embeddings.

Differences from the paper (deliberate simplifications, see docs/07_finetuning.md):
- Loss: `MultipleNegativesRankingLoss` (in-batch negatives / InfoNCE) instead
  of the paper's NT-Xent — same loss family, simpler to wire up with
  off-the-shelf sentence-transformers tooling.
- Representation: sentence-transformers' standard mean-pooled output, not the
  paper's concatenation of the last 4 `[CLS]` hidden states.
- Single train/test split, no 5-fold cross-validation.
- Backbone stays all-MiniLM-L6-v2 (not the paper's full BERT-base), to keep
  this stage "lightweight" per the original project brief and fit comfortably
  in 8GB VRAM.

One (doc_text, true_label_text) pair is generated per (doc, label) edge in the
training set's label matrix — a doc with 5 labels contributes 5 training pairs.
"""

from pathlib import Path

import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

from cascade.utils.io import ensure_dir, load_yaml
from cascade.utils.logging import get_logger

logger = get_logger(__name__)


def build_training_pairs(train_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dataset:
    label_text_by_id = dict(zip(labels_df["label_id"], labels_df["label_text"]))

    anchors, positives = [], []
    for _, row in train_df.iterrows():
        for label_id in row["label_ids"]:
            anchors.append(row["text"])
            positives.append(label_text_by_id[label_id])

    logger.info(f"Built {len(anchors)} (doc, true-label) training pairs from {len(train_df)} docs")
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def finetune_bi_encoder(
    dataset_cfg: dict,
    train_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str | None = None,
    num_train_epochs: float = 3.0,
    batch_size: int = 128,
    learning_rate: float = 2e-5,
    doc_truncate_tokens: int = 384,
    max_steps: int = -1,
    fp16: bool = True,
) -> Path:
    output_dir = output_dir or f"models/finetuned/{dataset_cfg['name']}/bi_encoder"
    ensure_dir(output_dir)

    train_dataset = build_training_pairs(train_df, labels_df)

    model = SentenceTransformer(base_model_name)
    model.max_seq_length = doc_truncate_tokens

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=f"{output_dir}/_checkpoints",
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=0.1,  # interpreted as a ratio of total steps (Transformers v5+)
        fp16=fp16,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    model.save(output_dir)
    logger.info(f"Saved fine-tuned bi-encoder -> {output_dir}")
    return Path(output_dir)


def main():
    import argparse

    from cascade.data.loaders import load_dataset_config, load_processed

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override for quick smoke tests")
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.dataset)
    train_df, _, labels_df, _ = load_processed(args.dataset)
    trunc = dataset_cfg.get("truncation", {})

    finetune_bi_encoder(
        dataset_cfg,
        train_df,
        labels_df,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        doc_truncate_tokens=trunc.get("encoder_doc_tokens", 384),
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
