"""Fine-tunes the bi-encoder stage's backbone on a dataset's training set,
following the spirit of the xCoRetriev paper's dense-retriever training
(Section 3.2.2 / 4): contrastive learning that pulls a document's embedding
toward its true labels' embeddings.

Uses **relevance-aware in-batch negative mining**, following RAG-Fuse's
`RelevanceMiner` (the reference implementation of a closely related paper by
the same authors — see docs/06_rag_labels.md sibling notes): plain in-batch
contrastive training (e.g. sentence-transformers' MultipleNegativesRankingLoss)
treats every other positive pair's label in the batch as a negative for the
current anchor. On multi-label data this is sometimes wrong — if doc_i is also
truly labeled with label_j (just not the specific edge sampled for this
batch), training would incorrectly push doc_i's embedding away from label_j.
We mask out these false negatives at loss time, using the full label matrix
(not just the current batch) to know which (doc, label) pairs are genuinely
irrelevant vs. just not the one sampled this step.

Differences from the paper (deliberate simplifications, see docs/03_stages_and_models.md):
- Representation: sentence-transformers' standard mean-pooled output, not the
  paper's concatenation of the last 4 `[CLS]` hidden states.
- Single train/test split, no 5-fold cross-validation.
- Backbone stays all-MiniLM-L6-v2 (not the paper's full BERT-base), to keep
  this stage "lightweight" per the original project brief and fit comfortably
  in 8GB VRAM.

One (doc_text, true_label_text) pair is generated per (doc, label) edge in the
training set's label matrix — a doc with 5 labels contributes 5 training pairs.
"""

import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup

from cascade.utils.io import ensure_dir, load_yaml
from cascade.utils.logging import get_logger

logger = get_logger(__name__)


def build_training_pairs(
    train_df: pd.DataFrame, labels_df: pd.DataFrame
) -> tuple[list[tuple[int, str, int, str]], dict[int, frozenset[int]]]:
    """Returns (pairs, relevance_map).

    pairs: one (doc_id, doc_text, label_id, label_text) tuple per (doc, true
    label) edge.
    relevance_map: doc_id -> frozenset of ALL true label_ids for that doc
    (used to mask false negatives at loss time, not just to build positives).
    """
    label_text_by_id = dict(zip(labels_df["label_id"], labels_df["label_text"]))

    pairs = []
    relevance_map = {}
    for _, row in train_df.iterrows():
        doc_id = row["doc_id"]
        relevance_map[doc_id] = frozenset(row["label_ids"])
        for label_id in row["label_ids"]:
            pairs.append((doc_id, row["text"], label_id, label_text_by_id[label_id]))

    logger.info(f"Built {len(pairs)} (doc, true-label) training pairs from {len(train_df)} docs")
    return pairs, relevance_map


def _encode_batch(model: SentenceTransformer, texts: list[str], device: str) -> torch.Tensor:
    features = model.tokenize(texts)
    features = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in features.items()}
    return model(features)["sentence_embedding"]


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
    scale: float = 20.0,
    max_steps: int = -1,
    fp16: bool = True,
    seed: int = 42,
) -> Path:
    output_dir = output_dir or f"models/finetuned/{dataset_cfg['name']}/bi_encoder"
    ensure_dir(output_dir)

    pairs, relevance_map = build_training_pairs(train_df, labels_df)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(base_model_name, device=device)
    model.max_seq_length = doc_truncate_tokens
    model.train()

    n_steps_per_epoch = (len(pairs) + batch_size - 1) // batch_size
    total_steps = max_steps if max_steps > 0 else int(n_steps_per_epoch * num_train_epochs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device == "cuda")

    rng = random.Random(seed)
    step = 0
    epoch = 0
    done = False
    while not done:
        rng.shuffle(pairs)
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]
            doc_ids = [p[0] for p in batch]
            anchor_texts = [p[1] for p in batch]
            label_ids = [p[2] for p in batch]
            candidate_texts = [p[3] for p in batch]
            n = len(batch)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=fp16 and device == "cuda"):
                anchor_emb = _encode_batch(model, anchor_texts, device)
                candidate_emb = _encode_batch(model, candidate_texts, device)
                scores = anchor_emb @ candidate_emb.T * scale  # (n, n)

                # Mask false negatives: candidate j is excluded from the
                # denominator for anchor i if label_ids[j] is ALSO a true
                # label of doc_ids[i] (and isn't the intended positive i==j).
                false_negative_mask = torch.zeros(n, n, dtype=torch.bool, device=device)
                for i in range(n):
                    doc_true_labels = relevance_map[doc_ids[i]]
                    for j in range(n):
                        if j != i and label_ids[j] in doc_true_labels:
                            false_negative_mask[i, j] = True
                scores = scores.masked_fill(false_negative_mask, float("-inf"))

                targets = torch.arange(n, device=device)
                loss = F.cross_entropy(scores, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            if step % 50 == 0 or step == 1:
                logger.info(f"step {step}/{total_steps} epoch {epoch} loss {loss.item():.4f}")
            if step >= total_steps:
                done = True
                break
        epoch += 1

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
