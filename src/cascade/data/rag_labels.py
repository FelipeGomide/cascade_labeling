"""Generates RAG-labels: LLM-written descriptions for each label, grounded in
training documents that carry that label (Retrieval-Augmented Generation), as
in the xCoRetriev paper's "quality" mechanism. We skip the paper's separate
prompt-optimization loop (it compensates for a weak LLM; with a clear fixed
prompt template and a capable instruct model it's unnecessary) and go
straight to RAG generation with a fixed prompt.

For each label: retrieve up to n_examples training docs assigned to it, ask
a local instruct LLM to describe what the label means given that context,
and cache (label_id, label_text, rag_description) to rag_labels.parquet.
"""

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cascade.utils.io import ensure_dir, load_yaml
from cascade.utils.logging import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """Given the following set of texts that are all tagged with the label "{label_text}", \
write a concise (1-2 sentence) description of what this label means, based on what these texts have \
in common. Do not just repeat the label name; explain its meaning and context.

Texts:
{examples_block}

Description of label "{label_text}":"""


def _build_label_to_doc_indices(train_df: pd.DataFrame, n_labels: int) -> list[list[int]]:
    label_to_docs: list[list[int]] = [[] for _ in range(n_labels)]
    for doc_idx, label_ids in enumerate(train_df["label_ids"]):
        for label_id in label_ids:
            label_to_docs[label_id].append(doc_idx)
    return label_to_docs


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def generate_rag_labels(
    dataset_cfg: dict,
    train_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_cfg: dict,
    seed: int = 42,
    n_labels_total: int | None = None,
) -> Path:
    """labels_df may be a subset (e.g. for smoke-testing); n_labels_total should
    be the full label space size so label_id indices resolve correctly."""
    n_labels = n_labels_total or len(labels_df)
    label_to_docs = _build_label_to_doc_indices(train_df, n_labels)

    model_name = model_cfg["model_name"]
    device = model_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    n_examples = model_cfg.get("n_examples_per_label", 5)
    truncate_words = model_cfg.get("example_doc_truncate_words", 150)
    max_new_tokens = model_cfg.get("max_new_tokens", 80)

    logger.info(f"Loading {model_name} on {device} for RAG-label generation")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # required for correct batched causal-LM generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if (model_cfg.get("fp16", True) and device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()

    rng = __import__("random").Random(seed)
    batch_size = model_cfg.get("batch_size", 16)
    rows = list(labels_df.itertuples(index=False))
    descriptions: list[str | None] = [None] * len(rows)

    progress = tqdm(
        range(0, len(rows), batch_size),
        total=(len(rows) + batch_size - 1) // batch_size,
        desc="RAG-labels",
        unit="batch",
    )
    for batch_start in progress:
        batch = rows[batch_start : batch_start + batch_size]
        prompts, prompt_positions = [], []

        for pos, row in enumerate(batch, start=batch_start):
            doc_indices = label_to_docs[row.label_id]
            if not doc_indices:
                descriptions[pos] = ""
                continue
            sample_idx = rng.sample(doc_indices, min(n_examples, len(doc_indices)))
            examples = [
                _truncate_words(train_df.iloc[idx]["text"], truncate_words) for idx in sample_idx
            ]
            examples_block = "\n".join(f"- {ex}" for ex in examples)
            prompt = PROMPT_TEMPLATE.format(label_text=row.label_text, examples_block=examples_block)
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt_str)
            prompt_positions.append(pos)

        if prompts:
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_len = inputs["input_ids"].shape[1]  # uniform thanks to left-padding
            for j, pos in enumerate(prompt_positions):
                generated = tokenizer.decode(output[j][input_len:], skip_special_tokens=True)
                descriptions[pos] = generated.strip()

        done = min(batch_start + batch_size, len(rows))
        progress.set_postfix(labels=f"{done}/{n_labels}")

    out_df = pd.DataFrame(
        {
            "label_id": labels_df["label_id"],
            "label_text": labels_df["label_text"],
            "rag_description": descriptions,
        }
    )
    out_df["augmented_text"] = out_df.apply(
        lambda r: f"{r['label_text']}. {r['rag_description']}" if r["rag_description"] else r["label_text"],
        axis=1,
    )

    out_path = Path(dataset_cfg["processed_dir"]) / "rag_labels.parquet"
    ensure_dir(out_path.parent)
    out_df.to_parquet(out_path)
    logger.info(f"Wrote {len(out_df)} RAG-label descriptions -> {out_path}")
    return out_path


def main():
    import argparse

    from cascade.data.loaders import load_dataset_config, load_processed

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.dataset)
    model_cfg = load_yaml("configs/models/rag_label_generator.yaml")
    train_df, _, labels_df, _ = load_processed(args.dataset)
    generate_rag_labels(dataset_cfg, train_df, labels_df, model_cfg, seed=args.seed)


if __name__ == "__main__":
    main()
