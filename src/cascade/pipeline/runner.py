"""Runs one experiment config end-to-end: load data -> build cascade -> predict ->
evaluate -> profile -> persist everything under results/runs/<exp_id>/."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.data.loaders import load_dataset_config, load_label_texts, load_processed
from cascade.eval.metrics import evaluate_predictions, recall_at_k
from cascade.pipeline.config import build_pipeline
from cascade.profiling.hardware import get_hardware_info
from cascade.profiling.monitor import ResourceMonitor
from cascade.utils.io import ensure_dir, load_yaml
from cascade.utils.logging import get_logger
from cascade.utils.seeding import set_seed

logger = get_logger(__name__)


def run_experiment(config_path: str, seed: int = 42) -> Path:
    set_seed(seed)
    exp_cfg = load_yaml(config_path)
    exp_id = exp_cfg["exp_id"]
    dataset_name = exp_cfg["dataset"]
    eval_ks = tuple(exp_cfg.get("eval_ks", [1, 5, 10]))

    dataset_cfg = load_dataset_config(dataset_name)
    train_df, test_df, labels_df, inv_propensity = load_processed(dataset_name)

    label_texts_by_mode = {False: load_label_texts(dataset_name, labels_df, use_rag_labels=False)}
    try:
        label_texts_by_mode[True] = load_label_texts(dataset_name, labels_df, use_rag_labels=True)
    except FileNotFoundError:
        label_texts_by_mode[True] = None

    sample_size = exp_cfg.get("test_sample_size")
    if sample_size is not None and sample_size < len(test_df):
        test_df = test_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    texts = test_df["text"].tolist()
    true_labels = test_df["label_ids"].tolist()
    doc_ids = test_df["doc_id"].tolist()

    pipeline = build_pipeline(exp_cfg, dataset_cfg, label_texts_by_mode)

    logger.info(f"Running '{exp_id}' on {dataset_name} ({len(texts)} test docs)")
    with ResourceMonitor(exp_id) as total_mon:
        final_ids, final_scores, per_stage_log, stage_outputs = pipeline.run_batch(texts)

    preds = [ids.tolist() for ids in final_ids]
    metrics = evaluate_predictions(preds, true_labels, inv_propensity, ks=eval_ks)

    # First-stage candidate recall@N: caps what later stages can possibly recover.
    first_stage_name, first_stage_cutoff = pipeline.stages[0][0].name, pipeline.stages[0][1]
    first_stage_preds = [ids.tolist() for ids in stage_outputs[first_stage_name]]
    metrics[f"first_stage_recall@{first_stage_cutoff}"] = recall_at_k(
        first_stage_preds, true_labels, first_stage_cutoff
    )

    out_dir = ensure_dir(f"results/runs/{exp_id}")
    shutil.copy(config_path, out_dir / "config.yaml")

    pd.DataFrame(
        {
            "doc_id": doc_ids,
            "ranked_label_ids": preds,
            "scores": [s.tolist() for s in final_scores],
        }
    ).to_parquet(out_dir / "predictions.parquet")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    resources = {
        "hardware": get_hardware_info(),
        "total": total_mon.stats,
        "per_stage": {
            name: {
                "wall_time_s": log["wall_time_s"],
                "mean_candidates": float(np.mean(log["candidate_counts"])),
            }
            for name, log in per_stage_log.items()
        },
    }
    with open(out_dir / "resources.json", "w") as f:
        json.dump(resources, f, indent=2)

    per_stage_rows = []
    for stage_name, log in per_stage_log.items():
        n = len(log["candidate_counts"])
        avg_latency = log["wall_time_s"] / n if n else 0.0
        for doc_id, count in zip(doc_ids, log["candidate_counts"]):
            per_stage_rows.append(
                {
                    "doc_id": doc_id,
                    "stage": stage_name,
                    "candidate_count": count,
                    "approx_latency_s": avg_latency,
                }
            )
    pd.DataFrame(per_stage_rows).to_parquet(out_dir / "per_stage.parquet")

    logger.info(f"'{exp_id}' done -> {out_dir}. Metrics: {metrics}")
    return out_dir


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
