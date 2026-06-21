"""Scans results/runs/*/ and builds a single tidy long-format table
(results/summary/all_runs.parquet), one row per (run x metric), ready for
plotting with seaborn/plotly without any reshaping.

Columns: exp_id, dataset, pipeline, stage_cutoffs, metric_name, k, value,
total_time_s, cpu_time_s, peak_vram_mb, mean_gpu_util_pct, energy_j,
throughput_docs_per_s, timestamp.

Note: we report cpu_time_s, peak_vram_mb, mean_gpu_util_pct and energy_j as
directly measured by ResourceMonitor, rather than a synthetic "gpu_time_s"
split (which we cannot measure precisely without per-op CUDA profiling).
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from cascade.utils.io import load_yaml

_METRIC_RE = re.compile(r"^(?P<name>[a-z_]+)@(?P<k>\d+)$")


def _parse_metric_key(key: str) -> tuple[str, int | None]:
    m = _METRIC_RE.match(key)
    if m:
        return m.group("name"), int(m.group("k"))
    return key, None


def aggregate_results(runs_dir: str = "results/runs", out_path: str = "results/summary/all_runs.parquet") -> pd.DataFrame:
    rows = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"
        resources_path = run_dir / "resources.json"
        preds_path = run_dir / "predictions.parquet"
        if not (config_path.exists() and metrics_path.exists() and resources_path.exists()):
            continue

        cfg = load_yaml(config_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(resources_path) as f:
            resources = json.load(f)

        pipeline_str = "->".join(step["stage"] for step in cfg["pipeline"])
        cutoffs_str = ">".join(str(step["top_k"]) for step in cfg["pipeline"])

        n_docs = len(pd.read_parquet(preds_path)) if preds_path.exists() else None
        total = resources.get("total", {})
        wall_time_s = total.get("wall_time_s")
        throughput = (n_docs / wall_time_s) if (n_docs and wall_time_s) else None
        timestamp = datetime.fromtimestamp(resources_path.stat().st_mtime).isoformat()

        common = {
            "exp_id": cfg["exp_id"],
            "dataset": cfg["dataset"],
            "pipeline": pipeline_str,
            "stage_cutoffs": cutoffs_str,
            "total_time_s": wall_time_s,
            "cpu_time_s": total.get("cpu_time_s"),
            "peak_vram_mb": total.get("peak_vram_mb"),
            "mean_gpu_util_pct": total.get("mean_gpu_util_pct"),
            "energy_j": total.get("energy_j"),
            "throughput_docs_per_s": throughput,
            "timestamp": timestamp,
        }

        for key, value in metrics.items():
            metric_name, k = _parse_metric_key(key)
            rows.append({**common, "metric_name": metric_name, "k": k, "value": value})

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df


def main():
    df = aggregate_results()
    print(f"Aggregated {df['exp_id'].nunique()} runs, {len(df)} rows -> results/summary/all_runs.parquet")


if __name__ == "__main__":
    main()
