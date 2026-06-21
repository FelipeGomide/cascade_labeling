"""Chains Stage instances into a cascade with per-stage candidate cutoffs.

Each stage narrows the candidate set: stage i receives, for each doc, the
label-id candidates surfaced by stage i-1 (top_k of stage i-1's output), and
the first stage scores the full label space (candidates=None).
"""

import time

import numpy as np

from cascade.stages.base import Stage


class CascadePipeline:
    def __init__(self, stages: list[tuple[Stage, int]], batch_size: int = 64):
        """stages: ordered list of (stage_instance, top_k_cutoff)."""
        self.stages = stages
        self.batch_size = batch_size

    def run_batch(
        self, texts: list[str]
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict, dict]:
        """Run the full cascade over a list of texts.

        Returns (final_label_ids_per_doc, final_scores_per_doc, per_stage_log,
        stage_outputs) where per_stage_log maps stage_name -> {"wall_time_s": float,
        "candidate_counts": list[int]}, and stage_outputs maps stage_name ->
        list[np.ndarray] of that stage's own ranked ids (for first-stage recall@N).
        """
        n = len(texts)
        candidate_ids: list[np.ndarray] | None = None
        per_stage_log: dict = {}
        stage_outputs: dict = {}

        for stage, top_k in self.stages:
            stage_ids: list[np.ndarray] = [None] * n  # type: ignore
            stage_scores: list[np.ndarray] = [None] * n  # type: ignore
            wall_time_s = 0.0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_texts = texts[start:end]
                batch_candidates = candidate_ids[start:end] if candidate_ids is not None else None

                t0 = time.perf_counter()
                batch_results = stage.rank_batch(batch_texts, batch_candidates, top_k=top_k)
                wall_time_s += time.perf_counter() - t0

                for i, (ids, scores) in enumerate(batch_results):
                    stage_ids[start + i] = ids
                    stage_scores[start + i] = scores

            per_stage_log[stage.name] = {
                "wall_time_s": wall_time_s,
                "candidate_counts": [len(ids) for ids in stage_ids],
            }
            stage_outputs[stage.name] = stage_ids
            candidate_ids = stage_ids
            final_scores = stage_scores

        return candidate_ids, final_scores, per_stage_log, stage_outputs
