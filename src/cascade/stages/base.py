"""Common interface for cascade stages.

Every stage scores an input text against label texts (the "labels-as-documents"
framing: the label space is treated as a small retrieval corpus). `candidate_label_ids`
restricts scoring to a subset (used by later cascade stages); `None` means score
against the full label space (used by single-stage baselines and the first stage).
"""

from abc import ABC, abstractmethod

import numpy as np


class Stage(ABC):
    name: str

    @abstractmethod
    def rank_batch(
        self,
        texts: list[str],
        candidate_label_ids: list[np.ndarray] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Rank labels for a batch of texts.

        candidate_label_ids: per-text array of label ids to restrict scoring to,
            or None to score against the full label space.
        top_k: number of top labels to return per text (None = return all scored).

        Returns a list of (ranked_label_ids, scores) per text, both numpy arrays,
        sorted by descending score.
        """
        raise NotImplementedError
