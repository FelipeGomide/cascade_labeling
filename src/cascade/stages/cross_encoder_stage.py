import numpy as np
import torch
from sentence_transformers import CrossEncoder

from cascade.stages.base import Stage
from cascade.utils.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderStage(Stage):
    """Cross-encoder that jointly reads (doc, label_text) pairs and outputs a
    relevance score. Only ever scores a small per-doc candidate set in
    practice (N_test x L pairs over the full label space is infeasible — see
    README caveat); restrict via candidate_label_ids in cascade usage.
    """

    name = "cross_encoder"

    def __init__(
        self,
        label_texts: list[str],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        fp16: bool = True,
        batch_size: int = 64,
        max_length: int = 512,
    ):
        self.label_texts = label_texts
        self.batch_size = batch_size
        device = device if torch.cuda.is_available() else "cpu"

        model_kwargs = {"torch_dtype": torch.float16} if fp16 and device == "cuda" else {}
        self.model = CrossEncoder(
            model_name, device=device, max_length=max_length, model_kwargs=model_kwargs
        )

    def rank_batch(
        self,
        texts: list[str],
        candidate_label_ids: list[np.ndarray] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if candidate_label_ids is None:
            logger.warning(
                "CrossEncoderStage scoring full label space "
                f"({len(self.label_texts)} labels) per doc — only feasible for small label sets."
            )

        results = []
        for i, text in enumerate(texts):
            cand = (
                np.asarray(candidate_label_ids[i])
                if candidate_label_ids is not None
                else np.arange(len(self.label_texts))
            )
            if len(cand) == 0:
                results.append((cand, np.array([], dtype=np.float32)))
                continue

            pairs = [(text, self.label_texts[c]) for c in cand]
            scores = self.model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
            )
            k = min(top_k or len(cand), len(cand))
            order = np.argsort(-scores)[:k]
            results.append((cand[order], scores[order]))
        return results
