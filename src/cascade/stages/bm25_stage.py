import numpy as np

import bm25s

from cascade.stages.base import Stage
from cascade.utils.logging import get_logger

logger = get_logger(__name__)


class BM25Stage(Stage):
    """BM25 over the label space: corpus = label texts, query = input document.

    This is the labels-as-documents framing: each label is a tiny "document" in
    an inverted index, and we retrieve the labels whose text best matches the
    query text via BM25 term-frequency scoring.
    """

    name = "bm25"

    def __init__(
        self,
        label_texts: list[str],
        k1: float = 1.5,
        b: float = 0.75,
        query_truncate_tokens: int = 512,
    ):
        self.label_texts = label_texts
        self.n_labels = len(label_texts)
        self.query_truncate_tokens = query_truncate_tokens

        corpus_tokens = bm25s.tokenize(label_texts, show_progress=False)
        self.retriever = bm25s.BM25(k1=k1, b=b)
        self.retriever.index(corpus_tokens, show_progress=False)

    def _truncate(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) > self.query_truncate_tokens:
            text = " ".join(tokens[: self.query_truncate_tokens])
        return text

    def rank_batch(
        self,
        texts: list[str],
        candidate_label_ids: list[np.ndarray] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        queries = [self._truncate(t) for t in texts]
        query_tokens = bm25s.tokenize(queries, show_progress=False)

        k = top_k or self.n_labels
        k = min(k, self.n_labels)
        label_ids, scores = self.retriever.retrieve(
            query_tokens, k=k, show_progress=False
        )

        results = []
        for i in range(len(texts)):
            ids_i = label_ids[i]
            scores_i = scores[i]
            if candidate_label_ids is not None:
                mask = np.isin(ids_i, candidate_label_ids[i])
                ids_i, scores_i = ids_i[mask], scores_i[mask]
            results.append((ids_i, scores_i))
        return results
