import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from cascade.stages.base import Stage
from cascade.utils.logging import get_logger

logger = get_logger(__name__)


class BiEncoderStage(Stage):
    """Bi-encoder over the label space: embeds the input text and each label's
    text into a shared space, ranks by cosine similarity.

    Label embeddings are precomputed once at init. When given no candidate
    restriction (first-stage / standalone use), search uses a FAISS flat index
    over the full label space. When restricted to a candidate set (cascade
    mode, following e.g. BM25), it scores only that per-doc candidate subset.
    """

    name = "bi_encoder"

    def __init__(
        self,
        label_texts: list[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        fp16: bool = True,
        batch_size: int = 256,
        doc_truncate_tokens: int = 384,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = doc_truncate_tokens
        if fp16 and self.device == "cuda":
            self.model = self.model.half()

        logger.info(f"Encoding {len(label_texts)} label texts with {model_name} on {self.device}")
        self.label_embeddings = self._encode(label_texts)  # (n_labels, dim), normalized

        self.index = faiss.IndexFlatIP(self.label_embeddings.shape[1])
        self.index.add(self.label_embeddings)

    def _encode(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return embeddings.astype(np.float32)

    def rank_batch(
        self,
        texts: list[str],
        candidate_label_ids: list[np.ndarray] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        query_embeddings = self._encode(texts)
        n_labels = self.label_embeddings.shape[0]
        k = min(top_k or n_labels, n_labels)

        if candidate_label_ids is None:
            scores, ids = self.index.search(query_embeddings, k)
            return [(ids[i], scores[i]) for i in range(len(texts))]

        results = []
        for i in range(len(texts)):
            cand = np.asarray(candidate_label_ids[i])
            if len(cand) == 0:
                results.append((cand, np.array([], dtype=np.float32)))
                continue
            sims = self.label_embeddings[cand] @ query_embeddings[i]
            order = np.argsort(-sims)[: min(k, len(cand))]
            results.append((cand[order], sims[order]))
        return results
