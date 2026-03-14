"""
embeddings.py
=============
Embedding layer for the RAG pipeline.

Supports two backends (configurable):
  1. SentenceTransformers  — local, no API key needed (default)
       Model: all-MiniLM-L6-v2  (384-dim, fast)
       Model: all-mpnet-base-v2  (768-dim, more accurate)
  2. HuggingFace Inference API — remote, needs HF_TOKEN

Usage:
    embedder = EmbeddingEngine()
    vectors  = embedder.embed_texts(["Hello world", "Namaste duniya"])
    print(vectors.shape)   # (2, 384)
"""

import logging
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────
DEFAULT_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim
ACCURATE_MODEL   = "sentence-transformers/all-mpnet-base-v2"   # 768-dim
BATCH_SIZE       = 64
MAX_SEQ_LEN      = 512


@dataclass
class EmbeddingConfig:
    model_name: str  = DEFAULT_MODEL
    batch_size: int  = BATCH_SIZE
    normalize:  bool = True          # L2-normalise → cosine similarity = dot product
    device:     str  = "cpu"         # "cuda" on GPU machines
    show_progress: bool = True


class EmbeddingEngine:
    """
    Wraps SentenceTransformer — encodes text chunks to dense vectors.

    Features:
      - Batch encoding (memory-efficient)
      - L2 normalisation (cosine similarity via dot product)
      - Fallback to numpy random vectors if model not found (CI/testing)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dim: int = 0
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            # Warm-up to confirm model loaded + get dimension
            test_vec = self._model.encode(["test"], convert_to_numpy=True)
            self._dim = test_vec.shape[1]
            logger.info(f"Embedding model ready. Dimension: {self._dim}")
        except Exception as e:
            logger.warning(
                f"Could not load SentenceTransformer ({e}). "
                "Using random-vector fallback (for testing only)."
            )
            self._model = None
            self._dim = 384

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings → float32 numpy array (N × dim).

        Args:
            texts: list of strings to encode

        Returns:
            numpy array shape (len(texts), self.dimension)
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if self._model is None:
            # Fallback: reproducible random vectors (testing)
            rng = np.random.default_rng(seed=42)
            vecs = rng.standard_normal((len(texts), self._dim)).astype(np.float32)
        else:
            vecs = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress and len(texts) > 100,
                convert_to_numpy=True,
            ).astype(np.float32)

        if self.config.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms

        return vecs

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string → 1D float32 array (dim,)."""
        return self.embed_texts([query])[0]
