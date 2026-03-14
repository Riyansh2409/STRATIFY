"""
reranker.py
===========
Cross-encoder reranker — improves top-k quality after initial retrieval.

Two-stage retrieval (standard RAG best practice):
  Stage 1 — Bi-encoder (fast)  : retrieve top-50 candidates from vector store
  Stage 2 — Cross-encoder (slow): rerank top-50 → return top-5 (much better accuracy)

Why reranking?
  Bi-encoders embed query and doc independently → fast but approximate.
  Cross-encoders see (query, doc) together → slower but much more accurate.
  Combined: speed of bi-encoder + accuracy of cross-encoder.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 12M params, fast CPU inference
  - Trained on MS MARCO passage ranking
  - Score range: ~-10 to +10 (higher = more relevant)
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RankedResult:
    """Reranked result with cross-encoder score."""
    chunk_id:      str
    text:          str
    bi_score:      float       # original vector similarity score
    cross_score:   float       # cross-encoder relevance score
    final_score:   float       # combined score (used for sorting)
    source_file:   str
    file_type:     str
    language:      str
    domain:        Optional[str]
    quality_score: float
    chunk_index:   int
    metadata:      dict

    def snippet(self, max_chars: int = 200) -> str:
        """Short preview of the chunk text."""
        return self.text[:max_chars] + ("..." if len(self.text) > max_chars else "")


class CrossEncoderReranker:
    """
    Reranks a list of SearchResult objects using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        ranked   = reranker.rerank(query="What is LoRA?", results=top50, top_k=5)
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str = "cpu",
        bi_weight: float = 0.3,     # weight for bi-encoder score in final blend
        cross_weight: float = 0.7,  # weight for cross-encoder score
    ):
        self.bi_weight    = bi_weight
        self.cross_weight = cross_weight
        self._model = None
        self._load_model(model_name, device)

    def _load_model(self, model_name: str, device: str):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name, device=device)
            logger.info(f"Cross-encoder loaded: {model_name}")
        except Exception as e:
            logger.warning(
                f"Could not load cross-encoder ({e}). "
                "Falling back to bi-encoder score only."
            )
            self._model = None

    def rerank(
        self,
        query: str,
        results: list,             # list[SearchResult]
        top_k: int = 5,
    ) -> list[RankedResult]:
        """
        Rerank a list of SearchResult objects.

        Args:
            query:   The user's query string
            results: Candidate SearchResult objects (from vector store)
            top_k:   How many to return after reranking

        Returns:
            List of RankedResult, sorted by final_score descending
        """
        if not results:
            return []

        top_k = min(top_k, len(results))

        if self._model is None:
            # No cross-encoder — return sorted by bi-encoder score
            return [
                RankedResult(
                    chunk_id=r.chunk_id, text=r.text,
                    bi_score=r.score, cross_score=r.score,
                    final_score=r.score,
                    source_file=r.source_file, file_type=r.file_type,
                    language=r.language, domain=r.domain,
                    quality_score=r.quality_score,
                    chunk_index=r.chunk_index, metadata=r.metadata,
                )
                for r in sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
            ]

        # Cross-encoder expects list of (query, passage) pairs
        pairs = [(query, r.text) for r in results]
        cross_scores = self._model.predict(pairs, show_progress_bar=False)

        # Normalise cross-encoder scores to [0, 1] with sigmoid
        import math
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))

        ranked = []
        for result, raw_cross in zip(results, cross_scores):
            cs = sigmoid(float(raw_cross))
            fs = self.bi_weight * result.score + self.cross_weight * cs
            ranked.append(RankedResult(
                chunk_id=result.chunk_id,
                text=result.text,
                bi_score=result.score,
                cross_score=cs,
                final_score=fs,
                source_file=result.source_file,
                file_type=result.file_type,
                language=result.language,
                domain=result.domain,
                quality_score=result.quality_score,
                chunk_index=result.chunk_index,
                metadata=result.metadata,
            ))

        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked[:top_k]
