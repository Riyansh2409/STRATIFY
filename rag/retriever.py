"""
retriever.py
============
RAG Retriever — the core query engine.

Flow per query:
  1. Embed query        → 384-dim vector
  2. Vector search      → top-50 candidates (FAISS or ChromaDB)
  3. Cross-encoder      → rerank top-50 → top-5 best passages
  4. Build context      → concatenate passages with source citations
  5. Return             → RAGResponse (context + sources + scores)

Key design decisions:
  - Retrieval and generation are decoupled — this module only retrieves.
    Generation (Llama-3 call) is handled by generator.py.
  - MMR (Maximal Marginal Relevance) deduplication ensures diverse results.
  - Source tracking lets the generator cite exactly which file each passage came from.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .embeddings import EmbeddingEngine, EmbeddingConfig
from .vector_store import VectorStore, SearchResult, build_vector_store
from .reranker import CrossEncoderReranker, RankedResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrieverConfig:
    # Vector search
    vector_backend:  str = "faiss"       # "faiss" | "chroma"
    initial_top_k:   int = 50            # candidates before reranking
    final_top_k:     int = 5             # passages sent to LLM

    # Reranking
    use_reranker:    bool = True
    bi_weight:       float = 0.3
    cross_weight:    float = 0.7

    # MMR diversity
    use_mmr:         bool = True
    mmr_lambda:      float = 0.7         # 1.0 = pure relevance, 0.0 = pure diversity

    # Context building
    max_context_chars: int = 4000        # keep context under LLM token limit
    add_source_tags:   bool = True       # wrap each passage in [Source: ...] tags

    # Metadata filtering
    default_filters: Optional[dict] = None  # e.g. {"language": "en"}


# ─────────────────────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Everything the generator needs to build an answer."""
    query:          str
    context:        str               # formatted passages string for prompt
    passages:       list[RankedResult]
    sources:        list[dict]        # [{file, page, score, domain}, ...]
    retrieval_stats: dict = field(default_factory=dict)

    def format_for_prompt(self) -> str:
        """
        Returns a ready-to-use context block for the LLM prompt.

        Format:
            [Context 1 | Source: report.pdf | Score: 0.89]
            ... passage text ...

            [Context 2 | Source: data.csv | Score: 0.76]
            ... passage text ...
        """
        return self.context

    def has_results(self) -> bool:
        return len(self.passages) > 0


# ─────────────────────────────────────────────────────────────
# MMR (Maximal Marginal Relevance)
# ─────────────────────────────────────────────────────────────

def _mmr_rerank(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidates: list[RankedResult],
    top_k: int,
    lam: float = 0.7,
) -> list[RankedResult]:
    """
    MMR selects diverse yet relevant results.

    At each step, pick the candidate that maximises:
        lam * relevance(c, query) - (1-lam) * max_similarity(c, selected)

    lam=1.0 → pure relevance (same as regular ranking)
    lam=0.0 → pure diversity (ignore query, maximise spread)
    """
    if len(candidates) <= top_k:
        return candidates

    selected_indices: list[int] = []
    candidate_pool = list(range(len(candidates)))

    for _ in range(top_k):
        best_idx = -1
        best_score = -np.inf

        for i in candidate_pool:
            relevance = float(np.dot(query_vec, candidate_vecs[i]))

            if selected_indices:
                selected_vecs = candidate_vecs[selected_indices]
                redundancy = float(np.max(selected_vecs @ candidate_vecs[i]))
            else:
                redundancy = 0.0

            mmr_score = lam * relevance - (1 - lam) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        if best_idx < 0:
            break
        selected_indices.append(best_idx)
        candidate_pool.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# ─────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────

def _build_context(
    passages: list[RankedResult],
    max_chars: int,
    add_source_tags: bool,
) -> tuple[str, list[dict]]:
    """
    Concatenate passages into a context string for the LLM prompt.

    Returns:
        (context_string, sources_list)
    """
    parts: list[str] = []
    sources: list[dict] = []
    total_chars = 0

    for i, p in enumerate(passages, 1):
        from pathlib import Path
        src_name = Path(p.source_file).name if p.source_file else "unknown"
        score_str = f"{p.final_score:.3f}"

        if add_source_tags:
            header = f"[Context {i} | Source: {src_name} | Score: {score_str}]"
            block  = f"{header}\n{p.text}"
        else:
            block = p.text

        if total_chars + len(block) > max_chars and parts:
            break

        parts.append(block)
        total_chars += len(block)
        sources.append({
            "index":         i,
            "file":          src_name,
            "source_file":   p.source_file,
            "domain":        p.domain,
            "language":      p.language,
            "bi_score":      round(p.bi_score, 4),
            "cross_score":   round(p.cross_score, 4),
            "final_score":   round(p.final_score, 4),
            "chunk_index":   p.chunk_index,
        })

    return "\n\n".join(parts), sources


# ─────────────────────────────────────────────────────────────
# Main Retriever
# ─────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Full retrieval pipeline: embed → search → rerank → MMR → context.

    Usage:
        # One-time setup (index your chunks first)
        retriever = RAGRetriever.from_chunks(chunks, config=RetrieverConfig())

        # Query
        response = retriever.retrieve("Explain LoRA fine-tuning")
        print(response.context)
        for src in response.sources:
            print(f"  [{src['index']}] {src['file']} score={src['final_score']}")

        # Save index for reuse
        retriever.save("data/rag_index/")

        # Load later
        retriever = RAGRetriever.load("data/rag_index/")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_engine: EmbeddingEngine,
        reranker: Optional[CrossEncoderReranker] = None,
        config: Optional[RetrieverConfig] = None,
    ):
        self.store    = vector_store
        self.embedder = embedding_engine
        self.reranker = reranker
        self.config   = config or RetrieverConfig()

    # ── Factory: build from chunks ─────────────────────────

    @classmethod
    def from_chunks(
        cls,
        chunks: list,
        config: Optional[RetrieverConfig] = None,
        store_kwargs: Optional[dict] = None,
    ) -> "RAGRetriever":
        """
        Build a fully indexed retriever from a list of ProcessedChunk objects.

        Args:
            chunks:       list[ProcessedChunk] from preprocessor.py
            config:       RetrieverConfig (optional)
            store_kwargs: extra kwargs for the vector store constructor
        """
        cfg = config or RetrieverConfig()

        embedder = EmbeddingEngine(EmbeddingConfig())

        kwargs = store_kwargs or {}
        if cfg.vector_backend == "faiss":
            kwargs.setdefault("index_path", "data/faiss.index")
        else:
            kwargs.setdefault("persist_dir", "data/chroma_db")

        store = build_vector_store(cfg.vector_backend, embedder, **kwargs)
        store.add_chunks(chunks)

        reranker = None
        if cfg.use_reranker:
            reranker = CrossEncoderReranker(
                bi_weight=cfg.bi_weight,
                cross_weight=cfg.cross_weight,
            )

        logger.info(
            f"RAGRetriever ready. Backend={cfg.vector_backend}, "
            f"Chunks={store.count()}, Reranker={'on' if reranker else 'off'}"
        )
        return cls(store, embedder, reranker, cfg)

    # ── Main query method ─────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> RAGResponse:
        """
        Retrieve the most relevant passages for a query.

        Args:
            query:   Natural language question
            top_k:   Override config.final_top_k
            filters: Metadata filters, e.g. {"language": "en", "domain": "finance"}

        Returns:
            RAGResponse with context string + ranked passages + sources
        """
        final_k  = top_k or self.config.final_top_k
        search_k = max(self.config.initial_top_k, final_k * 10)
        active_filters = filters or self.config.default_filters

        if not query.strip():
            return RAGResponse(query=query, context="", passages=[], sources={})

        # Step 1 — Embed query
        query_vec = self.embedder.embed_query(query)

        # Step 2 — Vector search (over-fetch for reranking)
        candidates: list[SearchResult] = self.store.search(
            query_vec,
            top_k=search_k,
            filters=active_filters,
        )
        logger.debug(f"Vector search returned {len(candidates)} candidates")

        if not candidates:
            return RAGResponse(
                query=query, context="", passages=[], sources=[],
                retrieval_stats={"candidates": 0, "final": 0},
            )

        # Step 3 — Rerank
        if self.reranker:
            ranked: list[RankedResult] = self.reranker.rerank(
                query, candidates, top_k=min(len(candidates), final_k * 4)
            )
        else:
            # No reranker — wrap SearchResult as RankedResult
            from .reranker import RankedResult as RR
            ranked = [
                RR(
                    chunk_id=r.chunk_id, text=r.text,
                    bi_score=r.score, cross_score=r.score,
                    final_score=r.score,
                    source_file=r.source_file, file_type=r.file_type,
                    language=r.language, domain=r.domain,
                    quality_score=r.quality_score,
                    chunk_index=r.chunk_index, metadata=r.metadata,
                )
                for r in candidates
            ]

        # Step 4 — MMR diversity
        if self.config.use_mmr and len(ranked) > final_k:
            # Embed top-ranked passages for MMR
            ranked_texts = [r.text for r in ranked]
            ranked_vecs  = self.embedder.embed_texts(ranked_texts)
            ranked = _mmr_rerank(
                query_vec, ranked_vecs, ranked,
                top_k=final_k,
                lam=self.config.mmr_lambda,
            )
        else:
            ranked = ranked[:final_k]

        # Step 5 — Build context
        context, sources = _build_context(
            ranked,
            max_chars=self.config.max_context_chars,
            add_source_tags=self.config.add_source_tags,
        )

        return RAGResponse(
            query=query,
            context=context,
            passages=ranked,
            sources=sources,
            retrieval_stats={
                "candidates":     len(candidates),
                "after_rerank":   len(ranked),
                "final":          len(sources),
                "context_chars":  len(context),
            },
        )

    # ── Persistence ───────────────────────────────────────

    def save(self, dir_path: str):
        """Save index to disk (FAISS only; ChromaDB auto-persists)."""
        from pathlib import Path
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        if hasattr(self.store, "save"):
            self.store.save()
        logger.info(f"RAG index saved → {dir_path}")
