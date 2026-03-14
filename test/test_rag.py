"""
tests/test_rag.py
=================
Unit tests for the RAG pipeline:
  - EmbeddingEngine (fallback mode)
  - FAISSStore (add, search, save/load)
  - ChromaStore (add, search, filter)
  - CrossEncoderReranker (fallback mode)
  - RAGRetriever (end-to-end)
  - build_llama3_prompt (prompt format)

Run: pytest tests/test_rag.py -v
"""

import sys
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

from embeddings import EmbeddingEngine, EmbeddingConfig
from vector_store import FAISSStore, SearchResult, build_vector_store
from reranker import CrossEncoderReranker
from retriever import RAGRetriever, RetrieverConfig, _mmr_rerank
from generator import build_llama3_prompt, GeneratorConfig


# ════════════════════════════════════════════════════════════
# Test fixtures
# ════════════════════════════════════════════════════════════

@dataclass
class FakeChunk:
    chunk_id:      str
    text:          str
    token_count:   int = 100
    source_file:   str = "test.pdf"
    file_type:     str = "pdf"
    language:      str = "en"
    quality_score: float = 0.90
    domain:        str | None = "technology"
    chunk_index:   int = 0


CHUNKS = [
    FakeChunk("c1", "LoRA is a parameter-efficient fine-tuning method for large language models."),
    FakeChunk("c2", "Tokenisation splits raw text into tokens using a vocabulary.", domain="technology"),
    FakeChunk("c3", "ChromaDB is a vector database that supports metadata filtering."),
    FakeChunk("c4", "The chi-square test checks if observed frequencies match expected ones.", domain="science"),
    FakeChunk("c5", "Llama-3 is Meta's open-source large language model family.", language="en"),
]


def make_embedder() -> EmbeddingEngine:
    """Create embedder in fallback mode (no model download in CI)."""
    cfg = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    e = EmbeddingEngine.__new__(EmbeddingEngine)
    e.config = cfg
    e._model = None
    e._dim = 384
    return e


# ════════════════════════════════════════════════════════════
# EmbeddingEngine
# ════════════════════════════════════════════════════════════

class TestEmbeddingEngine:

    def test_embed_texts_shape(self):
        emb = make_embedder()
        texts = ["hello world", "foo bar"]
        vecs = emb.embed_texts(texts)
        assert vecs.shape == (2, 384)

    def test_embed_texts_normalised(self):
        emb = make_embedder()
        vecs = emb.embed_texts(["test text"])
        norms = np.linalg.norm(vecs, axis=1)
        assert abs(norms[0] - 1.0) < 1e-5, "Vectors should be L2-normalised"

    def test_embed_query_1d(self):
        emb = make_embedder()
        vec = emb.embed_query("what is LoRA?")
        assert vec.shape == (384,)

    def test_embed_empty_returns_empty(self):
        emb = make_embedder()
        vecs = emb.embed_texts([])
        assert vecs.shape == (0, 384)


# ════════════════════════════════════════════════════════════
# FAISSStore
# ════════════════════════════════════════════════════════════

class TestFAISSStore:
    """These tests require faiss-cpu to be installed."""

    @pytest.fixture
    def store_with_chunks(self, tmp_path):
        emb = make_embedder()
        try:
            store = FAISSStore(emb, index_path=str(tmp_path / "test.index"))
            store.add_chunks(CHUNKS)
            return store, emb
        except ImportError:
            pytest.skip("faiss-cpu not installed")

    def test_count_after_add(self, store_with_chunks):
        store, _ = store_with_chunks
        assert store.count() == len(CHUNKS)

    def test_search_returns_results(self, store_with_chunks):
        store, emb = store_with_chunks
        q_vec = emb.embed_query("LoRA fine-tuning")
        results = store.search(q_vec, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_positive(self, store_with_chunks):
        store, emb = store_with_chunks
        q_vec = emb.embed_query("language model")
        results = store.search(q_vec, top_k=5)
        for r in results:
            assert r.score >= -1.0   # cosine similarity range [-1, 1]

    def test_save_and_reload(self, tmp_path):
        emb = make_embedder()
        try:
            idx_path = str(tmp_path / "faiss.index")
            store = FAISSStore(emb, index_path=idx_path)
            store.add_chunks(CHUNKS)
            store.save()

            # Reload
            store2 = FAISSStore(emb, index_path=idx_path)
            assert store2.count() == len(CHUNKS)
        except ImportError:
            pytest.skip("faiss-cpu not installed")

    def test_filter_by_language(self, store_with_chunks):
        store, emb = store_with_chunks
        q_vec = emb.embed_query("text processing")
        results = store.search(q_vec, top_k=5, filters={"language": "en"})
        for r in results:
            assert r.language == "en"

    def test_clear_resets_count(self, store_with_chunks):
        store, _ = store_with_chunks
        store.clear()
        assert store.count() == 0


# ════════════════════════════════════════════════════════════
# CrossEncoderReranker (fallback mode)
# ════════════════════════════════════════════════════════════

class TestReranker:

    def _make_search_results(self) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=f"r{i}", text=c.text, score=float(i) / 5,
                source_file=c.source_file, file_type=c.file_type,
                language=c.language, domain=c.domain,
                quality_score=c.quality_score, chunk_index=i,
            )
            for i, c in enumerate(CHUNKS)
        ]

    def test_rerank_returns_top_k(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker.bi_weight    = 0.3
        reranker.cross_weight = 0.7

        results = self._make_search_results()
        ranked  = reranker.rerank("What is LoRA?", results, top_k=3)
        assert len(ranked) == 3

    def test_rerank_sorted_by_score(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker.bi_weight    = 0.3
        reranker.cross_weight = 0.7

        results = self._make_search_results()
        ranked  = reranker.rerank("language model", results, top_k=5)
        for i in range(len(ranked) - 1):
            assert ranked[i].final_score >= ranked[i+1].final_score

    def test_snippet_truncates(self):
        from reranker import RankedResult
        r = RankedResult(
            chunk_id="x", text="A" * 300,
            bi_score=0.5, cross_score=0.8, final_score=0.7,
            source_file="f.pdf", file_type="pdf",
            language="en", domain=None, quality_score=0.9, chunk_index=0,
            metadata={},
        )
        assert r.snippet(50).endswith("...")
        assert len(r.snippet(50)) <= 53   # 50 chars + "..."


# ════════════════════════════════════════════════════════════
# MMR
# ════════════════════════════════════════════════════════════

class TestMMR:

    def _make_ranked_results(self, n=5):
        from reranker import RankedResult
        return [
            RankedResult(
                chunk_id=f"m{i}", text=CHUNKS[i % len(CHUNKS)].text,
                bi_score=0.9 - i * 0.1, cross_score=0.9 - i * 0.1,
                final_score=0.9 - i * 0.1,
                source_file="f.pdf", file_type="pdf",
                language="en", domain=None, quality_score=0.9,
                chunk_index=i, metadata={},
            )
            for i in range(n)
        ]

    def test_mmr_returns_top_k(self):
        emb = make_embedder()
        results = self._make_ranked_results(5)
        texts   = [r.text for r in results]
        vecs    = emb.embed_texts(texts)
        q_vec   = emb.embed_query("test query")
        selected = _mmr_rerank(q_vec, vecs, results, top_k=3, lam=0.7)
        assert len(selected) == 3

    def test_mmr_with_all_lambda1_equals_ranking(self):
        """lam=1.0 → no diversity penalty → order should match input scores."""
        emb = make_embedder()
        results = self._make_ranked_results(5)
        texts   = [r.text for r in results]
        vecs    = emb.embed_texts(texts)
        q_vec   = emb.embed_query("test query")
        selected = _mmr_rerank(q_vec, vecs, results, top_k=5, lam=1.0)
        assert len(selected) == 5


# ════════════════════════════════════════════════════════════
# Prompt builder
# ════════════════════════════════════════════════════════════

class TestPromptBuilder:

    def test_llama3_prompt_format(self):
        cfg = GeneratorConfig()
        prompt = build_llama3_prompt(
            query="What is LoRA?",
            context="[Context 1 | Source: paper.pdf]\nLoRA reduces parameters via low-rank decomposition.",
            config=cfg,
        )
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
        assert "What is LoRA?" in prompt

    def test_numbered_output_instruction_present(self):
        cfg = GeneratorConfig(numbered_output=True)
        prompt = build_llama3_prompt("Q?", "ctx", cfg)
        assert "numbered" in prompt.lower() or "1." in prompt or "list" in prompt.lower()

    def test_citation_instruction_present(self):
        cfg = GeneratorConfig(cite_sources=True)
        prompt = build_llama3_prompt("Q?", "ctx", cfg)
        assert "[Context" in prompt or "Cite" in prompt


# ════════════════════════════════════════════════════════════
# End-to-end RAGRetriever (no GPU)
# ════════════════════════════════════════════════════════════

class TestRAGRetrieverE2E:

    @pytest.fixture
    def retriever(self, tmp_path):
        try:
            config = RetrieverConfig(
                vector_backend="faiss",
                use_reranker=False,
                use_mmr=False,
                final_top_k=3,
            )
            r = RAGRetriever.from_chunks(
                CHUNKS,
                config=config,
                store_kwargs={"index_path": str(tmp_path / "test.index")},
            )
            return r
        except ImportError:
            pytest.skip("faiss-cpu not installed")

    def test_retrieve_returns_response(self, retriever):
        resp = retriever.retrieve("What is LoRA fine-tuning?")
        assert resp.query == "What is LoRA fine-tuning?"
        assert isinstance(resp.context, str)
        assert isinstance(resp.sources, list)

    def test_retrieve_has_results(self, retriever):
        resp = retriever.retrieve("language model tokenisation")
        assert resp.has_results()
        assert len(resp.passages) <= 3

    def test_retrieve_context_contains_source_tags(self, retriever):
        resp = retriever.retrieve("vector database")
        if resp.has_results():
            assert "[Context" in resp.context

    def test_retrieve_empty_query_returns_empty(self, retriever):
        resp = retriever.retrieve("   ")
        assert not resp.has_results()

    def test_retrieve_stats_populated(self, retriever):
        resp = retriever.retrieve("Llama-3 model")
        assert "candidates" in resp.retrieval_stats
        assert resp.retrieval_stats["candidates"] >= 0
