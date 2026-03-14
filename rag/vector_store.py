"""
vector_store.py
===============
Dual vector store — FAISS (fast local index) + ChromaDB (persistent, filterable).

Architecture:
  - FAISS  → blazing-fast ANN search (in-memory or disk-backed .index file)
  - ChromaDB → metadata filtering, persistence, collection management

Both implement the same VectorStore interface so the retriever can
swap backends with zero code changes.

Usage:
    # Build FAISS index from chunks
    store = FAISSStore(embedder, index_path="data/faiss.index")
    store.add_chunks(chunks)
    results = store.search("What is LoRA?", top_k=5)

    # Or use ChromaDB for persistent storage with metadata filters
    store = ChromaStore(embedder, persist_dir="data/chroma_db")
    store.add_chunks(chunks)
    results = store.search("Explain tokenisation", top_k=5,
                            filters={"language": "en", "domain": "technology"})
"""

import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """One retrieved chunk with its similarity score."""
    chunk_id:      str
    text:          str
    score:         float            # cosine similarity (0–1, higher = better)
    source_file:   str
    file_type:     str
    language:      str
    domain:        Optional[str]
    quality_score: float
    chunk_index:   int
    metadata:      dict = field(default_factory=dict)

    def __repr__(self):
        return (f"SearchResult(score={self.score:.4f}, "
                f"source={Path(self.source_file).name!r}, "
                f"text={self.text[:60]!r}...)")


# ─────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────

class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list) -> int:
        """Add ProcessedChunk objects. Returns count added."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search by embedding vector. Returns top_k results."""

    @abstractmethod
    def count(self) -> int:
        """Return total chunks stored."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored chunks."""


# ─────────────────────────────────────────────────────────────
# FAISS Store
# ─────────────────────────────────────────────────────────────

class FAISSStore(VectorStore):
    """
    FAISS-backed vector store.

    Index type: IndexFlatIP (inner product = cosine sim on L2-normalised vecs)
    For large corpora (>100k chunks): upgrade to IndexIVFFlat for speed.

    Files saved:
      {index_path}         → FAISS binary index
      {index_path}.meta    → JSON list of chunk metadata
    """

    def __init__(
        self,
        embedding_engine,
        index_path: str = "data/faiss.index",
        dim: Optional[int] = None,
    ):
        self.embedder    = embedding_engine
        self.index_path  = index_path
        self._dim        = dim or embedding_engine.dimension
        self._meta: list[dict] = []          # parallel to FAISS vectors
        self._index      = None
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            self._faiss = faiss
            # Try loading existing index
            if Path(self.index_path).exists():
                logger.info(f"Loading existing FAISS index: {self.index_path}")
                self._index = faiss.read_index(self.index_path)
                meta_path = self.index_path + ".meta"
                if Path(meta_path).exists():
                    with open(meta_path) as f:
                        self._meta = json.load(f)
                logger.info(f"FAISS index loaded — {self._index.ntotal} vectors")
            else:
                # Inner product (cosine on normalised vecs)
                self._index = faiss.IndexFlatIP(self._dim)
                logger.info(f"FAISS IndexFlatIP created. Dim={self._dim}")
        except ImportError:
            logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
            raise

    def add_chunks(self, chunks: list) -> int:
        texts      = [c.text for c in chunks]
        vectors    = self.embedder.embed_texts(texts)
        self._index.add(vectors)
        for c in chunks:
            self._meta.append({
                "chunk_id":      c.chunk_id,
                "text":          c.text,
                "source_file":   c.source_file,
                "file_type":     c.file_type,
                "language":      c.language,
                "domain":        c.domain,
                "quality_score": c.quality_score,
                "chunk_index":   c.chunk_index,
                "token_count":   c.token_count,
            })
        logger.info(f"Added {len(chunks)} chunks → FAISS total: {self._index.ntotal}")
        return len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        # Over-fetch then filter
        fetch_k = min(top_k * 10, self._index.ntotal) if filters else top_k
        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._meta[idx]
            # Apply metadata filters
            if filters:
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue
            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                score=float(score),
                source_file=meta["source_file"],
                file_type=meta["file_type"],
                language=meta["language"],
                domain=meta.get("domain"),
                quality_score=meta["quality_score"],
                chunk_index=meta["chunk_index"],
                metadata=meta,
            ))
            if len(results) == top_k:
                break

        return results

    def save(self):
        """Persist FAISS index + metadata to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, self.index_path)
        with open(self.index_path + ".meta", "w") as f:
            json.dump(self._meta, f)
        logger.info(f"FAISS index saved → {self.index_path} ({self.count()} vectors)")

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def clear(self) -> None:
        import faiss
        self._index = faiss.IndexFlatIP(self._dim)
        self._meta  = []


# ─────────────────────────────────────────────────────────────
# ChromaDB Store
# ─────────────────────────────────────────────────────────────

class ChromaStore(VectorStore):
    """
    ChromaDB-backed vector store — persistent, filterable by metadata.

    Use this when:
      - You need to filter by language/domain/source_file at query time
      - You want persistence without managing FAISS files manually
      - You need to update/delete individual chunks

    ChromaDB handles embedding storage internally using its own format.
    We pass pre-computed embeddings to avoid double-encoding.
    """

    COLLECTION_NAME = "llm_pipeline_chunks"

    def __init__(
        self,
        embedding_engine,
        persist_dir: str = "data/chroma_db",
        collection_name: str = COLLECTION_NAME,
    ):
        self.embedder = embedding_engine
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self):
        try:
            import chromadb
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB ready: '{self.collection_name}' "
                f"({self._collection.count()} docs) at {self.persist_dir}"
            )
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise

    def add_chunks(self, chunks: list) -> int:
        """Add chunks in batches of 500 (ChromaDB limit)."""
        BATCH = 500
        total_added = 0
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            texts      = [c.text        for c in batch]
            ids        = [c.chunk_id    for c in batch]
            vectors    = self.embedder.embed_texts(texts).tolist()
            metadatas  = [
                {
                    "source_file":   c.source_file,
                    "file_type":     c.file_type,
                    "language":      c.language,
                    "domain":        c.domain or "unknown",
                    "quality_score": c.quality_score,
                    "chunk_index":   c.chunk_index,
                    "token_count":   c.token_count,
                }
                for c in batch
            ]
            self._collection.add(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
            total_added += len(batch)
            logger.info(f"ChromaDB: added batch {i//BATCH + 1} ({len(batch)} docs)")

        logger.info(f"ChromaDB total: {self._collection.count()} documents")
        return total_added

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        if self._collection.count() == 0:
            return []

        where = None
        if filters:
            # ChromaDB where syntax: {"$and": [{"key": {"$eq": "val"}}, ...]}
            conditions = [
                {k: {"$eq": v}} for k, v in filters.items()
            ]
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        raw = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]
        ids       = raw["ids"][0]

        for doc, meta, dist, cid in zip(docs, metas, distances, ids):
            # ChromaDB cosine distance → similarity: 1 - distance
            score = float(1.0 - dist)
            results.append(SearchResult(
                chunk_id=cid,
                text=doc,
                score=score,
                source_file=meta.get("source_file", ""),
                file_type=meta.get("file_type", ""),
                language=meta.get("language", "unknown"),
                domain=meta.get("domain"),
                quality_score=float(meta.get("quality_score", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata=meta,
            ))

        return results

    def count(self) -> int:
        return self._collection.count() if self._collection else 0

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection cleared.")


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def build_vector_store(
    backend: str,
    embedding_engine,
    **kwargs,
) -> VectorStore:
    """
    Factory function — choose backend by name.

    Args:
        backend: "faiss" | "chroma"
        embedding_engine: EmbeddingEngine instance
        **kwargs: passed to the store constructor

    Example:
        store = build_vector_store("faiss", embedder,
                                   index_path="data/faiss.index")
        store = build_vector_store("chroma", embedder,
                                   persist_dir="data/chroma_db")
    """
    backend = backend.lower().strip()
    if backend == "faiss":
        return FAISSStore(embedding_engine, **kwargs)
    elif backend in ("chroma", "chromadb"):
        return ChromaStore(embedding_engine, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'faiss' or 'chroma'.")
