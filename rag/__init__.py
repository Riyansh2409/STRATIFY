"""
rag package
===========
Two-stage RAG pipeline: embed → FAISS/ChromaDB → rerank → MMR → Llama-3 generate.

Public API:
    from rag.embeddings   import EmbeddingEngine, EmbeddingConfig
    from rag.vector_store import FAISSStore, ChromaStore, build_vector_store, SearchResult
    from rag.reranker     import CrossEncoderReranker, RankedResult
    from rag.retriever    import RAGRetriever, RetrieverConfig, RAGResponse
    from rag.generator    import RAGGenerator, GeneratorConfig, GeneratedAnswer
"""
from rag.embeddings   import EmbeddingEngine, EmbeddingConfig
from rag.vector_store import FAISSStore, ChromaStore, build_vector_store, SearchResult
from rag.reranker     import CrossEncoderReranker, RankedResult
from rag.retriever    import RAGRetriever, RetrieverConfig, RAGResponse
from rag.generator    import RAGGenerator, GeneratorConfig, GeneratedAnswer

__all__ = [
    "EmbeddingEngine", "EmbeddingConfig",
    "FAISSStore", "ChromaStore", "build_vector_store", "SearchResult",
    "CrossEncoderReranker", "RankedResult",
    "RAGRetriever", "RetrieverConfig", "RAGResponse",
    "RAGGenerator", "GeneratorConfig", "GeneratedAnswer",
]
