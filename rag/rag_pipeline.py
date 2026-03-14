"""
rag_pipeline.py
===============
End-to-end RAG pipeline orchestrator.

Two modes:
  1. INDEX mode  — load preprocessed chunks → build vector index
  2. QUERY mode  — interactive query loop (or single query via --query flag)

Usage:
    # Build index from preprocessed data
    python rag_pipeline.py index \
        --data  data/processed/train.jsonl \
        --index data/rag_index/ \
        --backend faiss

    # Interactive query loop
    python rag_pipeline.py query \
        --index data/rag_index/ \
        --backend faiss

    # Single query
    python rag_pipeline.py query \
        --index data/rag_index/ \
        --query "Explain LoRA fine-tuning" \
        --top-k 5

    # Query with metadata filter
    python rag_pipeline.py query \
        --index data/rag_index/ \
        --query "What is tokenisation?" \
        --filter-lang en \
        --filter-domain technology
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from embeddings import EmbeddingEngine, EmbeddingConfig
from vector_store import build_vector_store
from reranker import CrossEncoderReranker
from retriever import RAGRetriever, RetrieverConfig
from generator import RAGGenerator, GeneratorConfig


# ─────────────────────────────────────────────────────────────
# Chunk loader (reads from preprocessor JSONL output)
# ─────────────────────────────────────────────────────────────

@dataclass
class MinimalChunk:
    """Lightweight chunk loaded from JSONL — no token_ids needed for RAG."""
    chunk_id:      str
    text:          str
    token_count:   int
    source_file:   str
    file_type:     str
    language:      str
    quality_score: float
    domain:        str | None
    chunk_index:   int


def load_chunks_from_jsonl(path: str) -> list[MinimalChunk]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(MinimalChunk(
                chunk_id=d.get("chunk_id", ""),
                text=d.get("text", ""),
                token_count=d.get("token_count", 0),
                source_file=d.get("source_file", ""),
                file_type=d.get("file_type", ""),
                language=d.get("language", "en"),
                quality_score=float(d.get("quality_score", 1.0)),
                domain=d.get("domain"),
                chunk_index=int(d.get("chunk_index", 0)),
            ))
    logger.info(f"Loaded {len(chunks):,} chunks from {path}")
    return chunks


# ─────────────────────────────────────────────────────────────
# Build index
# ─────────────────────────────────────────────────────────────

def build_index(args):
    logger.info("=" * 55)
    logger.info("  RAG Pipeline — INDEX mode")
    logger.info("=" * 55)

    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    chunks = load_chunks_from_jsonl(args.data)
    if not chunks:
        logger.error("No chunks found. Run run_pipeline.py first.")
        sys.exit(1)

    config = RetrieverConfig(
        vector_backend=args.backend,
        use_reranker=not args.no_reranker,
        use_mmr=not args.no_mmr,
        final_top_k=args.top_k,
    )

    store_kwargs: dict = {}
    index_dir = Path(args.index)
    index_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "faiss":
        store_kwargs["index_path"] = str(index_dir / "faiss.index")
    else:
        store_kwargs["persist_dir"] = str(index_dir / "chroma_db")

    retriever = RAGRetriever.from_chunks(
        chunks,
        config=config,
        store_kwargs=store_kwargs,
    )

    # Save FAISS index to disk
    retriever.save(str(index_dir))

    # Save config for loading later
    meta = {
        "backend":      args.backend,
        "chunk_count":  retriever.store.count(),
        "index_dir":    str(index_dir),
        "top_k":        args.top_k,
        "use_reranker": not args.no_reranker,
        "use_mmr":      not args.no_mmr,
        "data_source":  args.data,
    }
    with open(index_dir / "rag_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\n✓ Index built successfully.")
    logger.info(f"  Backend  : {args.backend}")
    logger.info(f"  Chunks   : {retriever.store.count():,}")
    logger.info(f"  Index dir: {index_dir}")
    logger.info(f"  Reranker : {'on' if not args.no_reranker else 'off'}")
    logger.info(f"  MMR      : {'on' if not args.no_mmr else 'off'}")
    logger.info(f"\nNow query with:")
    logger.info(f"  python rag_pipeline.py query --index {index_dir}\n")


# ─────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────

def load_retriever(args) -> RAGRetriever:
    """Load existing index."""
    index_dir  = Path(args.index)
    meta_path  = index_dir / "rag_meta.json"

    if not meta_path.exists():
        logger.error(f"No rag_meta.json found in {index_dir}. Run index mode first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    backend = meta.get("backend", args.backend)
    embedder = EmbeddingEngine(EmbeddingConfig())

    store_kwargs: dict = {}
    if backend == "faiss":
        store_kwargs["index_path"] = str(index_dir / "faiss.index")
    else:
        store_kwargs["persist_dir"] = str(index_dir / "chroma_db")

    store = build_vector_store(backend, embedder, **store_kwargs)

    config = RetrieverConfig(
        vector_backend=backend,
        use_reranker=meta.get("use_reranker", True) and not args.no_reranker,
        use_mmr=meta.get("use_mmr", True) and not args.no_mmr,
        final_top_k=args.top_k or meta.get("top_k", 5),
    )

    reranker = None
    if config.use_reranker:
        reranker = CrossEncoderReranker()

    logger.info(f"RAG index loaded: {store.count():,} chunks ({backend})")
    return RAGRetriever(store, embedder, reranker, config)


def run_query(retriever: RAGRetriever, generator: RAGGenerator, args):
    """Run a single query and print the result."""
    filters: dict = {}
    if hasattr(args, "filter_lang") and args.filter_lang:
        filters["language"] = args.filter_lang
    if hasattr(args, "filter_domain") and args.filter_domain:
        filters["domain"] = args.filter_domain

    rag_resp = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        filters=filters or None,
    )

    if not rag_resp.has_results():
        print("\n[No relevant passages found for this query.]")
        print("Tips: check --filter-lang / --filter-domain, or rephrase the query.\n")
        return

    answer = generator.generate(rag_resp)
    answer.print_answer()

    # Optionally save to JSON
    if hasattr(args, "output") and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(answer.to_dict(), f, indent=2)
        logger.info(f"Answer saved → {out_path}")


def query_mode(args):
    retriever = load_retriever(args)
    gen_config = GeneratorConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    generator = RAGGenerator(gen_config)

    if args.query:
        # Single query
        run_query(retriever, generator, args)
    else:
        # Interactive loop
        print("\n" + "=" * 55)
        print("  RAG Query Mode — Llama-3 8B")
        print(f"  Index: {args.index}  |  Backend: {retriever.store.count():,} chunks")
        print("  Type 'quit' or Ctrl+C to exit")
        print("=" * 55 + "\n")
        while True:
            try:
                query = input("Query > ").strip()
                if not query:
                    continue
                if query.lower() in ("quit", "exit", "q"):
                    break
                args.query = query
                run_query(retriever, generator, args)
            except KeyboardInterrupt:
                print("\nBye!")
                break


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def eval_mode(args):
    """
    Run evaluation on a JSONL file of query-answer pairs.

    Evaluation file format (one JSON object per line):
      {"query": "What is LoRA?", "expected": "LoRA is a parameter-efficient..."}

    Computes:
      - Retrieval hit rate (expected answer in retrieved context?)
      - Mean retrieval score
      - Mean latency
    """
    if not Path(args.eval_file).exists():
        logger.error(f"Eval file not found: {args.eval_file}")
        sys.exit(1)

    retriever = load_retriever(args)
    generator = RAGGenerator(GeneratorConfig(max_new_tokens=256))

    with open(args.eval_file) as f:
        eval_items = [json.loads(l) for l in f if l.strip()]

    logger.info(f"Evaluating {len(eval_items)} queries...")
    results = []
    hits = 0

    for item in eval_items:
        q = item["query"]
        expected = item.get("expected", "")
        rag_resp = retriever.retrieve(q)
        answer   = generator.generate(rag_resp)

        # Hit = expected answer keyword appears in retrieved context
        hit = expected.lower()[:50] in rag_resp.context.lower() if expected else False
        if hit:
            hits += 1

        results.append({
            "query":       q,
            "answer":      answer.answer[:200],
            "hit":         hit,
            "confidence":  answer.confidence,
            "latency_sec": answer.latency_sec,
            "sources":     [s["file"] for s in answer.sources],
        })

    hit_rate = hits / len(eval_items) if eval_items else 0
    mean_conf = sum(r["confidence"] for r in results) / len(results)
    mean_lat  = sum(r["latency_sec"] for r in results) / len(results)

    sep = "─" * 50
    print(f"\n{sep}")
    print("  RAG Evaluation Results")
    print(sep)
    print(f"  Queries evaluated : {len(eval_items)}")
    print(f"  Hit rate          : {hit_rate*100:.1f}%  ({hits}/{len(eval_items)})")
    print(f"  Mean confidence   : {mean_conf:.4f}")
    print(f"  Mean latency      : {mean_lat:.3f}s")
    print(sep + "\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "hit_rate":       round(hit_rate, 4),
                "mean_confidence": round(mean_conf, 4),
                "mean_latency":   round(mean_lat, 3),
                "results":        results,
            }, f, indent=2)
        logger.info(f"Eval results saved → {args.output}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RAG Pipeline — Llama-3 8B"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── index ──
    idx = sub.add_parser("index", help="Build vector index from JSONL chunks")
    idx.add_argument("--data",        required=True, help="Path to train.jsonl")
    idx.add_argument("--index",       default="data/rag_index/", help="Index output dir")
    idx.add_argument("--backend",     default="faiss", choices=["faiss", "chroma"])
    idx.add_argument("--top-k",       type=int, default=5)
    idx.add_argument("--no-reranker", action="store_true")
    idx.add_argument("--no-mmr",      action="store_true")

    # ── query ──
    qry = sub.add_parser("query", help="Query the RAG pipeline")
    qry.add_argument("--index",       default="data/rag_index/", help="Index dir")
    qry.add_argument("--backend",     default="faiss", choices=["faiss", "chroma"])
    qry.add_argument("--query",       default=None,   help="Single query (or interactive)")
    qry.add_argument("--top-k",       type=int, default=5)
    qry.add_argument("--max-tokens",  type=int, default=512)
    qry.add_argument("--temperature", type=float, default=0.2)
    qry.add_argument("--filter-lang",   default=None, help="Filter by language code")
    qry.add_argument("--filter-domain", default=None, help="Filter by domain")
    qry.add_argument("--output",      default=None,   help="Save answer JSON to path")
    qry.add_argument("--no-reranker", action="store_true")
    qry.add_argument("--no-mmr",      action="store_true")

    # ── eval ──
    evl = sub.add_parser("eval", help="Evaluate retrieval on labelled query set")
    evl.add_argument("--index",       default="data/rag_index/")
    evl.add_argument("--backend",     default="faiss", choices=["faiss", "chroma"])
    evl.add_argument("--eval-file",   required=True, help="JSONL with query+expected pairs")
    evl.add_argument("--output",      default=None,   help="Save eval results JSON")
    evl.add_argument("--top-k",       type=int, default=5)
    evl.add_argument("--no-reranker", action="store_true")
    evl.add_argument("--no-mmr",      action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "index":
        build_index(args)
    elif args.mode == "query":
        query_mode(args)
    elif args.mode == "eval":
        eval_mode(args)
