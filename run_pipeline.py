"""
run_pipeline.py
===============
Entry point — runs the full preprocessing pipeline end-to-end.

Usage:
    python run_pipeline.py --input data/raw/ --output data/processed/ --report reports/

Steps:
    1. Load all files from --input (PDF, Excel, CSV, JSON, DOCX, TXT)
    2. Run PreprocessingPipeline (clean → tokenise → chunk → metadata → QA filter)
    3. Export train.jsonl  (90%) and val.jsonl (10%) splits
    4. Run DatasetAnalyser — chi-square, t-test, charts (Figs 1.1–3.3)
    5. Save pipeline_stats.json + analysis_report.json
    6. Print summary table to console
"""

import argparse
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Add local modules to path ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))

from file_loader import load_directory
from preprocessor import PreprocessingPipeline
from statistical_analysis import DatasetAnalyser


# ──────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LLM Fine-tuning Preprocessing Pipeline (Llama-3 8B)"
    )
    p.add_argument("--input",     required=True,  help="Directory with raw files")
    p.add_argument("--output",    default="data/processed", help="Output directory")
    p.add_argument("--report",    default="reports",        help="Report + chart directory")
    p.add_argument("--val-split", type=float, default=0.10, help="Validation split ratio")
    p.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    p.add_argument("--model-id",  default="meta-llama/Meta-Llama-3-8B",
                   help="HuggingFace model id for tokenizer")
    p.add_argument("--chunk-size",    type=int, default=512)
    p.add_argument("--overlap",       type=int, default=50)
    p.add_argument("--quality-threshold", type=float, default=0.70)
    return p.parse_args()


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def train_val_split(chunks, val_ratio: float, seed: int):
    random.seed(seed)
    shuffled = chunks[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def print_summary_table(stats, report):
    """Pretty-print pipeline + analysis summary to console."""
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print("  PIPELINE SUMMARY")
    print('═'*60)
    print(f"  Files processed        : {stats.total_files}")
    print(f"  Total characters       : {stats.total_raw_chars:,}")
    print(f"  Chunks (before filter) : {stats.total_chunks_before_filter}")
    print(f"  Chunks (after filter)  : {stats.total_chunks_after_filter}")
    filter_pct = 100*(1 - stats.total_chunks_after_filter /
                      max(stats.total_chunks_before_filter, 1))
    print(f"  Filter drop rate       : {filter_pct:.1f}%")
    print(f"  Dropped: too short     : {stats.dropped_too_short}")
    print(f"  Dropped: low quality   : {stats.dropped_low_quality}")
    print(f"  Dropped: duplicates    : {stats.dropped_duplicates}")
    print(sep)
    print("  LANGUAGE DISTRIBUTION")
    print(sep)
    for lang, count in sorted(stats.language_distribution.items(),
                               key=lambda x: -x[1])[:8]:
        pct = 100 * count / max(stats.total_chunks_after_filter, 1)
        bar = "█" * int(pct / 2)
        print(f"  {lang:<8} {count:>6}  {bar} {pct:.1f}%")
    print(sep)
    print("  STATISTICAL TESTS")
    print(sep)
    for t in report.test_results:
        status = "✓ SIGNIFICANT" if t.p_value < t.alpha else "○ not significant"
        print(f"  {t.test_name[:38]:<38}  p={t.p_value:.4f}  {status}")
    print(sep)
    print("  CHARTS SAVED")
    print(sep)
    for fig_id, path in report.chart_paths.items():
        print(f"  {fig_id:<10} → {path}")
    print('═'*60 + "\n")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.report).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  LLM Fine-tuning Preprocessing Pipeline — Llama-3 8B")
    logger.info("=" * 60)
    logger.info(f"  Input dir  : {args.input}")
    logger.info(f"  Output dir : {args.output}")
    logger.info(f"  Model      : {args.model_id}")
    logger.info(f"  Chunk size : {args.chunk_size} tokens | Overlap: {args.overlap}")
    logger.info(f"  QA threshold: {args.quality_threshold}")

    # ── Step 1: Load files ─────────────────────────────────
    logger.info("\n[Step 1/5] Loading files...")
    documents = load_directory(args.input, recursive=True)
    if not documents:
        logger.error(f"No supported files found in: {args.input}")
        sys.exit(1)
    logger.info(f"  Loaded {len(documents)} documents.")

    # ── Step 2: Preprocess ─────────────────────────────────
    logger.info("\n[Step 2/5] Running preprocessing pipeline...")
    pipeline = PreprocessingPipeline(
        model_id=args.model_id,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        quality_threshold=args.quality_threshold,
    )
    chunks, stats = pipeline.run(documents)

    if not chunks:
        logger.error("No chunks survived QA filtering. "
                     "Check your data or lower --quality-threshold.")
        sys.exit(1)

    # ── Step 3: Export ─────────────────────────────────────
    logger.info("\n[Step 3/5] Exporting train / val splits...")
    train_chunks, val_chunks = train_val_split(chunks, args.val_split, args.seed)

    train_path = str(Path(args.output) / "train.jsonl")
    val_path   = str(Path(args.output) / "val.jsonl")
    stats_path = str(Path(args.output) / "pipeline_stats.json")

    pipeline.export_jsonl(train_chunks, train_path)
    pipeline.export_jsonl(val_chunks,   val_path)
    pipeline.export_stats_json(stats,   stats_path)

    logger.info(f"  Train: {len(train_chunks)} chunks → {train_path}")
    logger.info(f"  Val  : {len(val_chunks)}  chunks → {val_path}")

    # ── Step 4: Statistical analysis + charts ──────────────
    logger.info("\n[Step 4/5] Running statistical analysis + generating charts...")
    analyser = DatasetAnalyser(output_dir=args.report)

    # Placeholder model metrics (replace with real eval output after training)
    sample_metrics = {
        "Base Llama-3 8B": {
            "BLEU":      0.21,
            "ROUGE-L":   0.34,
            "BERTScore": 0.78,
            "Perplexity": 18.4,
        },
        "Fine-tuned (LoRA)": {
            "BLEU":      0.38,
            "ROUGE-L":   0.51,
            "BERTScore": 0.86,
            "Perplexity": 11.2,
        },
    }

    analysis_report = analyser.analyse(
        chunks,
        model_metrics=sample_metrics,
    )
    analyser.save_report_json(
        analysis_report,
        str(Path(args.report) / "analysis_report.json"),
    )

    # ── Step 5: Summary ────────────────────────────────────
    logger.info("\n[Step 5/5] Pipeline complete.")
    print_summary_table(stats, analysis_report)


if __name__ == "__main__":
    main()
