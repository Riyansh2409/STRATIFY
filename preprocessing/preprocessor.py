"""
preprocessor.py
===============
Core preprocessing pipeline for Llama-3 8B fine-tuning.

Stages (in order):
  1. Text Cleaning       — normalise, strip noise, detect language
  2. Tokenisation        — Llama-3 8B tokenizer (HuggingFace)
  3. Chunking            — 512-token windows, 50-token overlap
  4. Metadata Extraction — source, date, language, domain, token count
  5. QA Filtering        — deduplicate (MinHash), quality score, length

Output:
  List[ProcessedChunk] → ready for fine-tuning dataset or RAG ingestion.
"""

import re
import hashlib
import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from transformers import AutoTokenizer
from langdetect import detect as langdetect_detect, LangDetectException

from file_loader import LoadedDocument

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Config — change these without touching pipeline logic
# ─────────────────────────────────────────────────────────────

LLAMA3_MODEL_ID      = "meta-llama/Meta-Llama-3-8B"
CHUNK_TOKEN_SIZE     = 512
CHUNK_OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS     = 50
MAX_CHUNK_TOKENS     = 512
QUALITY_THRESHOLD    = 0.70     # chunks below this score are dropped


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class ProcessedChunk:
    """A single training-ready chunk with full metadata."""
    chunk_id: str                           # SHA256 of text (dedup key)
    text: str                               # clean chunk text
    token_count: int
    token_ids: list[int]                    # Llama-3 token ids

    # Provenance
    source_file: str
    file_type: str
    chunk_index: int                        # position within document
    total_chunks: int

    # Metadata
    language: str = "en"
    quality_score: float = 1.0
    domain: Optional[str] = None           # auto or manual tag
    extraction_date: str = ""
    page_hint: Optional[int] = None       # page number if from PDF

    # Extra
    extra: dict = field(default_factory=dict)


@dataclass
class PipelineStats:
    total_files: int = 0
    total_raw_chars: int = 0
    total_chunks_before_filter: int = 0
    total_chunks_after_filter: int = 0
    dropped_too_short: int = 0
    dropped_low_quality: int = 0
    dropped_duplicates: int = 0
    language_distribution: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Stage 1 — Text Cleaning
# ─────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Normalise and clean raw extracted text.
    Order matters: unicode → html → whitespace → special chars → numbers.
    """

    _HTML_TAG_RE      = re.compile(r"<[^>]+>")
    _URL_RE           = re.compile(r"https?://\S+|www\.\S+")
    _EMAIL_RE         = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _MULTI_SPACE_RE   = re.compile(r"[ \t]{2,}")
    _CONTROL_CHAR_RE  = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # 1. Unicode NFKC normalisation (handles ligatures, full-width, etc.)
        text = unicodedata.normalize("NFKC", text)

        # 2. Remove HTML/XML tags (common in scraped data)
        text = self._HTML_TAG_RE.sub(" ", text)

        # 3. Replace URLs and emails with placeholder tokens
        text = self._URL_RE.sub("[URL]", text)
        text = self._EMAIL_RE.sub("[EMAIL]", text)

        # 4. Remove control characters (keep \n and \t)
        text = self._CONTROL_CHAR_RE.sub("", text)

        # 5. Collapse multiple blank lines → single blank line
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)

        # 6. Collapse multiple spaces/tabs
        text = self._MULTI_SPACE_RE.sub(" ", text)

        # 7. Strip leading/trailing whitespace
        text = text.strip()

        return text


# ─────────────────────────────────────────────────────────────
# Stage 2 — Tokenisation (Llama-3 8B)
# ─────────────────────────────────────────────────────────────

class Llama3Tokenizer:
    """
    Wraps HuggingFace AutoTokenizer for Meta-Llama-3-8B.

    Falls back to tiktoken cl100k if model weights are not downloaded
    (useful for local testing without GPU).
    """

    def __init__(self, model_id: str = LLAMA3_MODEL_ID):
        logger.info(f"Loading tokenizer: {model_id}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                trust_remote_code=False,
            )
            # Llama-3 vocab size = 128,256
            logger.info(f"Tokenizer loaded. Vocab size: {self.tokenizer.vocab_size:,}")
        except Exception as e:
            logger.warning(f"Could not load Llama-3 tokenizer ({e}). "
                           "Falling back to tiktoken cl100k for testing.")
            import tiktoken
            self._fallback = tiktoken.get_encoding("cl100k_base")
            self.tokenizer = None

    def encode(self, text: str) -> list[int]:
        if self.tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self._fallback.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return self._fallback.decode(token_ids)

    def token_count(self, text: str) -> int:
        return len(self.encode(text))


# ─────────────────────────────────────────────────────────────
# Stage 3 — Chunking
# ─────────────────────────────────────────────────────────────

class TokenChunker:
    """
    Sliding-window token chunker.

    Strategy:
      - Encode full text once → token_ids list
      - Slide window of size=CHUNK_TOKEN_SIZE with step=(size - overlap)
      - Decode each window back to text
      - Preserve sentence boundaries where possible (soft boundary)

    Why token-level (not char-level)?
      Llama-3's training expects consistent token-length inputs.
      Char splits cause uneven token distributions that hurt loss curves.
    """

    def __init__(
        self,
        tokenizer: Llama3Tokenizer,
        chunk_size: int = CHUNK_TOKEN_SIZE,
        overlap: int = CHUNK_OVERLAP_TOKENS,
    ):
        self.tok = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def chunk(self, text: str) -> list[tuple[str, list[int]]]:
        """
        Returns list of (chunk_text, token_ids) tuples.
        """
        all_ids = self.tok.encode(text)
        total = len(all_ids)
        chunks: list[tuple[str, list[int]]] = []

        start = 0
        while start < total:
            end = min(start + self.chunk_size, total)
            window_ids = all_ids[start:end]
            window_text = self.tok.decode(window_ids)
            chunks.append((window_text.strip(), window_ids))
            if end == total:
                break
            start += self.step

        return chunks


# ─────────────────────────────────────────────────────────────
# Stage 4 — Metadata Extraction
# ─────────────────────────────────────────────────────────────

class MetadataExtractor:
    """
    Enrich each chunk with:
      - language (langdetect)
      - domain (keyword heuristic — extend as needed)
      - extraction_date
      - page_hint (for PDFs)
    """

    DOMAIN_KEYWORDS: dict[str, list[str]] = {
        "medical":    ["patient", "diagnosis", "clinical", "treatment", "hospital"],
        "legal":      ["contract", "clause", "jurisdiction", "liability", "court"],
        "finance":    ["revenue", "profit", "balance sheet", "equity", "cashflow"],
        "technology": ["algorithm", "neural", "model", "software", "api", "dataset"],
        "science":    ["experiment", "hypothesis", "molecule", "reaction", "quantum"],
    }

    def detect_language(self, text: str) -> str:
        try:
            sample = text[:500]
            return langdetect_detect(sample)
        except LangDetectException:
            return "unknown"

    def detect_domain(self, text: str) -> Optional[str]:
        lower = text.lower()
        scores: dict[str, int] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in lower)
            if count > 0:
                scores[domain] = count
        if not scores:
            return None
        return max(scores, key=scores.get)

    def extract_page_hint(self, text: str, file_type: str) -> Optional[int]:
        """Extract page number from PDF page markers like [Page 3]."""
        if file_type != "pdf":
            return None
        match = re.search(r"\[Page\s+(\d+)\]", text)
        return int(match.group(1)) if match else None


# ─────────────────────────────────────────────────────────────
# Stage 5 — QA Filtering
# ─────────────────────────────────────────────────────────────

class QualityFilter:
    """
    Filter and deduplicate chunks.

    Quality score (0.0–1.0) is a composite of:
      - unique_word_ratio    (penalise repetitive text)
      - alpha_ratio          (penalise number/symbol-heavy chunks)
      - no_stop_words_penalty (very short sentences with no connectors)
      - token_length_score   (reward mid-length, penalise extremes)

    Deduplication:
      - SHA256 of normalised text (exact dedup)
      - Falls back to 5-gram MinHash similarity for near-dedup
        (requires datasketch; gracefully skips if not installed)
    """

    def __init__(self, threshold: float = QUALITY_THRESHOLD):
        self.threshold = threshold
        self._seen_hashes: set[str] = set()

        # Try to load MinHash for near-dedup
        try:
            from datasketch import MinHash, MinHashLSH
            self._lsh = MinHashLSH(threshold=0.85, num_perm=128)
            self._minhash_cls = MinHash
            self._use_minhash = True
            logger.info("MinHash LSH near-dedup enabled.")
        except ImportError:
            self._use_minhash = False
            logger.info("datasketch not found — using exact SHA256 dedup only.")

    def _compute_hash(self, text: str) -> str:
        normalised = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalised.encode()).hexdigest()

    def _compute_minhash(self, text: str):
        m = self._minhash_cls(num_perm=128)
        # 5-gram shingling
        words = text.lower().split()
        for i in range(len(words) - 4):
            gram = " ".join(words[i:i+5])
            m.update(gram.encode())
        return m

    def is_duplicate(self, text: str, chunk_id: str) -> bool:
        # 1. Exact hash check
        h = self._compute_hash(text)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)

        # 2. Near-dedup via MinHash LSH
        if self._use_minhash and len(text.split()) >= 10:
            mh = self._compute_minhash(text)
            try:
                result = self._lsh.query(mh)
                if result:
                    return True
                self._lsh.insert(chunk_id, mh)
            except Exception:
                pass

        return False

    def quality_score(self, text: str, token_count: int) -> float:
        words = text.split()
        if not words:
            return 0.0

        # Feature 1: Unique word ratio (1.0 = all unique)
        unique_ratio = len(set(w.lower() for w in words)) / len(words)

        # Feature 2: Alpha character ratio (penalise mostly digits/symbols)
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / max(len(text), 1)

        # Feature 3: Token length score — optimal range 100–450
        if token_count < MIN_CHUNK_TOKENS:
            len_score = token_count / MIN_CHUNK_TOKENS
        elif token_count > MAX_CHUNK_TOKENS * 0.95:
            len_score = 0.8                    # slightly penalise full windows
        else:
            len_score = 1.0

        # Feature 4: Average word length (too short = noise)
        avg_word_len = sum(len(w) for w in words) / len(words)
        word_len_score = min(avg_word_len / 4.0, 1.0)

        # Weighted composite
        score = (
            unique_ratio   * 0.35 +
            alpha_ratio    * 0.30 +
            len_score      * 0.20 +
            word_len_score * 0.15
        )
        return round(min(score, 1.0), 4)

    def passes(self, text: str, token_count: int, chunk_id: str) -> tuple[bool, float, str]:
        """
        Returns (passes: bool, score: float, rejection_reason: str).
        """
        # Length check first (fast)
        if token_count < MIN_CHUNK_TOKENS:
            return False, 0.0, "too_short"

        score = self.quality_score(text, token_count)
        if score < self.threshold:
            return False, score, "low_quality"

        if self.is_duplicate(text, chunk_id):
            return False, score, "duplicate"

        return True, score, ""


# ─────────────────────────────────────────────────────────────
# Main Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────

class PreprocessingPipeline:
    """
    Runs all 5 stages in sequence on a list of LoadedDocuments.

    Usage:
        pipeline = PreprocessingPipeline()
        docs = load_directory("data/raw/")
        chunks, stats = pipeline.run(docs)
        pipeline.export_jsonl(chunks, "data/processed/train.jsonl")
    """

    def __init__(
        self,
        model_id: str = LLAMA3_MODEL_ID,
        chunk_size: int = CHUNK_TOKEN_SIZE,
        overlap: int = CHUNK_OVERLAP_TOKENS,
        quality_threshold: float = QUALITY_THRESHOLD,
    ):
        logger.info("Initialising preprocessing pipeline...")
        self.cleaner    = TextCleaner()
        self.tokenizer  = Llama3Tokenizer(model_id)
        self.chunker    = TokenChunker(self.tokenizer, chunk_size, overlap)
        self.meta       = MetadataExtractor()
        self.qa         = QualityFilter(quality_threshold)
        self.stats      = PipelineStats()
        logger.info("Pipeline ready.")

    def _process_document(self, doc: LoadedDocument) -> list[ProcessedChunk]:
        """Process one LoadedDocument → list of ProcessedChunk."""
        if doc.load_error:
            logger.warning(f"Skipping {doc.file_path} (load error: {doc.load_error})")
            return []

        # Stage 1 — Clean
        clean_text = self.cleaner.clean(doc.raw_text)
        if not clean_text:
            return []

        self.stats.total_raw_chars += len(clean_text)

        # Stage 3 — Chunk (tokenisation happens inside chunker)
        raw_chunks = self.chunker.chunk(clean_text)
        total_chunks = len(raw_chunks)
        self.stats.total_chunks_before_filter += total_chunks

        chunks: list[ProcessedChunk] = []
        today = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        for idx, (chunk_text, token_ids) in enumerate(raw_chunks):
            token_count = len(token_ids)
            chunk_id = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]

            # Stage 5 — QA Filter (early, before expensive metadata)
            passes, score, reason = self.qa.passes(chunk_text, token_count, chunk_id)
            if not passes:
                if reason == "too_short":
                    self.stats.dropped_too_short += 1
                elif reason == "low_quality":
                    self.stats.dropped_low_quality += 1
                elif reason == "duplicate":
                    self.stats.dropped_duplicates += 1
                continue

            # Stage 4 — Metadata
            lang    = self.meta.detect_language(chunk_text)
            domain  = self.meta.detect_domain(chunk_text)
            page    = self.meta.extract_page_hint(chunk_text, doc.file_type)

            # Update language stats
            self.stats.language_distribution[lang] = \
                self.stats.language_distribution.get(lang, 0) + 1

            chunks.append(ProcessedChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                token_count=token_count,
                token_ids=token_ids,
                source_file=doc.file_path,
                file_type=doc.file_type,
                chunk_index=idx,
                total_chunks=total_chunks,
                language=lang,
                quality_score=score,
                domain=domain,
                extraction_date=today,
                page_hint=page,
            ))

        self.stats.total_chunks_after_filter += len(chunks)
        return chunks

    def run(
        self,
        documents: list[LoadedDocument],
        verbose: bool = True,
    ) -> tuple[list[ProcessedChunk], PipelineStats]:
        """
        Run pipeline on all documents.

        Returns:
            (all_chunks, stats)
        """
        self.stats = PipelineStats()
        self.stats.total_files = len(documents)

        all_chunks: list[ProcessedChunk] = []

        for i, doc in enumerate(documents, 1):
            if verbose:
                logger.info(f"[{i}/{len(documents)}] Processing: {doc.file_path}")
            chunks = self._process_document(doc)
            all_chunks.extend(chunks)
            if verbose:
                logger.info(f"  → {len(chunks)} chunks retained")

        logger.info(
            f"\n{'='*50}\n"
            f"Pipeline complete.\n"
            f"  Files processed   : {self.stats.total_files}\n"
            f"  Raw chars         : {self.stats.total_raw_chars:,}\n"
            f"  Chunks (before QA): {self.stats.total_chunks_before_filter}\n"
            f"  Chunks (after QA) : {self.stats.total_chunks_after_filter}\n"
            f"  Dropped too short : {self.stats.dropped_too_short}\n"
            f"  Dropped low qual  : {self.stats.dropped_low_quality}\n"
            f"  Dropped duplicates: {self.stats.dropped_duplicates}\n"
            f"  Language dist     : {self.stats.language_distribution}\n"
            f"{'='*50}"
        )
        return all_chunks, self.stats

    # ─── Export helpers ────────────────────────────────────────

    def export_jsonl(self, chunks: list[ProcessedChunk], output_path: str) -> None:
        """
        Export chunks as JSONL for HuggingFace datasets.load_dataset().

        Each line:
          {"text": "...", "token_count": 412, "source": "...", ...}
        """
        import orjson
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in chunks:
                record = {
                    "text":           chunk.text,
                    "token_count":    chunk.token_count,
                    "source_file":    chunk.source_file,
                    "file_type":      chunk.file_type,
                    "chunk_id":       chunk.chunk_id,
                    "chunk_index":    chunk.chunk_index,
                    "language":       chunk.language,
                    "quality_score":  chunk.quality_score,
                    "domain":         chunk.domain,
                    "extraction_date": chunk.extraction_date,
                }
                f.write(orjson.dumps(record) + b"\n")
        logger.info(f"Exported {len(chunks)} chunks → {output_path}")

    def export_stats_json(self, stats: PipelineStats, output_path: str) -> None:
        """Export pipeline statistics for reporting / CI badge."""
        import orjson
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_files": stats.total_files,
            "total_raw_chars": stats.total_raw_chars,
            "chunks_before_filter": stats.total_chunks_before_filter,
            "chunks_after_filter": stats.total_chunks_after_filter,
            "filter_rate_pct": round(
                100 * (1 - stats.total_chunks_after_filter /
                       max(stats.total_chunks_before_filter, 1)), 2),
            "dropped_too_short": stats.dropped_too_short,
            "dropped_low_quality": stats.dropped_low_quality,
            "dropped_duplicates": stats.dropped_duplicates,
            "language_distribution": stats.language_distribution,
        }
        with open(output_path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        logger.info(f"Stats exported → {output_path}")
