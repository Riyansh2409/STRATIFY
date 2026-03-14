"""
tests/test_preprocessing.py
============================
Unit tests for:
  - TextCleaner
  - TokenChunker (with tiktoken fallback)
  - QualityFilter
  - FileLoader (CSV + JSON)
  - StatisticalTests (chi-square, t-test, ANOVA)

Run: pytest tests/ -v
"""

import sys
import json
import tempfile
import csv
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from preprocessor import TextCleaner, QualityFilter
from statistical_analysis import StatisticalTests


# ════════════════════════════════════════════════════════════
# TextCleaner Tests
# ════════════════════════════════════════════════════════════

class TestTextCleaner:
    cleaner = TextCleaner()

    def test_removes_html_tags(self):
        raw = "<h1>Hello</h1> <p>World</p>"
        cleaned = self.cleaner.clean(raw)
        assert "<h1>" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_replaces_urls(self):
        raw = "Visit https://huggingface.co/meta-llama for docs"
        cleaned = self.cleaner.clean(raw)
        assert "https://" not in cleaned
        assert "[URL]" in cleaned

    def test_replaces_emails(self):
        raw = "Contact us at support@example.com for help"
        cleaned = self.cleaner.clean(raw)
        assert "@example.com" not in cleaned
        assert "[EMAIL]" in cleaned

    def test_collapses_blank_lines(self):
        raw = "Line one\n\n\n\n\nLine two"
        cleaned = self.cleaner.clean(raw)
        assert "\n\n\n" not in cleaned

    def test_unicode_normalisation(self):
        # Full-width chars → ASCII
        raw = "ｈｅｌｌｏ ｗｏｒｌｄ"
        cleaned = self.cleaner.clean(raw)
        assert "hello" in cleaned.lower()

    def test_empty_string_returns_empty(self):
        assert self.cleaner.clean("") == ""

    def test_strips_control_characters(self):
        raw = "Hello\x00World\x08Test"
        cleaned = self.cleaner.clean(raw)
        assert "\x00" not in cleaned
        assert "Hello" in cleaned


# ════════════════════════════════════════════════════════════
# QualityFilter Tests
# ════════════════════════════════════════════════════════════

class TestQualityFilter:
    qa = QualityFilter(threshold=0.70)

    GOOD_TEXT = (
        "The transformer architecture has revolutionised natural language processing. "
        "Self-attention mechanisms allow models to weigh the importance of different "
        "words in a sequence, enabling better contextual understanding. Fine-tuning "
        "on domain-specific data further improves downstream task performance."
    )

    BAD_REPETITIVE = "word " * 80
    BAD_SHORT = "Hi there"

    def test_good_text_passes(self):
        token_count = len(self.GOOD_TEXT.split()) * 2  # rough estimate
        passes, score, reason = self.qa.passes(self.GOOD_TEXT, token_count, "chunk_good")
        assert passes, f"Good text should pass QA (score={score}, reason={reason})"

    def test_repetitive_text_fails(self):
        passes, score, _ = self.qa.passes(self.BAD_REPETITIVE, 160, "chunk_rep")
        assert not passes, "Highly repetitive text should fail QA"

    def test_short_text_fails(self):
        passes, score, reason = self.qa.passes(self.BAD_SHORT, 5, "chunk_short")
        assert not passes
        assert reason == "too_short"

    def test_exact_duplicate_detected(self):
        qa = QualityFilter(threshold=0.50)
        qa.passes(self.GOOD_TEXT, 100, "dup_a")     # first call — registers hash
        passes, _, reason = qa.passes(self.GOOD_TEXT, 100, "dup_b")  # second call
        assert not passes
        assert reason == "duplicate"

    def test_quality_score_between_0_and_1(self):
        score = self.qa.quality_score(self.GOOD_TEXT, 100)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ════════════════════════════════════════════════════════════
# FileLoader Tests (CSV + JSON, no heavy deps)
# ════════════════════════════════════════════════════════════

class TestFileLoader:

    def _write_csv(self, rows: list[dict]) -> str:
        tf = tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", newline="", delete=False, encoding="utf-8"
        )
        writer = csv.DictWriter(tf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        tf.close()
        return tf.name

    def _write_json(self, data: dict | list) -> str:
        tf = tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        )
        json.dump(data, tf)
        tf.close()
        return tf.name

    def test_csv_loads_rows(self):
        from file_loader import load_file
        path = self._write_csv([
            {"name": "Alice", "score": "0.9"},
            {"name": "Bob",   "score": "0.8"},
        ])
        doc = load_file(path)
        assert doc.file_type == "csv"
        assert "Alice" in doc.raw_text
        assert doc.load_error is None

    def test_json_loads_flat(self):
        from file_loader import load_file
        path = self._write_json({"model": "llama3", "version": "8B", "layers": 32})
        doc = load_file(path)
        assert doc.file_type == "json"
        assert "llama3" in doc.raw_text
        assert doc.load_error is None

    def test_json_nested_flatten(self):
        from file_loader import load_file
        data = {"model": {"name": "llama3", "params": {"layers": 32, "heads": 32}}}
        path = self._write_json(data)
        doc = load_file(path)
        assert "llama3" in doc.raw_text

    def test_missing_file_raises(self):
        from file_loader import load_file
        with pytest.raises(FileNotFoundError):
            load_file("/nonexistent/path/file.csv")


# ════════════════════════════════════════════════════════════
# StatisticalTests
# ════════════════════════════════════════════════════════════

class TestStatisticalTests:
    tests = StatisticalTests()

    def test_chi_square_goodness_of_fit_uniform(self):
        # Exactly uniform → p should be 1.0 (not significant)
        observed = {"en": 100, "hi": 100, "de": 100}
        result = self.tests.chi_square_goodness_of_fit(observed)
        assert result.p_value > 0.05, "Uniform distribution should NOT be significant"
        assert result.statistic == pytest.approx(0.0, abs=1e-6)

    def test_chi_square_goodness_of_fit_skewed(self):
        # Extremely skewed → should be significant
        observed = {"en": 980, "hi": 10, "de": 10}
        result = self.tests.chi_square_goodness_of_fit(observed)
        assert result.p_value < 0.05, "Skewed distribution should be significant"
        assert "SIGNIFICANT" in result.verdict

    def test_chi_square_independence(self):
        ct = pd.DataFrame({
            "pdf": {"technology": 50, "finance": 20},
            "csv": {"technology": 10, "finance": 60},
        })
        result = self.tests.chi_square_independence(ct)
        assert result.statistic > 0
        assert 0.0 <= result.p_value <= 1.0
        assert result.effect_size is not None

    def test_ttest_significant(self):
        np.random.seed(42)
        group_a = (np.random.normal(0.40, 0.05, 60)).tolist()   # fine-tuned (higher BLEU)
        group_b = (np.random.normal(0.21, 0.05, 60)).tolist()   # base model
        result = self.tests.independent_ttest(group_a, group_b, "FT", "Base")
        assert result.p_value < 0.001, "Large mean difference should be highly significant"
        assert result.effect_size > 0.8, "Expected large effect size (Cohen's d)"

    def test_ttest_not_significant(self):
        np.random.seed(0)
        group_a = np.random.normal(0.35, 0.10, 50).tolist()
        group_b = np.random.normal(0.36, 0.10, 50).tolist()
        result = self.tests.independent_ttest(group_a, group_b)
        assert result.p_value > 0.05, "Similar distributions should not be significant"

    def test_anova_significant(self):
        np.random.seed(7)
        groups = {
            "Llama-3": np.random.normal(0.38, 0.04, 40).tolist(),
            "Mistral":  np.random.normal(0.30, 0.04, 40).tolist(),
            "Phi-3":    np.random.normal(0.24, 0.04, 40).tolist(),
        }
        result = self.tests.one_way_anova(groups)
        assert result.p_value < 0.001
        assert "SIGNIFICANT" in result.verdict
        assert result.effect_size > 0.06   # at least medium η²

    def test_test_result_to_dict(self):
        observed = {"a": 80, "b": 20}
        result = self.tests.chi_square_goodness_of_fit(observed)
        d = result.to_dict()
        assert "test" in d
        assert "p_value" in d
        assert "verdict" in d
        assert isinstance(d["p_value"], float)
