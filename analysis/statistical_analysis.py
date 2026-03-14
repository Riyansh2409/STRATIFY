"""
statistical_analysis.py
=======================
Statistical analysis on preprocessed chunks + model evaluation metrics.

Tests included:
  1. Chi-Square Test (χ²)   — distribution fit, language / domain check
  2. T-Test                  — compare two model BLEU scores
  3. ANOVA                   — compare N model variants
  4. Hypothesis report       — p-value, effect size (Cohen's d), verdict

Charts (numbered, saved as PNG + embedded in HTML report):
  Fig 1.1 — Token count distribution (histogram)
  Fig 1.2 — Language distribution (bar chart)
  Fig 1.3 — Quality score distribution (KDE)
  Fig 2.1 — Chi-square contingency heatmap
  Fig 2.2 — p-value summary chart
  Fig 3.1 — BLEU / ROUGE metrics (grouped bar)
  Fig 3.2 — Training loss curve (placeholder)
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[float] = None
    effect_size: Optional[float] = None     # Cohen's d or Cramer's V
    verdict: str = ""                        # "significant" | "not significant"
    alpha: float = 0.05
    notes: str = ""

    def __post_init__(self):
        self.verdict = "SIGNIFICANT ✓" if self.p_value < self.alpha else "not significant"

    def to_dict(self) -> dict:
        return {
            "test": self.test_name,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "degrees_of_freedom": self.degrees_of_freedom,
            "effect_size": round(self.effect_size, 4) if self.effect_size else None,
            "verdict": self.verdict,
            "alpha": self.alpha,
            "notes": self.notes,
        }


@dataclass
class AnalysisReport:
    """Collects all test results and chart paths."""
    chunk_stats: dict = field(default_factory=dict)
    test_results: list[TestResult] = field(default_factory=list)
    chart_paths: dict[str, str] = field(default_factory=dict)   # fig_id → file path
    metric_table: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d effect size for two independent groups."""
    n1, n2 = len(group_a), len(group_b)
    m1, m2 = np.mean(group_a), np.mean(group_b)
    s1, s2 = np.std(group_a, ddof=1), np.std(group_b, ddof=1)
    pooled_std = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (m1 - m2) / pooled_std if pooled_std else 0.0


def _cramers_v(chi2: float, n: int, k: int, r: int) -> float:
    """Cramér's V — effect size for chi-square."""
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
    k_corr = k - (k-1)**2/(n-1)
    r_corr = r - (r-1)**2/(n-1)
    denom = min(k_corr-1, r_corr-1)
    return math.sqrt(phi2_corr / denom) if denom > 0 else 0.0


def _save_fig(fig, output_dir: str, fig_id: str) -> str:
    path = Path(output_dir) / f"{fig_id}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved chart → {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────
# Statistical Tests
# ─────────────────────────────────────────────────────────────

class StatisticalTests:

    @staticmethod
    def chi_square_goodness_of_fit(
        observed: dict[str, int],
        expected_proportions: Optional[dict[str, float]] = None,
        alpha: float = 0.05,
    ) -> TestResult:
        """
        Chi-square goodness-of-fit test.

        Use case: Is the language distribution in our dataset uniform?
                  Is the domain distribution skewed significantly?

        Args:
            observed: {"en": 420, "hi": 80, "de": 50}
            expected_proportions: None → assumes uniform distribution
        """
        categories = list(observed.keys())
        obs_values = np.array([observed[c] for c in categories], dtype=float)
        n = obs_values.sum()

        if expected_proportions:
            exp_props = np.array([expected_proportions[c] for c in categories])
            exp_values = exp_props * n
        else:
            exp_values = np.full(len(obs_values), n / len(obs_values))

        chi2, p = sp_stats.chisquare(f_obs=obs_values, f_exp=exp_values)
        df = len(categories) - 1

        # Cramér's V (1×k table → r=1)
        v = _cramers_v(chi2, int(n), len(categories), 2)

        return TestResult(
            test_name="Chi-Square Goodness-of-Fit",
            statistic=chi2,
            p_value=p,
            degrees_of_freedom=df,
            effect_size=v,
            alpha=alpha,
            notes=f"Categories: {', '.join(categories)}. N={int(n)}",
        )

    @staticmethod
    def chi_square_independence(
        contingency_table: pd.DataFrame,
        alpha: float = 0.05,
    ) -> TestResult:
        """
        Chi-square test of independence.

        Use case: Is domain distribution independent of file_type?
                  Are quality scores independent of language?

        Args:
            contingency_table: rows = group A, cols = group B
        """
        chi2, p, df, _ = sp_stats.chi2_contingency(contingency_table.values)
        n = contingency_table.values.sum()
        k, r = contingency_table.shape
        v = _cramers_v(chi2, int(n), k, r)

        return TestResult(
            test_name="Chi-Square Independence",
            statistic=chi2,
            p_value=p,
            degrees_of_freedom=df,
            effect_size=v,
            alpha=alpha,
            notes=f"Table shape: {k}×{r}. Cramér's V={v:.3f}",
        )

    @staticmethod
    def independent_ttest(
        group_a: list[float],
        group_b: list[float],
        group_a_name: str = "Model A",
        group_b_name: str = "Model B",
        alpha: float = 0.05,
    ) -> TestResult:
        """
        Independent samples t-test.

        Use case: Does fine-tuned model have significantly higher BLEU than base?

        Rule of thumb: needs ≥ 30 samples per group for reliable results.
        """
        t_stat, p = sp_stats.ttest_ind(group_a, group_b, equal_var=False)  # Welch's
        d = _cohens_d(group_a, group_b)
        df = len(group_a) + len(group_b) - 2

        return TestResult(
            test_name=f"Independent t-test ({group_a_name} vs {group_b_name})",
            statistic=t_stat,
            p_value=p,
            degrees_of_freedom=df,
            effect_size=abs(d),
            alpha=alpha,
            notes=(
                f"Mean A={np.mean(group_a):.4f}, Mean B={np.mean(group_b):.4f}. "
                f"Cohen's d={d:.3f} "
                f"({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'} effect)"
            ),
        )

    @staticmethod
    def one_way_anova(
        groups: dict[str, list[float]],
        alpha: float = 0.05,
    ) -> TestResult:
        """
        One-way ANOVA for comparing multiple model variants.

        Use case: Llama-3 vs Mistral vs Phi-3 ROUGE-L scores.
        """
        group_data = list(groups.values())
        f_stat, p = sp_stats.f_oneway(*group_data)

        # Effect size: eta-squared
        grand_mean = np.mean([x for g in group_data for x in g])
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data
        )
        ss_total = sum((x - grand_mean) ** 2 for g in group_data for x in g)
        eta_sq = ss_between / ss_total if ss_total else 0.0

        return TestResult(
            test_name=f"One-way ANOVA ({', '.join(groups.keys())})",
            statistic=f_stat,
            p_value=p,
            degrees_of_freedom=len(groups) - 1,
            effect_size=eta_sq,
            alpha=alpha,
            notes=f"η²={eta_sq:.4f} ({'small' if eta_sq<0.06 else 'medium' if eta_sq<0.14 else 'large'})",
        )


# ─────────────────────────────────────────────────────────────
# Chart Generator
# ─────────────────────────────────────────────────────────────

class ChartGenerator:
    """
    Generates numbered, labelled charts and saves them as PNG.
    Each chart method returns the file path.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Fig 1.1 ──────────────────────────────────────────────

    def fig_1_1_token_distribution(
        self,
        token_counts: list[int],
    ) -> str:
        """Fig 1.1 — Token count distribution histogram."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(token_counts, bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(token_counts), color=PALETTE[1], linestyle="--",
                   linewidth=1.5, label=f"Mean: {np.mean(token_counts):.0f}")
        ax.axvline(np.median(token_counts), color=PALETTE[2], linestyle=":",
                   linewidth=1.5, label=f"Median: {np.median(token_counts):.0f}")
        ax.set_xlabel("Token count per chunk")
        ax.set_ylabel("Frequency")
        ax.set_title("Fig 1.1 — Token Count Distribution", fontweight="bold", pad=12)
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        return _save_fig(fig, self.output_dir, "fig_1_1_token_distribution")

    # ── Fig 1.2 ──────────────────────────────────────────────

    def fig_1_2_language_distribution(
        self,
        language_counts: dict[str, int],
    ) -> str:
        """Fig 1.2 — Language distribution bar chart."""
        langs = sorted(language_counts, key=language_counts.get, reverse=True)[:12]
        counts = [language_counts[l] for l in langs]
        total = sum(counts)

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(langs, counts, color=PALETTE[0], width=0.6)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(counts)*0.01,
                    f"{count:,}\n({100*count/total:.1f}%)",
                    ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Language code (ISO 639-1)")
        ax.set_ylabel("Chunk count")
        ax.set_title("Fig 1.2 — Language Distribution", fontweight="bold", pad=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        return _save_fig(fig, self.output_dir, "fig_1_2_language_distribution")

    # ── Fig 1.3 ──────────────────────────────────────────────

    def fig_1_3_quality_scores(self, quality_scores: list[float]) -> str:
        """Fig 1.3 — Quality score KDE + threshold line."""
        from scipy.stats import gaussian_kde
        scores = np.array(quality_scores)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # If we have multiple points, attempt KDE
        if len(scores) > 1:
            try:
                xs = np.linspace(0, 1, 300)
                kde = gaussian_kde(scores, bw_method=0.1)
                ax.fill_between(xs, kde(xs), alpha=0.3, color=PALETTE[0])
                ax.plot(xs, kde(xs), color=PALETTE[0], linewidth=2)
            except Exception as e:
                logger.warning(f"KDE failed: {e}. Falling back to histogram.")
                ax.hist(scores, bins=min(20, len(scores)), alpha=0.3, color=PALETTE[0], density=True)
        else:
            # Single data point: KDE is mathematically impossible, show a vertical line
            ax.axvline(scores[0], color=PALETTE[0], linewidth=2, label="Score")
            ax.set_ylim(0, 1)

        ax.axvline(0.70, color=PALETTE[3], linestyle="--",
                   linewidth=1.5, label="Threshold: 0.70")
        ax.set_xlabel("Quality score")
        ax.set_ylabel("Density")
        ax.set_title("Fig 1.3 — Quality Score Distribution", fontweight="bold", pad=12)
        ax.legend(frameon=False)
        return _save_fig(fig, self.output_dir, "fig_1_3_quality_scores")

    # ── Fig 2.1 ──────────────────────────────────────────────

    def fig_2_1_chi_square_heatmap(
        self,
        contingency_table: pd.DataFrame,
    ) -> str:
        """Fig 2.1 — Chi-square contingency table heatmap."""
        import matplotlib.colors as mcolors
        fig, ax = plt.subplots(figsize=(max(6, len(contingency_table.columns)), 4))
        data = contingency_table.values.astype(float)
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="Count")

        ax.set_xticks(range(len(contingency_table.columns)))
        ax.set_yticks(range(len(contingency_table.index)))
        ax.set_xticklabels(contingency_table.columns, rotation=30, ha="right")
        ax.set_yticklabels(contingency_table.index)

        for i in range(len(contingency_table.index)):
            for j in range(len(contingency_table.columns)):
                ax.text(j, i, f"{int(data[i,j]):,}",
                        ha="center", va="center",
                        color="white" if data[i,j] > data.max()*0.6 else "black",
                        fontsize=9)

        ax.set_title("Fig 2.1 — Contingency Table (Chi-Square)", fontweight="bold", pad=12)
        return _save_fig(fig, self.output_dir, "fig_2_1_chi_square_heatmap")

    # ── Fig 2.2 ──────────────────────────────────────────────

    def fig_2_2_pvalue_summary(self, test_results: list[TestResult]) -> str:
        """Fig 2.2 — p-value summary lollipop chart for all tests."""
        names = [r.test_name.split("(")[0].strip() for r in test_results]
        pvals = [r.p_value for r in test_results]
        colors = [PALETTE[0] if p < 0.05 else PALETTE[3] for p in pvals]

        fig, ax = plt.subplots(figsize=(9, max(3, len(names)*0.7)))
        y_pos = range(len(names))
        ax.hlines(y_pos, 0, pvals, colors=colors, linewidth=2, alpha=0.7)
        ax.plot(pvals, y_pos, "o", color=PALETTE[0], markersize=9, zorder=5)
        ax.axvline(0.05, color=PALETTE[3], linestyle="--", linewidth=1,
                   label="α = 0.05")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("p-value")
        ax.set_title("Fig 2.2 — p-value Summary (all tests)", fontweight="bold", pad=12)
        ax.legend(frameon=False)
        ax.invert_yaxis()
        return _save_fig(fig, self.output_dir, "fig_2_2_pvalue_summary")

    # ── Fig 3.1 ──────────────────────────────────────────────

    def fig_3_1_metric_comparison(
        self,
        metrics: dict[str, dict[str, float]],
    ) -> str:
        """
        Fig 3.1 — Grouped bar chart: models × metrics.

        Args:
            metrics: {
                "Base Llama-3": {"BLEU": 0.21, "ROUGE-L": 0.34, "BERTScore": 0.78},
                "Fine-tuned":   {"BLEU": 0.38, "ROUGE-L": 0.51, "BERTScore": 0.86},
            }
        """
        model_names  = list(metrics.keys())
        metric_names = list(list(metrics.values())[0].keys())
        n_models  = len(model_names)
        n_metrics = len(metric_names)
        x = np.arange(n_metrics)
        width = 0.7 / n_models

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, (model, vals) in enumerate(metrics.items()):
            offsets = x + (i - n_models/2 + 0.5) * width
            bars = ax.bar(offsets, [vals[m] for m in metric_names],
                          width=width * 0.9, color=PALETTE[i % len(PALETTE)],
                          label=model, zorder=3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title("Fig 3.1 — Model Metric Comparison (BLEU / ROUGE / BERTScore)",
                     fontweight="bold", pad=12)
        ax.legend(frameon=False, loc="upper left")
        return _save_fig(fig, self.output_dir, "fig_3_1_metric_comparison")

    # ── Fig 3.2 ──────────────────────────────────────────────

    def fig_3_2_training_loss(
        self,
        steps: list[int],
        train_loss: list[float],
        val_loss: Optional[list[float]] = None,
    ) -> str:
        """Fig 3.2 — Training (and optional validation) loss curve."""
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps, train_loss, color=PALETTE[0], linewidth=2, label="Train loss")
        if val_loss:
            ax.plot(steps, val_loss, color=PALETTE[1], linewidth=2,
                    linestyle="--", label="Val loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Fig 3.2 — Training Loss Curve", fontweight="bold", pad=12)
        ax.legend(frameon=False)
        return _save_fig(fig, self.output_dir, "fig_3_2_training_loss")

    # ── Fig 3.3 ──────────────────────────────────────────────

    def fig_3_3_domain_distribution(
        self, domain_counts: dict[str, int]
    ) -> str:
        """Fig 3.3 — Domain distribution pie chart."""
        labels = list(domain_counts.keys())
        sizes  = list(domain_counts.values())
        other  = sum(sizes) - sum(sorted(sizes, reverse=True)[:6])
        top_labels = [l for l, s in sorted(zip(labels, sizes),
                       key=lambda x: x[1], reverse=True)][:6]
        top_sizes  = sorted(sizes, reverse=True)[:6]
        if other > 0:
            top_labels.append("other")
            top_sizes.append(other)

        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(
            top_sizes, labels=top_labels,
            colors=PALETTE[:len(top_labels)],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title("Fig 3.3 — Domain Distribution", fontweight="bold", pad=12)
        return _save_fig(fig, self.output_dir, "fig_3_3_domain_distribution")


# ─────────────────────────────────────────────────────────────
# High-level Analyser
# ─────────────────────────────────────────────────────────────

class DatasetAnalyser:
    """
    Run all statistical tests + generate all charts from processed chunks.

    Usage:
        from preprocessor import ProcessedChunk
        analyser = DatasetAnalyser(output_dir="reports/")
        report = analyser.analyse(chunks, model_metrics={"Base": {...}, "FT": {...}})
        analyser.save_report_json(report, "reports/analysis_report.json")
    """

    def __init__(self, output_dir: str = "reports"):
        self.tests  = StatisticalTests()
        self.charts = ChartGenerator(output_dir=str(Path(output_dir) / "figures"))
        self.output_dir = output_dir

    def analyse(
        self,
        chunks: list,                         # list[ProcessedChunk]
        model_metrics: Optional[dict] = None, # {model_name: {metric: score}}
        train_loss_data: Optional[dict] = None,
    ) -> AnalysisReport:
        report = AnalysisReport()

        # ── Collect raw values ─────────────────────────────
        token_counts    = [c.token_count    for c in chunks]
        quality_scores  = [c.quality_score  for c in chunks]
        languages       = {}
        domains         = {}
        file_types      = {}
        for c in chunks:
            languages[c.language]         = languages.get(c.language, 0) + 1
            if c.domain:
                domains[c.domain]         = domains.get(c.domain, 0) + 1
            file_types[c.file_type]       = file_types.get(c.file_type, 0) + 1

        # ── Summary stats ──────────────────────────────────
        report.chunk_stats = {
            "total_chunks": len(chunks),
            "mean_token_count": round(float(np.mean(token_counts)), 2),
            "median_token_count": float(np.median(token_counts)),
            "std_token_count": round(float(np.std(token_counts)), 2),
            "mean_quality_score": round(float(np.mean(quality_scores)), 4),
            "language_distribution": languages,
            "domain_distribution": domains,
            "file_type_distribution": file_types,
        }
        logger.info(f"Chunk stats: {report.chunk_stats}")

        # ── Statistical tests ──────────────────────────────

        # Test 1: Is language distribution uniform?
        if len(languages) >= 2:
            t1 = self.tests.chi_square_goodness_of_fit(languages)
            report.test_results.append(t1)
            logger.info(f"[Chi-Square GoF] {t1.verdict} | p={t1.p_value:.4f}")

        # Test 2: Independence of domain × file_type
        if domains and len(file_types) >= 2:
            ct_data = {}
            for c in chunks:
                d = c.domain or "unknown"
                ft = c.file_type
                ct_data.setdefault(d, {}).setdefault(ft, 0)
                ct_data[d][ft] += 1
            ct_df = pd.DataFrame(ct_data).T.fillna(0)
            if ct_df.shape[0] >= 2 and ct_df.shape[1] >= 2:
                t2 = self.tests.chi_square_independence(ct_df)
                report.test_results.append(t2)
                logger.info(f"[Chi-Square Ind] {t2.verdict} | p={t2.p_value:.4f}")

        # Test 3 & 4: If model metrics provided → t-test and ANOVA
        if model_metrics and len(model_metrics) >= 2:
            bleu_scores = {k: v.get("bleu_samples", [v.get("BLEU", 0)])
                           for k, v in model_metrics.items()}
            names = list(bleu_scores.keys())

            # t-test: first two models
            if all(isinstance(bleu_scores[names[i]], list)
                   and len(bleu_scores[names[i]]) > 1 for i in [0,1]):
                t3 = self.tests.independent_ttest(
                    bleu_scores[names[0]], bleu_scores[names[1]],
                    names[0], names[1]
                )
                report.test_results.append(t3)
                logger.info(f"[t-test] {t3.verdict} | p={t3.p_value:.4f} | d={t3.effect_size:.3f}")

            # ANOVA: all models
            if (len(names) >= 3 and
                    all(isinstance(bleu_scores[n], list) for n in names)):
                t4 = self.tests.one_way_anova(bleu_scores)
                report.test_results.append(t4)
                logger.info(f"[ANOVA] {t4.verdict} | p={t4.p_value:.4f}")

        # ── Generate charts ────────────────────────────────

        if token_counts:
            report.chart_paths["fig_1_1"] = \
                self.charts.fig_1_1_token_distribution(token_counts)
        if languages:
            report.chart_paths["fig_1_2"] = \
                self.charts.fig_1_2_language_distribution(languages)
        if quality_scores:
            report.chart_paths["fig_1_3"] = \
                self.charts.fig_1_3_quality_scores(quality_scores)

        # Chi-square heatmap (domain × file_type)
        if domains and len(file_types) >= 2 and "ct_df" in dir():
            report.chart_paths["fig_2_1"] = \
                self.charts.fig_2_1_chi_square_heatmap(ct_df)

        if report.test_results:
            report.chart_paths["fig_2_2"] = \
                self.charts.fig_2_2_pvalue_summary(report.test_results)

        if model_metrics:
            # Build metric dict for chart (use scalar values)
            chart_metrics = {
                model: {k: v for k, v in vals.items()
                        if isinstance(v, (int, float)) and k != "bleu_samples"}
                for model, vals in model_metrics.items()
            }
            if chart_metrics:
                report.chart_paths["fig_3_1"] = \
                    self.charts.fig_3_1_metric_comparison(chart_metrics)

            if domains:
                report.chart_paths["fig_3_3"] = \
                    self.charts.fig_3_3_domain_distribution(domains)

        if train_loss_data:
            report.chart_paths["fig_3_2"] = self.charts.fig_3_2_training_loss(
                train_loss_data["steps"],
                train_loss_data["train_loss"],
                train_loss_data.get("val_loss"),
            )

        report.metric_table = {
            k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float))}
            for k, v in (model_metrics or {}).items()
        }

        return report

    def save_report_json(self, report: AnalysisReport, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_stats": report.chunk_stats,
            "test_results": [t.to_dict() for t in report.test_results],
            "chart_paths": report.chart_paths,
            "metric_table": report.metric_table,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Analysis report saved → {path}")
