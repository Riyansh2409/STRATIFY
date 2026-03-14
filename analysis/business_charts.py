"""
business_charts.py
==================
Auto-generates business analysis charts from uploaded CSV / Excel files.

Charts produced (when data supports them):
  BC-1  — KPI Summary Card (key numeric column statistics)
  BC-2  — Revenue / Numeric Trend (line chart over a datetime column)
  BC-3  — Category Breakdown (top categories by value — bar chart)
  BC-4  — Market Share Pie (proportion of a categorical column)
  BC-5  — Correlation Heatmap (numeric columns)
  BC-6  — Top-N Ranked Bar (sorted by highest numeric value)
  BC-7  — Segment Comparison (grouped bar per category pair)

The module inspects the dataframe automatically — no column names needed.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Premium Business Palette ──────────────────────────────────────────────────
BIZ_PALETTE   = ["#2C6BED", "#22C0C0", "#F4A823", "#EF4C5A", "#6C63FF", "#27AE60"]
BG_COLOR      = "#F8F9FC"
CARD_COLOR    = "#FFFFFF"
ACCENT_DARK   = "#1A2D5A"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "figure.dpi":        150,
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    CARD_COLOR,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, output_dir: str, name: str) -> str:
    path = Path(output_dir) / f"{name}.png"
    fig.savefig(str(path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved business chart → {path}")
    return str(path)


def _smart_format(val: float) -> str:
    """Format large numbers as 1.2M, 45K etc."""
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.1f}K"
    return f"{val:.2f}"


def _detect_columns(df: pd.DataFrame):
    """Return best numeric and categorical columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Filter out id-like columns
    cat_cols = [c for c in cat_cols if df[c].nunique() < 50]
    # Try to detect a date column
    date_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        if any(kw in col.lower() for kw in ["date", "month", "year", "time", "period", "week"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().sum() > 3:
                    date_col = col
                    break
            except Exception:
                pass
    return num_cols, cat_cols, date_col


# ── Chart Functions ───────────────────────────────────────────────────────────

def bc1_kpi_summary(df: pd.DataFrame, num_cols: list, output_dir: str) -> Optional[str]:
    """BC-1 — KPI Summary Card: mean/max/min of top 4 numeric columns."""
    cols = num_cols[:4]
    if not cols:
        return None

    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3))
    if len(cols) == 1:
        axes = [axes]

    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("BC-1  KPI Summary", fontsize=14, fontweight="bold", color=ACCENT_DARK, y=1.02)

    for ax, col in zip(axes, cols):
        series = df[col].dropna()
        mean_v = series.mean()
        max_v  = series.max()
        ax.set_facecolor(CARD_COLOR)
        ax.axis("off")
        ax.text(0.5, 0.75, col.replace("_", " ").title(),
                ha="center", va="center", fontsize=10, color="#666", transform=ax.transAxes)
        ax.text(0.5, 0.45, _smart_format(mean_v),
                ha="center", va="center", fontsize=22, fontweight="bold",
                color=BIZ_PALETTE[0], transform=ax.transAxes)
        ax.text(0.5, 0.15, f"Max: {_smart_format(max_v)}",
                ha="center", va="center", fontsize=10, color="#999", transform=ax.transAxes)
        ax.add_patch(plt.Rectangle((0.05, 0.02), 0.9, 0.96, linewidth=1.5,
                                   edgecolor=BIZ_PALETTE[0], facecolor="none",
                                   transform=ax.transAxes, clip_on=False))

    plt.tight_layout()
    return _save(fig, output_dir, "bc1_kpi_summary")


def bc2_trend_line(df: pd.DataFrame, date_col: str, num_cols: list, output_dir: str) -> Optional[str]:
    """BC-2 — Trend line over time for top numeric columns."""
    if not date_col or not num_cols:
        return None

    cols = [c for c in num_cols[:3] if c != date_col]
    if not cols:
        return None

    df_sorted = df.sort_values(date_col)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("BC-2  Trends Over Time", fontsize=14, fontweight="bold", color=ACCENT_DARK)

    for i, col in enumerate(cols):
        ax.plot(df_sorted[date_col], df_sorted[col], color=BIZ_PALETTE[i],
                linewidth=2.2, label=col.replace("_", " ").title(), marker="o",
                markersize=3, alpha=0.85)
        ax.fill_between(df_sorted[date_col], df_sorted[col], alpha=0.08, color=BIZ_PALETTE[i])

    ax.set_xlabel("Period")
    ax.set_ylabel("Value")
    ax.legend(frameon=False)
    plt.xticks(rotation=30)
    plt.tight_layout()
    return _save(fig, output_dir, "bc2_trend_line")


def bc3_category_bar(df: pd.DataFrame, cat_col: str, num_col: str, output_dir: str) -> Optional[str]:
    """BC-3 — Top categories by total value (bar chart)."""
    if not cat_col or not num_col:
        return None

    grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(12)
    if grouped.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle(f"BC-3  {num_col.replace('_',' ').title()} by {cat_col.replace('_',' ').title()}",
                 fontsize=13, fontweight="bold", color=ACCENT_DARK)

    bars = ax.bar(grouped.index.astype(str), grouped.values,
                  color=BIZ_PALETTE[0], width=0.6, alpha=0.9, zorder=3)
    # Color gradient
    n = len(bars)
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.Blues(0.4 + 0.55 * (1 - i / max(n-1, 1))))

    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + grouped.max()*0.01,
                _smart_format(val), ha="center", va="bottom", fontsize=9, color=ACCENT_DARK)

    ax.set_xlabel(cat_col.replace("_", " ").title())
    ax.set_ylabel(num_col.replace("_", " ").title())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _smart_format(x)))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return _save(fig, output_dir, "bc3_category_bar")


def bc4_market_share_pie(df: pd.DataFrame, cat_col: str, output_dir: str) -> Optional[str]:
    """BC-4 — Market share donut (pie) chart by category."""
    if not cat_col:
        return None

    counts = df[cat_col].value_counts().head(7)
    if counts.empty:
        return None

    labels = counts.index.astype(str).tolist()
    sizes  = counts.values.tolist()

    # Aggregate tail into "Other"
    if len(df[cat_col].unique()) > 7:
        other = df[cat_col].value_counts().iloc[7:].sum()
        labels.append("Other")
        sizes.append(other)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f"BC-4  {cat_col.replace('_',' ').title()} Market Share",
                 fontsize=13, fontweight="bold", color=ACCENT_DARK)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels,
        colors=BIZ_PALETTE[:len(labels)],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color(ACCENT_DARK)

    # Donut hole
    centre_circle = plt.Circle((0, 0), 0.55, fc=BG_COLOR)
    ax.add_artist(centre_circle)

    plt.tight_layout()
    return _save(fig, output_dir, "bc4_market_share_pie")


def bc5_correlation_heatmap(df: pd.DataFrame, num_cols: list, output_dir: str) -> Optional[str]:
    """BC-5 — Correlation heatmap between numeric columns."""
    cols = [c for c in num_cols if df[c].nunique() > 2][:8]
    if len(cols) < 2:
        return None

    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(max(5, len(cols)), max(4, len(cols) - 1)))
    fig.suptitle("BC-5  Correlation Matrix", fontsize=13, fontweight="bold", color=ACCENT_DARK)

    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    labels = [c.replace("_", " ").title() for c in cols]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.6 else "black", fontsize=8)

    plt.tight_layout()
    return _save(fig, output_dir, "bc5_correlation_heatmap")


def bc6_top_ranked_bar(df: pd.DataFrame, cat_col: str, num_col: str, output_dir: str) -> Optional[str]:
    """BC-6 — Horizontal ranked bar chart (Top 10 performers)."""
    if not cat_col or not num_col:
        return None

    ranked = df.groupby(cat_col)[num_col].sum().sort_values(ascending=True).tail(10)
    if ranked.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(ranked) * 0.55)))
    fig.suptitle(f"BC-6  Top 10 — {num_col.replace('_',' ').title()} by {cat_col.replace('_',' ').title()}",
                 fontsize=13, fontweight="bold", color=ACCENT_DARK)

    colors = [BIZ_PALETTE[0]] * len(ranked)
    colors[-1] = "#F4A823"   # Highlight the #1 entry in gold

    bars = ax.barh(ranked.index.astype(str), ranked.values, color=colors, height=0.65, zorder=3)

    for bar, val in zip(bars, ranked.values):
        ax.text(val + ranked.max() * 0.01, bar.get_y() + bar.get_height()/2,
                _smart_format(val), va="center", fontsize=10, color=ACCENT_DARK)

    ax.set_xlabel(num_col.replace("_", " ").title())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _smart_format(x)))
    plt.tight_layout()
    return _save(fig, output_dir, "bc6_top_ranked")


def bc7_segment_comparison(df: pd.DataFrame, cat_col1: str, cat_col2: str,
                            num_col: str, output_dir: str) -> Optional[str]:
    """BC-7 — Grouped bar: segment vs sub-segment."""
    if not cat_col1 or not cat_col2 or not num_col:
        return None

    pivot = df.groupby([cat_col1, cat_col2])[num_col].sum().unstack(fill_value=0)
    if pivot.empty or pivot.shape[1] > 8:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle(f"BC-7  {num_col.replace('_',' ').title()} — {cat_col1} × {cat_col2}",
                 fontsize=13, fontweight="bold", color=ACCENT_DARK)

    n_groups  = len(pivot.index)
    n_bars    = len(pivot.columns)
    x         = np.arange(n_groups)
    width     = 0.7 / n_bars

    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - n_bars/2 + 0.5) * width, pivot[col],
               width=width * 0.9, color=BIZ_PALETTE[i % len(BIZ_PALETTE)],
               label=str(col), zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str), rotation=25, ha="right")
    ax.set_ylabel(num_col.replace("_", " ").title())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _smart_format(v)))
    ax.legend(title=cat_col2.replace("_", " ").title(), frameon=False, fontsize=9)
    plt.tight_layout()
    return _save(fig, output_dir, "bc7_segment_comparison")


# ── Master Runner ─────────────────────────────────────────────────────────────

def generate_business_charts(upload_dir: str, output_dir: str) -> dict:
    """
    Scans upload_dir for CSV/Excel files, reads them, and produces
    business analysis charts in output_dir. Returns {chart_id: path} dict.
    """
    chart_paths = {}
    upload_path = Path(upload_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all CSV / Excel files
    frames = []
    for pattern in ["*.csv", "*.xlsx", "*.xls"]:
        for fpath in upload_path.glob(pattern):
            try:
                df = pd.read_csv(str(fpath)) if fpath.suffix == ".csv" else pd.read_excel(str(fpath))
                frames.append(df)
                logger.info(f"Loaded business file: {fpath.name} ({len(df)} rows)")
            except Exception as e:
                logger.warning(f"Could not load {fpath.name}: {e}")

    if not frames:
        logger.info("No CSV/Excel files found — skipping business charts.")
        return chart_paths

    # Merge all frames into one
    df = pd.concat(frames, ignore_index=True)
    num_cols, cat_cols, date_col = _detect_columns(df)

    logger.info(f"Business data: {len(df)} rows | Numeric={num_cols} | Categorical={cat_cols}")

    # BC-1: KPI Summary
    if num_cols:
        path = bc1_kpi_summary(df, num_cols, str(output_path))
        if path: chart_paths["BC-1 KPI Summary"] = path

    # BC-2: Trend over time
    if date_col and num_cols:
        path = bc2_trend_line(df, date_col, num_cols, str(output_path))
        if path: chart_paths["BC-2 Trend Over Time"] = path

    # BC-3: Category total bar
    if cat_cols and num_cols:
        path = bc3_category_bar(df, cat_cols[0], num_cols[0], str(output_path))
        if path: chart_paths["BC-3 Category Breakdown"] = path

    # BC-4: Market share pie
    if cat_cols:
        path = bc4_market_share_pie(df, cat_cols[0], str(output_path))
        if path: chart_paths["BC-4 Market Share"] = path

    # BC-5: Correlation heatmap
    if len(num_cols) >= 2:
        path = bc5_correlation_heatmap(df, num_cols, str(output_path))
        if path: chart_paths["BC-5 Correlation Matrix"] = path

    # BC-6: Top-N ranked
    if cat_cols and num_cols:
        path = bc6_top_ranked_bar(df, cat_cols[0], num_cols[0], str(output_path))
        if path: chart_paths["BC-6 Top Ranked"] = path

    # BC-7: Segment comparison
    if len(cat_cols) >= 2 and num_cols:
        path = bc7_segment_comparison(df, cat_cols[0], cat_cols[1], num_cols[0], str(output_path))
        if path: chart_paths["BC-7 Segment Comparison"] = path

    return chart_paths
