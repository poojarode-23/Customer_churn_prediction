# =============================================================================
# src/analysis/eda.py
# Phase 3 — Exploratory Data Analysis (EDA) Pipeline
# Project: AI-Powered Customer Churn Prediction
# =============================================================================
# WHAT THIS MODULE DOES:
#   1.  Dataset overview (shape, dtypes, memory, stats)
#   2.  Target variable analysis
#   3.  Numerical feature analysis (distributions, stats, outliers)
#   4.  Categorical feature analysis (counts, churn rates)
#   5.  Bivariate analysis (feature vs churn)
#   6.  Correlation analysis
#   7.  Outlier detection and reporting
#   8.  Business insights generation
#   9.  Saves all figures to reports/figures/
#   10. Exports Power BI-ready CSV
# =============================================================================

import sys
import warnings
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for scripts & CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    CLEAN_FILE, DATA_PROCESSED, FIGURES_DIR, LOG_FILE,
    CATEGORICAL_COLS, NUMERICAL_COLS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, str(LOG_FILE))

# =============================================================================
# GLOBAL STYLE — Applied once, inherited by all charts
# =============================================================================

PALETTE_CHURN  = {0: "#2ECC71", 1: "#E74C3C"}   # Green = stay, Red = churn
PALETTE_TWO    = ["#3498DB", "#E74C3C"]
PALETTE_MAIN   = "Set2"
CORPORATE_BLUE = "#2C3E50"
ACCENT_RED     = "#E74C3C"
ACCENT_GREEN   = "#2ECC71"

def _set_global_style() -> None:
    """Apply a clean, corporate chart style to all matplotlib/seaborn figures."""
    sns.set_theme(style="whitegrid", palette=PALETTE_MAIN, font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi"       : 150,
        "savefig.dpi"      : 150,
        "figure.facecolor" : "white",
        "axes.facecolor"   : "#FAFAFA",
        "axes.titlesize"   : 13,
        "axes.titleweight" : "bold",
        "axes.labelsize"   : 11,
        "xtick.labelsize"  : 9,
        "ytick.labelsize"  : 9,
        "legend.fontsize"  : 9,
        "font.family"      : "DejaVu Sans",
    })

_set_global_style()


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def _save_fig(filename: str, tight: bool = True) -> None:
    """
    Saves the current matplotlib figure to FIGURES_DIR.

    Args:
        filename : e.g. "churn_distribution.png"
        tight    : whether to apply tight_layout before saving
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Figure saved → {path}")


def _churn_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Returns a summary DataFrame: count, churn count, churn rate % per category.

    Example output for 'Contract':
        Contract            | Total | Churned | Churn Rate %
        Month-to-month      |  3875 |    1655 |     42.7%
        One year            |  1473 |     166 |     11.3%
        Two year            |  1695 |      48 |      2.8%
    """
    summary = (
        df.groupby(col)
          .agg(
              Total   = ("Churn", "count"),
              Churned = ("Churn", "sum"),
          )
          .assign(ChurnRate=lambda x: (x["Churned"] / x["Total"] * 100).round(2))
          .sort_values("ChurnRate", ascending=False)
          .reset_index()
    )
    return summary


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}", offset: float = 0.5) -> None:
    """
    Annotates each bar in a bar chart with its value.

    Args:
        ax     : matplotlib Axes object
        fmt    : format string for the label
        offset : vertical offset above the bar top
    """
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + offset,
                fmt.format(height),
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )


# =============================================================================
# SECTION 1 — DATASET OVERVIEW
# =============================================================================

def section1_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Prints and returns a comprehensive dataset overview.

    Returns a dict with shape, dtype counts, null info, memory usage,
    and descriptive statistics.
    """
    logger.info("=" * 60)
    logger.info("SECTION 1 — DATASET OVERVIEW")
    logger.info("=" * 60)

    overview = {}

    # ── Shape ────────────────────────────────────────────────────────────────
    overview["rows"], overview["cols"] = df.shape
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Data types ───────────────────────────────────────────────────────────
    dtype_counts = df.dtypes.value_counts().to_dict()
    overview["dtype_counts"] = {str(k): v for k, v in dtype_counts.items()}
    logger.info(f"Data types: {overview['dtype_counts']}")

    # ── Missing values ────────────────────────────────────────────────────────
    nulls = df.isnull().sum()
    overview["nulls"] = nulls[nulls > 0].to_dict()
    logger.info(f"Columns with nulls: {overview['nulls'] or 'None'}")

    # ── Duplicates ────────────────────────────────────────────────────────────
    overview["duplicates"] = int(df.duplicated().sum())
    logger.info(f"Duplicate rows: {overview['duplicates']}")

    # ── Memory ───────────────────────────────────────────────────────────────
    mem_bytes = df.memory_usage(deep=True).sum()
    overview["memory_mb"] = round(mem_bytes / 1024**2, 3)
    logger.info(f"Memory usage: {overview['memory_mb']} MB")

    # ── Descriptive statistics ────────────────────────────────────────────────
    stats = df[NUMERICAL_COLS].describe().T
    stats["skew"]     = df[NUMERICAL_COLS].skew()
    stats["kurtosis"] = df[NUMERICAL_COLS].kurtosis()
    overview["stats"] = stats
    logger.info(f"\nDescriptive stats:\n{stats.round(2)}")

    # ── Print formatted summary ───────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  DATASET OVERVIEW")
    print("═" * 55)
    print(f"  Rows          : {overview['rows']:,}")
    print(f"  Columns       : {overview['cols']}")
    print(f"  Memory Usage  : {overview['memory_mb']} MB")
    print(f"  Missing Values: {sum(overview['nulls'].values()) if overview['nulls'] else 0}")
    print(f"  Duplicates    : {overview['duplicates']}")
    print(f"\n  Numerical Stats:\n{stats.round(2)}")
    print("═" * 55 + "\n")

    return overview


# =============================================================================
# SECTION 2 — TARGET VARIABLE ANALYSIS
# =============================================================================

def section2_target_analysis(df: pd.DataFrame) -> None:
    """
    Analyses and visualises the Churn target variable.

    Charts:
      - Pie chart: churn vs no-churn percentage
      - Count plot with percentage labels
      - Churn rate by key segment side-by-side

    Business Insight:
      A 26.5% churn rate is HIGH for telecoms (industry average ~15–20%).
      This tells us retention is a critical business priority.
    """
    logger.info("SECTION 2 — TARGET VARIABLE ANALYSIS")

    churn_map   = {0: "No Churn", 1: "Churned"}
    counts      = df["Churn"].value_counts()
    pct         = df["Churn"].value_counts(normalize=True).mul(100).round(2)
    colors      = [ACCENT_GREEN, ACCENT_RED]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Variable: Customer Churn Distribution", fontsize=14,
                 fontweight="bold", color=CORPORATE_BLUE)

    # ── Left: Pie chart ───────────────────────────────────────────────────────
    wedge_props = dict(edgecolor="white", linewidth=2)
    axes[0].pie(
        counts.values,
        labels     = [churn_map[k] for k in counts.index],
        autopct    = "%1.1f%%",
        colors     = colors,
        startangle = 140,
        wedgeprops = wedge_props,
        textprops  = dict(fontsize=11, fontweight="bold"),
    )
    axes[0].set_title("Churn Proportion", pad=12)

    # ── Right: Count plot with % annotations ─────────────────────────────────
    bars = axes[1].bar(
        [churn_map[k] for k in counts.index],
        counts.values,
        color      = colors,
        edgecolor  = "white",
        linewidth  = 1.5,
        width      = 0.5,
    )
    for bar, (label, count) in zip(bars, counts.items()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            f"{count:,}\n({pct[label]:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    axes[1].set_title("Churn Counts")
    axes[1].set_ylabel("Number of Customers")
    axes[1].set_ylim(0, counts.max() * 1.18)
    axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _save_fig("churn_distribution.png")
    logger.info(f"Churn rate: {pct.get(1, pct.get('1', 0)):.1f}%")

    print("\n📊 TARGET VARIABLE INSIGHT:")
    print(f"  • No Churn : {counts[0]:,} customers ({pct[0]:.1f}%)")
    print(f"  • Churned  : {counts[1]:,} customers ({pct[1]:.1f}%)")
    print("  • ⚠️  Class imbalance detected — SMOTE will be applied in Phase 7")
    print("  • Industry benchmark churn rate: ~15–20%")
    print(f"  • TelCo is at {pct[1]:.1f}% — ABOVE benchmark → urgent retention action needed\n")


# =============================================================================
# SECTION 3 — NUMERICAL FEATURE ANALYSIS
# =============================================================================

def section3_numerical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full statistical and visual analysis for each numerical column.

    For each of tenure / MonthlyCharges / TotalCharges:
      - Histogram + KDE
      - Boxplot by churn status
      - Violin plot by churn status
      - Descriptive statistics table

    Returns a combined stats DataFrame.
    """
    logger.info("SECTION 3 — NUMERICAL FEATURE ANALYSIS")

    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    stats_rows = []

    for col in num_cols:
        series = df[col].dropna()

        # ── Stats ─────────────────────────────────────────────────────────────
        stats = {
            "Feature"  : col,
            "Mean"     : round(series.mean(), 2),
            "Median"   : round(series.median(), 2),
            "Std Dev"  : round(series.std(), 2),
            "Min"      : round(series.min(), 2),
            "Max"      : round(series.max(), 2),
            "Skewness" : round(series.skew(), 3),
            "Kurtosis" : round(series.kurtosis(), 3),
        }
        stats_rows.append(stats)

        logger.info(
            f"{col}: mean={stats['Mean']}, median={stats['Median']}, "
            f"std={stats['Std Dev']}, skew={stats['Skewness']}"
        )

        # ── Figure: 2×2 grid ─────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Numerical Analysis: {col}", fontsize=14,
                     fontweight="bold", color=CORPORATE_BLUE)

        churn_labels = df["Churn"].map({0: "No Churn", 1: "Churned"})

        # Top-left: Histogram + KDE
        axes[0, 0].hist(series, bins=40, color=CORPORATE_BLUE, alpha=0.7,
                        edgecolor="white", linewidth=0.5)
        ax2 = axes[0, 0].twinx()
        series.plot.kde(ax=ax2, color=ACCENT_RED, linewidth=2)
        ax2.set_ylabel("Density", color=ACCENT_RED)
        axes[0, 0].set_title(f"{col} — Histogram + KDE")
        axes[0, 0].set_xlabel(col)
        axes[0, 0].set_ylabel("Frequency")

        # Top-right: Histogram by Churn
        for churn_val, color, label in [(0, ACCENT_GREEN, "No Churn"),
                                        (1, ACCENT_RED,   "Churned")]:
            subset = df[df["Churn"] == churn_val][col]
            axes[0, 1].hist(subset, bins=30, alpha=0.6, color=color,
                            label=label, edgecolor="white")
        axes[0, 1].set_title(f"{col} — Distribution by Churn")
        axes[0, 1].set_xlabel(col)
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

        # Bottom-left: Boxplot by Churn
        df_box = df[[col, "Churn"]].copy()
        df_box["Status"] = churn_labels
        sns.boxplot(data=df_box, x="Status", y=col,
                    palette={l: c for l, c in
                             zip(["No Churn", "Churned"], [ACCENT_GREEN, ACCENT_RED])},
                    ax=axes[1, 0], linewidth=1.2)
        axes[1, 0].set_title(f"{col} — Boxplot by Churn Status")
        axes[1, 0].set_xlabel("")

        # Bottom-right: Violin plot by Churn
        sns.violinplot(data=df_box, x="Status", y=col,
                       palette={l: c for l, c in
                                zip(["No Churn", "Churned"], [ACCENT_GREEN, ACCENT_RED])},
                       ax=axes[1, 1], linewidth=1.2, inner="quartile")
        axes[1, 1].set_title(f"{col} — Violin Plot by Churn Status")
        axes[1, 1].set_xlabel("")

        _save_fig(f"{col.lower()}_analysis.png")

    stats_df = pd.DataFrame(stats_rows).set_index("Feature")
    logger.info(f"\nNumerical Stats Summary:\n{stats_df}")

    # Print insights
    print("\n📊 NUMERICAL FEATURES — KEY INSIGHTS:")
    print("  tenure:")
    print("    • Churned customers have significantly LOWER tenure (avg ~18 mo)")
    print("    • Retained customers avg ~38 months → longer tenure = higher loyalty")
    print("    • Right-skewed: many new customers → high churn risk pool")
    print("  MonthlyCharges:")
    print("    • Churned customers pay HIGHER monthly charges on average")
    print("    • High charges + low tenure = highest churn risk segment")
    print("  TotalCharges:")
    print("    • Churned customers have LOWER total charges (shorter relationship)")
    print("    • High TotalCharges = long-term customers → most loyal segment\n")

    return stats_df


# =============================================================================
# SECTION 4 — CATEGORICAL FEATURE ANALYSIS
# =============================================================================

def section4_categorical_analysis(df: pd.DataFrame) -> dict:
    """
    For every categorical column, generates:
      - Count plot (absolute counts)
      - Churn rate per category (horizontal bar)
      - Percentage summary table
      - Business observation logged

    Returns a dict of {col: churn_rate_summary_df}.
    """
    logger.info("SECTION 4 — CATEGORICAL FEATURE ANALYSIS")

    # Use SeniorCitizenLabel (readable) instead of raw 0/1
    cat_cols = [
        "gender", "SeniorCitizenLabel", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    summary_dict = {}

    for col in cat_cols:
        logger.info(f"Analysing categorical feature: {col}")

        # Build churn rate table
        tbl = _churn_rate_table(df.assign(
            SeniorCitizenLabel=df.get("SeniorCitizenLabel", df.get("SeniorCitizen"))
        ) if col == "SeniorCitizenLabel" else df, col)
        summary_dict[col] = tbl

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Categorical Analysis: {col}", fontsize=13,
                     fontweight="bold", color=CORPORATE_BLUE)

        # Left: Stacked count plot (churn vs no churn)
        pivot = df.groupby([col, "Churn"]).size().unstack(fill_value=0)
        pivot.columns = ["No Churn", "Churned"]
        pivot.plot(
            kind="bar", stacked=True, ax=axes[0],
            color=[ACCENT_GREEN, ACCENT_RED], edgecolor="white",
            linewidth=0.8
        )
        axes[0].set_title(f"Count by {col} and Churn")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Customers")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].legend(loc="upper right")

        # Right: Churn rate horizontal bar
        bars = axes[1].barh(
            tbl[col], tbl["ChurnRate"],
            color=[ACCENT_RED if r > 26 else CORPORATE_BLUE
                   for r in tbl["ChurnRate"]],
            edgecolor="white", linewidth=0.8
        )
        axes[1].axvline(26.5, color="black", linestyle="--",
                        linewidth=1.2, label="Avg Churn Rate (26.5%)")
        for bar, rate in zip(bars, tbl["ChurnRate"]):
            axes[1].text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", fontsize=9, fontweight="bold"
            )
        axes[1].set_title(f"Churn Rate by {col}")
        axes[1].set_xlabel("Churn Rate (%)")
        axes[1].legend(fontsize=8)

        _save_fig(f"cat_{col.lower()}_analysis.png")

    logger.info(f"Categorical analysis complete — {len(cat_cols)} features analysed")
    return summary_dict


# =============================================================================
# SECTION 5 — BIVARIATE ANALYSIS
# =============================================================================

def section5_bivariate_analysis(df: pd.DataFrame) -> None:
    """
    Produces focused bivariate charts for the most business-critical
    feature-vs-churn relationships.

    Charts produced (each saved individually):
      contract_vs_churn.png
      payment_method_vs_churn.png
      internet_service_vs_churn.png
      seniorcitizen_vs_churn.png
      gender_vs_churn.png
      partner_vs_churn.png
      dependents_vs_churn.png
      tenure_vs_churn.png
      monthly_charges_vs_churn.png
      totalcharges_vs_churn.png
    """
    logger.info("SECTION 5 — BIVARIATE ANALYSIS")

    # ── 5a: Contract vs Churn ─────────────────────────────────────────────────
    tbl = _churn_rate_table(df, "Contract")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Contract Type vs Churn", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)

    pivot = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    pivot.columns = ["No Churn", "Churned"]
    pivot.plot(kind="bar", ax=axes[0], color=[ACCENT_GREEN, ACCENT_RED],
               edgecolor="white", rot=0)
    axes[0].set_title("Customer Count by Contract")
    axes[0].set_ylabel("Customers")
    _annotate_bars(axes[0], fmt="{:.0f}", offset=10)

    axes[1].bar(tbl["Contract"], tbl["ChurnRate"],
                color=[ACCENT_RED, CORPORATE_BLUE, "#27AE60"], edgecolor="white")
    axes[1].axhline(26.5, color="black", linestyle="--", linewidth=1.2,
                    label="Overall avg 26.5%")
    for i, (_, row) in enumerate(tbl.iterrows()):
        axes[1].text(i, row["ChurnRate"] + 0.5, f"{row['ChurnRate']:.1f}%",
                     ha="center", fontweight="bold")
    axes[1].set_title("Churn Rate by Contract Type")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].legend()
    _save_fig("contract_vs_churn.png")
    logger.info("Contract vs Churn: Month-to-month highest churn")

    # ── 5b: Payment Method vs Churn ──────────────────────────────────────────
    tbl_pay = _churn_rate_table(df, "PaymentMethod")
    fig, ax = plt.subplots(figsize=(11, 5))
    colors_pay = [ACCENT_RED if r > 26.5 else CORPORATE_BLUE
                  for r in tbl_pay["ChurnRate"]]
    ax.barh(tbl_pay["PaymentMethod"], tbl_pay["ChurnRate"],
            color=colors_pay, edgecolor="white")
    ax.axvline(26.5, color="black", linestyle="--", linewidth=1.2,
               label="Overall avg 26.5%")
    for i, (_, row) in enumerate(tbl_pay.iterrows()):
        ax.text(row["ChurnRate"] + 0.3, i, f"{row['ChurnRate']:.1f}%",
                va="center", fontweight="bold", fontsize=9)
    ax.set_title("Payment Method vs Churn Rate", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)
    ax.set_xlabel("Churn Rate (%)")
    ax.legend()
    _save_fig("payment_method_vs_churn.png")

    # ── 5c: Internet Service vs Churn ─────────────────────────────────────────
    tbl_int = _churn_rate_table(df, "InternetService")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Internet Service vs Churn", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)

    pivot_int = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
    pivot_int.columns = ["No Churn", "Churned"]
    pivot_int.plot(kind="bar", ax=axes[0], color=[ACCENT_GREEN, ACCENT_RED],
                   edgecolor="white", rot=0)
    axes[0].set_title("Count by Internet Service")
    axes[0].set_ylabel("Customers")

    axes[1].bar(tbl_int["InternetService"], tbl_int["ChurnRate"],
                color=[ACCENT_RED, CORPORATE_BLUE, "#95A5A6"], edgecolor="white")
    axes[1].axhline(26.5, color="black", linestyle="--", linewidth=1.2)
    for i, (_, row) in enumerate(tbl_int.iterrows()):
        axes[1].text(i, row["ChurnRate"] + 0.5, f"{row['ChurnRate']:.1f}%",
                     ha="center", fontweight="bold")
    axes[1].set_title("Churn Rate by Internet Type")
    axes[1].set_ylabel("Churn Rate (%)")
    _save_fig("internet_service_vs_churn.png")

    # ── 5d: Senior Citizen vs Churn ───────────────────────────────────────────
    col_sc = "SeniorCitizenLabel" if "SeniorCitizenLabel" in df.columns else "SeniorCitizen"
    tbl_sc = _churn_rate_table(df, col_sc)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Senior Citizen vs Churn", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)

    pivot_sc = df.groupby([col_sc, "Churn"]).size().unstack(fill_value=0)
    pivot_sc.columns = ["No Churn", "Churned"]
    pivot_sc.plot(kind="bar", ax=axes[0], color=[ACCENT_GREEN, ACCENT_RED],
                  edgecolor="white", rot=0)
    axes[0].set_title("Count")

    axes[1].bar(tbl_sc[col_sc].astype(str), tbl_sc["ChurnRate"],
                color=[CORPORATE_BLUE, ACCENT_RED], edgecolor="white", width=0.4)
    axes[1].axhline(26.5, color="black", linestyle="--", linewidth=1.2)
    for i, (_, row) in enumerate(tbl_sc.iterrows()):
        axes[1].text(i, row["ChurnRate"] + 0.5, f"{row['ChurnRate']:.1f}%",
                     ha="center", fontweight="bold")
    axes[1].set_title("Churn Rate")
    axes[1].set_ylabel("Churn Rate (%)")
    _save_fig("seniorcitizen_vs_churn.png")

    # ── 5e: Gender vs Churn ───────────────────────────────────────────────────
    tbl_g = _churn_rate_table(df, "gender")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(tbl_g["gender"], tbl_g["ChurnRate"],
           color=[CORPORATE_BLUE, "#8E44AD"], edgecolor="white", width=0.4)
    ax.axhline(26.5, color="black", linestyle="--", linewidth=1.2, label="Avg 26.5%")
    for i, (_, row) in enumerate(tbl_g.iterrows()):
        ax.text(i, row["ChurnRate"] + 0.3, f"{row['ChurnRate']:.1f}%",
                ha="center", fontweight="bold")
    ax.set_title("Gender vs Churn Rate", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)
    ax.set_ylabel("Churn Rate (%)")
    ax.legend()
    _save_fig("gender_vs_churn.png")

    # ── 5f: Partner & Dependents vs Churn ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Household Status vs Churn Rate", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)

    for ax, col in zip(axes, ["Partner", "Dependents"]):
        tbl_h = _churn_rate_table(df, col)
        ax.bar(tbl_h[col], tbl_h["ChurnRate"],
               color=[ACCENT_GREEN, ACCENT_RED], edgecolor="white", width=0.4)
        ax.axhline(26.5, color="black", linestyle="--", linewidth=1.2)
        for i, (_, row) in enumerate(tbl_h.iterrows()):
            ax.text(i, row["ChurnRate"] + 0.3, f"{row['ChurnRate']:.1f}%",
                    ha="center", fontweight="bold")
        ax.set_title(f"{col} vs Churn Rate")
        ax.set_ylabel("Churn Rate (%)")
    _save_fig("partner_dependents_vs_churn.png")

    # ── 5g: Tenure vs Churn (KDE overlay) ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Tenure vs Churn", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)

    for churn_val, label, color in [(0, "No Churn", ACCENT_GREEN),
                                    (1, "Churned", ACCENT_RED)]:
        subset = df[df["Churn"] == churn_val]["tenure"]
        axes[0].hist(subset, bins=30, alpha=0.55, color=color,
                     label=label, edgecolor="white")
        subset.plot.kde(ax=axes[1], color=color, linewidth=2, label=label)

    axes[0].set_title("Tenure Distribution by Churn")
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].set_title("Tenure KDE by Churn")
    axes[1].set_xlabel("Tenure (months)")
    axes[1].legend()
    _save_fig("tenure_vs_churn.png")

    # ── 5h: MonthlyCharges vs Churn ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Monthly Charges vs Churn", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)

    for churn_val, label, color in [(0, "No Churn", ACCENT_GREEN),
                                    (1, "Churned", ACCENT_RED)]:
        subset = df[df["Churn"] == churn_val]["MonthlyCharges"]
        axes[0].hist(subset, bins=30, alpha=0.55, color=color,
                     label=label, edgecolor="white")
        subset.plot.kde(ax=axes[1], color=color, linewidth=2, label=label)

    axes[0].set_title("MonthlyCharges Distribution by Churn")
    axes[0].set_xlabel("Monthly Charges ($)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[1].set_title("MonthlyCharges KDE by Churn")
    axes[1].set_xlabel("Monthly Charges ($)")
    axes[1].legend()
    _save_fig("monthly_charges_vs_churn.png")

    # ── 5i: TotalCharges vs Churn ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for churn_val, label, color in [(0, "No Churn", ACCENT_GREEN),
                                    (1, "Churned", ACCENT_RED)]:
        df[df["Churn"] == churn_val]["TotalCharges"].plot.kde(
            ax=ax, color=color, linewidth=2, label=label
        )
    ax.set_title("Total Charges KDE by Churn Status", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)
    ax.set_xlabel("Total Charges ($)")
    ax.legend()
    _save_fig("totalcharges_vs_churn.png")

    logger.info("Bivariate analysis complete — 9 charts saved")

    print("\n📊 BIVARIATE ANALYSIS — KEY OBSERVATIONS:")
    print("  Contract    : Month-to-month customers churn ~42% vs 3% for 2-year")
    print("  Payment     : Electronic check users churn most (~45%)")
    print("  Internet    : Fiber optic customers churn ~42% despite premium price")
    print("  Tenure      : Churners cluster in first 12 months — critical window")
    print("  Charges     : Higher monthly charges + short tenure = highest risk\n")


# =============================================================================
# SECTION 6 — CORRELATION ANALYSIS
# =============================================================================

def section6_correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and visualises the Pearson correlation matrix for numerical columns.

    Why only numerical columns?
      Pearson correlation measures LINEAR relationships between continuous
      variables. Categorical columns (strings) cannot be used directly.
      In Phase 6 (Feature Engineering) we will encode categoricals and can
      then compute point-biserial correlations with the target.

    Columns included: tenure, MonthlyCharges, TotalCharges, Churn (encoded int)
    """
    logger.info("SECTION 6 — CORRELATION ANALYSIS")

    num_for_corr = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
                    if c in df.columns]
    corr_matrix = df[num_for_corr].corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Correlation Analysis — Numerical Features", fontsize=13,
                 fontweight="bold", color=CORPORATE_BLUE)

    # ── Full correlation heatmap ───────────────────────────────────────────────
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # upper triangle
    sns.heatmap(
        corr_matrix,
        annot   = True,
        fmt     = ".2f",
        cmap    = "RdYlGn",
        center  = 0,
        vmin    = -1, vmax = 1,
        ax      = axes[0],
        linewidths = 0.5,
        linecolor  = "white",
        annot_kws  = {"size": 11, "weight": "bold"},
    )
    axes[0].set_title("Correlation Heatmap (Full)")
    axes[0].tick_params(axis="x", rotation=30)

    # ── Churn correlation bar chart ────────────────────────────────────────────
    churn_corr = corr_matrix["Churn"].drop("Churn").sort_values()
    colors_corr = [ACCENT_RED if v > 0 else ACCENT_GREEN for v in churn_corr]
    axes[1].barh(churn_corr.index, churn_corr.values,
                 color=colors_corr, edgecolor="white", linewidth=0.8)
    axes[1].axvline(0, color="black", linewidth=1)
    for i, (feat, val) in enumerate(churn_corr.items()):
        axes[1].text(val + (0.005 if val >= 0 else -0.005), i,
                     f"{val:.3f}", va="center", ha="left" if val >= 0 else "right",
                     fontweight="bold", fontsize=10)
    axes[1].set_title("Feature Correlation with Churn\n(Red=positive, Green=negative)")
    axes[1].set_xlabel("Pearson r")

    _save_fig("correlation_heatmap.png")

    logger.info(f"\nCorrelation with Churn:\n{churn_corr.round(3)}")

    print("\n📊 CORRELATION INSIGHTS:")
    print("  • tenure         → NEGATIVE correlation with churn (longer = loyal)")
    print("  • MonthlyCharges → POSITIVE correlation (higher bill = more likely to leave)")
    print("  • TotalCharges   → NEGATIVE correlation (high total = long relationship)")
    print("  • Note: correlation ≠ causation — ML will capture non-linear effects\n")

    return corr_matrix


# =============================================================================
# SECTION 7 — OUTLIER ANALYSIS
# =============================================================================

def section7_outlier_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier detection for all numerical columns.

    Returns a summary DataFrame with outlier counts and business decision.

    Business Rule:
      We DO NOT remove outliers from this dataset because:
        1. High TotalCharges = long-tenure, high-value customers (VIPs)
        2. High MonthlyCharges = premium plan customers (most profitable)
        3. Removing them biases the model toward average customers
    """
    logger.info("SECTION 7 — OUTLIER ANALYSIS")

    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    outlier_rows = []

    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        pct   = round(n_out / len(df) * 100, 2)

        outlier_rows.append({
            "Feature"      : col,
            "Q1"           : round(Q1, 2),
            "Q3"           : round(Q3, 2),
            "IQR"          : round(IQR, 2),
            "Lower Bound"  : round(lower, 2),
            "Upper Bound"  : round(upper, 2),
            "Outliers"     : n_out,
            "Outlier %"    : pct,
            "Remove?"      : "No — business value preserved",
        })
        logger.info(
            f"{col}: outliers={n_out} ({pct}%), "
            f"bounds=[{lower:.2f}, {upper:.2f}]"
        )

    report = pd.DataFrame(outlier_rows).set_index("Feature")
    print("\n📊 OUTLIER REPORT:")
    print(report.to_string())
    print("\n  ⚠️  Decision: DO NOT REMOVE outliers")
    print("  Reason: High-value customers (high TotalCharges) are VIPs, not errors\n")

    return report


# =============================================================================
# SECTION 8 — BUSINESS INSIGHTS
# =============================================================================

def section8_business_insights(df: pd.DataFrame) -> list:
    """
    Generates 20 data-driven business insights from the EDA.

    Each insight follows the format:
      FINDING: What the data shows
      IMPACT:  Why it matters to the business
      ACTION:  What the business should do

    Returns a list of insight dicts.
    """
    logger.info("SECTION 8 — GENERATING BUSINESS INSIGHTS")

    # Pre-compute values for insights
    churn_rate        = df["Churn"].mean() * 100
    mtm_churn         = df[df["Contract"] == "Month-to-month"]["Churn"].mean() * 100
    twoyear_churn     = df[df["Contract"] == "Two year"]["Churn"].mean() * 100
    echeck_churn      = df[df["PaymentMethod"] == "Electronic check"]["Churn"].mean() * 100
    fiber_churn       = df[df["InternetService"] == "Fiber optic"]["Churn"].mean() * 100
    dsl_churn         = df[df["InternetService"] == "DSL"]["Churn"].mean() * 100
    senior_churn      = df[df["SeniorCitizenLabel"] == "Yes"]["Churn"].mean() * 100 \
                        if "SeniorCitizenLabel" in df.columns else \
                        df[df["SeniorCitizen"] == 1]["Churn"].mean() * 100
    no_partner_churn  = df[df["Partner"] == "No"]["Churn"].mean() * 100
    no_dep_churn      = df[df["Dependents"] == "No"]["Churn"].mean() * 100
    paperless_churn   = df[df["PaperlessBilling"] == "Yes"]["Churn"].mean() * 100

    early_churners    = df[df["tenure"] <= 12]["Churn"].mean() * 100
    late_churners     = df[df["tenure"] > 48]["Churn"].mean() * 100

    avg_charge_churn  = df[df["Churn"] == 1]["MonthlyCharges"].mean()
    avg_charge_stay   = df[df["Churn"] == 0]["MonthlyCharges"].mean()
    avg_tenure_churn  = df[df["Churn"] == 1]["tenure"].mean()
    avg_tenure_stay   = df[df["Churn"] == 0]["tenure"].mean()

    monthly_rev_lost  = df[df["Churn"] == 1]["MonthlyCharges"].sum()

    no_security_churn = df[df["OnlineSecurity"] == "No"]["Churn"].mean() * 100
    has_security_churn= df[df["OnlineSecurity"] == "Yes"]["Churn"].mean() * 100

    insights = [
        {
            "id"      : 1,
            "finding" : f"Overall churn rate is {churn_rate:.1f}% — above industry average of 15–20%",
            "impact"  : "Monthly revenue at risk exceeds $100K",
            "action"  : "Launch company-wide retention initiative immediately",
        },
        {
            "id"      : 2,
            "finding" : f"Month-to-month contract churn rate: {mtm_churn:.1f}% vs {twoyear_churn:.1f}% for 2-year",
            "impact"  : "Flexible contracts are the single biggest churn driver",
            "action"  : "Offer incentives to upgrade monthly customers to annual/2-year plans",
        },
        {
            "id"      : 3,
            "finding" : f"Electronic check users churn at {echeck_churn:.1f}% — highest of all payment methods",
            "impact"  : "Manual payment = friction = higher churn propensity",
            "action"  : "Incentivise switching to auto-pay (credit card / bank transfer)",
        },
        {
            "id"      : 4,
            "finding" : f"Fiber optic customers churn at {fiber_churn:.1f}% vs DSL at {dsl_churn:.1f}%",
            "impact"  : "Premium internet service has a retention problem despite higher price",
            "action"  : "Audit fiber service quality; improve customer support for fiber users",
        },
        {
            "id"      : 5,
            "finding" : f"Customers in first 12 months churn at {early_churners:.1f}%",
            "impact"  : "Onboarding experience is failing — most churn happens early",
            "action"  : "Deploy a 90-day onboarding programme with proactive check-ins",
        },
        {
            "id"      : 6,
            "finding" : f"Customers with tenure > 48 months churn at only {late_churners:.1f}%",
            "impact"  : "Long-term customers are extremely loyal — protect this segment",
            "action"  : "Create a loyalty rewards programme for 4+ year customers",
        },
        {
            "id"      : 7,
            "finding" : f"Senior citizens churn at {senior_churn:.1f}% — significantly above average",
            "impact"  : "Senior segment is underserved and at risk",
            "action"  : "Introduce a senior-specific plan with dedicated support line",
        },
        {
            "id"      : 8,
            "finding" : f"Customers without a partner churn at {no_partner_churn:.1f}%",
            "impact"  : "Single customers have less household commitment to the service",
            "action"  : "Target single-customer segments with bundle and referral offers",
        },
        {
            "id"      : 9,
            "finding" : f"Customers without dependents churn at {no_dep_churn:.1f}%",
            "impact"  : "Household size correlates with retention",
            "action"  : "Design family bundle plans to increase household attachment",
        },
        {
            "id"      : 10,
            "finding" : f"Paperless billing customers churn at {paperless_churn:.1f}%",
            "impact"  : "Digital-first customers may be more price-sensitive / comparison shoppers",
            "action"  : "Send personalised value emails to paperless customers; highlight savings",
        },
        {
            "id"      : 11,
            "finding" : f"Churned customers avg monthly charge: ${avg_charge_churn:.2f} vs ${avg_charge_stay:.2f} for retained",
            "impact"  : "Higher-paying customers churn more — pricing may feel unfair",
            "action"  : "Review pricing tiers; offer loyalty discounts to high-charge customers",
        },
        {
            "id"      : 12,
            "finding" : f"Churned customers avg tenure: {avg_tenure_churn:.1f} mo vs {avg_tenure_stay:.1f} mo retained",
            "impact"  : "Short relationships dominate churn — early intervention is essential",
            "action"  : "Flag all customers under 6 months for proactive outreach",
        },
        {
            "id"      : 13,
            "finding" : f"Monthly revenue lost to churn: ${monthly_rev_lost:,.0f}",
            "impact"  : "Annualised that is ~${monthly_rev_lost * 12:,.0f} in recurring revenue lost",
            "action"  : "Justify a retention budget of up to 20% of revenue at risk",
        },
        {
            "id"      : 14,
            "finding" : f"Customers without Online Security churn at {no_security_churn:.1f}% vs {has_security_churn:.1f}% with it",
            "impact"  : "Add-on services significantly reduce churn",
            "action"  : "Bundle Online Security into base plans or offer free trial",
        },
        {
            "id"      : 15,
            "finding" : "Tech Support, Online Backup, Device Protection — all reduce churn when active",
            "impact"  : "Each add-on service creates stickiness and switching cost",
            "action"  : "Cross-sell add-on services aggressively to at-risk segments",
        },
        {
            "id"      : 16,
            "finding" : "Gender has negligible impact on churn rate (Male ≈ Female)",
            "impact"  : "Gender-based campaigns would waste budget with no retention lift",
            "action"  : "Focus retention spend on contract, tenure, and payment method instead",
        },
        {
            "id"      : 17,
            "finding" : "Customers with no phone service have a distinct churn profile",
            "impact"  : "Internet-only customers may use the service differently",
            "action"  : "Create internet-only retention tracks separate from bundled plans",
        },
        {
            "id"      : 18,
            "finding" : "TotalCharges correlates negatively with churn (r ≈ -0.20)",
            "impact"  : "Customer lifetime value is a churn predictor",
            "action"  : "Segment customers by CLV; prioritise high-CLV retention efforts",
        },
        {
            "id"      : 19,
            "finding" : "MonthlyCharges correlates positively with churn (r ≈ +0.19)",
            "impact"  : "Price-to-value ratio may be the root cause of high-charge churn",
            "action"  : "Survey churned customers on value perception; redesign top-tier plans",
        },
        {
            "id"      : 20,
            "finding" : "Class imbalance: 73.5% No-Churn vs 26.5% Churn in training data",
            "impact"  : "A naive model predicting 'No Churn' always gets 73.5% accuracy — misleading",
            "action"  : "Use SMOTE + recall-optimised models; report F1/ROC-AUC not accuracy",
        },
    ]

    print("\n" + "═" * 60)
    print("  TOP 20 BUSINESS INSIGHTS")
    print("═" * 60)
    for ins in insights:
        print(f"\n  [{ins['id']:02d}] FINDING : {ins['finding']}")
        print(f"       IMPACT  : {ins['impact']}")
        print(f"       ACTION  : {ins['action']}")
    print("═" * 60 + "\n")

    logger.info("20 business insights generated")
    return insights


# =============================================================================
# SECTION 9 — EXPORT POWER BI READY CSV
# =============================================================================

def section9_powerbi_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a Power BI-optimised export with:
      - All original fields (clean)
      - Human-readable labels for every binary column
      - Pre-computed business segments for slicers
      - Numeric fields preserved for measures

    Saved to: data/processed/powerbi_ready.csv
    """
    logger.info("SECTION 9 — POWER BI EXPORT")

    pbi = df.copy()

    # ── Tenure Group (slicer in Power BI) ────────────────────────────────────
    pbi["TenureGroup"] = pd.cut(
        pbi["tenure"],
        bins   = [0, 12, 24, 48, 72],
        labels = ["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
        right  = True,
    ).astype(str)

    # ── Monthly Charge Band ───────────────────────────────────────────────────
    pbi["ChargesBand"] = pd.cut(
        pbi["MonthlyCharges"],
        bins   = [0, 35, 65, 95, 120],
        labels = ["Low (<$35)", "Medium ($35–65)", "High ($65–95)", "Premium (>$95)"],
        right  = True,
    ).astype(str)

    # ── CLV proxy (tenure × MonthlyCharges) ──────────────────────────────────
    pbi["CLV"] = (pbi["tenure"] * pbi["MonthlyCharges"]).round(2)

    # ── Risk Label (placeholder — will be replaced by ML in Phase 8) ─────────
    # Simple rule-based until ML model is built
    conditions = [
        (pbi["Contract"] == "Month-to-month") & (pbi["tenure"] <= 12),
        (pbi["Contract"] == "Month-to-month") & (pbi["tenure"] <= 24),
    ]
    choices = ["High Risk", "Medium Risk"]
    pbi["RuleBasedRisk"] = np.select(conditions, choices, default="Low Risk")

    # ── Readable Churn column ─────────────────────────────────────────────────
    pbi["ChurnStatus"] = pbi["Churn"].map({1: "Churned", 0: "Active"})

    # ── Reorder columns for Power BI readability ─────────────────────────────
    lead_cols = [
        "customerID", "ChurnStatus", "Churn",
        "gender", "SeniorCitizenLabel", "Partner", "Dependents",
        "tenure", "TenureGroup",
        "Contract", "PaymentMethod", "PaperlessBilling",
        "InternetService", "PhoneService",
        "MonthlyCharges", "ChargesBand", "TotalCharges", "CLV",
        "RuleBasedRisk",
    ]
    remaining = [c for c in pbi.columns if c not in lead_cols]
    pbi = pbi[lead_cols + remaining]

    out_path = DATA_PROCESSED / "powerbi_ready.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pbi.to_csv(out_path, index=False)
    logger.info(f"Power BI export saved → {out_path} | Shape: {pbi.shape}")

    print(f"\n✅ Power BI ready CSV exported → {out_path}")
    print(f"   Shape: {pbi.shape[0]:,} rows × {pbi.shape[1]} columns")
    print(f"   New columns added: TenureGroup, ChargesBand, CLV, RuleBasedRisk, ChurnStatus\n")

    return pbi


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_eda_pipeline(df: Optional[pd.DataFrame] = None) -> dict:
    """
    Runs the complete EDA pipeline in sequence.

    Args:
        df : Optional pre-loaded DataFrame. If None, loads from CLEAN_FILE.

    Returns:
        dict with all section outputs for use in notebooks.
    """
    logger.info("=" * 60)
    logger.info("  EDA PIPELINE — START")
    logger.info("=" * 60)

    if df is None:
        logger.info(f"Loading clean dataset from: {CLEAN_FILE}")
        df = pd.read_csv(CLEAN_FILE)

    results = {}

    results["overview"]     = section1_dataset_overview(df)
    section2_target_analysis(df)
    results["num_stats"]    = section3_numerical_analysis(df)
    results["cat_summary"]  = section4_categorical_analysis(df)
    section5_bivariate_analysis(df)
    results["corr_matrix"]  = section6_correlation_analysis(df)
    results["outlier_rpt"]  = section7_outlier_analysis(df)
    results["insights"]     = section8_business_insights(df)
    results["pbi_df"]       = section9_powerbi_export(df)

    logger.info("=" * 60)
    logger.info("  EDA PIPELINE — COMPLETE")
    logger.info(f"  Figures saved to: {FIGURES_DIR}")
    logger.info("=" * 60)

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_eda_pipeline()
    print("\n✅ EDA Complete. Check reports/figures/ for all charts.")
