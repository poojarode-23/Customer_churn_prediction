# =============================================================================
# notebooks/02_data_cleaning.py
# Phase 2 — Interactive Data Cleaning Walkthrough
# Run cell-by-cell in Jupyter for a guided experience
# =============================================================================

# %% [markdown]
# # Phase 2: Data Cleaning
# **Goal:** Transform the raw IBM Telco CSV into a clean, reliable dataset
# ready for EDA and Machine Learning.
#
# ## Why Data Cleaning Matters
# - Real-world data is NEVER clean
# - ML models learn from data — dirty data = broken models
# - "Garbage In, Garbage Out" is the #1 reason ML projects fail
# - Industry estimate: data scientists spend 70–80% of time on data cleaning

# %%
# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import RAW_FILE, CLEAN_FILE
from src.data.data_cleaning import (
    load_raw_data,
    audit_dataset,
    fix_data_types,
    handle_missing_values,
    handle_duplicates,
    standardise_values,
    detect_outliers,
    validate_clean_data,
    save_clean_data,
    run_cleaning_pipeline,
)

print("All imports successful ✓")

# %% [markdown]
# ## Step 1: Load Raw Data

# %%
# ── Cell 2: Load ─────────────────────────────────────────────────────────────
df_raw = load_raw_data(RAW_FILE)
print(f"Shape: {df_raw.shape}")
df_raw.head()

# %% [markdown]
# ## Step 2: Audit the Dataset
# Before touching any data, we always audit first.
# This is a read-only step — we are OBSERVERS here.

# %%
# ── Cell 3: Audit ────────────────────────────────────────────────────────────
audit_report = audit_dataset(df_raw)

# %% [markdown]
# ### Key Audit Findings (IBM Telco):
# | Finding | Detail | Action |
# |---------|--------|--------|
# | `TotalCharges` is object type | Contains spaces instead of numbers | Convert to float |
# | ~11 rows have blank TotalCharges | New customers (tenure=0) | Fill with 0 |
# | No duplicate rows | Clean CRM export | No action needed |
# | Target: 26.5% churn, 73.5% no-churn | **Class imbalance!** | Handle in ML phase (SMOTE) |

# %%
# ── Cell 4: Visualise target distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Target Variable Distribution — Churn", fontsize=13, fontweight="bold")

churn_counts = df_raw["Churn"].value_counts()
churn_pct    = df_raw["Churn"].value_counts(normalize=True).mul(100)

# Bar chart
axes[0].bar(churn_counts.index, churn_counts.values,
            color=["#2ECC71", "#E74C3C"], edgecolor="white", linewidth=1.5)
axes[0].set_title("Count")
axes[0].set_ylabel("Customers")
for i, (label, val) in enumerate(churn_counts.items()):
    axes[0].text(i, val + 30, f"{val:,}", ha="center", fontweight="bold")

# Pie chart
axes[1].pie(churn_pct.values, labels=churn_pct.index,
            autopct="%1.1f%%", colors=["#2ECC71", "#E74C3C"],
            startangle=140, wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Percentage")

plt.tight_layout()
plt.savefig("reports/figures/target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📊 Business Insight: Dataset has class imbalance (73.5% No-Churn vs 26.5% Churn)")
print("   → We will use SMOTE in Phase 7 to handle this before training models")

# %%
# ── Cell 5: The TotalCharges Bug ─────────────────────────────────────────────
print("TotalCharges dtype BEFORE fix:", df_raw["TotalCharges"].dtype)
print("\nSample of blank TotalCharges rows:")
blank_mask = df_raw["TotalCharges"].str.strip() == ""
print(df_raw[blank_mask][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])

# %% [markdown]
# ## Step 3-6: Run Full Cleaning Pipeline

# %%
# ── Cell 6: Run pipeline ─────────────────────────────────────────────────────
# This runs all 9 steps in sequence
df_clean = run_cleaning_pipeline()

# %% [markdown]
# ## Step 7: Inspect Clean Dataset

# %%
# ── Cell 7: Post-cleaning checks ─────────────────────────────────────────────
print("Shape:", df_clean.shape)
print("\nData types:")
print(df_clean.dtypes)
print("\nMissing values:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# %%
# ── Cell 8: Outlier visualisation ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Numerical Columns — Distribution & Outliers", fontsize=13, fontweight="bold")

for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    ax.boxplot(df_clean[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor="#3498DB", alpha=0.6))
    ax.set_title(col)
    ax.set_ylabel("Value")

plt.tight_layout()
plt.savefig("reports/figures/post_clean_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Cleaning Summary
# | Step | Action | Rows Affected |
# |------|--------|---------------|
# | Fix TotalCharges dtype | string → float64 | All 7,043 |
# | Fill NaN TotalCharges | → 0.0 (new customers) | ~11 rows |
# | Remove duplicates | None found | 0 |
# | Collapse verbose "No" values | 6 columns simplified | All rows |
# | Encode Churn | Yes/No → 1/0 | All 7,043 |
# | Add SeniorCitizenLabel | 0/1 → Yes/No readable | All 7,043 |
# | Add ChurnLabel | Preserve original string | All 7,043 |
#
# **Output saved to:** `data/processed/telco_clean.csv`
