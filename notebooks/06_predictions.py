# =============================================================================
# notebooks/06_predictions.py
# Phase 6 — Interactive Prediction Walkthrough
# Run cell-by-cell in Jupyter
# =============================================================================

# %% [markdown]
# # Phase 6: Generating Predictions
#
# **Goal:** Score every customer with churn probability, risk category,
# and revenue-at-risk — and export a business-ready predictions.csv.
#
# ## From Model to Business Action
# A trained ML model is useless unless its outputs can be:
# - Read by non-technical stakeholders (risk categories, not raw probabilities)
# - Actioned by the retention team (priority-ranked customer lists)
# - Imported into Power BI (structured CSV with the right columns)
# - Validated for integrity (no nulls, valid ranges, correct logic)

# %%
# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from config.settings import DATA_PREDICTIONS, MODELS_DIR, FIGURES_DIR
from src.models.predict import run_prediction_pipeline, load_model_artefacts

print("✓ Imports successful")

# %% [markdown]
# ## Step 1: Run the Full Prediction Pipeline

# %%
predictions = run_prediction_pipeline()
print(f"Predictions shape: {predictions.shape}")
print(f"Columns: {list(predictions.columns)}")

# %% [markdown]
# ## Step 2: Inspect the Output

# %%
print("\nSample predictions (top 10 highest risk):")
display_cols = [
    "customerID", "ActualChurnLabel", "PredictedChurnLabel",
    "ChurnProbabilityPct", "RiskCategory", "RiskScore",
    "RevenueAtRisk", "Contract", "tenure"
]
display_cols = [c for c in display_cols if c in predictions.columns]
print(predictions[display_cols].head(10).to_string(index=False))

# %% [markdown]
# ## Step 3: Risk Category Deep-Dive

# %%
# Summary statistics by risk category
risk_summary = predictions.groupby("RiskCategory").agg(
    Customers       = ("customerID", "count"),
    AvgProbability  = ("ChurnProbabilityPct", "mean"),
    AvgMonthlyCharge= ("MonthlyCharges", "mean"),
    AvgTenure       = ("tenure", "mean"),
    TotalRevAtRisk  = ("RevenueAtRisk", "sum"),
    ActualChurnRate = ("ActualChurn", "mean"),
).round(2)

print("\nRisk Category Summary:")
print(risk_summary.to_string())

# %% [markdown]
# ## Step 4: Validate Model Performance on Full Dataset

# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_actual = predictions["ActualChurn"].astype(int)
y_pred   = predictions["PredictedChurn"].astype(int)
y_prob   = predictions["ChurnProbability"].astype(float)

print("Full Dataset Performance:")
print(classification_report(y_actual, y_pred, target_names=["No Churn", "Churned"]))
print(f"ROC-AUC (full dataset): {roc_auc_score(y_actual, y_prob):.4f}")

# %% [markdown]
# ## Step 5: High-Risk Customer List

# %%
high_risk = predictions[predictions["RiskCategory"] == "High Risk"].copy()

print(f"\nHigh Risk Customers: {len(high_risk):,}")
print(f"Avg churn probability: {high_risk['ChurnProbabilityPct'].mean():.1f}%")
print(f"Total monthly revenue at risk: ${high_risk['RevenueAtRisk'].sum():,.2f}")
print(f"\nTop 10 Priority Actions:")

action_cols = ["customerID", "ChurnProbabilityPct", "MonthlyCharges",
               "Contract", "tenure", "PaymentMethod"]
action_cols = [c for c in action_cols if c in high_risk.columns]
print(high_risk[action_cols].head(10).to_string(index=False))

# %% [markdown]
# ## Step 6: Revenue Impact Analysis

# %%
total_monthly_risk  = predictions["RevenueAtRisk"].sum()
annual_risk         = total_monthly_risk * 12
high_risk_rev       = high_risk["RevenueAtRisk"].sum()

print("\n── Revenue Impact ──────────────────────────")
print(f"Predicted churners          : {predictions['PredictedChurn'].sum():,}")
print(f"Monthly revenue at risk     : ${total_monthly_risk:>10,.2f}")
print(f"Annual revenue at risk      : ${annual_risk:>10,.2f}")
print(f"High-risk monthly rev       : ${high_risk_rev:>10,.2f}")
print(f"\nIf 30% of predicted churners are retained:")
retained_revenue = total_monthly_risk * 0.30 * 12
print(f"Annual revenue saved        : ${retained_revenue:>10,.2f}")

# %% [markdown]
# ## Step 7: Prediction Files Created

# %%
import os
pred_files = list(DATA_PREDICTIONS.iterdir())
print("\nPrediction files saved:")
for f in sorted(pred_files):
    size_kb = os.path.getsize(f) // 1024
    rows    = sum(1 for _ in open(f)) - 1
    print(f"  {f.name:<35} {rows:>6,} rows  {size_kb:>5} KB")

print("\nFigures generated:")
new_figs = [
    "probability_distribution.png",
    "risk_category_breakdown.png",
    "revenue_at_risk.png",
    "probability_vs_charges_scatter.png",
    "top20_high_risk_customers.png",
]
for fig in new_figs:
    path = FIGURES_DIR / fig
    exists = "✓" if path.exists() else "✗"
    print(f"  {exists}  {fig}")

# %% [markdown]
# ## Step 8: Power BI Connection Guide

# %%
print("""
Power BI — How to Import predictions.csv:
==========================================

1. Open Power BI Desktop
2. Home → Get Data → Text/CSV
3. Navigate to: data/predictions/predictions.csv
4. Click Load

Key columns for each dashboard page:
─────────────────────────────────────
Executive Summary:
  • COUNT(customerID)      → Total customers
  • SUM(RevenueAtRisk)     → Monthly revenue at risk
  • AVERAGE(ChurnProb..)   → Avg churn probability
  • COUNT where Predicted  → Predicted churners

AI Prediction Page:
  • RiskCategory           → Slicer
  • ChurnProbabilityPct    → Gauge / bar
  • PriorityRank           → Sort table
  • EstimatedCLVAtRisk     → Revenue impact

Customer Analysis:
  • Contract               → Slicer / groupby
  • TenureGroup            → Slicer
  • ChargesBand            → Slicer
  • PaymentMethod          → Slicer

Create these DAX measures (Phase 7):
  High Risk Count = COUNTROWS(FILTER(predictions, [RiskCategory]="High Risk"))
  Revenue At Risk = SUM(predictions[RevenueAtRisk])
  Predicted Churn Rate = DIVIDE(SUM([PredictedChurn]), COUNT([customerID]))
""")

# %% [markdown]
# ---
# # 🎤 Interview Q&A — Prediction Phase
#
# **Q1: What columns does your predictions.csv contain and why?**
# > customerID (identifier), ActualChurn + label (ground truth for validation),
# > PredictedChurn + label (model output), ChurnProbability (raw score),
# > ChurnProbabilityPct (human-readable %), RiskCategory (High/Medium/Low),
# > RiskScore (1–100 integer for dashboards), RevenueAtRisk (financial impact),
# > CLV (lifetime value), PriorityRank (retention team sort order), plus
# > all original customer attributes for Power BI filtering.
#
# **Q2: How do you turn a probability score into a business action?**
# > Three-tier risk categorisation:
# > High Risk (≥70%): Immediate personal outreach, offer retention deals
# > Medium Risk (40-69%): Targeted email campaign, loyalty offer
# > Low Risk (<40%): Standard engagement, monitor quarterly
# > The priority rank tells the retention team exactly who to call first.
#
# **Q3: What is the difference between PredictedChurn and RiskCategory?**
# > PredictedChurn is binary (0/1) based on the 0.40 threshold — used for
# > accuracy/recall metrics. RiskCategory is a three-tier business label that
# > provides more nuance: a customer at 68% is in a very different situation
# > from one at 41%, even though both get PredictedChurn=1.
#
# **Q4: How would you explain a 'High Risk' customer flag to a stakeholder?**
# > "Our model predicts this customer has a 70%+ chance of cancelling in
# > the next billing cycle, based on their contract type, tenure, payment
# > method, and service usage. They should receive a personal call from
# > the retention team this week with a targeted offer."
#
# **Q5: What is EstimatedCLVAtRisk and how is it calculated?**
# > It estimates the future revenue we'd lose if this customer churns.
# > Formula: MonthlyCharges × max(0, 36 - tenure). This uses 36 months
# > as an assumed average remaining lifetime. It's a conservative proxy —
# > a real CLV would incorporate margin, discount rate, and upsell probability.
#
# **Q6: Why do you score all 7,043 customers, not just the test set?**
# > The test set (1,409 rows) was held out for model evaluation — we know
# > who churned. In production, you need to score ALL active customers
# > weekly because you don't know in advance who will churn next. The full
# > dataset output mirrors the real-world use case of the model.
#
# **Q7: How would this prediction pipeline run in production?**
# > Weekly batch job: (1) Pull current customer data from CRM/database,
# > (2) Apply the same feature engineering pipeline, (3) Load model from
# > model registry, (4) Generate predictions, (5) Write to database/S3,
# > (6) Auto-refresh Power BI dashboard, (7) Email high-risk list to
# > retention team. Tools: Apache Airflow for scheduling, MLflow for model
# > versioning, AWS S3 for storage.
