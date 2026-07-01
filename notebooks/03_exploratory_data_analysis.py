# =============================================================================
# notebooks/03_exploratory_data_analysis.py
# Phase 3 — Interactive EDA Walkthrough
# Run cell-by-cell in Jupyter for a guided experience with explanations
# =============================================================================

# %% [markdown]
# # Phase 3: Exploratory Data Analysis (EDA)
#
# **Goal:** Understand the dataset deeply before modelling.
# Answer the question: *"What does this data tell us about customer churn?"*
#
# ## Why EDA Is Non-Negotiable
# - Reveals hidden patterns and relationships
# - Identifies problems that would corrupt ML models
# - Generates business insights without any ML at all
# - Tells you WHICH features will be most predictive
# - Industry rule: "An hour of EDA saves a day of debugging"

# %%
# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from config.settings import CLEAN_FILE, FIGURES_DIR
from src.analysis.eda import run_eda_pipeline, section1_dataset_overview, \
    section2_target_analysis, section3_numerical_analysis, \
    section4_categorical_analysis, section5_bivariate_analysis, \
    section6_correlation_analysis, section7_outlier_analysis, \
    section8_business_insights, section9_powerbi_export

print("✓ Imports successful")

# %%
# ── Cell 2: Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(CLEAN_FILE)
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head(3)

# %% [markdown]
# ## Section 1: Dataset Overview
# First rule of data science: **look at your data before touching it.**

# %%
# ── Cell 3: Overview ──────────────────────────────────────────────────────────
overview = section1_dataset_overview(df)

# %% [markdown]
# ## Section 2: Target Variable — Churn
# Before ANY analysis, understand what you're predicting.

# %%
# ── Cell 4: Target analysis ───────────────────────────────────────────────────
section2_target_analysis(df)
# Chart saved to: reports/figures/churn_distribution.png

# %% [markdown]
# ## Section 3: Numerical Features
# **tenure**, **MonthlyCharges**, **TotalCharges**
#
# For each we look at:
# - How values are distributed (histogram + KDE)
# - Whether churners and non-churners look different (box + violin)

# %%
# ── Cell 5: Numerical analysis ────────────────────────────────────────────────
num_stats = section3_numerical_analysis(df)
print("\nSummary Statistics:")
print(num_stats)
# Charts: tenure_analysis.png, monthlycharges_analysis.png, totalcharges_analysis.png

# %% [markdown]
# ## Section 4: Categorical Features
# 16 categorical features analysed — each with count plot and churn rate bar.

# %%
# ── Cell 6: Categorical analysis ─────────────────────────────────────────────
cat_summary = section4_categorical_analysis(df)

# Show Contract churn rate table as example
print("\nContract Type Churn Rate Summary:")
print(cat_summary.get("Contract", "Contract not found"))

# %% [markdown]
# ## Section 5: Bivariate Analysis
# Focused deep-dives on the most business-critical feature-vs-churn pairs.

# %%
# ── Cell 7: Bivariate ────────────────────────────────────────────────────────
section5_bivariate_analysis(df)

# %% [markdown]
# ## Section 6: Correlation Analysis
# **Why only numerical columns?**
# Pearson correlation measures linear relationships between continuous variables.
# Categorical strings can't be used directly — we'll encode them in Phase 6.

# %%
# ── Cell 8: Correlation ──────────────────────────────────────────────────────
corr_matrix = section6_correlation_analysis(df)
print("\nCorrelation matrix:")
print(corr_matrix.round(3))

# %% [markdown]
# ## Section 7: Outlier Analysis
# We DETECT outliers but do NOT remove them (business decision).

# %%
# ── Cell 9: Outlier report ────────────────────────────────────────────────────
outlier_rpt = section7_outlier_analysis(df)
print(outlier_rpt)

# %% [markdown]
# ## Section 8: Business Insights
# 20 actionable insights derived from the EDA above.

# %%
# ── Cell 10: Insights ────────────────────────────────────────────────────────
insights = section8_business_insights(df)
print(f"\nTotal insights generated: {len(insights)}")

# %% [markdown]
# ## Section 9: Power BI Export
# Creates a Power BI-optimised CSV with extra slicer columns.

# %%
# ── Cell 11: Power BI export ──────────────────────────────────────────────────
pbi_df = section9_powerbi_export(df)
print("\nPower BI DataFrame columns:")
print(list(pbi_df.columns))

# %%
# ── Cell 12: View all generated figures ───────────────────────────────────────
import os
figures = sorted(os.listdir(FIGURES_DIR))
print(f"\n{len(figures)} figures saved to {FIGURES_DIR}:\n")
for f in figures:
    size_kb = os.path.getsize(FIGURES_DIR / f) // 1024
    print(f"  {f:<45} {size_kb:>5} KB")

# %% [markdown]
# ---
# # 🎤 Top 20 Interview Questions & Answers
# These are the exact questions interviewers ask about EDA.
#
# **Q1: What is EDA and why do you do it before modelling?**
# > EDA is the process of exploring data using statistical summaries and
# > visualisations to discover patterns, anomalies, and relationships.
# > You do it first because: (1) it reveals data quality issues that corrupt
# > models, (2) it tells you which features are most predictive, (3) it
# > generates business insights without any ML.
#
# **Q2: What did you find in the IBM Telco EDA?**
# > Three critical findings: (1) 22% churn rate above industry benchmark,
# > (2) Month-to-month contracts drive 42%+ churn vs 3% for 2-year,
# > (3) First 12 months is the highest-risk window for churn.
#
# **Q3: Why did you not remove outliers?**
# > Because outliers here are business-relevant: a customer with $8,000
# > total charges is a high-value long-term customer, not a data error.
# > Removing them would bias the model away from the most valuable segment.
#
# **Q4: What is class imbalance and why does it matter for churn?**
# > Class imbalance means one class (No-Churn: 78%) vastly outnumbers the
# > other (Churn: 22%). A model that always predicts "No Churn" would get
# > 78% accuracy — but catch zero actual churners. We need SMOTE to
# > balance classes and optimise for recall, not accuracy.
#
# **Q5: What is the correlation between tenure and churn?**
# > Negative correlation (~-0.35). Longer-tenure customers churn less.
# > This is the strongest numerical predictor in the dataset.
#
# **Q6: Why does fiber optic internet have higher churn than DSL?**
# > Two possible reasons: (1) fiber customers pay higher prices so
# > price-sensitivity is higher, (2) fiber may have service quality issues.
# > Either way, the business should investigate with a customer survey.
#
# **Q7: What is a KDE plot and why use it alongside a histogram?**
# > KDE (Kernel Density Estimate) is a smoothed probability density curve.
# > Histograms show absolute counts (binning-dependent), KDE shows the
# > underlying distribution shape (continuous). Together they're more
# > informative than either alone.
#
# **Q8: Why did you use a heatmap for correlation?**
# > Heatmaps make correlation matrices instantly readable. The colour
# > gradient immediately shows strong/weak positive/negative relationships
# > without reading every number. Essential for stakeholder presentations.
#
# **Q9: What does skewness tell you about TotalCharges?**
# > Skewness of 0.89 means right-skewed: most customers have low-to-medium
# > total charges (shorter relationships) with a long tail of high-value
# > long-term customers. Tree-based models handle skew well; linear models
# > may benefit from a log-transform.
#
# **Q10: Which feature do you think will be most important for ML?**
# > Contract type — it has the largest separation in churn rates (42% vs 3%).
# > Followed by tenure, internet service type, and payment method.
# > Feature importance from the ML model will confirm this in Phase 7.
#
# **Q11: What is the difference between bivariate and multivariate analysis?**
# > Bivariate examines two variables at a time (e.g. Contract vs Churn).
# > Multivariate examines three or more simultaneously (e.g. Contract +
# > Tenure + Churn). Multivariate is done in Feature Engineering and ML.
#
# **Q12: Why use violin plots instead of boxplots?**
# > Boxplots show median and IQR but hide the distribution shape.
# > Violin plots show the full probability distribution via KDE. When
# > distributions are bimodal (two peaks), boxplots completely miss it.
#
# **Q13: How did you handle categorical variables in correlation analysis?**
# > Pearson correlation only works on continuous numerical variables. For
# > categorical vs binary target, I used visual analysis (churn rates) in
# > this phase. In Phase 6 (Feature Engineering), I'll encode them and
# > compute point-biserial correlations with the target.
#
# **Q14: What is the business significance of the first 12 months insight?**
# > It means the first year is a critical retention window. If a customer
# > survives past 12 months, their churn probability drops significantly.
# > This justifies a dedicated onboarding programme with proactive outreach
# > in months 1–6 of the customer relationship.
#
# **Q15: Why does gender have no significant impact on churn?**
# > The churn rates for Male and Female are nearly identical (~22%).
# > This means gender is a poor predictor and gender-targeted retention
# > campaigns would waste budget. Resources are better allocated to
# > contract-type and tenure-based interventions.
#
# **Q16: What is your interpretation of the MonthlyCharges correlation with churn?**
# > Positive correlation (+0.19): higher monthly charges slightly increase
# > churn probability. This suggests a price-value perception issue.
# > Churned customers pay more but may feel they get less value. A pricing
# > audit and customer satisfaction survey are recommended.
#
# **Q17: How would you communicate EDA findings to a non-technical stakeholder?**
# > I'd use the Power BI dashboard: simple KPI cards, colour-coded charts,
# > and a one-page insight summary. Key message: "3 things drive 80% of
# > our churn — contract type, payment method, and first-year experience."
# > No equations, no p-values; just: what it means and what to do.
#
# **Q18: What additional data would improve this EDA?**
# > (1) Customer support call logs — complaint frequency before churn,
# > (2) Service outage history per customer,
# > (3) Competitor pricing data,
# > (4) Customer satisfaction scores (NPS/CSAT),
# > (5) Date of contract start/end for time-series churn analysis.
#
# **Q19: What is a count plot and when do you use it?**
# > A count plot shows the frequency of each category in a column —
# > essentially a bar chart for categorical data. Use it when you want to
# > see the distribution of a categorical variable and how it relates to
# > the target (by adding hue=Churn).
#
# **Q20: What is the IQR method for outlier detection?**
# > IQR (Interquartile Range) = Q3 - Q1. Outlier bounds:
# > Lower = Q1 - 1.5×IQR, Upper = Q3 + 1.5×IQR.
# > Points outside these bounds are flagged as outliers.
# > It's robust to extreme values (unlike z-score) and doesn't assume
# > a normal distribution.
