# =============================================================================
# notebooks/01_business_understanding.py
# Phase 1 — Business Understanding
# Convert to .ipynb or run cell by cell in Jupyter
# =============================================================================

# %% [markdown]
# # Phase 1: Business Understanding
# **Project:** AI-Powered Customer Churn Prediction & BI Dashboard
# **Dataset:** IBM Telco Customer Churn
# **Author:** [Your Name]
# **Date:** 2025

# %% [markdown]
# ## 1. Problem Statement
# TelCo Inc. experiences significant recurring revenue loss due to customer churn.
# The company currently lacks a proactive, data-driven mechanism to identify
# at-risk customers before they leave. This project builds an ML-powered churn
# prediction system + Power BI dashboard to enable retention-focused action.

# %% [markdown]
# ## 2. Business Questions This Project Answers
BUSINESS_QUESTIONS = [
    "What is the current overall churn rate?",
    "Which contract type has the highest churn?",
    "Does payment method influence churn?",
    "How does tenure correlate with churn likelihood?",
    "Which customer segments are highest risk?",
    "How much monthly revenue is lost to churn?",
    "What is the Customer Lifetime Value of churned vs retained customers?",
    "Which customers should the retention team contact TODAY?"
]

for i, q in enumerate(BUSINESS_QUESTIONS, 1):
    print(f"Q{i}: {q}")

# %% [markdown]
# ## 3. KPI Definitions
KPI_TABLE = {
    "Churn Rate"        : "% of customers who left in a period | Target: < 15%",
    "Retention Rate"    : "% of customers retained            | Target: > 85%",
    "Revenue Lost"      : "Monthly revenue from churned customers | Minimize",
    "CLV"               : "Avg monthly charge × avg tenure        | Maximize",
    "High Risk Count"   : "Customers with churn probability > 70% | Track weekly",
    "Model ROC-AUC"     : "Model discrimination ability            | Target ≥ 0.85",
    "Model Recall"      : "% of actual churners caught             | Target ≥ 0.80",
}

print("\n── KPI Definitions ──")
for kpi, definition in KPI_TABLE.items():
    print(f"  {kpi:<20} : {definition}")

# %% [markdown]
# ## 4. Financial Impact Estimate
TOTAL_CUSTOMERS    = 7_043
AVG_MONTHLY_CHARGE = 65.0      # USD — approximate from dataset
CHURN_RATE         = 0.2655    # 26.55% — actual IBM Telco rate

churned_customers      = int(TOTAL_CUSTOMERS * CHURN_RATE)
monthly_revenue_at_risk = churned_customers * AVG_MONTHLY_CHARGE
annual_revenue_at_risk  = monthly_revenue_at_risk * 12
savings_10pct           = annual_revenue_at_risk * 0.10

print(f"\n── Financial Impact ──")
print(f"  Total Customers        : {TOTAL_CUSTOMERS:,}")
print(f"  Estimated Churned      : {churned_customers:,}")
print(f"  Monthly Revenue Lost   : ${monthly_revenue_at_risk:,.0f}")
print(f"  Annual Revenue Lost    : ${annual_revenue_at_risk:,.0f}")
print(f"  Savings (10% reduction): ${savings_10pct:,.0f} / year")

# %% [markdown]
# ## 5. Why Recall Over Precision?
# In churn prediction, the cost of a FALSE NEGATIVE (missing a churner)
# far exceeds the cost of a FALSE POSITIVE (incorrectly flagging a loyal customer).
# → Missing a churner = losing the customer forever (high cost)
# → False alarm       = unnecessary retention offer (low cost)
# Therefore we OPTIMISE FOR RECALL, not raw accuracy.
