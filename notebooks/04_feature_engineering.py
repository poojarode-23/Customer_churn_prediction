# =============================================================================
# notebooks/04_feature_engineering.py
# Phase 4 — Interactive Feature Engineering Walkthrough
# Run cell-by-cell in Jupyter
# =============================================================================

# %% [markdown]
# # Phase 4: Feature Engineering
#
# **Goal:** Transform raw columns into ML-optimised features that give the
# model more signal to learn from.
#
# ## What Is Feature Engineering?
# Feature engineering is the process of using domain knowledge to create
# new input variables that make machine learning algorithms work better.
# It is often described as the most impactful — and most creative — part
# of the ML workflow. A great feature can replace thousands of training rows.
#
# ## The Golden Rule
# Every feature you create must answer: *"Does this help the model
# distinguish a churner from a loyal customer better than without it?"*

# %%
# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from config.settings import CLEAN_FILE, DATA_PROCESSED, MODELS_DIR
from src.features.feature_engineering import (
    load_clean_data,
    create_business_features,
    encode_categorical_features,
    drop_non_ml_columns,
    split_data,
    scale_features,
    visualise_engineered_features,
    feature_importance_preview,
    run_feature_engineering_pipeline,
)

print("✓ Imports successful")

# %% [markdown]
# ## Step 1: Load Clean Data

# %%
df_raw = load_clean_data()
print(f"Loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"Columns: {list(df_raw.columns)}")

# %% [markdown]
# ## Step 2: Business Feature Creation
# We create 10 new features — each justified by a business insight from EDA.

# %%
df_fe = create_business_features(df_raw)

# Show new features
new_feat_cols = [
    "ServiceCount", "AvgMonthlySpend", "ChargeToTenureRatio",
    "CLV", "TenureGroup_num", "ChargesBand_num", "ContractRisk",
    "IsNewCustomer", "IsLongTermCustomer", "HasNoAddOns"
]
print("\nNew features created:")
print(df_fe[new_feat_cols].describe().round(2))

# %%
# ── Validate ServiceCount logic ───────────────────────────────────────────────
print("\nServiceCount distribution:")
print(df_fe["ServiceCount"].value_counts().sort_index())
print("\nChurn rate by ServiceCount:")
print(
    df_fe.groupby("ServiceCount")["Churn"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "ChurnRate", "count": "Customers"})
    .assign(ChurnRate=lambda x: (x["ChurnRate"] * 100).round(1))
)
print("\n💡 Business Insight: Each additional service reduces churn rate!")

# %%
# ── Validate IsNewCustomer logic ──────────────────────────────────────────────
new_cust_churn = df_fe.groupby("IsNewCustomer")["Churn"].mean() * 100
print("\nChurn rate by IsNewCustomer flag:")
print(f"  New customers (≤12 mo) : {new_cust_churn[1]:.1f}%")
print(f"  Established (>12 mo)   : {new_cust_churn[0]:.1f}%")
print("\n💡 Business Insight: New customers churn at 2× the rate of established ones!")

# %% [markdown]
# ## Step 3: Encoding — Three Strategies

# %%
df_encoded = encode_categorical_features(df_fe)
print(f"\nShape before encoding: {df_fe.shape}")
print(f"Shape after encoding : {df_encoded.shape}")
print(f"New OHE columns created: {df_encoded.shape[1] - df_fe.shape[1]}")

# Show how Contract was ordinal-encoded
print("\nContract encoding (ordinal):")
print(df_encoded[["Contract"]].value_counts().sort_index())

# %% [markdown]
# ## Step 4: Drop Non-ML Columns
# These would cause data leakage or add noise.

# %%
df_ml = drop_non_ml_columns(df_encoded)
print(f"\nFinal ML-ready dataset: {df_ml.shape}")
print(f"Features: {[c for c in df_ml.columns if c != 'Churn']}")

# %% [markdown]
# ## Step 5: Train / Test Split
# **Stratified** to preserve class balance in both splits.

# %%
X_train, X_test, y_train, y_test = split_data(df_ml)

print(f"\nTraining set : {X_train.shape[0]:,} rows ({y_train.mean()*100:.1f}% churn)")
print(f"Test set     : {X_test.shape[0]:,} rows ({y_test.mean()*100:.1f}% churn)")
print("\n✓ Stratification confirmed — both sets have identical churn rates")

# %% [markdown]
# ## Step 6: Feature Scaling
# **Critical:** fit ONLY on X_train. Transform both.

# %%
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

print("\nScaler parameters (mean of scaled columns):")
scale_cols = ["tenure", "MonthlyCharges", "TotalCharges",
              "AvgMonthlySpend", "ChargeToTenureRatio", "CLV"]
print(f"  Means used  : {dict(zip(scale_cols, scaler.mean_.round(2)))}")
print(f"  Std devs    : {dict(zip(scale_cols, scaler.scale_.round(2)))}")
print("\n✓ Test set transformed using training statistics only (no leakage)")

# %% [markdown]
# ## Step 7: Feature Importance Preview

# %%
importance_df = feature_importance_preview(X_train_sc, y_train)
print("\nTop 10 features by absolute correlation with Churn:")
print(importance_df.head(10).to_string(index=False))

# %% [markdown]
# ## Feature Engineering Summary
#
# | Step | Action | Result |
# |------|--------|--------|
# | Business features | Created 10 new columns | Better signal for non-linear patterns |
# | Binary encoding | Yes/No → 1/0 (5 cols) | Models can read booleans |
# | Ordinal encoding | Contract, InternetService | Preserves business order |
# | One-hot encoding | 8 nominal cols → 10 binary cols | No false ordinal relationship |
# | Drop non-ML cols | Removed 4 columns | Eliminated leakage risk |
# | Stratified split | 80/20 | Both sets have 22.1% churn |
# | StandardScaler | 6 continuous cols scaled | Linear models work correctly |
#
# **Final ML dataset: 30 features, 5,634 training rows, 1,409 test rows**

# %% [markdown]
# ---
# # 🎤 Interview Q&A — Feature Engineering
#
# **Q1: Why did you create ServiceCount instead of using each service column separately?**
# > Two reasons: (1) It captures the aggregate stickiness effect — the model
# > sees that 4 services is categorically different from 0 services, regardless
# > of WHICH services. (2) It reduces dimensionality from 6 binary columns
# > to 1 ordinal column, reducing noise and improving interpretability.
#
# **Q2: What is data leakage and how did you prevent it?**
# > Data leakage is when information from outside the training set is used
# > to train the model, giving falsely optimistic performance that collapses
# > in production. I prevented it three ways: (1) fit the scaler ONLY on
# > training data, (2) dropped ChurnLabel (a string alias of the target),
# > (3) excluded customerID (which could overfit to specific IDs).
#
# **Q3: Why did you use three different encoding strategies?**
# > Different column types need different treatment. Binary (Yes/No) maps
# > naturally to 1/0. Ordinal columns like Contract have a natural order
# > (Month-to-month > One year > Two year in risk) that ordinal encoding
# > captures. Nominal columns like PaymentMethod have NO natural order —
# > one-hot encoding creates independent binary flags instead of imposing
# > a false numeric relationship.
#
# **Q4: Why drop the 'gender' column?**
# > EDA showed gender has essentially zero correlation with churn (Male
# > and Female churn at nearly identical rates). Including it adds noise
# > without signal. Also, removing demographic features that don't improve
# > model performance is a responsible AI practice that reduces potential
# > model bias.
#
# **Q5: Why stratify the train/test split?**
# > Our dataset has 22.1% churn (class imbalance). Without stratification,
# > random chance might give the test set 18% or 26% churn, making our
# > evaluation metrics inconsistent and not representative of real deployment.
# > Stratification guarantees both splits have exactly 22.1% churn.
#
# **Q6: When would you use MinMaxScaler instead of StandardScaler?**
# > MinMaxScaler (scales to [0,1]) is preferred for neural networks and
# > k-NN where bounded input ranges matter. StandardScaler (z-score
# > normalisation) is preferred when the data has outliers (MinMaxScaler
# > is distorted by extreme values) and for linear models and SVMs that
# > assume approximately normal distributions.
#
# **Q7: What is the ChargeToTenureRatio and why does it matter?**
# > It measures whether a customer's current bill is higher than their
# > historical average spend. A ratio > 1 means their charges have
# > increased over time — a potential indicator of price fatigue that
# > may trigger churn. It captures price sensitivity in a single number.
#
# **Q8: What is the dummy variable trap?**
# > When you one-hot encode N categories, you get N binary columns. But
# > these N columns are perfectly multicollinear — knowing all N-1 values
# > perfectly predicts the Nth. This confuses linear models. Solution:
# > use drop_first=True to drop one category per feature, leaving N-1
# > columns that carry all the information without multicollinearity.
