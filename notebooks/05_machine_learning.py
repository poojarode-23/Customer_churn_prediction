# =============================================================================
# notebooks/05_machine_learning.py
# Phase 5 — Interactive ML Training Walkthrough
# Run cell-by-cell in Jupyter
# =============================================================================

# %% [markdown]
# # Phase 5: Machine Learning — Model Training & Evaluation
#
# **Goal:** Train, compare, and select the best churn prediction model.
#
# ## The ML Workflow
# ```
# Load Data → Define Models → Train + Cross-Validate
#           → Evaluate on Test → Compare → Select Best → Save
# ```
#
# ## Why Multiple Models?
# Different algorithms have different strengths:
# - **Logistic Regression:** Linear, interpretable, fast — great baseline
# - **Decision Tree:** Visual rules, zero black-box — good for stakeholder demos
# - **Random Forest:** Ensemble of 200 trees — robust, handles noise well
# - **Gradient Boosting:** Sequential error-correction — typically best AUC

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

from config.settings import DATA_PROCESSED, MODELS_DIR, EVAL_DIR, FIGURES_DIR
from src.models.train_models import (
    load_train_test_data, get_model_definitions,
    train_single_model, evaluate_on_test,
    plot_confusion_matrices, plot_roc_curves,
    plot_metrics_comparison, plot_feature_importance,
    plot_threshold_analysis, select_best_model,
    save_models_and_report, run_training_pipeline,
    get_sample_weights,
)
from sklearn.metrics import classification_report

print("✓ Imports successful")

# %% [markdown]
# ## Step 1: Load the Phase 4 Feature-Engineered Data

# %%
X_train, X_test, y_train, y_test = load_train_test_data()

print(f"Training : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"Test     : {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
print(f"\nClass balance (train): {y_train.value_counts().to_dict()}")
print(f"Class balance (test) : {y_test.value_counts().to_dict()}")

# %% [markdown]
# ## Step 2: Why Class Imbalance Matters
# 22% churn vs 78% no-churn. If a naive model always predicts "No Churn":
# - Accuracy = 78% (looks great!)
# - Recall   = 0%  (catches ZERO churners — useless)
# This is why accuracy alone is a TERRIBLE metric for imbalanced problems.

# %%
# Naive baseline — always predict "No Churn"
naive_accuracy = (y_test == 0).sum() / len(y_test)
naive_recall   = 0.0
print(f"Naive baseline accuracy : {naive_accuracy:.1%}  ← looks good!")
print(f"Naive baseline recall   : {naive_recall:.1%}    ← catches ZERO churners")
print("\nConclusion: Optimise for Recall + ROC-AUC, not Accuracy")

# %% [markdown]
# ## Step 3: Understanding class_weight='balanced'
# Instead of SMOTE (oversampling), we use class_weight='balanced'.
# This is equivalent but computationally cheaper and avoids synthetic data issues.

# %%
weights = get_sample_weights(y_train)
no_churn_weight = weights[y_train == 0][0]
churn_weight    = weights[y_train == 1][0]
print(f"No-Churn weight : {no_churn_weight:.3f}")
print(f"Churn weight    : {churn_weight:.3f}")
print(f"Ratio           : {churn_weight/no_churn_weight:.2f}× — churners weighted higher")

# %% [markdown]
# ## Step 4: Run the Full Training Pipeline

# %%
# This runs everything: train → CV → evaluate → visualise → select → save
results = run_training_pipeline()

# %% [markdown]
# ## Step 5: Deep-Dive into the Best Model

# %%
best_name  = results["best_name"]
best_model = results["trained_models"][best_name]
best_result= results["best_result"]

print(f"\nBest Model: {best_name}")
print("\nFull classification report:")
print(classification_report(
    results["y_test"],
    best_result["y_pred"],
    target_names=["No Churn", "Churned"]
))

# %% [markdown]
# ## Step 6: Business Interpretation of Results

# %%
y_prob_best = best_result["y_prob"]
y_pred_best = best_result["y_pred"]
y_test_eval = results["y_test"]

tp = int(((y_pred_best == 1) & (y_test_eval == 1)).sum())
fn = int(((y_pred_best == 0) & (y_test_eval == 1)).sum())
fp = int(((y_pred_best == 1) & (y_test_eval == 0)).sum())
tn = int(((y_pred_best == 0) & (y_test_eval == 0)).sum())

# Estimate revenue impact
avg_monthly_charge = 68.02   # From EDA
avg_tenure         = 32      # Estimated months remaining
est_clv_per_churner = avg_monthly_charge * avg_tenure

revenue_saved     = tp * est_clv_per_churner * 0.30   # Assume 30% retention success rate
wasted_on_fp      = fp * 25                            # ~$25 per false retention call

print("=" * 55)
print("  BUSINESS IMPACT ANALYSIS")
print("=" * 55)
print(f"  True Positives (caught churners)  : {tp:>5,}")
print(f"  False Negatives (missed churners) : {fn:>5,}  ← COSTLY")
print(f"  False Positives (false alarms)    : {fp:>5,}  ← low cost")
print(f"  True Negatives (correctly loyal)  : {tn:>5,}")
print()
print(f"  Estimated CLV per churner         : ${est_clv_per_churner:,.0f}")
print(f"  Revenue saved (30% retention rate): ${revenue_saved:,.0f}")
print(f"  Cost of false alarm calls         : ${wasted_on_fp:,.0f}")
print(f"  Net business value (test batch)   : ${revenue_saved - wasted_on_fp:,.0f}")
print("=" * 55)

# %% [markdown]
# ## Step 7: Load and Inspect Saved Model

# %%
# Demonstrate loading the saved model (for production use)
loaded_model = joblib.load(MODELS_DIR / "best_model.pkl")
loaded_name  = joblib.load(MODELS_DIR / "best_model_name.pkl")
print(f"Loaded model: {loaded_name}")
print(f"Model type  : {type(loaded_model).__name__}")

# Quick sanity check — predictions should match
y_prob_loaded = loaded_model.predict_proba(X_test)[:, 1]
assert (y_prob_loaded.round(6) == best_result["y_prob"].round(6)).all()
print("✓ Loaded model produces identical predictions — artefact integrity confirmed")

# %% [markdown]
# ## Step 8: Compare All Models

# %%
print("\nFull Model Comparison:")
print(results["comparison_df"].to_string())

# %% [markdown]
# ## Step 9: Decision Tree Rules (Explainability)
# One unique advantage of Decision Tree — you can print the actual rules.

# %%
from sklearn.tree import export_text
dt_model = results["trained_models"]["Decision Tree"]
feature_names = results["feature_names"]

tree_rules = export_text(dt_model, feature_names=feature_names, max_depth=3)
print("Decision Tree Rules (depth ≤ 3):")
print(tree_rules)
print("\n💡 These rules are what the model 'learned'. You can share this with business stakeholders.")

# %% [markdown]
# ---
# # 🎤 Top 15 Interview Questions — Machine Learning
#
# **Q1: Why did you choose ROC-AUC as your primary evaluation metric?**
# > ROC-AUC measures a model's ability to discriminate between churners
# > and non-churners across ALL possible thresholds. Unlike accuracy, it's
# > not affected by class imbalance. A score of 0.73+ means the model
# > correctly ranks a random churner above a random non-churner 73% of the time.
#
# **Q2: Why did you set the threshold at 0.40 instead of 0.50?**
# > Business decision: the cost of missing a churner (losing the customer
# > forever) far exceeds the cost of a false alarm (unnecessary retention offer).
# > Lowering the threshold from 0.50 to 0.40 increases recall from ~75% to
# > ~85%, catching 10% more real churners. The trade-off (more false alarms)
# > is acceptable given the cost asymmetry.
#
# **Q3: What is cross-validation and why use StratifiedKFold?**
# > Cross-validation trains and evaluates the model on 5 different
# > train/validation splits, then averages the scores. This gives a much
# > more reliable performance estimate than a single split. StratifiedKFold
# > ensures each fold has the same 22% churn ratio, preventing any fold from
# > being unrepresentative due to random chance.
#
# **Q4: What is the difference between Bagging (Random Forest) and Boosting (GradientBoosting)?**
# > Bagging trains N trees IN PARALLEL on random subsets of data and averages
# > predictions. Each tree is independent — it reduces variance (overfitting).
# > Boosting trains N trees SEQUENTIALLY where each tree corrects the errors
# > of the previous. It reduces bias (underfitting). Boosting typically achieves
# > higher accuracy but is more prone to overfitting on noisy data.
#
# **Q5: Why did Random Forest win over Gradient Boosting on this dataset?**
# > Our weighted selection score (0.40×AUC + 0.40×Recall + 0.20×F1) favoured
# > Random Forest slightly because it achieved the best Recall (84.6%) while
# > maintaining competitive AUC. Gradient Boosting had the best raw AUC (0.735)
# > but lower recall at our threshold. With more hyperparameter tuning, Gradient
# > Boosting would likely win — it usually does on tabular data.
#
# **Q6: What is class_weight='balanced' and how does it differ from SMOTE?**
# > class_weight='balanced' multiplies the loss function for minority class
# > samples by a factor inversely proportional to class frequency. SMOTE
# > actually creates synthetic minority class samples via interpolation.
# > class_weight is simpler, faster, and doesn't risk introducing noise from
# > synthetic samples. For tree-based models on tabular data, class_weight
# > typically performs equally well or better than SMOTE.
#
# **Q7: What is overfitting and how did you prevent it?**
# > Overfitting is when a model memorises training data but fails on new data.
# > I prevented it with: (1) max_depth limits on trees, (2) min_samples_leaf
# > constraints, (3) max_features='sqrt' in Random Forest (feature randomness),
# > (4) subsample=0.8 in Gradient Boosting, (5) cross-validation to detect
# > large train/test gaps, and (6) evaluating on a held-out test set.
#
# **Q8: Your model has 85% recall but only 50% F1. Is that good?**
# > In context, yes. F1 is the harmonic mean of Precision (0.35) and
# > Recall (0.85). The low precision means ~65% of flagged "churners" are
# > actually loyal customers (false alarms). However, given our business
# > objective — proactive retention outreach — contacting 65% false positives
# > is acceptable because: (1) the retention offer won't hurt a loyal customer,
# > (2) catching 85% of real churners justifies the cost.
#
# **Q9: What is feature importance in a Random Forest?**
# > Feature importance = the total reduction in Gini impurity (or MSE) that
# > each feature contributes across all trees, normalised to sum to 1.
# > High importance means the model splits on that feature often and
# > significantly — it's a strong predictor. In our model, tenure-related
# > features and contract type dominate, consistent with EDA findings.
#
# **Q10: How would you improve model performance further?**
# > (1) Hyperparameter tuning with GridSearchCV/RandomizedSearchCV,
# > (2) Try XGBoost or LightGBM (usually best on tabular data),
# > (3) Engineer interaction features (e.g. tenure × Contract),
# > (4) Try threshold tuning based on cost-benefit analysis,
# > (5) Collect additional features (support calls, NPS, usage data),
# > (6) Use Optuna or Bayesian optimization for automated HP search.
#
# **Q11: What is the dummy variable trap and did it affect your model?**
# > The dummy variable trap occurs when one-hot encoded columns are
# > perfectly multicollinear (the dropped column is predictable from others).
# > I used drop_first=True in pd.get_dummies() to avoid this. Tree-based
# > models are unaffected by multicollinearity, but Logistic Regression
# > would produce unstable coefficients without this fix.
#
# **Q12: Why does Logistic Regression have the lowest accuracy but highest recall?**
# > Logistic Regression with balanced class weights produces well-calibrated
# > probability scores that spread across the full [0,1] range. This means
# > more customers cross the 0.40 threshold → higher recall but more false
# > alarms → lower accuracy. Tree models tend to produce more extreme
# > probabilities (near 0 or 1), making threshold effects smaller.
#
# **Q13: What is the ROC curve and what does the area under it mean?**
# > The ROC (Receiver Operating Characteristic) curve plots True Positive Rate
# > (Recall) on the Y-axis vs False Positive Rate on the X-axis at every
# > possible threshold. AUC=0.5 is random guessing (diagonal line).
# > AUC=1.0 is a perfect model. Our AUC of ~0.72-0.73 means the model
# > correctly ranks a random churner above a random non-churner 72-73% of the time.
#
# **Q14: How would you deploy this model in production?**
# > (1) Save model + scaler + feature_columns to a model registry (MLflow),
# > (2) Create a REST API endpoint (FastAPI/Flask) that accepts customer JSON,
# > (3) Apply the same feature engineering pipeline as training,
# > (4) Return churn probability + risk category,
# > (5) Schedule weekly batch scoring on all active customers,
# > (6) Monitor data drift (customer behaviour changes over time),
# > (7) Retrain quarterly with new data.
#
# **Q15: What is precision-recall tradeoff and how did you decide where to set it?**
# > At every threshold, increasing recall decreases precision and vice versa.
# > I chose threshold=0.40 based on business cost analysis: a missed churner
# > costs ~$2,000 in lost CLV; a false alarm costs ~$25 (one retention call).
# > The cost ratio (80×) justifies accepting high false alarms (low precision)
# > in exchange for maximum churner capture (high recall).
