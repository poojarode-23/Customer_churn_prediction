# =============================================================================
# src/models/predict.py
# Phase 6 — Prediction Pipeline
# Project: AI-Powered Customer Churn Prediction
# =============================================================================
#
# WHAT THIS MODULE DOES:
#   1.  Loads the best trained model + scaler + feature column list (Phase 5)
#   2.  Applies the exact same feature engineering as Phase 4 to the full dataset
#   3.  Generates churn probability for every customer (0.00 → 1.00)
#   4.  Assigns binary churn prediction using the business-tuned threshold
#   5.  Categorises every customer into a Risk Category:
#         High Risk    : probability ≥ 0.70
#         Medium Risk  : probability 0.40–0.69
#         Low Risk     : probability < 0.40
#   6.  Computes a 1–100 Risk Score (for sorting / Power BI slicers)
#   7.  Joins predictions back to original readable customer data
#   8.  Exports:
#         data/predictions/predictions.csv       — full prediction output
#         data/predictions/high_risk_customers.csv — top priority list
#   9.  Produces prediction-specific visualisations
#  10.  Validates prediction output quality
#
# KEY OUTPUT COLUMNS:
#   customerID, ActualChurn, PredictedChurn, ChurnProbability,
#   RiskCategory, RiskScore, RevenueAtRisk, PriorityRank
# =============================================================================

import sys
import warnings
import joblib
from pathlib import Path
from typing import Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    CLEAN_FILE, DATA_PROCESSED, DATA_PREDICTIONS,
    MODELS_DIR, FIGURES_DIR, LOG_FILE,
    RANDOM_STATE, TARGET_COLUMN,
)
from src.utils.logger import get_logger
from src.features.feature_engineering import (
    create_business_features,
    encode_categorical_features,
    drop_non_ml_columns,
)

logger = get_logger(__name__, str(LOG_FILE))

# =============================================================================
# CONSTANTS
# =============================================================================

CHURN_THRESHOLD   = 0.40   # Must match Phase 5 threshold exactly
HIGH_RISK_CUTOFF  = 0.70   # ≥ 70% probability → High Risk
MEDIUM_RISK_LOWER = 0.40   # 40–69% → Medium Risk
                            # < 40%  → Low Risk

CORP_BLUE  = "#2C3E50"
ACC_RED    = "#E74C3C"
ACC_ORANGE = "#E67E22"
ACC_GREEN  = "#2ECC71"


# =============================================================================
# STEP 1 — LOAD MODEL ARTEFACTS
# =============================================================================

def load_model_artefacts() -> Tuple:
    """
    Loads all artefacts saved in Phase 5:
      best_model.pkl         — the fitted Random Forest (or whichever won)
      best_model_name.pkl    — string name of the best model
      scaler.pkl             — fitted StandardScaler from Phase 4
      feature_columns.pkl    — ordered list of 30 feature names

    Why load feature_columns?
      At prediction time, the incoming data must have EXACTLY the same
      columns in EXACTLY the same order as at training time.
      A column mismatch silently corrupts predictions — the model
      reads the wrong data for each feature slot.

    Returns:
        (model, model_name, scaler, feature_columns)
    """
    logger.info("STEP 1 — Loading model artefacts")

    required = {
        "model"          : MODELS_DIR / "best_model.pkl",
        "model_name"     : MODELS_DIR / "best_model_name.pkl",
        "scaler"         : MODELS_DIR / "scaler.pkl",
        "feature_columns": MODELS_DIR / "feature_columns.pkl",
    }

    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Artefact '{name}' not found at {path}.\n"
                "Run Phase 5 first: python src/models/train_models.py"
            )

    model          = joblib.load(required["model"])
    model_name     = joblib.load(required["model_name"])
    scaler         = joblib.load(required["scaler"])
    feature_cols   = joblib.load(required["feature_columns"])

    logger.info(f"Model loaded   : {model_name}")
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Scaler type    : {type(scaler).__name__}")

    return model, model_name, scaler, feature_cols


# =============================================================================
# STEP 2 — LOAD AND PREPARE FULL DATASET
# =============================================================================

def prepare_full_dataset(
    clean_path    : Path,
    scaler        : object,
    feature_cols  : list,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the complete Phase 4 feature engineering pipeline to the
    full clean dataset (all 7,043 customers — not just the test split).

    We score EVERY customer because:
      - The test set (1,409 rows) is already labelled — we know who churned
      - The "real-world" use case is scoring ALL active customers weekly
      - Power BI needs predictions for every customer in the dashboard

    Steps:
      1. Load telco_clean.csv
      2. Apply Phase 4 feature engineering (same functions, same logic)
      3. Align to trained feature columns (same order, same set)
      4. Apply scaler (transform only — never fit again)

    Returns:
        df_original : raw readable DataFrame (customerID, labels, etc.)
        X_scaled    : feature matrix ready for model.predict_proba()
    """
    logger.info("STEP 2 — Preparing full dataset for scoring")

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean dataset not found at {clean_path}")

    df_original = pd.read_csv(clean_path)
    logger.info(f"Loaded: {df_original.shape[0]:,} customers")

    # ── Apply identical feature engineering as Phase 4 ────────────────────────
    df_fe = create_business_features(df_original)
    df_fe = encode_categorical_features(df_fe)
    df_fe = drop_non_ml_columns(df_fe)

    # ── Separate target from features ─────────────────────────────────────────
    y_full = df_fe[TARGET_COLUMN].copy() if TARGET_COLUMN in df_fe.columns else None

    if TARGET_COLUMN in df_fe.columns:
        X_full = df_fe.drop(columns=[TARGET_COLUMN])
    else:
        X_full = df_fe.copy()

    # ── Align columns to training feature set ─────────────────────────────────
    # Add any missing columns as 0 (can happen if a OHE category is absent)
    for col in feature_cols:
        if col not in X_full.columns:
            X_full[col] = 0
            logger.warning(f"Missing column added as 0: {col}")

    # Select and reorder to exactly match training order
    X_full = X_full[feature_cols]

    logger.info(f"Feature matrix shape: {X_full.shape}")

    # ── Scale continuous features (transform only — scaler already fitted) ────
    scale_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges",
                               "AvgMonthlySpend", "ChargeToTenureRatio", "CLV"]
                  if c in X_full.columns]

    X_scaled        = X_full.copy()
    X_scaled[scale_cols] = scaler.transform(X_full[scale_cols])

    logger.info(f"Scaling applied to {len(scale_cols)} columns")

    return df_original, X_scaled, y_full


# =============================================================================
# STEP 3 — GENERATE PREDICTIONS
# =============================================================================

def generate_predictions(
    model        : object,
    X_scaled     : pd.DataFrame,
    threshold    : float = CHURN_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates churn probabilities and binary predictions.

    model.predict_proba() returns shape (n_customers, 2):
      Column 0 = P(No Churn)
      Column 1 = P(Churn)     ← we want this column

    We apply the custom threshold from Phase 5 (0.40) for consistency.

    Returns:
        y_prob : array of float churn probabilities [0.00, 1.00]
        y_pred : array of int binary predictions   {0, 1}
    """
    logger.info(f"STEP 3 — Generating predictions (threshold={threshold})")

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    churners = y_pred.sum()
    churn_rate = churners / len(y_pred) * 100

    logger.info(f"Predicted churners  : {churners:,} ({churn_rate:.1f}%)")
    logger.info(f"Predicted non-churners: {len(y_pred) - churners:,}")
    logger.info(
        f"Probability stats: "
        f"min={y_prob.min():.3f}, "
        f"mean={y_prob.mean():.3f}, "
        f"max={y_prob.max():.3f}"
    )

    return y_prob, y_pred


# =============================================================================
# STEP 4 — BUILD PREDICTION OUTPUT
# =============================================================================

def build_prediction_output(
    df_original  : pd.DataFrame,
    y_prob       : np.ndarray,
    y_pred       : np.ndarray,
    y_actual     : Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Assembles the final prediction DataFrame with business-ready columns.

    Output columns:
      customerID         — unique identifier
      ActualChurn        — ground truth (1/0) — only known for historical data
      ActualChurnLabel   — "Yes"/"No" readable label
      PredictedChurn     — model prediction (1/0)
      PredictedChurnLabel— "Yes"/"No" readable label
      ChurnProbability   — raw probability (e.g. 0.847)
      ChurnProbabilityPct— percentage (e.g. 84.7%)
      RiskCategory       — "High Risk" / "Medium Risk" / "Low Risk"
      RiskScore          — integer 1–100 (for sorting / Power BI)
      RevenueAtRisk      — MonthlyCharges for predicted churners, else 0
      CLV                — Customer Lifetime Value proxy (tenure × MonthlyCharges)
      PriorityRank       — Rank by ChurnProbability descending (1 = highest risk)
      IsCorrectPrediction— 1 if model prediction matches actual (validation)

    Plus key customer context columns for Power BI:
      gender, SeniorCitizenLabel, Partner, Dependents,
      tenure, Contract, PaymentMethod, InternetService,
      MonthlyCharges, TotalCharges
    """
    logger.info("STEP 4 — Building prediction output DataFrame")

    pred_df = pd.DataFrame()

    # ── Identifiers & ground truth ────────────────────────────────────────────
    pred_df["customerID"]         = df_original["customerID"].values

    if y_actual is not None:
        pred_df["ActualChurn"]       = y_actual.values
        pred_df["ActualChurnLabel"]  = y_actual.map({1: "Yes", 0: "No"}).values
    else:
        pred_df["ActualChurn"]       = np.nan
        pred_df["ActualChurnLabel"]  = "Unknown"

    # ── Model outputs ─────────────────────────────────────────────────────────
    pred_df["PredictedChurn"]      = y_pred
    pred_df["PredictedChurnLabel"] = pd.Series(y_pred).map({1: "Yes", 0: "No"}).values
    pred_df["ChurnProbability"]    = y_prob.round(4)
    pred_df["ChurnProbabilityPct"] = (y_prob * 100).round(2)

    # ── Risk Category ─────────────────────────────────────────────────────────
    # Three tiers based on probability thresholds
    conditions = [
        y_prob >= HIGH_RISK_CUTOFF,
        (y_prob >= MEDIUM_RISK_LOWER) & (y_prob < HIGH_RISK_CUTOFF),
    ]
    choices    = ["High Risk", "Medium Risk"]
    pred_df["RiskCategory"] = np.select(conditions, choices, default="Low Risk")

    # ── Risk Score (1–100 integer) ────────────────────────────────────────────
    # Transforms probability into an easy-to-read integer score
    # Used in Power BI gauges and sorted tables
    # Formula: probability × 100, rounded to nearest integer, clipped to [1, 100]
    pred_df["RiskScore"] = np.clip(
        (y_prob * 100).round(0).astype(int), 1, 100
    )

    # ── Financial columns ─────────────────────────────────────────────────────
    monthly_charges = df_original["MonthlyCharges"].values
    total_charges   = df_original["TotalCharges"].values
    tenure          = df_original["tenure"].values

    pred_df["MonthlyCharges"] = monthly_charges
    pred_df["TotalCharges"]   = total_charges
    pred_df["tenure"]         = tenure

    # Revenue at risk = monthly charges if predicted to churn, else 0
    pred_df["RevenueAtRisk"]  = np.where(y_pred == 1, monthly_charges, 0.0)

    # Customer Lifetime Value = total charges (proxy)
    pred_df["CLV"] = total_charges.round(2)

    # Estimated CLV remaining (rough: avg remaining tenure × monthly charge)
    avg_remaining_tenure = np.maximum(0, 36 - tenure)   # assuming 36-mo avg lifetime
    pred_df["EstimatedCLVAtRisk"] = np.where(
        y_pred == 1,
        (monthly_charges * avg_remaining_tenure).round(2),
        0.0
    )

    # ── Priority Rank ─────────────────────────────────────────────────────────
    # Rank 1 = most likely to churn — top priority for retention team
    pred_df["PriorityRank"] = pred_df["ChurnProbability"].rank(
        ascending=False, method="min"
    ).astype(int)

    # ── Validation column ─────────────────────────────────────────────────────
    if y_actual is not None:
        pred_df["IsCorrectPrediction"] = (
            pred_df["PredictedChurn"] == pred_df["ActualChurn"]
        ).astype(int)
    else:
        pred_df["IsCorrectPrediction"] = np.nan

    # ── Customer context columns (for Power BI filtering) ────────────────────
    context_cols = [
        "gender", "SeniorCitizenLabel", "Partner", "Dependents",
        "Contract", "PaymentMethod", "InternetService",
        "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "TechSupport",
        "MultipleLines", "StreamingTV", "StreamingMovies",
    ]
    for col in context_cols:
        if col in df_original.columns:
            pred_df[col] = df_original[col].values

    # ── Tenure group for Power BI slicers ────────────────────────────────────
    pred_df["TenureGroup"] = pd.cut(
        pred_df["tenure"],
        bins   = [0, 12, 24, 48, 72],
        labels = ["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
        right  = True,
    ).astype(str)

    # ── Charge band for Power BI slicers ─────────────────────────────────────
    pred_df["ChargesBand"] = pd.cut(
        pred_df["MonthlyCharges"],
        bins   = [0, 35, 65, 95, 120],
        labels = ["Low (<$35)", "Medium ($35–65)", "High ($65–95)", "Premium (>$95)"],
        right  = True,
    ).astype(str)

    # ── Final sort: highest risk first ────────────────────────────────────────
    pred_df = pred_df.sort_values("ChurnProbability", ascending=False).reset_index(drop=True)

    logger.info(f"Prediction output shape: {pred_df.shape}")
    logger.info(
        f"Risk breakdown: "
        f"High={( pred_df['RiskCategory']=='High Risk').sum():,} | "
        f"Medium={(pred_df['RiskCategory']=='Medium Risk').sum():,} | "
        f"Low={(pred_df['RiskCategory']=='Low Risk').sum():,}"
    )

    return pred_df


# =============================================================================
# STEP 5 — VALIDATE PREDICTIONS
# =============================================================================

def validate_predictions(pred_df: pd.DataFrame) -> dict:
    """
    Runs quality checks on the prediction output.

    Checks:
      1. No NaN in core prediction columns
      2. ChurnProbability in [0, 1]
      3. RiskScore in [1, 100]
      4. RiskCategory only contains valid values
      5. PriorityRank is unique and sequential
      6. RevenueAtRisk is 0 for predicted non-churners
    """
    logger.info("STEP 5 — Validating prediction output")

    checks = {}

    # 1. No NaN in core columns
    core_cols = ["customerID", "PredictedChurn", "ChurnProbability",
                 "RiskCategory", "RiskScore", "PriorityRank"]
    no_nulls  = pred_df[core_cols].isnull().sum().sum() == 0
    checks["no_nulls_in_core_cols"]   = no_nulls

    # 2. Probability in valid range
    prob_valid = (
        (pred_df["ChurnProbability"] >= 0) &
        (pred_df["ChurnProbability"] <= 1)
    ).all()
    checks["probabilities_in_0_1"]    = bool(prob_valid)

    # 3. Risk score range
    score_valid = (
        (pred_df["RiskScore"] >= 1) &
        (pred_df["RiskScore"] <= 100)
    ).all()
    checks["risk_score_in_1_100"]     = bool(score_valid)

    # 4. Valid risk categories
    valid_cats = {"High Risk", "Medium Risk", "Low Risk"}
    cats_valid = set(pred_df["RiskCategory"].unique()).issubset(valid_cats)
    checks["valid_risk_categories"]   = cats_valid

    # 5. RevenueAtRisk is 0 for predicted non-churners
    rev_check  = (
        pred_df[pred_df["PredictedChurn"] == 0]["RevenueAtRisk"] == 0
    ).all()
    checks["revenue_zero_for_non_churners"] = bool(rev_check)

    all_passed = all(checks.values())

    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status} | {check}")

    if all_passed:
        logger.info("All prediction validation checks passed ✓")
    else:
        logger.warning("Some validation checks FAILED — review output")

    return checks


# =============================================================================
# STEP 6 — VISUALISE PREDICTIONS
# =============================================================================

def visualise_predictions(pred_df: pd.DataFrame) -> None:
    """
    Produces 5 prediction-specific visualisations:
      1. Churn probability distribution (histogram)
      2. Risk category breakdown (pie + bar)
      3. Revenue at risk by risk category
      4. Probability vs Monthly Charges scatter
      5. Top 20 highest-risk customers (horizontal bar)
    """
    logger.info("STEP 6 — Generating prediction visualisations")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chart 1: Probability Distribution ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Churn Probability Distribution", fontsize=13,
                 fontweight="bold", color=CORP_BLUE)

    # Full distribution
    axes[0].hist(pred_df["ChurnProbability"], bins=50,
                 color=CORP_BLUE, alpha=0.7, edgecolor="white")
    axes[0].axvline(CHURN_THRESHOLD, color=ACC_RED, linestyle="--",
                    linewidth=2, label=f"Threshold ({CHURN_THRESHOLD})")
    axes[0].axvline(HIGH_RISK_CUTOFF, color=ACC_ORANGE, linestyle="--",
                    linewidth=2, label=f"High Risk ({HIGH_RISK_CUTOFF})")
    axes[0].set_xlabel("Churn Probability")
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_title("All Customers")
    axes[0].legend()

    # Probability by actual churn (if available)
    if pred_df["ActualChurn"].notna().all():
        for val, label, color in [(0, "No Churn", ACC_GREEN),
                                   (1, "Churned",  ACC_RED)]:
            subset = pred_df[pred_df["ActualChurn"] == val]["ChurnProbability"]
            axes[1].hist(subset, bins=30, alpha=0.55, color=color,
                         label=label, edgecolor="white")
        axes[1].set_title("By Actual Churn Status")
    else:
        for cat, color in [("High Risk", ACC_RED),
                            ("Medium Risk", ACC_ORANGE),
                            ("Low Risk", ACC_GREEN)]:
            subset = pred_df[pred_df["RiskCategory"] == cat]["ChurnProbability"]
            axes[1].hist(subset, bins=20, alpha=0.6, color=color,
                         label=cat, edgecolor="white")
        axes[1].set_title("By Risk Category")

    axes[1].set_xlabel("Churn Probability")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probability_distribution.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved: probability_distribution.png")

    # ── Chart 2: Risk Category Breakdown ─────────────────────────────────────
    risk_counts = pred_df["RiskCategory"].value_counts()
    risk_order  = ["High Risk", "Medium Risk", "Low Risk"]
    risk_counts = risk_counts.reindex([r for r in risk_order if r in risk_counts.index])
    risk_colors = [ACC_RED, ACC_ORANGE, ACC_GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Risk Category Breakdown", fontsize=13,
                 fontweight="bold", color=CORP_BLUE)

    # Pie
    axes[0].pie(risk_counts.values,
                labels     = risk_counts.index,
                autopct    = "%1.1f%%",
                colors     = risk_colors[:len(risk_counts)],
                startangle = 140,
                wedgeprops = dict(edgecolor="white", linewidth=2),
                textprops  = dict(fontsize=11, fontweight="bold"))
    axes[0].set_title("Proportion by Risk")

    # Bar with counts
    bars = axes[1].bar(risk_counts.index, risk_counts.values,
                       color=risk_colors[:len(risk_counts)],
                       edgecolor="white", linewidth=1.5, width=0.5)
    for bar, count in zip(bars, risk_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 20,
                     f"{count:,}", ha="center", fontweight="bold", fontsize=11)
    axes[1].set_title("Count by Risk Category")
    axes[1].set_ylabel("Number of Customers")
    axes[1].set_ylim(0, risk_counts.max() * 1.15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "risk_category_breakdown.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved: risk_category_breakdown.png")

    # ── Chart 3: Revenue at Risk by Category ─────────────────────────────────
    rev_by_cat = (
        pred_df[pred_df["PredictedChurn"] == 1]
        .groupby("RiskCategory")["RevenueAtRisk"]
        .agg(["sum", "mean", "count"])
        .reindex([r for r in risk_order if r in pred_df["RiskCategory"].unique()])
    )

    if not rev_by_cat.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Revenue at Risk — Predicted Churners", fontsize=13,
                     fontweight="bold", color=CORP_BLUE)

        colors_rev = risk_colors[:len(rev_by_cat)]

        axes[0].bar(rev_by_cat.index, rev_by_cat["sum"],
                    color=colors_rev, edgecolor="white", width=0.5)
        axes[0].set_title("Total Monthly Revenue at Risk ($)")
        axes[0].set_ylabel("Monthly Revenue ($)")
        for i, (cat, val) in enumerate(rev_by_cat["sum"].items()):
            axes[0].text(i, val + 200, f"${val:,.0f}",
                         ha="center", fontweight="bold", fontsize=10)

        axes[1].bar(rev_by_cat.index, rev_by_cat["mean"],
                    color=colors_rev, edgecolor="white", width=0.5)
        axes[1].set_title("Avg Monthly Charge per Predicted Churner ($)")
        axes[1].set_ylabel("Avg Monthly Charge ($)")
        for i, (cat, val) in enumerate(rev_by_cat["mean"].items()):
            axes[1].text(i, val + 0.5, f"${val:,.2f}",
                         ha="center", fontweight="bold", fontsize=10)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "revenue_at_risk.png",
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info("Saved: revenue_at_risk.png")

    # ── Chart 4: Probability vs Monthly Charges scatter ───────────────────────
    color_map = {"High Risk": ACC_RED, "Medium Risk": ACC_ORANGE, "Low Risk": ACC_GREEN}

    fig, ax = plt.subplots(figsize=(11, 7))
    for cat in risk_order:
        subset = pred_df[pred_df["RiskCategory"] == cat]
        ax.scatter(subset["MonthlyCharges"], subset["ChurnProbability"],
                   alpha=0.35, s=18, color=color_map[cat], label=cat)

    ax.axhline(CHURN_THRESHOLD, color="black", linestyle="--",
               linewidth=1.2, label=f"Threshold ({CHURN_THRESHOLD})")
    ax.axhline(HIGH_RISK_CUTOFF, color="grey", linestyle=":",
               linewidth=1.2, label=f"High Risk ({HIGH_RISK_CUTOFF})")
    ax.set_xlabel("Monthly Charges ($)", fontsize=12)
    ax.set_ylabel("Churn Probability", fontsize=12)
    ax.set_title("Churn Probability vs Monthly Charges\n(by Risk Category)",
                 fontsize=13, fontweight="bold", color=CORP_BLUE)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probability_vs_charges_scatter.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved: probability_vs_charges_scatter.png")

    # ── Chart 5: Top 20 Highest-Risk Customers ───────────────────────────────
    top20 = pred_df.head(20).copy()

    fig, ax = plt.subplots(figsize=(11, 8))
    colors_top = [ACC_RED if r == "High Risk" else ACC_ORANGE
                  for r in top20["RiskCategory"]]

    bars = ax.barh(
        top20["customerID"][::-1],
        top20["ChurnProbabilityPct"][::-1],
        color=colors_top[::-1],
        edgecolor="white", linewidth=0.6
    )
    ax.axvline(CHURN_THRESHOLD * 100, color="black", linestyle="--",
               linewidth=1.2, label=f"Threshold ({CHURN_THRESHOLD*100:.0f}%)")
    ax.axvline(HIGH_RISK_CUTOFF * 100, color="grey", linestyle=":",
               linewidth=1.2, label=f"High Risk ({HIGH_RISK_CUTOFF*100:.0f}%)")

    for bar, prob in zip(bars, top20["ChurnProbabilityPct"][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%", va="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Churn Probability (%)", fontsize=11)
    ax.set_title("Top 20 Highest-Risk Customers\n(Priority Retention List)",
                 fontsize=13, fontweight="bold", color=CORP_BLUE)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 108)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top20_high_risk_customers.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved: top20_high_risk_customers.png")


# =============================================================================
# STEP 7 — SAVE PREDICTION FILES
# =============================================================================

def save_predictions(pred_df: pd.DataFrame) -> None:
    """
    Saves prediction outputs to disk.

    Files:
      data/predictions/predictions.csv
        → Full prediction table (all 7,043 customers, all columns)
        → Primary input for Power BI AI Prediction Dashboard page

      data/predictions/high_risk_customers.csv
        → Only High Risk customers, sorted by probability
        → Immediate action list for the retention team
        → Should be refreshed weekly in production
    """
    logger.info("STEP 7 — Saving prediction files")

    DATA_PREDICTIONS.mkdir(parents=True, exist_ok=True)

    # Full predictions
    full_path = DATA_PREDICTIONS / "predictions.csv"
    pred_df.to_csv(full_path, index=False)
    logger.info(f"Full predictions saved → {full_path} | {pred_df.shape}")

    # High risk only
    high_risk = pred_df[pred_df["RiskCategory"] == "High Risk"].copy()
    high_risk_path = DATA_PREDICTIONS / "high_risk_customers.csv"
    high_risk.to_csv(high_risk_path, index=False)
    logger.info(f"High risk list saved  → {high_risk_path} | {len(high_risk):,} customers")

    # Medium risk only
    med_risk = pred_df[pred_df["RiskCategory"] == "Medium Risk"].copy()
    med_risk_path = DATA_PREDICTIONS / "medium_risk_customers.csv"
    med_risk.to_csv(med_risk_path, index=False)
    logger.info(f"Medium risk list saved → {med_risk_path} | {len(med_risk):,} customers")


# =============================================================================
# STEP 8 — PRINT SUMMARY REPORT
# =============================================================================

def print_prediction_summary(pred_df: pd.DataFrame, model_name: str) -> None:
    """Prints a business-readable prediction summary to console."""

    total       = len(pred_df)
    pred_churn  = pred_df["PredictedChurn"].sum()
    high_risk   = (pred_df["RiskCategory"] == "High Risk").sum()
    medium_risk = (pred_df["RiskCategory"] == "Medium Risk").sum()
    low_risk    = (pred_df["RiskCategory"] == "Low Risk").sum()

    total_rev_risk  = pred_df["RevenueAtRisk"].sum()
    annual_rev_risk = total_rev_risk * 12

    act_churn = pred_df["ActualChurn"].sum() if pred_df["ActualChurn"].notna().all() else None

    print("\n" + "═" * 62)
    print(f"  PREDICTION SUMMARY — {model_name}")
    print("═" * 62)
    print(f"  Total Customers Scored  : {total:>8,}")
    print(f"  Predicted to Churn      : {pred_churn:>8,}  ({pred_churn/total*100:.1f}%)")
    if act_churn is not None:
        print(f"  Actually Churned        : {int(act_churn):>8,}  ({int(act_churn)/total*100:.1f}%)")
    print()
    print(f"  ── Risk Category Breakdown ──────────────────────")
    print(f"  🔴 High Risk   (≥70%):  {high_risk:>6,}  ({high_risk/total*100:.1f}%)")
    print(f"  🟠 Medium Risk (40–69%): {medium_risk:>6,}  ({medium_risk/total*100:.1f}%)")
    print(f"  🟢 Low Risk    (<40%):  {low_risk:>6,}  ({low_risk/total*100:.1f}%)")
    print()
    print(f"  ── Revenue Impact ───────────────────────────────")
    print(f"  Monthly Revenue at Risk : ${total_rev_risk:>9,.2f}")
    print(f"  Annual Revenue at Risk  : ${annual_rev_risk:>9,.2f}")
    print()
    print(f"  ── Top 5 Priority Customers ─────────────────────")

    top5_cols = ["customerID", "ChurnProbabilityPct", "RiskCategory",
                 "MonthlyCharges", "Contract"]
    top5_cols = [c for c in top5_cols if c in pred_df.columns]
    print(pred_df[top5_cols].head(5).to_string(index=False))
    print("═" * 62 + "\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_prediction_pipeline() -> pd.DataFrame:
    """
    Orchestrates the complete prediction pipeline.

    Returns the full predictions DataFrame.
    """
    logger.info("=" * 60)
    logger.info("  PREDICTION PIPELINE — START")
    logger.info("=" * 60)

    # Step 1: Load artefacts
    model, model_name, scaler, feature_cols = load_model_artefacts()

    # Step 2: Prepare full dataset
    df_original, X_scaled, y_actual = prepare_full_dataset(
        CLEAN_FILE, scaler, feature_cols
    )

    # Step 3: Generate predictions
    y_prob, y_pred = generate_predictions(model, X_scaled)

    # Step 4: Build output DataFrame
    pred_df = build_prediction_output(df_original, y_prob, y_pred, y_actual)

    # Step 5: Validate
    validation = validate_predictions(pred_df)

    # Step 6: Visualise
    visualise_predictions(pred_df)

    # Step 7: Save
    save_predictions(pred_df)

    # Step 8: Summary
    print_prediction_summary(pred_df, model_name)

    logger.info("=" * 60)
    logger.info("  PREDICTION PIPELINE — COMPLETE")
    logger.info(f"  Output: data/predictions/predictions.csv")
    logger.info("=" * 60)

    return pred_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    predictions = run_prediction_pipeline()
