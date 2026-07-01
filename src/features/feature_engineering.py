# =============================================================================
# src/features/feature_engineering.py
# Phase 4 — Feature Engineering Pipeline
# Project: AI-Powered Customer Churn Prediction
# =============================================================================
#
# WHAT THIS MODULE DOES:
#   1.  Creates business-driven features from raw columns
#   2.  Encodes categorical variables (label + one-hot)
#   3.  Scales numerical features (StandardScaler + MinMaxScaler)
#   4.  Builds the final ML-ready feature matrix (X) and target vector (y)
#   5.  Saves artefacts:
#         data/processed/telco_features.csv   — full engineered dataset
#         data/processed/X_train.csv          — training features
#         data/processed/X_test.csv           — test features
#         data/processed/y_train.csv          — training labels
#         data/processed/y_test.csv           — test labels
#         models/saved/scaler.pkl             — fitted scaler
#         models/saved/feature_columns.pkl    — ordered column list
#   6.  Visualises feature distributions & correlation with target
#
# WHY FEATURE ENGINEERING MATTERS:
#   Raw data columns are rarely optimal for ML.  A well-engineered feature
#   can lift model AUC by 5–15 points with zero additional data.
#   Every feature below is justified by a business or statistical reason.
# =============================================================================

import sys
import warnings
import joblib
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    CLEAN_FILE, DATA_PROCESSED, FIGURES_DIR,
    MODELS_DIR, LOG_FILE,
    RANDOM_STATE, TEST_SIZE, TARGET_COLUMN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, str(LOG_FILE))

# =============================================================================
# CONSTANTS
# =============================================================================

# Binary yes/no columns — straightforward 1/0 encoding
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "SeniorCitizenLabel",
]

# Ordinal columns — natural order encodes business meaning
ORDINAL_MAPS = {
    "Contract": {
        "Month-to-month": 0,   # Highest risk
        "One year"       : 1,  # Medium risk
        "Two year"       : 2,  # Lowest risk
    },
    "InternetService": {
        "No"          : 0,
        "DSL"         : 1,
        "Fiber optic" : 2,
    },
}

# Nominal columns — no natural order → one-hot encode
NOMINAL_COLS = [
    "PaymentMethod",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Columns to DROP before ML (identifiers, leakage risks, redundant)
DROP_BEFORE_ML = [
    "customerID",       # Identifier — not a predictor
    "ChurnLabel",       # String version of target — leakage
    "ChurnStatus",      # Same as above
    "SeniorCitizen",    # Replaced by SeniorCitizenLabel (encoded)
    "gender",           # EDA showed no predictive power
    "TenureGroup",      # String bin — replaced by numeric TenureMonths
    "ChargesBand",      # String bin — replaced by MonthlyCharges directly
    "RuleBasedRisk",    # Rule-based placeholder — will be replaced by ML
]


# =============================================================================
# STEP 1 — LOAD CLEAN DATA
# =============================================================================

def load_clean_data(filepath: Path = CLEAN_FILE) -> pd.DataFrame:
    """
    Loads the Phase 2 clean dataset.

    We use telco_clean.csv (not powerbi_ready.csv) as our starting point
    because powerbi_ready already has some pre-engineered columns that we
    want to re-create in a controlled, ML-optimised way.
    """
    logger.info(f"Loading clean data from: {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {filepath}.\n"
            "Run Phase 2 first: python src/data/data_cleaning.py"
        )

    df = pd.read_csv(filepath)
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# STEP 2 — BUSINESS-DRIVEN FEATURE CREATION
# =============================================================================

def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 10 new features grounded in business logic.

    Each feature answers a specific business question and is expected to
    improve the model's ability to separate churners from retained customers.

    New features created:
      1.  ServiceCount          — How many add-ons does the customer have?
      2.  AvgMonthlySpend       — Sanity-check: TotalCharges / tenure
      3.  ChargeToTenureRatio   — Value perception: are they paying a lot for their tenure?
      4.  CLV                   — Customer Lifetime Value proxy
      5.  TenureGroup_num       — Ordinal tenure bin (4 risk bands)
      6.  ChargesBand_num       — Ordinal charge tier (4 price bands)
      7.  ContractRisk          — Numeric risk score from contract type
      8.  IsNewCustomer         — Flag: tenure ≤ 12 months
      9.  IsLongTermCustomer    — Flag: tenure > 48 months
      10. HasNoAddOns           — Flag: zero add-on services (highest risk profile)
    """
    logger.info("STEP 2 — Creating business-driven features")

    df = df.copy()

    # ── 1. Service Count ─────────────────────────────────────────────────────
    # Business rationale: more services = more switching cost = lower churn.
    # EDA confirmed that each add-on service reduces churn probability.
    addon_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = df[addon_services].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )
    logger.info(
        f"ServiceCount: min={df['ServiceCount'].min()}, "
        f"max={df['ServiceCount'].max()}, "
        f"mean={df['ServiceCount'].mean():.2f}"
    )

    # ── 2. Average Monthly Spend (from TotalCharges / tenure) ─────────────────
    # Business rationale: a customer paying $80/mo for 1 month is very different
    # from one paying $80/mo for 60 months. This captures the "sustained spend" signal.
    # Edge case: tenure=0 → avoid division by zero, use MonthlyCharges directly.
    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0,
        (df["TotalCharges"] / df["tenure"]).round(2),
        df["MonthlyCharges"]
    )

    # ── 3. Charge-to-Tenure Ratio ────────────────────────────────────────────
    # Business rationale: if MonthlyCharges >> AvgMonthlySpend, the customer's
    # bill has been rising — a potential churn trigger (price fatigue).
    # Ratio > 1 means current charges are above their historical average.
    df["ChargeToTenureRatio"] = (
        df["MonthlyCharges"] / (df["AvgMonthlySpend"] + 1e-6)
    ).round(4)

    # ── 4. Customer Lifetime Value (CLV) ─────────────────────────────────────
    # Business rationale: CLV = total revenue generated by this customer to date.
    # High CLV customers are worth more to retain — priority for retention spend.
    # Using TotalCharges as CLV proxy (actual CLV would require margin data).
    df["CLV"] = df["TotalCharges"].round(2)

    # ── 5. Tenure Group (ordinal numeric) ────────────────────────────────────
    # Business rationale: risk is not linear with tenure. The first year
    # is dramatically riskier than years 2–4. Ordinal encoding captures this
    # non-linear relationship better than raw tenure alone.
    #   0 = 0–12 mo  (onboarding, highest risk)
    #   1 = 13–24 mo (early retention)
    #   2 = 25–48 mo (established)
    #   3 = 49–72 mo (loyal)
    df["TenureGroup_num"] = pd.cut(
        df["tenure"],
        bins   = [0, 12, 24, 48, 72],
        labels = [0, 1, 2, 3],
        right  = True,
    ).astype(float).fillna(0).astype(int)

    # ── 6. Charges Band (ordinal numeric) ────────────────────────────────────
    # Business rationale: pricing tiers have non-linear effects on churn.
    # Premium customers ($95+) may feel least value-for-money.
    df["ChargesBand_num"] = pd.cut(
        df["MonthlyCharges"],
        bins   = [0, 35, 65, 95, 120],
        labels = [0, 1, 2, 3],
        right  = True,
    ).astype(float).fillna(0).astype(int)

    # ── 7. Contract Risk Score ────────────────────────────────────────────────
    # Business rationale: month-to-month is the single strongest churn predictor.
    # A numeric risk score lets the model learn a gradient rather than a category.
    #   2 = Month-to-month (highest risk)
    #   1 = One year
    #   0 = Two year (lowest risk)
    contract_risk = {
        "Month-to-month": 2,
        "One year"       : 1,
        "Two year"       : 0,
    }
    df["ContractRisk"] = df["Contract"].map(contract_risk).fillna(1)

    # ── 8. Is New Customer (first 12 months) ─────────────────────────────────
    # Business rationale: binary flag for the highest-risk period.
    # Useful for creating targeted interventions and for ML interactions.
    df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)

    # ── 9. Is Long-Term Customer (49+ months) ────────────────────────────────
    # Business rationale: binary flag for the most loyal segment.
    # Long-term customers rarely churn — this flag helps the model protect them.
    df["IsLongTermCustomer"] = (df["tenure"] > 48).astype(int)

    # ── 10. Has No Add-On Services ────────────────────────────────────────────
    # Business rationale: zero add-ons = lowest switching cost = highest flight risk.
    # EDA showed add-ons significantly reduce churn.
    df["HasNoAddOns"] = (df["ServiceCount"] == 0).astype(int)

    logger.info("10 business features created successfully")
    logger.info(
        f"New columns: ServiceCount, AvgMonthlySpend, ChargeToTenureRatio, "
        f"CLV, TenureGroup_num, ChargesBand_num, ContractRisk, "
        f"IsNewCustomer, IsLongTermCustomer, HasNoAddOns"
    )

    return df


# =============================================================================
# STEP 3 — ENCODE CATEGORICAL VARIABLES
# =============================================================================

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes all categorical columns into numeric form for ML.

    Three encoding strategies used:
      A) Binary encoding    → Yes/No columns → 1/0
      B) Ordinal encoding   → Columns with natural order (Contract, InternetService)
      C) One-hot encoding   → Nominal columns with no natural order (PaymentMethod, etc.)

    Why NOT label-encode everything?
      Label encoding imposes a false ordinal relationship on nominal columns.
      e.g. PaymentMethod: "Electronic check"=0, "Mailed check"=1 implies
      electronic < mailed, which is meaningless and confuses linear models.
      One-hot encoding avoids this by creating separate binary columns.

    Why NOT one-hot encode everything?
      One-hot encoding on high-cardinality columns creates too many features
      (curse of dimensionality). For columns with a natural order (Contract,
      InternetService) ordinal encoding is both simpler and more informative.
    """
    logger.info("STEP 3 — Encoding categorical features")

    df = df.copy()

    # ── A) Binary encoding: Yes → 1, No → 0 ─────────────────────────────────
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
            logger.info(f"Binary encoded: {col}")

    # ── B) Ordinal encoding: predefined mapping ───────────────────────────────
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
            logger.info(f"Ordinal encoded: {col} → {mapping}")

    # ── C) One-hot encoding: nominal multi-category columns ───────────────────
    nominal_present = [c for c in NOMINAL_COLS if c in df.columns]
    if nominal_present:
        df = pd.get_dummies(
            df,
            columns    = nominal_present,
            drop_first = True,   # Avoids dummy variable trap (multicollinearity)
            dtype      = int,
        )
        new_cols = [c for c in df.columns if any(c.startswith(n + "_") for n in nominal_present)]
        logger.info(f"One-hot encoded {len(nominal_present)} columns → {len(new_cols)} new binary columns")
        logger.info(f"OHE columns created: {new_cols}")

    logger.info(f"Shape after encoding: {df.shape}")
    return df


# =============================================================================
# STEP 4 — DROP NON-ML COLUMNS
# =============================================================================

def drop_non_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that must not enter the ML model:
      - Identifiers (customerID)    → not predictive, would overfit
      - Target string aliases       → data leakage
      - Redundant columns           → already captured in engineered features
      - Low-signal columns          → gender (EDA proved no predictive value)

    Data leakage is the #1 silent killer of ML models in production.
    If the model sees 'ChurnLabel' (which IS the target, just as a string),
    it learns a perfect but useless relationship.
    """
    logger.info("STEP 4 — Dropping non-ML columns")

    cols_to_drop = [c for c in DROP_BEFORE_ML if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
    logger.info(f"Shape after dropping: {df.shape}")

    return df


# =============================================================================
# STEP 5 — FEATURE SCALING
# =============================================================================

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales numerical features using StandardScaler (zero mean, unit variance).

    CRITICAL RULE: Fit the scaler on TRAINING data only.
    NEVER fit on the full dataset or test data.

    Why? Fitting on the full dataset causes data leakage:
    test set statistics (mean, std) would "leak" into the training process,
    giving an overly optimistic performance estimate that won't hold in production.

    Which columns are scaled?
    Only continuous numerical columns. Binary (0/1) and ordinal (0/1/2)
    encoded columns are already in a reasonable range — scaling them
    would distort their interpretability without improving model performance.

    StandardScaler vs MinMaxScaler:
      StandardScaler (z-score) — preferred for:
        • Logistic Regression, SVM, linear models (assumes normal distribution)
        • When outliers are present (MinMaxScaler is distorted by outliers)
      MinMaxScaler [0,1] — preferred for:
        • Neural networks, k-NN (sensitive to scale, not outliers)
    We use StandardScaler here as it works well across all our models.

    Returns:
        X_train_scaled, X_test_scaled, fitted_scaler
    """
    logger.info("STEP 5 — Scaling numerical features")

    # Identify continuous numerical columns to scale
    scale_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges",
                               "AvgMonthlySpend", "ChargeToTenureRatio", "CLV"]
                  if c in X_train.columns]

    logger.info(f"Columns to scale: {scale_cols}")

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])   # transform only

    logger.info(
        f"Scaler fitted on training set ({len(X_train):,} rows). "
        f"Test set transformed (not fitted)."
    )

    return X_train, X_test, scaler


# =============================================================================
# STEP 6 — TRAIN / TEST SPLIT
# =============================================================================

def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the engineered dataset into train and test sets.

    Stratified split:
      stratify=y ensures both splits have the same class ratio (~22% churn).
      Without stratification, random chance might give the test set 15% churn
      and the train set 25%, making evaluation misleading.

    Args:
        df           : fully engineered DataFrame (includes target column)
        target       : name of the target column ("Churn")
        test_size    : fraction for test set (0.20 = 80/20 split)
        random_state : seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"STEP 6 — Train/test split (test_size={test_size}, stratified)")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        stratify     = y,           # Preserve class distribution in both splits
        random_state = random_state,
    )

    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    logger.info(f"y_train churn rate: {y_train.mean()*100:.1f}%")
    logger.info(f"y_test  churn rate: {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 7 — VISUALISE ENGINEERED FEATURES
# =============================================================================

def visualise_engineered_features(df: pd.DataFrame) -> None:
    """
    Produces visualisations for the new engineered features,
    showing their relationship with the churn target.
    """
    logger.info("STEP 7 — Visualising engineered features")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    new_features = [
        "ServiceCount", "AvgMonthlySpend", "ChargeToTenureRatio",
        "TenureGroup_num", "ChargesBand_num", "ContractRisk",
        "IsNewCustomer", "IsLongTermCustomer", "HasNoAddOns",
    ]
    new_features = [f for f in new_features if f in df.columns]

    # ── Figure 1: New feature distributions by churn ──────────────────────────
    n_cols = 3
    n_rows = (len(new_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(new_features):
        ax = axes[i]
        churn_0 = df[df["Churn"] == 0][feat]
        churn_1 = df[df["Churn"] == 1][feat]

        unique_vals = df[feat].nunique()

        if unique_vals <= 6:
            # Categorical / flag: grouped bar chart
            tbl = df.groupby([feat, "Churn"]).size().unstack(fill_value=0)
            tbl.columns = ["No Churn", "Churned"]
            tbl.plot(kind="bar", ax=ax, color=["#2ECC71", "#E74C3C"],
                     edgecolor="white", rot=0)
            ax.set_xlabel(feat)
            ax.set_ylabel("Count")
        else:
            # Continuous: KDE overlay
            churn_0.plot.kde(ax=ax, color="#2ECC71", linewidth=2, label="No Churn")
            churn_1.plot.kde(ax=ax, color="#E74C3C", linewidth=2, label="Churned")
            ax.set_xlabel(feat)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        ax.set_title(f"{feat}\nvs Churn", fontsize=10, fontweight="bold")

    # Hide unused axes
    for j in range(len(new_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Engineered Features vs Churn", fontsize=14,
                 fontweight="bold", color="#2C3E50", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "engineered_features_vs_churn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Figure saved → {out}")

    # ── Figure 2: ServiceCount vs Churn Rate ─────────────────────────────────
    if "ServiceCount" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("ServiceCount — A Key Retention Driver", fontsize=13,
                     fontweight="bold", color="#2C3E50")

        service_churn = (
            df.groupby("ServiceCount")
              .agg(Total=("Churn", "count"), Churned=("Churn", "sum"))
              .assign(ChurnRate=lambda x: x["Churned"] / x["Total"] * 100)
              .reset_index()
        )

        axes[0].bar(service_churn["ServiceCount"], service_churn["Total"],
                    color="#3498DB", edgecolor="white")
        axes[0].set_title("Customer Count by Service Count")
        axes[0].set_xlabel("Number of Add-On Services")
        axes[0].set_ylabel("Customers")

        color_list = ["#E74C3C" if r > 26 else "#2ECC71"
                      for r in service_churn["ChurnRate"]]
        axes[1].bar(service_churn["ServiceCount"], service_churn["ChurnRate"],
                    color=color_list, edgecolor="white")
        axes[1].axhline(22.1, color="black", linestyle="--",
                        linewidth=1.2, label="Avg Churn 22.1%")
        for _, row in service_churn.iterrows():
            axes[1].text(row["ServiceCount"], row["ChurnRate"] + 0.5,
                         f"{row['ChurnRate']:.1f}%", ha="center",
                         fontsize=9, fontweight="bold")
        axes[1].set_title("Churn Rate by Number of Services")
        axes[1].set_xlabel("Number of Add-On Services")
        axes[1].set_ylabel("Churn Rate (%)")
        axes[1].legend()

        plt.tight_layout()
        out2 = FIGURES_DIR / "service_count_vs_churn.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Figure saved → {out2}")

    # ── Figure 3: Feature correlation heatmap (post-engineering) ─────────────
    num_cols_for_corr = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "ServiceCount", "AvgMonthlySpend", "ChargeToTenureRatio",
        "CLV", "TenureGroup_num", "ChargesBand_num", "ContractRisk",
        "IsNewCustomer", "IsLongTermCustomer", "HasNoAddOns", "Churn",
    ]
    num_cols_for_corr = [c for c in num_cols_for_corr if c in df.columns]

    corr = df[num_cols_for_corr].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr,
        annot      = True,
        fmt        = ".2f",
        cmap       = "RdYlGn",
        center     = 0,
        vmin=-1, vmax=1,
        ax         = ax,
        linewidths = 0.4,
        linecolor  = "white",
        annot_kws  = {"size": 8},
    )
    ax.set_title("Full Feature Correlation Matrix (Post Feature Engineering)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out3 = FIGURES_DIR / "post_fe_correlation_heatmap.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Figure saved → {out3}")


# =============================================================================
# STEP 8 — SAVE ARTEFACTS
# =============================================================================

def save_artefacts(
    df_features  : pd.DataFrame,
    X_train      : pd.DataFrame,
    X_test       : pd.DataFrame,
    y_train      : pd.Series,
    y_test       : pd.Series,
    scaler       : StandardScaler,
) -> None:
    """
    Saves all engineered datasets and model artefacts to disk.

    Files saved:
      data/processed/telco_features.csv   — full feature matrix (pre-split)
      data/processed/X_train.csv          — scaled training features
      data/processed/X_test.csv           — scaled test features
      data/processed/y_train.csv          — training labels
      data/processed/y_test.csv           — test labels
      models/saved/scaler.pkl             — fitted StandardScaler
      models/saved/feature_columns.pkl    — ordered list of feature names

    Why save feature_columns?
      At inference time (Phase 8), the incoming customer data must be
      transformed into EXACTLY the same column order that the model
      was trained on. A mismatch of even one column silently corrupts
      predictions. Saving the column list eliminates this risk.
    """
    logger.info("STEP 8 — Saving artefacts")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── CSV artefacts ─────────────────────────────────────────────────────────
    paths = {
        DATA_PROCESSED / "telco_features.csv" : df_features,
        DATA_PROCESSED / "X_train.csv"        : X_train,
        DATA_PROCESSED / "X_test.csv"         : X_test,
        DATA_PROCESSED / "y_train.csv"        : y_train.to_frame(),
        DATA_PROCESSED / "y_test.csv"         : y_test.to_frame(),
    }

    for path, data in paths.items():
        data.to_csv(path, index=False)
        logger.info(f"Saved → {path} | shape={data.shape}")

    # ── Model artefacts ───────────────────────────────────────────────────────
    scaler_path   = MODELS_DIR / "scaler.pkl"
    featcols_path = MODELS_DIR / "feature_columns.pkl"

    joblib.dump(scaler, scaler_path)
    joblib.dump(list(X_train.columns), featcols_path)

    logger.info(f"Saved → {scaler_path}")
    logger.info(f"Saved → {featcols_path} ({len(X_train.columns)} features)")


# =============================================================================
# STEP 9 — FEATURE IMPORTANCE PREVIEW (Correlation with Target)
# =============================================================================

def feature_importance_preview(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Computes point-biserial correlation between each feature and the binary
    target (Churn). This gives a quick preview of which features the ML model
    is likely to find most useful.

    Note: This is a LINEAR correlation only. Tree-based models (Random Forest,
    XGBoost) can leverage non-linear relationships that correlation misses.
    The true feature importance ranking will come from the ML models in Phase 5.
    """
    logger.info("STEP 9 — Feature importance preview (correlation with target)")

    corr_with_target = (
        pd.concat([X_train, y_train], axis=1)
          .corr()[TARGET_COLUMN]
          .drop(TARGET_COLUMN)
          .abs()
          .sort_values(ascending=False)
          .round(4)
    )

    print("\n" + "═" * 50)
    print("  FEATURE IMPORTANCE PREVIEW (|correlation| with Churn)")
    print("═" * 50)
    for feat, corr_val in corr_with_target.head(20).items():
        bar = "█" * int(corr_val * 40)
        print(f"  {feat:<35} {corr_val:.4f}  {bar}")
    print("═" * 50 + "\n")

    # Plot top 20
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = corr_with_target.head(20)
    colors = ["#E74C3C" if v > 0.15 else "#3498DB" for v in top20.values]
    ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1],
            edgecolor="white", linewidth=0.6)
    ax.set_title("Feature Correlation with Churn Target\n(Top 20 Features, Pre-ML)",
                 fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_xlabel("|Pearson Correlation|")
    ax.axvline(0.15, color="black", linestyle="--",
               linewidth=1, label="Threshold 0.15")
    ax.legend()

    for i, (feat, val) in enumerate(zip(top20.index[::-1], top20.values[::-1])):
        ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "feature_correlation_with_target.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Feature importance preview saved → {out}")

    return corr_with_target.reset_index().rename(
        columns={"index": "Feature", TARGET_COLUMN: "AbsCorrelation"}
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_feature_engineering_pipeline() -> dict:
    """
    Orchestrates all feature engineering steps in sequence.

    Pipeline:
      Load → Business Features → Encode → Drop Non-ML Cols
      → Split → Scale → Visualise → Save → Preview

    Returns a dict with all outputs for downstream use.
    """
    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING PIPELINE — START")
    logger.info("=" * 60)

    # Step 1: Load
    df = load_clean_data()

    # Step 2: Business features
    df = create_business_features(df)

    # Step 3: Encode categoricals
    df = encode_categorical_features(df)

    # Step 4: Drop non-ML columns
    df = drop_non_ml_columns(df)

    # Step 5: Train/test split (BEFORE scaling to prevent leakage)
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 6: Scale (fit on train, transform both)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 7: Visualise
    visualise_engineered_features(df)

    # Step 8: Save all artefacts
    save_artefacts(df, X_train_scaled, X_test_scaled, y_train, y_test, scaler)

    # Step 9: Feature importance preview
    importance_df = feature_importance_preview(X_train_scaled, y_train)

    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING PIPELINE — COMPLETE")
    logger.info(f"  Total features for ML: {X_train_scaled.shape[1]}")
    logger.info("=" * 60)

    print(f"\n✅ Feature Engineering Complete")
    print(f"   Training set : {X_train_scaled.shape[0]:,} rows × {X_train_scaled.shape[1]} features")
    print(f"   Test set     : {X_test_scaled.shape[0]:,} rows × {X_test_scaled.shape[1]} features")
    print(f"   Churn rate (train): {y_train.mean()*100:.1f}%")
    print(f"   Churn rate (test) : {y_test.mean()*100:.1f}%")
    print(f"\n   Files saved:")
    print(f"     data/processed/telco_features.csv")
    print(f"     data/processed/X_train.csv, X_test.csv")
    print(f"     data/processed/y_train.csv, y_test.csv")
    print(f"     models/saved/scaler.pkl")
    print(f"     models/saved/feature_columns.pkl")

    return {
        "df_features"  : df,
        "X_train"      : X_train_scaled,
        "X_test"       : X_test_scaled,
        "y_train"      : y_train,
        "y_test"       : y_test,
        "scaler"       : scaler,
        "importance"   : importance_df,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_feature_engineering_pipeline()
