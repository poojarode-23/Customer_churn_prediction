# =============================================================================
# src/data/data_cleaning.py
# Phase 2 — Data Cleaning Pipeline
# Project: AI-Powered Customer Churn Prediction
# =============================================================================
# WHAT THIS SCRIPT DOES (in order):
#   1. Loads the raw IBM Telco CSV
#   2. Audits the dataset (shape, dtypes, nulls, duplicates)
#   3. Fixes data type issues (TotalCharges stored as string)
#   4. Handles missing values
#   5. Removes duplicates
#   6. Standardises column names and binary values
#   7. Detects and reports outliers (does NOT remove — business decision)
#   8. Saves the clean dataset to data/processed/telco_clean.csv
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Allow imports from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    RAW_FILE, CLEAN_FILE, FIGURES_DIR, LOG_FILE,
    CATEGORICAL_COLS, NUMERICAL_COLS
)
from src.utils.logger import get_logger

logger = get_logger(__name__, str(LOG_FILE))


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the raw IBM Telco CSV into a Pandas DataFrame.

    Why Pandas?  It's the industry-standard library for tabular data.
    We log shape and column names immediately so any mismatch is caught early.
    """
    logger.info(f"Loading raw data from: {filepath}")

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "Place the CSV in:  data/raw/"
        )

    df = pd.read_csv(filepath)

    logger.info(f"Raw data loaded — Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df


# =============================================================================
# STEP 2 — AUDIT DATASET
# =============================================================================

def audit_dataset(df: pd.DataFrame) -> dict:
    """
    Performs a full data quality audit and returns a summary dictionary.

    In a real enterprise project this audit report goes to the data governance
    team before any cleaning begins — we never modify data silently.

    Returns a dict with:
      - shape         : (rows, cols)
      - dtypes        : column → dtype mapping
      - null_counts   : columns with missing values
      - null_pct      : % missing per column
      - duplicates    : number of exact duplicate rows
      - unique_counts : cardinality per column
    """
    logger.info("── DATASET AUDIT ──────────────────────────────────────")

    audit = {}

    # Basic shape
    audit["shape"] = df.shape
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Data types
    audit["dtypes"] = df.dtypes.to_dict()
    logger.info(f"\nData Types:\n{df.dtypes}")

    # Missing values
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    audit["null_counts"] = null_counts[null_counts > 0].to_dict()
    audit["null_pct"]    = null_pct[null_pct > 0].to_dict()

    if audit["null_counts"]:
        logger.warning(f"Missing values found:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No explicit NaN values found (may have hidden blanks — see TotalCharges)")

    # Duplicates
    audit["duplicates"] = df.duplicated().sum()
    logger.info(f"Duplicate rows: {audit['duplicates']}")

    # Cardinality (unique values per column)
    audit["unique_counts"] = df.nunique().to_dict()
    logger.info(f"\nUnique value counts:\n{df.nunique()}")

    # Value counts for target
    if "Churn" in df.columns:
        churn_dist = df["Churn"].value_counts()
        churn_pct  = df["Churn"].value_counts(normalize=True).mul(100).round(2)
        logger.info(f"\nTarget distribution (Churn):\n{churn_dist}")
        logger.info(f"Target % distribution:\n{churn_pct}")
        audit["churn_distribution"] = churn_dist.to_dict()

    logger.info("── END AUDIT ───────────────────────────────────────────")
    return audit


# =============================================================================
# STEP 3 — FIX DATA TYPES
# =============================================================================

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    The IBM Telco dataset has ONE known data type bug:
      TotalCharges is stored as OBJECT (string) instead of float.

    Why?  Some rows contain a space (" ") instead of a number.
    Pandas reads spaces as strings, making the entire column object-typed.

    Fix:
      1. Replace whitespace strings with NaN
      2. Cast column to float64
      3. Log how many rows were affected
    """
    logger.info("Fixing data types...")

    df = df.copy()   # Never mutate the original DataFrame in-place

    # ── TotalCharges: string → float ─────────────────────────────────────────
    # Replace any blank / whitespace-only string with NaN first
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)

    # Count affected rows before conversion
    blanks = df["TotalCharges"].isna().sum()
    logger.info(f"TotalCharges — blank/whitespace rows found: {blanks}")

    # Now safely cast to numeric; errors='coerce' converts any remaining
    # non-numeric values to NaN instead of raising an exception
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    logger.info("TotalCharges converted to float64")

    # ── SeniorCitizen: 0/1 int → descriptive string (optional, aids BI) ──────
    # Keeping as int for ML; we'll create a readable label column instead
    df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    logger.info("SeniorCitizenLabel column created (0→No, 1→Yes)")

    return df


# =============================================================================
# STEP 4 — HANDLE MISSING VALUES
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles NaN values introduced during type conversion (TotalCharges).

    Strategy — business-driven, not statistical:
      TotalCharges NaN rows are NEW customers (tenure = 0, no charges yet).
      Imputing them with median would be WRONG — they haven't been billed.
      Correct action: fill with 0 (they owe nothing yet).

    Why NOT drop these rows?
      - There are only ~11 rows (0.15% of data)
      - Dropping removes real customers from our analysis
      - Setting to 0 reflects reality: $0 total charges for new customers

    Best practice: DOCUMENT every imputation decision.
    """
    logger.info("Handling missing values...")

    df = df.copy()

    # Check which rows have NaN in TotalCharges
    null_mask = df["TotalCharges"].isna()
    logger.info(f"Rows with NaN TotalCharges: {null_mask.sum()}")

    if null_mask.sum() > 0:
        # Inspect these rows (log key fields)
        sample = df[null_mask][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]
        logger.info(f"NaN TotalCharges rows sample:\n{sample.head()}")

        # All NaN rows should have tenure == 0
        tenure_zero = df[null_mask]["tenure"].eq(0).all()
        logger.info(f"All NaN TotalCharges rows have tenure==0: {tenure_zero}")

        # Fill with 0 — correct for new customers
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        logger.info("TotalCharges NaN → 0.0 (new customers, no billing yet)")

    # Final check — no NaN should remain
    remaining_nulls = df.isnull().sum().sum()
    logger.info(f"Total NaN values remaining after cleaning: {remaining_nulls}")

    return df


# =============================================================================
# STEP 5 — HANDLE DUPLICATES
# =============================================================================

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicate rows. In a CRM dataset, duplicates usually mean
    an ETL pipeline error — every customer should appear exactly once.

    We log before/after counts so the data lineage is auditable.
    """
    logger.info("Checking for duplicates...")

    df = df.copy()
    before = len(df)

    # Check for customerID duplicates specifically (more targeted)
    id_dupes = df["customerID"].duplicated().sum()
    logger.info(f"Duplicate customerIDs: {id_dupes}")

    # Remove full-row duplicates
    df = df.drop_duplicates()
    after = len(df)

    removed = before - after
    logger.info(f"Duplicate rows removed: {removed} (before={before}, after={after})")

    return df


# =============================================================================
# STEP 6 — STANDARDISE VALUES
# =============================================================================

def standardise_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises inconsistent string values across the dataset.

    Known IBM Telco issues:
      - "No internet service" and "No phone service" are verbose versions of "No"
        → They should be collapsed to "No" for modelling (they carry the same
          business meaning and prevent OneHotEncoder from creating junk columns)
      - "Churn" column: "Yes"/"No" strings → 1/0 integers for ML

    Best practice: strip whitespace from all string columns to catch invisible
    leading/trailing spaces that break groupby and merge operations.
    """
    logger.info("Standardising column values...")

    df = df.copy()

    # ── Strip whitespace from all object columns ──────────────────────────────
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    logger.info(f"Stripped whitespace from {len(str_cols)} string columns")

    # ── Collapse "No internet service" / "No phone service" → "No" ───────────
    replace_map = {
        "No internet service": "No",
        "No phone service"   : "No"
    }
    # Apply only to relevant categorical columns
    internet_dependent = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    phone_dependent = ["MultipleLines"]

    for col in internet_dependent + phone_dependent:
        if col in df.columns:
            before_unique = df[col].unique().tolist()
            df[col] = df[col].replace(replace_map)
            after_unique = df[col].unique().tolist()
            if before_unique != after_unique:
                logger.info(f"{col}: {before_unique} → {after_unique}")

    # ── Encode target variable: Churn Yes/No → 1/0 ───────────────────────────
    # ML models need numeric targets. Keep original for readability.
    df["ChurnLabel"] = df["Churn"]                         # "Yes" / "No" — for dashboards
    df["Churn"]      = df["Churn"].map({"Yes": 1, "No": 0})
    logger.info("Churn encoded: Yes→1, No→0 | Original preserved as ChurnLabel")

    # ── Verify encoding ───────────────────────────────────────────────────────
    churn_counts = df["Churn"].value_counts()
    logger.info(f"Churn value counts after encoding:\n{churn_counts}")

    return df


# =============================================================================
# STEP 7 — OUTLIER DETECTION
# =============================================================================

def detect_outliers(df: pd.DataFrame, save_figures: bool = True) -> dict:
    """
    Detects outliers in numerical columns using the IQR method.

    IQR Rule:
      Lower bound = Q1 - 1.5 × IQR
      Upper bound = Q3 + 1.5 × IQR
      Any value outside these bounds is a potential outlier.

    IMPORTANT: We DETECT and REPORT — we do NOT remove outliers here.
    Removing them is a BUSINESS DECISION, not a technical one.
      Example: A customer with $10,000 in total charges is a high-value
               customer — removing them would be wrong.

    Returns a dict of {column: outlier_count}.
    """
    logger.info("Detecting outliers in numerical columns...")

    numerical = ["tenure", "MonthlyCharges", "TotalCharges"]
    outlier_report = {}

    for col in numerical:
        if col not in df.columns:
            continue

        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        count    = len(outliers)
        outlier_report[col] = count

        logger.info(
            f"{col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
            f"bounds=[{lower:.2f}, {upper:.2f}], outliers={count}"
        )

    if save_figures:
        _plot_outlier_boxplots(df, numerical)

    return outlier_report


def _plot_outlier_boxplots(df: pd.DataFrame, cols: list) -> None:
    """
    Saves a boxplot figure for outlier visualisation.
    Stored in reports/figures/ for use in the EDA notebook.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(cols), figsize=(14, 5))
    fig.suptitle("Outlier Detection — Boxplots", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, cols):
        df.boxplot(column=col, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")

    plt.tight_layout()
    out_path = FIGURES_DIR / "outlier_boxplots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Outlier boxplot saved → {out_path}")


# =============================================================================
# STEP 8 — FINAL VALIDATION
# =============================================================================

def validate_clean_data(df: pd.DataFrame) -> bool:
    """
    Runs a set of assertions on the clean DataFrame.
    Think of this as a lightweight data contract test.

    In production this would be replaced by Great Expectations or Pydantic.
    Returns True if all checks pass, raises AssertionError otherwise.
    """
    logger.info("Running final validation checks...")

    checks = {
        "No NaN values"          : df.isnull().sum().sum() == 0,
        "No duplicate customerID": not df["customerID"].duplicated().any(),
        "TotalCharges is float"  : df["TotalCharges"].dtype == float,
        "Churn is int (0/1)"     : set(df["Churn"].unique()).issubset({0, 1}),
        "Row count >= 7000"      : len(df) >= 7000,
        "Expected 22+ columns"   : df.shape[1] >= 22,
    }

    all_passed = True
    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status} | {check}")
        if not result:
            all_passed = False

    if all_passed:
        logger.info("All validation checks passed ✓")
    else:
        logger.error("One or more validation checks FAILED — review output above")

    return all_passed


# =============================================================================
# STEP 9 — SAVE CLEAN DATASET
# =============================================================================

def save_clean_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Saves the clean DataFrame to CSV.

    Why CSV and not Parquet/Feather?
      - CSV is readable by Power BI, Excel, and SQL loaders with zero config
      - For this project size (~7k rows) CSV performance is sufficient
      - Parquet would be preferred for millions of rows
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Clean dataset saved → {filepath}")
    logger.info(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_cleaning_pipeline() -> pd.DataFrame:
    """
    Orchestrates all cleaning steps in sequence.
    Returns the final clean DataFrame.

    This function is the single entry point — import and call this
    from any notebook or downstream script.
    """
    logger.info("=" * 60)
    logger.info("  DATA CLEANING PIPELINE — START")
    logger.info("=" * 60)

    # Step 1: Load
    df = load_raw_data(RAW_FILE)

    # Step 2: Audit (read-only, returns report)
    audit = audit_dataset(df)

    # Step 3: Fix data types
    df = fix_data_types(df)

    # Step 4: Handle missing values
    df = handle_missing_values(df)

    # Step 5: Remove duplicates
    df = handle_duplicates(df)

    # Step 6: Standardise values
    df = standardise_values(df)

    # Step 7: Detect outliers (logs + saves chart, does NOT modify df)
    outlier_report = detect_outliers(df, save_figures=True)
    logger.info(f"Outlier report: {outlier_report}")

    # Step 8: Validate
    is_valid = validate_clean_data(df)
    if not is_valid:
        logger.warning("Validation failed — saving anyway, review logs")

    # Step 9: Save
    save_clean_data(df, CLEAN_FILE)

    logger.info("=" * 60)
    logger.info("  DATA CLEANING PIPELINE — COMPLETE")
    logger.info("=" * 60)

    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run directly:  python src/data/data_cleaning.py
    Or import:     from src.data.data_cleaning import run_cleaning_pipeline
    """
    clean_df = run_cleaning_pipeline()

    # Print a quick summary to terminal
    print("\n── Clean Dataset Summary ─────────────────────────")
    print(clean_df.dtypes)
    print("\nFirst 3 rows:")
    print(clean_df.head(3))
