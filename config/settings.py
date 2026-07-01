# =============================================================================
# config/settings.py
# Project-wide configuration — single source of truth for all paths & settings
# =============================================================================

import os
from pathlib import Path

# ── Root ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data Paths ───────────────────────────────────────────────────────────────
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_PREDICTIONS = BASE_DIR / "data" / "predictions"

RAW_FILE       = DATA_RAW / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_FILE     = DATA_PROCESSED / "telco_clean.csv"
FEATURES_FILE  = DATA_PROCESSED / "telco_features.csv"
PREDICTIONS_FILE = DATA_PREDICTIONS / "predictions.csv"

# ── Model Paths ───────────────────────────────────────────────────────────────
MODELS_DIR     = BASE_DIR / "models" / "saved"
EVAL_DIR       = BASE_DIR / "models" / "evaluation"

# ── Reports ──────────────────────────────────────────────────────────────────
FIGURES_DIR    = BASE_DIR / "reports" / "figures"
INSIGHTS_DIR   = BASE_DIR / "reports" / "insights"

# ── Logs ─────────────────────────────────────────────────────────────────────
LOG_DIR        = BASE_DIR / "logs"
LOG_FILE       = LOG_DIR  / "pipeline.log"

# ── ML Settings ──────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
TARGET_COLUMN  = "Churn"
CHURN_THRESHOLD = 0.50        # Probability cut-off for binary prediction
HIGH_RISK_THRESHOLD = 0.70    # Customers above this → "High Risk" category

# ── Feature Lists ─────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

DROP_COLS = ["customerID"]   # Not predictive, kept for reporting only

# ── Power BI / SQL ────────────────────────────────────────────────────────────
DB_NAME        = "telco_churn"
DB_HOST        = "localhost"
DB_PORT        = 3306

