# =============================================================================
# tests/test_feature_engineering.py
# Unit tests for the feature engineering pipeline
# Run with: python -m pytest tests/test_feature_engineering.py -v
# =============================================================================

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_engineering import (
    create_business_features,
    encode_categorical_features,
    drop_non_ml_columns,
    split_data,
    scale_features,
)


# =============================================================================
# FIXTURE — minimal DataFrame mimicking telco_clean.csv
# =============================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID"      : ["A-001", "A-002", "A-003", "A-004", "A-005",
                              "A-006", "A-007", "A-008", "A-009", "A-010"],
        "gender"          : ["Male"] * 10,
        "SeniorCitizen"   : [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "SeniorCitizenLabel": ["No","Yes","No","No","Yes","No","No","Yes","No","No"],
        "Partner"         : ["Yes","No","Yes","No","Yes","No","Yes","No","Yes","No"],
        "Dependents"      : ["No","No","Yes","No","No","Yes","No","No","Yes","No"],
        "tenure"          : [1, 34, 0, 58, 12, 25, 72, 6, 48, 30],
        "PhoneService"    : ["Yes"] * 10,
        "MultipleLines"   : ["No","Yes","No","Yes","No","Yes","No","Yes","No","Yes"],
        "InternetService" : ["DSL","Fiber optic","No","DSL","Fiber optic",
                              "DSL","No","Fiber optic","DSL","DSL"],
        "OnlineSecurity"  : ["No","Yes","No","Yes","No","Yes","No","Yes","No","Yes"],
        "OnlineBackup"    : ["Yes","No","No","Yes","No","No","Yes","No","No","Yes"],
        "DeviceProtection": ["No","Yes","No","No","Yes","No","Yes","No","Yes","No"],
        "TechSupport"     : ["No","No","No","Yes","No","Yes","No","No","Yes","No"],
        "StreamingTV"     : ["Yes","Yes","No","No","Yes","No","No","Yes","Yes","No"],
        "StreamingMovies" : ["No","Yes","No","Yes","No","No","Yes","No","Yes","No"],
        "Contract"        : ["Month-to-month","One year","Month-to-month",
                              "Two year","Month-to-month","One year",
                              "Two year","Month-to-month","One year","Two year"],
        "PaperlessBilling": ["Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes"],
        "PaymentMethod"   : ["Electronic check","Mailed check","Bank transfer (automatic)",
                              "Credit card (automatic)","Electronic check","Mailed check",
                              "Bank transfer (automatic)","Credit card (automatic)",
                              "Electronic check","Mailed check"],
        "MonthlyCharges"  : [29.85, 56.95, 0.0, 97.35, 82.70,
                              53.85, 20.65, 75.50, 45.20, 65.10],
        "TotalCharges"    : [29.85, 1889.50, 0.0, 5611.45, 820.70,
                              1346.25, 1397.47, 453.0, 2169.6, 1953.0],
        "Churn"           : [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        "ChurnLabel"      : ["Yes","No","Yes","No","Yes","No","No","Yes","No","No"],
    })


# =============================================================================
# TESTS
# =============================================================================

class TestCreateBusinessFeatures:

    def test_service_count_created(self, sample_df):
        df = create_business_features(sample_df)
        assert "ServiceCount" in df.columns

    def test_service_count_range(self, sample_df):
        df = create_business_features(sample_df)
        assert df["ServiceCount"].min() >= 0
        assert df["ServiceCount"].max() <= 6

    def test_clv_equals_total_charges(self, sample_df):
        df = create_business_features(sample_df)
        assert (df["CLV"] == df["TotalCharges"]).all()

    def test_is_new_customer_correct(self, sample_df):
        df = create_business_features(sample_df)
        # tenure=1 and tenure=0 and tenure=12 → IsNewCustomer=1
        new_mask = df["tenure"] <= 12
        assert (df.loc[new_mask, "IsNewCustomer"] == 1).all()
        assert (df.loc[~new_mask, "IsNewCustomer"] == 0).all()

    def test_is_long_term_correct(self, sample_df):
        df = create_business_features(sample_df)
        long_mask = df["tenure"] > 48
        assert (df.loc[long_mask, "IsLongTermCustomer"] == 1).all()

    def test_avg_monthly_spend_no_division_error(self, sample_df):
        df = create_business_features(sample_df)
        assert df["AvgMonthlySpend"].isna().sum() == 0
        assert np.isinf(df["AvgMonthlySpend"]).sum() == 0

    def test_has_no_addons_flag(self, sample_df):
        df = create_business_features(sample_df)
        assert "HasNoAddOns" in df.columns
        # Every row where ServiceCount=0 should have HasNoAddOns=1
        zero_svc = df["ServiceCount"] == 0
        assert (df.loc[zero_svc, "HasNoAddOns"] == 1).all()

    def test_all_10_features_created(self, sample_df):
        df = create_business_features(sample_df)
        expected = [
            "ServiceCount", "AvgMonthlySpend", "ChargeToTenureRatio",
            "CLV", "TenureGroup_num", "ChargesBand_num", "ContractRisk",
            "IsNewCustomer", "IsLongTermCustomer", "HasNoAddOns"
        ]
        for feat in expected:
            assert feat in df.columns, f"Missing feature: {feat}"


class TestEncodeCategorical:

    def _with_features(self, sample_df):
        return create_business_features(sample_df)

    def test_binary_cols_are_int(self, sample_df):
        df = self._with_features(sample_df)
        df = encode_categorical_features(df)
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            assert df[col].dtype in [int, np.int64, np.int32], f"{col} should be int"
            assert set(df[col].unique()).issubset({0, 1}), f"{col} values not in {{0,1}}"

    def test_contract_ordinal_encoded(self, sample_df):
        df = self._with_features(sample_df)
        df = encode_categorical_features(df)
        assert set(df["Contract"].unique()).issubset({0, 1, 2})

    def test_ohe_columns_created(self, sample_df):
        df = self._with_features(sample_df)
        df = encode_categorical_features(df)
        assert "PaymentMethod_Electronic check" in df.columns

    def test_no_string_columns_remain(self, sample_df):
        df = self._with_features(sample_df)
        df = encode_categorical_features(df)
        df = drop_non_ml_columns(df)
        str_cols = [c for c in df.columns if df[c].dtype == object]
        assert len(str_cols) == 0, f"String columns remain: {str_cols}"


class TestDropColumns:

    def test_customer_id_removed(self, sample_df):
        df = create_business_features(sample_df)
        df = encode_categorical_features(df)
        df = drop_non_ml_columns(df)
        assert "customerID" not in df.columns

    def test_churn_label_removed(self, sample_df):
        df = create_business_features(sample_df)
        df = encode_categorical_features(df)
        df = drop_non_ml_columns(df)
        assert "ChurnLabel" not in df.columns

    def test_target_churn_remains(self, sample_df):
        df = create_business_features(sample_df)
        df = encode_categorical_features(df)
        df = drop_non_ml_columns(df)
        assert "Churn" in df.columns


class TestSplitAndScale:

    def _get_ml_df(self, sample_df):
        df = create_business_features(sample_df)
        df = encode_categorical_features(df)
        df = drop_non_ml_columns(df)
        return df

    def test_split_preserves_rows(self, sample_df):
        df = self._get_ml_df(sample_df)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.3, random_state=42)
        assert len(X_train) + len(X_test) == len(df)

    def test_no_target_in_X(self, sample_df):
        df = self._get_ml_df(sample_df)
        X_train, X_test, y_train, y_test = split_data(df)
        assert "Churn" not in X_train.columns
        assert "Churn" not in X_test.columns

    def test_scaler_fitted_on_train_only(self, sample_df):
        df = self._get_ml_df(sample_df)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.3, random_state=42)
        X_tr_sc, X_te_sc, scaler = scale_features(X_train, X_test)
        # After StandardScaler, training set scaled columns should be ~N(0,1)
        if "tenure" in X_tr_sc.columns:
            assert abs(X_tr_sc["tenure"].mean()) < 0.5   # close to 0
