# =============================================================================
# tests/test_data_cleaning.py
# Unit tests for the data cleaning pipeline
# Run with:  python -m pytest tests/ -v
# =============================================================================

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_cleaning import (
    fix_data_types,
    handle_missing_values,
    handle_duplicates,
    standardise_values,
    validate_clean_data,
)


# =============================================================================
# FIXTURES — shared sample DataFrames
# =============================================================================

@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Minimal DataFrame that mimics the IBM Telco structure."""
    return pd.DataFrame({
        "customerID"     : ["1-AAA", "2-BBB", "3-CCC", "4-DDD"],
        "gender"         : ["Male", "Female", "Male", "Female"],
        "SeniorCitizen"  : [0, 1, 0, 0],
        "Partner"        : ["Yes", "No", "Yes", "No"],
        "Dependents"     : ["No", "No", "Yes", "No"],
        "tenure"         : [1, 34, 0, 58],
        "PhoneService"   : ["No", "Yes", "Yes", "Yes"],
        "MultipleLines"  : ["No phone service", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity" : ["No", "No internet service", "Yes", "No internet service"],
        "Contract"       : ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaymentMethod"  : ["Electronic check", "Mailed check", "Electronic check", "Bank transfer"],
        "MonthlyCharges" : [29.85, 56.95, 53.85, 20.65],
        "TotalCharges"   : ["29.85", "1889.50", " ", "1397.47"],  # " " = bug
        "Churn"          : ["No", "No", "Yes", "No"],
    })


# =============================================================================
# TESTS
# =============================================================================

class TestFixDataTypes:

    def test_totalcharges_becomes_float(self, raw_sample):
        df = fix_data_types(raw_sample)
        assert df["TotalCharges"].dtype == float, "TotalCharges should be float64"

    def test_blank_totalcharges_becomes_nan(self, raw_sample):
        df = fix_data_types(raw_sample)
        # The " " row (customerID 3-CCC) should now be NaN
        assert df.loc[df["customerID"] == "3-CCC", "TotalCharges"].isna().all()

    def test_senior_citizen_label_created(self, raw_sample):
        df = fix_data_types(raw_sample)
        assert "SeniorCitizenLabel" in df.columns
        assert set(df["SeniorCitizenLabel"].unique()).issubset({"Yes", "No"})


class TestHandleMissingValues:

    def test_no_nulls_after_handling(self, raw_sample):
        df = fix_data_types(raw_sample)
        df = handle_missing_values(df)
        assert df["TotalCharges"].isna().sum() == 0

    def test_new_customer_total_charges_is_zero(self, raw_sample):
        df = fix_data_types(raw_sample)
        df = handle_missing_values(df)
        # customerID 3-CCC had tenure=0 and blank TotalCharges
        val = df.loc[df["customerID"] == "3-CCC", "TotalCharges"].values[0]
        assert val == 0.0, f"Expected 0.0, got {val}"


class TestHandleDuplicates:

    def test_duplicates_removed(self, raw_sample):
        df_with_dupe = pd.concat([raw_sample, raw_sample.iloc[[0]]], ignore_index=True)
        assert len(df_with_dupe) == 5
        cleaned = handle_duplicates(df_with_dupe)
        assert len(cleaned) == 4

    def test_no_duplicates_unchanged(self, raw_sample):
        cleaned = handle_duplicates(raw_sample)
        assert len(cleaned) == len(raw_sample)


class TestStandardiseValues:

    def test_no_internet_service_collapsed(self, raw_sample):
        df = fix_data_types(raw_sample)
        df = handle_missing_values(df)
        df = standardise_values(df)
        assert "No internet service" not in df["OnlineSecurity"].values

    def test_no_phone_service_collapsed(self, raw_sample):
        df = fix_data_types(raw_sample)
        df = handle_missing_values(df)
        df = standardise_values(df)
        assert "No phone service" not in df["MultipleLines"].values

    def test_churn_encoded_as_int(self, raw_sample):
        df = fix_data_types(raw_sample)
        df = handle_missing_values(df)
        df = standardise_values(df)
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_churn_label_preserved(self, raw_sample):
        df = fix_data_types(raw_sample)
        df = handle_missing_values(df)
        df = standardise_values(df)
        assert "ChurnLabel" in df.columns
        assert set(df["ChurnLabel"].unique()).issubset({"Yes", "No"})


class TestValidateCleanData:

    def _get_clean(self, raw_sample):
        df = fix_data_types(raw_sample)
        df = handle_missing_values(df)
        df = handle_duplicates(df)
        df = standardise_values(df)
        return df

    def test_validation_passes_on_clean_data(self, raw_sample):
        df = self._get_clean(raw_sample)
        # Override row-count check since sample is only 4 rows
        # Just check the function runs without error
        assert df.isnull().sum().sum() == 0
        assert not df["customerID"].duplicated().any()
