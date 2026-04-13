"""
Unit tests for src/utils/generic_preprocessor.py

Verify auto-detection of protected attributes and target column,
and correct encoding behaviour.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.utils.generic_preprocessor import (
    auto_detect_protected_attrs,
    auto_detect_target,
    preprocess_generic,
)
from src.utils.schema import DatasetConfig


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame({
        "age":            rng.integers(20, 60, n),
        "sex":            rng.choice(["Male", "Female"], n),
        "race":           rng.choice(["White", "Black", "Asian"], n),
        "education_num":  rng.integers(8, 16, n),
        "hours_per_week": rng.integers(30, 60, n),
        "income":         rng.choice([0, 1], n),
    })


class TestAutoDetectProtectedAttrs:
    def test_detects_sex_and_race(self, sample_df):
        found = auto_detect_protected_attrs(sample_df, target_col="income")
        assert "sex" in found
        assert "race" in found

    def test_excludes_target_col(self, sample_df):
        found = auto_detect_protected_attrs(sample_df, target_col="income")
        assert "income" not in found

    def test_returns_list(self, sample_df):
        result = auto_detect_protected_attrs(sample_df)
        assert isinstance(result, list)

    def test_no_duplicates(self, sample_df):
        result = auto_detect_protected_attrs(sample_df, target_col="income")
        assert len(result) == len(set(result))


class TestAutoDetectTarget:
    def test_detects_income(self, sample_df):
        target = auto_detect_target(sample_df)
        assert target == "income"

    def test_fallback_to_last_col(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "z": [0, 1]})
        target = auto_detect_target(df)
        assert target == "z"


class TestPreprocessGeneric:
    @pytest.fixture
    def config(self):
        return DatasetConfig(
            protected_attrs=["sex"],
            target_col="income",
            privileged_values={"sex": "Male"},
            positive_label=1,
            dataset_name="Test",
        )

    def test_returns_three_objects(self, sample_df, config):
        result = preprocess_generic(sample_df, config)
        assert len(result) == 3

    def test_X_has_no_target_col(self, sample_df, config):
        X, y, _ = preprocess_generic(sample_df, config)
        assert "income" not in X.columns

    def test_y_is_binary(self, sample_df, config):
        X, y, _ = preprocess_generic(sample_df, config)
        assert set(y.unique()).issubset({0, 1})

    def test_X_is_all_numeric(self, sample_df, config):
        X, y, _ = preprocess_generic(sample_df, config)
        assert X.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all(), \
            "All X columns must be numeric after preprocessing"

    def test_sex_encoded_as_binary(self, sample_df, config):
        X, y, _ = preprocess_generic(sample_df, config)
        assert set(X["sex"].unique()).issubset({0.0, 1.0})

    def test_df_clean_preserves_original_sex_values(self, sample_df, config):
        _, _, df_clean = preprocess_generic(sample_df, config)
        assert set(df_clean["sex"].unique()).issubset({"Male", "Female"})

    def test_no_rows_dropped_without_nan(self, sample_df, config):
        X, y, df_clean = preprocess_generic(sample_df, config)
        assert len(X) == len(sample_df)

    def test_drops_rows_with_nan_in_key_cols(self, config):
        df = pd.DataFrame({
            "sex":    ["Male", None, "Female", "Male"],
            "income": [1, 0, 1, 0],
        })
        X, y, _ = preprocess_generic(df, config)
        assert len(X) == 3, "Row with None in sex should be dropped"
