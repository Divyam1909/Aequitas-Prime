"""
Unit tests for src/bias_engine/counterfactual.py

Uses a deliberately biased stub model to verify that decision flips
produce CRITICAL risk and stable decisions produce CLEAR risk.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.bias_engine.counterfactual import run_counterfactual_check, CounterfactualResult


class _BiasedStubModel:
    """Always approves when sex==1 (Male), always denies when sex==0."""
    def predict(self, X):
        return (X["sex"].values >= 1).astype(int)

    def predict_proba(self, X):
        preds = self.predict(X)
        # High confidence predictions
        proba = np.where(preds == 1,
                         np.column_stack([0.05 * np.ones(len(X)), 0.95 * np.ones(len(X))]),
                         np.column_stack([0.95 * np.ones(len(X)), 0.05 * np.ones(len(X))]))
        return proba


class _FairStubModel:
    """Always predicts 1 regardless of demographics."""
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([0.1 * np.ones(len(X)), 0.9 * np.ones(len(X))])


FEATURE_NAMES = ["age", "education_num", "hours_per_week", "sex"]

MALE_FEATURES   = {"age": 35, "education_num": 13, "hours_per_week": 40, "sex": 1}
FEMALE_FEATURES = {"age": 35, "education_num": 13, "hours_per_week": 40, "sex": 0}


class TestCounterfactualCheck:
    def test_biased_model_flips_decision(self, base_config):
        """Biased model must produce CRITICAL when decision flips for Male→Female."""
        model = _BiasedStubModel()
        result = run_counterfactual_check(
            model, MALE_FEATURES, base_config, FEATURE_NAMES
        )
        assert isinstance(result, CounterfactualResult)
        assert result.risk_level == "CRITICAL", \
            f"Expected CRITICAL from biased model, got {result.risk_level}"
        assert not result.is_fair

    def test_fair_model_clear_risk(self, base_config):
        """Fair model (same prediction always) must produce CLEAR."""
        model = _FairStubModel()
        result = run_counterfactual_check(
            model, MALE_FEATURES, base_config, FEATURE_NAMES
        )
        assert result.risk_level == "CLEAR", \
            f"Expected CLEAR from fair model, got {result.risk_level}"
        assert result.is_fair

    def test_result_fields_populated(self, base_config):
        model = _BiasedStubModel()
        result = run_counterfactual_check(
            model, MALE_FEATURES, base_config, FEATURE_NAMES
        )
        assert result.protected_attr == "sex"
        assert result.original_value in (0, 1)
        assert result.flipped_value in (0, 1)
        assert result.original_value != result.flipped_value
        assert 0.0 <= result.original_confidence <= 1.0
        assert isinstance(result.message, str) and len(result.message) > 0

    def test_confidence_delta_non_negative(self, base_config):
        model = _BiasedStubModel()
        result = run_counterfactual_check(
            model, MALE_FEATURES, base_config, FEATURE_NAMES
        )
        assert result.confidence_delta >= 0
