"""
Unit tests for src/bias_engine/detector.py

Verify that the 6 fairness metrics produce values in the expected direction
against synthetic datasets with known ground-truth bias levels.
"""
from __future__ import annotations
import numpy as np
import pytest
from src.bias_engine.detector import (
    compute_disparate_impact,
    compute_spd,
    compute_all_metrics,
    compute_metrics_for_all_attrs,
    DI_THRESHOLD,
    SPD_THRESHOLD,
)
from src.utils.schema import DatasetConfig


def _biased_pred(df):
    """Deterministic predictions matching true labels."""
    return df["income"].to_numpy()


class TestDisparateImpact:
    def test_biased_di_below_threshold(self, biased_df, base_config):
        di = compute_disparate_impact(biased_df, base_config)
        assert di < DI_THRESHOLD, f"Expected biased DI < {DI_THRESHOLD}, got {di:.3f}"

    def test_fair_di_near_one(self, fair_df, base_config):
        di = compute_disparate_impact(fair_df, base_config)
        assert 0.80 <= di <= 1.20, f"Expected fair DI ≈ 1.0, got {di:.3f}"

    def test_di_positive(self, biased_df, base_config):
        di = compute_disparate_impact(biased_df, base_config)
        assert di > 0, "DI must be positive"

    def test_di_uses_positive_label(self, biased_df, base_config):
        di_normal = compute_disparate_impact(biased_df, base_config)
        flipped = DatasetConfig(
            protected_attrs=["sex"],
            target_col="income",
            privileged_values={"sex": "Male"},
            positive_label=0,
            dataset_name="Synthetic",
        )
        di_flipped = compute_disparate_impact(biased_df, flipped)
        assert abs(di_normal - di_flipped) > 0.05, \
            "Flipping positive_label should change DI materially"


class TestSPD:
    def test_biased_spd_negative(self, biased_df, base_config):
        spd = compute_spd(biased_df, base_config)
        assert spd < SPD_THRESHOLD, f"Expected SPD < {SPD_THRESHOLD}, got {spd:.3f}"

    def test_fair_spd_near_zero(self, fair_df, base_config):
        spd = compute_spd(fair_df, base_config)
        assert abs(spd) < 0.10, f"Expected fair SPD ≈ 0, got {spd:.3f}"


class TestComputeAllMetrics:
    def test_returns_all_fields_with_y_pred(self, biased_df, base_config):
        y_pred = _biased_pred(biased_df)
        m = compute_all_metrics(biased_df, base_config, y_pred=y_pred, attr="sex")
        assert not np.isnan(m.disparate_impact)
        assert not np.isnan(m.statistical_parity_diff)
        assert not np.isnan(m.equalized_odds_diff)
        assert not np.isnan(m.equal_opportunity_diff)

    def test_biased_severity_not_clear(self, biased_df, base_config):
        y_pred = _biased_pred(biased_df)
        m = compute_all_metrics(biased_df, base_config, y_pred=y_pred, attr="sex")
        assert m.severity in ("CRITICAL", "WARNING"), \
            f"Expected CRITICAL/WARNING for biased data, got {m.severity}"

    def test_fair_severity_clear(self, fair_df, base_config):
        y_pred = _biased_pred(fair_df)
        m = compute_all_metrics(fair_df, base_config, y_pred=y_pred, attr="sex")
        assert m.severity == "CLEAR", \
            f"Expected CLEAR for fair data, got {m.severity}"

    def test_compute_all_attrs_returns_dict_with_key(self, biased_df, base_config):
        result = compute_metrics_for_all_attrs(biased_df, base_config)
        assert isinstance(result, dict)
        assert "sex" in result

    def test_group_counts_sum_to_total(self, biased_df, base_config):
        m = compute_all_metrics(biased_df, base_config, attr="sex")
        assert m.privileged_count + m.unprivileged_count == len(biased_df)

    def test_post_training_metrics_nan_without_y_pred(self, biased_df, base_config):
        m = compute_all_metrics(biased_df, base_config)
        assert np.isnan(m.equalized_odds_diff)
        assert np.isnan(m.equal_opportunity_diff)
