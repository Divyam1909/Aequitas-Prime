"""
Unit tests for src/bias_engine/mitigator.py (Reweighing).

Verify that reweighing assigns higher weights to underrepresented
(unprivileged + positive outcome) combinations.
"""
from __future__ import annotations
import numpy as np
import pytest
from src.bias_engine.mitigator import apply_reweighing


class TestReweighing:
    def test_returns_weights_same_length(self, biased_df, base_config):
        rw = apply_reweighing(biased_df, base_config, attr="sex")
        assert len(rw.weights) == len(biased_df)

    def test_weights_positive(self, biased_df, base_config):
        rw = apply_reweighing(biased_df, base_config, attr="sex")
        assert (rw.weights > 0).all(), "All sample weights must be positive"

    def test_unprivileged_positive_upweighted(self, biased_df, base_config):
        """
        Reweighing must up-weight female+positive and down-weight male+positive
        because females are underrepresented in positive outcomes.
        """
        rw = apply_reweighing(biased_df, base_config, attr="sex")
        df = biased_df.copy()
        df["_w"] = rw.weights

        w_female_pos = df.loc[(df["sex"] == "Female") & (df["income"] == 1), "_w"].mean()
        w_male_pos   = df.loc[(df["sex"] == "Male")   & (df["income"] == 1), "_w"].mean()

        assert w_female_pos > w_male_pos, (
            f"Female+positive should be upweighted vs Male+positive. "
            f"Got female={w_female_pos:.3f}, male={w_male_pos:.3f}"
        )

    def test_group_weight_map_populated(self, biased_df, base_config):
        rw = apply_reweighing(biased_df, base_config, attr="sex")
        assert isinstance(rw.group_weight_map, dict)
        assert len(rw.group_weight_map) > 0

    def test_weights_sum_approximately_equal_n(self, biased_df, base_config):
        """AIF360 Reweighing normalises so that sum(weights) ≈ N."""
        rw = apply_reweighing(biased_df, base_config, attr="sex")
        assert abs(rw.weights.sum() - len(biased_df)) < len(biased_df) * 0.05, \
            "Reweighing weights should sum to approximately N"
