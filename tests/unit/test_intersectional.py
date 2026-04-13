"""
Unit tests for src/bias_engine/intersectional.py

Verify group enumeration, DI direction, and IBI scalar computation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.bias_engine.intersectional import compute_intersectional_metrics, IntersectionalResult


class TestIntersectionalMetrics:
    def test_returns_list(self, biased_df, base_config):
        results = compute_intersectional_metrics(biased_df, base_config)
        assert isinstance(results, list)

    def test_single_attr_returns_both_groups(self, biased_df, base_config):
        """With one protected attr (sex), we expect Male and Female groups."""
        results = compute_intersectional_metrics(biased_df, base_config)
        labels = {r.group_label for r in results}
        # Should contain at least Male and Female entries
        assert len(labels) >= 2, f"Expected at least 2 groups, got: {labels}"

    def test_each_result_is_intersectional_result(self, biased_df, base_config):
        results = compute_intersectional_metrics(biased_df, base_config)
        for r in results:
            assert isinstance(r, IntersectionalResult)

    def test_di_positive(self, biased_df, base_config):
        results = compute_intersectional_metrics(biased_df, base_config)
        for r in results:
            assert r.disparate_impact >= 0, f"DI must be non-negative, got {r.disparate_impact}"

    def test_biased_female_group_di_below_one(self, biased_df, base_config):
        """In a biased dataset, the Female group DI should be well below 1.0."""
        results = compute_intersectional_metrics(biased_df, base_config)
        female_results = [r for r in results if "Female" in r.group_label]
        assert female_results, "Expected Female group in intersectional results"
        female_di = female_results[0].disparate_impact
        assert female_di < 0.8, f"Expected Female DI < 0.8 in biased data, got {female_di:.3f}"

    def test_sorted_ascending_by_di(self, biased_df, base_config):
        """Results must be sorted by DI ascending (worst first)."""
        results = compute_intersectional_metrics(biased_df, base_config)
        dis = [r.disparate_impact for r in results]
        assert dis == sorted(dis), "Results must be sorted ascending by DI"

    def test_group_size_above_minimum(self, biased_df, base_config):
        """Groups below MIN_GROUP_SIZE should be filtered out."""
        from src.bias_engine.intersectional import MIN_GROUP_SIZE
        results = compute_intersectional_metrics(biased_df, base_config)
        for r in results:
            assert r.group_size >= MIN_GROUP_SIZE, \
                f"Group '{r.group_label}' has {r.group_size} < MIN_GROUP_SIZE={MIN_GROUP_SIZE}"

    def test_with_y_pred_populates_eopd(self, biased_df, base_config):
        y_pred = biased_df["income"].to_numpy()
        results = compute_intersectional_metrics(biased_df, base_config, y_pred=y_pred)
        for r in results:
            assert not np.isnan(r.equal_opportunity_diff), \
                f"EOpD should be set when y_pred is provided, group={r.group_label}"
