"""
Unit tests for src/bias_engine/proxy_scanner.py

Verify that a feature with high mutual information to the protected attribute
is detected as HIGH risk, while an uncorrelated feature is NEGLIGIBLE.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.bias_engine.proxy_scanner import scan_proxies, ProxyResult, RISK_THRESHOLDS


class TestProxyScanner:
    def test_correlated_feature_flagged_high(self, biased_df, base_config):
        """
        'zip_code' is constructed as near-perfect copy of sex → must be HIGH.
        """
        df = biased_df.copy()
        # zip_code = 10001 for Male, 10002 for Female (nearly perfect proxy)
        df["zip_code"] = (df["sex"] == "Male").astype(int)

        from src.utils.generic_preprocessor import preprocess_generic
        X, y, _ = preprocess_generic(df, base_config)

        results = scan_proxies(X, base_config)
        assert "sex" in results

        zip_results = [r for r in results["sex"] if r.feature == "zip_code"]
        assert zip_results, "zip_code should appear in proxy scan results"
        assert zip_results[0].risk_level == "HIGH", \
            f"Expected HIGH risk for perfect proxy, got {zip_results[0].risk_level}"

    def test_random_feature_low_or_negligible(self, biased_df, base_config):
        """A pure-noise feature should not be flagged as HIGH or MEDIUM."""
        rng = np.random.default_rng(99)
        df = biased_df.copy()
        df["noise"] = rng.integers(0, 100, len(df))

        from src.utils.generic_preprocessor import preprocess_generic
        X, y, _ = preprocess_generic(df, base_config)

        results = scan_proxies(X, base_config)
        noise_results = [r for r in results.get("sex", []) if r.feature == "noise"]
        for r in noise_results:
            assert r.risk_level in ("LOW", "NEGLIGIBLE"), \
                f"Random noise feature should not be HIGH/MEDIUM, got {r.risk_level}"

    def test_returns_dict_keyed_by_attr(self, preprocessed_biased, base_config):
        X, y, df_clean = preprocessed_biased
        results = scan_proxies(X, base_config)
        assert isinstance(results, dict)

    def test_each_result_is_proxy_result(self, preprocessed_biased, base_config):
        X, y, df_clean = preprocessed_biased
        results = scan_proxies(X, base_config)
        for attr, proxy_list in results.items():
            for r in proxy_list:
                assert isinstance(r, ProxyResult)
                assert r.risk_level in ("HIGH", "MEDIUM", "LOW", "NEGLIGIBLE")
                assert r.mutual_info >= 0
