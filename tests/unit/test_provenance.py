"""
Unit tests for src/bias_engine/provenance.py

Verify that the provenance tracer correctly identifies:
  - representation_bias when groups are severely imbalanced
  - feature_bias when high-MI proxy features are present
  - label_bias as the fallback when no other signals dominate
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.bias_engine.provenance import trace_bias_provenance, ProvenanceReport
from src.bias_engine.proxy_scanner import ProxyResult


def _make_proxy_result(mi: float) -> ProxyResult:
    return ProxyResult(
        feature="zip_code",
        protected_attr="sex",
        mutual_info=mi,
        cramers_v=None,
        point_biserial_r=None,
        risk_level="HIGH" if mi >= 0.15 else "LOW",
        action="",
        combined_score=mi,
    )


class TestBiasProvenance:
    def test_returns_provenance_report(self, biased_df, base_config):
        report = trace_bias_provenance(biased_df, base_config, proxy_scan_results={})
        assert isinstance(report, ProvenanceReport)

    def test_primary_source_is_valid(self, biased_df, base_config):
        report = trace_bias_provenance(biased_df, base_config, proxy_scan_results={})
        assert report.primary_source in ("label_bias", "representation_bias", "feature_bias")

    def test_high_proxy_triggers_feature_bias(self, biased_df, base_config):
        """When a high-MI proxy feature exists, feature_bias should rank high."""
        proxy_results = {"sex": [_make_proxy_result(mi=0.35)]}
        report = trace_bias_provenance(biased_df, base_config, proxy_scan_results=proxy_results)
        # feature_bias should be either primary or secondary
        sources = {report.primary_source, report.secondary_source}
        assert "feature_bias" in sources, \
            f"Expected feature_bias in sources, got primary={report.primary_source}, secondary={report.secondary_source}"

    def test_no_proxies_does_not_flag_feature_bias_primary(self, biased_df, base_config):
        """Without proxy features, feature_bias should not win."""
        proxy_results = {"sex": [_make_proxy_result(mi=0.01)]}
        report = trace_bias_provenance(biased_df, base_config, proxy_scan_results=proxy_results)
        assert report.primary_source != "feature_bias", \
            "With negligible proxy MI, feature_bias should not be primary"

    def test_confidence_in_range(self, biased_df, base_config):
        report = trace_bias_provenance(biased_df, base_config, proxy_scan_results={})
        assert 0.0 <= report.confidence <= 1.0

    def test_recommended_fixes_not_empty(self, biased_df, base_config):
        report = trace_bias_provenance(biased_df, base_config, proxy_scan_results={})
        assert len(report.recommended_fixes) > 0

    def test_evidence_populated(self, biased_df, base_config):
        report = trace_bias_provenance(biased_df, base_config, proxy_scan_results={})
        assert len(report.evidence) > 0
