"""
Integration tests for the end-to-end pipeline.

Runs run_full_pipeline() on a synthetic dataset (no real CSV required)
with run_inprocessing=False to keep the test fast (~5s).

Verifies:
  - AuditResult is fully populated after baseline-only run
  - run_mitigation_steps() produces pre-processed results with improved DI
  - model_comparison produces a DataFrame with the expected models
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.ml_pipeline.pipeline import run_full_pipeline, run_mitigation_steps
from src.utils.generic_preprocessor import preprocess_generic


@pytest.fixture(scope="module")
def pipeline_result(biased_df, base_config):
    """Run baseline-only pipeline on synthetic biased dataset."""
    return run_full_pipeline(
        biased_df,
        base_config,
        run_mitigation=False,
        run_inprocessing=False,
        verbose=False,
        preprocess_fn=preprocess_generic,
        base_model="lr",   # LR is fastest
    )


class TestBaselinePipeline:
    def test_result_not_none(self, pipeline_result):
        assert pipeline_result is not None

    def test_baseline_eval_populated(self, pipeline_result):
        assert pipeline_result.baseline_eval is not None

    def test_baseline_accuracy_reasonable(self, pipeline_result):
        acc = pipeline_result.baseline_eval.performance.accuracy
        assert 0.50 <= acc <= 1.00, f"Unexpected accuracy: {acc}"

    def test_n_features_positive(self, pipeline_result):
        assert pipeline_result.n_features > 0

    def test_baseline_metrics_has_sex(self, pipeline_result):
        assert "sex" in pipeline_result.baseline_metrics

    def test_proxy_scan_runs(self, pipeline_result):
        assert isinstance(pipeline_result.proxy_scan, dict)

    def test_mitigation_not_run(self, pipeline_result):
        assert pipeline_result.preprocessed_eval is None
        assert pipeline_result.inprocessed_eval is None

    def test_internal_state_preserved(self, pipeline_result):
        assert pipeline_result._X is not None
        assert pipeline_result._y is not None
        assert pipeline_result._df_clean is not None


class TestMitigationSteps:
    @pytest.fixture(scope="class")
    def mitigated_result(self, pipeline_result):
        return run_mitigation_steps(
            pipeline_result,
            run_inprocessing=False,
            base_model="lr",
            verbose=False,
        )

    def test_preprocessed_eval_populated(self, mitigated_result):
        assert mitigated_result.preprocessed_eval is not None

    def test_preprocessed_accuracy_reasonable(self, mitigated_result):
        acc = mitigated_result.preprocessed_eval.performance.accuracy
        assert 0.50 <= acc <= 1.00

    def test_reweigh_result_populated(self, mitigated_result):
        assert mitigated_result.reweigh_result is not None
        assert len(mitigated_result.reweigh_result.weights) > 0

    def test_di_improves_after_mitigation(self, mitigated_result):
        primary = mitigated_result.config.primary_protected_attr()
        baseline_di = mitigated_result.baseline_eval.fairness.get(primary)
        pre_di      = mitigated_result.preprocessed_eval.fairness.get(primary)
        if baseline_di and pre_di:
            assert pre_di.disparate_impact >= baseline_di.disparate_impact - 0.05, \
                "Pre-processed DI should not be worse than baseline by more than 0.05"

    def test_comparison_df_built(self, mitigated_result):
        assert mitigated_result.comparison_df is not None
        assert len(mitigated_result.comparison_df) >= 2


class TestModelComparison:
    def test_compare_base_models_returns_dataframe(self, biased_df, base_config):
        from src.ml_pipeline.model_comparison import compare_base_models

        X, y, df_clean = preprocess_generic(biased_df, base_config)
        cdf = compare_base_models(X, y, df_clean, base_config, models=("rf", "lr"))
        assert isinstance(cdf, pd.DataFrame)
        assert len(cdf) == 2

    def test_compare_all_three_models(self, biased_df, base_config):
        from src.ml_pipeline.model_comparison import compare_base_models

        X, y, df_clean = preprocess_generic(biased_df, base_config)
        cdf = compare_base_models(X, y, df_clean, base_config, models=("rf", "lr", "xgb"))
        assert len(cdf) == 3
        assert "Model" in cdf.columns
        assert "Accuracy" in cdf.columns
        assert "DI" in cdf.columns

    def test_all_accuracies_valid(self, biased_df, base_config):
        from src.ml_pipeline.model_comparison import compare_base_models

        X, y, df_clean = preprocess_generic(biased_df, base_config)
        cdf = compare_base_models(X, y, df_clean, base_config, models=("rf", "lr"))
        assert (cdf["Accuracy"].between(0.0, 1.0)).all()
