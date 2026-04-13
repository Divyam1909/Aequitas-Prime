"""
Model evaluation — performance metrics + fairness metrics post-training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src.utils.schema import DatasetConfig
from src.bias_engine.detector import compute_all_metrics, MetricsResult
from src.ml_pipeline.trainer import TrainResult


@dataclass
class PerformanceMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float


@dataclass
class EvalResult:
    model_type: str
    performance: PerformanceMetrics
    fairness: dict[str, MetricsResult]   # attr → MetricsResult


def evaluate(
    train_result: TrainResult,
    df_test_clean: pd.DataFrame,
    config: DatasetConfig,
) -> EvalResult:
    """
    Evaluate both performance and fairness for a trained model.

    Parameters
    ----------
    train_result : TrainResult from trainer.py
    df_test_clean : raw-label test slice (same rows as X_test, with string protected attrs)
    config : DatasetConfig
    """
    y_test = train_result.y_test
    y_pred = train_result.y_pred

    # ── Performance ──────────────────────────────────────────────────────────
    try:
        auc = roc_auc_score(y_test, y_pred)
    except Exception:
        auc = float("nan")

    perf = PerformanceMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(auc),
    )

    # ── Fairness ─────────────────────────────────────────────────────────────
    fairness: dict[str, MetricsResult] = {}
    for attr in config.protected_attrs:
        # Align df_test_clean to the same rows as X_test
        df_aligned = df_test_clean.loc[train_result.X_test.index] if hasattr(train_result.X_test, "index") else df_test_clean
        try:
            m = compute_all_metrics(
                df_aligned, config,
                y_pred=y_pred,
                attr=attr,
            )
        except Exception as e:
            print(f"  [evaluator] fairness metric error for {attr}: {e}")
            m = MetricsResult(protected_attr=attr, privileged_value=config.privileged_values.get(attr))
        fairness[attr] = m

    return EvalResult(
        model_type=train_result.model_type,
        performance=perf,
        fairness=fairness,
    )


def compare_evals(evals: list[EvalResult], primary_attr: str) -> pd.DataFrame:
    """
    Build a comparison DataFrame across model types.
    Useful for the Before/After UI table.
    """
    rows = []
    for ev in evals:
        m = ev.fairness.get(primary_attr)
        row = {
            "Model":     ev.model_type,
            "Accuracy":  round(ev.performance.accuracy, 4),
            "F1":        round(ev.performance.f1, 4),
            "ROC-AUC":   round(ev.performance.roc_auc, 4),
            "DI":        round(m.disparate_impact, 3)        if m and not np.isnan(m.disparate_impact) else None,
            "SPD":       round(m.statistical_parity_diff, 3) if m and not np.isnan(m.statistical_parity_diff) else None,
            "EOD":       round(m.equalized_odds_diff, 3)     if m and not np.isnan(m.equalized_odds_diff) else None,
            "EOpD":      round(m.equal_opportunity_diff, 3)  if m and not np.isnan(m.equal_opportunity_diff) else None,
            "PP":        round(m.predictive_parity_diff, 3)  if m and not np.isnan(m.predictive_parity_diff) else None,
            "FNRP":      round(m.fnr_parity_diff, 3)         if m and not np.isnan(m.fnr_parity_diff) else None,
            "Severity":  m.severity if m else "UNKNOWN",
        }
        rows.append(row)
    return pd.DataFrame(rows)
