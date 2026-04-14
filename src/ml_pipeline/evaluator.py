"""
Model evaluation — performance metrics + fairness metrics post-training.

V2 additions:
  - Dispatches on config.task_type:
      "binary"     — existing 6-metric fairness evaluation (unchanged)
      "multiclass" — macro-averaged performance + per-class DI / EOpD
      "regression" — MAE / MSE / R² + mean outcome parity by group
  - PerformanceMetrics extended with mae, mse, r2 for regression
  - evaluate() returns EvalResult for all task types
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
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
    # V2 regression fields (nan for classification)
    mae: float = float("nan")
    mse: float = float("nan")
    r2: float  = float("nan")


@dataclass
class MulticlassMetrics:
    """Per-class Disparate Impact and EOpD — for multiclass tasks."""
    class_label: Any
    disparate_impact: float
    equal_opportunity_diff: float
    privileged_approval_rate: float
    unprivileged_approval_rate: float


@dataclass
class RegressionFairnessMetrics:
    """Mean outcome parity metrics for regression tasks."""
    protected_attr: str
    mean_outcome_priv: float
    mean_outcome_unpriv: float
    mean_outcome_diff: float       # absolute difference
    mean_outcome_ratio: float      # unpriv / priv
    mse_priv: float
    mse_unpriv: float
    mse_parity_diff: float         # |MSE_priv - MSE_unpriv|


@dataclass
class EvalResult:
    model_type: str
    performance: PerformanceMetrics
    fairness: dict[str, MetricsResult]            # attr → MetricsResult  (binary)
    multiclass_fairness: dict[str, list[MulticlassMetrics]] = field(default_factory=dict)
    regression_fairness: dict[str, RegressionFairnessMetrics] = field(default_factory=dict)
    task_type: str = "binary"


# ── Binary evaluation (unchanged from V1) ─────────────────────────────────────

def _eval_binary(
    train_result: TrainResult,
    df_test_clean: pd.DataFrame,
    config: DatasetConfig,
) -> EvalResult:
    y_test = train_result.y_test
    y_pred = train_result.y_pred

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

    fairness: dict[str, MetricsResult] = {}
    for attr in config.protected_attrs:
        df_aligned = (
            df_test_clean.loc[train_result.X_test.index]
            if hasattr(train_result.X_test, "index")
            else df_test_clean
        )
        try:
            m = compute_all_metrics(df_aligned, config, y_pred=y_pred, attr=attr)
        except Exception as e:
            print(f"  [evaluator] fairness metric error for {attr}: {e}")
            m = MetricsResult(
                protected_attr=attr,
                privileged_value=config.privileged_values.get(attr),
            )
        fairness[attr] = m

    return EvalResult(
        model_type=train_result.model_type,
        performance=perf,
        fairness=fairness,
        task_type="binary",
    )


# ── Multiclass evaluation ─────────────────────────────────────────────────────

def _eval_multiclass(
    train_result: TrainResult,
    df_test_clean: pd.DataFrame,
    config: DatasetConfig,
) -> EvalResult:
    y_test = train_result.y_test
    y_pred = train_result.y_pred

    try:
        auc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    perf = PerformanceMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        precision=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        roc_auc=float(auc),
    )

    # Per-class DI and EOpD relative to privileged group
    df_aligned = (
        df_test_clean.loc[train_result.X_test.index]
        if hasattr(train_result.X_test, "index")
        else df_test_clean
    )
    mc_fairness: dict[str, list[MulticlassMetrics]] = {}

    for attr in config.protected_attrs:
        if attr not in df_aligned.columns:
            continue
        priv_val = config.privileged_values.get(attr)
        priv_mask = df_aligned[attr].astype(str) == str(priv_val)
        class_labels = sorted(np.unique(y_test))
        results = []
        for cls in class_labels:
            # Approval rate = fraction predicted as this class
            priv_rate   = float((y_pred[priv_mask.values] == cls).mean())
            unpriv_rate = float((y_pred[~priv_mask.values] == cls).mean())
            di = unpriv_rate / priv_rate if priv_rate > 0 else float("nan")

            # TPR difference for this class
            priv_tpr   = float(((y_pred[priv_mask.values] == cls) & (y_test.values[priv_mask.values] == cls)).sum() /
                               max((y_test.values[priv_mask.values] == cls).sum(), 1))
            unpriv_tpr = float(((y_pred[~priv_mask.values] == cls) & (y_test.values[~priv_mask.values] == cls)).sum() /
                                max((y_test.values[~priv_mask.values] == cls).sum(), 1))
            eopd = abs(priv_tpr - unpriv_tpr)

            results.append(MulticlassMetrics(
                class_label=cls,
                disparate_impact=di,
                equal_opportunity_diff=eopd,
                privileged_approval_rate=priv_rate,
                unprivileged_approval_rate=unpriv_rate,
            ))
        mc_fairness[attr] = results

    # Provide a compatible binary-style fairness dict (uses majority-class DI)
    # so the rest of the pipeline (comparison table, report) doesn't break
    compat_fairness: dict[str, MetricsResult] = {}
    for attr in config.protected_attrs:
        cls_results = mc_fairness.get(attr, [])
        if cls_results:
            # Use the class with worst (lowest) DI as the headline metric
            worst = min(cls_results, key=lambda c: c.disparate_impact if not np.isnan(c.disparate_impact) else 1.0)
            m = MetricsResult(
                protected_attr=attr,
                privileged_value=config.privileged_values.get(attr),
                disparate_impact=worst.disparate_impact,
                equal_opportunity_diff=worst.equal_opportunity_diff,
                privileged_approval_rate=worst.privileged_approval_rate,
                unprivileged_approval_rate=worst.unprivileged_approval_rate,
            )
            m.failures = []
            if not np.isnan(m.disparate_impact) and m.disparate_impact < 0.80:
                m.failures.append(f"DI = {m.disparate_impact:.3f}")
            m.severity = "CRITICAL" if len(m.failures) > 2 else ("WARNING" if m.failures else "CLEAR")
            compat_fairness[attr] = m

    return EvalResult(
        model_type=train_result.model_type,
        performance=perf,
        fairness=compat_fairness,
        multiclass_fairness=mc_fairness,
        task_type="multiclass",
    )


# ── Regression evaluation ─────────────────────────────────────────────────────

def _eval_regression(
    train_result: TrainResult,
    df_test_clean: pd.DataFrame,
    config: DatasetConfig,
) -> EvalResult:
    y_test = train_result.y_test
    y_pred = train_result.y_pred

    perf = PerformanceMetrics(
        accuracy=float("nan"),   # not applicable for regression
        f1=float("nan"),
        precision=float("nan"),
        recall=float("nan"),
        roc_auc=float("nan"),
        mae=float(mean_absolute_error(y_test, y_pred)),
        mse=float(mean_squared_error(y_test, y_pred)),
        r2=float(r2_score(y_test, y_pred)),
    )

    df_aligned = (
        df_test_clean.loc[train_result.X_test.index]
        if hasattr(train_result.X_test, "index")
        else df_test_clean
    )

    reg_fairness: dict[str, RegressionFairnessMetrics] = {}
    compat_fairness: dict[str, MetricsResult] = {}

    for attr in config.protected_attrs:
        if attr not in df_aligned.columns:
            continue
        priv_val  = config.privileged_values.get(attr)
        priv_mask = df_aligned[attr].astype(str) == str(priv_val)
        pm = priv_mask.values

        y_pred_arr = np.array(y_pred)
        y_test_arr = np.array(y_test)

        mean_p = float(y_pred_arr[pm].mean()) if pm.sum() > 0 else float("nan")
        mean_u = float(y_pred_arr[~pm].mean()) if (~pm).sum() > 0 else float("nan")

        mse_p = float(mean_squared_error(y_test_arr[pm],  y_pred_arr[pm]))  if pm.sum() > 0 else float("nan")
        mse_u = float(mean_squared_error(y_test_arr[~pm], y_pred_arr[~pm])) if (~pm).sum() > 0 else float("nan")

        diff  = abs(mean_p - mean_u) if not (np.isnan(mean_p) or np.isnan(mean_u)) else float("nan")
        ratio = mean_u / mean_p if (mean_p and mean_p > 0) else float("nan")
        mse_parity = abs(mse_p - mse_u) if not (np.isnan(mse_p) or np.isnan(mse_u)) else float("nan")

        reg_fairness[attr] = RegressionFairnessMetrics(
            protected_attr=attr,
            mean_outcome_priv=mean_p,
            mean_outcome_unpriv=mean_u,
            mean_outcome_diff=diff,
            mean_outcome_ratio=ratio,
            mse_priv=mse_p,
            mse_unpriv=mse_u,
            mse_parity_diff=mse_parity,
        )

        # Provide a MetricsResult-compatible entry using ratio as DI proxy
        m = MetricsResult(
            protected_attr=attr,
            privileged_value=priv_val,
            disparate_impact=ratio if not np.isnan(ratio) else float("nan"),
            statistical_parity_diff=(-diff) if not np.isnan(diff) else float("nan"),
            privileged_approval_rate=mean_p,
            unprivileged_approval_rate=mean_u,
        )
        m.failures = []
        if not np.isnan(diff) and diff > 0.1 * abs(mean_p):
            m.failures.append(f"Mean outcome gap = {diff:.3f}")
        m.severity = "WARNING" if m.failures else "CLEAR"
        compat_fairness[attr] = m

    return EvalResult(
        model_type=train_result.model_type,
        performance=perf,
        fairness=compat_fairness,
        regression_fairness=reg_fairness,
        task_type="regression",
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate(
    train_result: TrainResult,
    df_test_clean: pd.DataFrame,
    config: DatasetConfig,
) -> EvalResult:
    """
    Dispatch to the correct evaluation function based on task_type.
    """
    task = getattr(config, "task_type", "binary")
    if task == "regression":
        return _eval_regression(train_result, df_test_clean, config)
    elif task == "multiclass":
        return _eval_multiclass(train_result, df_test_clean, config)
    else:
        return _eval_binary(train_result, df_test_clean, config)


def compare_evals(evals: list[EvalResult], primary_attr: str) -> pd.DataFrame:
    """
    Build a comparison DataFrame across model types.
    Works for binary and multiclass (uses compat fairness dict).
    For regression, uses MAE / MSE instead of accuracy / F1.
    """
    rows = []
    for ev in evals:
        m = ev.fairness.get(primary_attr)

        if ev.task_type == "regression":
            row = {
                "Model":    ev.model_type,
                "MAE":      round(ev.performance.mae, 4),
                "MSE":      round(ev.performance.mse, 4),
                "R²":       round(ev.performance.r2, 4),
                "DI":       round(m.disparate_impact, 3) if m and not np.isnan(m.disparate_impact) else None,
                "Severity": m.severity if m else "UNKNOWN",
            }
        else:
            row = {
                "Model":    ev.model_type,
                "Accuracy": round(ev.performance.accuracy, 4),
                "F1":       round(ev.performance.f1, 4),
                "ROC-AUC":  round(ev.performance.roc_auc, 4),
                "DI":       round(m.disparate_impact, 3)        if m and not np.isnan(m.disparate_impact) else None,
                "SPD":      round(m.statistical_parity_diff, 3) if m and not np.isnan(m.statistical_parity_diff) else None,
                "EOD":      round(m.equalized_odds_diff, 3)     if m and not np.isnan(m.equalized_odds_diff) else None,
                "EOpD":     round(m.equal_opportunity_diff, 3)  if m and not np.isnan(m.equal_opportunity_diff) else None,
                "PP":       round(m.predictive_parity_diff, 3)  if m and not np.isnan(m.predictive_parity_diff) else None,
                "FNRP":     round(m.fnr_parity_diff, 3)         if m and not np.isnan(m.fnr_parity_diff) else None,
                "Severity": m.severity if m else "UNKNOWN",
            }
        rows.append(row)
    return pd.DataFrame(rows)
