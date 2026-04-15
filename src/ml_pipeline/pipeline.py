"""
End-to-end pipeline orchestrator.

run_full_pipeline(df, config) → AuditResult

V2 fixes & additions:
  1. DATA LEAKAGE FIX: apply_reweighing() is now called on TRAINING rows only
     (fit on train, apply to train) via train_preprocessed_presplit(). This
     eliminates the test-distribution leakage present in V1.
  2. SHAP: AuditResult gains shap_importance (pd.DataFrame) and shap_explainer
     fields, computed from the baseline model after training.
  3. label_decode: built from config.label_encodings after preprocessing and
     stored in AuditResult for the counterfactual UI.

Steps:
  1. Preprocess
  2. Baseline model + metrics
  3. Proxy scan
  4. Reweighing on train-only → Pre-processed model + metrics  [V2 leakage fix]
  5. ExponentiatedGradient → In-processed model + metrics
  6. SHAP global importance (optional)
  7. Return AuditResult with everything
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from src.utils.schema import DatasetConfig
from src.bias_engine.detector import compute_metrics_for_all_attrs, MetricsResult
from src.bias_engine.proxy_scanner import scan_proxies, ProxyResult
from src.bias_engine.mitigator import apply_reweighing, ReweighResult
from src.ml_pipeline.trainer import (
    TrainResult, train_baseline,
    train_preprocessed, train_preprocessed_presplit, train_inprocessed,
)
from src.ml_pipeline.evaluator import EvalResult, evaluate, compare_evals


@dataclass
class AuditResult:
    # Dataset info
    dataset_name: str
    n_rows: int
    n_features: int
    config: DatasetConfig

    # Pre-training bias on raw data
    baseline_metrics: dict[str, MetricsResult] = field(default_factory=dict)

    # Proxy scanner
    proxy_scan: dict[str, list[ProxyResult]] = field(default_factory=dict)

    # Reweighing info
    reweigh_result: ReweighResult | None = None

    # Trained models
    baseline_train: TrainResult | None = None
    preprocessed_train: TrainResult | None = None
    inprocessed_train: TrainResult | None = None

    # Post-training evaluations
    baseline_eval: EvalResult | None = None
    preprocessed_eval: EvalResult | None = None
    inprocessed_eval: EvalResult | None = None

    # Comparison table
    comparison_df: pd.DataFrame | None = None

    # V2: SHAP feature importance from the baseline model
    shap_importance: pd.DataFrame | None = None
    shap_explainer: Any | None = None   # cached for the Ghost tab

    # V2: label decode map {attr: {encoded_int: str_label}} for counterfactual display
    label_decode: dict[str, dict[int, str]] = field(default_factory=dict)

    # Internal — raw preprocessed data for downstream use
    _X: pd.DataFrame | None = None
    _y: pd.Series | None = None
    _df_clean: pd.DataFrame | None = None


def _build_label_decode(config: DatasetConfig, df_clean: pd.DataFrame) -> dict[str, dict[int, str]]:
    """Build {attr: {0: 'Female', 1: 'Male'}} decode map for counterfactual display."""
    from src.bias_engine.counterfactual import build_label_decode
    return build_label_decode(config, df_clean)


def _compute_shap(
    train_result: TrainResult,
    max_samples: int = 300,
    verbose: bool = True,
) -> tuple[pd.DataFrame | None, Any]:
    """
    Compute global SHAP feature importance for the baseline model.
    Returns (importance_df, explainer) or (None, None) on failure.
    """
    try:
        from src.explainability.shap_explainer import (
            build_explainer, compute_global_shap, global_feature_importance,
        )
        if verbose:
            print("  [pipeline] Computing SHAP feature importance...")
        exp = build_explainer(train_result.model, train_result.X_train)
        sv, fnames = compute_global_shap(exp, train_result.X_test, max_samples=max_samples)
        imp = global_feature_importance(sv, fnames)
        return imp, exp
    except Exception as e:
        if verbose:
            print(f"  [pipeline] SHAP computation skipped: {e}")
        return None, None


def run_mitigation_steps(
    result: AuditResult,
    run_inprocessing: bool = False,
    inprocess_method: str = "expgrad",
    base_model: str = "rf",
    verbose: bool = True,
    target_attr: str | None = None,
) -> AuditResult:
    """
    Run only the mitigation steps on an AuditResult that already has a
    trained baseline model.

    V2: Uses train_preprocessed_presplit() to avoid data leakage —
    reweighing is fitted on training rows only.

    Parameters
    ----------
    target_attr : str, optional
        Protected attribute to target for bias mitigation.
        Defaults to the most biased attribute by lowest Disparate Impact.
    """
    def log(msg):
        if verbose:
            print(f"  [mitigation] {msg}")

    X        = result._X
    y        = result._y
    df_clean = result._df_clean
    config   = result.config

    # Default to the most biased attribute (lowest DI) for meaningful mitigation
    if target_attr is None:
        if result.baseline_eval and result.baseline_eval.fairness:
            primary = min(
                result.baseline_eval.fairness.keys(),
                key=lambda a: result.baseline_eval.fairness[a].disparate_impact
                              if not np.isnan(result.baseline_eval.fairness[a].disparate_impact) else 1.0
            )
        else:
            primary = config.primary_protected_attr()
    else:
        primary = target_attr

    # ── V2 Leakage fix: fit reweighing on training rows only ─────────────────
    log("Step 1/2 — Reweighing on train-only (V2 leakage fix) + pre-processed model...")
    train_idx = result.baseline_train.X_train.index
    df_train_only = df_clean.loc[train_idx]
    rw = apply_reweighing(df_train_only, config, attr=primary)
    result.reweigh_result = rw

    result.preprocessed_train = train_preprocessed_presplit(
        X_train=result.baseline_train.X_train,
        X_test=result.baseline_train.X_test,
        y_train=result.baseline_train.y_train,
        y_test=result.baseline_train.y_test,
        weights_train=rw.weights,
        config=config,
        base_model=base_model,
    )
    result.preprocessed_eval = evaluate(result.preprocessed_train, df_clean, config)
    pre_di = result.preprocessed_eval.fairness.get(primary)
    pre_di_val = f"{pre_di.disparate_impact:.3f}" if pre_di else "N/A"
    log(f"  Pre-processed accuracy={result.preprocessed_eval.performance.accuracy:.3f}  DI={pre_di_val}")

    if run_inprocessing:
        log(f"Step 2/2 — In-processing ({inprocess_method})...")
        result.inprocessed_train = train_inprocessed(
            X, y, config,
            method=inprocess_method,
            constraint="equalized_odds",
            base_model=base_model,
        )
        result.inprocessed_eval = evaluate(result.inprocessed_train, df_clean, config)
    else:
        log("Step 2/2 — In-processing skipped.")

    evals = [e for e in [result.baseline_eval, result.preprocessed_eval, result.inprocessed_eval] if e]
    result.comparison_df = compare_evals(evals, primary)
    log("Mitigation complete.")
    return result


def run_full_pipeline(
    df: pd.DataFrame,
    config: DatasetConfig,
    run_inprocessing: bool = True,
    inprocess_method: str = "expgrad",
    inprocess_max_iter: int = 50,
    verbose: bool = True,
    run_mitigation: bool = True,
    preprocess_fn=None,
    base_model: str = "rf",
    compute_shap: bool = True,
) -> AuditResult:
    """
    Full bias audit + mitigation pipeline.

    V2 changes:
      - Reweighing fitted on train-only (no test leakage).
      - SHAP feature importance computed after baseline training.
      - label_decode populated for counterfactual display.
      - compute_shap flag (default True) — set False to skip SHAP for speed.
    """
    def log(msg):
        if verbose:
            print(f"  [pipeline] {msg}")

    result = AuditResult(
        dataset_name=config.dataset_name,
        n_rows=len(df),
        n_features=0,
        config=config,
    )

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    log("Step 1/6 — Preprocessing data...")
    if preprocess_fn is None:
        from src.utils.adult_preprocessor import preprocess as preprocess_fn
    X, y, df_clean = preprocess_fn(df, config)
    result._X = X
    result._y = y
    result._df_clean = df_clean
    result.n_features = X.shape[1]

    # V2: build label decode map for counterfactual display
    result.label_decode = _build_label_decode(config, df_clean)

    # ── Step 2: Baseline bias (pre-training) ─────────────────────────────────
    log("Step 2/6 — Computing baseline bias metrics...")
    result.baseline_metrics = compute_metrics_for_all_attrs(df_clean, config)
    for attr, m in result.baseline_metrics.items():
        log(f"  {attr}: DI={m.disparate_impact:.3f}  [{m.severity}]")

    # ── Step 3: Proxy scan ────────────────────────────────────────────────────
    log("Step 3/6 — Running shadow proxy scan (V2: combined_score fusion)...")
    result.proxy_scan = scan_proxies(X, config)

    # ── Step 4: Baseline model ────────────────────────────────────────────────
    log("Step 4/6 — Training baseline model...")
    result.baseline_train = train_baseline(X, y, config, base_model=base_model)
    result.baseline_eval  = evaluate(result.baseline_train, df_clean, config)
    log(f"  Baseline accuracy={result.baseline_eval.performance.accuracy:.3f}")

    # V2: SHAP after baseline
    if compute_shap:
        result.shap_importance, result.shap_explainer = _compute_shap(
            result.baseline_train, verbose=verbose
        )

    # ── Step 5: Pre-processed model (Reweighing) ─────────────────────────────
    primary_attr = config.primary_protected_attr()
    if run_mitigation:
        # V2 LEAKAGE FIX: fit reweigher on training rows only
        log("Step 5/6 — Reweighing (train-only, V2 leakage fix) + pre-processed model...")
        train_idx     = result.baseline_train.X_train.index
        df_train_only = df_clean.loc[train_idx]
        rw = apply_reweighing(df_train_only, config, attr=primary_attr)
        result.reweigh_result = rw

        result.preprocessed_train = train_preprocessed_presplit(
            X_train=result.baseline_train.X_train,
            X_test=result.baseline_train.X_test,
            y_train=result.baseline_train.y_train,
            y_test=result.baseline_train.y_test,
            weights_train=rw.weights,
            config=config,
            base_model=base_model,
        )
        result.preprocessed_eval = evaluate(result.preprocessed_train, df_clean, config)
        pre_di = result.preprocessed_eval.fairness.get(primary_attr)
        pre_di_val = f"{pre_di.disparate_impact:.3f}" if pre_di else "N/A"
        log(f"  Pre-processed accuracy={result.preprocessed_eval.performance.accuracy:.3f}  DI={pre_di_val}")

        # ── Step 6: In-processed model (ExponentiatedGradient) ────────────────
        if run_inprocessing:
            log(f"Step 6/6 — In-processing ({inprocess_method})...")
            result.inprocessed_train = train_inprocessed(
                X, y, config,
                method=inprocess_method,
                constraint="equalized_odds",
                base_model=base_model,
            )
            result.inprocessed_eval = evaluate(result.inprocessed_train, df_clean, config)
            in_di = result.inprocessed_eval.fairness.get(primary_attr)
            in_di_val = f"{in_di.disparate_impact:.3f}" if in_di else "N/A"
            log(f"  In-processed accuracy={result.inprocessed_eval.performance.accuracy:.3f}  DI={in_di_val}")
        else:
            log("Step 6/6 — In-processing skipped.")
    else:
        log("Steps 5–6 — Mitigation skipped (call run_mitigation_steps() to apply).")

    evals = [e for e in [result.baseline_eval, result.preprocessed_eval, result.inprocessed_eval] if e]
    if len(evals) > 1:
        result.comparison_df = compare_evals(evals, primary_attr)

    log("Pipeline complete.")
    return result
