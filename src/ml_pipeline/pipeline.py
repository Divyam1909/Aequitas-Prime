"""
End-to-end pipeline orchestrator.

run_full_pipeline(df, config) → AuditResult

Steps:
  1. Preprocess
  2. Baseline model + metrics
  3. Proxy scan
  4. Reweighing → Pre-processed model + metrics
  5. ExponentiatedGradient → In-processed model + metrics
  6. Return AuditResult with everything
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
from src.ml_pipeline.trainer import TrainResult, train_baseline, train_preprocessed, train_inprocessed
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

    # Internal — raw preprocessed data for downstream use
    _X: pd.DataFrame | None = None
    _y: pd.Series | None = None
    _df_clean: pd.DataFrame | None = None


def run_mitigation_steps(
    result: AuditResult,
    run_inprocessing: bool = False,
    inprocess_method: str = "expgrad",
    base_model: str = "rf",
    verbose: bool = True,
) -> AuditResult:
    """
    Run only the mitigation steps (reweighing + optional in-processing) on an
    AuditResult that already has a trained baseline model.

    Called by the Surgeon tab when the user explicitly triggers mitigation.
    Modifies `result` in-place and also returns it.
    """
    def log(msg):
        if verbose:
            print(f"  [mitigation] {msg}")

    X        = result._X
    y        = result._y
    df_clean = result._df_clean
    config   = result.config
    primary  = config.primary_protected_attr()

    log("Step 1/2 — Reweighing + pre-processed model...")
    rw = apply_reweighing(df_clean, config, attr=primary)
    result.reweigh_result     = rw
    result.preprocessed_train = train_preprocessed(X, y, rw.weights, config, base_model=base_model)
    result.preprocessed_eval  = evaluate(result.preprocessed_train, df_clean, config)
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
) -> AuditResult:
    """
    Full bias audit + mitigation pipeline.

    Parameters
    ----------
    df : raw DataFrame (output of load_csv)
    config : DatasetConfig
    run_inprocessing : whether to run the slower ExponentiatedGradient step
    inprocess_method : "expgrad" or "gridsearch"
    inprocess_max_iter : epochs for ExponentiatedGradient
    verbose : print progress
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

    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    log("Step 1/6 — Preprocessing data...")
    if preprocess_fn is None:
        from src.utils.adult_preprocessor import preprocess as preprocess_fn
    X, y, df_clean = preprocess_fn(df, config)
    result._X = X
    result._y = y
    result._df_clean = df_clean
    result.n_features = X.shape[1]

    # ── Step 2: Baseline bias (pre-training, raw labels) ────────────────────
    log("Step 2/6 — Computing baseline bias metrics...")
    result.baseline_metrics = compute_metrics_for_all_attrs(df_clean, config)
    for attr, m in result.baseline_metrics.items():
        log(f"  {attr}: DI={m.disparate_impact:.3f}  [{m.severity}]")

    # ── Step 3: Proxy scan ───────────────────────────────────────────────────
    log("Step 3/6 — Running shadow proxy scan...")
    result.proxy_scan = scan_proxies(X, config)

    # ── Step 4: Baseline model ───────────────────────────────────────────────
    log("Step 4/6 — Training baseline model...")
    result.baseline_train = train_baseline(X, y, config, base_model=base_model)
    result.baseline_eval  = evaluate(result.baseline_train, df_clean, config)
    log(f"  Baseline accuracy={result.baseline_eval.performance.accuracy:.3f}")

    # ── Step 5: Pre-processed model (Reweighing) ─────────────────────────────
    primary_attr = config.primary_protected_attr()
    if run_mitigation:
        log("Step 5/6 — Reweighing + training pre-processed model...")
        rw = apply_reweighing(df_clean, config, attr=primary_attr)
        result.reweigh_result = rw

        result.preprocessed_train = train_preprocessed(X, y, rw.weights, config, base_model=base_model)
        result.preprocessed_eval  = evaluate(result.preprocessed_train, df_clean, config)
        pre_di = result.preprocessed_eval.fairness.get(primary_attr)
        pre_di_val = f"{pre_di.disparate_impact:.3f}" if pre_di else "N/A"
        log(f"  Pre-processed accuracy={result.preprocessed_eval.performance.accuracy:.3f}  DI={pre_di_val}")

        # ── Step 6: In-processed model (ExponentiatedGradient) ──────────────
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

    # ── Build comparison table (only entries that exist) ─────────────────────
    evals = [e for e in [result.baseline_eval, result.preprocessed_eval, result.inprocessed_eval] if e]
    if len(evals) > 1:
        result.comparison_df = compare_evals(evals, primary_attr)

    log("Pipeline complete.")
    return result
