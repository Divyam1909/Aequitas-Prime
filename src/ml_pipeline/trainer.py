"""
ML Training — three model tracks:
  1. Baseline       : chosen base model, no mitigation
  2. Pre-processed  : chosen base model with AIF360 Reweighing sample weights
  3. In-processed   : Fairlearn ExponentiatedGradient (equalized odds)

Base models available:
  "rf"    — RandomForestClassifier  (default, tree-based, handles non-linearity)
  "lr"    — LogisticRegression      (linear, fast, interpretable)
  "xgb"   — XGBoostClassifier       (gradient boosting, usually best accuracy)
  "stack" — StackingClassifier      (LR + RF + XGB meta-learned by LR)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.utils.schema import DatasetConfig

# ── Model labels for display ──────────────────────────────────────────────────
MODEL_DISPLAY = {
    "rf":    "Random Forest",
    "lr":    "Logistic Regression",
    "xgb":   "XGBoost",
    "stack": "Stacked Ensemble (LR + RF + XGB)",
}


def _build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )


def _build_lr() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced", solver="lbfgs",
    )


def _build_xgb() -> Any:
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False,
        )
    except ImportError:
        # Graceful fallback to sklearn GradientBoosting if XGBoost absent
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, random_state=42)


def _build_stacking() -> StackingClassifier:
    """
    Three-model stack: LR + RF + XGB base learners, LR meta-learner.
    Uses 3-fold cross-validation for out-of-fold predictions.
    """
    estimators = [
        ("lr",  _build_lr()),
        ("rf",  RandomForestClassifier(n_estimators=50, random_state=42,
                                        class_weight="balanced", n_jobs=-1)),
    ]
    try:
        from xgboost import XGBClassifier
        estimators.append(("xgb", XGBClassifier(
            n_estimators=50, random_state=42,
            eval_metric="logloss", verbosity=0,
        )))
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        estimators.append(("gbm", GradientBoostingClassifier(
            n_estimators=50, random_state=42,
        )))

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500, random_state=42),
        cv=3,
        passthrough=False,
        n_jobs=-1,
    )


# Registry — callable so each call gets a fresh, unfitted instance
MODEL_REGISTRY: dict[str, Any] = {
    "rf":    _build_rf,
    "lr":    _build_lr,
    "xgb":   _build_xgb,
    "stack": _build_stacking,
}


@dataclass
class TrainResult:
    model: Any
    model_type: str          # "baseline_rf" | "baseline_lr" | etc.
    base_model: str          # "rf" | "lr" | "xgb" | "stack"
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    S_test: pd.Series        # protected attribute for fairness eval
    feature_names: list[str] = field(default_factory=list)


def _get_s_test(X_te: pd.DataFrame, config: DatasetConfig) -> pd.Series:
    attr = config.primary_protected_attr()
    if attr in X_te.columns:
        return X_te[attr]
    alt = f"{attr}_binary"
    if alt in X_te.columns:
        return X_te[alt]
    return pd.Series(dtype=float)


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    config: DatasetConfig,
    base_model: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """Train the chosen base model with no fairness mitigation."""
    if base_model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown base_model '{base_model}'. Choose from: {list(MODEL_REGISTRY)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    clf = MODEL_REGISTRY[base_model]()
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    return TrainResult(
        model=clf,
        model_type=f"baseline_{base_model}",
        base_model=base_model,
        X_train=X_tr, X_test=X_te,
        y_train=y_tr, y_test=y_te,
        y_pred=y_pred,
        S_test=_get_s_test(X_te, config),
        feature_names=list(X.columns),
    )


def train_preprocessed(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    config: DatasetConfig,
    base_model: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """Train with AIF360 Reweighing sample weights applied to the base model."""
    if base_model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown base_model '{base_model}'. Choose from: {list(MODEL_REGISTRY)}")

    X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(
        X, y, weights, test_size=test_size, random_state=random_state, stratify=y,
    )
    clf = MODEL_REGISTRY[base_model]()

    # StackingClassifier in sklearn ≥1.4 propagates sample_weight to base
    # estimators that support it.  Wrap in try/except for any edge-case.
    try:
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
    except TypeError:
        # Fallback: fit without weights (stacking meta-learner may not support it)
        clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)

    return TrainResult(
        model=clf,
        model_type=f"preprocessed_{base_model}",
        base_model=base_model,
        X_train=X_tr, X_test=X_te,
        y_train=y_tr, y_test=y_te,
        y_pred=y_pred,
        S_test=_get_s_test(X_te, config),
        feature_names=list(X.columns),
    )


def train_inprocessed(
    X: pd.DataFrame,
    y: pd.Series,
    config: DatasetConfig,
    method: str = "expgrad",
    constraint: str = "equalized_odds",
    base_model: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """
    Fairlearn in-processing.  Protected attrs are kept out of X but passed
    as sensitive_features to the fairness estimator.

    base_model selects the underlying estimator wrapped by ExponentiatedGradient
    or GridSearch.  'stack' is not supported here (Fairlearn requires a simple
    sklearn estimator); falls back to 'rf'.
    """
    from src.bias_engine.inprocessor import train_expgrad_model, train_gridsearch_model

    eff_base = "rf" if base_model == "stack" else base_model
    estimator = MODEL_REGISTRY[eff_base]()

    attr = config.primary_protected_attr()
    prot_cols = [c for c in X.columns
                 if c in config.protected_attrs or c.endswith("_binary")]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    S_tr = (X_tr[attr] if attr in X_tr.columns
            else X_tr.get(f"{attr}_binary", pd.Series(dtype=float)))
    S_te = _get_s_test(X_te, config)

    X_tr_f = X_tr.drop(columns=prot_cols, errors="ignore")
    X_te_f = X_te.drop(columns=prot_cols, errors="ignore")

    if method == "expgrad":
        result = train_expgrad_model(X_tr_f, y_tr, S_tr,
                                     constraint=constraint,
                                     estimator=estimator)
    else:
        result = train_gridsearch_model(X_tr_f, y_tr, S_tr,
                                        constraint=constraint,
                                        estimator=estimator)

    y_pred = result.model.predict(X_te_f)

    return TrainResult(
        model=result.model,
        model_type=f"inprocessed_{eff_base}",
        base_model=eff_base,
        X_train=X_tr_f, X_test=X_te_f,
        y_train=y_tr, y_test=y_te,
        y_pred=y_pred,
        S_test=S_te,
        feature_names=list(X_tr_f.columns),
    )
