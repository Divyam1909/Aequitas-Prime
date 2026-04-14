"""
ML Training — three model tracks:
  1. Baseline       : chosen base model, no mitigation
  2. Pre-processed  : chosen base model with AIF360 Reweighing sample weights
                      (V2: uses train_preprocessed_presplit to avoid data leakage)
  3. In-processed   : Fairlearn ExponentiatedGradient (equalized odds)

V2 additions:
  - train_preprocessed_presplit(): accepts already-split train/test data so
    reweighing weights fitted on train-only can be applied directly.
  - Multiclass support: task_type="multiclass" trains without class_weight="balanced"
    and uses stratified split on multi-valued target.
  - Regression support: task_type="regression" uses sklearn regressors.

Base models available:
  "rf"    — RandomForestClassifier / RandomForestRegressor
  "lr"    — LogisticRegression / Ridge
  "xgb"   — XGBoostClassifier / XGBoostRegressor
  "stack" — StackingClassifier (LR + RF + XGB meta-learned by LR)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from src.utils.schema import DatasetConfig

# ── Model labels for display ──────────────────────────────────────────────────
MODEL_DISPLAY = {
    "rf":    "Random Forest",
    "lr":    "Logistic Regression",
    "xgb":   "XGBoost",
    "stack": "Stacked Ensemble (LR + RF + XGB)",
}


# ── Classification builders ───────────────────────────────────────────────────
def _build_rf(multiclass: bool = False) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight=None if multiclass else "balanced",
        n_jobs=-1,
    )


def _build_lr(multiclass: bool = False) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight=None if multiclass else "balanced",
        solver="lbfgs",
        multi_class="auto",
    )


def _build_xgb(multiclass: bool = False) -> Any:
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, random_state=42)


def _build_stacking() -> StackingClassifier:
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
        cv=3, passthrough=False, n_jobs=-1,
    )


# ── Regression builders ───────────────────────────────────────────────────────
def _build_rf_reg() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def _build_lr_reg() -> Ridge:
    return Ridge(alpha=1.0)


def _build_xgb_reg() -> Any:
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=100, random_state=42)


# Registries
MODEL_REGISTRY: dict[str, Any] = {
    "rf":    _build_rf,
    "lr":    _build_lr,
    "xgb":   _build_xgb,
    "stack": _build_stacking,
}
MODEL_REGISTRY_REG: dict[str, Any] = {
    "rf":  _build_rf_reg,
    "lr":  _build_lr_reg,
    "xgb": _build_xgb_reg,
    "stack": _build_rf_reg,  # no stacking for regression; fallback to RF
}


@dataclass
class TrainResult:
    model: Any
    model_type: str
    base_model: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    S_test: pd.Series
    feature_names: list[str] = field(default_factory=list)
    task_type: str = "binary"


def _get_s_test(X_te: pd.DataFrame, config: DatasetConfig) -> pd.Series:
    attr = config.primary_protected_attr()
    if attr in X_te.columns:
        return X_te[attr]
    alt = f"{attr}_binary"
    if alt in X_te.columns:
        return X_te[alt]
    return pd.Series(dtype=float)


def _stratify_arg(y: pd.Series, task_type: str):
    """Return stratify argument: y for binary/multiclass, None for regression."""
    return y if task_type != "regression" else None


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    config: DatasetConfig,
    base_model: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """Train the chosen base model with no fairness mitigation."""
    task = config.task_type if hasattr(config, "task_type") else "binary"

    if task == "regression":
        registry = MODEL_REGISTRY_REG
    else:
        registry = MODEL_REGISTRY
        # Pass multiclass flag to avoid class_weight="balanced" for multiclass
        _mc = (task == "multiclass")

    if base_model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown base_model '{base_model}'. Choose from: {list(MODEL_REGISTRY)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_arg(y, task),
    )

    if task == "regression":
        clf = MODEL_REGISTRY_REG[base_model]()
    elif task == "multiclass":
        clf = MODEL_REGISTRY[base_model](multiclass=True) if base_model in ("rf", "lr") else MODEL_REGISTRY[base_model]()
    else:
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
        task_type=task,
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
    """
    Train with AIF360 Reweighing sample weights.

    NOTE: This legacy function re-splits the full dataset. Prefer
    train_preprocessed_presplit() when you have already split the data and
    want to avoid data leakage from fitting the reweigher on test rows.
    """
    task = config.task_type if hasattr(config, "task_type") else "binary"

    if base_model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown base_model '{base_model}'.")

    X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(
        X, y, weights,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_arg(y, task),
    )

    clf = MODEL_REGISTRY[base_model]() if task != "regression" else MODEL_REGISTRY_REG[base_model]()
    try:
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
    except TypeError:
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
        task_type=task,
    )


def train_preprocessed_presplit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    weights_train: np.ndarray,
    config: DatasetConfig,
    base_model: str = "rf",
) -> TrainResult:
    """
    V2: Train with reweighing weights fitted on TRAINING data only.

    Uses the pre-existing train/test split from the baseline model so that
    the reweigher is never exposed to test-set rows.  This eliminates the
    data leakage present in the legacy train_preprocessed().

    Parameters
    ----------
    weights_train : sample weights for X_train rows only (shape = n_train).
    """
    task = config.task_type if hasattr(config, "task_type") else "binary"

    if base_model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown base_model '{base_model}'.")

    clf = MODEL_REGISTRY[base_model]() if task != "regression" else MODEL_REGISTRY_REG[base_model]()
    try:
        clf.fit(X_train, y_train, sample_weight=weights_train)
    except TypeError:
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return TrainResult(
        model=clf,
        model_type=f"preprocessed_{base_model}",
        base_model=base_model,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        y_pred=y_pred,
        S_test=_get_s_test(X_test, config),
        feature_names=list(X_train.columns),
        task_type=task,
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
    Fairlearn in-processing (binary classification only).
    'stack' falls back to 'rf' as Fairlearn requires a simple estimator.
    """
    from src.bias_engine.inprocessor import train_expgrad_model, train_gridsearch_model
    task = config.task_type if hasattr(config, "task_type") else "binary"

    eff_base  = "rf" if base_model == "stack" else base_model
    estimator = MODEL_REGISTRY[eff_base]()

    attr = config.primary_protected_attr()
    prot_cols = [c for c in X.columns
                 if c in config.protected_attrs or c.endswith("_binary")]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_arg(y, task),
    )
    S_tr = (X_tr[attr] if attr in X_tr.columns
            else X_tr.get(f"{attr}_binary", pd.Series(dtype=float)))
    S_te = _get_s_test(X_te, config)

    X_tr_f = X_tr.drop(columns=prot_cols, errors="ignore")
    X_te_f = X_te.drop(columns=prot_cols, errors="ignore")

    if method == "expgrad":
        result = train_expgrad_model(X_tr_f, y_tr, S_tr,
                                     constraint=constraint, estimator=estimator)
    else:
        result = train_gridsearch_model(X_tr_f, y_tr, S_tr,
                                        constraint=constraint, estimator=estimator)

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
        task_type=task,
    )
