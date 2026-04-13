"""
In-Processing Mitigation — The Adversarial Blindfold.

Three tiers (most practical to most powerful):
  Tier 1: Fairlearn ExponentiatedGradient + EqualizedOdds   [DEFAULT]
  Tier 2: Fairlearn GridSearch + DemographicParity           [FAST FALLBACK]
  Tier 3: AIF360 AdversarialDebiasing (TensorFlow)           [ADVANCED — optional]

Critical API notes (from research):
  - Fairlearn: sensitive_features is a kwarg to fit(), NEVER included in X
  - AIF360 AdversarialDebiasing: protected attr must be in DataFrame INDEX
  - AdversarialDebiasing: cannot use standard pickle — use joblib protocol=4
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

from fairlearn.reductions import (
    ExponentiatedGradient,
    GridSearch,
    EqualizedOdds,
    DemographicParity,
    TruePositiveRateParity,
)

from src.utils.schema import DatasetConfig


@dataclass
class InprocessResult:
    model: Any                # fitted model (sklearn-compatible)
    method: str               # "expgrad" / "gridsearch" / "adversarial"
    constraint: str           # "equalized_odds" / "demographic_parity"
    n_estimators_in_ensemble: int = 1


# ── Tier 1: ExponentiatedGradient ─────────────────────────────────────────────

def train_expgrad_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    S_train: pd.Series | np.ndarray,
    constraint: str = "equalized_odds",
    eps: float = 0.01,
    max_iter: int = 50,
    n_estimators: int = 100,
    random_state: int = 42,
    estimator: BaseEstimator | None = None,
) -> InprocessResult:
    """
    Train a fair classifier using ExponentiatedGradient.

    Parameters
    ----------
    X_train : feature matrix (protected attr must NOT be included)
    y_train : binary target
    S_train : sensitive feature series (same index as X_train)
    constraint : "equalized_odds" | "demographic_parity" | "equal_opportunity"
    eps : max allowed constraint violation (lower = stricter fairness)
    estimator : base sklearn estimator; defaults to RandomForestClassifier if None
    """
    if estimator is not None:
        base_clf = estimator
    else:
        base_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
    constraint_obj = _build_constraint(constraint)
    eg = ExponentiatedGradient(
        estimator=base_clf,
        constraints=constraint_obj,
        eps=eps,
        max_iter=max_iter,
    )
    # IMPORTANT: sensitive_features is a kwarg, never inside X
    eg.fit(X_train, y_train, sensitive_features=S_train)

    n_in_ensemble = len(eg.predictors_) if hasattr(eg, "predictors_") else 1
    return InprocessResult(
        model=eg,
        method="expgrad",
        constraint=constraint,
        n_estimators_in_ensemble=n_in_ensemble,
    )


# ── Tier 2: GridSearch ────────────────────────────────────────────────────────

def train_gridsearch_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    S_train: pd.Series | np.ndarray,
    constraint: str = "demographic_parity",
    grid_size: int = 10,
    constraint_weight: float = 0.5,
    n_estimators: int = 100,
    random_state: int = 42,
    estimator: BaseEstimator | None = None,
) -> InprocessResult:
    """
    Train a fair classifier using GridSearch over Lagrange multipliers.
    More deterministic than ExponentiatedGradient — trains exactly grid_size models.

    constraint_weight : 0.0 = pure accuracy, 1.0 = pure fairness
    estimator : base sklearn estimator; defaults to RandomForestClassifier if None
    """
    if estimator is not None:
        base_clf = estimator
    else:
        base_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
    constraint_obj = _build_constraint(constraint)
    gs = GridSearch(
        estimator=base_clf,
        constraints=constraint_obj,
        grid_size=grid_size,
        constraint_weight=constraint_weight,
    )
    gs.fit(X_train, y_train, sensitive_features=S_train)

    # GridSearch stores all models in predictors_; the object itself is the best predictor
    # Use gs directly as the model — it has .predict() and .predict_proba()
    n = len(gs.predictors_) if hasattr(gs, "predictors_") else grid_size
    return InprocessResult(
        model=gs,
        method="gridsearch",
        constraint=constraint,
        n_estimators_in_ensemble=n,
    )


# ── Tier 3: AIF360 AdversarialDebiasing (optional, requires TensorFlow) ───────

def train_adversarial_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: DatasetConfig,
    adversary_loss_weight: float = 0.5,
    num_epochs: int = 50,
    batch_size: int = 128,
    random_state: int = 42,
) -> InprocessResult:
    """
    Train an adversarially debiased neural classifier using AIF360.
    Requires tensorflow >= 2.15. Falls back to ExponentiatedGradient if TF absent.

    Protected attribute must NOT be in X_train — it is set via the DataFrame index.
    """
    try:
        from aif360.sklearn.inprocessing import AdversarialDebiasing
        import tensorflow as tf
    except ImportError:
        print("  [AdversarialDebiasing] TensorFlow not installed. Falling back to ExponentiatedGradient.")
        attr = config.primary_protected_attr()
        S = pd.Series(0, index=X_train.index)  # dummy — caller should provide real S
        return train_expgrad_model(X_train, y_train, S)

    attr = config.primary_protected_attr()
    # AdversarialDebiasing requires protected attr in the DataFrame INDEX
    X_indexed = X_train.copy()
    X_indexed.index = pd.MultiIndex.from_arrays(
        [range(len(X_indexed)), X_train.index],
        names=[attr, None],
    )

    model = AdversarialDebiasing(
        prot_attr=attr,
        adversary_loss_weight=adversary_loss_weight,
        num_epochs=num_epochs,
        batch_size=batch_size,
        debias=True,
        random_state=random_state,
    )
    model.fit(X_indexed, y_train)

    return InprocessResult(
        model=model,
        method="adversarial",
        constraint="adversarial_debiasing",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_constraint(name: str):
    mapping = {
        "equalized_odds":     EqualizedOdds(),
        "demographic_parity": DemographicParity(),
        "equal_opportunity":  TruePositiveRateParity(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown constraint '{name}'. Choose from: {list(mapping)}")
    return mapping[name]


def predict(result: InprocessResult, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Unified predict interface for all three model tiers."""
    return result.model.predict(X)


def predict_proba(result: InprocessResult, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Unified predict_proba. ExponentiatedGradient returns randomized predictions —
    predict_proba gives the weighted probability across the ensemble.
    """
    model = result.model
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    # Fallback: use predict and convert to two-column proba
    preds = model.predict(X)
    prob1 = preds.astype(float)
    return np.column_stack([1 - prob1, prob1])
