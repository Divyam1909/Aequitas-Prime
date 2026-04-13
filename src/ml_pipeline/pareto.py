"""
Fairness-Accuracy Pareto Frontier.

Sweeps constraint_weight from 0 (pure accuracy) to 1 (pure fairness) using
Fairlearn GridSearch. Plots the tradeoff curve — users pick their operating point.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.reductions import GridSearch, EqualizedOdds, DemographicParity

from src.utils.schema import DatasetConfig
from src.bias_engine.detector import compute_disparate_impact, compute_all_metrics


@dataclass
class ParetoPoint:
    constraint_weight: float
    accuracy: float
    f1: float
    disparate_impact: float
    equalized_odds_diff: float
    is_optimal: bool = False


def sweep_fairness_frontier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    S_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    S_test: pd.Series,
    df_test_clean: pd.DataFrame,
    config: DatasetConfig,
    steps: int = 8,
    constraint: str = "equalized_odds",
    n_estimators: int = 50,
    random_state: int = 42,
) -> list[ParetoPoint]:
    """
    Train `steps` models, each with a different fairness constraint weight.
    Returns a list of ParetoPoints suitable for a Plotly scatter plot.
    """
    weights = np.linspace(0.0, 1.0, steps)
    points: list[ParetoPoint] = []

    for w in weights:
        base_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        # Fresh constraint object each iteration — GridSearch is stateful
        constraint_cls = EqualizedOdds() if constraint == "equalized_odds" else DemographicParity()
        gs = GridSearch(
            estimator=base_clf,
            constraints=constraint_cls,
            grid_size=5,
            constraint_weight=float(w),
        )
        try:
            gs.fit(X_train, y_train, sensitive_features=S_train)
            y_pred = gs.predict(X_test)
        except Exception as e:
            print(f"  [pareto] w={w:.2f} failed: {e}")
            continue

        acc = float(accuracy_score(y_test, y_pred))
        f1  = float(f1_score(y_test, y_pred, zero_division=0))

        # Fairness metrics on test set
        try:
            attr = config.primary_protected_attr()
            df_aligned = df_test_clean.loc[X_test.index]
            m = compute_all_metrics(df_aligned, config, y_pred=y_pred, attr=attr)
            di  = m.disparate_impact
            eod = m.equalized_odds_diff
        except Exception:
            di  = float("nan")
            eod = float("nan")

        points.append(ParetoPoint(
            constraint_weight=float(w),
            accuracy=acc,
            f1=f1,
            disparate_impact=di if not np.isnan(di) else 0.0,
            equalized_odds_diff=eod if not np.isnan(eod) else 1.0,
        ))

    _mark_optimal(points)
    return points


def _mark_optimal(points: list[ParetoPoint]) -> None:
    """
    Mark the 'knee' point: largest DI gain per unit accuracy loss.
    Compares each point against the first point (w=0, pure accuracy).
    """
    if len(points) < 2:
        return
    baseline = points[0]
    best_ratio = -1.0
    best_idx   = 0
    for i, p in enumerate(points[1:], 1):
        di_gain  = p.disparate_impact - baseline.disparate_impact
        acc_loss = baseline.accuracy  - p.accuracy + 1e-9  # avoid div/0
        ratio = di_gain / acc_loss
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx   = i
    points[best_idx].is_optimal = True


def pareto_to_dataframe(points: list[ParetoPoint]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "constraint_weight": p.constraint_weight,
            "accuracy":          round(p.accuracy, 4),
            "f1":                round(p.f1, 4),
            "disparate_impact":  round(p.disparate_impact, 3),
            "eod":               round(p.equalized_odds_diff, 3),
            "is_optimal":        p.is_optimal,
        }
        for p in points
    ])
