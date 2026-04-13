"""
LIME explainability — model-agnostic local explanations.
Used primarily for the in-processing model (ExponentiatedGradient ensemble)
where SHAP TreeExplainer is less reliable.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


@dataclass
class LimeExplanation:
    feature_contributions: list[tuple[str, float]]   # (feature_label, weight)
    prediction_proba: list[float]                     # [P(0), P(1)]
    intercept: float

    def top_drivers(self, n: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(self.feature_contributions, key=lambda t: abs(t[1]), reverse=True)
        return ranked[:n]

    def as_html(self) -> str:
        """Simple HTML table for embedding in Streamlit."""
        rows = "".join(
            f"<tr><td>{feat}</td><td style='color:{'green' if w>0 else 'red'}'>{w:+.3f}</td></tr>"
            for feat, w in self.top_drivers(6)
        )
        return f"<table><tr><th>Feature</th><th>Contribution</th></tr>{rows}</table>"


def build_lime_explainer(
    X_train: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    categorical_feature_indices: list[int] | None = None,
    class_names: list[str] | None = None,
) -> LimeTabularExplainer:
    """
    Build a LimeTabularExplainer from training data.

    Parameters
    ----------
    categorical_feature_indices : list of column indices that are categorical
    """
    if isinstance(X_train, pd.DataFrame):
        X_arr = X_train.values
    else:
        X_arr = X_train

    return LimeTabularExplainer(
        training_data=X_arr,
        feature_names=feature_names,
        categorical_features=categorical_feature_indices or [],
        class_names=class_names or ["Denied", "Approved"],
        mode="classification",
        random_state=42,
    )


def explain_with_lime(
    explainer: LimeTabularExplainer,
    model: Any,
    X_row: np.ndarray | pd.DataFrame,
    num_features: int = 6,
) -> LimeExplanation:
    """
    Generate a LIME explanation for a single row.

    Parameters
    ----------
    model : any sklearn-compatible classifier with predict_proba
    X_row : single row (1D array or single-row DataFrame)
    """
    if isinstance(X_row, pd.DataFrame):
        X_arr = X_row.values[0]
    elif X_row.ndim == 2:
        X_arr = X_row[0]
    else:
        X_arr = X_row

    def predict_fn(x):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)
        preds = model.predict(x).astype(float)
        return np.column_stack([1 - preds, preds])

    exp = explainer.explain_instance(
        X_arr,
        predict_fn,
        num_features=num_features,
    )

    return LimeExplanation(
        feature_contributions=exp.as_list(),
        prediction_proba=list(exp.predict_proba),
        intercept=float(exp.intercept[1]) if isinstance(exp.intercept, dict) else float(exp.intercept),
    )
