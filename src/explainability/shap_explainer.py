"""
SHAP explainability — global feature importance + local per-prediction explanations.

Uses TreeExplainer for RandomForest (fast, exact).
Falls back to LinearExplainer / KernelExplainer for non-tree models.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import shap


@dataclass
class LocalExplanation:
    feature_names: list[str]
    feature_values: list[Any]
    shap_values: list[float]
    prediction: int
    base_value: float

    def top_drivers(self, n: int = 3) -> list[tuple[str, Any, float]]:
        """Return top-n features by absolute SHAP value."""
        ranked = sorted(
            zip(self.feature_names, self.feature_values, self.shap_values),
            key=lambda t: abs(t[2]),
            reverse=True,
        )
        return ranked[:n]


def build_explainer(model: Any, X_background: pd.DataFrame | np.ndarray) -> shap.Explainer:
    """
    Build the most appropriate SHAP explainer for the given model type.
    - RandomForest / tree ensembles → TreeExplainer (fast, exact)
    - Everything else               → LinearExplainer or KernelExplainer
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    tree_types = (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)

    # Direct tree model → fast TreeExplainer
    if isinstance(model, tree_types):
        return shap.TreeExplainer(model)

    # Fairlearn ExponentiatedGradient / GridSearch — use first fitted predictor
    if hasattr(model, "predictors_") and len(model.predictors_) > 0:
        candidate = model.predictors_[0]
        if isinstance(candidate, tree_types):
            return shap.TreeExplainer(candidate)

    # Linear models
    try:
        return shap.LinearExplainer(model, X_background)
    except Exception:
        pass

    # Generic fallback — slow but universal
    bg = shap.sample(X_background, min(100, len(X_background)))
    return shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, 1],
        bg,
    )


def compute_global_shap(
    explainer: shap.Explainer,
    X: pd.DataFrame | np.ndarray,
    max_samples: int = 500,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for up to max_samples rows.
    Returns (shap_values_2d, feature_names).
    shap_values_2d shape: (n_samples, n_features) for the positive class.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_arr = X

    # Sample for speed
    n = min(max_samples, len(X_arr))
    idx = np.random.choice(len(X_arr), n, replace=False)
    X_sample = X_arr[idx]

    try:
        sv = explainer.shap_values(X_sample)
        # list of 2 arrays (one per class) → take class-1
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        # 3D array (n_samples, n_features, n_classes) → take class-1
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv, feature_names
    except Exception as e:
        print(f"  [SHAP] global computation failed: {e}")
        return np.zeros((n, len(feature_names))), feature_names


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Mean absolute SHAP per feature — sorted descending.
    Returns DataFrame with columns [feature, importance].
    """
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def explain_single(
    explainer: shap.Explainer,
    X_row: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    prediction: int,
) -> LocalExplanation:
    """Compute SHAP explanation for a single prediction row."""
    if isinstance(X_row, pd.DataFrame):
        X_arr = X_row.values
        feat_vals = X_row.iloc[0].tolist()
    else:
        X_arr = X_row.reshape(1, -1)
        feat_vals = X_arr[0].tolist()

    try:
        sv = explainer.shap_values(X_arr)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        sv_row = sv[0].tolist()
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base = float(ev[1])
        else:
            base = float(ev)
    except Exception as e:
        print(f"  [SHAP] local explanation failed: {e}")
        sv_row = [0.0] * len(feature_names)
        base = 0.0

    return LocalExplanation(
        feature_names=feature_names,
        feature_values=feat_vals,
        shap_values=sv_row,
        prediction=prediction,
        base_value=base,
    )


def protected_attr_in_top_n(
    importance_df: pd.DataFrame,
    protected_cols: list[str],
    n: int = 5,
) -> bool:
    """Return True if any protected attribute is in the top-n SHAP features."""
    top_features = set(importance_df["feature"].head(n).tolist())
    return bool(top_features & set(protected_cols))
