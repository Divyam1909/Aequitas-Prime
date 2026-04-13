"""
Counterfactual Ghost Engine — The Post-Processing Fairness Firewall.

For every live prediction, flip each protected attribute value and re-run the
model. If the decision changes, that prediction is demographically dependent
and should be flagged for human review.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from src.utils.schema import DatasetConfig


@dataclass
class CounterfactualResult:
    protected_attr: str
    original_value: Any
    flipped_value: Any
    original_decision: str       # "Approved" / "Denied"
    counterfactual_decision: str
    original_confidence: float
    counterfactual_confidence: float
    is_fair: bool                # True  = same decision regardless of demographics
    confidence_delta: float      # |original_conf - cf_conf|
    risk_level: str              # "CLEAR" / "FLAGGED" / "CRITICAL"
    message: str                 # human-readable summary


@dataclass
class MultiCounterfactualResult:
    protected_attr: str
    results: list[CounterfactualResult]
    worst_risk_level: str
    summary: str


DECISION_LABELS = {1: "Approved", 0: "Denied"}


def _risk_level(same_decision: bool, conf_delta: float) -> str:
    if not same_decision:
        return "CRITICAL"
    if conf_delta >= 0.15:
        return "FLAGGED"
    return "CLEAR"


def _predict_row(model: Any, X_row: pd.DataFrame) -> tuple[int, float]:
    """Return (class, confidence) for a single-row DataFrame."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row)[0]
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred])
    else:
        pred = int(model.predict(X_row)[0])
        conf = 1.0
    return pred, conf


def run_counterfactual_check(
    model: Any,
    input_features: dict[str, Any],
    config: DatasetConfig,
    feature_names: list[str],
    attr: str | None = None,
    flip_values: dict[str, Any] | None = None,
) -> CounterfactualResult:
    """
    Run a single binary counterfactual check for one protected attribute.

    Parameters
    ----------
    model : fitted sklearn-compatible model
    input_features : dict of {feature_name: value} for the original input
    config : DatasetConfig
    feature_names : ordered list of feature names the model was trained on
    attr : protected attribute to flip. Defaults to primary.
    flip_values : override the flipped value. If None, binary-flips (1↔0).
    """
    attr = attr or config.primary_protected_attr()

    # Build original row DataFrame in the correct column order
    orig_row = pd.DataFrame([{k: input_features.get(k, 0) for k in feature_names}])
    orig_pred, orig_conf = _predict_row(model, orig_row)

    # Build counterfactual row — flip the protected attribute
    cf_features = dict(input_features)
    orig_val = cf_features.get(attr, 0)

    if flip_values and attr in flip_values:
        cf_val = flip_values[attr]
    else:
        # Binary flip: 1 → 0, 0 → 1
        cf_val = 1 - int(orig_val)

    cf_features[attr] = cf_val
    cf_row = pd.DataFrame([{k: cf_features.get(k, 0) for k in feature_names}])
    cf_pred, cf_conf = _predict_row(model, cf_row)

    same = (orig_pred == cf_pred)
    delta = abs(orig_conf - cf_conf)
    risk  = _risk_level(same, delta)

    orig_label = DECISION_LABELS.get(orig_pred, str(orig_pred))
    cf_label   = DECISION_LABELS.get(cf_pred,   str(cf_pred))

    if risk == "CLEAR":
        msg = (f"Counterfactual Check PASSED. "
               f"Decision '{orig_label}' is stable regardless of {attr} value. "
               f"No demographic dependency detected.")
    elif risk == "FLAGGED":
        msg = (f"CAUTION: Decision stays '{orig_label}' but confidence drops "
               f"{delta:.0%} when {attr} changes from {orig_val} to {cf_val}. "
               f"Borderline case — manual review recommended.")
    else:
        msg = (f"ALERT: Decision FLIPS from '{orig_label}' to '{cf_label}' "
               f"when {attr} changes from {orig_val} to {cf_val}. "
               f"This prediction is demographically dependent.")

    return CounterfactualResult(
        protected_attr=attr,
        original_value=orig_val,
        flipped_value=cf_val,
        original_decision=orig_label,
        counterfactual_decision=cf_label,
        original_confidence=orig_conf,
        counterfactual_confidence=cf_conf,
        is_fair=same,
        confidence_delta=delta,
        risk_level=risk,
        message=msg,
    )


def run_multi_counterfactual(
    model: Any,
    input_features: dict[str, Any],
    config: DatasetConfig,
    feature_names: list[str],
    attr: str | None = None,
    all_attr_values: list[Any] | None = None,
) -> MultiCounterfactualResult:
    """
    Run counterfactual checks against ALL possible values of the protected attribute
    (not just binary flip). Useful for race which has 5+ values.

    Parameters
    ----------
    all_attr_values : list of all possible encoded values for the attr.
                      If None, assumes binary (0, 1).
    """
    attr = attr or config.primary_protected_attr()
    values = all_attr_values if all_attr_values else [0, 1]

    results = []
    for val in values:
        flip_vals = {attr: val}
        r = run_counterfactual_check(
            model, input_features, config, feature_names,
            attr=attr, flip_values=flip_vals,
        )
        results.append(r)

    risk_order = {"CRITICAL": 2, "FLAGGED": 1, "CLEAR": 0}
    worst = max(results, key=lambda r: risk_order.get(r.risk_level, 0))

    n_flip = sum(1 for r in results if not r.is_fair)
    if n_flip == 0:
        summary = f"All {len(results)} demographic scenarios produce the same decision. No bias detected."
    else:
        summary = (f"Decision flips in {n_flip}/{len(results)} demographic scenarios. "
                   f"Worst case: {worst.risk_level}.")

    return MultiCounterfactualResult(
        protected_attr=attr,
        results=results,
        worst_risk_level=worst.risk_level,
        summary=summary,
    )
