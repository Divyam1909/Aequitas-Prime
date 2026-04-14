"""
Counterfactual Ghost Engine — The Post-Processing Fairness Firewall.

V2 changes:
  - Uses original string labels in messages ("Male → Female" not "1 → 0")
  - run_counterfactual_check now accepts label_decode dict for display
  - run_multi_counterfactual now iterates over ALL original string values
    (not just binary 0/1) using config.all_encoded_values()
  - CounterfactualResult gains original_label_str / flipped_label_str fields
  - Backward-compatible: all existing call sites still work without changes

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
    # V2: human-readable string versions of the above
    original_label_str: str = ""
    flipped_label_str: str = ""

    original_decision: str = ""       # "Approved" / "Denied"
    counterfactual_decision: str = ""
    original_confidence: float = 0.0
    counterfactual_confidence: float = 0.0
    is_fair: bool = True
    confidence_delta: float = 0.0
    risk_level: str = "CLEAR"         # "CLEAR" / "FLAGGED" / "CRITICAL"
    message: str = ""


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


def _decode(config: DatasetConfig, attr: str, encoded_val: Any) -> str:
    """
    Convert an encoded integer to its original string label.
    Uses config.label_encodings if populated; falls back to str(encoded_val).
    """
    return config.decode_label(attr, int(encoded_val))


def run_counterfactual_check(
    model: Any,
    input_features: dict[str, Any],
    config: DatasetConfig,
    feature_names: list[str],
    attr: str | None = None,
    flip_values: dict[str, Any] | None = None,
    label_decode: dict[str, dict[int, str]] | None = None,
) -> CounterfactualResult:
    """
    Run a single binary counterfactual check for one protected attribute.

    V2: label_decode is an optional {attr: {encoded_int: str_label}} mapping
    used purely for display purposes in result messages.  When absent, the
    function falls back to config.decode_label() and then to str().

    Parameters
    ----------
    model         : fitted sklearn-compatible model
    input_features: dict {feature_name: value} for the original input
    config        : DatasetConfig
    feature_names : ordered list of feature names the model was trained on
    attr          : protected attribute to flip. Defaults to primary.
    flip_values   : override the flipped value. If None, binary-flips (1↔0).
    label_decode  : optional {attr: {0: "Female", 1: "Male"}} for display
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

    same  = (orig_pred == cf_pred)
    delta = abs(orig_conf - cf_conf)
    risk  = _risk_level(same, delta)

    orig_label = DECISION_LABELS.get(orig_pred, str(orig_pred))
    cf_label   = DECISION_LABELS.get(cf_pred,   str(cf_pred))

    # V2: resolve human-readable demographic labels
    def _str_label(val: Any) -> str:
        if label_decode and attr in label_decode:
            return label_decode[attr].get(int(val), str(val))
        return config.decode_label(attr, int(val))

    orig_str = _str_label(orig_val)
    cf_str   = _str_label(cf_val)

    # Build message with proper string labels
    if risk == "CLEAR":
        msg = (
            f"Counterfactual Check PASSED. "
            f"Decision '{orig_label}' is stable regardless of {attr} value "
            f"({orig_str} → {cf_str}). No demographic dependency detected."
        )
    elif risk == "FLAGGED":
        msg = (
            f"CAUTION: Decision stays '{orig_label}' but confidence drops "
            f"{delta:.0%} when {attr} changes from {orig_str} to {cf_str}. "
            f"Borderline case — manual review recommended."
        )
    else:
        msg = (
            f"ALERT: Decision FLIPS from '{orig_label}' to '{cf_label}' "
            f"when {attr} changes from {orig_str} to {cf_str}. "
            f"This prediction is demographically dependent."
        )

    return CounterfactualResult(
        protected_attr=attr,
        original_value=orig_val,
        flipped_value=cf_val,
        original_label_str=orig_str,
        flipped_label_str=cf_str,
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
    label_decode: dict[str, dict[int, str]] | None = None,
) -> MultiCounterfactualResult:
    """
    Run counterfactual checks against ALL possible values of the protected
    attribute, using original string labels for display.

    V2: when all_attr_values is None, uses config.all_encoded_values(attr)
    which is populated from label_encodings during preprocessing — so race
    with 5 encoded values automatically tests all 5 instead of just 0/1.

    Parameters
    ----------
    all_attr_values : list of all possible encoded values for the attr.
                      If None, uses config.all_encoded_values(attr).
    label_decode    : optional {attr: {encoded_int: str_label}} for display.
    """
    attr = attr or config.primary_protected_attr()
    values = all_attr_values if all_attr_values is not None else config.all_encoded_values(attr)

    results = []
    for val in values:
        flip_vals = {attr: val}
        r = run_counterfactual_check(
            model, input_features, config, feature_names,
            attr=attr, flip_values=flip_vals, label_decode=label_decode,
        )
        results.append(r)

    risk_order = {"CRITICAL": 2, "FLAGGED": 1, "CLEAR": 0}
    worst = max(results, key=lambda r: risk_order.get(r.risk_level, 0))

    n_flip = sum(1 for r in results if not r.is_fair)
    if n_flip == 0:
        summary = (
            f"All {len(results)} demographic scenarios produce the same "
            "decision. No bias detected."
        )
    else:
        summary = (
            f"Decision flips in {n_flip}/{len(results)} demographic scenarios. "
            f"Worst case: {worst.risk_level} "
            f"({worst.original_label_str} → {worst.flipped_label_str})."
        )

    return MultiCounterfactualResult(
        protected_attr=attr,
        results=results,
        worst_risk_level=worst.risk_level,
        summary=summary,
    )


def build_label_decode(
    config: DatasetConfig,
    df_clean: pd.DataFrame,
) -> dict[str, dict[int, str]]:
    """
    Build a label_decode dict from DatasetConfig.label_encodings.
    Falls back to inferring from df_clean if label_encodings is empty.

    Returns {attr: {encoded_int: original_str_label}}
    """
    decode: dict[str, dict[int, str]] = {}
    for attr in config.protected_attrs:
        enc = config.label_encodings.get(attr, {})
        if enc:
            # Invert: str_label → int  ⟹  int → str_label
            decode[attr] = {v: k for k, v in enc.items()}
        elif attr in df_clean.columns:
            # Fallback: privileged value → 1, everything else → 0 (generic encoding)
            priv_val = str(config.privileged_values.get(attr, ""))
            unpriv_vals = [
                str(v) for v in df_clean[attr].unique()
                if str(v) != priv_val
            ]
            decode[attr] = {
                1: priv_val,
                0: unpriv_vals[0] if len(unpriv_vals) == 1 else f"non-{priv_val}",
            }
    return decode
