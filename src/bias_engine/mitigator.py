"""
Pre-Processing Mitigation — The Generative Surgeon.

Two techniques:
  1. Reweighing (AIF360): assigns sample weights to balance group outcomes
  2. DisparateImpactRemover (AIF360): repairs feature distributions to reduce correlation
     with the protected attribute. Applied to continuous features only.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover

from src.utils.schema import DatasetConfig
from src.bias_engine.detector import _coerce_priv_val


@dataclass
class ReweighResult:
    weights: np.ndarray
    group_weight_map: dict[str, float]   # e.g. {"Male+Positive": 0.72, "Female+Positive": 2.31}
    original_approval_rates: dict[str, float]
    technique: str = "Reweighing"


@dataclass
class BeforeAfterResult:
    attr: str
    di_before: float
    di_after: float
    spd_before: float
    spd_after: float
    approval_rate_priv_before: float
    approval_rate_priv_after: float
    approval_rate_unpriv_before: float
    approval_rate_unpriv_after: float


def _to_bld(df: pd.DataFrame, config: DatasetConfig, attr: str) -> BinaryLabelDataset:
    """Convert a DataFrame to AIF360 BinaryLabelDataset for one protected attribute."""
    # AIF360 needs numeric protected attribute encoded as 1=privileged, 0=unprivileged
    df_aif = df[[attr, config.target_col]].copy()
    priv_val_raw = config.privileged_values[attr]
    col = df_aif[attr]
    if col.dtype == object or pd.api.types.is_string_dtype(col):
        # String column: compare directly
        df_aif[attr] = (col == priv_val_raw).astype(float)
    else:
        # Numeric column: coerce priv_val to the column's dtype before comparing
        try:
            priv_val_typed = col.dtype.type(priv_val_raw)
        except (ValueError, TypeError):
            priv_val_typed = priv_val_raw
        df_aif[attr] = (col == priv_val_typed).astype(float)
    df_aif[config.target_col] = df_aif[config.target_col].astype(float)

    numeric_priv = 1.0  # always 1.0 after the encoding above

    return BinaryLabelDataset(
        df=df_aif,
        label_names=[config.target_col],
        protected_attribute_names=[attr],
        favorable_label=float(config.positive_label),
        unfavorable_label=0.0,
        privileged_protected_attributes=[[numeric_priv]],
    )


def apply_reweighing(
    df: pd.DataFrame,
    config: DatasetConfig,
    attr: str | None = None,
) -> ReweighResult:
    """
    Compute AIF360 Reweighing weights for a given protected attribute.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the protected attribute column and the target column.
        Protected attribute should already be numeric (0/1) or string matching
        config.privileged_values.
    config : DatasetConfig
    attr : str, optional
        Protected attribute to reweigh for. Defaults to primary.

    Returns
    -------
    ReweighResult with weights array and per-group multiplier info.
    """
    attr = attr or config.primary_protected_attr()
    df_work = df.copy()

    # Resolve column name (race → race_binary after encoding)
    if attr not in df_work.columns:
        alt = f"{attr}_binary"
        if alt in df_work.columns:
            df_work = df_work.rename(columns={alt: attr})
        else:
            raise ValueError(f"Protected attribute '{attr}' not found in DataFrame columns: {list(df_work.columns)}")

    bld = _to_bld(df_work, config, attr)

    # After encoding, privileged = 1
    rw = Reweighing(
        unprivileged_groups=[{attr: 0.0}],
        privileged_groups=[{attr: 1.0}],
    )
    rw.fit(bld)
    transformed = rw.transform(bld)
    weights = transformed.instance_weights.copy()

    # Build human-readable group weight map
    # priv_mask uses the numeric column in df_work (already converted in _to_bld copy)
    # Use privileged_values to build mask from original column
    priv_val  = _coerce_priv_val(df_work[attr], config.privileged_values.get(attr, 1))
    priv_mask = df_work[attr] == priv_val
    pos_mask  = df_work[config.target_col] == config.positive_label
    group_weight_map = {
        "Privileged+Positive":    float(weights[priv_mask  & pos_mask ].mean()),
        "Privileged+Negative":    float(weights[priv_mask  & ~pos_mask].mean()),
        "Unprivileged+Positive":  float(weights[~priv_mask & pos_mask ].mean()),
        "Unprivileged+Negative":  float(weights[~priv_mask & ~pos_mask].mean()),
    }

    approval_rates = {
        "Privileged":   float(pos_mask[priv_mask ].mean()),
        "Unprivileged": float(pos_mask[~priv_mask].mean()),
    }

    return ReweighResult(
        weights=weights,
        group_weight_map=group_weight_map,
        original_approval_rates=approval_rates,
    )


def apply_disparate_impact_remover(
    df: pd.DataFrame,
    config: DatasetConfig,
    repair_level: float = 0.8,
    attr: str | None = None,
) -> pd.DataFrame:
    """
    Apply DisparateImpactRemover to continuous/ordinal features.
    Repairs feature distributions to reduce correlation with the protected attribute.

    Parameters
    ----------
    repair_level : float
        0.0 = no repair, 1.0 = full repair. 0.8 is a good balance.

    Returns
    -------
    Repaired DataFrame (same shape as input).
    """
    attr = attr or config.primary_protected_attr()
    df_work = df.copy()

    if attr not in df_work.columns:
        alt = f"{attr}_binary"
        if alt in df_work.columns:
            df_work = df_work.rename(columns={alt: attr})

    # Apply remover to each non-protected, non-target numeric column
    sensitive_cols = set(config.protected_attrs) | {config.target_col, attr}
    repair_cols = [
        c for c in df_work.select_dtypes(include=[np.number]).columns
        if c not in sensitive_cols
    ]

    dir_model = DisparateImpactRemover(
        repair_level=repair_level,
        sensitive_attribute=attr,
    )

    # DIR operates on AIF360 BinaryLabelDataset
    df_dir = df_work[[attr, config.target_col] + repair_cols].copy()
    df_dir[attr] = df_dir[attr].astype(float)
    df_dir[config.target_col] = df_dir[config.target_col].astype(float)

    bld = BinaryLabelDataset(
        df=df_dir,
        label_names=[config.target_col],
        protected_attribute_names=[attr],
        favorable_label=float(config.positive_label),
        unfavorable_label=0.0,
        privileged_protected_attributes=[[1.0]],
    )

    repaired_bld = dir_model.fit_transform(bld)
    repaired_df = repaired_bld.convert_to_dataframe()[0]

    # Merge repaired features back into original df
    result = df_work.copy()
    for col in repair_cols:
        if col in repaired_df.columns:
            result[col] = repaired_df[col].values
    return result


def compute_before_after(
    df_original: pd.DataFrame,
    df_reweighed: pd.DataFrame,
    weights_original: np.ndarray,
    weights_new: np.ndarray,
    config: DatasetConfig,
    attr: str | None = None,
) -> BeforeAfterResult:
    """
    Compare pre-training DI and SPD before and after reweighing.
    Uses weighted approval rates to reflect the effective reweighed distribution.
    """
    attr = attr or config.primary_protected_attr()

    def weighted_approval(df, weights, attr_col):
        if attr_col not in df.columns:
            attr_col = f"{attr_col}_binary"
        raw       = df[attr_col]
        priv_val  = _coerce_priv_val(raw, config.privileged_values.get(attr, 1))
        priv_mask = raw == priv_val
        pos_mask  = df[config.target_col] == config.positive_label
        w_priv_pos  = weights[priv_mask  & pos_mask ].sum()
        w_priv_all  = weights[priv_mask ].sum()
        w_unpriv_pos = weights[~priv_mask & pos_mask].sum()
        w_unpriv_all = weights[~priv_mask].sum()
        ar_priv   = w_priv_pos   / w_priv_all   if w_priv_all   > 0 else 0
        ar_unpriv = w_unpriv_pos / w_unpriv_all if w_unpriv_all > 0 else 0
        return float(ar_priv), float(ar_unpriv)

    attr_col = attr if attr in df_original.columns else f"{attr}_binary"
    ar_priv_b,   ar_unpriv_b   = weighted_approval(df_original, weights_original, attr_col)
    ar_priv_a,   ar_unpriv_a   = weighted_approval(df_reweighed, weights_new, attr_col)

    di_b  = ar_unpriv_b / ar_priv_b   if ar_priv_b  > 0 else float("nan")
    di_a  = ar_unpriv_a / ar_priv_a   if ar_priv_a  > 0 else float("nan")
    spd_b = ar_unpriv_b - ar_priv_b
    spd_a = ar_unpriv_a - ar_priv_a

    return BeforeAfterResult(
        attr=attr,
        di_before=di_b, di_after=di_a,
        spd_before=spd_b, spd_after=spd_a,
        approval_rate_priv_before=ar_priv_b,   approval_rate_priv_after=ar_priv_a,
        approval_rate_unpriv_before=ar_unpriv_b, approval_rate_unpriv_after=ar_unpriv_a,
    )
