"""
Intersectional Bias Analysis — checks bias across combinations of protected
attributes (e.g. "Black Female" vs. "White Female" vs. "Black Male").

Standard tools check sex OR race. We check all combinations simultaneously.
This is legally significant (Kimberlé Crenshaw's intersectionality theory)
and is required by EU AI Act recitals for high-risk AI systems.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd

from src.utils.schema import DatasetConfig


@dataclass
class IntersectionalResult:
    group_label: str           # e.g. "Female × Black"
    group_attrs: dict          # e.g. {"sex": "Female", "race": "Black"}
    group_size: int
    approval_rate: float       # P(Y=1 | group)
    pred_approval_rate: float  # P(Ŷ=1 | group) — requires y_pred
    disparate_impact: float    # vs privileged baseline
    equal_opportunity_diff: float
    severity: str              # "CRITICAL" / "HIGH" / "MEDIUM" / "OK"


SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.60,
    "HIGH":     0.80,
    "MEDIUM":   0.90,
}
MIN_GROUP_SIZE = 30   # skip groups with insufficient samples


def _severity(di: float) -> str:
    if di < SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if di < SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if di < SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "OK"


def compute_intersectional_metrics(
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    y_pred: np.ndarray | None = None,
) -> list[IntersectionalResult]:
    """
    Compute DI and EOpD for every combination of protected attribute values.

    Parameters
    ----------
    df_clean : DataFrame with original string label columns (output of adult_preprocessor)
    config : DatasetConfig
    y_pred : model predictions aligned to df_clean rows. If None, uses true labels.

    Returns
    -------
    list of IntersectionalResult, sorted by DI ascending (worst first)
    """
    target = config.target_col
    pos    = config.positive_label

    # Only consider attributes that are present in df_clean as string columns
    active_attrs = [a for a in config.protected_attrs if a in df_clean.columns]
    if len(active_attrs) < 1:
        return []

    # Privileged baseline: all attrs set to privileged values
    priv_mask = pd.Series(True, index=df_clean.index)
    for attr in active_attrs:
        priv_mask &= (df_clean[attr] == config.privileged_values[attr])

    priv_approval = float((df_clean.loc[priv_mask, target] == pos).mean())
    if priv_approval == 0:
        return []

    # For EOpD: TPR of privileged group
    if y_pred is not None:
        priv_y     = df_clean.loc[priv_mask, target]
        priv_preds = y_pred[priv_mask.values]
        priv_tpr   = float(((priv_preds == pos) & (priv_y == pos)).sum() /
                           max((priv_y == pos).sum(), 1))
    else:
        priv_tpr = None

    # Enumerate all value combinations
    attr_value_sets = [df_clean[a].unique().tolist() for a in active_attrs]
    results = []

    for combo in product(*attr_value_sets):
        attrs_combo = dict(zip(active_attrs, combo))
        # Build mask for this combo
        mask = pd.Series(True, index=df_clean.index)
        for attr, val in attrs_combo.items():
            mask &= (df_clean[attr] == val)

        group_size = int(mask.sum())
        if group_size < MIN_GROUP_SIZE:
            continue

        group_y = df_clean.loc[mask, target]
        approval = float((group_y == pos).mean())
        di = approval / priv_approval if priv_approval > 0 else float("nan")

        # Pred approval rate and EOpD
        if y_pred is not None:
            group_preds = y_pred[mask.values]
            pred_approval = float((group_preds == pos).mean())
            group_tpr = float(((group_preds == pos) & (group_y == pos)).sum() /
                              max((group_y == pos).sum(), 1))
            eopd = abs(group_tpr - priv_tpr) if priv_tpr is not None else float("nan")
        else:
            pred_approval = float("nan")
            eopd = float("nan")

        label = " × ".join(f"{v}" for v in combo)

        results.append(IntersectionalResult(
            group_label=label,
            group_attrs=attrs_combo,
            group_size=group_size,
            approval_rate=approval,
            pred_approval_rate=pred_approval,
            disparate_impact=di,
            equal_opportunity_diff=eopd,
            severity=_severity(di),
        ))

    results.sort(key=lambda r: r.disparate_impact)
    return results


def get_heatmap_data(
    results: list[IntersectionalResult],
    row_attr: str,
    col_attr: str,
    value: str = "disparate_impact",
) -> pd.DataFrame:
    """
    Build a 2D pivot DataFrame for a Plotly heatmap.
    rows = values of row_attr, cols = values of col_attr, cells = DI or approval_rate.
    """
    records = []
    for r in results:
        if row_attr in r.group_attrs and col_attr in r.group_attrs:
            records.append({
                "row_val": r.group_attrs[row_attr],
                "col_val": r.group_attrs[col_attr],
                "value":   getattr(r, value, float("nan")),
            })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.pivot(index="row_val", columns="col_val", values="value")
