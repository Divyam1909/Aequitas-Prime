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
from src.bias_engine.detector import _coerce_priv_val


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


def _pick_intersectional_attrs(
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    max_attrs: int = 3,
    max_cardinality: int = 10,
) -> list[str]:
    """
    Select the best subset of protected attrs for intersectional analysis.

    Strategy:
    - Keep only attrs present in df_clean
    - Drop attrs whose unique-value count exceeds max_cardinality (too many combos)
    - Among survivors, prefer the most biased ones (lowest DI proxy: largest outcome gap)
    - Return at most max_attrs columns whose combined product of unique values ≤ 5_000
    """
    target  = config.target_col
    pos     = config.positive_label
    candidates = []
    for attr in config.protected_attrs:
        if attr not in df_clean.columns:
            continue
        n_unique = df_clean[attr].nunique()
        if n_unique < 2 or n_unique > max_cardinality:
            continue
        # Measure outcome gap as a bias proxy
        pv = _coerce_priv_val(df_clean[attr], config.privileged_values[attr])
        priv_rate   = float((df_clean.loc[df_clean[attr] == pv,  target] == pos).mean())
        unpriv_rate = float((df_clean.loc[df_clean[attr] != pv,  target] == pos).mean())
        gap = abs(priv_rate - unpriv_rate)
        candidates.append((attr, n_unique, gap))

    # Sort by gap descending (most biased first), then cardinality ascending (fewer combos)
    candidates.sort(key=lambda t: (-t[2], t[1]))

    selected, combo_count = [], 1
    for attr, n_unique, _ in candidates:
        if len(selected) >= max_attrs:
            break
        if combo_count * n_unique > 5_000:
            continue  # skip — would blow up the product
        selected.append(attr)
        combo_count *= n_unique

    return selected


def compute_intersectional_metrics(
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    y_pred: np.ndarray | None = None,
    attrs: list[str] | None = None,
) -> list[IntersectionalResult]:
    """
    Compute DI and EOpD for every combination of protected attribute values.

    Parameters
    ----------
    df_clean : DataFrame with original string label columns
    config : DatasetConfig
    y_pred : model predictions aligned to df_clean rows. If None, uses true labels.
    attrs : explicit list of attrs to use. If None, auto-selects via _pick_intersectional_attrs
            to prevent combinatorial explosion.

    Returns
    -------
    list of IntersectionalResult, sorted by DI ascending (worst first)
    """
    target = config.target_col
    pos    = config.positive_label

    # Auto-select a safe subset of attrs if not provided explicitly
    if attrs is None:
        active_attrs = _pick_intersectional_attrs(df_clean, config)
    else:
        active_attrs = [a for a in attrs if a in df_clean.columns]

    if len(active_attrs) < 1:
        return []

    # Privileged baseline: all selected attrs set to privileged values
    priv_mask = pd.Series(True, index=df_clean.index)
    for attr in active_attrs:
        pv = _coerce_priv_val(df_clean[attr], config.privileged_values[attr])
        priv_mask &= (df_clean[attr] == pv)

    priv_approval = float((df_clean.loc[priv_mask, target] == pos).mean())
    if priv_approval == 0 or priv_mask.sum() == 0:
        return []

    # For EOpD: TPR of privileged group
    if y_pred is not None:
        priv_y     = df_clean.loc[priv_mask, target]
        priv_preds = y_pred[priv_mask.values]
        priv_tpr   = float(((priv_preds == pos) & (priv_y == pos)).sum() /
                           max((priv_y == pos).sum(), 1))
    else:
        priv_tpr = None

    # Enumerate all value combinations (bounded by _pick_intersectional_attrs)
    attr_value_sets = [df_clean[a].unique().tolist() for a in active_attrs]
    results = []

    for combo in product(*attr_value_sets):
        attrs_combo = dict(zip(active_attrs, combo))
        mask = pd.Series(True, index=df_clean.index)
        for attr, val in attrs_combo.items():
            mask &= (df_clean[attr] == val)

        group_size = int(mask.sum())
        if group_size < MIN_GROUP_SIZE:
            continue

        group_y  = df_clean.loc[mask, target]
        approval = float((group_y == pos).mean())
        di       = approval / priv_approval if priv_approval > 0 else float("nan")

        if y_pred is not None:
            group_preds   = y_pred[mask.values]
            pred_approval = float((group_preds == pos).mean())
            group_tpr     = float(((group_preds == pos) & (group_y == pos)).sum() /
                                  max((group_y == pos).sum(), 1))
            eopd = abs(group_tpr - priv_tpr) if priv_tpr is not None else float("nan")
        else:
            pred_approval = float("nan")
            eopd          = float("nan")

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
