"""
Intersectional Bias Analysis — checks bias across combinations of protected
attributes (e.g. "Black Female" vs. "White Female" vs. "Black Male").

V2 changes:
  - Groups smaller than MIN_GROUP_SIZE_FULL are no longer silently dropped.
    They are reported with a Wilson-score confidence interval on the approval
    rate and a small_sample_warning=True flag.
  - MIN_GROUP_SIZE_SKIP (=5) still applies — groups with <5 samples are
    excluded because the Wilson CI is too wide to be meaningful.
  - IntersectionalResult gains:
      small_sample_warning : bool
      approval_rate_ci_lower : float
      approval_rate_ci_upper : float

Standard tools check sex OR race. We check all combinations simultaneously.
This is legally significant (Kimberlé Crenshaw's intersectionality theory)
and is required by EU AI Act recitals for high-risk AI systems.
"""

from __future__ import annotations
from dataclasses import dataclass, field
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

    # V2: small-sample warning and Wilson CI
    small_sample_warning: bool = False
    approval_rate_ci_lower: float = float("nan")
    approval_rate_ci_upper: float = float("nan")


SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.60,
    "HIGH":     0.80,
    "MEDIUM":   0.90,
}

MIN_GROUP_SIZE_FULL = 30   # below this → include with CI warning
MIN_GROUP_SIZE_SKIP = 5    # below this → skip (Wilson CI too wide to be useful)


def _severity(di: float) -> str:
    if di < SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if di < SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if di < SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "OK"


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Returns (lower, upper) ∈ [0, 1].
    Used for the approval rate of small demographic groups.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = successes / n
    centre = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    half   = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(np.clip(centre - half, 0, 1)), float(np.clip(centre + half, 0, 1))


def _pick_intersectional_attrs(
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    max_attrs: int = 3,
    max_cardinality: int = 10,
) -> list[str]:
    """
    Select the best subset of protected attrs for intersectional analysis.
    """
    target     = config.target_col
    pos        = config.positive_label
    candidates = []
    for attr in config.protected_attrs:
        if attr not in df_clean.columns:
            continue
        n_unique = df_clean[attr].nunique()
        if n_unique < 2 or n_unique > max_cardinality:
            continue
        pv = _coerce_priv_val(df_clean[attr], config.privileged_values[attr])
        priv_rate   = float((df_clean.loc[df_clean[attr] == pv,  target] == pos).mean())
        unpriv_rate = float((df_clean.loc[df_clean[attr] != pv,  target] == pos).mean())
        gap = abs(priv_rate - unpriv_rate)
        candidates.append((attr, n_unique, gap))

    candidates.sort(key=lambda t: (-t[2], t[1]))

    selected, combo_count = [], 1
    for attr, n_unique, _ in candidates:
        if len(selected) >= max_attrs:
            break
        if combo_count * n_unique > 5_000:
            continue
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

    V2: groups between MIN_GROUP_SIZE_SKIP and MIN_GROUP_SIZE_FULL are
    included with small_sample_warning=True and Wilson CI on approval_rate.
    Groups with fewer than MIN_GROUP_SIZE_SKIP samples are still excluded.

    Returns
    -------
    list of IntersectionalResult sorted by DI ascending (worst first)
    """
    target = config.target_col
    pos    = config.positive_label

    if attrs is None:
        active_attrs = _pick_intersectional_attrs(df_clean, config)
    else:
        active_attrs = [a for a in attrs if a in df_clean.columns]

    if len(active_attrs) < 1:
        return []

    # Privileged baseline: all selected attrs at their privileged values
    priv_mask = pd.Series(True, index=df_clean.index)
    for attr in active_attrs:
        pv = _coerce_priv_val(df_clean[attr], config.privileged_values[attr])
        priv_mask &= (df_clean[attr] == pv)

    priv_approval = float((df_clean.loc[priv_mask, target] == pos).mean())
    if priv_approval == 0 or priv_mask.sum() == 0:
        return []

    # TPR of privileged group for EOpD
    if y_pred is not None:
        priv_y     = df_clean.loc[priv_mask, target]
        priv_preds = y_pred[priv_mask.values]
        priv_tpr   = float(((priv_preds == pos) & (priv_y == pos)).sum() /
                           max((priv_y == pos).sum(), 1))
    else:
        priv_tpr = None

    attr_value_sets = [df_clean[a].unique().tolist() for a in active_attrs]
    results = []

    for combo in product(*attr_value_sets):
        attrs_combo = dict(zip(active_attrs, combo))
        mask = pd.Series(True, index=df_clean.index)
        for attr, val in attrs_combo.items():
            mask &= (df_clean[attr] == val)

        group_size = int(mask.sum())

        # V2: skip only truly tiny groups; warn on small ones
        if group_size < MIN_GROUP_SIZE_SKIP:
            continue

        small_warn = group_size < MIN_GROUP_SIZE_FULL

        group_y  = df_clean.loc[mask, target]
        approval = float((group_y == pos).mean())
        di       = approval / priv_approval if priv_approval > 0 else float("nan")

        # Wilson CI for approval rate
        successes = int((group_y == pos).sum())
        ci_lo, ci_hi = _wilson_ci(successes, group_size)

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
            small_sample_warning=small_warn,
            approval_rate_ci_lower=ci_lo,
            approval_rate_ci_upper=ci_hi,
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
