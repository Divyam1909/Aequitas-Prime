"""
Shadow Proxy Scanner — detects features that are statistical proxies for
protected attributes even after the protected attribute itself is removed.

V2 changes:
  - combined_score: weighted fusion of MI + Cramér's V + Point-Biserial
  - Risk level now driven by combined_score (not MI alone)
  - ProxyResult.combined_score field added
  - Weights: MI=0.50, Cramér's V=0.30, Point-Biserial=0.20

Three complementary measures:
  - Mutual Information (works for any feature type)
  - Cramér's V (for categorical vs categorical)
  - Point-Biserial correlation (for continuous vs binary protected attr)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from src.utils.schema import DatasetConfig


@dataclass
class ProxyResult:
    feature: str
    protected_attr: str
    mutual_info: float
    cramers_v: float | None
    point_biserial_r: float | None
    # V2: fused score and risk driven by it
    combined_score: float = 0.0
    risk_level: str = "NEGLIGIBLE"   # "HIGH" / "MEDIUM" / "LOW" / "NEGLIGIBLE"
    action: str = ""


# ── Risk thresholds on combined_score (0–1 scale) ───────────────────────────
RISK_THRESHOLDS = {
    "HIGH":       0.15,
    "MEDIUM":     0.08,
    "LOW":        0.03,
}

# Fusion weights (must sum to 1.0)
_W_MI  = 0.50
_W_CV  = 0.30
_W_PBR = 0.20

# Typical maximum values used for normalisation
_MAX_MI  = 1.0   # MI is unbounded but rarely > 1 on tabular data
_MAX_CV  = 1.0   # Cramér's V is already in [0, 1]
_MAX_PBR = 1.0   # |point-biserial r| is in [0, 1]

ACTIONS = {
    "HIGH":       "Consider removing or capping — strong statistical proxy for the protected attribute.",
    "MEDIUM":     "Monitor closely — may re-introduce demographic bias as a back-door proxy.",
    "LOW":        "Low risk. Track across model versions.",
    "NEGLIGIBLE": "No significant association with the protected attribute.",
}


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V statistic for two categorical series."""
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def _fused_score(mi: float, cv: float | None, pbr: float | None) -> float:
    """
    Compute a weighted combination of all three association measures.

    Any missing signal gets its weight redistributed proportionally to the
    available signals so the result always stays in [0, 1].
    """
    available = {"mi": mi / _MAX_MI}
    weights   = {"mi": _W_MI}

    if cv is not None:
        available["cv"]  = cv / _MAX_CV
        weights["cv"]    = _W_CV
    if pbr is not None:
        available["pbr"] = pbr / _MAX_PBR
        weights["pbr"]   = _W_PBR

    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0

    score = sum(available[k] * weights[k] for k in available) / total_w
    return float(np.clip(score, 0.0, 1.0))


def _risk_from_score(score: float) -> str:
    if score >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if score >= RISK_THRESHOLDS["LOW"]:
        return "LOW"
    return "NEGLIGIBLE"


def scan_proxies(
    df: pd.DataFrame,
    config: DatasetConfig,
    exclude_cols: list[str] | None = None,
) -> dict[str, list[ProxyResult]]:
    """
    Scan all non-protected, non-target features for proxy risk.

    V2: risk level is now determined by combined_score (weighted fusion of
    MI, Cramér's V, and Point-Biserial) rather than MI alone.

    Returns
    -------
    dict mapping protected_attr → sorted list of ProxyResult (highest combined_score first)
    """
    exclude = set(config.protected_attrs) | {config.target_col} | set(exclude_cols or [])
    feature_cols = [c for c in df.columns if c not in exclude]

    results: dict[str, list[ProxyResult]] = {}

    for attr in config.protected_attrs:
        resolved_attr = attr if attr in df.columns else f"{attr}_binary"
        if resolved_attr not in df.columns:
            continue
        attr_col_name = resolved_attr

        protected_series = df[attr_col_name]
        feat_cols_for_attr = [c for c in feature_cols if c != attr_col_name]

        valid_mask = protected_series.notna()
        X_clean = df.loc[valid_mask, feat_cols_for_attr].fillna(0)
        y_clean = protected_series[valid_mask]

        if y_clean.dtype == object:
            y_num = pd.Categorical(y_clean).codes
        else:
            y_num = y_clean.values.astype(int)

        mi_scores = mutual_info_classif(
            X_clean.values, y_num, random_state=42, n_neighbors=3
        )

        attr_results = []
        for i, feat in enumerate(feat_cols_for_attr):
            mi = float(mi_scores[i])

            feat_series = df[feat].dropna()
            n_unique = feat_series.nunique()
            cv = None
            pbr = None

            # Cramér's V for low-cardinality features
            if n_unique <= 20:
                aligned = df[[feat, attr_col_name]].dropna()
                if len(aligned) > 10:
                    cv = _cramers_v(
                        aligned[feat].astype(str),
                        aligned[attr_col_name].astype(str),
                    )

            # Point-Biserial for continuous features vs binary protected attr
            if n_unique > 20 and len(y_num) > 10:
                try:
                    res = stats.pointbiserialr(y_num, X_clean[feat].values)
                    pbr = float(abs(res.statistic))
                except Exception:
                    pass

            combined = _fused_score(mi, cv, pbr)
            risk = _risk_from_score(combined)

            attr_results.append(ProxyResult(
                feature=feat,
                protected_attr=attr,
                mutual_info=mi,
                cramers_v=cv,
                point_biserial_r=pbr,
                combined_score=combined,
                risk_level=risk,
                action=ACTIONS[risk],
            ))

        # Sort by combined_score descending (V2: was MI-only)
        attr_results.sort(key=lambda r: r.combined_score, reverse=True)
        results[attr] = attr_results

    return results


def get_flagged_proxies(scan_results: dict[str, list[ProxyResult]]) -> list[ProxyResult]:
    """Return only HIGH and MEDIUM risk proxies across all protected attrs."""
    flagged = []
    for results in scan_results.values():
        flagged.extend([r for r in results if r.risk_level in ("HIGH", "MEDIUM")])
    flagged.sort(key=lambda r: r.combined_score, reverse=True)
    return flagged


def proxy_heatmap_data(scan_results: dict[str, list[ProxyResult]]) -> pd.DataFrame:
    """
    Build a DataFrame suitable for a Plotly heatmap.
    Rows = features, Columns = protected attributes, Values = combined_score (V2).
    """
    rows = {}
    for attr, results in scan_results.items():
        for r in results:
            if r.feature not in rows:
                rows[r.feature] = {}
            rows[r.feature][attr] = r.combined_score   # V2: was mutual_info
    return pd.DataFrame(rows).T.fillna(0)
