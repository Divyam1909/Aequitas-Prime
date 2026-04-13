"""
Shadow Proxy Scanner — detects features that are statistical proxies for
protected attributes even after the protected attribute itself is removed.

Uses three complementary measures:
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
    risk_level: str       # "HIGH" / "MEDIUM" / "LOW" / "NEGLIGIBLE"
    action: str


RISK_THRESHOLDS = {
    "HIGH":       0.15,
    "MEDIUM":     0.08,
    "LOW":        0.03,
}

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


def scan_proxies(
    df: pd.DataFrame,
    config: DatasetConfig,
    exclude_cols: list[str] | None = None,
) -> dict[str, list[ProxyResult]]:
    """
    Scan all non-protected, non-target features for proxy risk.

    Parameters
    ----------
    df : pd.DataFrame
        The clean (encoded) DataFrame. Protected attrs must be present as columns.
    config : DatasetConfig
    exclude_cols : list[str], optional
        Additional columns to skip (e.g. sample_weight).

    Returns
    -------
    dict mapping protected_attr → sorted list of ProxyResult (highest MI first)
    """
    exclude = set(config.protected_attrs) | {config.target_col} | set(exclude_cols or [])
    feature_cols = [c for c in df.columns if c not in exclude]

    results: dict[str, list[ProxyResult]] = {}

    for attr in config.protected_attrs:
        # Support renamed columns (e.g. "race" → "race_binary" after encoding)
        resolved_attr = attr if attr in df.columns else f"{attr}_binary"
        if resolved_attr not in df.columns:
            continue
        attr_col_name = resolved_attr  # actual column in df

        protected_series = df[attr_col_name]
        # Exclude the resolved column name too
        feat_cols_for_attr = [c for c in feature_cols if c != attr_col_name]
        # For MI: needs a clean array with no NaN
        valid_mask = protected_series.notna()
        X_clean = df.loc[valid_mask, feat_cols_for_attr].fillna(0)
        y_clean = protected_series[valid_mask]

        # Encode protected attr to numeric for MI if needed
        if y_clean.dtype == object:
            y_num = pd.Categorical(y_clean).codes
        else:
            y_num = y_clean.values.astype(int)

        # Mutual information (discrete_features=False → treats all as continuous)
        mi_scores = mutual_info_classif(
            X_clean.values, y_num, random_state=42, n_neighbors=3
        )

        attr_results = []
        for i, feat in enumerate(feat_cols_for_attr):
            mi = float(mi_scores[i])

            # Cramér's V (only meaningful for categorical features — low cardinality)
            feat_series = df[feat].dropna()
            n_unique = feat_series.nunique()
            cv = None
            if n_unique <= 20:
                aligned = df[[feat, attr_col_name]].dropna()
                if len(aligned) > 10:
                    cv = _cramers_v(aligned[feat].astype(str), aligned[attr_col_name].astype(str))

            # Point-Biserial (only for continuous feature vs binary protected)
            pb_r = None
            if n_unique > 20 and len(y_num) > 10:
                try:
                    result = stats.pointbiserialr(y_num, X_clean[feat].values)
                    pb_r = float(abs(result.statistic))
                except Exception:
                    pass

            # Risk level driven by MI score
            if mi >= RISK_THRESHOLDS["HIGH"]:
                risk = "HIGH"
            elif mi >= RISK_THRESHOLDS["MEDIUM"]:
                risk = "MEDIUM"
            elif mi >= RISK_THRESHOLDS["LOW"]:
                risk = "LOW"
            else:
                risk = "NEGLIGIBLE"

            attr_results.append(ProxyResult(
                feature=feat,
                protected_attr=attr,
                mutual_info=mi,
                cramers_v=cv,
                point_biserial_r=pb_r,
                risk_level=risk,
                action=ACTIONS[risk],
            ))

        # Sort by MI descending
        attr_results.sort(key=lambda r: r.mutual_info, reverse=True)
        results[attr] = attr_results

    return results


def get_flagged_proxies(scan_results: dict[str, list[ProxyResult]]) -> list[ProxyResult]:
    """Return only HIGH and MEDIUM risk proxies across all protected attrs."""
    flagged = []
    for results in scan_results.values():
        flagged.extend([r for r in results if r.risk_level in ("HIGH", "MEDIUM")])
    flagged.sort(key=lambda r: r.mutual_info, reverse=True)
    return flagged


def proxy_heatmap_data(scan_results: dict[str, list[ProxyResult]]) -> pd.DataFrame:
    """
    Build a DataFrame suitable for a Plotly heatmap.
    Rows = features, Columns = protected attributes, Values = mutual information score.
    """
    rows = {}
    for attr, results in scan_results.items():
        for r in results:
            if r.feature not in rows:
                rows[r.feature] = {}
            rows[r.feature][attr] = r.mutual_info
    return pd.DataFrame(rows).T.fillna(0)
