"""
Bias Detection Engine — computes 6 standard fairness metrics.

Pre-training metrics (need only df + config):
  - Disparate Impact (DI)
  - Statistical Parity Difference (SPD)

Post-training metrics (need y_pred):
  - Equalized Odds Difference (EOD)
  - Equal Opportunity Difference (EOpD)
  - Predictive Parity / Calibration (PP)
  - False Negative Rate Parity (FNRP)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from src.utils.schema import DatasetConfig

# ── Thresholds (fail conditions) ──────────────────────────────────────────────
DI_THRESHOLD     = 0.80   # below = FAIL  (legal 4/5ths rule)
SPD_THRESHOLD    = -0.10  # below = FAIL
EOD_THRESHOLD    = 0.10   # above = FAIL
EOPD_THRESHOLD   = 0.10   # above = FAIL
PP_THRESHOLD     = 0.10   # above = FAIL
FNRP_THRESHOLD   = 0.10   # above = FAIL


@dataclass
class MetricsResult:
    """Holds all computed fairness metrics for one protected attribute."""
    protected_attr: str
    privileged_value: object

    # Pre-training
    disparate_impact: float = float("nan")
    statistical_parity_diff: float = float("nan")

    # Post-training (require y_pred)
    equalized_odds_diff: float = float("nan")
    equal_opportunity_diff: float = float("nan")
    predictive_parity_diff: float = float("nan")
    fnr_parity_diff: float = float("nan")

    # Group stats for UI
    privileged_approval_rate: float = float("nan")
    unprivileged_approval_rate: float = float("nan")
    privileged_count: int = 0
    unprivileged_count: int = 0

    # Aggregated verdict
    failures: list[str] = field(default_factory=list)
    severity: str = "UNKNOWN"   # CLEAR / WARNING / CRITICAL


def _coerce_priv_val(col: pd.Series, priv_val: object) -> object:
    """
    Cast priv_val to match the dtype of col so that equality comparisons work
    even when the UI always stores values as strings but the column is numeric.
    E.g. col dtype=int64, priv_val="23" → returns 23 (int).
    """
    if not pd.api.types.is_object_dtype(col) and not pd.api.types.is_string_dtype(col):
        try:
            return col.dtype.type(priv_val)
        except (ValueError, TypeError):
            pass
    return priv_val


def _split_groups(
    df: pd.DataFrame, attr: str, priv_val: object, target_col: str
) -> tuple[pd.Series, pd.Series]:
    """Return (privileged_target_series, unprivileged_target_series)."""
    col = df[attr]
    priv_val = _coerce_priv_val(col, priv_val)
    priv_mask = col == priv_val
    return df.loc[priv_mask, target_col], df.loc[~priv_mask, target_col]


def compute_disparate_impact(
    df: pd.DataFrame, config: DatasetConfig, attr: str | None = None
) -> float:
    """P(Y=1|unprivileged) / P(Y=1|privileged)"""
    attr = attr or config.primary_protected_attr()
    priv_y, unpriv_y = _split_groups(df, attr, config.privileged_values[attr], config.target_col)
    p_priv   = (priv_y   == config.positive_label).mean()
    p_unpriv = (unpriv_y == config.positive_label).mean()
    if p_priv == 0:
        return float("nan")
    return float(p_unpriv / p_priv)


def compute_spd(
    df: pd.DataFrame, config: DatasetConfig, attr: str | None = None
) -> float:
    """P(Y=1|unprivileged) − P(Y=1|privileged)"""
    attr = attr or config.primary_protected_attr()
    priv_y, unpriv_y = _split_groups(df, attr, config.privileged_values[attr], config.target_col)
    p_priv   = (priv_y   == config.positive_label).mean()
    p_unpriv = (unpriv_y == config.positive_label).mean()
    return float(p_unpriv - p_priv)


def _confusion_rates(y_true: pd.Series, y_pred: np.ndarray, pos_label: int = 1):
    """Return TPR, FPR, PPV (precision), FNR for a group."""
    y_true = np.array(y_true)
    tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
    fp = ((y_pred == pos_label) & (y_true != pos_label)).sum()
    tn = ((y_pred != pos_label) & (y_true != pos_label)).sum()
    fn = ((y_pred != pos_label) & (y_true == pos_label)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")  # precision
    fnr = fn / (tp + fn) if (tp + fn) > 0 else float("nan")
    return tpr, fpr, ppv, fnr


def compute_all_metrics(
    df: pd.DataFrame,
    config: DatasetConfig,
    y_pred: np.ndarray | None = None,
    attr: str | None = None,
) -> MetricsResult:
    """
    Compute all 6 fairness metrics for one protected attribute.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain protected attr column and target column (true labels).
    config : DatasetConfig
    y_pred : np.ndarray, optional
        Model predictions. Required for post-training metrics.
    attr : str, optional
        Which protected attribute to analyse. Defaults to the primary one.
    """
    attr = attr or config.primary_protected_attr()
    priv_val = _coerce_priv_val(df[attr], config.privileged_values[attr])
    priv_mask = df[attr] == priv_val

    priv_y   = df.loc[priv_mask,  config.target_col]
    unpriv_y = df.loc[~priv_mask, config.target_col]

    p_priv   = (priv_y   == config.positive_label).mean()
    p_unpriv = (unpriv_y == config.positive_label).mean()

    result = MetricsResult(
        protected_attr=attr,
        privileged_value=priv_val,
        privileged_approval_rate=float(p_priv),
        unprivileged_approval_rate=float(p_unpriv),
        privileged_count=int(priv_mask.sum()),
        unprivileged_count=int((~priv_mask).sum()),
    )

    # Pre-training metrics
    result.disparate_impact = float(p_unpriv / p_priv) if p_priv > 0 else float("nan")
    result.statistical_parity_diff = float(p_unpriv - p_priv)

    # Post-training metrics
    if y_pred is not None:
        priv_pred   = y_pred[priv_mask.values]
        unpriv_pred = y_pred[~priv_mask.values]

        tpr_p,  fpr_p,  ppv_p,  fnr_p  = _confusion_rates(priv_y,   priv_pred,   config.positive_label)
        tpr_u,  fpr_u,  ppv_u,  fnr_u  = _confusion_rates(unpriv_y, unpriv_pred, config.positive_label)

        result.equalized_odds_diff    = float(max(abs(tpr_p - tpr_u), abs(fpr_p - fpr_u)))
        result.equal_opportunity_diff = float(abs(tpr_p - tpr_u))
        result.predictive_parity_diff = float(abs(ppv_p - ppv_u))  if not (np.isnan(ppv_p) or np.isnan(ppv_u)) else float("nan")
        result.fnr_parity_diff        = float(abs(fnr_p - fnr_u))  if not (np.isnan(fnr_p) or np.isnan(fnr_u)) else float("nan")

    # Determine pass/fail
    result.failures = _flag_failures(result)
    result.severity = _severity(result.failures)
    return result


def _flag_failures(r: MetricsResult) -> list[str]:
    failures = []
    if not np.isnan(r.disparate_impact) and r.disparate_impact < DI_THRESHOLD:
        failures.append(f"Disparate Impact = {r.disparate_impact:.3f} (< {DI_THRESHOLD})")
    if not np.isnan(r.statistical_parity_diff) and r.statistical_parity_diff < SPD_THRESHOLD:
        failures.append(f"Statistical Parity Diff = {r.statistical_parity_diff:.3f} (< {SPD_THRESHOLD})")
    if not np.isnan(r.equalized_odds_diff) and r.equalized_odds_diff > EOD_THRESHOLD:
        failures.append(f"Equalized Odds Diff = {r.equalized_odds_diff:.3f} (> {EOD_THRESHOLD})")
    if not np.isnan(r.equal_opportunity_diff) and r.equal_opportunity_diff > EOPD_THRESHOLD:
        failures.append(f"Equal Opportunity Diff = {r.equal_opportunity_diff:.3f} (> {EOPD_THRESHOLD})")
    if not np.isnan(r.predictive_parity_diff) and r.predictive_parity_diff > PP_THRESHOLD:
        failures.append(f"Predictive Parity Diff = {r.predictive_parity_diff:.3f} (> {PP_THRESHOLD})")
    if not np.isnan(r.fnr_parity_diff) and r.fnr_parity_diff > FNRP_THRESHOLD:
        failures.append(f"FNR Parity Diff = {r.fnr_parity_diff:.3f} (> {FNRP_THRESHOLD})")
    return failures


def _severity(failures: list[str]) -> str:
    n = len(failures)
    if n == 0:
        return "CLEAR"
    elif n <= 2:
        return "WARNING"
    else:
        return "CRITICAL"


def compute_metrics_for_all_attrs(
    df: pd.DataFrame,
    config: DatasetConfig,
    y_pred: np.ndarray | None = None,
) -> dict[str, MetricsResult]:
    """Run compute_all_metrics for every protected attribute in config."""
    return {
        attr: compute_all_metrics(df, config, y_pred=y_pred, attr=attr)
        for attr in config.protected_attrs
    }
