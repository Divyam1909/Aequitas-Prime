"""
Causal Fairness Analysis — mediation decomposition of direct vs. indirect
discrimination.

Definitions
-----------
  Total Effect (TE)     : overall association between the protected attribute A
                          and outcome Y, ignoring pathways.
  Direct Effect (DE)    : A → Y path not mediated by any selected mediator M.
                          Corresponds to "direct discrimination" in law.
  Indirect Effect (IE)  : A → M → Y path.  Corresponds to "structural" or
                          "indirect discrimination" transmitted through mediators
                          such as zip-code, occupation, or credit score.

Method
------
  Baron-Kenny path-coefficient decomposition (OLS, parametric):
    1.  Regress Y on A (+ confounders) → total effect β_total
    2.  Regress M on A (+ confounders) → α  (A→M path)
    3.  Regress Y on A + M (+ confounders) → β_direct, γ  (A→Y|M, M→Y)
    4.  Indirect = α × γ  (product of coefficients)
    5.  Direct   = β_direct

  Sobel test p-value provided for the indirect path.
  Percent mediated = IE / TE × 100.

  If statsmodels is unavailable the module still works through a pure-numpy
  OLS implementation so there are no hard external dependencies.

V2 notes
--------
  - Supports any binary or continuous protected attribute A.
  - Supports multiple mediators; analyses each independently and reports the
    top-k by |indirect effect|.
  - Returns CausalFairnessResult for each (attr, mediator) pair.
  - Uses df_clean (post-preprocessing) — numeric columns only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from src.utils.schema import DatasetConfig


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class MediationPathResult:
    """Mediation decomposition for one (protected_attr, mediator) pair."""
    protected_attr: str
    mediator: str
    total_effect: float          # β_total  (A → Y, unadjusted for M)
    direct_effect: float         # β_direct (A → Y | M)
    indirect_effect: float       # α × γ    (A → M → Y)
    sobel_z: float               # Sobel test statistic
    sobel_p: float               # approximate p-value (two-sided)
    percent_mediated: float      # IE / TE × 100  (may be nan if TE ≈ 0)
    n_samples: int
    interpretation: str          # plain-English summary


@dataclass
class CausalFairnessReport:
    """Full causal fairness report for one dataset."""
    protected_attr: str
    mediators_tested: list[str]
    paths: list[MediationPathResult]           # one per mediator
    dominant_path: MediationPathResult | None  # highest |IE|
    summary: str                               # one-paragraph plain English

    # Auto-detected candidate mediators (for UI display)
    candidate_mediators: list[str] = field(default_factory=list)


# ── OLS helpers ───────────────────────────────────────────────────────────────

def _ols_coef(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form OLS: returns (coefficients, standard_errors).
    X should already include a constant column.
    """
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        n, k = X.shape
        y_hat = X @ beta
        resid = y - y_hat
        sigma2 = (resid @ resid) / max(n - k, 1)
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
        return beta, se
    except np.linalg.LinAlgError:
        k = X.shape[1]
        return np.zeros(k), np.ones(k) * float("nan")


def _add_const(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones (intercept)."""
    return np.column_stack([np.ones(len(X)), X])


# ── Mediator candidate selection ─────────────────────────────────────────────

def _select_mediator_candidates(
    df: pd.DataFrame,
    config: DatasetConfig,
    attr: str,
    max_candidates: int = 8,
    max_cardinality: int = 20,
) -> list[str]:
    """
    Heuristic: columns that are:
      - Not the target
      - Not a protected attribute
      - Numeric (or low-cardinality integer that could be one-hot encoded)
      - Correlated ≥ 0.05 with the protected attribute

    Returns up to max_candidates column names sorted by |corr(col, attr)|.
    """
    exclude = set(config.protected_attrs) | {config.target_col}
    candidates = []

    if attr not in df.columns:
        return []

    a_vals = pd.to_numeric(df[attr], errors="coerce").dropna()

    for col in df.columns:
        if col in exclude:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().mean() > 0.5:
            continue
        if df[col].nunique() > max_cardinality and df[col].dtype == object:
            continue
        aligned = s.reindex(a_vals.index).dropna()
        a_aligned = a_vals.reindex(aligned.index)
        if len(aligned) < 30:
            continue
        try:
            corr = abs(float(np.corrcoef(a_aligned.values, aligned.values)[0, 1]))
        except Exception:
            corr = 0.0
        if corr >= 0.05:
            candidates.append((col, corr))

    candidates.sort(key=lambda t: -t[1])
    return [c for c, _ in candidates[:max_candidates]]


# ── Sobel test ────────────────────────────────────────────────────────────────

def _sobel_p(alpha: float, gamma: float, se_alpha: float, se_gamma: float) -> tuple[float, float]:
    """
    Sobel (1982) test for the indirect effect α × γ.
    Returns (z-statistic, two-sided p-value).
    """
    from math import sqrt, isnan
    import math
    se_sobel = sqrt(gamma**2 * se_alpha**2 + alpha**2 * se_gamma**2)
    if se_sobel == 0 or isnan(se_sobel):
        return float("nan"), float("nan")
    z = (alpha * gamma) / se_sobel
    # Approximate p-value from standard normal
    try:
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(z))))
    except ImportError:
        # Rough approximation without scipy
        az = abs(z)
        p = float(2 * math.exp(-0.5 * az * az) / (az + 1e-9) / (2 * math.pi) ** 0.5) if az > 0 else 1.0
        p = min(max(p, 0.0), 1.0)
    return float(z), float(p)


# ── Single mediation path ─────────────────────────────────────────────────────

def _mediation_path(
    df: pd.DataFrame,
    config: DatasetConfig,
    attr: str,
    mediator: str,
    confounders: list[str] | None = None,
) -> MediationPathResult:
    """
    Run Baron-Kenny mediation for a single (attr, mediator) pair.
    """
    target = config.target_col

    # Numeric-only slice
    cols_needed = [attr, mediator, target] + (confounders or [])
    sub = df[cols_needed].copy()
    for c in cols_needed:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    n = len(sub)

    if n < 30:
        return MediationPathResult(
            protected_attr=attr, mediator=mediator,
            total_effect=float("nan"), direct_effect=float("nan"),
            indirect_effect=float("nan"), sobel_z=float("nan"), sobel_p=float("nan"),
            percent_mediated=float("nan"), n_samples=n,
            interpretation=f"Insufficient data (n={n}) for mediation analysis.",
        )

    A = sub[attr].values
    M = sub[mediator].values
    Y = sub[target].values
    C = sub[confounders].values if confounders else np.empty((n, 0))

    # ── Step 1: Y ~ A (+ C)  →  total effect ──────────────────────────────────
    X1 = _add_const(np.column_stack([A, C]) if C.shape[1] > 0 else A.reshape(-1, 1))
    beta1, se1 = _ols_coef(X1, Y)
    beta_total = float(beta1[1])   # coefficient on A (index 0 = intercept)
    se_total   = float(se1[1])

    # ── Step 2: M ~ A (+ C)  →  α path ───────────────────────────────────────
    X2 = _add_const(np.column_stack([A, C]) if C.shape[1] > 0 else A.reshape(-1, 1))
    beta2, se2 = _ols_coef(X2, M)
    alpha    = float(beta2[1])
    se_alpha = float(se2[1])

    # ── Step 3: Y ~ A + M (+ C)  →  direct effect + γ ────────────────────────
    AM = np.column_stack([A, M])
    if C.shape[1] > 0:
        AM = np.column_stack([AM, C])
    X3 = _add_const(AM)
    beta3, se3 = _ols_coef(X3, Y)
    beta_direct = float(beta3[1])   # A coefficient (with M in model)
    gamma       = float(beta3[2])   # M coefficient
    se_gamma    = float(se3[2])

    # ── Step 4: Indirect effect ───────────────────────────────────────────────
    indirect = alpha * gamma
    z, p     = _sobel_p(alpha, gamma, se_alpha, se_gamma)

    pct = (indirect / beta_total * 100) if abs(beta_total) > 1e-9 else float("nan")

    # ── Interpretation ────────────────────────────────────────────────────────
    direction = "positive" if beta_total > 0 else "negative"
    bias_type = []
    if abs(beta_direct) > 0.01:
        bias_type.append(f"direct discrimination (β_direct={beta_direct:.3f})")
    if abs(indirect) > 0.01 and not np.isnan(p) and p < 0.1:
        pct_str = f"{abs(pct):.1f}%" if not np.isnan(pct) else "unknown%"
        bias_type.append(
            f"indirect discrimination via '{mediator}' "
            f"({pct_str} of total effect, Sobel p={p:.3f})"
        )
    if not bias_type:
        bias_type.append("no significant mediation detected")

    interp = (
        f"'{attr}' has a {direction} total effect on '{target}' "
        f"(β_total={beta_total:.3f}, n={n}). "
        + " | ".join(bias_type) + "."
    )

    return MediationPathResult(
        protected_attr=attr,
        mediator=mediator,
        total_effect=beta_total,
        direct_effect=beta_direct,
        indirect_effect=indirect,
        sobel_z=z,
        sobel_p=p,
        percent_mediated=pct,
        n_samples=n,
        interpretation=interp,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_causal_analysis(
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    attr: str | None = None,
    mediators: list[str] | None = None,
    confounders: list[str] | None = None,
    max_mediators: int = 5,
) -> CausalFairnessReport:
    """
    Run mediation-based causal fairness analysis.

    Parameters
    ----------
    df_clean   : post-preprocessing DataFrame (numeric columns preferred)
    config     : DatasetConfig for this dataset
    attr       : protected attribute to analyse (default: primary_protected_attr)
    mediators  : explicit list of mediator columns; auto-detected if None
    confounders: list of confounder columns to include in all regressions
    max_mediators : max number of mediators to analyse

    Returns
    -------
    CausalFairnessReport with per-mediator MediationPathResult objects
    """
    if attr is None:
        attr = config.primary_protected_attr()

    # Auto-detect mediator candidates if not provided
    if mediators is None:
        candidates = _select_mediator_candidates(df_clean, config, attr)
    else:
        candidates = [m for m in mediators if m in df_clean.columns]

    mediators_used = candidates[:max_mediators]
    paths: list[MediationPathResult] = []

    for med in mediators_used:
        path = _mediation_path(df_clean, config, attr, med, confounders)
        paths.append(path)

    # Sort by |indirect effect| descending
    paths.sort(key=lambda p: abs(p.indirect_effect) if not np.isnan(p.indirect_effect) else 0, reverse=True)

    dominant = paths[0] if paths else None

    # Build plain-English summary
    if dominant and not np.isnan(dominant.indirect_effect):
        pct_str = f"{abs(dominant.percent_mediated):.1f}%" if not np.isnan(dominant.percent_mediated) else "a notable portion"
        summary = (
            f"Causal mediation analysis for '{attr}' identified {len(paths)} potential indirect "
            f"pathway(s). The strongest indirect path runs through '{dominant.mediator}', "
            f"accounting for {pct_str} of the total effect "
            f"(Sobel z={dominant.sobel_z:.2f}, p={dominant.sobel_p:.3f}). "
            f"Direct discrimination (A→Y, not through mediators) coefficient: "
            f"{dominant.direct_effect:.3f}. "
        )
        if not np.isnan(dominant.sobel_p) and dominant.sobel_p < 0.05:
            summary += (
                "The indirect pathway is statistically significant (p<0.05), suggesting "
                "that at least part of the disparity is transmitted structurally. "
                "Mitigation should address the mediating variable in addition to the "
                "protected attribute itself."
            )
        else:
            summary += (
                "The indirect pathway does not reach conventional significance (p≥0.05). "
                "The observed disparity appears to be primarily direct rather than mediated."
            )
    elif paths:
        summary = (
            f"Mediation analysis ran for {len(paths)} pathway(s) from '{attr}', "
            "but insufficient data or near-zero effects prevented confident decomposition. "
            "Consider collecting more samples or checking mediator selection."
        )
    else:
        summary = (
            f"No suitable mediator candidates found for '{attr}'. "
            "Ensure the dataset contains features that may lie on the causal path "
            "between the protected attribute and the outcome."
        )

    return CausalFairnessReport(
        protected_attr=attr,
        mediators_tested=mediators_used,
        paths=paths,
        dominant_path=dominant,
        summary=summary,
        candidate_mediators=candidates,
    )


def causal_waterfall_data(report: CausalFairnessReport) -> pd.DataFrame:
    """
    Build a DataFrame suitable for a waterfall / stacked-bar chart showing
    total, direct, and indirect effects per mediator.
    """
    rows = []
    for p in report.paths:
        rows.append({
            "mediator":        p.mediator,
            "total_effect":    p.total_effect,
            "direct_effect":   p.direct_effect,
            "indirect_effect": p.indirect_effect,
            "sobel_p":         p.sobel_p,
            "pct_mediated":    p.percent_mediated,
            "n":               p.n_samples,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["mediator", "total_effect", "direct_effect",
                 "indirect_effect", "sobel_p", "pct_mediated", "n"]
    )
