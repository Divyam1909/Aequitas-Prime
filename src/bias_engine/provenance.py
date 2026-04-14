"""
Bias Provenance Tracer — identifies WHERE the bias originates.

V2 changes:
  - Replaced magic-weight heuristics (0.6, 0.15, etc.) with principled
    normalised scoring:
      * Representation: Gini-based imbalance index → [0, 1]
      * Feature:        mean combined_score of top proxies → [0, 1]
      * Label:          normalised label-ratio entropy → [0, 1]
  - Confidence is now the margin between top-1 and top-2 scores (clearer
    interpretation than the old partial-credit formula).
  - Each signal is independently scaled so no single hand-tuned constant
    can dominate.

Three root causes:
  Label Bias        → historical human decisions were biased (fix: Reweighing)
  Representation    → underrepresented groups (fix: more data / augmentation)
  Feature Bias      → proxy features encode protected attribute (fix: DIR)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.utils.schema import DatasetConfig
from src.bias_engine.proxy_scanner import ProxyResult


@dataclass
class ProvenanceReport:
    primary_source: str                  # "label_bias" | "representation_bias" | "feature_bias"
    secondary_source: str | None
    confidence: float                    # 0.0–1.0
    evidence: list[str] = field(default_factory=list)
    recommended_fixes: list[str] = field(default_factory=list)
    caution_notes: list[str] = field(default_factory=list)
    # V2: raw normalised scores for transparency
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def primary_label(self) -> str:
        labels = {
            "label_bias":          "Label Bias",
            "representation_bias": "Representation Bias",
            "feature_bias":        "Feature / Proxy Bias",
        }
        return labels.get(self.primary_source, self.primary_source)

    @property
    def primary_description(self) -> str:
        descs = {
            "label_bias": (
                "Historical human decisions in the training data were biased. "
                "The target labels (Y) themselves reflect past discrimination, "
                "not ground truth."
            ),
            "representation_bias": (
                "Certain demographic groups are severely underrepresented in the "
                "training data. The model has seen too few examples of minority "
                "group members to learn fair patterns."
            ),
            "feature_bias": (
                "Features in the dataset act as statistical proxies for the "
                "protected attribute. Even after removing the protected column, "
                "the model re-learns demographic bias through the back door."
            ),
        }
        return descs.get(self.primary_source, "")


LABEL_BIAS_FIXES = [
    "Apply Reweighing (AIF360): assigns higher weights to unprivileged positive cases.",
    "Apply DisparateImpactRemover to continuous features before training.",
    "Use in-processing with Fairlearn ExponentiatedGradient + EqualizedOdds constraint.",
    "Consider re-labelling a sample of training data using human auditors.",
]

REPRESENTATION_FIXES = [
    "Collect more data from underrepresented groups before retraining.",
    "Apply SMOTE (Synthetic Minority Oversampling) for moderate imbalances.",
    "Use CTGAN (Conditional GAN) to synthesize realistic minority group samples.",
    "Oversample the minority group using Reweighing as a lightweight alternative.",
]

FEATURE_BIAS_FIXES = [
    "Remove or cap the highest-score proxy features identified by the Shadow Proxy Scanner.",
    "Apply DisparateImpactRemover (AIF360) to repair feature distributions.",
    "Use Fairlearn's ExponentiatedGradient with DemographicParity.",
    "Monitor SHAP importance of protected attribute — should be near zero after mitigation.",
]


# ── Normalised signal functions (each returns a float in [0, 1]) ─────────────

def _representation_score(
    df_clean: pd.DataFrame,
    attr: str,
    imbalance_threshold: float = 3.0,
) -> tuple[float, str]:
    """
    Gini-based group imbalance index.

    For k groups of sizes n_i, Gini-Simpson diversity = 1 - Σ(n_i/N)².
    Max Gini for k groups = (k-1)/k.  We invert and normalise:
      imbalance = 1 - diversity / max_diversity  ∈ [0, 1]
    A perfectly balanced dataset → 0. One group dominates → → 1.
    """
    if attr not in df_clean.columns:
        return 0.0, ""
    counts = df_clean[attr].value_counts()
    if len(counts) < 2:
        return 0.0, ""

    N = counts.sum()
    k = len(counts)
    gini = 1.0 - ((counts / N) ** 2).sum()
    max_gini = (k - 1) / k if k > 1 else 1.0
    imbalance = 1.0 - (gini / max_gini) if max_gini > 0 else 0.0
    imbalance = float(np.clip(imbalance, 0.0, 1.0))

    ratio = counts.max() / max(counts.min(), 1)
    if imbalance >= 0.30 or ratio >= imbalance_threshold:
        msg = (
            f"Group size imbalance (Gini-imbalance={imbalance:.2f}): "
            f"largest group is {ratio:.1f}× the smallest "
            f"({counts.idxmax()}={counts.max():,} vs {counts.idxmin()}={counts.min():,})."
        )
    else:
        msg = (
            f"Group sizes are relatively balanced (Gini-imbalance={imbalance:.2f}, ratio={ratio:.1f}×). "
            "Representation bias is unlikely as the primary cause."
        )
    return imbalance, msg


def _feature_score(
    proxy_scan_results: dict[str, list[ProxyResult]],
    high_proxy_threshold: float = 0.10,
) -> tuple[float, str]:
    """
    Mean combined_score of top-4 proxy features across all protected attrs.

    combined_score is already normalised to [0, 1] in the V2 proxy scanner.
    Score = mean of top-min(4, n_proxies) combined_scores → [0, 1].
    """
    all_proxies = [
        r for results in proxy_scan_results.values()
        for r in results
    ]
    if not all_proxies:
        return 0.0, "No proxy features found. Feature leakage is not the primary cause."

    top_proxies = sorted(all_proxies, key=lambda r: r.combined_score, reverse=True)[:4]
    score = float(np.mean([r.combined_score for r in top_proxies]))
    score = float(np.clip(score, 0.0, 1.0))

    high_proxies = [r for r in all_proxies if r.combined_score >= high_proxy_threshold]
    if high_proxies:
        top = top_proxies[0]
        msg = (
            f"{len(high_proxies)} shadow proxy feature(s) detected. "
            f"Highest: '{top.feature}' (combined_score={top.combined_score:.3f}) — "
            f"strongly correlated with '{top.protected_attr}'."
        )
    else:
        msg = "No high-score shadow proxies detected. Feature leakage is not the primary cause."

    return score, msg


def _label_score(
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    attr: str,
) -> tuple[float, str]:
    """
    Label-distribution skew normalised via log-ratio entropy.

    score = log2(max_rate / min_rate) / log2(100)   — capped at 1.0.

    This is principled: the denominator corresponds to a 100× rate ratio,
    which represents extreme historical discrimination. A 2× ratio → 0.14,
    a 4× ratio → 0.29, a 10× ratio → 0.50.
    """
    if attr not in df_clean.columns:
        return 0.0, ""
    target  = config.target_col
    pos     = config.positive_label
    col     = df_clean[attr]
    priv_val = config.privileged_values[attr]
    if not pd.api.types.is_object_dtype(col):
        try:
            priv_val = col.dtype.type(priv_val)
        except (ValueError, TypeError):
            pass

    priv_mask   = col == priv_val
    priv_rate   = (df_clean.loc[priv_mask,  target] == pos).mean()
    unpriv_rate = (df_clean.loc[~priv_mask, target] == pos).mean()

    if priv_rate > 0 and unpriv_rate > 0:
        label_ratio = max(priv_rate, unpriv_rate) / min(priv_rate, unpriv_rate)
    elif max(priv_rate, unpriv_rate) > 0:
        label_ratio = 100.0
    else:
        label_ratio = 1.0

    # log2(ratio) / log2(100) maps ratio=1 → 0, ratio=100 → 1
    score = float(np.clip(np.log2(max(label_ratio, 1.0)) / np.log2(100.0), 0.0, 1.0))

    higher_group = "privileged" if priv_rate >= unpriv_rate else "unprivileged"
    if label_ratio >= 1.5:
        msg = (
            f"Label distribution skew: {higher_group} group positive rate "
            f"({max(priv_rate, unpriv_rate):.1%}) is {label_ratio:.1f}× higher than "
            f"the other group ({min(priv_rate, unpriv_rate):.1%}). "
            "Historical discrimination likely encoded in target labels."
        )
    else:
        msg = (
            f"Label rates are similar across groups ({priv_rate:.1%} vs {unpriv_rate:.1%}). "
            "Label bias is unlikely as the primary cause."
        )
    return score, msg


def trace_bias_provenance(
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    proxy_scan_results: dict[str, list[ProxyResult]],
    primary_attr: str | None = None,
    imbalance_ratio_threshold: float = 3.0,
    high_proxy_threshold: float = 0.10,
) -> ProvenanceReport:
    """
    Diagnose the primary source of bias from three independent normalised signals.

    V2: Each signal is independently normalised to [0, 1] using principled
    formulas (Gini imbalance, log-ratio entropy, mean combined proxy score).
    Confidence = margin between top-1 and top-2 source scores.

    Parameters
    ----------
    df_clean                : DataFrame with original string protected attr columns
    proxy_scan_results      : output of scan_proxies()
    primary_attr            : which protected attr to analyse
    imbalance_ratio_threshold : ratio threshold for representation evidence message
    high_proxy_threshold    : combined_score threshold for feature evidence message
    """
    attr = primary_attr or config.primary_protected_attr()
    evidence: list[str] = []

    # ── Signal 1: Representation imbalance ────────────────────────────────────
    rep_score, rep_msg = _representation_score(df_clean, attr, imbalance_ratio_threshold)
    evidence.append(rep_msg)

    # ── Signal 2: Proxy / feature leakage ─────────────────────────────────────
    feat_score, feat_msg = _feature_score(proxy_scan_results, high_proxy_threshold)
    evidence.append(feat_msg)

    # ── Signal 3: Label distribution skew ─────────────────────────────────────
    label_score, label_msg = _label_score(df_clean, config, attr)
    evidence.append(label_msg)

    scores = {
        "label_bias":          label_score,
        "representation_bias": rep_score,
        "feature_bias":        feat_score,
    }

    # ── Pick primary and secondary ─────────────────────────────────────────────
    sorted_scores = sorted(scores.items(), key=lambda t: t[1], reverse=True)
    primary   = sorted_scores[0][0]
    secondary = sorted_scores[1][0] if sorted_scores[1][1] > 0.05 else None

    # Confidence = margin between top-1 and top-2, scaled up to 1.0
    top_score  = sorted_scores[0][1]
    runner_up  = sorted_scores[1][1]
    margin     = top_score - runner_up  # ∈ [0, 1]

    # Also factor in absolute magnitude: a score of 0.05 vs 0.01 is low confidence
    confidence = float(np.clip(0.4 * top_score + 0.6 * margin, 0.0, 1.0))

    # ── Fixes and cautions ────────────────────────────────────────────────────
    fix_map = {
        "label_bias":          LABEL_BIAS_FIXES,
        "representation_bias": REPRESENTATION_FIXES,
        "feature_bias":        FEATURE_BIAS_FIXES,
    }
    fixes = fix_map.get(primary, [])

    cautions = []
    if primary in ("label_bias", "representation_bias"):
        cautions.append(
            "Demographic Parity and Calibration are mathematically incompatible "
            "when base rates differ across groups. Optimizing one will worsen the other."
        )
    if feat_score > 0.10:
        cautions.append(
            "Shadow proxy features remain even after removing the protected attribute. "
            "Feature mitigation (DIR or proxy removal) should complement sample-weight methods."
        )

    return ProvenanceReport(
        primary_source=primary,
        secondary_source=secondary,
        confidence=confidence,
        evidence=evidence,
        recommended_fixes=fixes,
        caution_notes=cautions,
        scores=scores,
    )
