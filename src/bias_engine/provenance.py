"""
Bias Provenance Tracer — identifies WHERE the bias originates.

Three root causes require different fixes:
  Label Bias        → historical human decisions were biased (fix: Reweighing)
  Representation    → underrepresented groups in training data (fix: more data / augmentation)
  Feature Bias      → proxy features encode protected attribute (fix: remove proxies / DIR)
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

    @property
    def primary_label(self) -> str:
        labels = {
            "label_bias":         "Label Bias",
            "representation_bias":"Representation Bias",
            "feature_bias":       "Feature / Proxy Bias",
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
    "Remove or cap the highest-MI proxy features identified by the Shadow Proxy Scanner.",
    "Apply DisparateImpactRemover (AIF360) to repair feature distributions.",
    "Use Fairlearn's ExponentiatedGradient with DemographicParity to force feature-blind decisions.",
    "Monitor the SHAP importance of the protected attribute — it should be near zero after mitigation.",
]


def trace_bias_provenance(
    df_clean: pd.DataFrame,
    config: DatasetConfig,
    proxy_scan_results: dict[str, list[ProxyResult]],
    primary_attr: str | None = None,
    imbalance_ratio_threshold: float = 3.0,
    high_proxy_threshold: float = 0.15,
) -> ProvenanceReport:
    """
    Diagnose the primary source of bias from four signals:
      1. Group size imbalance → Representation Bias
      2. High-MI proxy features → Feature Bias
      3. Label distribution skew → Label Bias
      4. Disparate Impact magnitude → Label Bias confidence boost

    Parameters
    ----------
    df_clean : DataFrame with original string protected attr columns
    proxy_scan_results : output of scan_proxies()
    primary_attr : which protected attr to analyse (default: config.primary_protected_attr())
    imbalance_ratio_threshold : ratio of largest to smallest group to flag representation bias
    high_proxy_threshold : MI threshold to flag feature bias
    """
    attr = primary_attr or config.primary_protected_attr()
    evidence = []
    scores = {"label_bias": 0.0, "representation_bias": 0.0, "feature_bias": 0.0}

    # ── Signal 1: Representation imbalance ───────────────────────────────────
    if attr in df_clean.columns:
        counts = df_clean[attr].value_counts()
        if len(counts) >= 2:
            ratio = counts.max() / max(counts.min(), 1)
            if ratio >= imbalance_ratio_threshold:
                scores["representation_bias"] += 0.6
                evidence.append(
                    f"Group size imbalance: largest group is {ratio:.1f}× larger than smallest "
                    f"({counts.idxmax()}={counts.max():,} vs {counts.idxmin()}={counts.min():,})."
                )
            else:
                evidence.append(
                    f"Group sizes are relatively balanced (ratio={ratio:.1f}×). "
                    "Representation bias is unlikely as the primary cause."
                )

    # ── Signal 2: Proxy features ─────────────────────────────────────────────
    high_mi_proxies = [
        r for results in proxy_scan_results.values()
        for r in results
        if r.mutual_info >= high_proxy_threshold
    ]
    if high_mi_proxies:
        proxy_score = 0.6 * min(len(high_mi_proxies), 4) / 4
        scores["feature_bias"] += proxy_score
        top_proxy = max(high_mi_proxies, key=lambda r: r.mutual_info)
        evidence.append(
            f"{len(high_mi_proxies)} shadow proxy feature(s) detected. "
            f"Highest: '{top_proxy.feature}' (MI={top_proxy.mutual_info:.3f}) — "
            f"strongly correlated with '{top_proxy.protected_attr}'."
        )
    else:
        evidence.append("No high-MI shadow proxies detected. Feature leakage is not the primary cause.")

    # ── Signal 3: Label distribution skew (label bias) ───────────────────────
    if attr in df_clean.columns:
        target = config.target_col
        pos    = config.positive_label
        col    = df_clean[attr]
        priv_val = config.privileged_values[attr]
        # Coerce priv_val to column dtype for numeric columns (e.g. age as int)
        if not pd.api.types.is_object_dtype(col):
            try:
                priv_val = col.dtype.type(priv_val)
            except (ValueError, TypeError):
                pass
        priv_mask   = col == priv_val
        priv_rate   = (df_clean.loc[priv_mask,  target] == pos).mean()
        unpriv_rate = (df_clean.loc[~priv_mask, target] == pos).mean()
        # Symmetric ratio — detects bias regardless of which group is labelled privileged
        if priv_rate > 0 and unpriv_rate > 0:
            label_ratio = max(priv_rate, unpriv_rate) / min(priv_rate, unpriv_rate)
        elif max(priv_rate, unpriv_rate) > 0:
            label_ratio = 10.0  # one group has zero positive rate — extreme bias
        else:
            label_ratio = 1.0
        higher_group = "privileged" if priv_rate >= unpriv_rate else "unprivileged"
        if label_ratio >= 1.5:
            # Scale: ratio 1.5 → 0.55,  ratio 2 → 0.70,  ratio 4+ → 0.90
            label_score = min(0.40 + 0.15 * np.log2(label_ratio), 0.90)
            scores["label_bias"] += label_score
            evidence.append(
                f"Label distribution skew: {higher_group} group positive rate ({max(priv_rate, unpriv_rate):.1%}) "
                f"is {label_ratio:.1f}× higher than the other group ({min(priv_rate, unpriv_rate):.1%}). "
                "Historical discrimination likely encoded in target labels."
            )
        else:
            evidence.append(
                f"Label rates are similar across groups ({priv_rate:.1%} vs {unpriv_rate:.1%}). "
                "Label bias is unlikely as the primary cause."
            )

    # ── Pick primary and secondary ────────────────────────────────────────────
    sorted_scores = sorted(scores.items(), key=lambda t: t[1], reverse=True)
    primary   = sorted_scores[0][0]
    secondary = sorted_scores[1][0] if sorted_scores[1][1] > 0.1 else None
    # Confidence: top signal + partial credit for corroborating evidence
    top_score  = sorted_scores[0][1]
    other_sum  = sum(s for _, s in sorted_scores[1:])
    confidence = min(top_score + 0.15 * other_sum, 1.0)

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
    if scores["feature_bias"] > 0.3:
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
    )
