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
    imbalance_ratio_threshold: float = 3.0,
    high_proxy_threshold: float = 0.15,
) -> ProvenanceReport:
    """
    Diagnose the primary source of bias from three signals:
      1. Group size imbalance → Representation Bias
      2. High-MI proxy features → Feature Bias
      3. Default (label distribution skew) → Label Bias

    Parameters
    ----------
    df_clean : DataFrame with original string protected attr columns
    proxy_scan_results : output of scan_proxies()
    imbalance_ratio_threshold : ratio of largest to smallest group to flag representation bias
    high_proxy_threshold : MI threshold to flag feature bias
    """
    attr = config.primary_protected_attr()
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
        scores["feature_bias"] += 0.5 * min(len(high_mi_proxies), 4) / 4
        top_proxy = max(high_mi_proxies, key=lambda r: r.mutual_info)
        evidence.append(
            f"{len(high_mi_proxies)} shadow proxy feature(s) detected. "
            f"Highest: '{top_proxy.feature}' (MI={top_proxy.mutual_info:.3f}) — "
            f"strongly correlated with '{top_proxy.protected_attr}'."
        )
    else:
        evidence.append("No high-MI shadow proxies detected. Feature leakage is not the primary cause.")

    # ── Signal 3: Label distribution skew (label bias default) ───────────────
    if attr in df_clean.columns:
        target = config.target_col
        pos    = config.positive_label
        priv_rate   = (df_clean[df_clean[attr] == config.privileged_values[attr]][target] == pos).mean()
        unpriv_rate = (df_clean[df_clean[attr] != config.privileged_values[attr]][target] == pos).mean()
        label_ratio = priv_rate / max(unpriv_rate, 1e-9)
        if label_ratio > 2.0:
            scores["label_bias"] += 0.7
            evidence.append(
                f"Label distribution skew: privileged group positive rate ({priv_rate:.1%}) "
                f"is {label_ratio:.1f}× higher than unprivileged ({unpriv_rate:.1%}). "
                "Historical discrimination likely encoded in target labels."
            )

    # ── Pick primary and secondary ────────────────────────────────────────────
    sorted_scores = sorted(scores.items(), key=lambda t: t[1], reverse=True)
    primary   = sorted_scores[0][0]
    secondary = sorted_scores[1][0] if sorted_scores[1][1] > 0.1 else None
    confidence = min(sorted_scores[0][1], 1.0)

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
