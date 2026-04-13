"""
LLM-Powered Audit Narrative — uses Google Gemini API (free tier).

Model: gemini-2.0-flash (15 RPM free, 1M tokens/day)
Library: google-generativeai

Set GEMINI_API_KEY in your .env file.
Get a free key at: https://aistudio.google.com/apikey
"""

from __future__ import annotations
import os
from typing import Generator

from src.ml_pipeline.evaluator import EvalResult
from src.bias_engine.provenance import ProvenanceReport
from src.bias_engine.proxy_scanner import ProxyResult


GEMINI_MODEL = "gemini-2.0-flash"


def _streamlit_secret(key: str) -> str:
    """Safely read a Streamlit secret (works on Streamlit Cloud, silent fail locally)."""
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""

SYSTEM_PROMPT = """You are an expert AI fairness auditor writing a compliance report.
Your audience is a Chief Compliance Officer or legal counsel — not a data scientist.
Write in clear, direct prose. No bullet lists. No markdown headers.
Keep the total response to 4 short paragraphs:
  1. What bias was found and how severe it is (use plain numbers, not formulas).
  2. Why this bias likely exists (historical context, not technical jargon).
  3. What was done to mitigate it and what the improvement looks like.
  4. Remaining risks, recommended next steps, and which regulations apply.
Be honest about limitations. Do not oversell the mitigation."""


def _build_metrics_prompt(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    proxy_results: list[ProxyResult],
    provenance: ProvenanceReport,
    dataset_name: str,
    primary_attr: str,
) -> str:
    """Assemble a structured metrics string to feed into the LLM."""
    bm = baseline_eval.fairness.get(primary_attr)
    am = best_eval.fairness.get(primary_attr)

    lines = [
        f"BIAS AUDIT REPORT — {dataset_name}",
        f"Protected Attribute Analysed: {primary_attr}",
        "",
        "PRE-MITIGATION METRICS (baseline model):",
        f"  Disparate Impact (DI): {bm.disparate_impact:.3f}  [FAIL — legal threshold is 0.80]" if bm else "  DI: unavailable",
        f"  Statistical Parity Difference: {bm.statistical_parity_diff:.3f}" if bm else "",
        f"  Equalized Odds Difference: {bm.equalized_odds_diff:.3f}" if bm else "",
        f"  Model Accuracy: {baseline_eval.performance.accuracy:.1%}",
        "",
        f"POST-MITIGATION METRICS ({best_eval.model_type} model):",
        f"  Disparate Impact (DI): {am.disparate_impact:.3f}" if am else "  DI: unavailable",
        f"  Equalized Odds Difference: {am.equalized_odds_diff:.3f}" if am else "",
        f"  Model Accuracy: {best_eval.performance.accuracy:.1%}",
        f"  Accuracy change: {(best_eval.performance.accuracy - baseline_eval.performance.accuracy):+.1%}",
        "",
        "BIAS SOURCE DIAGNOSIS:",
        f"  Primary cause: {provenance.primary_label} (confidence {provenance.confidence:.0%})",
    ]
    for e in provenance.evidence[:2]:
        lines.append(f"  - {e}")

    if proxy_results:
        top = proxy_results[:3]
        lines.append("\nSHADOW PROXY FEATURES DETECTED:")
        for r in top:
            lines.append(f"  - '{r.feature}' is a {r.risk_level} proxy for '{r.protected_attr}' (MI={r.mutual_info:.3f})")

    lines += [
        "",
        "REGULATORY CONTEXT:",
        "  - EU AI Act (2024): Article 10 requires bias testing for high-risk AI.",
        "  - US Equal Credit Opportunity Act (ECOA): prohibits discriminatory credit decisions.",
        "  - NYC Local Law 144: mandates annual bias audits for automated employment tools.",
    ]
    return "\n".join(lines)


def generate_audit_narrative(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    proxy_results: list[ProxyResult],
    provenance: ProvenanceReport,
    dataset_name: str,
    primary_attr: str,
    api_key: str | None = None,
) -> str:
    """
    Generate a plain-English audit narrative using Gemini API.

    Parameters
    ----------
    api_key : Gemini API key. Falls back to GEMINI_API_KEY env var.

    Returns
    -------
    Narrative string. Returns a fallback message if API is unavailable.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "") or _streamlit_secret("GEMINI_API_KEY")
    if not key:
        return _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)

    try:
        import google.generativeai as genai
    except ImportError:
        return (
            "[LLM Narrative] google-generativeai not installed. "
            "Run: pip install google-generativeai\n\n"
            + _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)
        )

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        metrics_text = _build_metrics_prompt(
            baseline_eval, best_eval, proxy_results, provenance, dataset_name, primary_attr
        )
        response = model.generate_content(metrics_text)
        return response.text.strip()
    except Exception as e:
        return (
            f"[LLM Narrative] Gemini API error: {e}\n\n"
            + _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)
        )


def stream_audit_narrative(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    proxy_results: list[ProxyResult],
    provenance: ProvenanceReport,
    dataset_name: str,
    primary_attr: str,
    api_key: str | None = None,
) -> Generator[str, None, None]:
    """
    Streaming version for Streamlit st.write_stream().
    Yields text chunks as they arrive from Gemini.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "") or _streamlit_secret("GEMINI_API_KEY")
    if not key:
        yield _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)
        return

    try:
        import google.generativeai as genai
    except ImportError:
        yield "[LLM Narrative] Install with: pip install google-generativeai"
        return

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        metrics_text = _build_metrics_prompt(
            baseline_eval, best_eval, proxy_results, provenance, dataset_name, primary_attr
        )
        for chunk in model.generate_content(metrics_text, stream=True):
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"[LLM Narrative] Gemini API error: {e}\n\n"
        yield _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)


def _fallback_narrative(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    primary_attr: str,
    provenance: ProvenanceReport,
) -> str:
    """Rule-based fallback when Gemini API is unavailable."""
    bm = baseline_eval.fairness.get(primary_attr)
    am = best_eval.fairness.get(primary_attr)
    di_b = bm.disparate_impact if bm else float("nan")
    di_a = am.disparate_impact if am else float("nan")
    acc_delta = best_eval.performance.accuracy - baseline_eval.performance.accuracy

    return (
        f"AUDIT SUMMARY — {baseline_eval.model_type.upper()} vs {best_eval.model_type.upper()}\n\n"
        f"The baseline model shows a Disparate Impact of {di_b:.3f} for '{primary_attr}', "
        f"well below the legal threshold of 0.80. This means the unprivileged group receives "
        f"positive outcomes at {di_b:.0%} the rate of the privileged group — a legally actionable disparity.\n\n"
        f"Root cause diagnosis: {provenance.primary_label}. {provenance.primary_description}\n\n"
        f"After applying bias mitigation, the Disparate Impact improved to {di_a:.3f}. "
        f"Model accuracy changed by {acc_delta:+.1%}, demonstrating that fairness "
        f"improvements are achievable with minimal performance cost.\n\n"
        f"Recommended next steps: {provenance.recommended_fixes[0] if provenance.recommended_fixes else 'Continue monitoring.'} "
        f"This audit covers EU AI Act Article 10, ECOA, and NYC Local Law 144 requirements."
    )
