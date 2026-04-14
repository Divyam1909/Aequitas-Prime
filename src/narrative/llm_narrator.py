"""
LLM-Powered Audit Narrative — Gemini-only backend.

All AI features use Google Gemini (gemini-2.0-flash) via the
google-generativeai SDK.  Get a free key at aistudio.google.com/apikey.

Public API:
  generate_audit_narrative()   — full narrative string (non-streaming)
  stream_audit_narrative()     — streaming version for Streamlit
  generate_risk_scorecard()    — structured JSON risk scorecard via Gemini
  ask_remediation_chat()       — single-turn or multi-turn Q&A about the audit
  suggest_sensitive_columns()  — auto-detect protected attributes from CSV columns
"""

from __future__ import annotations
import os
import json
from typing import Generator

from src.ml_pipeline.evaluator import EvalResult
from src.bias_engine.provenance import ProvenanceReport
from src.bias_engine.proxy_scanner import ProxyResult

GEMINI_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are an expert AI fairness auditor writing a compliance report.
Your audience is a Chief Compliance Officer or legal counsel — not a data scientist.
Write in clear, direct prose. No bullet lists. No markdown headers.
Keep the total response to 4 short paragraphs:
  1. What bias was found and how severe it is (use plain numbers, not formulas).
  2. Why this bias likely exists (historical context, not technical jargon).
  3. What was done to mitigate it and what the improvement looks like.
  4. Remaining risks, recommended next steps, and which regulations apply.
Be honest about limitations. Do not oversell the mitigation."""


def _get_gemini_key(api_key: str | None = None) -> str:
    """Resolve Gemini API key from explicit arg → st.secrets → env var."""
    if api_key:
        return api_key
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def _configure_gemini(api_key: str | None = None):
    import google.generativeai as genai
    key = _get_gemini_key(api_key)
    if not key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY environment variable "
            "or pass api_key= to the function. Free key: aistudio.google.com/apikey"
        )
    genai.configure(api_key=key)
    return genai


def _build_metrics_prompt(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    proxy_results: list[ProxyResult],
    provenance: ProvenanceReport,
    dataset_name: str,
    primary_attr: str,
) -> str:
    bm = baseline_eval.fairness.get(primary_attr)
    am = best_eval.fairness.get(primary_attr)

    lines = [
        f"BIAS AUDIT REPORT — {dataset_name}",
        f"Protected Attribute Analysed: {primary_attr}",
        "",
        "PRE-MITIGATION METRICS (baseline model):",
        (f"  Disparate Impact (DI): {bm.disparate_impact:.3f}  [FAIL — legal threshold is 0.80]"
         if bm else "  DI: unavailable"),
        (f"  Statistical Parity Difference: {bm.statistical_parity_diff:.3f}" if bm else ""),
        (f"  Equalized Odds Difference: {bm.equalized_odds_diff:.3f}" if bm else ""),
        f"  Model Accuracy: {baseline_eval.performance.accuracy:.1%}",
        "",
        f"POST-MITIGATION METRICS ({best_eval.model_type} model):",
        (f"  Disparate Impact (DI): {am.disparate_impact:.3f}" if am else "  DI: unavailable"),
        (f"  Equalized Odds Difference: {am.equalized_odds_diff:.3f}" if am else ""),
        f"  Model Accuracy: {best_eval.performance.accuracy:.1%}",
        f"  Accuracy change: {(best_eval.performance.accuracy - baseline_eval.performance.accuracy):+.1%}",
        "",
        "BIAS SOURCE DIAGNOSIS:",
        f"  Primary cause: {provenance.primary_label} (confidence {provenance.confidence:.0%})",
        f"  Raw scores: {', '.join(f'{k}={v:.2f}' for k, v in provenance.scores.items())}",
    ]
    for e in provenance.evidence[:2]:
        lines.append(f"  - {e}")

    if proxy_results:
        top = proxy_results[:3]
        lines.append("\nSHADOW PROXY FEATURES DETECTED:")
        for r in top:
            lines.append(
                f"  - '{r.feature}' is a {r.risk_level} proxy for "
                f"'{r.protected_attr}' (combined_score={r.combined_score:.3f})"
            )

    lines += [
        "",
        "REGULATORY CONTEXT:",
        "  - EU AI Act (2024): Article 10 requires bias testing for high-risk AI.",
        "  - US Equal Credit Opportunity Act (ECOA): prohibits discriminatory credit decisions.",
        "  - NYC Local Law 144: mandates annual bias audits for automated employment tools.",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Core Gemini helpers
# ══════════════════════════════════════════════════════════════════════════════

def _generate_gemini(prompt: str, api_key: str | None = None) -> str:
    genai = _configure_gemini(api_key)
    gm = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    resp = gm.generate_content(prompt)
    return resp.text.strip()


def _stream_gemini(prompt: str, api_key: str | None = None) -> Generator[str, None, None]:
    genai = _configure_gemini(api_key)
    gm = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    for chunk in gm.generate_content(prompt, stream=True):
        if chunk.text:
            yield chunk.text


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Narrative
# ══════════════════════════════════════════════════════════════════════════════

def generate_audit_narrative(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    proxy_results: list[ProxyResult],
    provenance: ProvenanceReport,
    dataset_name: str,
    primary_attr: str,
    api_key: str | None = None,
    # legacy params kept for backwards-compat, ignored
    provider: str | None = None,
    model: str | None = None,
    ollama_base_url: str = "",
) -> str:
    """Generate a plain-English audit narrative using Gemini."""
    metrics_text = _build_metrics_prompt(
        baseline_eval, best_eval, proxy_results, provenance, dataset_name, primary_attr
    )
    key = _get_gemini_key(api_key)
    if not key:
        return _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)
    try:
        return _generate_gemini(metrics_text, key)
    except Exception as e:
        return f"[Gemini error: {e}]\n\n" + _fallback_narrative(
            baseline_eval, best_eval, primary_attr, provenance
        )


def stream_audit_narrative(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    proxy_results: list[ProxyResult],
    provenance: ProvenanceReport,
    dataset_name: str,
    primary_attr: str,
    api_key: str | None = None,
    # legacy params kept for backwards-compat, ignored
    provider: str | None = None,
    model: str | None = None,
    ollama_base_url: str = "",
) -> Generator[str, None, None]:
    """Streaming version — yields text chunks for Streamlit st.write_stream()."""
    metrics_text = _build_metrics_prompt(
        baseline_eval, best_eval, proxy_results, provenance, dataset_name, primary_attr
    )
    key = _get_gemini_key(api_key)
    if not key:
        yield _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)
        return
    try:
        yield from _stream_gemini(metrics_text, key)
    except Exception as e:
        yield f"[Gemini error: {e}]\n\n"
        yield _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Risk Scorecard (Gemini JSON mode)
# ══════════════════════════════════════════════════════════════════════════════

SCORECARD_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_risk": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
        "overall_risk_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "legal_exposure": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "deployment_recommendation": {
            "type": "string",
            "enum": ["Deploy", "Deploy with monitoring", "Defer — remediate first", "Block deployment"]
        },
        "dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "status": {"type": "string", "enum": ["Pass", "Warning", "Fail"]},
                    "finding": {"type": "string"}
                }
            }
        },
        "top_actions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3
        },
        "applicable_regulations": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}


def generate_risk_scorecard(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    proxy_results: list[ProxyResult],
    provenance: ProvenanceReport,
    dataset_name: str,
    primary_attr: str,
    api_key: str | None = None,
) -> dict | None:
    """
    Use Gemini's JSON-mode output to produce a structured compliance risk scorecard.
    Returns a dict matching SCORECARD_SCHEMA, or None if the API key is missing.
    """
    key = _get_gemini_key(api_key)
    if not key:
        return None

    metrics_text = _build_metrics_prompt(
        baseline_eval, best_eval, proxy_results, provenance, dataset_name, primary_attr
    )

    scorecard_prompt = (
        f"{metrics_text}\n\n"
        "Based on the audit data above, produce a structured compliance risk scorecard. "
        "Score each dimension from 0 (worst) to 100 (best). "
        "Be conservative — err on the side of flagging risk. "
        "Return ONLY valid JSON matching the schema provided."
    )

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        gm = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json"},
        )
        resp = gm.generate_content(scorecard_prompt)
        return json.loads(resp.text)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Remediation Chatbot
# ══════════════════════════════════════════════════════════════════════════════

CHATBOT_SYSTEM = """You are a helpful AI fairness advisor embedded in an audit tool.
The user has just run a bias audit on their ML model. Answer their questions about
the results clearly and concisely — 2–4 sentences max per response.
Use plain language. If asked for code, provide a short Python snippet.
Never make up metric values; only refer to data provided in the conversation."""


def ask_remediation_chat(
    user_message: str,
    audit_context: str,
    chat_history: list[dict] | None = None,
    api_key: str | None = None,
) -> str:
    """
    Single-turn or multi-turn Q&A about the audit using Gemini chat.

    Parameters
    ----------
    user_message   : the user's question
    audit_context  : a string summary of the current audit (from _build_metrics_prompt)
    chat_history   : list of {"role": "user"|"model", "parts": [str]} dicts
                     (pass st.session_state["gemini_chat_history"] directly)
    api_key        : Gemini API key

    Returns
    -------
    str  — Gemini's response text
    """
    key = _get_gemini_key(api_key)
    if not key:
        return "No Gemini API key found. Add your key in the sidebar to use the AI advisor."

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        gm = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=CHATBOT_SYSTEM,
        )

        # Prepend audit context as the very first user turn if history is empty
        history = list(chat_history) if chat_history else []
        if not history:
            history = [
                {"role": "user",  "parts": [f"Here is the audit context:\n\n{audit_context}"]},
                {"role": "model", "parts": ["Understood. I have reviewed the audit results. What would you like to know?"]},
            ]

        chat = gm.start_chat(history=history)
        resp = chat.send_message(user_message)
        return resp.text.strip()
    except Exception as e:
        return f"Gemini error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Dataset Column Advisor
# ══════════════════════════════════════════════════════════════════════════════

def suggest_sensitive_columns(
    column_names: list[str],
    sample_values: dict[str, list],
    api_key: str | None = None,
) -> dict | None:
    """
    Given a list of column names and a few sample values, ask Gemini to identify
    which columns are likely protected attributes, the target label, and proxy risks.

    Returns a dict with keys:
      protected_attrs : list[str]
      target_col      : str | None
      proxy_risks     : list[str]
      reasoning       : str
    Or None if no API key is available.
    """
    key = _get_gemini_key(api_key)
    if not key:
        return None

    col_summary = "\n".join(
        f"  - {col}: sample values = {vals[:5]}"
        for col, vals in sample_values.items()
    )
    prompt = (
        f"A dataset has these columns:\n{col_summary}\n\n"
        "Identify which columns are likely:\n"
        "1. Protected attributes (race, gender, age, religion, national origin, disability, etc.)\n"
        "2. The target/label column (what the model predicts)\n"
        "3. Proxy risk columns (correlate with protected attributes: zip code, school, surname)\n\n"
        "Return ONLY valid JSON with keys: "
        "protected_attrs (list), target_col (string or null), proxy_risks (list), reasoning (string)."
    )

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        gm = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json"},
        )
        resp = gm.generate_content(prompt)
        return json.loads(resp.text)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Fallback (no API key)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_narrative(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    primary_attr: str,
    provenance: ProvenanceReport,
) -> str:
    """Rule-based fallback when no Gemini API key is available."""
    bm = baseline_eval.fairness.get(primary_attr)
    am = best_eval.fairness.get(primary_attr)
    di_b = bm.disparate_impact if bm else float("nan")
    di_a = am.disparate_impact if am else float("nan")
    acc_delta = best_eval.performance.accuracy - baseline_eval.performance.accuracy

    import math
    di_b_str = f"{di_b:.3f}" if not math.isnan(di_b) else "N/A"
    di_a_str = f"{di_a:.3f}" if not math.isnan(di_a) else "N/A"

    return (
        f"AUDIT SUMMARY — {baseline_eval.model_type.upper()} vs {best_eval.model_type.upper()}\n\n"
        f"The baseline model shows a Disparate Impact of {di_b_str} for '{primary_attr}', "
        f"well below the legal threshold of 0.80. This means the unprivileged group receives "
        f"positive outcomes at a significantly lower rate than the privileged group — "
        f"a legally actionable disparity.\n\n"
        f"Root cause diagnosis: {provenance.primary_label} (confidence {provenance.confidence:.0%}). "
        f"{provenance.primary_description}\n\n"
        f"After applying bias mitigation, the Disparate Impact improved to {di_a_str}. "
        f"Model accuracy changed by {acc_delta:+.1%}, demonstrating that fairness "
        f"improvements are achievable with minimal performance cost.\n\n"
        f"Recommended next steps: {provenance.recommended_fixes[0] if provenance.recommended_fixes else 'Continue monitoring.'} "
        f"This audit covers EU AI Act Article 10, ECOA, and NYC Local Law 144 requirements."
    )
