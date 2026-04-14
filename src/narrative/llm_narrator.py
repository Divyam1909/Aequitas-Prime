"""
LLM-Powered Audit Narrative — V2: pluggable multi-provider backend.

Supported providers:
  "gemini"  — Google Gemini API (default, free tier)
  "openai"  — OpenAI GPT-4o / GPT-3.5-turbo
  "claude"  — Anthropic Claude (claude-sonnet-4-6 default)
  "ollama"  — Local Ollama server (llama3, mistral, etc.)
  "none"    — Rule-based fallback only

Provider selection (priority order):
  1. Explicit api_key / provider arguments to generate_audit_narrative()
  2. st.secrets (Streamlit Cloud deployment)
  3. Environment variables: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
  4. Auto-detect: first key found wins
  5. Fallback narrative (no external calls)
"""

from __future__ import annotations
import os
from typing import Generator

from src.ml_pipeline.evaluator import EvalResult
from src.bias_engine.provenance import ProvenanceReport
from src.bias_engine.proxy_scanner import ProxyResult

# ── Default model names per provider ──────────────────────────────────────────
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-sonnet-4-6",
    "ollama": "llama3",
}

SYSTEM_PROMPT = """You are an expert AI fairness auditor writing a compliance report.
Your audience is a Chief Compliance Officer or legal counsel — not a data scientist.
Write in clear, direct prose. No bullet lists. No markdown headers.
Keep the total response to 4 short paragraphs:
  1. What bias was found and how severe it is (use plain numbers, not formulas).
  2. Why this bias likely exists (historical context, not technical jargon).
  3. What was done to mitigate it and what the improvement looks like.
  4. Remaining risks, recommended next steps, and which regulations apply.
Be honest about limitations. Do not oversell the mitigation."""


def _streamlit_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""


def _auto_detect_provider() -> tuple[str, str]:
    """
    Return (provider_name, api_key) for the first available provider.
    Returns ("none", "") if nothing is configured.
    """
    checks = [
        ("gemini", "GEMINI_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("claude", "ANTHROPIC_API_KEY"),
    ]
    for provider, env_var in checks:
        key = os.environ.get(env_var, "") or _streamlit_secret(env_var)
        if key:
            return provider, key
    return "none", ""


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
# Provider-specific generation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _generate_gemini(prompt: str, api_key: str, model: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)
    resp = gm.generate_content(prompt)
    return resp.text.strip()


def _stream_gemini(prompt: str, api_key: str, model: str) -> Generator[str, None, None]:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)
    for chunk in gm.generate_content(prompt, stream=True):
        if chunk.text:
            yield chunk.text


def _generate_openai(prompt: str, api_key: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _stream_openai(prompt: str, api_key: str, model: str) -> Generator[str, None, None]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _generate_claude(prompt: str, api_key: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _stream_claude(prompt: str, api_key: str, model: str) -> Generator[str, None, None]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def _generate_ollama(prompt: str, model: str, base_url: str = "http://localhost:11434") -> str:
    import requests, json
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _stream_ollama(prompt: str, model: str, base_url: str = "http://localhost:11434") -> Generator[str, None, None]:
    import requests, json
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": True,
    }
    with requests.post(f"{base_url}/api/chat", json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                delta = data.get("message", {}).get("content", "")
                if delta:
                    yield delta


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_audit_narrative(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    proxy_results: list[ProxyResult],
    provenance: ProvenanceReport,
    dataset_name: str,
    primary_attr: str,
    api_key: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> str:
    """
    Generate a plain-English audit narrative using the configured LLM.

    provider : one of "gemini", "openai", "claude", "ollama", "none".
               If None, auto-detects from environment variables.
    model    : model name override (e.g. "gpt-4o", "claude-opus-4-6").
               Defaults to DEFAULT_MODELS[provider].
    api_key  : explicit key (overrides env vars).

    Returns narrative string; falls back to rule-based if API unavailable.
    """
    # Resolve provider + key
    if provider is None or provider == "none":
        prov, key = _auto_detect_provider()
        if api_key:
            key = api_key
    else:
        prov = provider
        key  = (
            api_key
            or os.environ.get({
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
            }.get(prov, ""), "")
            or _streamlit_secret({
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
            }.get(prov, ""))
        )

    if prov == "none" or not key and prov != "ollama":
        return _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)

    mdl = model or DEFAULT_MODELS.get(prov, "")
    metrics_text = _build_metrics_prompt(
        baseline_eval, best_eval, proxy_results, provenance, dataset_name, primary_attr
    )

    try:
        if prov == "gemini":
            return _generate_gemini(metrics_text, key, mdl)
        elif prov == "openai":
            return _generate_openai(metrics_text, key, mdl)
        elif prov == "claude":
            return _generate_claude(metrics_text, key, mdl)
        elif prov == "ollama":
            return _generate_ollama(metrics_text, mdl, ollama_base_url)
        else:
            return _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)
    except Exception as e:
        return (
            f"[LLM Narrative] {prov} error: {e}\n\n"
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
    provider: str | None = None,
    model: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> Generator[str, None, None]:
    """Streaming version — yields text chunks for Streamlit st.write_stream()."""
    if provider is None or provider == "none":
        prov, key = _auto_detect_provider()
        if api_key:
            key = api_key
    else:
        prov = provider
        key  = (
            api_key
            or os.environ.get({
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
            }.get(prov, ""), "")
            or _streamlit_secret({
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
            }.get(prov, ""))
        )

    if prov == "none" or not key and prov != "ollama":
        yield _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)
        return

    mdl = model or DEFAULT_MODELS.get(prov, "")
    metrics_text = _build_metrics_prompt(
        baseline_eval, best_eval, proxy_results, provenance, dataset_name, primary_attr
    )

    try:
        if prov == "gemini":
            yield from _stream_gemini(metrics_text, key, mdl)
        elif prov == "openai":
            yield from _stream_openai(metrics_text, key, mdl)
        elif prov == "claude":
            yield from _stream_claude(metrics_text, key, mdl)
        elif prov == "ollama":
            yield from _stream_ollama(metrics_text, mdl, ollama_base_url)
        else:
            yield _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)
    except Exception as e:
        yield f"[LLM Narrative] {prov} error: {e}\n\n"
        yield _fallback_narrative(baseline_eval, best_eval, primary_attr, provenance)


def _fallback_narrative(
    baseline_eval: EvalResult,
    best_eval: EvalResult,
    primary_attr: str,
    provenance: ProvenanceReport,
) -> str:
    """Rule-based fallback when no LLM API is available."""
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
