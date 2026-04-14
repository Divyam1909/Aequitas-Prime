"""
Aequitas Prime — Streamlit Dashboard  V2
Tabs: Bias X-Ray | BYOM | Surgeon | Intersectional | Causal | CF Panel | Ghost | Report
V2 additions:
  - Wizard mode: linear step-by-step guide
  - Bootstrap CI on 6-metric table
  - Combined-score proxy scanner
  - Intersectional small-group Wilson CI
  - Group SHAP comparison in Surgeon
  - Causal Fairness tab (mediation)
  - Google Gemini AI: narrative, risk scorecard, remediation chatbot, column advisor
  - Audit persistence: Save + History
  - String labels in Counterfactual Panel
"""

import os
import io
import sys
import base64
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aequitas Prime",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root { --primary: #7c3aed; --danger: #dc2626; --warn: #d97706; --ok: #16a34a; }
  .stApp { background: #0d1117 !important; color: #e6edf3; }
  [data-testid="stHeader"] {
    background: #0d1117 !important;
    border-bottom: 1px solid #21262d !important;
  }
  #MainMenu  { visibility: hidden !important; }
  footer     { visibility: hidden !important; }
  .block-container {
    padding-left:  2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1180px;
  }
  [data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: 0.8rem; }
  [data-testid="stSidebar"] .stMarkdown { font-size: 0.85rem; }
  .stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-bottom: 2px solid #21262d !important;
    gap: 3px !important;
    padding: 0 !important;
    display: flex !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-bottom: none !important;
    border-radius: 7px 7px 0 0 !important;
    padding: 10px 6px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    white-space: nowrap !important;
    flex: 1 1 0 !important;
    text-align: center !important;
    transition: background 0.15s, color 0.15s !important;
    min-width: 0 !important;
    max-width: none !important;
    justify-content: center !important;
  }
  .stTabs [data-baseweb="tab"] p {
    font-size: 0.78rem !important;
    text-align: center !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    background: #1e2433 !important;
    color: #e6edf3 !important;
  }
  .stTabs [aria-selected="true"] {
    background: #7c3aed !important;
    color: #ffffff !important;
    border-color: #7c3aed !important;
    box-shadow: 0 2px 10px rgba(124,58,237,0.4) !important;
  }
  .stTabs [aria-selected="true"] p { color: #ffffff !important; }
  .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
  .stTabs [data-baseweb="tab-border"]    { display: none !important; }
  [data-testid="stMetric"] {
    background: #161b22;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    border: 1px solid #30363d;
    transition: border-color 0.2s;
  }
  [data-testid="stMetric"]:hover { border-color: #7c3aed; }
  [data-testid="stMetricLabel"]   { font-size: 0.78rem !important; color: #8b949e !important; }
  [data-testid="stMetricValue"]   { font-size: 1.55rem !important; font-weight: 700; }
  .glow-btn > div > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white; border: none; border-radius: 10px;
    font-size: 1rem; padding: 0.65rem 2rem; font-weight: 600;
    box-shadow: 0 0 18px rgba(124,58,237,0.45);
    transition: all 0.25s; width: 100%;
  }
  .glow-btn > div > button:hover {
    box-shadow: 0 0 36px rgba(168,85,247,0.85);
    transform: translateY(-1px);
  }
  .badge-critical { background:#dc2626; color:#fff; padding:3px 11px; border-radius:20px; font-weight:700; font-size:0.76rem; }
  .badge-warning  { background:#d97706; color:#fff; padding:3px 11px; border-radius:20px; font-weight:700; font-size:0.76rem; }
  .badge-clear    { background:#16a34a; color:#fff; padding:3px 11px; border-radius:20px; font-weight:700; font-size:0.76rem; }
  .badge-high     { background:#ea580c; color:#fff; padding:3px 11px; border-radius:20px; font-weight:700; font-size:0.76rem; }
  .hint-box {
    background: #161b2e;
    border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    font-size: 0.84rem;
    color: #c9d1d9;
    margin: 0.4rem 0 0.9rem 0;
    line-height: 1.55;
  }
  .hint-box b { color: #a78bfa; }
  .alert-box-critical {
    background: #1e0a0a; border: 1px solid #dc2626;
    border-left: 4px solid #dc2626; border-radius: 8px;
    padding: 0.75rem 1rem; font-size: 0.88rem; color: #fca5a5;
    margin: 0.4rem 0; line-height: 1.6;
  }
  .alert-box-critical b { color: #f87171; }
  .alert-box-critical .alert-title { font-size: 1rem; font-weight: 700; color: #ef4444; margin-bottom: 0.3rem; }
  .alert-box-warn {
    background: #1e1200; border: 1px solid #d97706;
    border-left: 4px solid #d97706; border-radius: 8px;
    padding: 0.75rem 1rem; font-size: 0.88rem; color: #fcd34d;
    margin: 0.4rem 0; line-height: 1.6;
  }
  .alert-box-ok {
    background: #0a1e0f; border: 1px solid #16a34a;
    border-left: 4px solid #16a34a; border-radius: 8px;
    padding: 0.75rem 1rem; font-size: 0.88rem; color: #86efac;
    margin: 0.4rem 0; line-height: 1.6;
  }
  .section-header {
    font-size: 1.1rem; font-weight: 700; color: #e6edf3;
    padding: 0.4rem 0; border-bottom: 1px solid #21262d;
    margin: 1.1rem 0 0.5rem 0;
  }
  hr { border-color: #21262d !important; margin: 1rem 0 !important; }
  [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
  [data-testid="stExpander"] {
    background: #161b22; border: 1px solid #30363d !important;
    border-radius: 8px; overflow: hidden;
  }
  [data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def badge(severity: str) -> str:
    cls = {"CRITICAL":"badge-critical","WARNING":"badge-warning",
           "CLEAR":"badge-clear","OK":"badge-clear","HIGH":"badge-high"}.get(severity,"badge-warning")
    return f'<span class="{cls}">{severity}</span>'

def hint(text: str):
    st.markdown(f'<div class="hint-box">{text}</div>', unsafe_allow_html=True)

def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def di_gauge(di_value: float, title: str = "Disparate Impact") -> go.Figure:
    color = "#dc2626" if di_value < 0.80 else ("#d97706" if di_value < 0.90 else "#16a34a")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=di_value,
        delta={"reference": 0.80, "valueformat": ".3f",
               "increasing": {"color": "#16a34a"}, "decreasing": {"color": "#dc2626"}},
        number={"valueformat": ".3f", "font": {"color": color, "size": 44}},
        title={"text": title, "font": {"color": "#8b949e", "size": 13}},
        gauge={
            "axis": {"range": [0, 1.25], "tickcolor": "#30363d", "tickwidth": 1,
                     "tickfont": {"color": "#6b7280", "size": 10}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#161b22", "borderwidth": 0,
            "steps": [
                {"range": [0,    0.80], "color": "#2d1515"},
                {"range": [0.80, 0.90], "color": "#2d2410"},
                {"range": [0.90, 1.25], "color": "#0f2d1a"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 2},
                          "thickness": 0.75, "value": 0.80},
        }
    ))
    fig.update_layout(paper_bgcolor="#0d1117", height=240, margin=dict(t=20,b=0,l=10,r=10))
    return fig

DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font_color="#e6edf3", margin=dict(t=45, b=35, l=20, r=20),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)

METRIC_HELP = {
    "Disparate Impact":        ("≥ 0.80",  "Ratio of positive outcome rates: unprivileged ÷ privileged. The US/EU legal 4/5ths rule. 1.0 = perfectly equal. Below 0.80 is legally actionable discrimination."),
    "Statistical Parity Diff": ("≥ −0.10", "Raw percentage-point gap in selection rates between groups. 0.0 means both groups are selected at the same rate. Negative = unprivileged group is selected less."),
    "Equalized Odds Diff":     ("≤ 0.10",  "Maximum of the True Positive Rate gap and False Positive Rate gap between groups. Ensures the model is equally accurate for both groups — not just overall."),
    "Equal Opportunity Diff":  ("≤ 0.10",  "Gap in True Positive Rate (recall) between groups. Measures whether qualified people in both groups are correctly approved at the same rate."),
    "Predictive Parity":       ("≤ 0.10",  "Gap in Precision (PPV) between groups. When the model says yes, is it equally reliable for both groups? High gap = model's approvals mean different things for different demographics."),
    "FNR Parity":              ("≤ 0.10",  "Gap in False Negative Rate between groups. How often are qualified people wrongly rejected? A high gap means one group faces more unjust denials."),
}


def show_six_metric_table(bm, attr_name: str, show_ci: bool = False, ci_obj: "dict | None" = None):
    """
    Render the 6-metric PASS/FAIL table for one MetricsResult object.
    V2: show_ci=True adds 95% confidence interval columns when available.
    ci_obj: optional dict of {metric_field: ConfidenceInterval} from compute_bootstrap_ci.
    """
    if bm is None:
        st.warning("Metrics not available.")
        return
    rows = []
    metric_ci_map = {
        "Disparate Impact":        "ci_disparate_impact",
        "Statistical Parity Diff": "ci_statistical_parity_diff",
        "Equalized Odds Diff":     "ci_equalized_odds_diff",
        "Equal Opportunity Diff":  "ci_equal_opportunity_diff",
        "Predictive Parity":       "ci_predictive_parity_diff",
        "FNR Parity":              "ci_fnr_parity_diff",
    }
    metrics_vals = [
        ("Disparate Impact",        bm.disparate_impact,        lambda v: v >= 0.80),
        ("Statistical Parity Diff", bm.statistical_parity_diff, lambda v: v >= -0.10),
        ("Equalized Odds Diff",     bm.equalized_odds_diff,     lambda v: v <= 0.10),
        ("Equal Opportunity Diff",  bm.equal_opportunity_diff,  lambda v: v <= 0.10),
        ("Predictive Parity",       bm.predictive_parity_diff,  lambda v: v <= 0.10),
        ("FNR Parity",              bm.fnr_parity_diff,         lambda v: v <= 0.10),
    ]
    for name, val, passes in metrics_vals:
        threshold, meaning = METRIC_HELP[name]
        is_nan = isinstance(val, float) and np.isnan(val)
        status = "— N/A" if is_nan else ("✅ PASS" if passes(val) else "❌ FAIL")
        row = {
            "Metric":           name,
            "Value":            "—" if is_nan else f"{val:.3f}",
            "Pass Threshold":   threshold,
            "Status":           status,
            "Plain-English Meaning": meaning,
        }
        if show_ci:
            ci_field = metric_ci_map.get(name)
            ci = getattr(bm, ci_field, None) if ci_field and hasattr(bm, ci_field) else None
            if ci is not None and not (isinstance(ci.lower, float) and np.isnan(ci.lower)):
                row["95% CI"] = f"[{ci.lower:.3f}, {ci.upper:.3f}]"
            else:
                row["95% CI"] = "—"
        rows.append(row)

    col_cfg = {
        "Status":                st.column_config.TextColumn(width="small"),
        "Plain-English Meaning": st.column_config.TextColumn(width="large"),
    }
    if show_ci:
        col_cfg["95% CI"] = st.column_config.TextColumn(width="medium")

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, column_config=col_cfg)
    fail_count = sum(1 for r in rows if "FAIL" in r["Status"])
    if fail_count == 0:
        st.success(f"All metrics PASS for **{attr_name}**. No statistically significant bias detected.")
    elif fail_count <= 2:
        st.warning(f"**{fail_count} metric(s) FAIL** for {attr_name} — WARNING-level bias. Mitigation recommended.")
    else:
        st.error(f"**{fail_count} metric(s) FAIL** for {attr_name} — CRITICAL bias. Immediate mitigation required.")


def build_config_ui(df: pd.DataFrame, is_demo: bool = False):
    """
    Render the dataset configuration form.
    Returns a DatasetConfig if all required fields are filled, else None.
    """
    from src.utils.schema import DatasetConfig
    from src.utils.generic_preprocessor import auto_detect_protected_attrs, auto_detect_target

    section("⚙️ Configure Your Audit")
    hint(
        "Aequitas Prime has <b>auto-detected</b> candidate fields below — verify and adjust as needed. "
        "Select which columns represent protected demographics, which column is the outcome label, "
        "and what the positive outcome looks like."
    )

    cols = df.columns.tolist()
    default_name = "UCI Adult Income" if is_demo else "My Dataset"
    ds_name = st.text_input("Dataset Name", value=default_name, key="cfg_ds_name",
                             help="A friendly name shown in reports and the PDF audit.")

    # ── Target column ──────────────────────────────────────────────────────────
    auto_target = "income" if is_demo else (auto_detect_target(df) or cols[-1])
    auto_target_idx = cols.index(auto_target) if auto_target in cols else len(cols) - 1
    target_col = st.selectbox(
        "Target Column (what to predict)", options=cols, index=auto_target_idx, key="cfg_target",
        help="The column your model is predicting. Must have exactly two distinct values (binary classification)."
    )

    # ── Positive outcome value ─────────────────────────────────────────────────
    if target_col in df.columns:
        target_vals = sorted(df[target_col].dropna().astype(str).unique().tolist())

        def _pick_default_positive(vals: list[str]) -> str:
            # 1. Prefer values starting with ">" (income thresholds: >50K, >100K)
            for v in vals:
                if str(v).startswith(">"):
                    return v
            # 2. Prefer clearly positive/favourable labels
            positive_hints = {
                "yes", "1", "true", "approved", "high", "positive",
                "good", "hired", "accept", "accepted", "pass", "passed",
                "grant", "granted", "non-default", "non_default",
                "no churn", "no_churn", "retain", "retained",
            }
            for v in vals:
                if str(v).lower() in positive_hints:
                    return v
            # 3. Prefer values containing "not" or "non" (non-fraud, not-default)
            for v in vals:
                vl = str(v).lower()
                if vl.startswith("not") or vl.startswith("non"):
                    return v
            # 4. For two-value columns, prefer the minority class (often the positive/event class)
            if len(vals) == 2:
                raw_counts = df[target_col].astype(str).value_counts()
                minority = raw_counts.idxmin()
                if minority in vals:
                    return minority
            # 5. Fallback: last alphabetically
            return vals[-1]

        default_pos = _pick_default_positive(target_vals)
        default_pos_idx = target_vals.index(default_pos) if default_pos in target_vals else 0
        positive_label = st.selectbox(
            "Positive Outcome Value", options=target_vals, index=default_pos_idx, key="cfg_pos",
            help=(
                "Which value of the target column represents a *good* outcome — "
                "e.g. loan approved, hired, income above threshold, not default? "
                "Fairness metrics measure whether both groups receive this outcome equally."
            )
        )
    else:
        st.error("Select a valid target column.")
        return None

    # ── Protected attributes ───────────────────────────────────────────────────
    non_target = [c for c in cols if c != target_col]
    auto_protected = ["sex", "race"] if is_demo else auto_detect_protected_attrs(df, target_col)
    # Only keep auto-detected that actually exist
    auto_protected = [a for a in auto_protected if a in non_target]

    st.markdown("**Protected Attributes** — demographic columns to audit for bias")
    hint(
        "These are columns representing characteristics that are <b>legally protected</b> from discrimination "
        "(sex, race, age, religion, nationality, disability, etc.). "
        "Aequitas Prime will measure whether outcomes are equally distributed across groups in each column."
    )
    selected_attrs = []
    for attr in auto_protected:
        checked = st.checkbox(
            f"✓ {attr}", value=True, key=f"cfg_attr_{attr}",
            help=f"Auto-detected as a protected attribute (low cardinality or name-matched). Uncheck to exclude."
        )
        if checked:
            selected_attrs.append(attr)

    remaining = [c for c in non_target if c not in auto_protected]
    if remaining:
        extras = st.multiselect(
            "Add more protected attributes:", options=remaining, key="cfg_extra_attrs",
            help="Any column you additionally want to test for demographic bias — even if not auto-detected."
        )
        selected_attrs.extend(extras)

    if not selected_attrs:
        st.warning("Select at least one protected attribute to continue.")
        return None

    # Warn about high-cardinality numeric attrs that may produce unreliable metrics
    for attr in selected_attrs:
        if df[attr].dtype != object and df[attr].nunique() > 20:
            st.warning(
                f"⚠️ **{attr}** is a continuous numeric column with {df[attr].nunique()} unique values. "
                "Fairness metrics compare one specific privileged value against all others — "
                "for continuous attributes like age consider binning the column first (e.g. age < 40 vs ≥ 40)."
            )

    # ── Privileged values ──────────────────────────────────────────────────────
    st.markdown("**Privileged Group Values** — the historically dominant group in each attribute")
    hint(
        "The 'privileged' group is the reference group with historically higher positive outcome rates "
        "(e.g. Male, White, Non-Disabled). Fairness metrics are computed relative to this group. "
        "This does <b>not</b> imply moral judgement — it's a statistical reference point."
    )
    # Well-known privileged group defaults — applied for any dataset, not just demo
    KNOWN_PRIVILEGED = {"sex": "Male", "race": "White", "gender": "Male"}
    privileged_values = {}
    pv_cols = st.columns(min(len(selected_attrs), 3))
    for i, attr in enumerate(selected_attrs):
        with pv_cols[i % len(pv_cols)]:
            attr_vals = sorted(df[attr].dropna().astype(str).unique().tolist())
            # Use known default if available, else fall back to last alphabetically
            # (minority/disadvantaged groups often sort last, so last = privileged is wrong;
            # but without domain knowledge we just keep it explicit for the user)
            default_priv = KNOWN_PRIVILEGED.get(attr, attr_vals[0])
            if default_priv not in attr_vals:
                default_priv = attr_vals[0]
            default_priv_idx = attr_vals.index(default_priv)
            priv_val = st.selectbox(
                f"Privileged: **{attr}**", options=attr_vals,
                index=default_priv_idx, key=f"cfg_priv_{attr}",
                help=(
                    f"Which value of '{attr}' belongs to the historically privileged group? "
                    "Fairness metrics compute whether the other group(s) receive equal treatment."
                )
            )
            privileged_values[attr] = priv_val

    from src.utils.schema import DatasetConfig
    return DatasetConfig(
        protected_attrs=selected_attrs,
        target_col=target_col,
        privileged_values=privileged_values,
        positive_label=positive_label,
        dataset_name=ds_name,
    )


def make_shareable_link(result) -> tuple[str, bytes]:
    """
    Encode key audit metrics into a base64 URL param and generate an SVG badge.
    Returns (encoded_string, svg_bytes).
    """
    primary = result.config.primary_protected_attr()
    bm = result.baseline_eval.fairness.get(primary) if result.baseline_eval else None
    payload = {
        "dataset":   result.dataset_name,
        "n_rows":    result.n_rows,
        "attr":      primary,
        "di":        round(bm.disparate_impact, 3) if bm and not np.isnan(bm.disparate_impact) else None,
        "severity":  bm.severity if bm else "UNKNOWN",
        "ts":        pd.Timestamp.now().isoformat()[:10],
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

    severity  = payload["severity"]
    di_val    = payload["di"]
    sev_color = {"CRITICAL": "#dc2626", "WARNING": "#d97706", "CLEAR": "#16a34a"}.get(severity, "#6b7280")
    di_text   = f"DI={di_val:.2f}" if di_val is not None else "DI=N/A"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="240" height="20" role="img">
  <title>Aequitas Prime Fairness Audit</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0"  stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1"  stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="240" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="130" height="20" fill="#555"/>
    <rect x="130" width="110" height="20" fill="{sev_color}"/>
    <rect width="240" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="65"  y="15" fill="#010101" fill-opacity=".3">Aequitas Audit | {di_text}</text>
    <text x="65"  y="14">Aequitas Audit | {di_text}</text>
    <text x="185" y="15" fill="#010101" fill-opacity=".3">{severity}</text>
    <text x="185" y="14">{severity}</text>
  </g>
</svg>"""
    return encoded, svg.encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ Aequitas Prime")
    st.markdown('<span style="color:#8b949e;font-size:0.85rem">Fairness-as-a-Service Platform</span>', unsafe_allow_html=True)
    st.divider()

    model_ready = "pipeline_result" in st.session_state
    byom_ready  = "byom_metrics" in st.session_state
    status_color = "#16a34a" if model_ready else "#dc2626"
    status_text  = "Audit Ready" if model_ready else "No Audit Run"
    st.markdown(f'<span style="color:{status_color};font-weight:600">● {status_text}</span>', unsafe_allow_html=True)
    if byom_ready:
        st.markdown('<span style="color:#7c3aed;font-weight:600">● BYOM Loaded</span>', unsafe_allow_html=True)

    if model_ready:
        r = st.session_state["pipeline_result"]
        mitigation_done = r.preprocessed_eval is not None
        mit_color = "#16a34a" if mitigation_done else "#6b7280"
        mit_text  = "Mitigation Done" if mitigation_done else "Mitigation Pending"
        st.markdown(f"""
        <div style="background:#161b22;border-radius:8px;padding:0.6rem 0.8rem;margin-top:0.5rem;
                    font-size:0.82rem;border:1px solid #30363d;">
          <div style="color:#8b949e">Dataset</div>
          <div style="color:#e6edf3;font-weight:600">{r.dataset_name}</div>
          <div style="color:#8b949e;margin-top:0.3rem">Rows / Features</div>
          <div style="color:#e6edf3">{r.n_rows:,} / {r.n_features}</div>
          <div style="color:#8b949e;margin-top:0.3rem">Mitigation</div>
          <div style="color:{mit_color}">{mit_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── V2: Wizard mode toggle ─────────────────────────────────────────────────
    wizard_mode = st.toggle("🧙 Guided Wizard Mode", value=False, key="wizard_mode",
                             help="Switch to a step-by-step linear flow (Upload → Configure → Detect → Mitigate → Report).")

    st.divider()

    # ── Gemini AI config ───────────────────────────────────────────────────────
    st.markdown('<span style="color:#8b949e;font-size:0.82rem">GOOGLE GEMINI AI</span>', unsafe_allow_html=True)
    _gemini_key = st.text_input(
        "Gemini API Key", type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        label_visibility="collapsed", key="llm_key_gemini",
        help="Free key at aistudio.google.com/apikey  •  15 req/min, 1M tokens/day",
    )
    if _gemini_key:
        os.environ["GEMINI_API_KEY"] = _gemini_key
    st.session_state["_llm_api_key"] = _gemini_key
    st.session_state["_llm_provider"] = "gemini"
    if _gemini_key:
        st.markdown('<span style="color:#3fb950;font-size:0.75rem">✓ Gemini connected — AI features enabled.</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#6b7280;font-size:0.75rem">Leave blank — rule-based fallback auto-generates.</span>', unsafe_allow_html=True)

    st.divider()

    # ── V2: Audit Persistence ──────────────────────────────────────────────────
    if model_ready:
        if st.button("💾 Save Audit Snapshot", use_container_width=True,
                     help="Save this audit to local history for comparison across runs."):
            try:
                from src.utils.persistence import save_audit
                _aid = save_audit(st.session_state["pipeline_result"])
                st.success(f"Saved: {_aid[:20]}…")
            except Exception as _e:
                st.error(f"Save failed: {_e}")

    if st.button("📜 Audit History", use_container_width=True,
                 help="View previously saved audits."):
        st.session_state["_show_history"] = not st.session_state.get("_show_history", False)

    if st.session_state.get("_show_history", False):
        try:
            from src.utils.persistence import audit_history_dataframe
            _hist = audit_history_dataframe()
            if _hist.empty:
                st.caption("No saved audits yet.")
            else:
                st.dataframe(_hist.drop(columns=["audit_id"], errors="ignore"),
                             use_container_width=True, hide_index=True,
                             column_config={"ΔDI": st.column_config.NumberColumn(format="%.3f")})
        except Exception as _e:
            st.caption(f"History unavailable: {_e}")

    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem;color:#6b7280;line-height:1.8">
      <b style="color:#8b949e">Navigation</b><br>
      🔍 <b>Bias X-Ray</b> — upload any CSV, detect bias<br>
      🤖 <b>BYOM</b> — audit your own model's predictions<br>
      ⚗️ <b>Surgeon</b> — mitigate & compare<br>
      🧩 <b>Intersectional</b> — compound group bias<br>
      🔬 <b>Causal</b> — direct vs indirect discrimination<br>
      🔀 <b>CF Panel</b> — dataset-level flip evidence<br>
      👻 <b>Ghost</b> — live single-prediction firewall<br>
      📄 <b>Report</b> — PDF + shareable badge
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("v2.0.0 · Google Hackathon 2025")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab4b, tab5, tab6, tab7 = st.tabs([
    "🔍 Bias X-Ray",
    "🤖 BYOM",
    "⚗️ Surgeon",
    "🧩 Intersectional",
    "🔬 Causal",
    "🔀 CF Panel",
    "👻 Ghost",
    "📄 Report",
])


# ══════════════════════════════════════════════════════════════════════════════
# V2: WIZARD MODE (linear guided flow shown when wizard_mode toggle is on)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("wizard_mode", False):
    st.markdown("## 🧙 Guided Wizard Mode")
    hint(
        "Follow these steps in order. Each step unlocks the next. "
        "Toggle <b>Guided Wizard Mode</b> off in the sidebar to return to the full expert layout."
    )

    _wiz_step = st.session_state.get("_wiz_step", 1)
    _wiz_steps = ["1. Upload Dataset", "2. Configure Audit", "3. Run Bias X-Ray",
                  "4. Run Mitigation", "5. Generate Report"]
    st.progress((_wiz_step - 1) / (len(_wiz_steps) - 1), text=_wiz_steps[_wiz_step - 1])

    # --- Step 1: Upload ---
    if _wiz_step >= 1:
        st.markdown("### Step 1 — Upload Your Dataset")
        _wup = st.file_uploader("Upload CSV", type=["csv"], key="wiz_upload")
        _wdemo = st.button("Or use UCI Adult Income demo", key="wiz_demo")
        if _wup is not None:
            try:
                _wdf = pd.read_csv(_wup, sep=None, engine="python", na_values=["?", " ?"])
                _wdf[_wdf.select_dtypes("object").columns] = _wdf.select_dtypes("object").apply(lambda c: c.str.strip())
                st.session_state["raw_df"]     = _wdf
                st.session_state["_use_adult"] = False
                if _wiz_step == 1:
                    st.session_state["_wiz_step"] = 2
                    st.rerun()
            except Exception as _we:
                st.error(f"Cannot read CSV: {_we}")
        if _wdemo:
            from src.utils.data_loader import load_csv
            try:
                st.session_state["raw_df"]     = load_csv("data/raw/adult.csv")
                st.session_state["_use_adult"] = True
                st.session_state["_wiz_step"]  = 2
                st.rerun()
            except Exception as _we:
                st.error(f"Cannot load demo: {_we}")

    # --- Step 2: Configure ---
    if _wiz_step >= 2 and "raw_df" in st.session_state:
        st.divider()
        st.markdown("### Step 2 — Configure Audit")
        _wconfig = build_config_ui(st.session_state["raw_df"],
                                    is_demo=st.session_state.get("_use_adult", False))
        if _wconfig and st.button("✅ Confirm Configuration", key="wiz_cfg_ok"):
            st.session_state["_wiz_config"] = _wconfig
            st.session_state["_wiz_step"]   = 3
            st.rerun()

    # --- Step 3: Run X-Ray ---
    if _wiz_step >= 3 and "raw_df" in st.session_state:
        st.divider()
        st.markdown("### Step 3 — Run Bias X-Ray")
        _wc = st.session_state.get("_wiz_config")
        if _wc and "pipeline_result" not in st.session_state:
            st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
            if st.button("🔬 Run Bias X-Ray Now", use_container_width=True, key="wiz_run"):
                with st.spinner("Running bias audit..."):
                    try:
                        from src.ml_pipeline.pipeline import run_full_pipeline
                        _is_adult = st.session_state.get("_use_adult", False)
                        _prep = None
                        if _is_adult:
                            from src.utils.adult_preprocessor import preprocess as _prep
                        else:
                            from src.utils.generic_preprocessor import preprocess_generic as _prep
                        _wr = run_full_pipeline(
                            st.session_state["raw_df"], _wc,
                            run_mitigation=False, run_inprocessing=False,
                            verbose=False, preprocess_fn=_prep,
                        )
                        st.session_state["pipeline_result"] = _wr
                        st.session_state["_wiz_step"] = 4
                        st.rerun()
                    except Exception as _we:
                        st.error(f"Pipeline error: {_we}")
            st.markdown("</div>", unsafe_allow_html=True)
        elif "pipeline_result" in st.session_state:
            _wr = st.session_state["pipeline_result"]
            _wp = _wr.config.primary_protected_attr()
            _wbm = _wr.baseline_eval.fairness.get(_wp) if _wr.baseline_eval else None
            if _wbm:
                _wdi = _wbm.disparate_impact
                st.metric("Disparate Impact (primary attr)", f"{_wdi:.3f}",
                          delta="⚠ Below legal threshold" if _wdi < 0.80 else "✅ Compliant",
                          delta_color="inverse" if _wdi < 0.80 else "normal")
            if st.button("Continue to Mitigation →", key="wiz_to_mit"):
                st.session_state["_wiz_step"] = 4
                st.rerun()

    # --- Step 4: Mitigation ---
    if _wiz_step >= 4 and "pipeline_result" in st.session_state:
        st.divider()
        st.markdown("### Step 4 — Apply Bias Mitigation")
        _wr = st.session_state["pipeline_result"]
        if _wr.preprocessed_eval is None:
            if st.button("⚗️ Run Mitigation", use_container_width=True, key="wiz_mit"):
                with st.spinner("Applying Reweighing..."):
                    try:
                        from src.ml_pipeline.pipeline import run_mitigation_steps
                        _wr = run_mitigation_steps(_wr, run_inprocessing=False, verbose=False)
                        st.session_state["pipeline_result"] = _wr
                        st.session_state["_wiz_step"] = 5
                        st.rerun()
                    except Exception as _we:
                        st.error(f"Mitigation error: {_we}")
        else:
            _wp = _wr.config.primary_protected_attr()
            _pm = _wr.preprocessed_eval.fairness.get(_wp) if _wr.preprocessed_eval else None
            if _pm:
                st.metric("DI after Reweighing", f"{_pm.disparate_impact:.3f}",
                          delta=f"{'✅ Fixed' if _pm.disparate_impact >= 0.80 else '⚠ Still low'}")
            if st.button("Continue to Report →", key="wiz_to_rep"):
                st.session_state["_wiz_step"] = 5
                st.rerun()

    # --- Step 5: Report ---
    if _wiz_step >= 5 and "pipeline_result" in st.session_state:
        st.divider()
        st.markdown("### Step 5 — Generate Compliance Report")
        st.info("Go to the **📄 Report** tab to generate a PDF and AI narrative, or use the Save button in the sidebar.")
        if st.button("💾 Save Audit to History", key="wiz_save"):
            try:
                from src.utils.persistence import save_audit
                _aid = save_audit(st.session_state["pipeline_result"])
                st.success(f"Saved! Audit ID: {_aid[:24]}")
            except Exception as _we:
                st.error(f"Save failed: {_we}")

    st.stop()   # Don't render tabs below in wizard mode


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BIAS X-RAY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🔍 Bias X-Ray — Detect & Measure")
    hint(
        "Upload <b>any</b> binary-classification CSV dataset. Aequitas Prime will auto-detect "
        "protected attributes, let you configure the audit, then scan for bias using "
        "<b>6 international fairness metrics</b>. Or click <b>Use Adult Income Sample Data</b> to try a demo."
    )

    # ── Data source selection ─────────────────────────────────────────────────
    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload CSV dataset", type=["csv"],
            help="Any binary-classification CSV. Protected attributes are auto-detected from column names and cardinality."
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("📊 Use Adult Income\nSample Data", use_container_width=True,
                               help="Load the UCI Adult Income dataset (48,842 rows, sex + race as protected attributes).")

    # ── Load data into session state ──────────────────────────────────────────
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded, sep=None, engine="python", na_values=["?", " ?"])
            str_cols = df_raw.select_dtypes("object").columns
            df_raw[str_cols] = df_raw[str_cols].apply(lambda c: c.str.strip())
            st.session_state["raw_df"]     = df_raw
            st.session_state["_use_adult"] = False
            # Clear previous run if dataset changed
            if st.session_state.get("_last_upload_name") != uploaded.name:
                for k in ["pipeline_result","shap_importance","shap_explainer","feat_names",
                           "pareto_points","narrative","provenance","batch_cf_results"]:
                    st.session_state.pop(k, None)
                st.session_state["_last_upload_name"] = uploaded.name
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if use_sample:
        from src.utils.data_loader import load_csv
        try:
            df_raw = load_csv("data/raw/adult.csv")
            st.session_state["raw_df"]     = df_raw
            st.session_state["_use_adult"] = True
            for k in ["pipeline_result","shap_importance","shap_explainer","feat_names",
                       "pareto_points","narrative","provenance","batch_cf_results"]:
                st.session_state.pop(k, None)
            st.success("Adult Income dataset loaded — configure below and click Run.")
        except Exception as e:
            st.error(f"Could not load sample data: {e}")

    # ── Show config UI if data is available ───────────────────────────────────
    raw_df    = st.session_state.get("raw_df")
    is_demo   = st.session_state.get("_use_adult", False)

    if raw_df is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem;color:#6b7280">
          <div style="font-size:3.5rem">⚖️</div>
          <div style="font-size:1.1rem;margin-top:0.6rem">
            Upload a CSV or click <b style="color:#a78bfa">Use Adult Income Sample Data</b> to begin.
          </div>
          <div style="font-size:0.85rem;margin-top:0.4rem;color:#484f58">
            Supports any binary-classification dataset — income, credit, hiring, healthcare, and more.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    with st.expander(f"📋 Dataset Preview — {len(raw_df):,} rows × {len(raw_df.columns)} columns", expanded=False):
        st.dataframe(raw_df.head(10), use_container_width=True)

    # ── Gemini Dataset Column Advisor ─────────────────────────────────────────
    _adv_key = st.session_state.get("_llm_api_key", "")
    if _adv_key and not is_demo:
        _adv_cache_key = f"_col_advice_{st.session_state.get('_last_upload_name','')}"
        if _adv_cache_key not in st.session_state:
            if st.button("✨ Gemini: Auto-detect Protected Attributes", key="btn_col_advisor",
                         help="Gemini analyses your column names and sample values to suggest protected attributes, target column, and proxy risks."):
                with st.spinner("Gemini is analysing your dataset columns..."):
                    try:
                        from src.narrative.llm_narrator import suggest_sensitive_columns
                        _sample_vals = {
                            col: raw_df[col].dropna().unique()[:5].tolist()
                            for col in raw_df.columns
                        }
                        _advice = suggest_sensitive_columns(
                            list(raw_df.columns), _sample_vals, api_key=_adv_key
                        )
                        if _advice:
                            st.session_state[_adv_cache_key] = _advice
                    except Exception as _ae:
                        st.warning(f"Gemini column advisor error: {_ae}")

        if _adv_cache_key in st.session_state:
            _adv = st.session_state[_adv_cache_key]
            with st.expander("✨ Gemini Column Advisor — click to see suggestions", expanded=True):
                _ca1, _ca2 = st.columns(2)
                with _ca1:
                    st.markdown("**Likely protected attributes:**")
                    for _pa in _adv.get("protected_attrs", []):
                        st.markdown(f"- `{_pa}`")
                    if _adv.get("target_col"):
                        st.markdown(f"**Likely target column:** `{_adv['target_col']}`")
                with _ca2:
                    if _adv.get("proxy_risks"):
                        st.markdown("**Proxy risk columns:**")
                        for _pr in _adv.get("proxy_risks", []):
                            st.markdown(f"- `{_pr}`")
                if _adv.get("reasoning"):
                    st.caption(_adv["reasoning"])
                st.info("Use these suggestions when configuring the audit below.")

    st.divider()

    config = build_config_ui(raw_df, is_demo=is_demo)

    st.divider()

    run_inproc = st.checkbox(
        "Include in-processing mitigation in Surgeon tab",
        value=False,
        help="Adds ExponentiatedGradient (Fairlearn) training in the Surgeon tab. Takes ~2 extra minutes."
    )
    st.session_state["_run_inproc"] = run_inproc

    from src.ml_pipeline.trainer import MODEL_DISPLAY
    base_model_key = st.selectbox(
        "Base model",
        options=list(MODEL_DISPLAY.keys()),
        format_func=lambda k: MODEL_DISPLAY[k],
        index=0,
        help="Which model to train for baseline, pre-processed, and in-processed comparisons."
    )
    st.session_state["_base_model"] = base_model_key

    st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
    run_clicked = st.button("🔬 Run Full Bias X-Ray", use_container_width=True,
                            disabled=(config is None))
    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked and config is not None:
        with st.spinner("Running bias audit pipeline (baseline model + metrics)..."):
            try:
                from src.ml_pipeline.pipeline import run_full_pipeline
                if is_demo:
                    from src.utils.adult_preprocessor import preprocess as prep_fn
                else:
                    from src.utils.generic_preprocessor import preprocess_generic as prep_fn
                result = run_full_pipeline(
                    raw_df, config,
                    run_mitigation=False,
                    run_inprocessing=False,
                    verbose=False,
                    preprocess_fn=prep_fn,
                    base_model=base_model_key,
                )
                st.session_state["pipeline_result"] = result
                st.session_state["_config_snapshot"] = config
                for k in ["shap_importance","shap_explainer","feat_names",
                           "pareto_points","narrative","provenance","batch_cf_results"]:
                    st.session_state.pop(k, None)
                st.success("Bias X-Ray complete! Scroll down for results.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback; st.code(traceback.format_exc())
                st.stop()

    if "pipeline_result" not in st.session_state:
        st.stop()

    result  = st.session_state["pipeline_result"]

    # Pick the most-biased protected attr as "primary" for the 6-metric table and
    # provenance — prefer the attr with the lowest valid DI (most legally concerning).
    def _primary_by_bias(res) -> str:
        if res.baseline_eval is None:
            return res.config.primary_protected_attr()
        def _di_key(attr):
            m = res.baseline_eval.fairness.get(attr)
            if m is None or np.isnan(m.disparate_impact) or m.privileged_count == 0:
                return 2.0  # sort last — undefined/empty-group metrics
            return m.disparate_impact  # lower DI = more biased = show first
        return min(res.config.protected_attrs, key=_di_key)

    primary = _primary_by_bias(result)
    bm      = result.baseline_eval.fairness.get(primary) if result.baseline_eval else None

    st.divider()

    # ── DI Gauges ─────────────────────────────────────────────────────────────
    section("Disparate Impact Gauges")
    hint(
        "<b>Disparate Impact (DI)</b> = P(positive outcome | unprivileged) ÷ P(positive outcome | privileged). "
        "A DI below <b>0.80</b> violates the US 4/5ths rule (EEOC) and the EU AI Act Article 10. "
        "<b>Green ≥ 0.90</b> = compliant. <b>Yellow 0.80–0.90</b> = borderline. <b>Red &lt; 0.80</b> = legally actionable."
    )
    _GAUGE_COLS = 3  # max gauges per row
    _all_attrs = result.config.protected_attrs
    for _row_start in range(0, len(_all_attrs), _GAUGE_COLS):
        _row_attrs = _all_attrs[_row_start:_row_start + _GAUGE_COLS]
        gauge_cols = st.columns(len(_row_attrs))
        for i, attr in enumerate(_row_attrs):
            m = result.baseline_eval.fairness.get(attr) if result.baseline_eval else None
            if m:
                with gauge_cols[i]:
                    _di_raw = m.disparate_impact
                    _di_nan = isinstance(_di_raw, float) and np.isnan(_di_raw)
                    di_val  = 0.0 if _di_nan else _di_raw
                    st.plotly_chart(di_gauge(di_val, f"DI — {attr}"),
                                    use_container_width=True, config={"displayModeBar": False})
                    priv_rate   = m.privileged_approval_rate
                    unpriv_rate = m.unprivileged_approval_rate
                    _priv_nan   = isinstance(priv_rate, float) and np.isnan(priv_rate)

                    if _di_nan or _priv_nan or m.privileged_count == 0:
                        # Privileged group was empty — config may have a bad privileged value
                        st.markdown(f"""
                        <div class="alert-box-warn">
                          <b>⚠️ {attr.upper()}: DI = N/A — Privileged group not found</b><br>
                          No rows matched the privileged value <code>{m.privileged_value}</code> for column <b>{attr}</b>.
                          Check the <b>Privileged Group Value</b> setting in the config above and re-run.
                        </div>
                        """, unsafe_allow_html=True)
                    elif di_val < 0.80:
                        ratio_pct = f"{di_val:.1%}"
                        st.markdown(f"""
                        <div class="alert-box-critical">
                          <div class="alert-title">🚨 BIAS DETECTED — {attr.upper()}</div>
                          <b>DI = {di_val:.3f}</b> (legal minimum: 0.80)<br>
                          Unprivileged group receives positive outcomes at only <b>{ratio_pct}</b> the rate of the privileged group.<br>
                          <span style="opacity:0.8;font-size:0.82rem">
                            Privileged: {priv_rate:.1%} &nbsp;|&nbsp; Unprivileged: {unpriv_rate:.1%}
                          </span><br><br>
                          <b>Next step:</b> Go to the <b>⚗️ Surgeon</b> tab to apply bias mitigation.
                        </div>
                        """, unsafe_allow_html=True)
                    elif di_val < 0.90:
                        st.markdown(f"""
                        <div class="alert-box-warn">
                          <b>⚠️ {attr.upper()}: DI = {di_val:.3f} — Borderline</b><br>
                          Passes the 0.80 legal threshold but regulators often scrutinise DI below 0.90.
                          Approval gap: {priv_rate:.1%} (privileged) vs {unpriv_rate:.1%} (unprivileged). <b>Monitoring recommended.</b>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-box-ok">
                          <b>✅ {attr.upper()}: DI = {di_val:.3f} — Compliant</b><br>
                          Above the 0.80 threshold. Approval rates are similar:
                          {priv_rate:.1%} (privileged) vs {unpriv_rate:.1%} (unprivileged). No immediate action needed.
                        </div>
                        """, unsafe_allow_html=True)

    st.divider()

    # ── Six-Metric Table ───────────────────────────────────────────────────────
    section("6-Metric Fairness Summary")
    hint(
        "Six internationally recognized metrics, each measuring a different dimension of fairness. "
        "A model can pass Disparate Impact but still fail Equalized Odds — that's why all six matter. "
        "<b>Hover the 'Plain-English Meaning' column for a non-technical explanation of each metric.</b>"
    )

    # Let the user pick any protected attribute — defaults to the most biased one
    _attr_options = result.config.protected_attrs
    _default_idx  = _attr_options.index(primary) if primary in _attr_options else 0
    _smcol1, _smcol2 = st.columns([3, 1])
    with _smcol1:
        selected_metric_attr = st.selectbox(
            "Show 6-metric breakdown for:",
            options=_attr_options,
            index=_default_idx,
            key="six_metric_attr_selector",
            help=(
                "The most biased attribute is pre-selected. "
                "Switch to inspect the full metric breakdown for any other protected attribute."
            ),
        )
    with _smcol2:
        _show_ci = st.toggle("Show 95% CI", value=False, key="ci_toggle",
                              help="Bootstrap 95% confidence intervals (500 resamples). Takes ~5s on first load.")

    _bm_selected = result.baseline_eval.fairness.get(selected_metric_attr) if result.baseline_eval else None
    if _bm_selected:
        # V2: compute bootstrap CI on demand
        _ci_computed = st.session_state.get(f"_ci_{selected_metric_attr}")
        if _show_ci and _ci_computed is None:
            with st.spinner("Computing bootstrap confidence intervals (500 resamples)..."):
                try:
                    from src.bias_engine.detector import compute_bootstrap_ci
                    _ci_result = compute_bootstrap_ci(result._df_clean, result.config,
                                                       y_pred=None, attr=selected_metric_attr)
                    st.session_state[f"_ci_{selected_metric_attr}"] = _ci_result
                    _ci_computed = _ci_result
                except Exception as _ce:
                    st.caption(f"CI computation unavailable: {_ce}")

        # Attach CI fields to _bm_selected if available
        if _show_ci and _ci_computed is not None:
            for _ci_field in ["ci_disparate_impact","ci_statistical_parity_diff",
                               "ci_equalized_odds_diff","ci_equal_opportunity_diff",
                               "ci_predictive_parity_diff","ci_fnr_parity_diff"]:
                if hasattr(_ci_computed, _ci_field):
                    try:
                        setattr(_bm_selected, _ci_field, getattr(_ci_computed, _ci_field))
                    except Exception:
                        pass

        show_six_metric_table(_bm_selected, selected_metric_attr, show_ci=_show_ci)
    else:
        st.info(f"No metrics available for **{selected_metric_attr}**.")

    st.divider()

    # ── Proxy Scanner ──────────────────────────────────────────────────────────
    section("🕵️ Shadow Proxy Scanner")
    hint(
        "<b>What is a shadow proxy?</b> Even if you remove the protected column (e.g. 'sex'), "
        "the model can re-learn demographic bias through correlated features like 'occupation' or 'zip code'. "
        "<b>Mutual Information (MI)</b> measures how much each feature leaks demographic information. "
        "MI ≥ 0.15 = HIGH risk — strong back-door bias channel even without the protected column."
    )
    from src.bias_engine.proxy_scanner import get_flagged_proxies, proxy_heatmap_data
    flagged = get_flagged_proxies(result.proxy_scan)
    if flagged:
        st.warning(f"**{len(flagged)} shadow proxy feature(s) detected** — removing the protected "
                   "attribute alone will NOT eliminate bias.")
        proxy_df = pd.DataFrame([{
            "Feature":           r.feature,
            "Protected Attr":    r.protected_attr,
            "Combined Score":    round(getattr(r, "combined_score", r.mutual_info), 4),
            "Mutual Info":       round(r.mutual_info, 4),
            "Cramér V":          round(getattr(r, "cramers_v", float("nan")), 4)
                                 if not np.isnan(getattr(r, "cramers_v", float("nan"))) else "—",
            "Risk Level":        r.risk_level,
            "Recommended Action": r.action,
        } for r in flagged])
        st.dataframe(proxy_df, use_container_width=True, hide_index=True,
                     column_config={"Recommended Action": st.column_config.TextColumn(width="large")})
        hm_data = proxy_heatmap_data(result.proxy_scan)
        if not hm_data.empty:
            fig_hm = px.imshow(
                hm_data.head(15), text_auto=".3f",
                color_continuous_scale=[[0,"#0d1117"],[0.5,"#7c3aed"],[1,"#dc2626"]],
                title="Combined Risk Score Heatmap — Features vs Protected Attributes (V2: MI + Cramér V + PBR)",
                labels=dict(color="Combined Score"),
            )
            fig_hm.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})
    else:
        st.success("No high-risk proxy features detected. The dataset appears clean of back-door bias channels.")

    st.divider()

    # ── Provenance ─────────────────────────────────────────────────────────────
    section("🧬 Bias Provenance — Where Did the Bias Come From?")
    hint(
        "Bias can originate from three sources, each requiring a <b>different fix</b>:<br>"
        "<b>Label Bias</b> — historical human decisions were discriminatory (fix: reweighing, label correction).<br>"
        "<b>Representation Bias</b> — one group is severely underrepresented (fix: collect more data, oversample).<br>"
        "<b>Feature Bias</b> — features encode demographics as proxies (fix: remove or repair those features)."
    )
    from src.bias_engine.provenance import trace_bias_provenance
    prov = trace_bias_provenance(result._df_clean, result.config, result.proxy_scan,
                                 primary_attr=primary)
    st.session_state["provenance"] = prov
    p1, p2, p3 = st.columns(3)
    p1.metric("Primary Bias Source",  prov.primary_label,
              help="The most likely root cause of bias in this dataset, based on statistical evidence.")
    p2.metric("Confidence",           f"{prov.confidence:.0%}",
              help="How strongly the evidence points to this source. Above 70% = high confidence.")
    p3.metric("Secondary Source",     (prov.secondary_source or "None").replace("_"," ").title(),
              help="A secondary contributing cause, if present.")
    with st.expander("View full evidence & recommended fixes"):
        for e in prov.evidence:
            st.markdown(f"- {e}")
        st.markdown("**Recommended Fixes (priority order):**")
        for i, fix in enumerate(prov.recommended_fixes, 1):
            st.markdown(f"{i}. {fix}")
        if prov.caution_notes:
            st.warning("**Caution:** " + " | ".join(prov.caution_notes))
    st.info(f"**Top Recommended Action:** {prov.recommended_fixes[0] if prov.recommended_fixes else 'Continue monitoring.'}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BYOM (Bring Your Own Model)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🤖 BYOM — Bring Your Own Model")
    hint(
        "Already have a trained model? Upload a CSV containing your model's <b>predictions</b> alongside "
        "the true labels. Aequitas Prime will compute all 6 fairness metrics on your predictions without "
        "needing access to your model's internals. Works with XGBoost, PyTorch, HuggingFace, scikit-learn, or any framework."
    )

    st.markdown("### Required CSV format")
    st.markdown("""
    Your CSV must include:
    - **Feature columns** — the same inputs your model used
    - **A true-label column** — the actual ground truth
    - **A predicted-label column** — your model's binary predictions (0/1 or class labels)
    - **Protected attribute columns** — sex, race, etc.
    """)

    byom_file = st.file_uploader("Upload predictions CSV", type=["csv"], key="byom_uploader",
                                  help="CSV with features + true label + your model's predicted label.")

    if byom_file is not None:
        try:
            byom_df = pd.read_csv(byom_file, sep=None, engine="python", na_values=["?", " ?"])
            str_cols = byom_df.select_dtypes("object").columns
            byom_df[str_cols] = byom_df[str_cols].apply(lambda c: c.str.strip())
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            byom_df = None

        if byom_df is not None:
            st.success(f"Loaded {len(byom_df):,} rows × {len(byom_df.columns)} columns.")
            with st.expander("Preview (first 5 rows)"):
                st.dataframe(byom_df.head(), use_container_width=True)

            bcols = byom_df.columns.tolist()

            st.divider()
            section("Configure Your BYOM Audit")

            ba1, ba2 = st.columns(2)
            with ba1:
                from src.utils.generic_preprocessor import auto_detect_target
                auto_true = auto_detect_target(byom_df)
                auto_true_idx = bcols.index(auto_true) if auto_true in bcols else len(bcols)-1
                byom_true_col = st.selectbox(
                    "True Label Column", options=bcols, index=auto_true_idx, key="byom_true",
                    help="Column containing the actual ground-truth labels."
                )
            with ba2:
                pred_candidates = [c for c in bcols if any(k in c.lower() for k in ["pred","forecast","output","score","decision"])]
                pred_default_idx = bcols.index(pred_candidates[0]) if pred_candidates else min(1, len(bcols)-1)
                byom_pred_col = st.selectbox(
                    "Predicted Label Column", options=bcols, index=pred_default_idx, key="byom_pred",
                    help="Column containing your model's predictions (binary: same values as true label)."
                )

            # Positive outcome
            true_vals = sorted(byom_df[byom_true_col].dropna().astype(str).unique().tolist()) if byom_true_col else []
            byom_pos = st.selectbox("Positive Outcome Value", options=true_vals, key="byom_pos",
                                     help="Which label value represents the 'good' outcome?") if true_vals else None

            # Protected attrs
            from src.utils.generic_preprocessor import auto_detect_protected_attrs
            byom_non_pred = [c for c in bcols if c not in [byom_true_col, byom_pred_col]]
            byom_auto_prot = auto_detect_protected_attrs(byom_df, target_col=byom_true_col)
            byom_auto_prot = [a for a in byom_auto_prot if a in byom_non_pred]

            st.markdown("**Protected Attributes**")
            byom_selected_attrs = []
            for attr in byom_auto_prot:
                if st.checkbox(f"✓ {attr}", value=True, key=f"byom_attr_{attr}"):
                    byom_selected_attrs.append(attr)
            byom_remaining = [c for c in byom_non_pred if c not in byom_auto_prot]
            if byom_remaining:
                byom_extras = st.multiselect("Add more:", options=byom_remaining, key="byom_extra_attrs")
                byom_selected_attrs.extend(byom_extras)

            # Privileged values
            byom_priv_vals = {}
            if byom_selected_attrs:
                st.markdown("**Privileged Group Values**")
                bpv_cols = st.columns(min(len(byom_selected_attrs), 3))
                for i, attr in enumerate(byom_selected_attrs):
                    with bpv_cols[i % len(bpv_cols)]:
                        av = sorted(byom_df[attr].dropna().astype(str).unique().tolist())
                        byom_priv_vals[attr] = st.selectbox(
                            f"Privileged: {attr}", options=av, key=f"byom_priv_{attr}",
                            help=f"The historically dominant group in the '{attr}' column."
                        )

            byom_ready = (byom_true_col and byom_pred_col and byom_pos and byom_selected_attrs)

            if st.button("🤖 Run BYOM Fairness Audit", use_container_width=True, disabled=not byom_ready):
                with st.spinner("Computing fairness metrics on your model's predictions..."):
                    try:
                        from src.utils.schema import DatasetConfig
                        from src.bias_engine.detector import compute_metrics_for_all_attrs

                        byom_config = DatasetConfig(
                            protected_attrs=byom_selected_attrs,
                            target_col=byom_true_col,
                            privileged_values=byom_priv_vals,
                            positive_label=byom_pos,
                            dataset_name="BYOM Upload",
                        )

                        # Binarize target and get predictions as numpy
                        df_audit = byom_df.copy()
                        for col in df_audit.select_dtypes("object").columns:
                            df_audit[col] = df_audit[col].str.strip()
                        df_audit[byom_true_col] = (
                            df_audit[byom_true_col].astype(str) == str(byom_pos)
                        ).astype(int)
                        byom_config = DatasetConfig(
                            protected_attrs=byom_selected_attrs,
                            target_col=byom_true_col,
                            privileged_values=byom_priv_vals,
                            positive_label=1,  # now binarized
                            dataset_name="BYOM Upload",
                        )
                        y_pred_byom = (
                            df_audit[byom_pred_col].astype(str) == str(byom_pos)
                        ).astype(int).values

                        byom_metrics = compute_metrics_for_all_attrs(
                            df_audit, byom_config, y_pred=y_pred_byom
                        )
                        st.session_state["byom_metrics"] = byom_metrics
                        st.session_state["byom_config"]  = byom_config
                        st.session_state["byom_n_rows"]  = len(df_audit)
                        st.success("BYOM audit complete!")
                    except Exception as e:
                        st.error(f"BYOM audit error: {e}")
                        import traceback; st.code(traceback.format_exc())

    # ── Show BYOM results ──────────────────────────────────────────────────────
    if "byom_metrics" in st.session_state:
        byom_metrics = st.session_state["byom_metrics"]
        byom_config  = st.session_state["byom_config"]
        byom_n_rows  = st.session_state.get("byom_n_rows", "?")

        st.divider()
        st.markdown(f"### Audit Results — {byom_config.dataset_name} ({byom_n_rows:,} rows)")

        for attr, bm in byom_metrics.items():
            st.markdown(f"#### Protected Attribute: `{attr}`")
            c1, c2, c3 = st.columns(3)
            di_val = bm.disparate_impact if not np.isnan(bm.disparate_impact) else None
            c1.metric("Disparate Impact", f"{di_val:.3f}" if di_val else "N/A",
                      help="DI < 0.80 = legally actionable bias under the 4/5ths rule.")
            c2.metric("Privileged Approval",   f"{bm.privileged_approval_rate:.1%}",
                      help="Fraction of the privileged group that received a positive prediction.")
            c3.metric("Unprivileged Approval", f"{bm.unprivileged_approval_rate:.1%}",
                      help="Fraction of the unprivileged group that received a positive prediction.")
            if di_val:
                st.plotly_chart(di_gauge(di_val, f"DI — {attr}"),
                                use_container_width=True, config={"displayModeBar": False})
            show_six_metric_table(bm, attr)
            st.divider()

    elif byom_file is None:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#6b7280">
          <div style="font-size:2.5rem">🤖</div>
          <div style="font-size:1rem;margin-top:0.5rem">
            Upload a predictions CSV above to audit any model's fairness.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GENERATIVE SURGEON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## ⚗️ The Generative Surgeon — Bias Mitigation")
    hint(
        "The Surgeon applies two bias correction techniques and shows before/after comparisons. "
        "<b>Pre-processing (Reweighing)</b> rebalances the training data. "
        "<b>In-processing (ExponentiatedGradient)</b> bakes a fairness constraint into the learning algorithm. "
        "Click <b>Run Mitigation</b> to trigger — this is kept separate so you can run the X-Ray instantly first."
    )

    if "pipeline_result" not in st.session_state:
        st.info("Complete the Bias X-Ray first (Tab 1).")
        st.stop()

    result  = st.session_state["pipeline_result"]
    primary = result.config.primary_protected_attr()

    # ── Mitigation trigger ────────────────────────────────────────────────────
    if result.preprocessed_eval is None:
        st.warning("Mitigation has not been run yet.")
        run_inproc_surgeon = st.session_state.get("_run_inproc", False)
        st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
        run_mit = st.button("⚗️ Run Mitigation Analysis", use_container_width=True,
                            help="Applies Reweighing (pre-processing) and optionally ExponentiatedGradient (in-processing).")
        st.markdown("</div>", unsafe_allow_html=True)
        if run_mit:
            with st.spinner("Applying bias mitigation techniques..."):
                try:
                    from src.ml_pipeline.pipeline import run_mitigation_steps
                    bm_key = st.session_state.get("_base_model", "rf")
                    result = run_mitigation_steps(
                        result,
                        run_inprocessing=run_inproc_surgeon,
                        base_model=bm_key,
                        verbose=False,
                    )
                    st.session_state["pipeline_result"] = result
                    st.success("Mitigation complete! Results below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Mitigation error: {e}")
                    import traceback; st.code(traceback.format_exc())
        if result.preprocessed_eval is None:
            st.stop()

    be = result.baseline_eval
    pe = result.preprocessed_eval
    ie = result.inprocessed_eval
    bm = be.fairness.get(primary) if be else None
    pm = pe.fairness.get(primary) if pe else None
    im = ie.fairness.get(primary) if ie else None

    best_eval = ie if ie else pe
    best_m    = im if im else pm
    acc_delta = best_eval.performance.accuracy - be.performance.accuracy
    di_delta  = (best_m.disparate_impact - bm.disparate_impact) if (best_m and bm) else 0

    # ── Headline metrics ───────────────────────────────────────────────────────
    section("Results at a Glance")
    hint(
        "Goal: <b>maximize Disparate Impact</b> (closer to 1.0) while <b>minimising accuracy loss</b>. "
        "A good mitigation improves DI by 0.3+ while losing less than 2% accuracy."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy — Baseline",  f"{be.performance.accuracy:.1%}",
              help="Model accuracy before any bias correction. Higher is better.")
    c2.metric("Accuracy — After",     f"{best_eval.performance.accuracy:.1%}",
              delta=f"{acc_delta:+.1%}",
              help="Accuracy after bias mitigation. A small drop (<2%) is acceptable.")
    c3.metric("DI — Baseline",        f"{bm.disparate_impact:.3f}" if bm else "—",
              help="Disparate Impact before mitigation. Below 0.80 = illegal under 4/5ths rule.")
    c4.metric("DI — After",           f"{best_m.disparate_impact:.3f}" if best_m else "—",
              delta=f"{di_delta:+.3f}" if di_delta else None,
              help="Disparate Impact after mitigation. Target: ≥ 0.80 (legal threshold).")

    st.divider()

    # ── Approval rate chart ────────────────────────────────────────────────────
    section("Approval Rate by Group — Before vs After")
    hint(
        "Shows what percentage of each demographic group receives a positive outcome. "
        "A <b>fair model</b> has similar approval rates for both groups. "
        "The dashed line marks the <b>4/5ths legal threshold</b>."
    )
    if bm and pm:
        groups       = ["Privileged", "Unprivileged"]
        before_rates = [bm.privileged_approval_rate, bm.unprivileged_approval_rate]
        after_rates  = [pm.privileged_approval_rate, pm.unprivileged_approval_rate]
        fig_ba = go.Figure()
        fig_ba.add_trace(go.Bar(name="Before Mitigation", x=groups, y=before_rates,
                                marker_color="#dc2626", marker_opacity=0.85,
                                text=[f"{v:.1%}" for v in before_rates], textposition="outside"))
        fig_ba.add_trace(go.Bar(name="After Mitigation", x=groups, y=after_rates,
                                marker_color="#16a34a", marker_opacity=0.85,
                                text=[f"{v:.1%}" for v in after_rates], textposition="outside"))
        fig_ba.add_hline(y=bm.privileged_approval_rate * 0.80,
                         line_dash="dash", line_color="#fbbf24", line_width=1.5,
                         annotation_text="4/5ths rule", annotation_position="top right",
                         annotation_font_color="#fbbf24")
        fig_ba.update_layout(barmode="group", yaxis_tickformat=".0%",
                             yaxis_range=[0, max(max(before_rates), max(after_rates)) * 1.25],
                             **DARK_LAYOUT)
        st.plotly_chart(fig_ba, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── All-metrics comparison ─────────────────────────────────────────────────
    section("All Metrics — Baseline vs Pre-processed vs In-processed")
    hint(
        "<b>Baseline</b> = unmodified model. <b>Pre-processed</b> = model trained on reweighed data. "
        "<b>In-processed</b> = ExponentiatedGradient with EqualizedOdds constraint (if enabled)."
    )
    if result.comparison_df is not None:
        st.dataframe(result.comparison_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Debiased Dataset Download ──────────────────────────────────────────────
    section("Download Debiased Dataset (CSV)")
    hint(
        "Exports the preprocessed dataset with a <b>sample_weight</b> column added by the Reweighing algorithm. "
        "Pass the <code>sample_weight</code> column to your own model's <code>fit()</code> call to train "
        "with fairness-aware sample weighting — compatible with any sklearn-API model."
    )
    if result.reweigh_result is not None and result._df_clean is not None:
        train_idx = result.baseline_train.X_train.index
        df_out = result._df_clean.loc[train_idx].copy()
        df_out["sample_weight"] = result.reweigh_result.weights
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Reweighed Dataset (.csv)",
            data=csv_bytes,
            file_name=f"{result.dataset_name.replace(' ', '_')}_debiased.csv",
            mime="text/csv",
            use_container_width=True,
            help="Dataset with sample_weight column. Use it to train fairness-aware models externally.",
        )
        st.caption(f"{len(df_out):,} rows (training split) × {len(df_out.columns)} columns (original + sample_weight). "
                   "Protected attributes retained for auditability.")
    else:
        st.info("Run mitigation above to generate the debiased dataset.")

    st.divider()

    # ── SHAP Importance ────────────────────────────────────────────────────────
    section("SHAP Feature Importance — What Is the Model Actually Using?")
    hint(
        "<b>SHAP (SHapley Additive exPlanations)</b> uses game theory to quantify each feature's contribution. "
        "The bar chart shows the <b>mean absolute SHAP value</b> — how much each feature moves the prediction on average. "
        "<b>If a protected attribute appears near the top, bias leakage is confirmed.</b>"
    )
    # V2: use pre-computed SHAP from pipeline if available
    _pipeline_shap = getattr(result, "shap_importance", None)
    try:
        from src.explainability.shap_explainer import (
            build_explainer, compute_global_shap, global_feature_importance, protected_attr_in_top_n
        )
        model      = result.baseline_train.model
        X_test     = result.baseline_train.X_test
        feat_names = result.baseline_train.feature_names

        if "shap_importance" not in st.session_state:
            if _pipeline_shap is not None:
                st.session_state["shap_importance"] = _pipeline_shap
                st.session_state["shap_explainer"]  = getattr(result, "shap_explainer", None)
                st.session_state["feat_names"]      = feat_names
            else:
                with st.spinner("Computing SHAP values (one-time, ~15 seconds)..."):
                    exp = build_explainer(model, X_test)
                    sv, fnames = compute_global_shap(exp, X_test, max_samples=300)
                    st.session_state["shap_importance"] = global_feature_importance(sv, fnames)
                    st.session_state["shap_explainer"]  = exp
                    st.session_state["feat_names"]      = fnames

        imp = st.session_state["shap_importance"]
        prot_cols = result.config.protected_attrs + [f"{a}_binary" for a in result.config.protected_attrs]
        in_top5   = protected_attr_in_top_n(imp, prot_cols)

        if in_top5:
            st.warning("⚠️ A protected attribute is in the top-5 SHAP features — the model is using demographics to make decisions.")
        else:
            st.success("✅ No protected attribute in the top-5 SHAP features. The model relies on legitimate predictors.")

        colors = ["#dc2626" if f in prot_cols else "#7c3aed" for f in imp.head(12)["feature"]]
        fig_shap = go.Figure(go.Bar(
            x=imp.head(12)["importance"], y=imp.head(12)["feature"],
            orientation="h", marker_color=colors,
            text=[f"{v:.4f}" for v in imp.head(12)["importance"]], textposition="outside",
        ))
        fig_shap.update_yaxes(autorange="reversed")
        fig_shap.update_layout(
            title="Top 12 Global SHAP Features (red = protected attribute)",
            xaxis_title="Mean |SHAP value|", **DARK_LAYOUT, height=380,
        )
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

        # ── V2: Group SHAP Comparison ──────────────────────────────────────────
        st.divider()
        section("V2: Group SHAP Comparison — Privileged vs Unprivileged")
        hint(
            "Same model, different populations. <b>Do different demographic groups rely on different features?</b> "
            "If the top features differ between privileged and unprivileged groups, the model has learned "
            "group-specific decision patterns — a sign of structural bias even when DI looks acceptable."
        )
        try:
            _exp = st.session_state.get("shap_explainer")
            if _exp is None and result.baseline_train:
                _exp = build_explainer(model, X_test)
                st.session_state["shap_explainer"] = _exp

            if _exp is not None:
                _primary_col = primary if primary in X_test.columns else f"{primary}_binary"
                _priv_val    = result.config.privileged_values.get(primary)
                _priv_mask   = (X_test[_primary_col].astype(str) == str(_priv_val)).values if _primary_col in X_test.columns else None

                if _priv_mask is not None and _priv_mask.sum() >= 10 and (~_priv_mask).sum() >= 10:
                    _X_priv   = X_test[_priv_mask]
                    _X_unpriv = X_test[~_priv_mask]
                    _sv_p, _fn = compute_global_shap(_exp, _X_priv,   max_samples=min(150, len(_X_priv)))
                    _sv_u, _   = compute_global_shap(_exp, _X_unpriv, max_samples=min(150, len(_X_unpriv)))
                    _imp_p = global_feature_importance(_sv_p, _fn).head(10).set_index("feature")["importance"]
                    _imp_u = global_feature_importance(_sv_u, _fn).head(10).set_index("feature")["importance"]
                    _all_feats = sorted(set(_imp_p.index) | set(_imp_u.index),
                                        key=lambda f: -(_imp_p.get(f, 0) + _imp_u.get(f, 0)))[:10]
                    fig_gcmp = go.Figure()
                    fig_gcmp.add_trace(go.Bar(
                        name="Privileged", x=_all_feats,
                        y=[float(_imp_p.get(f, 0)) for f in _all_feats],
                        marker_color="#7c3aed", opacity=0.85,
                    ))
                    fig_gcmp.add_trace(go.Bar(
                        name="Unprivileged", x=_all_feats,
                        y=[float(_imp_u.get(f, 0)) for f in _all_feats],
                        marker_color="#f59e0b", opacity=0.85,
                    ))
                    fig_gcmp.update_layout(
                        barmode="group",
                        title=f"SHAP Feature Importance by Group ({_primary_col})",
                        yaxis_title="Mean |SHAP|", **DARK_LAYOUT, height=380,
                    )
                    st.plotly_chart(fig_gcmp, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption("Not enough samples in one group for SHAP group comparison.")
        except Exception as _ge:
            st.caption(f"Group SHAP comparison unavailable: {_ge}")

    except Exception as e:
        st.warning(f"SHAP computation unavailable: {e}")

    st.divider()

    # ── Multi-Model Comparison ─────────────────────────────────────────────────
    section("🔬 Compare All Base Models (RF vs LR vs XGB)")
    hint(
        "Trains Random Forest, Logistic Regression, and XGBoost on the same split and "
        "reports accuracy + fairness metrics side-by-side. Use this to verify that the "
        "detected bias is <b>algorithm-independent</b> — a hallmark of dataset-level bias."
    )
    if st.button("Run Model Comparison (~30s)", key="model_cmp_btn"):
        with st.spinner("Training RF, LR, and XGBoost baselines..."):
            try:
                from src.ml_pipeline.model_comparison import compare_base_models
                cmp_df = compare_base_models(
                    result._X, result._y, result._df_clean, result.config
                )
                st.session_state["model_cmp_df"] = cmp_df
                st.success("Comparison complete.")
            except Exception as e:
                st.error(f"Model comparison error: {e}")
                import traceback; st.code(traceback.format_exc())

    if "model_cmp_df" in st.session_state:
        cdf = st.session_state["model_cmp_df"]
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        fig_cmp = go.Figure()
        for col, color in [("Accuracy","#7c3aed"), ("DI","#f59e0b"), ("F1","#22d3ee")]:
            if col in cdf.columns:
                fig_cmp.add_trace(go.Bar(
                    name=col, x=cdf["Model"], y=cdf[col],
                    marker_color=color, text=[f"{v:.3f}" for v in cdf[col]],
                    textposition="outside",
                ))
        fig_cmp.update_layout(
            barmode="group", title="Performance vs Fairness across Base Models",
            yaxis_title="Score", **DARK_LAYOUT, height=360,
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div style="color:#6b7280;text-align:center;padding:1rem">Click above to compare models.</div>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Pareto Frontier ────────────────────────────────────────────────────────
    section("⚖️ Fairness-Accuracy Pareto Frontier")
    hint(
        "We train 6 models with different fairness constraint weights (0 = pure accuracy, 1 = pure fairness) "
        "and plot them all. <b>You choose your operating point.</b> "
        "The gold star marks the 'knee' — maximum fairness gain for minimum accuracy loss. "
        "This is the correct scientific framing: fairness is a tradeoff, not a binary pass/fail."
    )
    if st.button("Compute Pareto Frontier (~60s)", key="pareto_btn"):
        with st.spinner("Sweeping fairness-accuracy tradeoff across 6 constraint weights..."):
            try:
                from src.ml_pipeline.pareto import sweep_fairness_frontier, pareto_to_dataframe
                X_tr = result.preprocessed_train.X_train.drop(
                    columns=result.config.protected_attrs + [f"{a}_binary" for a in result.config.protected_attrs],
                    errors="ignore")
                y_tr = result.preprocessed_train.y_train
                X_te = result.preprocessed_train.X_test.drop(
                    columns=result.config.protected_attrs + [f"{a}_binary" for a in result.config.protected_attrs],
                    errors="ignore")
                y_te = result.preprocessed_train.y_test
                primary_col = primary if primary in result.preprocessed_train.X_train.columns else f"{primary}_binary"
                S_tr = result.preprocessed_train.X_train.get(primary_col, X_tr.iloc[:,0]*0)
                S_te = result.preprocessed_train.X_test.get(primary_col,  X_te.iloc[:,0]*0)
                pts = sweep_fairness_frontier(X_tr, y_tr, S_tr, X_te, y_te, S_te,
                                              result._df_clean, result.config, steps=6)
                st.session_state["pareto_points"] = pareto_to_dataframe(pts)
                st.success("Frontier computed.")
            except Exception as e:
                st.error(f"Pareto sweep error: {e}")

    if "pareto_points" in st.session_state:
        pdf     = st.session_state["pareto_points"]
        optimal = pdf[pdf["is_optimal"]]
        fig_par = go.Figure()
        fig_par.add_trace(go.Scatter(
            x=pdf["eod"], y=pdf["accuracy"], mode="markers+lines",
            marker=dict(size=10, color="#7c3aed", line=dict(color="#a78bfa", width=1)),
            line=dict(color="#7c3aed", dash="dot", width=1.5), name="Trained models",
            hovertemplate="<b>weight=%{text}</b><br>EOD=%{x:.3f}<br>Accuracy=%{y:.1%}",
            text=[f"{w:.2f}" for w in pdf["constraint_weight"]],
        ))
        if not optimal.empty:
            fig_par.add_trace(go.Scatter(
                x=optimal["eod"], y=optimal["accuracy"], mode="markers+text",
                marker=dict(size=22, color="#fbbf24", symbol="star", line=dict(color="#fff", width=1)),
                text=["Optimal"], textposition="top center", name="Recommended operating point",
            ))
        fig_par.update_layout(
            xaxis_title="Equalized Odds Difference (lower = fairer)",
            yaxis_title="Accuracy", yaxis_tickformat=".1%",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"), **DARK_LAYOUT,
        )
        st.plotly_chart(fig_par, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(pdf.rename(columns={"constraint_weight":"Weight","accuracy":"Accuracy",
                                          "eod":"Equalized Odds Diff","disparate_impact":"DI",
                                          "is_optimal":"Optimal"}),
                     use_container_width=True, hide_index=True)
    else:
        st.markdown('<div style="color:#6b7280;text-align:center;padding:1rem">Click above to compute the frontier.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INTERSECTIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🧩 Intersectional Bias Analysis")
    hint(
        "<b>What is intersectionality?</b> (Kimberlé Crenshaw, 1989) A Black woman may face discrimination "
        "invisible when you look at race <i>or</i> sex alone — because the disadvantages compound. "
        "Standard bias tools check one attribute at a time. <b>Aequitas Prime checks every combination automatically.</b> "
        "The EU AI Act (Recital 44) explicitly recognises intersectional discrimination. "
        "Groups with fewer than 30 samples are excluded for statistical validity."
    )

    if "pipeline_result" not in st.session_state:
        st.info("Complete the Bias X-Ray first (Tab 1).")
        st.stop()

    result = st.session_state["pipeline_result"]
    from src.bias_engine.intersectional import (
        compute_intersectional_metrics, get_heatmap_data, _pick_intersectional_attrs
    )

    df_test = result._df_clean.loc[result.baseline_train.X_test.index]
    y_pred  = result.baseline_train.y_pred

    # ── Attribute selector ─────────────────────────────────────────────────────
    # Auto-pick safe attrs (prevents combinatorial explosion), let user override
    _auto_attrs = _pick_intersectional_attrs(df_test, result.config)
    _all_eligible = [
        a for a in result.config.protected_attrs
        if a in df_test.columns and df_test[a].nunique() >= 2
    ]
    hint(
        "Intersectional analysis cross-tabulates every combination of the selected attributes. "
        "To prevent slowdowns, attributes with >10 unique values are excluded by default. "
        "<b>Select 2–3 attributes</b> for best results."
    )
    selected_inter_attrs = st.multiselect(
        "Attributes to cross-tabulate:",
        options=_all_eligible,
        default=_auto_attrs,
        key="inter_attrs_selector",
        help="Auto-selected: the most biased low-cardinality attributes. Adding high-cardinality columns will slow computation significantly.",
    )
    if not selected_inter_attrs:
        st.warning("Select at least one attribute above.")
        st.stop()

    # Warn if the combination count will be large
    _combo_count = 1
    for _a in selected_inter_attrs:
        _combo_count *= df_test[_a].nunique()
    if _combo_count > 10_000:
        st.error(
            f"⚠️ {_combo_count:,} combinations — this will be very slow. "
            "Please reduce the number of selected attributes or choose lower-cardinality ones."
        )
        st.stop()
    elif _combo_count > 1_000:
        st.warning(f"{_combo_count:,} combinations to check — may take a few seconds.")

    # Cache key: pipeline run + selected attrs
    _cache_key = f"inter_{id(result)}_{'_'.join(sorted(selected_inter_attrs))}"
    if st.session_state.get("_inter_cache_key") != _cache_key:
        with st.spinner(f"Computing intersectional metrics across {_combo_count:,} combinations..."):
            inter = compute_intersectional_metrics(
                df_test, result.config, y_pred=y_pred, attrs=selected_inter_attrs
            )
        st.session_state["inter_results"]    = inter
        st.session_state["_inter_cache_key"] = _cache_key
    else:
        inter = st.session_state["inter_results"]

    if not inter:
        st.warning("Not enough data for intersectional analysis (need ≥5 samples per intersectional group).")
        st.stop()

    # V2: count small-sample groups (5-29 rows)
    _n_small = sum(1 for r in inter if getattr(r, "small_sample_warning", False))
    if _n_small > 0:
        st.info(
            f"**{_n_small} small group(s) included with caution** — sample size 5–29. "
            "Wilson 95% confidence intervals are shown. Interpret these results carefully."
        )

    n_crit = sum(1 for r in inter if r.severity == "CRITICAL")
    n_high = sum(1 for r in inter if r.severity == "HIGH")
    worst  = inter[0]

    # ── Intersectionality Index ────────────────────────────────────────────────
    section("🏆 Intersectionality Bias Index (IBI)")
    hint(
        "The <b>Intersectionality Bias Index (IBI)</b> is a single scalar that quantifies the degree of "
        "compounding disadvantage across all intersectional demographic groups. "
        "It is computed as the <b>weighted mean deviation from perfect DI (1.0)</b>, weighted by group size. "
        "<b>0 = perfectly fair. 100 = maximum measurable intersectional bias.</b>"
    )

    total_weight = sum(r.group_size for r in inter if not np.isnan(r.disparate_impact))
    weighted_di  = (sum(r.disparate_impact * r.group_size for r in inter
                        if not np.isnan(r.disparate_impact)) / total_weight) if total_weight > 0 else 1.0
    ibi = round((1 - min(weighted_di, 1.0)) * 100, 1)

    ibi_color = "#dc2626" if ibi > 30 else ("#d97706" if ibi > 10 else "#16a34a")
    ibi_label = "CRITICAL" if ibi > 30 else ("MODERATE" if ibi > 10 else "LOW")

    ib1, ib2, ib3, ib4 = st.columns(4)
    ib1.metric("Intersectionality Bias Index", f"{ibi:.1f} / 100",
               help="IBI = (1 − weighted_mean_DI) × 100. Higher = more intersectional bias.")
    ib2.metric("Severity",          ibi_label,
               help="LOW (<10), MODERATE (10–30), CRITICAL (>30).")
    ib3.metric("Groups Analysed",   len(inter),
               help="Number of intersectional demographic combinations with ≥30 samples.")
    ib4.metric("CRITICAL Groups",   n_crit,
               help="Groups with Disparate Impact < 0.60 — severe compounding discrimination.")

    # Radar chart for top worst groups
    top_groups = inter[:min(6, len(inter))]
    if len(top_groups) >= 3:
        section("Radar Chart — Top Worst Intersectional Groups")
        hint(
            "Each axis shows a different fairness dimension for the worst-affected demographic combinations. "
            "A perfectly fair model would fill the entire chart to the outer edge."
        )
        radar_cats = ["Disparate Impact", "Approval Rate", "Inverse EOpD"]
        fig_radar = go.Figure()
        for r in top_groups:
            eopd_inv = max(0.0, 1.0 - abs(r.equal_opportunity_diff)) if not np.isnan(r.equal_opportunity_diff) else 0.5
            values   = [
                min(r.disparate_impact, 1.0) if not np.isnan(r.disparate_impact) else 0.0,
                min(r.approval_rate * 2, 1.0),   # scale so 50% approval = edge
                eopd_inv,
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=radar_cats + [radar_cats[0]],
                fill="toself", opacity=0.35,
                name=r.group_label[:30],
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#8b949e", size=10),
                                gridcolor="#30363d"),
                angularaxis=dict(tickfont=dict(color="#c9d1d9", size=11), gridcolor="#30363d"),
            ),
            paper_bgcolor="#0d1117", font_color="#e6edf3",
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
            height=420, margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        hint(
            "Outer edge = ideal (DI=1.0, high approval, zero EOpD gap). "
            "Groups collapsed toward the centre face severe compounding discrimination. "
            "Use this chart in presentations — it communicates intersectional inequality visually."
        )

    st.divider()

    # ── Heatmap ────────────────────────────────────────────────────────────────
    attrs = result.config.protected_attrs
    if len(attrs) >= 2:
        section(f"DI Heatmap: {attrs[0].title()} × {attrs[1].title()}")
        hm = get_heatmap_data(inter, row_attr=attrs[0], col_attr=attrs[1])
        if not hm.empty:
            fig_hm = px.imshow(
                hm, text_auto=".2f",
                color_continuous_scale=[[0,"#7f1d1d"],[0.5,"#d97706"],[0.8,"#16a34a"],[1,"#166534"]],
                zmin=0.2, zmax=1.1,
                title=f"Disparate Impact: {attrs[0].title()} × {attrs[1].title()}",
                labels=dict(color="DI"),
            )
            fig_hm.update_traces(textfont_size=13)
            fig_hm.update_layout(**DARK_LAYOUT, height=380)
            st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

    st.divider()
    section("Worst-Case Groups — Ranked by Disparate Impact")
    hint(
        "<b>Approval Rate</b>: % of this group receiving a positive outcome. "
        "<b>EOppD</b>: Equal Opportunity Difference — gap in True Positive Rate vs privileged baseline. "
        "<b>IBI Contribution</b>: this group's share of the total Intersectionality Bias Index. "
        "<b>Severity</b>: CRITICAL (DI &lt;0.60) / HIGH (0.60–0.80) / MEDIUM (0.80–0.90) / OK (≥0.90)."
    )
    rows = []
    for r in inter:
        ibi_contrib = round((1 - min(r.disparate_impact, 1.0)) * r.group_size / max(total_weight, 1) * 100, 2) if not np.isnan(r.disparate_impact) else 0
        _ci_lo = getattr(r, "approval_rate_ci_lower", float("nan"))
        _ci_hi = getattr(r, "approval_rate_ci_upper", float("nan"))
        _small = getattr(r, "small_sample_warning", False)
        _ci_str = f"[{_ci_lo:.2f}, {_ci_hi:.2f}]" if not (np.isnan(_ci_lo) or np.isnan(_ci_hi)) else "—"
        rows.append({
            "Group":              r.group_label,
            "Size":               r.group_size,
            "Approval Rate":      f"{r.approval_rate:.1%}",
            "95% CI (Approval)":  _ci_str if _small else "—",
            "⚠ Small":            "⚠" if _small else "",
            "Disparate Impact":   round(r.disparate_impact, 3),
            "EOppD":              round(r.equal_opportunity_diff, 3) if not np.isnan(r.equal_opportunity_diff) else "—",
            "IBI Contribution":   f"{ibi_contrib:.2f}%",
            "Severity":           r.severity,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={
                     "95% CI (Approval)": st.column_config.TextColumn(width="medium",
                                                                        help="Wilson 95% CI for approval rate (shown for groups with n=5–29 only)"),
                     "⚠ Small": st.column_config.TextColumn(width="small",
                                                              help="Group has 5–29 samples — treat with caution"),
                 })

    if worst.severity in ("CRITICAL", "HIGH"):
        st.error(
            f"**Most discriminated group: {worst.group_label}** — "
            f"DI = {worst.disparate_impact:.3f} [{worst.severity}] | "
            f"Approval rate: {worst.approval_rate:.1%}. "
            "A standard single-attribute audit would have missed this compounding effect."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4b — CAUSAL FAIRNESS (V2 NEW)
# ══════════════════════════════════════════════════════════════════════════════
with tab4b:
    st.markdown("## 🔬 Causal Fairness — Direct vs Indirect Discrimination")
    hint(
        "<b>Why causal analysis?</b> Standard fairness metrics tell you <i>that</i> a disparity exists, "
        "not <i>why</i>. Causal mediation analysis decomposes the total effect of a protected attribute "
        "into <b>direct discrimination</b> (A→Y) and <b>indirect discrimination</b> (A→M→Y, through mediators "
        "like occupation or zip code). The EU AI Act and ECOA increasingly expect this level of root-cause analysis."
    )

    if "pipeline_result" not in st.session_state:
        st.info("Complete the Bias X-Ray first (Tab 1).")
        st.stop()

    result  = st.session_state["pipeline_result"]
    primary = result.config.primary_protected_attr()

    # ── Configuration ──────────────────────────────────────────────────────────
    section("⚙️ Configure Causal Analysis")
    _cf_attr = st.selectbox(
        "Protected attribute to analyse causally:",
        options=result.config.protected_attrs,
        index=0,
        key="causal_attr",
        help="The protected attribute whose effect on the outcome you want to decompose."
    )

    if result._df_clean is not None:
        _candidate_cols = [c for c in result._df_clean.columns
                           if c not in result.config.protected_attrs
                           and c != result.config.target_col]
        _auto_meds = [c for c in _candidate_cols
                      if result._df_clean[c].dtype != object
                      and result._df_clean[c].nunique() <= 20][:6]
    else:
        _candidate_cols, _auto_meds = [], []

    _sel_meds = st.multiselect(
        "Mediator candidates (features that may transmit bias):",
        options=_candidate_cols,
        default=_auto_meds[:4],
        key="causal_meds",
        help="Features that plausibly lie on the causal path from the protected attribute to the outcome. E.g. occupation → income."
    )

    _confounders = st.multiselect(
        "Confounders to control for (optional):",
        options=[c for c in _candidate_cols if c not in _sel_meds],
        default=[],
        key="causal_confounders",
        help="Variables that affect both the protected attribute and the outcome but are NOT mediators."
    )

    st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
    _run_causal = st.button("🔬 Run Causal Mediation Analysis", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if _run_causal:
        with st.spinner("Running Baron-Kenny mediation analysis..."):
            try:
                from src.bias_engine.causal_fairness import run_causal_analysis
                _causal_report = run_causal_analysis(
                    result._df_clean, result.config,
                    attr=_cf_attr,
                    mediators=_sel_meds if _sel_meds else None,
                    confounders=_confounders if _confounders else None,
                )
                st.session_state["causal_report"] = _causal_report
                st.success("Causal analysis complete!")
            except Exception as _ce:
                st.error(f"Causal analysis error: {_ce}")
                import traceback; st.code(traceback.format_exc())

    if "causal_report" in st.session_state:
        _cr = st.session_state["causal_report"]

        # ── Summary ────────────────────────────────────────────────────────────
        section("📋 Summary")
        st.markdown(
            f'<div class="hint-box">{_cr.summary}</div>',
            unsafe_allow_html=True
        )

        if not _cr.paths:
            st.info("No mediation paths could be estimated with the selected configuration.")
            st.stop()

        # ── Dominant path metrics ───────────────────────────────────────────────
        if _cr.dominant_path:
            _dp = _cr.dominant_path
            _dm1, _dm2, _dm3, _dm4 = st.columns(4)
            _dm1.metric("Total Effect",    f"{_dp.total_effect:.3f}",
                        help="Overall effect of the protected attribute on the outcome (ignoring mediators).")
            _dm2.metric("Direct Effect",   f"{_dp.direct_effect:.3f}",
                        help="Effect of the protected attribute that is NOT explained by mediators. Direct discrimination.")
            _dm3.metric("Indirect Effect", f"{_dp.indirect_effect:.3f}",
                        help=f"Effect transmitted through '{_dp.mediator}'. Indirect/structural discrimination.")
            _pct = _dp.percent_mediated
            _dm4.metric("% Mediated",
                        f"{abs(_pct):.1f}%" if not np.isnan(_pct) else "—",
                        help="Fraction of the total effect that runs through the mediator pathway.")

        # ── Waterfall chart: direct vs indirect per mediator ───────────────────
        section("Effect Decomposition per Mediator")
        hint(
            "Each row is a different mediator. <b>Purple = direct effect</b> (A→Y, not through mediator). "
            "<b>Orange = indirect effect</b> (A→M→Y, through mediator). "
            "A large orange bar means the disparity is transmitted structurally — "
            "fixing the mediator (e.g. changing occupational sorting) would reduce the bias."
        )
        from src.bias_engine.causal_fairness import causal_waterfall_data
        _wf_df = causal_waterfall_data(_cr)
        if not _wf_df.empty:
            _fig_wf = go.Figure()
            _fig_wf.add_trace(go.Bar(
                name="Direct Effect",
                x=_wf_df["mediator"],
                y=_wf_df["direct_effect"].round(4),
                marker_color="#7c3aed", opacity=0.85,
                text=[f"{v:.3f}" for v in _wf_df["direct_effect"]], textposition="outside",
            ))
            _fig_wf.add_trace(go.Bar(
                name="Indirect Effect (via mediator)",
                x=_wf_df["mediator"],
                y=_wf_df["indirect_effect"].round(4),
                marker_color="#f59e0b", opacity=0.85,
                text=[f"{v:.3f}" for v in _wf_df["indirect_effect"]], textposition="outside",
            ))
            _fig_wf.update_layout(
                barmode="group", title="Direct vs Indirect Effect Decomposition by Mediator",
                yaxis_title="Effect Size (OLS coefficient)", **DARK_LAYOUT, height=380,
            )
            st.plotly_chart(_fig_wf, use_container_width=True, config={"displayModeBar": False})

        # ── Sobel test table ────────────────────────────────────────────────────
        section("Sobel Test Results — Statistical Significance of Indirect Paths")
        hint(
            "The Sobel test assesses whether the indirect effect (A→M→Y) is statistically significant. "
            "<b>p &lt; 0.05</b> = the indirect path is real, not sampling noise. "
            "High % mediated + low p-value = strong evidence of structural/indirect discrimination."
        )
        _sobel_rows = []
        for _p in _cr.paths:
            _sobel_rows.append({
                "Mediator":        _p.mediator,
                "Total Effect":    round(_p.total_effect, 4),
                "Direct Effect":   round(_p.direct_effect, 4),
                "Indirect Effect": round(_p.indirect_effect, 4),
                "Sobel Z":         round(_p.sobel_z, 3) if not np.isnan(_p.sobel_z) else "—",
                "Sobel p":         round(_p.sobel_p, 4) if not np.isnan(_p.sobel_p) else "—",
                "% Mediated":      f"{abs(_p.percent_mediated):.1f}%" if not np.isnan(_p.percent_mediated) else "—",
                "n":               _p.n_samples,
                "Significant":     "✅" if (not np.isnan(_p.sobel_p) and _p.sobel_p < 0.05) else "—",
            })
        st.dataframe(pd.DataFrame(_sobel_rows), use_container_width=True, hide_index=True)

        # ── Per-path interpretation ─────────────────────────────────────────────
        with st.expander("View plain-English interpretation per mediator"):
            for _p in _cr.paths:
                st.markdown(f"**{_p.mediator}**: {_p.interpretation}")

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#6b7280">
          <div style="font-size:2.5rem">🔬</div>
          <div style="font-size:1rem;margin-top:0.5rem">
            Configure mediators above and click <b style="color:#a78bfa">Run Causal Mediation Analysis</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COUNTERFACTUAL PANEL
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔀 Counterfactual Panel — Dataset-Level Bias Evidence")
    hint(
        "This panel <b>proves that bias exists</b> in the trained model — not just in the dataset statistics. "
        "For each test row, we ask: <i>'What would the model predict if this person's demographic were different?'</i> "
        "If the decision flips, that prediction is <b>demographically unstable</b> — direct evidence of individual-level discrimination. "
        "Run the batch check to see what percentage of the model's predictions are affected."
    )

    if "pipeline_result" not in st.session_state:
        st.info("Complete the Bias X-Ray first (Tab 1).")
        st.stop()

    result     = st.session_state["pipeline_result"]
    model      = result.baseline_train.model
    feat_names = result.baseline_train.feature_names
    X_test     = result.baseline_train.X_test
    primary    = result.config.primary_protected_attr()
    pos_label  = result.config.positive_label

    # Determine the primary encoded column name in X_test
    primary_feat = primary if primary in feat_names else f"{primary}_binary"

    # ── Batch Counterfactual Check ─────────────────────────────────────────────
    section("Batch Counterfactual Check — How Many Predictions Flip?")
    hint(
        "We sample up to 500 test rows and flip the primary protected attribute (e.g. Male→Female) for each. "
        "CRITICAL = the model decision changes. FLAGGED = same decision but confidence drops ≥15%. CLEAR = no effect. "
        "<b>A high CRITICAL rate is smoking-gun evidence of systematic individual-level discrimination.</b>"
    )

    n_sample = min(500, len(X_test))

    # V2: label decode map for string labels in CF display
    _label_decode = getattr(result, "label_decode", {})

    if st.button(f"▶ Run Batch Check ({n_sample} rows)", key="batch_cf_btn"):
        with st.spinner(f"Running counterfactual check on {n_sample} rows..."):
            try:
                from src.bias_engine.counterfactual import run_counterfactual_check
                sample_idx = X_test.sample(n=n_sample, random_state=42).index
                X_sample   = X_test.loc[sample_idx]

                cf_results = []
                for idx, row in X_sample.iterrows():
                    input_dict = row.to_dict()
                    cf = run_counterfactual_check(
                        model, input_dict, result.config, feat_names,
                        attr=primary_feat, label_decode=_label_decode,
                    )
                    _orig_str = getattr(cf, "original_label_str", str(cf.original_value))
                    _flip_str = getattr(cf, "flipped_label_str",  str(cf.flipped_value))
                    cf_results.append({
                        "row_idx":           idx,
                        "risk_level":        cf.risk_level,
                        "original_decision": cf.original_decision,
                        "cf_decision":       cf.counterfactual_decision,
                        "flip":              f"{_orig_str} → {_flip_str}",
                        "confidence_delta":  round(cf.confidence_delta, 3),
                        "is_fair":           cf.is_fair,
                    })

                cf_df = pd.DataFrame(cf_results)
                st.session_state["batch_cf_results"] = cf_df
                st.success(f"Batch check complete on {n_sample} rows.")
            except Exception as e:
                st.error(f"Batch check error: {e}")
                import traceback; st.code(traceback.format_exc())

    if "batch_cf_results" in st.session_state:
        cf_df = st.session_state["batch_cf_results"]
        n_total    = len(cf_df)
        n_critical = (cf_df["risk_level"] == "CRITICAL").sum()
        n_flagged  = (cf_df["risk_level"] == "FLAGGED").sum()
        n_clear    = (cf_df["risk_level"] == "CLEAR").sum()
        pct_crit   = n_critical / n_total

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows Checked",      f"{n_total:,}",
                  help="Total number of test rows sampled for counterfactual analysis.")
        m2.metric("CRITICAL (flip)",   f"{n_critical} ({pct_crit:.1%})",
                  help="Rows where the model's decision changes when the protected attribute is flipped. Direct evidence of discrimination.",
                  delta=f"-{pct_crit:.1%} fairness" if pct_crit > 0.1 else None,
                  delta_color="inverse")
        m3.metric("FLAGGED (unstable)", f"{n_flagged} ({n_flagged/n_total:.1%})",
                  help="Same decision but confidence drops ≥15% on demographic swap. Borderline cases needing review.")
        m4.metric("CLEAR (stable)",    f"{n_clear} ({n_clear/n_total:.1%})",
                  help="Decision is identical regardless of demographic. No bias detected for these rows.")

        if pct_crit > 0.20:
            st.error(f"🚨 **{pct_crit:.0%} of predictions flip** when {primary} is changed — CRITICAL systematic bias. "
                     "This model is making decisions based on demographic identity, not merit.")
        elif pct_crit > 0.05:
            st.warning(f"⚠️ **{pct_crit:.0%} of predictions flip** — significant demographic dependency detected.")
        else:
            st.success(f"✅ Only {pct_crit:.0%} of predictions flip — the model is largely demographically stable.")

        # Bar chart
        fig_bar = go.Figure(go.Bar(
            x=["CRITICAL\n(Decision Flips)", "FLAGGED\n(Confidence Drops)", "CLEAR\n(Stable)"],
            y=[n_critical, n_flagged, n_clear],
            marker_color=["#dc2626", "#d97706", "#16a34a"],
            text=[f"{v} ({v/n_total:.0%})" for v in [n_critical, n_flagged, n_clear]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title=f"Counterfactual Risk Distribution — {n_total} test rows",
            yaxis_title="Number of rows", **DARK_LAYOUT, height=340,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # Table of critical rows
        critical_rows = cf_df[cf_df["risk_level"] == "CRITICAL"].head(20)
        if not critical_rows.empty:
            _show_cols = [c for c in ["row_idx","flip","original_decision","cf_decision","confidence_delta"] if c in cf_df.columns]
            with st.expander(f"View CRITICAL rows (first 20 of {n_critical})"):
                st.dataframe(critical_rows[_show_cols].rename(
                    columns={"row_idx":"Row","flip":"Demographic Flip","original_decision":"Original",
                             "cf_decision":"After Flip","confidence_delta":"Conf Delta"}
                ), use_container_width=True, hide_index=True)

    st.divider()

    # ── Individual Row Explorer ────────────────────────────────────────────────
    section("Individual Row Explorer — Pick Any Row")
    hint(
        "Select a test-set row and instantly see how the model's prediction changes when the protected "
        "attribute is flipped. This is the <b>Counterfactual Ghost</b> applied to real dataset rows — "
        "no manual form filling needed."
    )

    n_display = min(200, len(X_test))
    row_indices = X_test.sample(n=n_display, random_state=0).index.tolist()

    sel_idx = st.selectbox(
        "Select a test row (by dataset index):",
        options=row_indices,
        format_func=lambda i: f"Row {i}",
        key="cf_panel_row_sel",
        help="Pick any row from the test set. The counterfactual ghost will flip the primary protected attribute."
    )

    if sel_idx is not None:
        row_data   = X_test.loc[sel_idx]
        input_dict = row_data.to_dict()

        # Show row features
        with st.expander("Row features", expanded=False):
            feat_df = pd.DataFrame({"Feature": list(input_dict.keys()), "Value": list(input_dict.values())})
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        from src.bias_engine.counterfactual import run_counterfactual_check

        row_df = pd.DataFrame([{f: input_dict.get(f, 0) for f in feat_names}])
        if hasattr(model, "predict_proba"):
            proba   = model.predict_proba(row_df)[0]
            pred    = int(np.argmax(proba))
            conf    = float(proba[pred])
        else:
            pred = int(model.predict(row_df)[0])
            conf = 1.0

        cf = run_counterfactual_check(model, input_dict, result.config, feat_names,
                                       attr=primary_feat, label_decode=_label_decode)

        # V2: use string labels for demographic flip display
        _orig_str = getattr(cf, "original_label_str", str(cf.original_value))
        _flip_str = getattr(cf, "flipped_label_str",  str(cf.flipped_value))

        orig_label = "Approved" if pred == 1 else "Denied"
        orig_color = "#16a34a"  if pred == 1 else "#dc2626"
        risk_colors = {"CLEAR": "#16a34a", "FLAGGED": "#d97706", "CRITICAL": "#dc2626"}
        risk_icons  = {"CLEAR": "✅", "FLAGGED": "⚠️", "CRITICAL": "🚨"}
        rc = risk_colors.get(cf.risk_level, "#6b7280")
        ri = risk_icons.get(cf.risk_level, "❓")

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown(f"""
            <div style="background:#161b22;border-radius:12px;padding:1.2rem;border:1px solid #30363d;text-align:center">
              <div style="color:#8b949e;font-size:0.85rem;margin-bottom:0.3rem">ORIGINAL PREDICTION</div>
              <div style="font-size:2rem;font-weight:700;color:{orig_color}">{orig_label}</div>
              <div style="color:#8b949e;margin-top:0.4rem">{primary_feat} = <b style="color:#e6edf3">{_orig_str}</b></div>
              <div style="color:#8b949e">Confidence: <b style="color:#e6edf3">{conf:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            flip_msg = (f"Decision <b>FLIPS to {cf.counterfactual_decision}</b> when {primary_feat} changes."
                        if not cf.is_fair else
                        f"Decision stays <b>{cf.original_decision}</b>. Confidence delta: {cf.confidence_delta:.0%}.")
            st.markdown(f"""
            <div style="background:#161b22;border-radius:12px;padding:1.2rem;border:2px solid {rc};text-align:center">
              <div style="color:#8b949e;font-size:0.85rem;margin-bottom:0.3rem">GHOST CHECK</div>
              <div style="font-size:2rem;font-weight:700;color:{rc}">{ri} {cf.risk_level}</div>
              <div style="color:#c9d1d9;margin-top:0.4rem;font-size:0.88rem">{flip_msg}</div>
              <div style="color:#8b949e;font-size:0.8rem;margin-top:0.3rem">Flipped {primary_feat}: <b>{_orig_str} → {_flip_str}</b></div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — COUNTERFACTUAL GHOST (Live Inference)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 👻 Counterfactual Ghost — Live Inference Firewall")
    hint(
        "<b>What is a Counterfactual Fairness Check?</b> For any prediction, we create a 'ghost' — "
        "an identical person with only the protected attribute changed (e.g. Male → Female). "
        "If the model's decision changes, the prediction is demographically unstable and likely discriminatory. "
        "<b>CLEAR</b> = stable. <b>FLAGGED</b> = same decision but confidence drops. "
        "<b>CRITICAL</b> = decision flips — direct evidence of individual-level discrimination."
    )

    if "pipeline_result" not in st.session_state:
        st.info("Complete the Bias X-Ray first (Tab 1).")
        st.stop()

    result     = st.session_state["pipeline_result"]
    model      = result.baseline_train.model
    feat_names = result.baseline_train.feature_names
    X_sample   = result.baseline_train.X_test
    primary    = result.config.primary_protected_attr()
    primary_feat = primary if primary in feat_names else f"{primary}_binary"

    section("Enter Candidate Details")
    hint(
        "Fill in the candidate's features below. The protected attribute is <b>intentionally excluded</b> "
        "from this form — the Ghost tests all demographic values automatically."
    )

    display_feats = [f for f in feat_names if f not in [primary_feat, f"{primary}_binary"]]
    form_cols = st.columns(3)
    input_vals: dict = {}

    for i, feat in enumerate(display_feats):
        col = form_cols[i % 3]
        sample_val = float(X_sample[feat].median())
        fmin = float(X_sample[feat].min())
        fmax = float(X_sample[feat].max())
        # Auto-detect step: integer column → step 1, float column → step 0.01
        is_int_col = (X_sample[feat] == X_sample[feat].round()).all() and fmax < 1e6
        step = 1.0 if is_int_col else 0.01
        with col:
            input_vals[feat] = st.number_input(
                feat, value=float(sample_val),
                min_value=fmin, max_value=fmax, step=step,
                key=f"ghost_inp_{feat}",
                help=f"Feature '{feat}'. Median value pre-filled. Range: {fmin:.2g} – {fmax:.2g}."
            )

    # Include the protected attr at its median (Ghost will flip this)
    if primary_feat in feat_names:
        input_vals[primary_feat] = float(X_sample[primary_feat].median())

    st.markdown('<div class="glow-btn">', unsafe_allow_html=True)
    run_ghost = st.button("👻 Run Live Prediction + Ghost Check", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_ghost:
        from src.bias_engine.counterfactual import run_counterfactual_check
        _ghost_label_decode = getattr(result, "label_decode", {})
        row = pd.DataFrame([{f: input_vals.get(f, 0) for f in feat_names}])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row)[0]
            pred  = int(np.argmax(proba))
            conf  = float(proba[pred])
        else:
            pred = int(model.predict(row)[0])
            conf = 1.0

        decision       = "✅ Approved" if pred == 1 else "❌ Denied"
        decision_color = "#16a34a"     if pred == 1 else "#dc2626"

        cf = run_counterfactual_check(model, input_vals, result.config, feat_names,
                                       attr=primary_feat, label_decode=_ghost_label_decode)
        _g_orig_str = getattr(cf, "original_label_str", str(cf.original_value))
        _g_flip_str = getattr(cf, "flipped_label_str",  str(cf.flipped_value))

        res_col, cf_col = st.columns(2)
        with res_col:
            st.markdown(f"""
            <div style="background:#161b22;border-radius:12px;padding:1.2rem;border:1px solid #30363d;text-align:center">
              <div style="color:#8b949e;font-size:0.85rem;margin-bottom:0.3rem">MODEL DECISION</div>
              <div style="font-size:2.2rem;font-weight:700;color:{decision_color}">{decision}</div>
              <div style="color:#8b949e;margin-top:0.5rem">Confidence: <b style="color:#e6edf3">{conf:.1%}</b></div>
              <div style="color:#8b949e">{primary_feat}: <b style="color:#a78bfa">{_g_orig_str}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with cf_col:
            if cf.risk_level == "CLEAR":
                color, icon, label = "#16a34a", "✅", "CLEAR"
                msg = f"Decision is <b>demographically stable</b>. Changing {primary_feat} does not affect the outcome."
            elif cf.risk_level == "FLAGGED":
                color, icon, label = "#d97706", "⚠️", "FLAGGED"
                msg = f"Same decision but confidence <b>drops {cf.confidence_delta:.0%}</b> on demographic swap. Recommend manual review."
            else:
                color, icon, label = "#dc2626", "🚨", "CRITICAL"
                msg = f"Decision <b>FLIPS</b> when {primary_feat} changes (<b>{_g_orig_str} → {_g_flip_str}</b>). Direct individual-level discrimination."

            st.markdown(f"""
            <div style="background:#161b22;border-radius:12px;padding:1.2rem;border:2px solid {color};text-align:center">
              <div style="color:#8b949e;font-size:0.85rem;margin-bottom:0.3rem">GHOST CHECK RESULT</div>
              <div style="font-size:2.2rem;font-weight:700;color:{color}">{icon} {label}</div>
              <div style="color:#c9d1d9;margin-top:0.5rem;font-size:0.88rem">{msg}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        section("Why This Decision? (SHAP Local Explanation)")
        hint(
            "<b>SHAP values</b> explain this specific prediction. "
            "Green bars = features <b>pushing toward approval</b>. "
            "Red bars = features <b>pushing toward denial</b>. "
            "Length = how strongly that feature influenced this particular decision."
        )
        try:
            exp    = st.session_state.get("shap_explainer")
            fnames = st.session_state.get("feat_names", feat_names)
            if not exp:
                from src.explainability.shap_explainer import build_explainer
                exp = build_explainer(model, X_sample)
                st.session_state["shap_explainer"] = exp

            from src.explainability.shap_explainer import explain_single
            local   = explain_single(exp, row[fnames], fnames, pred)
            drivers = local.top_drivers(6)

            fig_local = go.Figure(go.Bar(
                x=[abs(v) for _, _, v in drivers],
                y=[f"{f} = {val:.2f}" for f, val, _ in drivers],
                orientation="h",
                marker_color=["#16a34a" if v > 0 else "#dc2626" for _, _, v in drivers],
                text=[f"{v:+.3f}" for _, _, v in drivers], textposition="outside",
            ))
            fig_local.update_yaxes(autorange="reversed")
            fig_local.update_layout(xaxis_title="|SHAP value|", **DARK_LAYOUT, height=300)
            st.plotly_chart(fig_local, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.caption(f"SHAP explanation unavailable: {e}")

        st.divider()
        _g_key = st.session_state.get("_llm_api_key", "")
        if _g_key:
            if st.button("✨ Explain with Gemini AI"):
                with st.spinner("Gemini is explaining this prediction..."):
                    try:
                        try:
                            top3 = [(f, round(v, 3)) for f, _, v in drivers[:3]]
                        except Exception:
                            top3 = []
                        _cf_msg = getattr(cf, "message", cf.risk_level)
                        _expl_prompt = (
                            f"A prediction model decided: {decision} (confidence {conf:.0%}).\n"
                            f"Top factors: {top3}.\n"
                            f"Counterfactual Ghost result: {cf.risk_level} — {_cf_msg}\n"
                            f"Demographic flip: {_g_orig_str} → {_g_flip_str}.\n\n"
                            "Explain this decision in 2 plain sentences a non-technical person could understand. "
                            "Then explain what the Counterfactual Ghost result means for fairness in one sentence."
                        )
                        import google.generativeai as genai
                        genai.configure(api_key=_g_key)
                        gm = genai.GenerativeModel("gemini-2.0-flash")
                        st.info(gm.generate_content(_expl_prompt).text)
                    except Exception as _ge:
                        st.warning(f"Gemini error: {_ge}")
        else:
            hint("Add your <b>Gemini API key</b> in the sidebar to get plain-English explanations of any prediction.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AUDIT REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## 📄 Compliance Audit Report")
    hint(
        "This tab generates: an <b>AI-written compliance narrative</b>, a <b>downloadable PDF audit report</b>, "
        "a <b>shareable audit link</b>, and an embeddable <b>Fairness Badge</b> for your README or website."
    )

    if "pipeline_result" not in st.session_state:
        st.info("Complete the Bias X-Ray first (Tab 1).")
        st.stop()

    result  = st.session_state["pipeline_result"]
    primary = result.config.primary_protected_attr()

    # ── AI Narrative ───────────────────────────────────────────────────────────
    section("Gemini AI — Audit Narrative")
    hint(
        "Gemini translates bias metrics into a plain-English compliance document for officers and regulators. "
        "Covers: what bias was found, why it exists, what was done, what risks remain, and which regulations apply. "
        "<b>This is what a compliance officer needs to sign off on AI deployment.</b>"
    )

    _llm_key  = st.session_state.get("_llm_api_key", "")
    prov      = st.session_state.get("provenance")
    best_eval = result.inprocessed_eval or result.preprocessed_eval

    if "narrative" not in st.session_state and prov and best_eval:
        from src.narrative.llm_narrator import _fallback_narrative
        st.session_state["narrative"] = _fallback_narrative(result.baseline_eval, best_eval, primary, prov)
    elif "narrative" not in st.session_state and prov:
        from src.narrative.llm_narrator import _fallback_narrative
        st.session_state["narrative"] = _fallback_narrative(result.baseline_eval, result.baseline_eval, primary, prov)

    _narr_label = "✨ Generate Narrative with Gemini AI" if _llm_key else "✨ Generate Rule-Based Narrative"
    if st.button(_narr_label, use_container_width=True):
        if prov:
            from src.narrative.llm_narrator import stream_audit_narrative
            from src.bias_engine.proxy_scanner import get_flagged_proxies
            flagged = get_flagged_proxies(result.proxy_scan)
            eval_for_narr = best_eval or result.baseline_eval
            spinner_msg = "Gemini is writing your compliance narrative..." if _llm_key else "Generating rule-based narrative..."
            with st.spinner(spinner_msg):
                narrative = ""
                box = st.empty()
                for chunk in stream_audit_narrative(
                    result.baseline_eval, eval_for_narr, flagged, prov,
                    result.dataset_name, primary,
                    api_key=_llm_key or None,
                ):
                    narrative += chunk
                    box.markdown(narrative)
            st.session_state["narrative"] = narrative
            st.success("Narrative generated. Scroll down to download the PDF.")
        else:
            st.warning("Run Tab 1 first to generate provenance data.")

    if "narrative" in st.session_state:
        st.markdown("""
        <div style="background:#161b22;border-radius:10px;padding:1.2rem 1.5rem;border:1px solid #30363d;
                    line-height:1.8;font-size:0.92rem;color:#c9d1d9;">
        """, unsafe_allow_html=True)
        st.markdown(st.session_state["narrative"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Gemini Risk Scorecard ──────────────────────────────────────────────────
    section("Gemini AI — Compliance Risk Scorecard")
    hint(
        "Gemini analyses the audit metrics and produces a <b>structured risk scorecard</b> using JSON-mode output. "
        "Each dimension is scored 0–100 with a Pass / Warning / Fail status, and Gemini gives a deployment recommendation."
    )
    if _llm_key:
        if st.button("📊 Generate Risk Scorecard", use_container_width=True, key="btn_scorecard"):
            with st.spinner("Gemini is scoring your model's compliance risk..."):
                try:
                    from src.narrative.llm_narrator import generate_risk_scorecard
                    from src.bias_engine.proxy_scanner import get_flagged_proxies
                    _sc_prov = st.session_state.get("provenance")
                    _sc_best = result.inprocessed_eval or result.preprocessed_eval or result.baseline_eval
                    _sc_flagged = get_flagged_proxies(result.proxy_scan)
                    scorecard = generate_risk_scorecard(
                        result.baseline_eval, _sc_best,
                        _sc_flagged, _sc_prov,
                        result.dataset_name, primary,
                        api_key=_llm_key,
                    )
                    st.session_state["gemini_scorecard"] = scorecard
                except Exception as _se:
                    st.error(f"Scorecard generation failed: {_se}")

        if "gemini_scorecard" in st.session_state and st.session_state["gemini_scorecard"]:
            sc = st.session_state["gemini_scorecard"]
            _risk_colors = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149", "Critical": "#ff0000"}
            _rec_colors  = {"Deploy": "#3fb950", "Deploy with monitoring": "#d29922",
                            "Defer — remediate first": "#f85149", "Block deployment": "#ff0000"}
            _sc_risk = sc.get("overall_risk", "Unknown")
            _sc_rec  = sc.get("deployment_recommendation", "Unknown")
            _sc_score = sc.get("overall_risk_score", 0)
            _sc_legal = sc.get("legal_exposure", "Unknown")

            _sc_risk_color = _risk_colors.get(_sc_risk, "#fff")
            _sc_legal_color = _risk_colors.get(_sc_legal, "#fff")
            _sc_rec_color = _rec_colors.get(_sc_rec, "#fff")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"<div style='text-align:center'><span style='font-size:2rem;font-weight:700;color:{_sc_risk_color}'>{_sc_risk}</span><br><span style='color:#8b949e;font-size:0.75rem'>RISK LEVEL</span></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='text-align:center'><span style='font-size:2rem;font-weight:700'>{_sc_score}</span><span style='color:#8b949e'>/100</span><br><span style='color:#8b949e;font-size:0.75rem'>RISK SCORE</span></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div style='text-align:center'><span style='font-size:1.1rem;font-weight:700;color:{_sc_legal_color}'>{_sc_legal}</span><br><span style='color:#8b949e;font-size:0.75rem'>LEGAL EXPOSURE</span></div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div style='text-align:center'><span style='font-size:0.85rem;font-weight:700;color:{_sc_rec_color}'>{_sc_rec}</span><br><span style='color:#8b949e;font-size:0.75rem'>RECOMMENDATION</span></div>", unsafe_allow_html=True)

            if sc.get("dimensions"):
                st.markdown("**Dimension Breakdown**")
                _status_colors = {"Pass": "#3fb950", "Warning": "#d29922", "Fail": "#f85149"}
                _dim_rows = []
                for d in sc["dimensions"]:
                    _dim_rows.append({
                        "Dimension": d.get("name", ""),
                        "Score": d.get("score", 0),
                        "Status": d.get("status", ""),
                        "Finding": d.get("finding", ""),
                    })
                st.dataframe(pd.DataFrame(_dim_rows), use_container_width=True, hide_index=True)

            if sc.get("top_actions"):
                st.markdown("**Top Remediation Actions**")
                for i, action in enumerate(sc["top_actions"], 1):
                    st.markdown(f"{i}. {action}")

            if sc.get("applicable_regulations"):
                st.caption("Applicable regulations: " + " | ".join(sc["applicable_regulations"]))
    else:
        hint("Add your <b>Gemini API key</b> in the sidebar to generate a structured risk scorecard.")

    st.divider()

    # ── PDF Download ───────────────────────────────────────────────────────────
    section("Download Compliance Report (PDF)")
    hint(
        "The PDF contains: cover page with severity badge, executive summary, 6-metric table, "
        "shadow proxy scan, AI narrative, regulatory compliance statement, and a signature block. "
        "<b>Designed to be filed with regulators or presented to a compliance board.</b>"
    )
    if st.button("📥 Generate & Download PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            try:
                from src.api.routers.audit import _serialize_result
                from src.utils.report_generator import generate_pdf_bytes
                data      = _serialize_result(result)
                narrative = st.session_state.get("narrative", "")
                pdf_bytes = generate_pdf_bytes(data, narrative=narrative)
                st.download_button(
                    label="⬇️ Download Audit Report PDF",
                    data=pdf_bytes,
                    file_name=f"aequitas_audit_{result.dataset_name.replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success(f"PDF ready — {len(pdf_bytes) / 1024:.0f} KB")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.divider()

    # ── Shareable Audit Link + SVG Badge ──────────────────────────────────────
    section("🔗 Shareable Audit Link & Fairness Badge")
    hint(
        "Generate a unique URL that encodes your audit results — share it with your team, regulators, or the public. "
        "The <b>Fairness Badge</b> is an SVG you can embed in your GitHub README, product website, or documentation "
        "to signal that your model has been independently audited. Makes fairness visible and builds trust."
    )

    if st.button("🔗 Generate Shareable Link & Badge", use_container_width=True):
        try:
            encoded, svg_bytes = make_shareable_link(result)
            st.session_state["share_encoded"] = encoded
            st.session_state["share_svg"]     = svg_bytes
            st.success("Link and badge generated!")
        except Exception as e:
            st.error(f"Could not generate link: {e}")

    if "share_encoded" in st.session_state:
        encoded   = st.session_state["share_encoded"]
        svg_bytes = st.session_state["share_svg"]

        # Shareable URL (works locally + in cloud)
        share_url = f"?audit={encoded}"
        st.text_input("Shareable Audit URL (copy this):", value=share_url, key="share_url_display",
                      help="Share this URL. Anyone opening it will see a summary of your audit results.")

        # Badge preview + embed code
        svg_b64 = base64.b64encode(svg_bytes).decode()
        badge_img = f"data:image/svg+xml;base64,{svg_b64}"
        st.markdown(f'<img src="{badge_img}" alt="Fairness Audit Badge" style="height:28px;margin:0.5rem 0">', unsafe_allow_html=True)

        primary  = result.config.primary_protected_attr()
        bm       = result.baseline_eval.fairness.get(primary) if result.baseline_eval else None
        di_val   = bm.disparate_impact if bm and not np.isnan(bm.disparate_impact) else None
        severity = bm.severity if bm else "UNKNOWN"

        embed_md  = f"![Fairness Audit]({badge_img})"
        embed_html = f'<img src="{badge_img}" alt="Aequitas Prime Fairness Audit — {severity}" height="20">'

        with st.expander("Embed code"):
            st.markdown("**Markdown (GitHub README):**")
            st.code(embed_md, language="markdown")
            st.markdown("**HTML:**")
            st.code(embed_html, language="html")

        st.download_button(
            label="⬇️ Download Badge SVG",
            data=svg_bytes,
            file_name=f"fairness_badge_{result.dataset_name.replace(' ','_')}.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )

    st.divider()

    # ── Regulatory Coverage ────────────────────────────────────────────────────
    section("Regulatory Coverage")
    hint(
        "This audit covers four major regulatory frameworks. Failing to audit AI under these frameworks "
        "carries fines up to <b>€35M (EU AI Act)</b> or class-action liability (ECOA)."
    )
    regs = [
        ("EU AI Act (2024) — Article 10",  "Requires bias testing for all high-risk AI systems before deployment. Full enforcement: Aug 2026.",     "€35M or 7% of global turnover"),
        ("US ECOA (Equal Credit)",         "Equal Credit Opportunity Act — prohibits discriminatory credit/loan decisions based on protected class.",  "Civil liability + CFPB enforcement"),
        ("NYC Local Law 144 (2023)",        "Mandatory annual bias audits for any automated employment decision tool used in NYC.",                   "$1,500/day fine per violation"),
        ("US EEOC — Title VII",             "Non-discrimination in employment. AI hiring tools must not produce disparate impact on protected groups.", "EEOC enforcement + class action"),
    ]
    for reg, desc, penalty in regs:
        col_reg, col_desc, col_pen = st.columns([2, 4, 2])
        with col_reg:   st.markdown(f"✅ **{reg}**")
        with col_desc:  st.markdown(f"<span style='color:#c9d1d9;font-size:0.88rem'>{desc}</span>", unsafe_allow_html=True)
        with col_pen:   st.markdown(f"<span style='color:#d97706;font-size:0.82rem'>{penalty}</span>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#21262d;margin:0.4rem 0'>", unsafe_allow_html=True)

    st.divider()

    # ── Gemini Remediation Chatbot ─────────────────────────────────────────────
    section("Gemini AI — Fairness Advisor Chat")
    hint(
        "Ask Gemini anything about your audit results. "
        "Examples: <i>\"Why is my Disparate Impact below 0.80?\"</i>, "
        "<i>\"What Python code fixes label bias?\"</i>, "
        "<i>\"Which EU AI Act articles apply to me?\"</i>"
    )

    _chat_key = st.session_state.get("_llm_api_key", "")
    if _chat_key:
        # Build audit context string once
        if "gemini_audit_context" not in st.session_state:
            try:
                from src.narrative.llm_narrator import _build_metrics_prompt
                from src.bias_engine.proxy_scanner import get_flagged_proxies
                _ctx_prov = st.session_state.get("provenance")
                _ctx_best = result.inprocessed_eval or result.preprocessed_eval or result.baseline_eval
                _ctx_flagged = get_flagged_proxies(result.proxy_scan)
                if _ctx_prov:
                    st.session_state["gemini_audit_context"] = _build_metrics_prompt(
                        result.baseline_eval, _ctx_best,
                        _ctx_flagged, _ctx_prov,
                        result.dataset_name, primary,
                    )
            except Exception:
                st.session_state["gemini_audit_context"] = "Audit context unavailable."

        if "gemini_chat_history" not in st.session_state:
            st.session_state["gemini_chat_history"] = []
        if "gemini_chat_display" not in st.session_state:
            st.session_state["gemini_chat_display"] = []

        # Render chat history
        for _msg in st.session_state["gemini_chat_display"]:
            with st.chat_message(_msg["role"], avatar="🧑" if _msg["role"] == "user" else "✨"):
                st.markdown(_msg["content"])

        _user_input = st.chat_input("Ask Gemini about your audit...", key="gemini_chat_input")
        if _user_input:
            # Show user message immediately
            st.session_state["gemini_chat_display"].append({"role": "user", "content": _user_input})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(_user_input)

            with st.chat_message("assistant", avatar="✨"):
                with st.spinner("Gemini is thinking..."):
                    from src.narrative.llm_narrator import ask_remediation_chat
                    _reply = ask_remediation_chat(
                        user_message=_user_input,
                        audit_context=st.session_state.get("gemini_audit_context", ""),
                        chat_history=st.session_state["gemini_chat_history"],
                        api_key=_chat_key,
                    )
                st.markdown(_reply)

            # Append to Gemini history format for multi-turn context
            st.session_state["gemini_chat_history"].append(
                {"role": "user",  "parts": [_user_input]}
            )
            st.session_state["gemini_chat_history"].append(
                {"role": "model", "parts": [_reply]}
            )
            st.session_state["gemini_chat_display"].append(
                {"role": "assistant", "content": _reply}
            )

        if st.session_state["gemini_chat_display"]:
            if st.button("🗑 Clear chat", key="clear_gemini_chat"):
                st.session_state["gemini_chat_history"] = []
                st.session_state["gemini_chat_display"] = []
                st.session_state.pop("gemini_audit_context", None)
                st.rerun()
    else:
        hint("Add your <b>Gemini API key</b> in the sidebar to chat with the AI fairness advisor.")
