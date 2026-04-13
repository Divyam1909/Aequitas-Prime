# Aequitas Prime

**Fairness-as-a-Service — Algorithmic Bias Detection, Mitigation & Compliance Platform**

> Upload any tabular ML dataset (or your own model's predictions) → get a complete fairness audit in ~60 seconds. Detect bias, understand where it comes from, fix it, prove the fix worked, and download a compliance-grade PDF report.

---

## What It Does

Aequitas Prime is a full-stack algorithmic fairness platform with a 7-tab Streamlit dashboard and a FastAPI backend. It covers the entire bias lifecycle:

| Stage | What it does |
|-------|-------------|
| **Detect** | 6 fairness metrics, shadow proxy scan, bias provenance tracing |
| **Understand** | Intersectional analysis, SHAP/LIME explainability, counterfactual panel |
| **Fix** | Pre-processing (Reweighing), in-processing (ExponentiatedGradient, GridSearch, AdversarialDebiasing) |
| **Prove** | Before/after comparison, Pareto frontier, debiased dataset export |
| **Report** | PDF compliance report, AI-generated narrative, shareable badge, regulatory mapping |

Works with **any binary-classification CSV** — income, credit scoring, hiring, healthcare, recidivism — not just the UCI Adult dataset.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run ui/app.py
```

Opens at [http://localhost:8501](http://localhost:8501)

> **Dataset:** Upload your own CSV or click **"Use Adult Income Sample Data"** to try the demo (48,842-row UCI Census dataset). Nothing loads automatically — you are always in control.

---

## The 7 Tabs

### 🔍 Bias X-Ray
Upload **any CSV**. Aequitas Prime auto-detects protected attributes (sex, race, age, etc.) using keyword matching and cardinality heuristics. A configuration form lets you confirm/adjust the detected columns, select the target, set the positive outcome, and identify privileged group values — no code changes required.

Produces:
- Disparate Impact gauges (with color-coded legal threshold alerts)
- Full 6-metric PASS/FAIL table with plain-English explanations
- Shadow Proxy Scanner (mutual information + Cramér's V)
- Bias Provenance card (label bias / representation bias / feature bias)

### 🤖 BYOM — Bring Your Own Model
Upload a predictions CSV from **any model or framework** (XGBoost, PyTorch, HuggingFace, etc.) with columns `[features, true_label, predicted_label]`. No model access needed — Aequitas Prime runs all 6 fairness metrics on your predictions directly.

### ⚗️ Surgeon — Bias Mitigation
Manually triggered (not automatic). Click **Run Mitigation** to apply:
- **Pre-processing:** Reweighing (AIF360) — adds sample weights to correct group outcome imbalance
- **In-processing:** ExponentiatedGradient (Fairlearn) — trains with a hard EqualizedOdds constraint
- **Pareto Frontier:** sweeps 6 fairness constraint weights, shows every accuracy/fairness tradeoff, marks the optimal operating point

Includes a **debiased dataset CSV download** — the reweighed dataset with a `sample_weight` column, compatible with any sklearn-API model.

### 🧩 Intersectional Analysis
Computes bias across every **combination** of protected attributes simultaneously (e.g., "Black Female" vs "White Male"). Standard tools check one attribute at a time and miss compounding discrimination.

Features:
- **Intersectionality Bias Index (IBI)** — a single scalar (0–100) quantifying total compounding disadvantage
- **Radar chart** of the 6 worst-affected groups across DI, Approval Rate, and Equal Opportunity
- DI heatmap (sex × race grid)
- Ranked worst-case group table with IBI contribution per group

### 🔀 Counterfactual Panel
**Proves** that the trained model is biased — not just the data.

- **Batch Check:** Samples 500 test rows, flips the protected attribute for each, shows what percentage of decisions change (CRITICAL = decision flips, FLAGGED = confidence drops, CLEAR = stable)
- **Row Explorer:** Pick any test-set row, see the original prediction vs the counterfactual side-by-side

A 20%+ CRITICAL rate is smoking-gun evidence of systematic individual-level discrimination.

### 👻 Ghost — Live Inference Firewall
For any live prediction, flip the protected attribute and check if the decision changes. Fill in a candidate's features, get the model's decision, and immediately see if changing their demographic would change the outcome. Includes SHAP local explanation and optional Gemini plain-English interpretation.

### 📄 Report
- **AI Narrative:** Google Gemini generates a compliance-grade narrative (free tier). Falls back to rule-based narrative if no key is set.
- **PDF Download:** Compliance report with severity badge, metrics table, proxy scan, narrative, and regulatory sign-off block — designed to be filed with regulators.
- **Shareable Audit Link:** Encodes audit results into a URL parameter for sharing.
- **Fairness Badge:** SVG badge (embeddable in GitHub README, websites, or documentation) showing audit severity and DI score.

---

## Project Structure

```
google-hackathon/
├── data/
│   └── raw/
│       └── adult.csv                    # UCI Adult Income dataset (48,842 rows)
├── models/                              # Saved model artifacts (.joblib)
├── reports/                             # Generated PDF audit reports
├── scripts/
│   ├── download_data.py                 # Downloads adult.csv if missing
│   ├── prebake_demo.py                  # Pre-trains and caches demo pipeline result
│   └── generate_samples.py             # Generates sample CSVs for testing
├── src/
│   ├── bias_engine/
│   │   ├── detector.py                  # 6 fairness metrics (DI, SPD, EOD, EOpD, PP, FNRP)
│   │   ├── proxy_scanner.py             # Shadow proxy detection (MI, Cramér's V, PB correlation)
│   │   ├── mitigator.py                 # Pre-processing: Reweighing + DisparateImpactRemover
│   │   ├── inprocessor.py               # In-processing: ExponentiatedGradient / GridSearch / AdversarialDebiasing
│   │   ├── counterfactual.py            # Counterfactual Ghost Engine (CLEAR / FLAGGED / CRITICAL)
│   │   ├── intersectional.py            # Intersectional bias + Intersectionality Bias Index
│   │   └── provenance.py                # Root cause tracer (label / representation / feature bias)
│   ├── ml_pipeline/
│   │   ├── trainer.py                   # Train 3 model tracks (baseline, pre-processed, in-processed)
│   │   ├── evaluator.py                 # Performance (Acc, F1, AUC) + fairness evaluation
│   │   ├── pareto.py                    # Fairness-accuracy Pareto frontier sweep + knee detection
│   │   ├── model_store.py               # Save/load model artifacts (joblib + JSON metadata)
│   │   └── pipeline.py                  # End-to-end orchestrator → AuditResult
│   ├── explainability/
│   │   ├── shap_explainer.py            # SHAP (TreeExplainer / LinearExplainer / KernelExplainer auto-select)
│   │   └── lime_explainer.py            # LIME tabular explainer
│   ├── narrative/
│   │   └── llm_narrator.py              # Gemini API narrative (free tier) + rule-based fallback
│   ├── api/
│   │   ├── main.py                      # FastAPI app + CORS + startup model loading
│   │   └── routers/
│   │       ├── audit.py                 # POST /audit/upload, GET /audit/result/{job_id}
│   │       ├── inference.py             # POST /predict (live prediction with fairness firewall)
│   │       └── report.py                # GET /report/{job_id} → PDF download
│   └── utils/
│       ├── schema.py                    # DatasetConfig dataclass (central schema)
│       ├── data_loader.py               # CSV loader + column validator
│       ├── adult_preprocessor.py        # UCI Adult Income specific preprocessor + ADULT_CONFIG
│       ├── generic_preprocessor.py      # Generic preprocessor for any CSV (auto-detect + encode)
│       └── report_generator.py          # PDF report generator (ReportLab)
├── ui/
│   └── app.py                           # Streamlit dashboard (7 tabs)
├── tests/                               # Test suite
├── .streamlit/
│   └── config.toml                      # Dark theme config
├── requirements.txt
└── tasks.md                             # Full project task tracker
```

---

## The Full Technical Pipeline

```
CSV Upload (any dataset) OR Adult Income sample
    │
    ▼
generic_preprocessor / adult_preprocessor
    → strip whitespace, drop NaN, binary-encode protected attrs
    → ordinal-encode categoricals
    → return (X: encoded features, y: binary target, df_clean: original strings)
    │
    ├──► Shadow Proxy Scanner
    │       → mutual_info_classif per feature vs each protected attr
    │       → Cramér's V for categorical cols
    │       → Point-Biserial for continuous cols
    │       → Flag HIGH (MI≥0.15) / MEDIUM (MI≥0.08) risk proxies
    │
    ├──► Bias Provenance Tracer
    │       → label distribution skew → Label Bias
    │       → group size imbalance → Representation Bias
    │       → high-MI proxy features → Feature Bias
    │       → returns primary + secondary source + confidence + recommended fixes
    │
    ├──► Baseline Track: RandomForest (no mitigation)
    │       → 6 fairness metrics → DI / SPD / EOD / EOpD / PP / FNRP
    │       → SEVERITY: CLEAR (0 fails) / WARNING (1–2 fails) / CRITICAL (3+ fails)
    │
    ├──► Pre-processing Track (on-demand in Surgeon tab)
    │       → AIF360 Reweighing → per-group sample weights
    │       → train RF with sample_weight → re-evaluate 6 metrics
    │       → export reweighed CSV for external use
    │
    ├──► In-processing Track (optional, on-demand)
    │       → Fairlearn ExponentiatedGradient + EqualizedOdds constraint
    │       → train on full dataset with demographic parity baked in
    │       → re-evaluate 6 metrics
    │
    ├──► Fairness-Accuracy Pareto Sweep
    │       → Fairlearn GridSearch at 6 constraint_weight values (0.0→1.0)
    │       → knee-point detection (max fairness gain / accuracy loss)
    │
    ├──► SHAP Explainability
    │       → auto-select TreeExplainer / LinearExplainer / KernelExplainer
    │       → global: mean |SHAP| per feature → sorted importance chart
    │       → local: per-prediction feature contributions
    │       → flag if protected attr in top-5 (proxy leakage check)
    │
    ├──► Intersectional Analysis
    │       → enumerate all combinations of protected attr values
    │       → compute DI and EOpD per group (min size: 30)
    │       → Intersectionality Bias Index = (1 − weighted_mean_DI) × 100
    │       → radar chart + ranked worst-group table
    │
    ├──► Counterfactual Panel (dataset-level)
    │       → for N sampled test rows: flip primary protected attr
    │       → classify CRITICAL / FLAGGED / CLEAR per row
    │       → batch statistics + individual row explorer
    │
    └──► AuditResult (dataclass)
            ├── Streamlit dashboard (7 tabs)
            ├── ReportLab PDF (compliance report)
            ├── Gemini LLM narrative (optional)
            ├── Shareable audit URL + SVG badge
            └── FastAPI JSON serialization
```

---

## The 6 Fairness Metrics

| Metric | Formula | Pass Threshold | What it Catches |
|--------|---------|---------------|-----------------|
| **Disparate Impact (DI)** | P(Ŷ=1\|unprivileged) ÷ P(Ŷ=1\|privileged) | ≥ 0.80 | Unequal outcome rates — the core US/EU legal standard |
| **Statistical Parity Diff (SPD)** | P(Ŷ=1\|unpriv) − P(Ŷ=1\|priv) | ≥ −0.10 | Raw percentage-point gap in selection rates |
| **Equalized Odds Diff (EOD)** | max(\|TPR gap\|, \|FPR gap\|) | ≤ 0.10 | Model being less accurate for one group |
| **Equal Opportunity Diff (EOpD)** | \|TPR_unpriv − TPR_priv\| | ≤ 0.10 | Qualified people wrongly rejected at different rates |
| **Predictive Parity (PP)** | \|PPV_unpriv − PPV_priv\| | ≤ 0.10 | Model's approvals meaning different things per group |
| **FNR Parity (FNRP)** | \|FNR_unpriv − FNR_priv\| | ≤ 0.10 | Disproportionate wrongful denials across groups |

**Severity:** CLEAR (0 fails) → WARNING (1–2 fails) → CRITICAL (3+ fails)

---

## Bias Mitigation Techniques

### Pre-Processing: Reweighing (AIF360)
Assigns inverse-frequency weights to training samples based on group × label combinations. Upweights underrepresented combinations (e.g., "Female+Income>50K"), downweights overrepresented ones. The model trains on a rebalanced version of history without any changes to features or labels.

### In-Processing Tier 1 (default): ExponentiatedGradient (Fairlearn)
Wraps any sklearn estimator with a fairness constraint. Trains a randomized ensemble of models, each optimizing a different Lagrangian relaxation, then combines them to satisfy the constraint at inference time. Constraint: `EqualizedOdds` (equal TPR and FPR across groups).

> **Implementation note:** `sensitive_features` is passed as a keyword argument to `fit()` — never included in `X`. This is the most common API error with Fairlearn, and is correctly avoided here.

### In-Processing Tier 2: GridSearch (Fairlearn)
Deterministic alternative — sweeps a grid of `constraint_weight` values (0=pure accuracy, 1=pure fairness) and trains one model per point. Used for the Pareto frontier. Useful when reproducibility is required.

### In-Processing Tier 3: AdversarialDebiasing (AIF360, TensorFlow required)
Neural adversarial training: a classifier predicts the target while an adversary simultaneously tries to predict the protected attribute from the classifier's internals. Falls back gracefully to ExponentiatedGradient if TensorFlow is absent.

---

## Using Your Own Dataset

1. Upload any binary-classification CSV in Tab 1 (Bias X-Ray)
2. Aequitas Prime auto-detects candidate protected attributes using:
   - **Keyword matching** against known protected-attribute names (sex, gender, race, ethnicity, age, religion, nationality, disability, marital status, etc.)
   - **Cardinality heuristic**: string columns with 2–20 unique values
3. A configuration form appears — confirm the detected columns, select:
   - Target column (what to predict)
   - Positive outcome value (loan approved, hired, not defaulted, etc.)
   - Protected attributes and their privileged group values
4. Click **Run Full Bias X-Ray** — the full pipeline runs with your configuration

### BYOM (Bring Your Own Model)
If your model is already trained (or is not an sklearn model), go to Tab 2 — BYOM. Upload a CSV with:
- Feature columns
- A true label column (ground truth)
- A predicted label column (your model's output)

Aequitas Prime runs all 6 fairness metrics on your predictions without needing model internals.

---

## Gemini API Key (Optional)

Powers the AI Audit Narrative in the Report tab. The entire platform works without it (rule-based fallback narrative auto-generates).

**To enable (free):**
1. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no credit card, 15 requests/min
2. Paste it in the sidebar, or set `GEMINI_API_KEY` as an environment variable

---

## FastAPI Backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

| Endpoint | Description |
|----------|-------------|
| `POST /audit/upload` | Upload CSV, trigger pipeline in background, returns job_id |
| `GET /audit/status/{job_id}` | Poll job status |
| `GET /audit/result/{job_id}` | Full audit result as JSON |
| `POST /predict` | Live prediction with counterfactual fairness check + SHAP |
| `GET /report/{job_id}` | Download PDF compliance report |

> **Note:** The API currently uses the UCI Adult Income configuration by default. For custom dataset support via API, pass a `DatasetConfig` JSON payload (future enhancement).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Fairness metrics** | IBM AIF360 ≥ 0.6.1 |
| **Bias mitigation** | Microsoft Fairlearn ≥ 0.10 |
| **ML model** | scikit-learn RandomForestClassifier |
| **Explainability** | SHAP ≥ 0.44 + LIME ≥ 0.2 |
| **UI dashboard** | Streamlit ≥ 1.35 + Plotly |
| **REST API** | FastAPI + Uvicorn |
| **PDF reports** | ReportLab |
| **LLM narrative** | Google Gemini 2.0 Flash (free tier) |
| **Data layer** | pandas + numpy + scipy |
| **Neural mitigation** | TensorFlow ≥ 2.15 (optional) |

---

## Supported Regulatory Frameworks

| Framework | Jurisdiction | Key Requirement | Max Penalty |
|-----------|-------------|-----------------|-------------|
| **EU AI Act — Article 10** | European Union | Bias testing for all high-risk AI before deployment (enforcement: Aug 2026) | €35M or 7% global turnover |
| **US ECOA (Equal Credit)** | United States | No discriminatory credit/loan decisions by protected class | Civil liability + CFPB enforcement |
| **NYC Local Law 144** | New York City | Annual bias audits for automated employment decisions | $1,500/day per violation |
| **US EEOC — Title VII** | United States | AI hiring tools must not produce disparate impact on protected groups | EEOC enforcement + class action |

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError` | Activate virtualenv: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` |
| Pipeline takes too long | Uncheck "Include in-processing" in Tab 1. Baseline + Reweighing only takes ~15–20 seconds. |
| Gemini narrative missing | Add API key in sidebar, or leave blank — rule-based fallback generates automatically. |
| Port 8501 busy | `streamlit run ui/app.py --server.port 8502` |
| `adult.csv` not found | `python scripts/download_data.py` |
| TensorFlow warnings on startup | Normal — TF is optional. Set `TF_CPP_MIN_LOG_LEVEL=3` to suppress. |
| Custom CSV not loading | Ensure the CSV has a clear binary target column and at least one demographic column. |
