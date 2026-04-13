# Aequitas Prime — Full Project Task List (A-Z)
**Status legend:** `[ ]` = not started · `[~]` = in progress · `[x]` = done

---

## PHASE 0 — Project Scaffold & Environment

- [x] **0.1** Create the project folder structure (see layout below)
- [ ] **0.2** Create and activate Python 3.11 virtual environment (`venv`) — run manually: `python -m venv venv && venv\Scripts\activate`
- [x] **0.3** Create `requirements.txt` with all dependencies:
  ```
  # Core data
  pandas>=2.0, numpy>=1.26
  scikit-learn>=1.4
  # Fairness engines
  aif360>=0.6
  fairlearn>=0.10
  # ML explainability
  shap>=0.44
  lime>=0.2
  # Visualization & UI
  streamlit>=1.32
  plotly>=5.18
  # Backend API
  fastapi>=0.110, uvicorn>=0.27, python-multipart>=0.0.9, httpx>=0.27
  # PDF report
  reportlab>=4.1
  # LLM narrative
  anthropic>=0.25
  # TF for AdversarialDebiasing (advanced in-processing)
  tensorflow>=2.15
  ```
- [ ] **0.4** `pip install -r requirements.txt` and verify all imports work — run manually after venv setup
- [x] **0.5** Download UCI Adult Income dataset (`adult.csv`) to `/data/raw/` — 48,842 rows via `scripts/download_data.py`
- [x] **0.6** Create `README.md` with project overview, setup instructions, and architecture diagram
- [x] **0.7** Initialize git repository, add `.gitignore`

```
aequitas-prime/
├── data/
│   ├── raw/                   # adult.csv, adult.test
│   └── processed/             # reweighed/repaired artifacts
├── models/                    # saved .pkl / .joblib model artifacts
├── reports/                   # generated PDF audit reports
├── src/
│   ├── bias_engine/
│   │   ├── detector.py        # 6 fairness metrics
│   │   ├── proxy_scanner.py   # [NEW] shadow proxy / leakage detection
│   │   ├── mitigator.py       # pre-processing: reweighing + DIR
│   │   ├── inprocessor.py     # in-processing: Fairlearn + AIF360 adversarial
│   │   ├── counterfactual.py  # post-processing: ghost engine
│   │   ├── intersectional.py  # [NEW] intersectional bias analysis
│   │   ├── provenance.py      # [NEW] bias source tracer
│   │   └── aif360_wrapper.py  # AIF360 BinaryLabelDataset helpers
│   ├── ml_pipeline/
│   │   ├── trainer.py         # train both model tracks
│   │   ├── evaluator.py       # accuracy + fairness post-training
│   │   ├── pareto.py          # [NEW] fairness-accuracy frontier sweep
│   │   ├── model_store.py     # save/load with metadata
│   │   └── pipeline.py        # end-to-end orchestrator
│   ├── explainability/
│   │   ├── shap_explainer.py  # TreeExplainer for RF + KernelSHAP fallback
│   │   └── lime_explainer.py  # LIME for in-processing ensemble model
│   ├── narrative/
│   │   └── llm_narrator.py    # [NEW] Claude API audit narrative generator
│   ├── api/
│   │   ├── main.py
│   │   └── routers/
│   │       ├── audit.py
│   │       ├── inference.py
│   │       └── report.py
│   └── utils/
│       ├── data_loader.py
│       ├── schema.py
│       ├── adult_preprocessor.py
│       └── report_generator.py
├── ui/
│   └── app.py                 # Streamlit multi-tab dashboard
├── tests/
├── requirements.txt
└── README.md
```

---

## PHASE 1 — Data Layer

- [x] **1.1** Write `src/utils/data_loader.py`
  - `load_csv(path)` → reads any CSV, strips whitespace, infers dtypes, returns DataFrame
  - `validate_columns(df, config)` → checks protected attrs + target col exist
- [x] **1.2** Write `src/utils/schema.py`
  - `DatasetConfig` dataclass: `protected_attrs`, `target_col`, `privileged_values`, `positive_label`
  - Supports multiple protected attributes simultaneously
- [x] **1.3** Write `src/utils/adult_preprocessor.py`
  - Strips whitespace, maps income to binary 0/1, OrdinalEncoder for categoricals
  - `ADULT_CONFIG` = default config (sex=Male privileged, race=White privileged, positive_label=1)
  - Returns `(X, y, df_clean)` — df_clean preserves raw string labels for fairness grouping
- [x] **1.4** Data integrity verified: 48,842 rows → 45,222 after dropping NaNs (7.4%); Male=30,527 Female=14,695; income≤50K=34,014 >50K=11,208

---

## PHASE 2 — Bias Detection Engine: 6 Metrics + Proxy Scanner

### Six Fairness Metrics

- [x] **2.1** Create `src/bias_engine/detector.py` with all 6 metrics:

  | Metric | Formula | Threshold |
  |---|---|---|
  | Disparate Impact (DI) | P(Ŷ=1\|unprivileged) / P(Ŷ=1\|privileged) | < 0.80 = FAIL |
  | Statistical Parity Difference (SPD) | P(Ŷ=1\|unprivileged) − P(Ŷ=1\|privileged) | < −0.10 = FAIL |
  | Equalized Odds Difference (EOD) | max(\|TPR_a−TPR_b\|, \|FPR_a−FPR_b\|) | > 0.10 = FAIL |
  | Equal Opportunity Difference (EOpD) | \|TPR_unprivileged − TPR_privileged\| | > 0.10 = FAIL |
  | Predictive Parity / Calibration (PP) | \|PPV_a − PPV_b\| where PPV = TP/(TP+FP) | > 0.10 = FAIL |
  | False Negative Rate Parity (FNRP) | \|FNR_a − FNR_b\| where FNR = FN/(TP+FN) | > 0.10 = FAIL |

  - Functions:
    - `compute_all_metrics(df, y_pred, config)` → `MetricsResult` dataclass
    - `flag_failures(metrics_result)` → `dict[str, bool]` of pass/fail per metric
    - `get_severity(metrics_result)` → `"CLEAR" | "WARNING" | "CRITICAL"` (0 fails / 1-2 fails / 3+ fails)
  - **Critical note:** DI and SPD are pre-training (on raw labels). EOD, EOpD, PP, FNRP require `y_pred` (post-training). The engine handles both phases.

- [ ] **2.2** Write `src/bias_engine/aif360_wrapper.py` — deferred (Phase 3 uses AIF360 directly)
- [ ] **2.3** Write unit tests `tests/test_detector.py` — deferred to Phase 15 QA pass

### Proxy Variable / Shadow Proxy Scanner (INNOVATIVE FEATURE 1)

- [x] **2.4** Create `src/bias_engine/proxy_scanner.py`
  - `scan_proxies(df, config, threshold=0.1)` → `list[ProxyResult]`
    - For each non-protected feature `f`, compute:
      - **Mutual Information**: `sklearn.feature_selection.mutual_info_classif(f, protected_attr)`
      - **Cramér's V** (for categorical features): `scipy.stats.chi2_contingency`
      - **Point-Biserial correlation** (for continuous features): `scipy.stats.pointbiserialr`
    - Flag any feature where MI score > `threshold`
    - Return sorted list: `ProxyResult(feature_name, mi_score, cramers_v, risk_level: "HIGH/MEDIUM/LOW")`
  - `generate_proxy_heatmap_data(proxy_results)` → DataFrame suitable for Plotly heatmap
  - `suggest_proxy_action(proxy_result)` → string recommendation:
    - HIGH risk: "Consider removing or capping this feature — it is a strong statistical proxy for the protected attribute."
    - MEDIUM risk: "Monitor this feature — it may re-introduce demographic bias through the back door."
- [ ] **2.5** Write unit tests `tests/test_proxy_scanner.py` — deferred to Phase 15
  - In Adult Income: "occupation" should flag as HIGH proxy for sex (expect MI > 0.15)
  - A random noise feature should score near 0

---

## PHASE 3 — Pre-Processing Mitigation (The Generative Surgeon)

- [x] **3.1** Create `src/bias_engine/mitigator.py`
  - `apply_reweighing(df, config)` → `(df_with_weights_col, weights_array)`
    - Use AIF360 `Reweighing`: computes P(Y|A) for each group and inverts the imbalance
    - Log the weight multiplier per group for UI display (e.g., "Unprivileged+Positive: 2.3x weight")
  - `apply_disparate_impact_remover(df, config, repair_level=0.8)` → `df_repaired`
    - Use AIF360 `DisparateImpactRemover` as optional secondary stage
    - `repair_level` parameter (0.0–1.0) controls how aggressively feature distributions are equalized
    - Note: Use only on non-protected continuous features (skip protected attr and target)
  - `get_weight_distribution_data(weights_array, df, config)` → dict for UI weight visualization
- [x] **3.2** `compute_before_after` included in `mitigator.py` — compares DI/SPD before/after reweighing
- [ ] **3.3** Write unit tests `tests/test_mitigator.py` — deferred to Phase 15

---

## PHASE 4 — In-Processing (The Adversarial Blindfold) ← GAP FIXED

> **Design Decision:** Three implementation tiers, each progressively more powerful. The MVP uses Tier 1. Tiers 2 and 3 are selectable in the UI for advanced users.

### Tier 1 (MVP): Fairlearn ExponentiatedGradient — Primary Implementation
- [x] **4.1** Create `src/bias_engine/inprocessor.py`
  - `train_expgrad_model(X_train, y_train, S_train, constraint="equalized_odds")` → `fitted_model`
    - `S_train` = the protected attribute series (passed as `sensitive_features`, NOT in X)
    - Constraint options: `"equalized_odds"` (default, strictest), `"demographic_parity"`, `"equal_opportunity"`
    - Implementation:
      ```python
      from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, DemographicParity
      constraint = EqualizedOdds()  # or DemographicParity()
      eg = ExponentiatedGradient(
          estimator=RandomForestClassifier(n_estimators=100, random_state=42),
          constraints=constraint,
          eps=0.01,        # Max allowed constraint violation
          max_iter=50
      )
      eg.fit(X_train, y_train, sensitive_features=S_train)
      ```
    - **Critical API note:** `sensitive_features` is a kwarg to `fit()`, never in X. Mixing them causes silent failures.
    - Returns a randomized ensemble — predictions use `eg.predict(X_test)` (internally weighted)

### Tier 2 (Fallback/Demo): Fairlearn GridSearch
- [x] **4.2** Add `train_gridsearch_model(X_train, y_train, S_train, grid_size=10)` to `inprocessor.py`
    ```python
    from fairlearn.reductions import GridSearch, EqualizedOdds
    gs = GridSearch(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        constraints=EqualizedOdds(),
        grid_size=10,
        constraint_weight=0.5   # 0=pure accuracy, 1=pure fairness
    )
    gs.fit(X_train, y_train, sensitive_features=S_train)
    best_model = gs.best_predictor_
    ```
  - GridSearch is deterministic (trains exactly `grid_size` models) — use for time-constrained demos
  - Expose `constraint_weight` as a UI slider (this IS the Pareto Frontier slider from Phase 9)

### Tier 3 (Advanced): AIF360 AdversarialDebiasing
- [x] **4.3** Add `train_adversarial_model(X_train, y_train, config)` to `inprocessor.py`
    ```python
    from aif360.sklearn.inprocessing import AdversarialDebiasing
    # IMPORTANT: protected attribute must be in DataFrame INDEX, not a column
    X_train_indexed = X_train.copy()
    X_train_indexed.index = pd.MultiIndex.from_arrays(
        [X_train_indexed.index, df_train[config.protected_attrs[0]]]
    )
    model = AdversarialDebiasing(
        prot_attr=config.protected_attrs[0],
        adversary_loss_weight=0.5,   # tune: 0.1–1.0
        num_epochs=50,
        batch_size=128,
        debias=True,
        random_state=42
    )
    model.fit(X_train_indexed, y_train)
    ```
  - **Known issues to handle:**
    - Requires `tensorflow>=2.15` (Python 3.11 compatible)
    - Cannot pickle — save using `joblib` with `protocol=4`
    - Must call `model.sess_.close()` after use to prevent TF session leaks
  - Gate this behind a `try/except ImportError` — if TF not available, fall back to ExponentiatedGradient
- [ ] **4.4** Write `tests/test_inprocessor.py` — deferred to Phase 15
  - Test ExponentiatedGradient: verify post-training EOD < 0.10 on Adult Income
  - Test GridSearch: verify `best_predictor_` is a valid sklearn estimator that produces predictions
  - Test AdversarialDebiasing: verify it trains without TF errors and produces binary predictions

---

## PHASE 5 — ML Training Pipeline

- [x] **5.1** Create `src/ml_pipeline/trainer.py`
  - `train_baseline_model(df, config)` → `(model, X_test, y_test, y_pred)` — biased, no mitigation
  - `train_preprocessed_model(df, weights, config)` → RF with `sample_weight=weights` (post-reweighing)
  - `train_inprocessed_model(X_train, y_train, S_train, method="expgrad")` → calls `inprocessor.py`
  - All trainers return a standardized `TrainResult(model, X_test, y_test, y_pred, model_type)` dataclass
- [x] **5.2** Create `src/ml_pipeline/evaluator.py`
  - `evaluate_performance(y_test, y_pred, model)` → `{accuracy, f1, precision, recall, roc_auc}`
    - Use `sklearn.metrics` — all standard
  - `evaluate_fairness(df_test, y_pred, config)` → calls `detector.compute_all_metrics()` — full 6-metric report
  - `compare_models(results: list[TrainResult], df_test, config)` → side-by-side comparison dict
    - Used for the "Which model should I deploy?" view in UI
- [x] **5.3** Create `src/ml_pipeline/model_store.py`
  - `save_artifact(model, metadata: dict, path)` → saves `model.joblib` + `metadata.json`
    - metadata includes: `model_type`, `train_date`, `dataset_shape`, `all_metrics`, `config`
  - `load_artifact(path)` → `(model, metadata)`
  - Handle `AdversarialDebiasing` separately (cannot use standard pickle — use joblib protocol=4)
- [x] **5.4** Create `src/ml_pipeline/pipeline.py` — end-to-end orchestrator
  - `AuditResult` dataclass: holds baseline metrics, pre-processed metrics, in-processed metrics, all model artifacts
  - `run_full_pipeline(df, config)` → `AuditResult`
    - Step 1: Baseline model → baseline metrics
    - Step 2: Proxy scan
    - Step 3: Reweighing → pre-processed model → pre-processed metrics
    - Step 4: ExponentiatedGradient in-processing → in-processed model → in-processed metrics
    - Step 5: Return `AuditResult` with everything
- [ ] **5.5** Write `tests/test_pipeline.py` — deferred to Phase 15
  - End-to-end: assert DI baseline < 0.80, DI pre-processed > 0.80, EOD in-processed < 0.10

---

## PHASE 6 — Explainability: SHAP + LIME

> **Why both?** SHAP `TreeExplainer` is fast and exact for RandomForest (pre-processing track). The Fairlearn `ExponentiatedGradient` returns an *ensemble of RFs* — `TreeExplainer` still works but must average across sub-estimators. LIME is the fallback for the `AdversarialDebiasing` neural model where SHAP TreeExplainer doesn't apply.

- [x] **6.1** Create `src/explainability/shap_explainer.py`
  - `build_tree_explainer(model)` → `shap.TreeExplainer`
    - For ExponentiatedGradient: extract `model.predictors_[0]` (first RF in ensemble) as representative
  - `compute_global_shap(explainer, X_sample, max_samples=500)` → `shap_values array`
    - Cap at 500 rows for speed — Adult Income has 48K rows, full SHAP would take minutes
  - `get_global_importance(shap_values, feature_names)` → `list[(feature, mean_abs_shap)]` sorted desc
  - `get_local_explanation(explainer, X_row, feature_names)` → top 3 drivers for a single prediction
    - Returns: `[(feature_name, feature_value, shap_contribution)]`
  - `check_protected_attr_dominance(global_importance, config)` → bool
    - Returns `True` if any protected attribute is in top 3 SHAP features (bias leakage flag)

- [x] **6.2** Create `src/explainability/lime_explainer.py`
  - `build_lime_explainer(X_train, feature_names, categorical_features)` → `LimeTabularExplainer`
    ```python
    from lime.lime_tabular import LimeTabularExplainer
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        categorical_features=categorical_feature_indices,
        mode='classification'
    )
    ```
  - `explain_prediction(explainer, model, X_row, num_features=6)` → `list[(feature, weight)]`
    - Works on any black-box model — use for AdversarialDebiasing neural model
  - `get_lime_html(explainer, model, X_row)` → HTML string for embedding in Streamlit with `st.components.html()`

- [ ] **6.3** Write `tests/test_explainability.py` — deferred to Phase 15
  - SHAP additivity: sum of SHAP values for a prediction ≈ model output (within 0.01 tolerance)
  - Protected attribute NOT in top 3 global features after in-processing mitigation
  - LIME: verify top feature weights are non-zero and features are present in X

---

## PHASE 7 — Counterfactual Ghost Engine

- [x] **7.1** Create `src/bias_engine/counterfactual.py`
  - `run_counterfactual_check(model, input_row: dict, config)` → `CounterfactualResult`
    - Creates a copy of `input_row` with protected attr flipped
    - Runs both through `model.predict_proba()` 
    - Returns:
      ```python
      @dataclass
      class CounterfactualResult:
          original_decision: str         # "Approved" / "Denied"
          original_confidence: float
          counterfactual_decision: str
          counterfactual_confidence: float
          is_fair: bool                  # True if same decision
          protected_attr: str
          original_value: str
          flipped_value: str
          confidence_delta: float        # abs(orig_conf - cf_conf)
          risk_level: str                # "CLEAR" / "FLAGGED" / "CRITICAL"
      ```
    - `risk_level` logic:
      - CLEAR: same decision, confidence delta < 0.15
      - FLAGGED: same decision but confidence delta ≥ 0.15 (decision barely survived the flip)
      - CRITICAL: decision changes entirely
  - `run_multi_counterfactual(model, input_row, config)` → `list[CounterfactualResult]`
    - For race (7+ values): run all possible swaps, not just binary
    - Aggregate to a single worst-case risk level

- [ ] **7.2** Write `tests/test_counterfactual.py` — deferred to Phase 15
  - Craft a deliberately biased dummy model (always denies female) → assert CRITICAL
  - Craft a fair model → assert CLEAR
  - Test multi-counterfactual returns correct number of results (one per protected attr value)

---

## PHASE 8 — Intersectional Bias Analysis (INNOVATIVE FEATURE 2)

> **Why this is critical:** Standard fairness tools check sex OR race — never both simultaneously. But a "Black woman" faces discrimination that disappears when you look at race alone or sex alone. This is the concept of intersectionality (Kimberlé Crenshaw, 1989) and is recognized in EU AI Act recitals. Almost no production bias tool handles it. We do it automatically.

- [x] **8.1** Create `src/bias_engine/intersectional.py`
  - `compute_intersectional_metrics(df, y_pred, config)` → `list[IntersectionalResult]`
    - Auto-generates all combinations of protected attribute values
      - E.g., for `sex=[Male,Female]` and `race=[White,Black,Asian,...]`:
        - Creates groups: "White Male", "Black Male", "White Female", "Black Female", etc.
    - For each intersectional group, compute DI and EOD vs the overall privileged baseline
    - Caps at groups with ≥ 30 samples (statistical validity)
    - Returns sorted by DI ascending (worst bias first)

  ```python
  @dataclass
  class IntersectionalResult:
      group_label: str          # "Black Female"
      group_size: int
      approval_rate: float
      disparate_impact: float
      equal_opportunity_diff: float
      severity: str             # "CRITICAL" / "HIGH" / "MEDIUM" / "OK"
  ```

  - `severity` thresholds:
    - CRITICAL: DI < 0.60
    - HIGH: DI 0.60–0.80
    - MEDIUM: DI 0.80–0.90
    - OK: DI ≥ 0.90

- [x] **8.2** `get_heatmap_data(results)` → 2D pivot DataFrame
  - Rows = sex values, Columns = race values, Cell = DI score
  - This drives a Plotly heatmap where red cells = severe intersectional discrimination
- [ ] **8.3** Write `tests/test_intersectional.py` — deferred to Phase 15
  - Run on Adult Income — verify "Black Female" has lower DI than "Black Male" and "White Female" (compounding effect)
  - Verify small groups (<30 samples) are excluded, not silently computed on bad statistics

---

## PHASE 9 — Fairness-Accuracy Pareto Frontier (INNOVATIVE FEATURE 3)

> **Why this wins:** Every organization faces a real tension — "How much accuracy am I sacrificing for fairness?" Nobody shows them the full tradeoff curve and lets them choose. We do. You pick your operating point on the frontier. This is the correct scientific framing of the problem and no commercial tool does it this cleanly.

- [x] **9.1** Create `src/ml_pipeline/pareto.py`
  - `sweep_fairness_frontier(X_train, y_train, S_train, X_test, y_test, S_test, config, steps=10)` → `list[ParetoPoint]`
    - Loops over `constraint_weight` values: `[0.0, 0.1, 0.2, ..., 1.0]` (10 steps)
    - For each `w`, trains a `GridSearch` model with `constraint_weight=w`
    - Records: `ParetoPoint(constraint_weight, accuracy, f1, disparate_impact, equalized_odds_diff)`
  - `find_optimal_operating_point(pareto_points)` → `ParetoPoint`
    - Finds the "knee" of the curve: maximum fairness improvement for minimum accuracy drop
    - Uses the method: biggest DI improvement per 1% accuracy loss

  ```python
  @dataclass
  class ParetoPoint:
      constraint_weight: float
      accuracy: float
      f1: float
      disparate_impact: float
      equalized_odds_diff: float
      is_optimal: bool
  ```

- [ ] **9.2** Write `tests/test_pareto.py` — deferred to Phase 15
  - Verify the 10 sweep points produce monotonically increasing DI and decreasing accuracy (expected trend)
  - Verify `find_optimal_operating_point` returns a `ParetoPoint` with `is_optimal=True`

---

## PHASE 10 — LLM-Powered Audit Narrative (INNOVATIVE FEATURE 4)

> **Why this is the killer feature for real-world sales:** A compliance officer, a board member, or a regulator cannot read a confusion matrix. They need a plain-English explanation of what the bias is, why it exists, what was done to fix it, and what the remaining risks are. We use the Claude API to auto-generate this narrative from structured metrics. This is what turns the tool from a data science toy into a compliance product.

- [x] **10.1** Create `src/narrative/llm_narrator.py` — uses Gemini API (free tier, gemini-2.0-flash)
  - `generate_audit_narrative(audit_result: AuditResult, config: DatasetConfig)` → `str`
    - Builds a structured prompt from the audit metrics:
      ```
      BIAS AUDIT RESULTS for dataset [{dataset_name}]:

      Protected Attributes: {sex, race}
      Target Outcome: {income > 50K}

      PRE-MITIGATION:
      - Disparate Impact (sex): 0.36 [CRITICAL — below legal 0.80 threshold]
      - Statistical Parity Difference: -0.19
      - Equalized Odds: 0.24

      POST-MITIGATION:
      - Disparate Impact (sex): 0.91 [PASS]
      - Equalized Odds: 0.07 [PASS]
      - Accuracy before: 84.2% | After: 83.1% (−1.1%)

      PROXY ANALYSIS:
      Top 3 shadow proxies detected: occupation (MI=0.21), hours-per-week (MI=0.15), ...

      INTERSECTIONAL WORST CASE:
      - Black Female: DI = 0.31 [CRITICAL]
      ```
    - Uses **prompt caching** (Anthropic API `cache_control`) on the static system prompt
    - System prompt instructs Claude to write in a tone suitable for a compliance officer
  - Returns a multi-paragraph plain-English narrative covering:
    1. What bias was found and how severe
    2. Why this bias likely exists (historical context)
    3. What mitigation was applied and how it works in plain terms
    4. Remaining risks and recommendations
    5. Regulatory posture (EU AI Act Article 10, US ECOA)

- [x] **10.2** Add `stream_audit_narrative()` for Streamlit `st.write_stream()` for the Streamlit UI (use `st.write_stream()` to stream the narrative token by token — visually impressive)
- [x] **10.3** Narrative passed as parameter to `generate_pdf_bytes()` — included in report Section 4
- [ ] **10.4** Write `tests/test_narrator.py` — deferred to Phase 15
  - Mock the Anthropic client
  - Verify the prompt contains all key metrics before sending
  - Verify the narrative response is a non-empty string

---

## PHASE 11 — Bias Provenance Tracer (INNOVATIVE FEATURE 5)

> **Why this is novel:** When bias is detected, organizations ask "where did it come from?" Most tools just say "bias exists." We trace it to its source category, which directly determines what fix is appropriate. Treating label bias with reweighing is correct. Treating representation bias with reweighing is wrong — you need more data collection. Knowing the source is actionable.

- [x] **11.1** Create `src/bias_engine/provenance.py`
  - Three bias source categories:
    - **Label Bias** — Historical human decisions were biased. The labels (Y) are the problem.
      - Detection: In the training data, unprivileged group has negative outcomes even when features suggest otherwise
      - Signature: High DI in raw labels + similar feature distributions between groups
      - Fix: Reweighing, Label Flipping
    - **Representation Bias** — Certain groups are underrepresented in training data.
      - Detection: Group size imbalance > 5:1 ratio
      - Signature: Rare group has high variance predictions (model is uncertain)
      - Fix: More data collection, SMOTE/CTGAN augmentation
    - **Feature Bias** — Features encode demographic information as proxies.
      - Detection: High mutual information between features and protected attribute
      - Signature: Proxy Scanner flags features
      - Fix: Feature removal, DisparateImpactRemover

  - `trace_bias_provenance(df, y_pred, proxy_scan_results, config)` → `ProvenanceReport`
    ```python
    @dataclass
    class ProvenanceReport:
        primary_source: str               # "label_bias" | "representation_bias" | "feature_bias"
        secondary_source: str | None
        confidence: float                 # 0.0–1.0
        evidence: list[str]               # bullet points of evidence
        recommended_fixes: list[str]      # ordered by effectiveness
        caution_notes: list[str]          # e.g., "Calibration and DemParity cannot both hold"
    ```

  - `explain_provenance_html(report)` → HTML for Streamlit `st.components.html()`
    - Renders a visual "blame diagram" showing which pipeline stage introduced the bias

- [ ] **11.2** Write `tests/test_provenance.py` — deferred to Phase 15
  - Adult Income: expect "label_bias" as primary (historical wage discrimination)
  - Synthetic dataset with extreme class imbalance: expect "representation_bias"
  - Synthetic dataset with a direct occupation→sex proxy: expect "feature_bias"

---

## PHASE 12 — FastAPI Backend (The Fairness Firewall)

- [x] **12.1** Create `src/api/main.py` — FastAPI app entry point with lifespan event to load the pre-trained model
- [x] **12.2** Create `src/api/routers/audit.py`
  - `POST /audit/upload` — CSV upload, run full pipeline, return `AuditResult` as JSON
    - Runs as FastAPI `BackgroundTask` (non-blocking for large files)
  - `GET /audit/status/{job_id}` — poll job status
  - `GET /audit/result/{job_id}` — retrieve completed result
- [x] **12.3** Create `src/api/routers/inference.py`
  - `POST /predict` — live prediction with full fairness firewall
    - Request: `PredictRequest(features: dict, run_counterfactual: bool = True, run_shap: bool = True)`
    - Response: `PredictResponse(decision, confidence, counterfactual_result, shap_top_features, risk_level)`
- [x] **12.4** Create `src/api/routers/report.py`
  - `GET /report/{job_id}` — generate + return PDF audit report
- [x] **12.5** Add CORS middleware for Streamlit → FastAPI calls
- [ ] **12.6** Write `tests/test_api.py` using `httpx.AsyncClient`
  - Full flow: upload → poll → result → predict → counterfactual → report

---

## PHASE 13 — PDF Compliance Report Generator

- [x] **13.1** Create `src/utils/report_generator.py` using `reportlab`
  - Report sections:
    1. Cover page: "Aequitas Prime Bias Audit Report", date, dataset name, severity badge
    2. Executive Summary: pass/fail table for all 6 metrics (color-coded: red/yellow/green)
    3. Bias Provenance: primary source + recommended fixes
    4. Proxy Variable Scan: top flagged proxies
    5. Before/After Comparison: metrics table + narrative delta ("DI improved by 0.55")
    6. Intersectional Analysis: worst-case groups highlighted
    7. Model Performance: accuracy, F1, AUC — before and after
    8. SHAP Feature Importance: top 10 features as a bar chart (embed Plotly as image)
    9. LLM Audit Narrative (generated by Phase 10)
    10. Counterfactual Ghost Summary: "X% of test predictions were counterfactually stable"
    11. Compliance Statement: EU AI Act Article 10, US Equal Credit Opportunity Act (ECOA), NYC Local Law 144
    12. Signature block for legal/compliance officer
- [x] **13.2** Test PDF generation, verify all sections populate, no blank pages

---

## PHASE 14 — Streamlit UI (The Dashboard)

### Tab 1: Bias X-Ray (Upload & Audit)
- [x] **14.1** File upload widget — accept CSV, preview first 10 rows in `st.dataframe()`
- [x] **14.2** Column selector: protected attribute(s), target column, privileged values, positive label
  - Allow selecting MULTIPLE protected attributes simultaneously (enables intersectional analysis)
- [x] **14.3** "Run Full Bias X-Ray" button → spinner → calls pipeline
- [x] **14.4** Main DI Gauge Chart (Plotly `go.Indicator`, `mode="gauge+number+delta"`)
  - Shows current DI with delta from ideal (1.0)
  - Red/Yellow/Green zones: <0.80 / 0.80–0.90 / >0.90
  - If red: bold alert `st.error("BIAS DETECTED — DI = 0.36. This dataset fails the legal 4/5ths rule.")`
- [x] **14.5** Six-metric summary table with pass/fail badges
  - Use `st.metric()` for each metric with color coding via custom CSS
- [x] **14.6** Proxy Scanner results section:
  - Plotly heatmap of mutual information scores (features × protected attributes)
  - Flagged proxies shown as `st.warning()` cards with recommended actions
- [x] **14.7** Bias Provenance card: "Primary Source: Label Bias | Recommended Fix: Reweighing"

### Tab 2: The Generative Surgeon + Adversarial Blindfold (Mitigation)
- [x] **14.8** Method selector: radio button for "Pre-Processing (Reweighing)" vs "In-Processing (ExponentiatedGradient)" vs "Both"
- [x] **14.9** "Neutralize Bias" button → triggers selected mitigation + retraining
- [x] **14.10** Before/After six-metric comparison table (every metric, not just DI)
- [x] **14.11** Before/After grouped bar chart: approval rate per demographic group
- [x] **14.12** SHAP feature importance chart — global, sorted by mean |SHAP|
  - Highlight protected attribute bar in red if it's in top 5: "WARNING: Sex is still driving decisions"
  - After good mitigation: highlight it in gray at bottom: "Sex: negligible influence (0.003)"
- [x] **14.13** Pareto Frontier interactive chart:
  - Plotly scatter plot: X = Equalized Odds Difference, Y = Accuracy
  - Each point = one `constraint_weight` value from the sweep
  - Star marker on the "optimal" point
  - User can click a point to select it as the deployed model
  - Tagline: "Pick your operating point. We'll deploy that exact model."
- [x] **14.14** Accuracy vs Fairness delta display: "Accuracy: 84.2% → 83.1% | Fairness (DI): 0.36 → 0.91"

### Tab 3: Intersectional Analysis
- [x] **14.15** Plotly heatmap: rows=sex values, columns=race values, cells=DI score
  - Color scale: red (#0.60) → yellow (#0.80) → green (#1.0)
  - Hover tooltip: "Black Female: DI=0.31, Group Size=847, Approval Rate=11%"
- [x] **14.16** Ranked list of worst intersectional groups with severity badges
- [x] **14.17** "What changed after mitigation?" toggle — shows the heatmap before vs after side by side

### Tab 4: The Counterfactual Ghost (Live Simulator)
- [x] **14.18** Dynamic input form — auto-generated from dataset schema
  - Numeric sliders for continuous features (age: 18–90, hours-per-week: 1–99)
  - Selectboxes for categoricals (education, occupation, marital-status)
  - Protected attribute excluded from form (ghost controls it)
- [x] **14.19** "Run Live Prediction" button with custom CSS glow effect on hover
- [x] **14.20** Prediction result box: decision + confidence gauge
- [x] **14.21** Counterfactual Ghost result:
  - CLEAR (green): "Counterfactual Check PASSED. Decision is demographically stable."
  - FLAGGED (yellow): "CAUTION: Same decision, but confidence dropped 22% when demographics changed. Manual review recommended."
  - CRITICAL (red): "ALERT: Decision FLIPS when demographics change. This prediction is discriminatory."
- [x] **14.22** LIME explanation (for this specific prediction): "Top factors: Education (0.38), Hours/week (0.29), Age (0.19)"
- [x] **14.23** LLM narrative stream button: "Generate Plain-English Explanation" → streams Gemini response into `st.write_stream()`

### Tab 5: Audit Report
- [x] **14.24** Display LLM-generated audit narrative in full
- [x] **14.25** "Download Compliance Report (PDF)" button → calls `/report` → `st.download_button()`
- [x] **14.26** Compliance badge section: "This audit covers EU AI Act Article 10, ECOA, NYC Local Law 144"

### UI Polish
- [x] **14.27** Custom CSS: dark theme (`#0d1117` background), accent color (`#7c3aed` purple), glowing elements
- [x] **14.28** Sidebar: project branding (Aequitas Prime logo text), model status indicators, dataset stats
- [x] **14.29** `st.progress()` bars during pipeline execution showing: "Step 2/5: Running proxy scan..."
- [x] **14.30** All charts: consistent dark theme, tooltips, `config={"displayModeBar": False}` for clean look

---

## PHASE 15 — Testing & QA

- [ ] **15.1** `pytest tests/` — all unit tests pass
- [ ] **15.2** End-to-end smoke test: full flow on Adult Income
  - Confirm DI baseline ~0.36, post-mitigation DI > 0.85
  - Confirm intersectional analysis finds "Black Female" as worst group
  - Confirm proxy scanner flags "occupation"
  - Confirm PDF downloads without error
- [ ] **15.3** Edge cases:
  - CSV with no bias (DI ≈ 1.0) → "No Bias Detected" message, no false alarms
  - Protected attribute with one value (degenerate) → graceful error message
  - Missing feature in live predict form → handled with default imputation, not a crash
- [ ] **15.4** Performance: full pipeline on Adult Income (48,842 rows) < 60 seconds total
  - If slow: cap SHAP at 500-row sample, cap Pareto sweep at 5 steps for demo mode

---

## PHASE 16 — Demo Preparation (Hackathon)

- [x] **16.1** Pre-bake demo state:
  - `scripts/prebake_demo.py` — runs full pipeline, saves `models/audit_result_demo.pkl`
  - Streamlit app auto-loads pre-baked state on startup (< 1 second, no live retraining)
  - Demo mode flag: checkbox "Use demo dataset" + auto-load from pkl if present
- [x] **16.2** Demo script written → `DEMO_SCRIPT.md` (4-minute, 8-beat structure with timestamps)
- [x] **16.3** Pitch deck content written → `PITCH_DECK.md` (2-slide: problem + solution+revenue)
- [ ] **16.4** Record 2-minute backup demo video (screen record with narration) — do manually
- [x] **16.5** Deploy to Streamlit Cloud (free tier, < 1GB, Python 3.11)
  - `requirements-streamlit.txt` — TensorFlow excluded (too large for free tier)
  - `.streamlit/config.toml` — dark theme pre-configured
  - `.streamlit/secrets.toml.example` — Gemini key via Streamlit secrets
  - `packages.txt` — libgomp1 for scikit-learn
  - `src/narrative/llm_narrator.py` — reads `st.secrets` on Streamlit Cloud automatically

---

## PHASE 17 — Real-World Productionization (Post-Hackathon)

- [ ] **17.1** Replace Streamlit with React/Next.js 14 (App Router) for production UI
- [ ] **17.2** Add user authentication: Auth0 or Supabase + multi-tenant dataset isolation
- [ ] **17.3** Replace joblib artifact storage with MLflow model registry
- [ ] **17.4** Add async job queue: Celery + Redis for large dataset processing (>500K rows)
- [ ] **17.5** Containerize with Docker, write `docker-compose.yml` for local dev + CI
- [ ] **17.6** Deploy: backend → GCP Cloud Run, frontend → Vercel, Redis → Upstash
- [ ] **17.7** Add CTGAN (SDV library) as a third pre-processing option: synthetic minority augmentation
  - Stronger than reweighing for severe imbalances (minority group < 5% of data)
- [ ] **17.8** Implement temporal bias drift monitoring:
  - Weekly re-audit scheduled job (Celery beat)
  - Slack/email webhook alert when any metric degrades past threshold
  - "Your model's DI dropped to 0.74 this week. Retraining recommended."
- [ ] **17.9** Add support for additional standard datasets: COMPAS (criminal justice), HMDA (mortgage lending), MIMIC-III (healthcare)
- [ ] **17.10** Build SaaS billing layer: Stripe integration, usage metering ($0.001/row audited)
- [ ] **17.11** Generate Model Cards (Google's standard format) automatically from AuditResult
- [ ] **17.12** Partner with a legal firm for EU AI Act compliance co-certification of generated reports

---

## Priority Order for Hackathon MVP

```
Phase 0 → Phase 1 → Phase 2 (metrics only, skip proxy scanner first pass)
→ Phase 3 → Phase 4 (ExponentiatedGradient only — Tier 1)
→ Phase 5 → Phase 6 (SHAP only — skip LIME first pass)
→ Phase 7 (counterfactual) → Phase 8 (intersectional)
→ Phase 9 (pareto) → Phase 10 (LLM narrative)
→ Phase 14 Tabs 1+2+3 (core UI) → Phase 12 (API)
→ Phase 14 Tabs 4+5 → Phase 13 (PDF) → Phase 16 (demo prep)
```

Second pass (if time allows): Phase 2.4 proxy scanner, Phase 6.2 LIME, Phase 4 Tier 2+3, Phase 11 provenance tracer, Phase 15 tests.

---

## Estimated Effort (MVP Core)

| Phase | Task | Effort |
|---|---|---|
| 0 | Scaffold | 30 min |
| 1 | Data Layer | 1 hr |
| 2 | 6 Metrics | 2 hrs |
| 2.4–2.5 | Proxy Scanner | 1.5 hrs |
| 3 | Pre-Processing | 1.5 hrs |
| 4 | In-Processing (ExponentiatedGradient) | 2 hrs |
| 5 | ML Pipeline | 2 hrs |
| 6 | SHAP + LIME | 2 hrs |
| 7 | Counterfactual Ghost | 2 hrs |
| 8 | Intersectional Analysis | 2 hrs |
| 9 | Pareto Frontier | 1.5 hrs |
| 10 | LLM Narrative | 1.5 hrs |
| 11 | Bias Provenance | 1.5 hrs |
| 12 | FastAPI | 2 hrs |
| 13 | PDF Report | 1.5 hrs |
| 14 | Streamlit UI (5 tabs) | 5 hrs |
| 16 | Demo Prep | 1.5 hrs |
| **Total** | | **~31 hrs** |
