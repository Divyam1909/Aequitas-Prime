# Aequitas Prime — 2-Slide Pitch Deck

---

## SLIDE 1 — The Problem

### Title: Biased AI Is Costing the World $500 Billion Annually

**Headline stat:**
> The EU AI Act (2024) mandates bias audits for all high-risk AI systems by August 2026.
> No accessible, automated compliance tool exists.

**Three numbers:**
- **67%** of hiring AI systems exhibit gender bias (MIT Media Lab, 2023)
- **€35M** max fine under EU AI Act for non-compliant AI
- **$300B** US ECOA liability exposure from discriminatory lending AI (CFPB, 2024)

**The Gap:**
| What exists today | What's missing |
|---|---|
| IBM OpenScale — $50K/yr enterprise-only | A tool a startup can afford |
| AIF360 — raw Python library, no UI | A tool a compliance officer can use |
| Manual audits — 6–12 weeks, $200K+ | An automated audit in minutes |
| Tools check sex OR race separately | Intersectional analysis (EU AI Act recital 44 requires it) |

**Visual:** Red X across a confusion matrix. Green checkmark next to "Compliant".

---

## SLIDE 2 — The Solution + Revenue

### Title: Aequitas Prime — Automated Fairness, in Minutes

**Product:** One-click bias audit that detects, explains, mitigates, and certifies AI bias.

**Five Innovations (no other tool has all five):**

| # | Innovation | Why It Wins |
|---|---|---|
| 1 | **Shadow Proxy Scanner** | Detects bias back-doors even after protected column removal |
| 2 | **Intersectional Analysis** | Catches "Black Female" bias that standard tools miss |
| 3 | **Pareto Frontier** | Shows the fairness-accuracy tradeoff curve — you pick the operating point |
| 4 | **Counterfactual Ghost** | Checks every live prediction for demographic stability |
| 5 | **AI Compliance Report** | Gemini-generated plain-English narrative + PDF for regulators |

**Demo (Live):** Upload → Bias Detected (DI=0.36) → Fix (DI=0.91) → PDF Report → Done.

**Revenue Model:**

| Tier | Price | Target |
|---|---|---|
| Free | $0 | Open-source, self-hosted. Community + awareness. |
| Starter | $299/mo | Startups: 500K rows/month, PDF reports, email support |
| Professional | $1,499/mo | Mid-market: 10M rows, API access, custom configs, SLA |
| Enterprise | Custom ($50K+/yr) | Banks, insurers, hospitals. On-prem, EU AI Act co-certification |

**Go-to-Market:**
- Phase 1: Win hackathon → open-source GitHub launch → 1K stars
- Phase 2: EU AI Act compliance wave (Aug 2026 deadline creates urgency)
- Phase 3: Partner with EU AI Act compliance consultancies for referrals
- Phase 4: Expand to COMPAS (criminal justice), HMDA (mortgage), MIMIC-III (healthcare)

**Why Now:**
EU AI Act full enforcement begins August 2026. Every company deploying high-risk AI needs this.
The compliance window is 18 months. Aequitas Prime is ready today.

**Team / Stack:**
Python · FastAPI · Streamlit · AIF360 · Fairlearn · SHAP · Google Gemini · ReportLab · Google Cloud Run

**Live Demo:** [localhost:8501] | **GitHub:** [repo link]
