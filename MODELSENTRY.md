# MODELSENTRY.md
# Project Memory File — Read this before making ANY architectural or product decision.
# Last updated: May 2026 | Version: 1.0

---

## WHAT THIS FILE IS

This file is the single source of truth for ModelSentry. Every major product, architectural, pricing, and scope decision is recorded here. When working with Claude Code or any AI tool, paste this file into context first. It exists to prevent decisions that contradict what has already been decided.

If you are about to make a decision that conflicts with something in this file, stop and flag it rather than proceeding.

---

## NORTH STAR

> "ModelSentry gives data teams early warning when their production ML models start degrading — so they can fix problems before they impact the business."

**North star metric:** Number of models actively monitored in production across all customer workspaces.

---

## THE PROBLEM WE SOLVE

Alex's team spent three months building a churn prediction model. It went to production, stakeholders loved it, and everyone moved on. Eight weeks later, the product team shipped a new onboarding flow that changed three key behavioral features. Nobody told the data team. The model kept scoring customers — confidently, silently, wrongly. A month later, CS noticed churn spiking in segments the model flagged as low risk. Alex spent a week tracing the failure back to a feature distribution shift that happened 45 days prior. Six weeks of bad predictions. Decisions made on corrupted scores. Trust in the model — and the team — damaged.

**ModelSentry would have caught it on day one.**

Three core failure modes we detect:
1. Feature drift — input distributions shift from baseline
2. Data volume drift — prediction volume spikes or drops unexpectedly
3. Silent performance degradation — accuracy drops while confidence stays high

---

## PRIMARY PERSONA

**Name:** Alex (archetypal)
**Title:** Senior Data Scientist
**Company:** 50–200 person SaaS company
**Models in production:** 3–8 models
**Current monitoring:** Manual ad-hoc queries every few days
**Primary pain:** Models degrade silently for weeks before anyone notices
**Budget authority:** Can expense tools under ~$500/mo without approval
**Technical comfort:** High — comfortable with Python SDK and API integration
**Language:** Python
**Stack:** Predominantly AWS, scikit-learn, XGBoost, some PyTorch

**Economic buyer:** Head of Data / ML Lead — holds team budget, gets blamed when models fail, approves Growth tier purchases.

---

## WHAT WE ARE

- A pure technical monitoring tool
- An early warning system for ML model degradation
- A product-led growth SaaS targeting data scientists at 50–200 person SaaS companies
- "Sentry for ML models"

---

## WHAT WE ARE NOT

- An organizational communication bridge (no Jira, GitHub commit monitoring, CI/CD hooks)
- An LLM / generative AI monitoring tool (v2 roadmap item, not v1)
- An end-to-end MLOps platform
- An enterprise-only product
- A raw data storage or processing service

---

## CORE ARCHITECTURAL PRINCIPLES

### PRINCIPLE 1 — RAW DATA NEVER LEAVES CUSTOMER INFRASTRUCTURE

This is non-negotiable. It is both an architectural decision and our primary sales differentiator.

The SDK computes statistical profiles (distributions, PSI scores, KS statistics, null rates, cardinality) locally within the customer's environment. Only compact, anonymized statistical summaries are transmitted to ModelSentry cloud. ModelSentry never stores, sees, or processes customer raw prediction data or input features.

**Practical implementation:**
```
Customer environment:
  Raw prediction data
    → SDK computes statistical profiles locally
    → Transmits ONLY summaries via encrypted HTTPS
    
ModelSentry cloud:
  Receives statistical profiles only
  Runs drift detection against stored baselines
  Serves dashboard and alerts
  NEVER receives raw data
```

**Why this matters for sales:** When Alex's Head of Data asks "what data does this send externally?" the answer is "only statistical summaries — your raw data never leaves your environment." This closes the security objection instantly.

### PRINCIPLE 2 — FRAMEWORK AGNOSTIC

The SDK intercepts at the data boundary (inputs and outputs of the predict function), not at model internals. This means it works with any framework — scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow, or any custom NumPy implementation — without requiring framework-specific integration code.

### PRINCIPLE 3 — ZERO-CONFIG BASELINE

ModelSentry automatically establishes a statistical baseline from the first N days of production traffic. No manual configuration of baselines is required. This is a P0 feature — if Alex has to manually configure baselines, we've broken the "just works" promise.

### PRINCIPLE 4 — SIMPLICITY OVER FEATURES

Every feature must answer: "Does this serve Alex's ability to know her model is breaking before her boss does?" If not, it doesn't belong in the MVP. Scope creep is the primary risk to shipping.

---

## SDK DESIGN

**Language:** Python only at launch. REST API in v1.1 to cover all other languages.

**Target integration time:** Under 10 minutes from `pip install` to first profile transmitted.

**Intended usage pattern:**
```python
import modelsentry as ms

ms.init(api_key="your-key", model_id="churn-v3")

@ms.monitor()
def predict(features):
    return model.predict(features)  # works with ANY framework
```

Alternative for teams that can't use decorators:
```python
ms.log(model_id="churn-v3", features=X, predictions=y_pred)
```

**What the SDK must NOT do:**
- Transmit raw feature values
- Transmit raw predictions
- Add more than 5ms latency to prediction calls
- Require changes to model training code
- Require knowledge of which framework was used

---

## FEATURE PRIORITY TIERS

### P0 — Launch blockers (all 6 must ship before GA)

| Feature | Key requirement |
|---|---|
| Feature drift detection | PSI + KS test. No labels required. Fires alert when drift score exceeds threshold. |
| Performance degradation alerts | Tracks accuracy, F1, AUC, RMSE. Supports classification and regression. |
| Python SDK | pip installable. One decorator instruments any model. Raw data stays local. |
| Monitoring dashboard | Shows health, drift scores, alert history across all models. Loads < 3 seconds. |
| Baseline auto-detection | Automatically built from first N days. Zero manual config. |
| Multi-model support | 3–8 models per workspace from a single dashboard. |

### P1 — Should have (build after P0 is stable, before GA if possible)

| Feature | Notes |
|---|---|
| Slack alerting | Include model name, drift score, affected features, dashboard link |
| Email alerting | Configurable per model per user |
| Data volume monitoring | Alert on unexpected spikes/drops vs. baseline |
| Model versioning | Track performance across versions, compare after retraining |
| Custom alert thresholds | Per-model sensitivity controls |
| Retraining recommendations | AI-generated suggestions based on drift severity and trend |
| Team collaboration | Multi-user workspace access, not full incident management |

### P2 — Strategic roadmap (do NOT build until paying customers request it)

| Feature | Why deferred |
|---|---|
| LLM / generative AI monitoring | Different architecture entirely. Separate market. v2. |
| REST API | Build when customers outgrow the dashboard. v1.1. |
| SSO / SAML | Enterprise tier only. After SOC 2. |
| On-premise deployment | Defer until a customer explicitly pays for it. |
| R language SDK | Academic market only. REST API covers this. |

---

## PRICING

### Model
Per workspace, tiered by number of models monitored.
Primary dimension: models monitored (not seats, not usage volume).

### Tiers

| Tier | Price | Models | Purpose |
|---|---|---|---|
| Starter | Free | 1 | Zero-friction trial. No credit card. 7-day data retention. |
| Pro | $99/month | 3 | Individual Alex. Expensable without approval. 90-day retention. |
| Growth | $299/month | 10 | Team purchase. Head of Data approves. 1-year retention. 5 users. |
| Enterprise | Custom | Unlimited | SSO, SOC 2, REST API, dedicated support. Procurement-grade. |

### What's included at each tier
- **Starter:** Dashboard, baseline auto-detection, drift detection, 7-day retention
- **Pro:** Everything in Starter + email alerting, 90-day retention, 3 models
- **Growth:** Everything in Pro + Slack alerting, custom thresholds, team access (5 users), 1-year retention
- **Enterprise:** Everything in Growth + SSO, SOC 2 documentation, REST API, dedicated support, custom retention

---

## INFRASTRUCTURE

### Cloud provider
**AWS primary.** Reasons: Alex's existing infrastructure is predominantly AWS. Largest partner ecosystem. AWS Marketplace is a viable future sales channel.

**Apply to AWS Activate Founders program immediately** — provides $5,000–$100,000 in free credits depending on track.

### Architecture requirements
- **Containerized from day one** — Docker + ECS or Kubernetes. Enables future cloud portability without rewriting.
- **Use standard managed services** — PostgreSQL on RDS (not Aurora-specific features), standard S3-compatible storage. Avoid vendor lock-in at the service level.
- **Cloud agnostic abstractions** — wrap AWS-specific services behind interface layers so switching clouds is a configuration change, not a rewrite.

### Key AWS services
- **Compute:** ECS Fargate or EC2 (start small, autoscale)
- **Database:** RDS PostgreSQL — stores profiles, drift scores, alerts, users, workspace config
- **Storage:** S3 — statistical profile archive, dashboard assets
- **Queue:** SQS — decouples profile ingestion from drift detection processing
- **CDN:** CloudFront — dashboard delivery

---

## SECURITY

### Launch baseline (all required before first paying customer)
- TLS/HTTPS everywhere — all API endpoints, all dashboard traffic
- Data encrypted at rest — RDS encryption enabled, S3 server-side encryption
- MFA required — enforced for all user accounts
- Tenant isolation — each customer's data is strictly siloed at the database level
- Basic audit logs — who logged in, when, what changed
- Secure password requirements — minimum length, complexity, bcrypt hashing

### Deferred (do not build for MVP)
- SSO / SAML — Enterprise tier only
- SOC 2 Type I — Target month 12, use Vanta or Drata for automation
- SOC 2 Type II — Target month 18

### The privacy advantage
Because raw data never leaves customer infrastructure, ModelSentry's compliance surface area is dramatically smaller than tools that store raw prediction data. We never handle customer PII, PHI, or raw feature data. This simplifies our SOC 2 scope and eliminates HIPAA exposure for the MVP.

---

## NON-FUNCTIONAL REQUIREMENTS

| Requirement | Target | Notes |
|---|---|---|
| Availability | 99.5% uptime | ~43 hours downtime/year acceptable at early stage |
| Dashboard load time | < 3 seconds | Instrument from day one with real monitoring |
| SDK latency overhead | < 5ms | Profile computation must not slow down prediction calls |
| SDK data transmission | Async, non-blocking | Never block the predict() call waiting for transmission |
| Localization | English + timezone auto-detect | US market MVP. Timezone prevents alert timestamp confusion. |
| Framework support | Framework agnostic | Intercept at data boundary, not model internals |

---

## COMPETITIVE POSITIONING

| Competitor | Price | Gap ModelSentry fills |
|---|---|---|
| Evidently / NannyML | Free (open source) | No hosted dashboard, no alerting, requires self-hosting expertise |
| WhyLabs | $0–$500/mo | Limited features, complex setup for small teams |
| Arize AI | $500–$1,000+/mo | Too expensive for 50–200 person SaaS teams |
| Fiddler AI | $1,500–$2,500+/mo | Enterprise-only, massively over-engineered |
| **ModelSentry** | **$99–$299/mo** | **Simple, affordable, just works. The missing middle.** |

**Our differentiators:**
1. Raw data never leaves customer infrastructure (privacy by design)
2. Zero-config baseline auto-detection (fastest time to value)
3. Priced for practitioners, not procurement teams
4. Framework agnostic — works with anything

---

## SUCCESS METRICS

### North star
Number of models actively monitored in production.

### Activation (most important early metric)
- Time from signup to first model connected: **< 10 minutes**
- Time from signup to first drift alert or baseline confirmation: **< 24 hours**
- Activation rate (signup → model connected): **40%**
- Day 7 retention of activated users: **50%**

### Revenue targets
| Month | Paying customers | MRR |
|---|---|---|
| 3 | 5 | $750 |
| 6 | 20 | $3,500 |
| 12 | 60 | $12,000 |

---

## BUILD PHASES

### Phase 0 — Validation Sprint (Weeks 1–8)
**No production code.** Research and discovery only.
- Conduct 10 customer discovery interviews
- Build landing page to gauge signup intent
- Complete legal formation (LLC, trademark, domain)
- Build local proof-of-concept SDK
- **Gate:** 3 of 10 interviews produce "I would pay for that" → proceed to Phase 1

### Phase 1 — P0 MVP (Weeks 9–20)
Build all 6 P0 features to production standard.
- Python SDK with local statistical profile computation
- Ingestion API and time-series storage on AWS
- Drift detection engine (PSI, KS test, baseline comparison)
- Baseline auto-detection
- Multi-model dashboard
- Performance degradation alerting
- Security baseline

### Phase 2 — Beta & P1 Features (Weeks 21–32)
- Launch to 5–10 beta users
- Build P1 features in response to real usage feedback
- Stripe billing integration
- **Gate:** 3+ beta users convert to paid → general availability

### Phase 3 — GA & Growth (Month 9+)
- Public launch
- Content marketing
- SOC 2 Type I initiation
- REST API (v1.1)
- Enterprise tier

---

## LEGAL & BUSINESS

- **Entity:** ModelSentry LLC — file in North Carolina via sosnc.gov ($125)
- **Domain:** modelsentry.com — register immediately
- **Trademark:** USPTO Class 042, Intent to Use basis — file within first 2 weeks
- **Social handles:** @ModelSentry on X, LinkedIn company page, GitHub org (modelsentry)
- **AWS Activate:** Apply immediately at aws.amazon.com/activate

---

## OPEN QUESTIONS (do not assume answers — validate with customers)

1. What should the default baseline window be? (Hypothesis: 14 days)
2. What statistical tests should be defaults for different model types?
3. Should retraining be triggered automatically or always require human approval? (Hypothesis: always human approval in v1)
4. What onboarding flow achieves < 10 minute time-to-first-model?
5. Will customers want self-hosted / on-premise? (Defer until a customer pays for it)

---

## INSTRUCTIONS FOR CLAUDE CODE

When using this file with Claude Code, begin every session with:

> "Read MODELSENTRY.md before proceeding. Do not make architectural decisions that contradict decisions recorded in that file. If you are unsure whether a decision conflicts with something in MODELSENTRY.md, ask rather than assume."

Key constraints Claude Code must respect:
- The SDK must NEVER transmit raw feature values or raw predictions
- Statistical profiles are computed locally in the customer environment
- The SDK must be framework agnostic — no framework-specific imports at the top level
- All API endpoints must use HTTPS
- Tenant data isolation is mandatory at the database query level
- Do not add features not listed in P0 or P1 without explicit approval
- Do not use Aurora-specific, DynamoDB-specific, or other vendor-lock-in services without approval

---

*MODELSENTRY.md — Keep this file updated as decisions change. The cost of an outdated decision file is an AI agent that builds the wrong thing.*
