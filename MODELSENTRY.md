# MODELSENTRY.md
# Project Memory File — Read this before making ANY architectural or product decision.
# Last updated: May 2026 | Version: 2.0
# IMPORTANT: This file was significantly updated after POC completion. Version 1.0 is obsolete.

---

## WHAT THIS FILE IS

This file is the single source of truth for ModelSentry. Every major product,
architectural, pricing, and scope decision is recorded here. When working with
Claude Code or any AI tool, paste this file into context first. It exists to
prevent decisions that contradict what has already been decided.

If you are about to make a decision that conflicts with something in this file,
stop and flag it rather than proceeding. If MODELSENTRY.md does not cover it,
ask the human before assuming.

---

## CURRENT PROJECT STATUS

**Phase:** Phase 1 — COMPLETE. Beta outreach in progress as of May 2026.
**POC:** COMPLETE — all 6 steps done, 55/55 tests passing
**Phase 1:** COMPLETE — 148/148 tests passing. `pip install modelsentry` live on PyPI.
**Next milestone:** 5 beta users actively using product on real models → gate for Phase 2

### What has been built (do not rebuild or redesign without explicit instruction)

- `sdk/modelsentry/profiler.py` — 322 lines, statistical profile computation, 19 tests
- `sdk/modelsentry/drift.py` — 287 lines, PSI + KS drift detection, 19 tests
- `sdk/modelsentry/monitor.py` — 304 lines, @ms.monitor() decorator, 17 tests
- `sdk/modelsentry/storage.py` — local JSON profile persistence (~/.modelsentry/)
- `sdk/modelsentry/server.py` — FastAPI dashboard server, localhost:8080
- `sdk/modelsentry/alerts.py` — SMTP email alert module (AlertConfig + send_drift_alert)
- `sdk/modelsentry/cli.py` — `modelsentry serve` CLI with email alert flags
- `sdk/modelsentry/__init__.py` — clean public API, version 0.1.0
- `sdk/pyproject.toml` — Poetry config, Python 3.11, all Phase 1 deps
- `dashboard/index.html` — single-page dashboard, 7 requirements met
- `landing/index.html` — landing page live at getmodelsentry.com
- `notebooks/poc_validation.ipynb` — 17-cell validation notebook, all 3 assumptions PASS

### CLI interface (current — as shipped)

```bash
modelsentry serve \
  --model churn-v3 \
  --alert-email you@company.com \
  --smtp-host smtp.gmail.com \
  --smtp-port 587 \
  --smtp-user you@gmail.com \
  --smtp-password "app-password"
```

`--alert-email` is optional. `--smtp-host` defaults to `smtp.gmail.com`, `--smtp-port` to `587`.

### POC validation results (confirmed working on synthetic data)

- Assumption 1 PASS: Local statistical profiling works
- Assumption 2 PASS: Framework-agnostic @ms.monitor() works on any Python function
- Assumption 3 PASS: Deliberately introduced drift is detectable
  - age feature (+15yr mean shift): PSI=2.43, severity=critical, KS p=0.0000
  - income feature (shifted distribution): PSI=0.83, severity=critical
  - tenure/country/tier (unchanged): PSI<0.03, severity=stable
- SDK overhead: 0.6μs per call — 8,000× under the 5ms budget

---

## NORTH STAR

> "ModelSentry gives data teams early warning when their production ML models
> start degrading — so they can fix problems before they impact the business."

**North star metric:** Number of models actively monitored in production.

---

## THE PROBLEM WE SOLVE

Alex's team spent three months building a churn prediction model. It went to
production, stakeholders loved it, and everyone moved on. Eight weeks later,
the product team shipped a new onboarding flow that changed three key behavioral
features. Nobody told the data team. The model kept scoring customers —
confidently, silently, wrongly. A month later, CS noticed churn spiking in
segments the model flagged as low risk. Alex spent a week tracing the failure
back to a feature distribution shift that happened 45 days prior. Six weeks of
bad predictions. Decisions made on corrupted scores. Trust in the model — and
the team — damaged.

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

**Economic buyer:** Head of Data / ML Lead — holds team budget, gets blamed
when models fail, approves Growth tier purchases.

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
- A cloud-hosted dashboard yet (Phase 1 is intentionally local-first)

---

## CORE ARCHITECTURAL PRINCIPLES

### PRINCIPLE 1 — RAW DATA NEVER LEAVES CUSTOMER INFRASTRUCTURE

This is non-negotiable. It is both an architectural decision and our primary
sales differentiator.

The SDK computes statistical profiles locally. Only anonymized statistical
summaries are ever stored or transmitted. ModelSentry never stores, sees, or
processes customer raw prediction data or input features.

**Phase 1 (local dashboard):**
```
Customer machine:
  Raw prediction data
    → SDK computes statistical profiles locally (profiler.py)
    → Profiles stored in ~/.modelsentry/{model_id}/ as JSON
    → Local dashboard server reads profiles and renders them
    → Raw data never transmitted, never written to disk
```

**Phase 2 (cloud):**
```
Customer environment:
  Raw prediction data
    → SDK computes profiles locally
    → Transmits ONLY summaries via encrypted HTTPS
    → Cloud stores profiles, serves dashboard, sends alerts
    → Raw data NEVER transmitted
```

**Sales impact:** "Nothing leaves your machine" is the strongest possible
privacy story. It eliminates the #1 enterprise security objection before it
is even raised.

### PRINCIPLE 2 — FRAMEWORK AGNOSTIC

SDK intercepts at the data boundary (inputs/outputs of predict function), not
model internals. Works with any framework. Verified in POC.

### PRINCIPLE 3 — ZERO-CONFIG BASELINE

ModelSentry automatically establishes a statistical baseline from the first N
predictions observed. No manual configuration required.

### PRINCIPLE 4 — PROOF OF LIFE (CRITICAL UX REQUIREMENT)

**Added in Version 2.0. This is non-negotiable.**

A dashboard that only shows content when drift is detected has a fatal UX flaw:
users cannot tell if the tool is working or broken during normal operation.

The dashboard MUST always show:
- How many predictions have been monitored since install
- When the last profile was computed (timestamp proves the system is alive)
- Current feature distributions vs. baseline — even when all stable
- Per-feature drift scores with green/yellow/red indicators
- An intentional "all clear" state — not an empty page

**A silent dashboard = users assume it is broken = uninstall.**

### PRINCIPLE 5 — SIMPLICITY OVER FEATURES

Every feature must answer: "Does this serve Alex's ability to know her model is
breaking before her boss does?" If not, it does not belong in the MVP.

---

## PHASE 1 BUILD — LOCAL DASHBOARD SERVER

### What we are building

A lightweight local web server Alex runs alongside her model. It reads profiles
computed by the SDK, detects drift, and serves a visual dashboard at
localhost:8080. No cloud, no account, no data leaving her machine.

### CLI interface

```bash
modelsentry serve --model churn-v3 --port 8080
```

Alex opens `http://localhost:8080` and sees her model's health.

### Dashboard requirements (must all be present)

1. **Model health overview** — green/yellow/red status per model
2. **Prediction volume** — total predictions monitored, count in last window
3. **Per-feature distributions** — current vs. baseline charts/histograms
4. **Drift scores** — PSI and KS per feature, color-coded by severity
5. **Last updated timestamp** — must update regularly to prove system is alive
6. **Alert history** — recent drift events with timestamps and severity
7. **"All systems nominal" state** — explicitly shown when no drift detected

### Dashboard technology

- **Backend:** FastAPI serving JSON API endpoints
- **Frontend:** Single-page HTML/JS — no React build toolchain for Phase 1
- **Auto-refresh:** Every 60 seconds (configurable)
- **Process:** Runs as background process alongside customer's model
- **Binding:** localhost only — never 0.0.0.0

### Profile storage

```
~/.modelsentry/
  {model_id}/
    baseline.json           ← baseline profile (first N predictions)
    profiles/               ← rolling window of recent profiles
      2026-05-06T14:00.json
      2026-05-06T15:00.json
    drift_reports/          ← computed drift reports
      2026-05-06T15:00.json
```

No database. JSON files. Persists across server restarts.

### Email alert

- Sends when drift crosses threshold
- SMTP or SendGrid (decide during build — ask human)
- Configurable: recipient, threshold, model_id
- Sends directly from customer's machine — no cloud backend needed
- One email per drift event

### Why local-first for Phase 1

1. Zero cloud infrastructure cost or complexity during beta
2. Zero signup friction — install pip, run command, open browser
3. Strongest privacy story — "nothing leaves your machine"
4. Faster to build — no auth, no database, no deployment pipeline
5. Beta feedback tells us what the cloud dashboard needs before we build it

---

## GO-TO-MARKET — BETA PROGRAM

### Decision: Beta users over discovery interviews

Rather than pure discovery interviews, we recruit 5 beta users who actually
install and use the product. Beta users provide richer signal — they find real
bugs, real edge cases, real workflow friction that no interview reveals.

### Beta pitch

"I built something. Install it with pip. It monitors your production models and
shows you a local dashboard of model health in real time. Looking for 5 data
scientists to try it on real models and tell me what is broken. Free forever
for beta users."

### Beta readiness checklist

- [x] Local dashboard server running and showing continuous monitoring state
- [x] Email alert working on drift detection
- [x] getmodelsentry.com live with landing page + waitlist signup
- [x] Full flow demoed end-to-end in under 10 minutes on a real model
- [x] README with clear install and quickstart instructions
- [x] Published to PyPI — `pip install modelsentry` works

**Status: outreach in progress as of 2026-05-07.**

### Legal formation (intentionally deferred)

Decision: LLC and trademark filing are deferred until beta validates the idea.
**Only exception: register modelsentry.com domain immediately (~$15).**

File LLC and trademark when:
- At least 3 beta users are actively using the product on real models
- At least 1 says "I would pay for this"

---

## SDK PUBLIC API — CURRENT STATE

### profiler.py

```python
from modelsentry.profiler import profile, compute_psi
from modelsentry.profiler import Profile, FeatureProfile, Distribution
from modelsentry.profiler import NumericStats, PredictionProfile

prof: Profile = profile(
    features: pd.DataFrame,
    predictions: np.ndarray,
    *,
    n_bins: int = 10,
    top_n_categories: int = 50,
    baseline_edges: dict[str, tuple[float, ...]] | None = None,
) -> Profile

psi: float = compute_psi(
    expected_counts: np.ndarray,
    actual_counts: np.ndarray,
    epsilon: float = 1e-4,
) -> float
```

### drift.py

```python
from modelsentry.drift import detect_drift
from modelsentry.drift import DriftReport, FeatureDriftResult

report: DriftReport = detect_drift(
    baseline: Profile,
    current: Profile,
    *,
    psi_warning: float = 0.1,
    psi_critical: float = 0.25,
    ks_alpha: float = 0.05,
) -> DriftReport

# DriftReport fields:
# .schema_version: str ("1.0")
# .overall_severity: Literal["stable", "warning", "critical"]
# .feature_results: dict[str, FeatureDriftResult]
# .missing_in_current: tuple[str, ...]
# .missing_in_baseline: tuple[str, ...]

# FeatureDriftResult fields:
# .name: str
# .dtype: Literal["numeric", "categorical"]
# .severity: Literal["stable", "warning", "critical"]
# .psi: float
# .ks_statistic: float | None (numeric only)
# .ks_p_value: float | None (numeric only)
# .notes: tuple[str, ...]
```

### monitor.py

```python
import modelsentry as ms

ms.init(
    api_key: str,           # stored, not transmitted in Phase 1
    model_id: str,
    profile_window: int = 100,
    profile_handler: Callable | None = None,
    logger: logging.Logger | None = None,
)

@ms.monitor(model_id: str | None = None)
def predict(features): ...

# Operational helpers (import directly):
from modelsentry.monitor import flush, shutdown, get_latest_profile
```

### Package surface

```python
import modelsentry as ms
# Exports: ms.init, ms.monitor, ms.Profile, ms.DriftReport, ms.__version__
```

---

## KNOWN TECHNICAL DEBT

These are known issues to fix in Phase 2. Do not fix in Phase 1 without
explicit instruction — they are acceptable for beta.

1. **Mutable dicts in frozen dataclasses:** `feature_profiles`, `value_counts`,
   `class_counts` are dicts inside frozen dataclasses. `frozen=True` blocks
   field reassignment but not `.update()`. Fix with `MappingProxyType` in Phase 2.

2. **KS test uses histogram reconstruction:** drift.py reconstructs synthetic
   samples from bin midpoints (Option A, approximate at ~10% per bin width)
   rather than computing the statistic directly from empirical CDFs (Option B,
   exact). Acceptable for Phase 1. Upgrade in Phase 2.

3. **Integer predictions treated as regression:** Low-cardinality integer class
   IDs (0/1) are profiled as regression. String labels work correctly as
   classification. Flagged in profiler.py docstring. Fix in Phase 2.

---

## FEATURE PRIORITY TIERS

### Phase 1 P0 — Must ship for beta (in build order)

| Feature | Status |
|---|---|
| Python SDK with @ms.monitor() | ✅ DONE |
| Local profile storage (~/.modelsentry/) | ✅ DONE |
| FastAPI local server with profile endpoints | ✅ DONE |
| HTML/JS dashboard with continuous monitoring display | ✅ DONE |
| `modelsentry serve` CLI command | ✅ DONE |
| Email alert on drift threshold breach | ✅ DONE |
| Full integration test (install → monitor → dashboard → alert) | ✅ DONE |
| Landing page at getmodelsentry.com with waitlist | ✅ DONE |
| Published to PyPI (`pip install modelsentry` works) | ✅ DONE — v0.1.0 |

### P1 — After beta validates core (do not build yet)

| Feature | Notes |
|---|---|
| Cloud-hosted dashboard | After beta confirms local dashboard works |
| Slack alerting | Include model name, drift score, dashboard link |
| Custom alert thresholds | Per-model sensitivity controls |
| Data volume monitoring | Alert on unexpected prediction volume changes |
| Model versioning | Compare performance across model versions |
| Retraining recommendations | AI-generated suggestions on when to retrain |
| Team collaboration | Multi-user workspace access |
| Baseline auto-detection from rolling window | Currently requires explicit call |

### P2 — After paying customers (do not build yet)

| Feature | Why deferred |
|---|---|
| LLM / generative AI monitoring | Different architecture. Separate market. v2. |
| REST API | When customers outgrow local dashboard |
| SSO / SAML | Enterprise tier only. After SOC 2. |
| On-premise deployment | Defer until customer explicitly pays for it |
| R language SDK | REST API covers this when built |
| SOC 2 Type I | Target month 12 |

---

## PRICING

Per workspace, tiered by number of models monitored.

| Tier | Price | Models | Key features |
|---|---|---|---|
| Starter | Free | 1 | Local dashboard, 7-day retention, no credit card |
| Pro | $99/month | 3 | Email alerts, 90-day retention |
| Growth | $299/month | 10 | Slack + email, custom thresholds, 5 users, 1yr retention |
| Enterprise | Custom | Unlimited | SSO, SOC 2, REST API, dedicated support |

**Note:** Pricing applies to the future cloud product. All beta users get Pro
features free forever regardless of tier.

---

## INFRASTRUCTURE

### Phase 1 — No cloud infrastructure

Local dashboard server runs on customer's machine. No AWS required.
Stack: FastAPI + static HTML/JS + JSON files in ~/.modelsentry/

### Phase 2 — AWS primary, containerized

- Compute: ECS Fargate (not EC2 bare metal)
- Database: RDS PostgreSQL (not Aurora-specific features)
- Storage: S3 (standard, not S3-specific lock-in features)
- Queue: SQS (decouple ingestion from detection)
- CDN: CloudFront
- Apply to AWS Activate Founders before Phase 2: aws.amazon.com/activate
- Containerized from day one: Docker + ECS, cloud agnostic abstractions

---

## SECURITY

### Phase 1 (local — minimal surface area)
- No network transmission at all in Phase 1
- Profiles stored locally — user controls their own data
- Dashboard binds to localhost only — never 0.0.0.0
- No authentication needed for local-only access

### Phase 2 (cloud — required before first paying customer)
- TLS/HTTPS everywhere
- Encryption at rest (RDS + S3)
- MFA required for all accounts
- Tenant data isolation at database query level
- Basic audit logs
- Bcrypt password hashing

### Deferred
- SSO/SAML: Enterprise tier only
- SOC 2 Type I: Month 12
- SOC 2 Type II: Month 18

---

## NON-FUNCTIONAL REQUIREMENTS

| Requirement | Target | Notes |
|---|---|---|
| SDK latency overhead | < 5ms | Currently 0.6μs — maintain this |
| SDK data transmission | Async, non-blocking | Never block predict() |
| Dashboard load time | < 3 seconds | Local server should be near-instant |
| Dashboard refresh | Every 60 seconds (configurable) | Proof of life |
| Server startup | < 5 seconds | `modelsentry serve` must start fast |
| Local server binding | localhost only | Never 0.0.0.0 |
| Availability (Phase 2) | 99.5% uptime | ~43 hrs/year |
| Localization | English + timezone auto-detect | US market MVP |

---

## COMPETITIVE POSITIONING

| Competitor | Price | Gap ModelSentry fills |
|---|---|---|
| Evidently / NannyML | Free (OSS) | No dashboard, no alerting, requires self-hosting expertise |
| WhyLabs | $0–$500/mo | Complex cloud setup, limited features |
| Arize AI | $500–$1,000+/mo | Too expensive for 50–200 person SaaS teams |
| Fiddler AI | $1,500–$2,500+/mo | Enterprise-only, massively over-engineered |
| **ModelSentry** | **$99–$299/mo** | **Simple, affordable, just works. The missing middle.** |

**Differentiators:**
1. Raw data never leaves customer infrastructure (privacy by design)
2. Local-first — no cloud setup, no account, works in 5 minutes
3. Continuous proof of life — always shows monitoring state
4. Priced for practitioners, not procurement teams
5. Framework agnostic

---

## SUCCESS METRICS

### Beta success criteria (gate for Phase 2)
- 5 beta users actively using the product on real models
- At least 3 running on real production data (not toy data)
- At least 1 says "I would pay for this"
- At least 1 real drift event detected on real production data
- Average: dashboard opened at least 3x per week per user

### Activation (Phase 2 cloud product)
- Time from pip install to dashboard showing live data: < 10 minutes
- Activation rate (install → dashboard running): 50%
- Day 7 retention: 50%

### Revenue (Phase 2)
| Month | Customers | MRR |
|---|---|---|
| 3 | 5 | $750 |
| 6 | 20 | $3,500 |
| 12 | 60 | $12,000 |

---

## BUILD PHASES

### Phase 0 — POC ✅ COMPLETE (May 2026)
55/55 tests. Validation notebook. All 3 assumptions proven.

### Phase 1 — Local Dashboard Server ✅ COMPLETE (May 2026)
148/148 tests. `pip install modelsentry` live. getmodelsentry.com live.
All 7 build steps done. Beta outreach in progress.

Gate: 5 beta users actively using on real models → Phase 2

### Phase 2 — Cloud Platform
After beta validates local product:
Cloud API, cloud dashboard, Stripe billing, Slack alerts, multi-user workspaces

### Phase 3 — GA & Growth
Public launch, content marketing, SOC 2, REST API, Enterprise tier

---

## KNOWN ISSUES

### Subagent hallucination (fixed in Version 2.0 subagent files)

During POC build, all three custom subagents hallucinated reports — citing line
numbers, fields, and imports that did not exist in the actual code. Root cause:
YAML frontmatter tool permissions were not correctly granting file read access.

**Fix applied:** Subagent files have been rewritten in Version 2.0 with correct
`allowedTools` syntax. Updated files must be committed before Phase 1 sessions.

**Current workaround:** Until confirmed fixed, treat subagent reports as
supplementary only. Trust Claude Code's own grep-based self-verification.

---

## OPEN QUESTIONS (validate with beta users — do not assume)

1. What is the right default profile_window? (Currently: 100 predictions)
2. Should `modelsentry serve` auto-open the browser on startup?
3. SMTP or SendGrid for email alerts? (Ask human before implementing)
4. Should the dashboard show partial data when a model has < profile_window predictions?
5. Should profiles persist across server restarts? (Current plan: yes, via JSON files)
6. What happens when prediction volume is very high (millions/day)?
7. Will beta users want on-premise cloud deployment? (Defer until asked)

---

## INSTRUCTIONS FOR CLAUDE CODE

### Session start prompt (use every time)

```
Read CLAUDE.md and MODELSENTRY.md completely. Confirm you understand:
1. Current phase: Phase 1 — Local Dashboard Server
2. POC is complete — do not modify profiler.py, drift.py, or monitor.py
3. Proof of Life requirement: dashboard must show continuous monitoring state
4. Local-first architecture: no cloud infrastructure in Phase 1
5. Dashboard binds to localhost only
State the current Phase 1 build step and wait for instruction.
```

### Hard constraints

- SDK must NEVER transmit raw feature values or raw predictions
- SDK overhead must remain < 5ms per call (currently 0.6μs — protect this)
- Never modify profiler.py, drift.py, or monitor.py without explicit instruction
- Dashboard must show continuous state — not only on drift events
- Local server: localhost only, never 0.0.0.0
- Do not add Phase 2 features during Phase 1
- Python 3.11.9 only
- All dependencies via Poetry — never pip install directly
- Do not use Aurora-specific, DynamoDB-specific, or other lock-in services

---

*MODELSENTRY.md v2.1 — Updated post-Phase-1-completion, May 2026.*
*This file is the source of truth. An outdated file means Claude Code builds the wrong thing.*
