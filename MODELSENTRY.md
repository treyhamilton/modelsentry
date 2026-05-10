# MODELSENTRY.md
# Project Memory File — Read this before making ANY architectural or product decision.
# Last updated: May 2026 | Version: 2.2
# IMPORTANT: v2.1 had outdated SDK API, wrong domain, missing beta infrastructure.

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

**Phase:** Phase 1 COMPLETE. Beta recruitment in progress as of May 2026.
**POC:** COMPLETE — 55/55 tests
**Phase 1:** COMPLETE — 153/153 tests passing. `pip install modelsentry` live on PyPI v0.1.1
**Next milestone:** 5 beta users actively using product on real models → gate for Phase 2

### What has been built (do not rebuild or redesign without explicit instruction)

**SDK modules:**
- `sdk/modelsentry/profiler.py` — 322 lines, statistical profile computation, 19 tests
- `sdk/modelsentry/drift.py` — 287 lines, PSI + KS drift detection, 19 tests
- `sdk/modelsentry/monitor.py` — framework-agnostic decorator, default local storage, 17+ tests
- `sdk/modelsentry/storage.py` — local JSON profile persistence (~/.modelsentry/)
- `sdk/modelsentry/server.py` — FastAPI dashboard server, localhost:8080
- `sdk/modelsentry/alerts.py` — SMTP email alert module (AlertConfig + send_drift_alert)
- `sdk/modelsentry/cli.py` — `modelsentry serve` CLI with all email alert flags
- `sdk/modelsentry/__init__.py` — clean public API, version 0.1.1
- `sdk/pyproject.toml` — Poetry config, Python 3.11, all Phase 1 deps

**Frontend and landing:**
- `dashboard/index.html` — single-page dashboard, all 7 Proof of Life requirements met
- `landing/index.html` — main landing page at getmodelsentry.com
- `landing/waitlist.html` — dedicated dark-themed beta signup page (Netlify Forms)
- `landing/success.html` — post-submission page with install instructions and personal note
- `netlify.toml` — Netlify publish config and redirect rules

**Infrastructure:**
- `notebooks/poc_validation.ipynb` — 17-cell validation notebook, all 3 assumptions PASS
- `.github/ISSUE_TEMPLATE/bug_report.md` — GitHub bug report template
- `.github/ISSUE_TEMPLATE/feature_request.md` — GitHub feature request template
- `.github/ISSUE_TEMPLATE/config.yml` — disables blank issues, adds email contact link
- `docs/beta-acceptance-email.md` — email template sent within 24hrs of signup
- `docs/beta-feedback-email.md` — email template sent after 2 weeks of use

### CLI interface (current — as shipped in v0.1.1)

```bash
modelsentry serve \
  --model churn-v3 \
  --port 8080 \
  --profile-window 500 \
  --alert-email you@company.com \
  --smtp-host smtp.gmail.com \
  --smtp-port 587 \
  --smtp-user you@gmail.com \
  --smtp-password "app-password"
```

`--alert-email` is optional. `--smtp-host` defaults to `smtp.gmail.com`, `--smtp-port` to `587`.
`--profile-window` is informational (shown in startup banner). Set in ms.init() in model code.

### Quickstart (zero config — just works in v0.1.1)

```python
import modelsentry as ms

ms.init(model_id="churn-v3")  # api_key optional, storage auto-configured

@ms.monitor()
def predict(features_df):
    return model.predict(features_df)
```

Profiles auto-save to `~/.modelsentry/churn-v3/`. Baseline auto-detected on first profile.

### POC validation results (confirmed working on synthetic data)

- Assumption 1 PASS: Local statistical profiling works
- Assumption 2 PASS: Framework-agnostic @ms.monitor() works on any Python function
- Assumption 3 PASS: Deliberately introduced drift is detectable
  - age feature (+15yr mean shift): PSI=2.43, severity=critical, KS p=0.0000
  - income feature (shifted distribution): PSI=0.83, severity=critical
  - tenure/country/tier (unchanged): PSI<0.03, severity=stable
- SDK overhead: 0.6μs per call — 8,000× under the 5ms budget

---

## DEMO ENVIRONMENT

### Demo Environment (added May 2026)

A self-contained demo script for sales calls and personal QA.

**Location:** demos/ folder
**Models:** churn-v3, lead-score-v2, fraud-detect-v4
**Setup:**
  cd demos
  cp .env.example .env        # fill in SMTP_USER + SMTP_PASSWORD
  pip install -r requirements.txt
  pip install -e ../sdk/      # required — use local SDK, not PyPI

**Run modes:**
  python demo.py walkthrough  # sales calls — first time
  python demo.py fast         # sales calls — repeat
  python demo.py slow         # personal QA — 25 min unattended

**Keypress map (fast/walkthrough mode):**
  1 → WARNING drift on churn-v3
  2 → CRITICAL drift on churn-v3
  3 → WARNING drift on lead-score-v2
  4 → CRITICAL drift on lead-score-v2
  5 → WARNING drift on fraud-detect-v4
  6 → CRITICAL drift on fraud-detect-v4
  r → reset all models to stable
  q → quit

**Architecture note:** The demo bypasses the SDK's default profile_handler because
monitor.py does not forward baseline_edges to profile() — see Known Technical Debt
item #4. The demo manages its own raw buffer and calls profile() with aligned edges directly.

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
predictions observed. No manual configuration required. Default storage handler
saves profiles to ~/.modelsentry/ automatically — no profile_handler lambda needed.

### PRINCIPLE 4 — PROOF OF LIFE (CRITICAL UX REQUIREMENT)

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

## TWO-PROCESS MODEL (HOW IT WORKS)

ModelSentry runs as two independent processes. Beta users must understand this.

**Process 1 — SDK (runs inside their model, continuously)**
The @ms.monitor() decorator captures inputs and outputs on every predict() call.
Every 500 predictions (configurable via profile_window), it computes a statistical
profile and saves it to ~/.modelsentry/{model_id}/. On the first profile, it
auto-saves that as the baseline. This runs on a background thread with < 1ms
overhead. Their model code is unaffected.

**Process 2 — Dashboard (run it when you want to check in)**
`modelsentry serve` reads profile files whenever opened. Does not need to run
continuously. Open it when you want to investigate or after receiving a drift
alert email. Auto-refreshes every 60 seconds.

**Storage footprint:** ~5–20 KB per profile. At 10,000 predictions/day with
profile_window=500, that's ~20 profiles/day or ~200–400 KB/day. Very manageable.

---

## GO-TO-MARKET — BETA PROGRAM

### Decision: Open product, relationship program (NOT gated access)

**CRITICAL DECISION — DO NOT REVERSE WITHOUT DISCUSSION**

ModelSentry is fully open. Anyone can `pip install modelsentry` and use it
without permission. The GitHub repo is public. PyPI is public.

The beta program is a **relationship program**, not an access gate. It exists
to identify engaged users so we can build personal relationships with them
before they become paying customers.

**What beta users get that random users don't:**
- Personal onboarding from Trey within 24 hours
- Direct line for bug reports and fast fixes
- Input on what we build next
- Free Pro tier features when cloud product launches
- "Founding user" status

This decision is based on:
1. Every successful comparable tool (Evidently, WhyLabs, NannyML) launched fully open
2. Product-led growth only works if Alex can find and try it without asking permission
3. Open source SDK is standard for developer tools — trust is built by showing the code
4. "Your data never leaves your machine" is more credible when users can verify the code

### Beta communication infrastructure

**Public-facing (website):**
- Waitlist form: `getmodelsentry.com/waitlist` (Netlify Forms, dark-themed)
- Success page: `getmodelsentry.com/success.html` (shows install instructions immediately)
- Email notifications: `getmodelsentry@gmail.com` gets notified on every signup

**Private forms (sent via email to specific users):**
- Beta Onboarding form: `https://tally.so/r/vGEXlD` — sent within 24hrs of signup
- Beta Feedback form: `https://tally.so/r/2E1j0j` — sent after 2 weeks of use

**Email templates:**
- Beta acceptance email: `docs/beta-acceptance-email.md` — personal, from Trey, < 200 words
- Beta feedback email: `docs/beta-feedback-email.md` — sent at 2-week mark

**Bug reporting:**
- GitHub Issues: primary channel (public, templates set up)
- Email: `getmodelsentry@gmail.com` for personal/sensitive questions

### Beta onboarding form fields (tally.so/r/vGEXlD)
Name, job title, company name, company size, models in production,
primary ML framework, current monitoring approach, what would make
ModelSentry a must-have.

### Beta feedback form fields (tally.so/r/2E1j0j)
Setup ease (1-5), drift detected (yes/no/not yet), what almost made
them quit, feature requests, willingness to pay, NPS (recommend yes/maybe/no).

### Beta readiness checklist — ALL COMPLETE

- [x] Local dashboard server running and showing continuous monitoring state
- [x] Email alert working on drift detection
- [x] getmodelsentry.com live with landing page + waitlist signup
- [x] getmodelsentry.com/waitlist — dark-themed form, Netlify Forms
- [x] getmodelsentry.com/success.html — install instructions shown immediately
- [x] Full flow demoed end-to-end in under 10 minutes on a real model
- [x] README with clear install and quickstart instructions
- [x] Published to PyPI — `pip install modelsentry` works (v0.1.1)
- [x] GitHub Issue templates (bug report + feature request)
- [x] Beta Onboarding Tally form created
- [x] Beta Feedback Tally form created
- [x] Email templates written (acceptance + feedback)

**Status: Ready for outreach. Recruiting 5 beta users.**

### Outreach strategy
- Target: warm contacts who are data scientists or ML engineers with models in production
- Approach: personal message, not a template blast
- Goal: 5 people who will install and use it on real models
- Calls: offer 15-minute onboarding call to warm contacts
- Forms: send onboarding Tally form to all beta signups within 24 hours

### Legal formation (intentionally deferred)

File LLC and trademark when:
- At least 3 beta users are actively using on real models
- At least 1 says "I would pay for this"

**Domain registered:** getmodelsentry.com (Namecheap, May 2026)
**Email:** getmodelsentry@gmail.com
**Hosting:** Netlify (connected to GitHub, auto-deploys on push to main)

---

## SDK PUBLIC API — CURRENT STATE (v0.1.1)

### monitor.py — UPDATED in v0.1.1

```python
import modelsentry as ms

ms.init(
    *,
    api_key: str = "",          # optional — stored, not transmitted in Phase 1
    model_id: str,              # required
    profile_window: int = 500,  # changed from 100 to 500 in v0.1.1
    profile_handler: Callable | None = None,  # None = auto-save to ~/.modelsentry/
    storage_path: Path | str | None = None,   # NEW in v0.1.1 — override storage location
    logger: logging.Logger | None = None,
)

@ms.monitor(model_id: str | None = None)
def predict(features): ...

# Operational helpers (import directly):
from modelsentry.monitor import flush, shutdown, get_latest_profile
```

**Key v0.1.1 changes:**
- `api_key` is now optional (default `""`) — no longer causes TypeError
- `profile_window` default changed from 100 to 500
- Default behavior: profiles auto-save to `~/.modelsentry/` without any profile_handler
- Auto-baseline: first profile is automatically saved as baseline if none exists
- `storage_path` parameter added for custom storage location

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
# .detected_at: str | None  (ISO timestamp, added in Phase 1)

# FeatureDriftResult fields:
# .name, .dtype, .severity, .psi, .ks_statistic, .ks_p_value, .notes
```

### Package surface

```python
import modelsentry as ms
# Exports: ms.init, ms.monitor, ms.Profile, ms.DriftReport, ms.__version__
# Version: 0.1.1
```

---

## KNOWN TECHNICAL DEBT

Fix in Phase 2. Do not fix in Phase 1 without explicit instruction.

1. **Mutable dicts in frozen dataclasses:** `feature_profiles`, `value_counts`,
   `class_counts` are dicts inside frozen dataclasses. Fix with `MappingProxyType`.

2. **KS test uses histogram reconstruction (Option A):** Approximate at ~10% per
   bin width. Upgrade to Option B (empirical CDFs) in Phase 2.

3. **Integer predictions treated as regression:** Low-cardinality int class IDs
   (0/1) profiled as regression. String labels work correctly. Fix in Phase 2.

4. **bin_edges never forwarded by monitor.py (P0 — affects all users):**
   _compute_and_dispatch in monitor.py calls profile(df, preds) without
   passing baseline_edges. Result: every numeric feature reports
   severity="warning" with bin-edges mismatch notes even when distributions
   are stable. Critical drift can never escalate past warning for numeric
   features. Fix: load baseline in _compute_and_dispatch, extract bin edges,
   pass baseline_edges= into profile(). Pair fix with an end-to-end
   integration test covering the full @ms.monitor() → save → load →
   detect_drift chain. Demo workaround: manages a parallel raw buffer in
   demo.py — not viable for real customers since profile_handler only
   receives Profile objects, not raw data.

---

## FEATURE PRIORITY TIERS

### Phase 1 P0 — All complete ✅

| Feature | Status |
|---|---|
| Python SDK with @ms.monitor() | ✅ DONE |
| Default local storage (zero config) | ✅ DONE — v0.1.1 |
| FastAPI local server with profile endpoints | ✅ DONE |
| HTML/JS dashboard with continuous monitoring display | ✅ DONE |
| `modelsentry serve` CLI command with email flags | ✅ DONE |
| Email alert on drift threshold breach | ✅ DONE |
| Full integration test (install → monitor → dashboard → alert) | ✅ DONE |
| Landing page at getmodelsentry.com | ✅ DONE |
| Waitlist page + success page | ✅ DONE |
| Published to PyPI (`pip install modelsentry` works) | ✅ DONE — v0.1.1 |
| GitHub Issue templates | ✅ DONE |
| Beta communication forms (onboarding + feedback) | ✅ DONE |
| Beta acceptance email template | ✅ DONE |

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
| Baseline auto-detection from rolling window | Currently auto-set from first window |

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

**Hosting:** Netlify (landing pages)
**Domain:** getmodelsentry.com (Namecheap)
**Email:** getmodelsentry@gmail.com
**Forms:** Netlify Forms (public waitlist) + Tally (private beta forms)
**Repo:** github.com/treyhamilton/modelsentry (public)
**PyPI:** pypi.org/project/modelsentry (v0.1.1)

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
3. Zero config — three lines of Python and it just works
4. Continuous proof of life — always shows monitoring state
5. Priced for practitioners, not procurement teams
6. Framework agnostic — works with any Python ML library

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
153/153 tests. `pip install modelsentry` v0.1.1 live.
getmodelsentry.com live. Beta communication infrastructure complete.
Recruiting 5 beta users.

Gate: 5 beta users actively using on real models → Phase 2

### Phase 2 — Cloud Platform
After beta validates local product:
Cloud API, cloud dashboard, Stripe billing, Slack alerts, multi-user workspaces

### Phase 3 — GA & Growth
Public launch, content marketing, SOC 2, REST API, Enterprise tier

---

## KNOWN ISSUES

### Subagent hallucination (fixed in Version 2.0 subagent files)

During POC build, all three custom subagents hallucinated reports. Root cause:
YAML frontmatter tool permissions were not correctly granting file read access.

**Fix applied:** Subagent files rewritten in Version 2.0 with correct
`allowedTools` syntax. Fix was confirmed working during Phase 1 build —
security-checker successfully read and verified storage.py and alerts.py.

---

## OPEN QUESTIONS (validate with beta users)

1. What happens when prediction volume is very high (millions/day)?
2. Will beta users want on-premise cloud deployment? (Defer until asked)
3. Should the dashboard show partial data when model has < profile_window predictions?
4. What is the right profile_window for high-frequency vs low-frequency models?

---

## INSTRUCTIONS FOR CLAUDE CODE

### Session start prompt (use every time)

```
Read CLAUDE.md and MODELSENTRY.md completely. Confirm you understand:
1. Phase 1 is COMPLETE — 153/153 tests passing, pip install modelsentry v0.1.1 live
2. We are in beta recruitment — do not build Phase 2 features
3. Core SDK modules are frozen: profiler.py, drift.py — do not modify
4. The product is fully open — no gated access
5. Domain is getmodelsentry.com (not modelsentry.com)
Then wait for instruction before proceeding.
```

### Hard constraints

- SDK must NEVER transmit raw feature values or raw predictions
- SDK overhead must remain < 5ms per call (currently 0.6μs — protect this)
- Never modify profiler.py or drift.py without explicit instruction
- Dashboard must show continuous state — not only on drift events
- Local server: localhost only, never 0.0.0.0
- Do not add Phase 2 features during beta period
- Python 3.11.9 only
- All dependencies via Poetry — never pip install directly
- Do not use Aurora-specific, DynamoDB-specific, or other lock-in services
- Domain is getmodelsentry.com — never reference modelsentry.com

---

*MODELSENTRY.md v2.2 — Updated post-Phase-1-completion, beta infrastructure complete, May 2026.*
*This file is the source of truth. An outdated file means Claude Code builds the wrong thing.*
