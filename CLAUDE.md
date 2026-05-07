# CLAUDE.md
# Primary instruction file for Claude Code — read this at the start of every session.
# Last updated: May 2026 | Version: 2.0
# IMPORTANT: POC is complete. We are now in Phase 1 — Local Dashboard Server.

---

## FIRST THING TO DO EVERY SESSION

1. Read this file completely
2. Read MODELSENTRY.md completely
3. Confirm you understand both before proceeding
4. State the current phase AND the current Phase 1 build step
5. Wait for instruction — do not start building

If unsure about any decision, check MODELSENTRY.md first.
If MODELSENTRY.md does not cover it, ask the human before assuming.

---

## PROJECT CONTEXT

ModelSentry is an ML model monitoring SDK and SaaS. We give data teams early
warning when their production ML models start degrading — so they can fix
problems before they impact the business.

**Current phase: Phase 1 — Local Dashboard Server**

Phase 1 goal: Build a beta-ready product that runs entirely on the customer's
machine. A Python local web server that reads statistical profiles from the SDK,
detects drift, and serves a visual dashboard at localhost:8080.

### What ALREADY EXISTS — do not rebuild or modify without explicit instruction

- `sdk/modelsentry/profiler.py` — statistical profiling, 19 tests ✅
- `sdk/modelsentry/drift.py` — PSI + KS drift detection, 19 tests ✅
- `sdk/modelsentry/monitor.py` — @ms.monitor() decorator, 17 tests ✅
- `sdk/modelsentry/__init__.py` — public API, version 0.1.0 ✅
- `sdk/pyproject.toml` — Poetry config ✅
- `notebooks/poc_validation.ipynb` — validation notebook ✅
- **Total: 55/55 tests passing**

### What we are building in Phase 1

```
modelsentry/
├── sdk/modelsentry/
│   ├── storage.py          ← NEW: profile storage to ~/.modelsentry/
│   ├── server.py           ← NEW: FastAPI local dashboard server
│   └── cli.py              ← NEW: modelsentry serve CLI entry point
├── dashboard/
│   └── index.html          ← NEW: single-page dashboard frontend
└── api/                    ← Reserved for Phase 2 cloud API
```

---

## WORKFLOW — FOLLOW THIS FOR EVERY TASK

### The Explore → Plan → Execute pipeline

**For ANY task that touches more than one file:**

**Step 1 — Explore first**
Use the Explore subagent to read and understand relevant existing files.
Never assume what is in a file — always read it first.
Pay special attention to the existing SDK modules — they define the Profile
and DriftReport objects that the new server will consume.

**Step 2 — Plan in Plan Mode (/plan)**
Produce a specific, file-by-file plan before touching anything.
A good plan names:
- Every file being created or modified
- Every function being added or changed
- The order of operations
- Any risks, edge cases, or design decisions requiring human input

**Step 3 — Present plan for approval**
Do not execute until the human approves the plan.
If the human pushes back, revise the plan — do not improvise during execution.

**Step 4 — Execute in small, reviewable steps**
Build one module at a time. Stop after each module and show the result.
Never build multiple modules in a single execution step.

**Step 5 — Write tests alongside code**
Every new function gets a pytest test immediately after.
Do not move to the next module until tests pass for the current one.

**Step 6 — Run /ultrareview before any commit to main**
Use `/ultrareview` for parallel multi-agent code review before merging.

---

## REPOSITORY STRUCTURE

```
modelsentry/
├── CLAUDE.md                       ← This file
├── MODELSENTRY.md                  ← Product decisions — read before every session
├── .claude/
│   ├── agents/
│   │   ├── statistical-reviewer.md ← Fixed in v2.0 — now reads files correctly
│   │   ├── test-writer.md          ← Fixed in v2.0
│   │   └── security-checker.md    ← Fixed in v2.0
│   └── settings.json
├── sdk/                            ← Python SDK (what customers install)
│   ├── modelsentry/
│   │   ├── __init__.py             ← Public API (COMPLETE — do not modify)
│   │   ├── monitor.py              ← @ms.monitor() decorator (COMPLETE)
│   │   ├── profiler.py             ← Statistical profiling (COMPLETE)
│   │   ├── drift.py                ← Drift detection (COMPLETE)
│   │   ├── client.py               ← Future cloud transmission (Phase 2)
│   │   ├── storage.py              ← Phase 1: local JSON profile storage
│   │   ├── server.py               ← Phase 1: FastAPI dashboard server
│   │   └── cli.py                  ← Phase 1: CLI entry point
│   ├── tests/
│   │   ├── test_profiler.py        ← 19 tests (COMPLETE)
│   │   ├── test_drift.py           ← 19 tests (COMPLETE)
│   │   ├── test_monitor.py         ← 17 tests (COMPLETE)
│   │   ├── test_storage.py         ← Phase 1 (to be written)
│   │   └── test_server.py          ← Phase 1 (to be written)
│   └── pyproject.toml
├── dashboard/
│   └── index.html                  ← Phase 1: single-page frontend
├── api/                            ← Phase 2: cloud API (FastAPI)
├── notebooks/
│   └── poc_validation.ipynb        ← COMPLETE — do not modify
└── docs/decisions/                 ← Architecture decision records
```

---

## CODE STANDARDS — NON-NEGOTIABLE

### Python
- Python 3.11.9 only (pyenv local is set — never change this)
- Type hints on every function signature — no exceptions
- Docstrings on every public method and class
- Maximum function length: 50 lines — if longer, split it
- No wildcard imports (`from x import *` is forbidden)

### Testing
- pytest for all tests
- Tests live in sdk/tests/ mirroring module structure
- Every public function must have at least one test
- Test file naming: test_{module_name}.py
- Run tests with: `cd sdk && poetry run pytest`
- Full suite must stay green after every change — 55 tests minimum

### Dependencies
- All dependencies managed via Poetry (sdk/pyproject.toml)
- Never use pip install directly — always `poetry add`
- No new dependency without explicit human approval
- Currently approved: numpy, pandas, scipy, pytest, pytest-cov, scikit-learn
- Phase 1 additions to request approval for: fastapi, uvicorn, click (or typer), httpx

### Git
- Commit after every working module — not before tests pass
- Commit message format: `type(scope): description`
  Examples: `feat(storage): add local profile JSON persistence`
- Never commit directly to main — always use a feature branch
- Branch naming: `feature/module-name`
- Run `/ultrareview` before merging to main

---

## THE CORE ARCHITECTURAL RULE — NEVER VIOLATE THIS

### Raw data NEVER leaves the customer environment

The SDK computes statistical profiles locally. Only anonymized summaries are
stored or transmitted. Raw feature values and raw predictions are NEVER written
to disk, transmitted over network, or stored in any persistent form.

**In practice for Phase 1:**
- profiler.py accepts DataFrames/arrays and returns Profile objects only
- storage.py writes Profile objects to disk as JSON — never raw data
- server.py reads Profile JSON files and serves statistical summaries only
- dashboard never displays raw feature values — only distributions and scores

**Before every commit to profiler.py, storage.py, or server.py:**
Run the security-checker subagent.

### Dashboard binds to localhost only

The local server must bind to `127.0.0.1` (localhost) only.
Never `0.0.0.0`. Never expose the dashboard to external network interfaces.
This is a security requirement, not a preference.

---

## MODULE RESPONSIBILITIES — PHASE 1 NEW MODULES

### sdk/modelsentry/storage.py
**Purpose:** Persist Profile objects and DriftReports as JSON files locally.
**Storage location:** ~/.modelsentry/{model_id}/
**Input:** Profile objects and DriftReport objects from profiler.py and drift.py
**Output:** JSON files on disk; reads back Profile/DriftReport objects
**Must include:**
- save_profile(profile, model_id, timestamp)
- load_profiles(model_id, limit=N) → list[Profile]
- save_baseline(profile, model_id)
- load_baseline(model_id) → Profile | None
- save_drift_report(report, model_id, timestamp)
- load_drift_reports(model_id, limit=N) → list[DriftReport]
- get_prediction_count(model_id) → int
**Must NOT do:** store raw feature values, store raw predictions
**Dependencies:** profiler.py, drift.py (for dataclass serialization), stdlib json/pathlib

### sdk/modelsentry/server.py
**Purpose:** Serve the local dashboard via FastAPI at localhost:{port}
**Input:** Reads from storage.py (JSON profile files)
**Output:** HTTP endpoints serving JSON + static HTML dashboard
**Must include:**
- GET /api/models — list monitored models and overall health
- GET /api/models/{model_id}/status — current health, last updated timestamp
- GET /api/models/{model_id}/profiles — recent profiles
- GET /api/models/{model_id}/drift — recent drift reports
- GET / — serve dashboard HTML
**Must NOT do:** bind to 0.0.0.0, serve raw feature data, require auth in Phase 1
**Dependencies:** fastapi, uvicorn, storage.py

### sdk/modelsentry/cli.py
**Purpose:** `modelsentry serve` CLI entry point
**Must include:**
- `modelsentry serve --model MODEL_ID --port PORT --host localhost`
- Starts the FastAPI server as a background process
- Optionally opens browser on startup (ask human — open question)
**Dependencies:** click or typer (ask human before adding), server.py

### dashboard/index.html
**Purpose:** Single-page dashboard frontend
**Must include all 7 dashboard requirements from MODELSENTRY.md:**
1. Model health overview (green/yellow/red)
2. Prediction volume counter
3. Per-feature distribution charts vs. baseline
4. Drift scores per feature (PSI + KS), color-coded
5. Last updated timestamp (auto-refresh every 60 seconds)
6. Alert history with timestamps
7. Explicit "all systems nominal" state when no drift
**Technology:** Vanilla HTML/JS + CSS — no build toolchain, no npm, no React
**External CDN allowed:** Chart.js from cdnjs.cloudflare.com for charts

---

## EXISTING MODULE RESPONSIBILITIES (do not change)

### sdk/modelsentry/profiler.py — COMPLETE
Computes statistical profiles locally. Returns Profile objects.
Raw data never retained. 19 tests passing.

### sdk/modelsentry/drift.py — COMPLETE
Compares two Profile objects. Returns DriftReport.
Severity: stable / warning / critical. PSI thresholds: 0.1 / 0.25.
19 tests passing.

### sdk/modelsentry/monitor.py — COMPLETE
@ms.monitor() decorator. Framework agnostic. < 5ms overhead (currently 0.6μs).
Async-safe. Error isolation — monitoring failures never crash predictions.
17 tests passing.

### sdk/modelsentry/client.py — PLACEHOLDER (Phase 2)
Will transmit Profile objects to cloud API.
Do not implement in Phase 1.

---

## WHAT YOU OWN vs WHAT REQUIRES HUMAN APPROVAL

### You own (proceed without asking):
- Writing new functions within the spec above
- Writing tests for code you wrote in this session
- Suggesting refactors with clear rationale and no behavior change
- Fixing bugs in code you wrote in the same session
- Adding docstrings and type hints

### Requires human approval before proceeding:
- Any new external dependency (poetry add anything)
- Any change to existing Profile or DriftReport dataclass structure
- Any change to profiler.py, drift.py, or monitor.py
- Any deviation from the module responsibilities defined above
- Any database schema or storage format decision
- Any API endpoint contract decision
- Any security-relevant decision (binding, auth, encryption)
- Whether to auto-open browser on `modelsentry serve`
- Choice of SMTP vs SendGrid for email alerts

### Never do without explicit instruction:
- Add features not in Phase 1 P0 list
- Transmit raw feature values anywhere
- Modify profiler.py, drift.py, or monitor.py
- Bind local server to 0.0.0.0 or any non-localhost address
- Change the Python version
- Modify .claude/ files
- Push to main branch
- Add AWS-specific, Aurora-specific lock-in services
- Build Phase 2 cloud features during Phase 1

---

## CUSTOM SUBAGENTS

**Important:** Subagents were rewritten in Version 2.0 with corrected tool
permissions. They should now correctly read files. If a subagent report
contains references to code that does not exist in the actual file (wrong
field names, wrong line numbers, wrong imports), treat the report as
unreliable and use Claude Code's own grep-based verification instead.

### statistical-reviewer
**Trigger:** After writing or modifying profiler.py or drift.py
**Purpose:** Validates statistical correctness and confirms no raw data leakage
**How to invoke:** "Run the statistical-reviewer subagent on [filename]"

### test-writer
**Trigger:** After writing any new module
**Purpose:** Writes comprehensive pytest tests for new code
**How to invoke:** "Run the test-writer subagent on [filename]"

### security-checker
**Trigger:** Before any commit touching profiler.py, storage.py, or server.py
**Purpose:** Scans for raw data exposure, hardcoded secrets, wrong binding
**How to invoke:** "Run the security-checker subagent before this commit"

---

## PHASE 1 BUILD ORDER — FOLLOW THIS SEQUENCE

Do not skip steps. Do not build ahead. Each step complete and tested before next.

- [ ] Step 1: storage.py — local JSON profile persistence + tests
- [ ] Step 2: server.py — FastAPI server with API endpoints + tests
- [ ] Step 3: dashboard/index.html — single-page frontend with all 7 requirements
- [ ] Step 4: cli.py — `modelsentry serve` entry point
- [ ] Step 5: Email alert module + tests
- [ ] Step 6: Integration test — full flow from @ms.monitor() to dashboard to alert
- [ ] Step 7: Landing page at modelsentry.com

**Current step:** Step 1 — storage.py

---

## HOW TO START EACH SESSION

Paste this at the start of every Claude Code session:

```
Read CLAUDE.md and MODELSENTRY.md completely. Confirm you understand:
1. Current phase: Phase 1 — Local Dashboard Server
2. POC is complete — do not modify profiler.py, drift.py, or monitor.py
3. Proof of Life requirement: dashboard must show continuous monitoring state
4. Local-first: no cloud, localhost binding only
5. Current build step in Phase 1
Then wait for my instruction before proceeding.
```

---

## DEFINITION OF DONE — PHASE 1

Phase 1 is complete when:
1. `pip install modelsentry && modelsentry serve --model churn-v3` opens a
   live dashboard at localhost:8080 showing monitoring state
2. The dashboard shows continuous monitoring state even when no drift detected
3. A deliberate drift event triggers an email alert
4. The dashboard updates automatically every 60 seconds
5. The full flow (install → monitor → dashboard → alert) works in < 10 minutes
6. All tests pass (55 existing + new Phase 1 tests)
7. A landing page at modelsentry.com has a working waitlist signup

---

*CLAUDE.md v2.0 — Updated after POC completion, May 2026.*
*Update this file when phase changes or new constraints are decided.*
*Do not update this file without human instruction.*
