# CLAUDE.md
# Primary instruction file for Claude Code — read this at the start of every session.
# Last updated: May 2026

---

## FIRST THING TO DO EVERY SESSION

1. Read this file completely
2. Read MODELSENTRY.md completely
3. Confirm you understand both before proceeding
4. State which phase we are in (POC / Phase 1 / Phase 2 / Phase 3)

If you are unsure about any decision, check MODELSENTRY.md before proceeding.
If MODELSENTRY.md doesn't cover it, ask the human before assuming.

---

## PROJECT CONTEXT

ModelSentry is an ML model monitoring SDK and SaaS. We give data teams early warning
when their production ML models start degrading — so they can fix problems before they
impact the business.

Current phase: **POC — Proof of Concept**

POC goal: Prove three technical assumptions before building anything production-grade:
1. Local statistical profiling works (features, distributions, PSI, KS test)
2. Framework-agnostic instrumentation is achievable (@ms.monitor() decorator)
3. Deliberately introduced drift is detectable

---

## WORKFLOW — FOLLOW THIS FOR EVERY TASK

### The Explore → Plan → Execute pipeline

For ANY task that touches more than one file:

**Step 1 — Explore first**
Before writing any code, use the Explore subagent to read and understand the
relevant existing files. Never assume what's in a file — always read it first.

**Step 2 — Plan in Plan Mode**
Use /plan to produce a specific, file-by-file plan before touching anything.
A good plan names:
- Every file being created or modified
- Every function being added or changed
- The order of operations
- Any risks or edge cases

**Step 3 — Present plan for approval**
Do not execute until the human approves the plan. If the human pushes back,
revise the plan — do not improvise during execution.

**Step 4 — Execute in small steps**
Build one module at a time. Stop after each module and show the result.
Never build multiple modules in a single execution step.

**Step 5 — Write tests alongside code**
Every new function gets a pytest test written immediately after. Do not move
to the next module until tests pass for the current one.

---

## REPOSITORY STRUCTURE

```
modelsentry/
├── CLAUDE.md                   ← This file
├── MODELSENTRY.md              ← Product decisions — read before every session
├── .claude/
│   ├── agents/                 ← Custom subagents
│   │   ├── statistical-reviewer.md
│   │   ├── test-writer.md
│   │   └── security-checker.md
│   └── settings.json
├── sdk/                        ← The Python SDK (what customers install)
│   ├── modelsentry/
│   │   ├── __init__.py
│   │   ├── monitor.py          ← @ms.monitor() decorator
│   │   ├── profiler.py         ← Statistical profile computation
│   │   ├── drift.py            ← Drift detection logic
│   │   └── client.py           ← Transmits profiles to API
│   └── tests/
├── api/                        ← Backend API (FastAPI) — Phase 1
├── dashboard/                  ← Frontend — Phase 1
├── notebooks/                  ← POC validation notebooks
└── docs/decisions/             ← Architecture decision records
```

---

## CODE STANDARDS — NON-NEGOTIABLE

### Python
- Python 3.11.9 only (pyenv local is set — do not change this)
- Type hints on every function signature — no exceptions
- Docstrings on every public method and class
- Maximum function length: 50 lines — if longer, split it
- No wildcard imports (`from x import *` is forbidden)

### Testing
- pytest for all tests
- Tests live in sdk/tests/ mirroring the module structure
- Every public function must have at least one test
- Test file naming: test_{module_name}.py
- Run tests with: `cd sdk && poetry run pytest`

### Dependencies
- All dependencies managed via Poetry (pyproject.toml)
- Never use pip install directly — always `poetry add`
- No new dependency without explicit human approval
- Approved POC dependencies: numpy, pandas, scipy, pytest

### Git
- Commit after every working module — not before
- Commit message format: `type(scope): description`
  Examples: `feat(profiler): add PSI computation`, `test(drift): add KS test coverage`
- Never commit to main directly — always work on a feature branch
- Branch naming: `feature/module-name` e.g. `feature/profiler`

---

## THE CORE ARCHITECTURAL RULE — NEVER VIOLATE THIS

### Raw data NEVER leaves the customer environment

The SDK computes statistical profiles locally. Only anonymized statistical
summaries are transmitted. Raw feature values and raw predictions are NEVER
transmitted anywhere.

**In practice this means:**
- profiler.py accepts DataFrames/arrays and returns Profile objects only
- client.py transmits Profile objects only — never raw data
- No function should serialize raw feature values to any output
- No logging of raw feature values anywhere in the codebase

**Before every commit to profiler.py or client.py, run the security-checker subagent.**

---

## MODULE RESPONSIBILITIES

### sdk/modelsentry/profiler.py
**Purpose:** Compute statistical profiles from raw data locally.
**Input:** pandas DataFrame or numpy array of features + predictions
**Output:** A Profile object containing statistical summaries only
**Must include:** feature distributions, PSI scores, null rates, cardinality, basic stats
**Must NOT do:** transmit data, write to disk, import from monitor.py or client.py
**Dependencies:** numpy, pandas, scipy only

### sdk/modelsentry/drift.py
**Purpose:** Detect drift by comparing two Profile objects.
**Input:** baseline Profile + current Profile
**Output:** DriftReport object with per-feature drift scores and overall status
**Must include:** PSI test, KS test, overall drift classification (none/warning/critical)
**Must NOT do:** access raw data, transmit anything, import from monitor.py or client.py
**Dependencies:** numpy, scipy only (no pandas needed here)

### sdk/modelsentry/monitor.py
**Purpose:** Framework-agnostic instrumentation decorator.
**Input:** Any Python function that accepts features and returns predictions
**Output:** Decorated function with identical signature + side effect of capturing profiles
**Must include:** @ms.monitor() decorator, async-safe profile capture, <5ms overhead
**Must NOT do:** block the predict() call, raise exceptions that affect prediction flow
**Dependencies:** profiler.py only (not client.py — keep transmission separate for POC)

### sdk/modelsentry/__init__.py
**Purpose:** Public API surface for the SDK.
**Exports:** init(), monitor decorator, and nothing else for POC
**Must be:** clean, minimal, what a customer sees when they `import modelsentry`

### sdk/modelsentry/client.py
**Purpose:** Transmit Profile objects to ModelSentry API.
**NOT needed for POC** — during POC, profiles are saved locally or logged to console.
**Build in Phase 1** when the API exists to receive them.

---

## WHAT YOU OWN vs WHAT REQUIRES HUMAN APPROVAL

### You own (proceed without asking):
- Writing new functions within the spec above
- Writing tests for code you just wrote
- Suggesting refactors with clear rationale
- Fixing bugs in code you wrote in the same session
- Adding docstrings and type hints

### Requires human approval before proceeding:
- Any change to how Profile objects are structured
- Any change to what data is captured in profiler.py
- Any new external dependency (poetry add anything)
- Any deviation from the module responsibilities above
- Any database schema decision
- Any API contract decision
- Anything that touches client.py transmission logic

### Never do without explicit instruction:
- Add features not in MODELSENTRY.md P0 or P1
- Transmit raw feature values anywhere
- Change the Python version
- Modify .claude/ files
- Push to main branch
- Add AWS-specific, Aurora-specific, or other lock-in services

---

## CUSTOM SUBAGENTS — WHEN TO USE THEM

### statistical-reviewer
**Trigger:** After writing or modifying profiler.py or drift.py
**Purpose:** Validates statistical implementation correctness and confirms no raw data leakage
**How to invoke:** "Run the statistical-reviewer subagent on profiler.py"

### test-writer
**Trigger:** After writing any new module or function
**Purpose:** Writes comprehensive pytest tests for new code
**How to invoke:** Automatically triggered after each module completion

### security-checker
**Trigger:** Before every git commit that touches profiler.py or client.py
**Purpose:** Scans for raw data transmission, hardcoded secrets, missing encryption
**How to invoke:** "Run the security-checker subagent before this commit"

---

## POC BUILD ORDER — FOLLOW THIS SEQUENCE

Do not skip steps. Do not build ahead. Each step must be complete and tested
before the next begins.

- [ ] Step 1: pyproject.toml — Poetry project setup for the SDK
- [ ] Step 2: profiler.py — Statistical profile computation + tests
- [ ] Step 3: drift.py — Drift detection logic + tests
- [ ] Step 4: monitor.py — Framework-agnostic decorator + tests
- [ ] Step 5: __init__.py — Clean public API surface
- [ ] Step 6: POC validation notebook — Demo that catches deliberate drift

**Current step:** Step 1 — pyproject.toml

---

## HOW TO START EACH SESSION

Paste this at the start of every Claude Code session:

```
Read CLAUDE.md and MODELSENTRY.md. Confirm you understand the project context,
current phase, and constraints. State the current POC build step we are on.
Then wait for my instruction before proceeding.
```

---

## DEFINITION OF DONE — POC

The POC is complete when:
1. A scikit-learn model can be instrumented with @ms.monitor() in 2 lines of code
2. Deliberately introduced feature drift is detected and reported
3. All tests pass (pytest green)
4. A Jupyter notebook demonstrates the full flow end to end
5. Raw data is confirmed never transmitted (security-checker passes)
6. The demo can be run in front of a data scientist in under 10 minutes

---

*CLAUDE.md — Update this file when project phase changes or new constraints are decided.*
*Do not update this file without human instruction.*
