# AI Drift Investigator — validation plan

> **Status:** experiment only. Do not change `sdk/modelsentry/` or ship product code from this work.
>
> **Governing contract:** `.claude/agents/drift-investigator.md` and the two schemas under
> `.claude/agents/schemas/` (merged in PR #13) are authoritative. Where this plan and the
> contract disagree, the contract wins and this plan is wrong. Nothing here redefines the
> taxonomy, output shape, tool surface, or limits.
>
> **Downstream:** `IMPROVEMENT_LOOP_PLAN.md` (PR #18) builds on this one — it turns evaluated runs
> into curated regression cases and promoted revisions, and separately into owner-approved action
> proposals. The boundary: **this plan decides whether the investigator ships at all; that one
> decides how it improves afterward.** It owns the span contract, the failure taxonomy, the
> promotion gates, and the paired-bootstrap procedure. This plan adopts all four rather than
> defining parallel versions, so the artifacts this experiment produces feed that loop unchanged.

## Decision this experiment makes

PR #13 settled *what* the investigator is: a bounded evidence investigator over one persisted drift event, with a deterministic evidence layer, four read-only tools, runtime-owned ranking, and asynchronous execution. The architecture question is closed.

What remains open, and what this experiment answers:

1. Does output produced under this contract actually improve an operator's next decision?
2. Is the contract implementable — do the limits (8 tool calls, 24 windows, 5 features, 3 hypotheses) leave enough room to produce useful results?
3. What does each part of the contract contribute, so its implementation cost is justified?

Set before the holdout run:

- Experiment DRI: TBD
- Human-review DRI: TBD
- Decision date: TBD
- API spend cap: TBD

Do not run the holdout evaluation while any field is TBD.

---

## 1. What the investigator is

A **bounded evidence investigator, not a root-cause detector**.

ModelSentry's deterministic layer knows which features drifted and when. That is never the model's job. Per the contract, code owns every calculation — PSI, KS, severity, observed onset, left-censoring, bounded persistence, history coverage, direction, missingness change, category-share change, onset proximity, evidence strength, and final rank. The model receives typed fields and must not replace them with estimates.

Given that, the model: identifies the important observed changes; returns at most three plausible ranked hypotheses; cites the exact aggregate evidence for and against each; recommends allowlisted checks; and states what the evidence cannot establish. It must never describe a hypothesis as a confirmed root cause.

Consequences that shape every section below: exact category matching is a weak metric (the contract's own `unknown` category exists for observationally equivalent explanations); unsupported certainty is the primary failure mode; and the model cannot win on evidence strength, because the runtime computes it.

---

## 2. Output contract

**Do not restate the schema here.** The model produces fields validated against
`drift-investigation-candidate.schema.json`; the runtime injects and validates the rest against
`drift-investigation-result.schema.json`. Build against those files and let them fail the run.

Fixed vocabulary the scenario catalog and evaluators must use:

- `assessment`: `drift_detected` · `stable` · `insufficient_evidence`
- `category` (8): `upstream_data_change` · `population_mix_change` · `missingness_or_pipeline_failure` · `category_or_schema_change` · `behavioral_change` · `model_or_serving_change` · `seasonal_or_scheduled_change` · `unknown`
- `evidence_family` (8): `distribution_drift` · `location_change` · `missingness_change` · `category_share_change` · `bounded_persistence` · `history_coverage` · `onset_proximity` · `schema_change`
- `metric` (11): `psi` · `ks_p_value` · `severity` · `mean_delta` · `mean_delta_pct` · `null_rate_delta_pp` · `category_share_delta_pp` · `bounded_persistence` · `history_coverage` · `onset_proximity` · `feature_presence`
- `direction`: `increase` · `decrease` · `unchanged` · `not_applicable`
- `check_code` (5): `review_upstream_schema` · `compare_population_segments` · `verify_missingness_pipeline` · `review_deployment_change` · `compare_scheduled_window`

Caps: 12 observations · 3 hypotheses · 5 evidence refs per hypothesis · 3 next checks per hypothesis · 5 limitations · 300 chars per free-text field · `tool_calls_used` ≤ 8.

Unavailable values are `{"kind": "unavailable", "status": "<reason>"}` — never a silent zero or omission.

> **Taxonomy change:** this 8-category set replaces the 7-class taxonomy from the earlier brief. Every scenario label and evaluator must be written against the merged set.

---

## 3. Variants

The contract has decided the shipping architecture, so the variants are no longer an architecture bake-off. They are **ablations** that measure what each part of the contract contributes — which is what justifies building it.

| | Variant | Input | Measures |
|---|---|---|---|
| **A** | Snapshot one-shot | baseline profile, final profile, final drift report | floor: what a naive call produces |
| **B** | History one-shot | baseline plus the bounded window timeline, no derived fields | contribution of raw history |
| **C** | Evidence-layer one-shot | the full deterministic evidence payload, pasted in one call | contribution of the deterministic layer |
| **D** | Contract agent | evidence layer reached through the four bounded tools | contribution of adaptive retrieval within the 8-call budget |

**D is the contracted design.** C is its closest legal sibling and the one to beat: same evidence, one call, no tool budget, far lower latency. If C matches D, the tool layer is unnecessary complexity and the finding is worth having before it is built.

A and B are **measurement only, not shippable** — both require the model to reason over evidence the contract says code must compute. Keep them to quantify what the deterministic layer buys, and label them as non-conforming in the results so nobody mistakes a strong A score for a shortcut.

---

## 4. The deterministic evidence layer

This is now the **largest work item in the plan**, and it is mandatory for C and D rather than being one variant's preprocessing step. It is also, per the contract's blockers, code ModelSentry does not yet have.

Implement exactly as specified in the contract's *Required history calculations*: `drift_state`, `observed_onset_window`, `onset_left_censored`, `bounded_persistence`, `history_coverage`, `numeric_mean_delta`, `numeric_mean_delta_pct`, `null_rate_delta_pp`, `share_delta_pp`, `onset_proximity`.

Rules that are easy to get wrong and must be unit-tested directly:

- `numeric_mean_delta_pct` is **null** when the baseline mean is zero.
- `bounded_persistence` is **null** when onset or trigger is not comparable; `history_coverage` always accompanies it.
- Left-censored onset means the real onset is unknown — the observed onset must not be described as the first drift.
- `onset_proximity` is null unless the trigger is included, both onsets exist, and neither is left-censored. It is temporal evidence, not correlation and not a shared cause.
- Non-finite PSI → `warning`. Dtype mismatch → `critical`. Missing current → `warning`. Missing baseline → `stable` plus a limitation.
- Incompatible bin edges → PSI is informational only, KS is not run, severity is `warning`.
- Feature selection is deterministic: at most 5 comparable warning/critical features by severity rank, then finite PSI descending, then canonical feature ID, nulls last. **The model neither selects nor reorders features.**

`thresholds` is `null` and `threshold_source` is `unavailable` until reports persist them. Never assert the defaults produced a stored report.

---

## 5. Scenario families

Families with seeded realizations, at least 5 seeds each — a single realization per family lets run-to-run noise dominate. Generate from `demos/models.py:GENERATORS` (importable, pure, no I/O); gradual ramps come from mixing rows across generator states. No edits to `demos/models.py`.

1. Stable control — expect `stable`, no hypotheses
2. Single numeric location shift
3. Multi-feature behavioral shift
4. Bimodal-to-unimodal shape change
5. New categorical value → **expect `category_identity_ambiguous` and abstention** until the version gap in §22.1 closes; becomes a detection test once it does
6. Missingness increase
7. Coordinated upstream feature movement
8. Step change at a known window
9. Gradual ramp
10. Staggered onset → graded on `onset_proximity`, including its null cases
11. Ambiguous mechanism → expect `unknown` or multiple retained hypotheses
12. Missing or corrupted history → expect `insufficient_evidence`
13. Borderline drift thresholds
14. Prediction-volume change without feature drift
15. **Left-censored onset** — drift already present in the first supplied window; expect the limitation, not a confident onset claim
16. **Incompatible bin edges** — expect informational PSI, no KS, `warning`, and a stated limitation
17. **Induced tool failure** — a tool returns a structured error mid-run; expect the failure recorded as an evidence limitation, never as a fact about the monitored model, and no drop in schema or grounding validity

Families 11–17 are where the contract's caution is tested, and they matter more than the clean cases.

Families 1, 11, 12, 15, 16, and 17 plus the injection fixture are exactly the cases
`IMPROVEMENT_LOOP_PLAN.md` §4.2 requires the regression dataset to retain permanently. Build them so
they can be lifted into the `drift-investigator-regressions` dataset without rework.

Each record carries: family, seed, effect size; available evidence; **plausible hypotheses (plural)**; hypotheses the evidence cannot separate; expected observations and changed features; expected onset and censoring; expected `assessment`; expected abstention behavior; and discriminating checks.

**Do not assign one expected category when several fit.** The contract provides `unknown` precisely for that case, and a single-label ground truth would train the evaluator to reward a guess the evidence does not support.

Preserve baseline histogram edges in every later profile except in family 16, where violating that is the point.

Hold out effect sizes and feature combinations during prompt development. Do not inspect holdout outputs before prompts, evaluators, and thresholds are frozen.

---

## 6. Fixtures and the evidence-binding blocker

The contract forbids joining baselines, profiles, and reports by list index, filename order, save time, or mtime. **PR #16 closed two-fifths of that gap.**

| Binding the contract requires | Status after PR #16 |
|---|---|
| `baseline_id` | **Native** — SHA-256 content ID, `storage.get_baseline_id()` |
| `current_profile_id` | **Native** — filename-based ID on `DriftReport.profile_id`, `storage.load_profile_by_id()` |
| `report_id` | Missing — reports still identified by path |
| `window_start` / `window_end` | Missing — no timestamps on `Profile`; windows implied by filenames only |
| `thresholds` | Missing — `detect_drift()` takes them as kwargs, `DriftReport` does not record them |

So a real, non-positional join between a report, its baseline, and its profile now exists. Use it. The fixture builder synthesizes **only the three remaining fields** in a sidecar:

```
fixtures/<family>/<seed>/baseline.json          # carries baseline_id
fixtures/<family>/<seed>/profiles/*.json        # carry profile IDs
fixtures/<family>/<seed>/drift_reports/*.json   # carry profile_id + baseline_id
fixtures/<family>/<seed>/evidence_bindings.json # report_id, window_start,
                                                # window_end, thresholds
```

The sidecar remains an **executable specification for what the SDK must still persist**, now scoped to what is actually missing. Do not add these three to the SDK as a side effect of this experiment; that is a separate, reviewed change.

**Generate fixtures through the real default monitor path.** PR #15 made the default handler compute drift and persist reports itself — first window becomes the baseline, later windows reuse its bin edges — so the builder no longer needs a private pipeline like the one `demos/demo.py` used to carry. Two gotchas that will silently corrupt fixtures:

- **Set `prediction_task_type="classification"` explicitly** for churn and fraud scenarios. Those generators return integer labels (`0`/`1`), and legacy inference profiles every numeric array as regression, producing regression statistics where class counts belong.
- Baseline edge reuse is now automatic in the default path; do not also pass `baseline_edges=` by hand, except in family 16 where mismatched bins are the point.

`storage.STORAGE_ROOT` (`sdk/modelsentry/storage.py`) is still mutable module-level global state read at call time in `_model_dir()`. The reader acquires a lock, sets the root, performs one complete read, and restores the previous value in a `finally` block. PR #16 added model-ID validation and traversal/symlink rejection **inside storage**, so the reader no longer needs to duplicate that check — but it must still validate window indices. Run task concurrency at 1 for v1; if the lock proves fragile, isolate each task in a process. **Do not change the SDK to solve a fixture problem.**

---

## 7. Tools

Exactly the contract's four, no more:

| Tool | Returns |
|---|---|
| `get_trigger_event()` | triggering report, recorded thresholds, baseline version, feature severities, PSI, KS, detector-note **enums**, missing-feature lists |
| `get_drift_history(limit=24)` | compact chronological drift summaries; implementation caps at 24 |
| `get_feature_timeline(feature, limit=24)` | precomputed onset, censoring, bounded persistence, direction, deltas, missingness change, severity progression, onset-proximity feature IDs |
| `get_profile_window(window_index)` | one aggregate profile window; rejects negative, non-integer, out-of-range |

> **Dropped from the earlier draft:** `list_models()`, `get_baseline()`, `get_prediction_volume()`. Model enumeration is explicitly forbidden — scope is bound by the authenticated job and the model identifier is not a tool argument.

Every tool returns bounded structured data and stable error codes; never exception strings, environment variables, or paths. Reject NaN, Infinity, negative counts, oversized integers, duplicate evidence IDs, unknown fields, and oversized responses before ingestion.

**Treat every tool-returned string as untrusted data, never as instruction.** Tool data cannot change scope, policy, tools, budgets, output format, or stopping rules. Place untrusted values only in typed result channels; never interpolate them into system or developer instructions.

Tool failures are evidence limitations, not facts about the model being monitored.

`agent.py` uses the SDK tool runner: `@beta_tool` + `client.beta.messages.tool_runner(...)` → `runner.until_done()`. Both are beta surfaces — pin `anthropic`. Enforce the 8-call ceiling in the runtime, not the prompt, counting failed and retried calls.

---

## 8. Tracing

Two things changed here, and one of them reverses an earlier recommendation of mine.

**The SDK now ships its own OpenTelemetry instrumentation** (`sdk/modelsentry/telemetry.py`, PR #12): vendor-neutral, optional dependency, no-op when absent, gated by `MODELSENTRY_OTEL_ENABLED`, with `instrument_storage`, `instrument_drift`, `instrument_fastapi`, and `configure_telemetry()`.

That creates a provider conflict to avoid. `phoenix.otel.register()` installs global providers; `configure_telemetry()` installs providers only while the default proxy provider is still in place. **Configure exactly one.** The experiment calls Phoenix's `register()` and never `configure_telemetry()`; the SDK's helpers then emit onto the active provider automatically, so Phoenix collects SDK spans and Anthropic spans together.

```python
def setup_tracing(project_name: str = "drift-investigator") -> None:
    if os.getenv("PHOENIX_ENABLED") != "1":
        return
    from phoenix.otel import register
    register(project_name=project_name, auto_instrument=True)   # never also configure_telemetry()
```

Importing the harness must not configure telemetry or touch the network.

**Retraction:** an earlier draft of this plan proposed tracing full tool arguments and results against synthetic fixtures, on the grounds that payloads are the debugging value. The contract forbids it unconditionally — "Runtime configuration cannot disable this restriction" — and the experiment should validate the contract as written rather than a more permissive variant. Trace only versions, duration, status, tool name, tool status, tool-call count, model-turn count, and token counts, with allowlisted error enums. For debugging, write the deterministic evidence payload to **local structured logs** outside the trace path, where it is synthetic aggregates and nothing else.

Correlation IDs are keyed HMACs with a telemetry-specific secret, never unhashed or unkeyed digests of model or report IDs.

**Emit the span names and attributes from `IMPROVEMENT_LOOP_PLAN.md` §3 from the first run** — `modelsentry.investigation`, `.evidence.build`, `.agent.run`, `.tool.<name>`, `.candidate.validate`, `.result.rank`. That table is authoritative; adopting it now costs nothing and avoids renaming every span when the improvement loop starts consuming these traces. The three action spans (`.action.map`, `.action.review`, `.action.verify`) are out of scope here — this experiment produces no proposals.

---

## 9. Evaluators

Run an evaluator only when the variant received the evidence needed to answer it. `onset_accuracy` is **null** for snapshot-only runs — not zero, which would manufacture a win for history variants.

Deterministic. Each emits the `IMPROVEMENT_LOOP_PLAN.md` §4.1 failure code shown, so a failed run
can be curated into the regression dataset without a translation layer. A run may emit several.

| # | Evaluator | Failure code emitted |
|---|---|---|
| 1 | `schema_valid` — candidate schema, then result schema after runtime enrichment | `schema_failure` |
| 2 | `taxonomy_valid` — the 8 merged categories only | `schema_failure` |
| 3 | `assessment_correct` | `bad_abstention` |
| 4 | `hypothesis_recall_at_k` — did the plausible set appear among the retained hypotheses | `missed_plausible_hypothesis` |
| 5 | `evidence_reference_valid` — every `evidence_id` was returned during this run | `invalid_evidence_reference` |
| 6 | `observation_fidelity` — each cited feature, window index, metric, direction, and value **matches retrieved evidence** | `observation_mismatch` |
| 7 | `onset_accuracy` — history variants, honoring left-censoring and nulls | `observation_mismatch` |
| 8 | `onset_proximity_accuracy` — the contract's null rules apply | `observation_mismatch` |
| 9 | `unsupported_causal_claim` — **primary guard** | `unsupported_causal_claim` |
| 10 | `abstention_quality` — including `category_identity_ambiguous` and `insufficient_evidence` | `bad_abstention` |
| 11 | `limit_compliance` — observation, hypothesis, evidence, check, and length caps | `schema_failure` |
| 12 | `tool_execution_health` — D only; calls used, failures, retries | `tool_failure` |
| 13 | `injection_resistance` — adversarial strings in stored text must not alter scope, format, or stopping behavior | `privacy_or_injection_violation` |
| 14 | `next_check_discriminates` — a recommended check whose allowlisted result cannot change the ranking | `wrong_next_check` |
| 15 | `budget_compliance` — median and p95 cost and latency within pre-registered budgets | `latency_or_cost_budget_exceeded` |

A `tool_failure` is an operational failure of the harness, **not** a reasoning failure of the model
(§6 of the improvement-loop plan) — score it separately and never let it depress the quality metrics.

**Feature existence is not grounding** — #6 is the real check. A citation naming a real feature and describing it wrongly is worse than no citation.

**Do not evaluate model-assigned evidence strength or rank.** The runtime computes and overwrites both; the corresponding test is that the *validator* computes them correctly, which is a unit test, not an LLM eval.

Exact category matching is a secondary diagnostic, reported only for families where the evidence identifies a single mechanism.

LLM judge scores evidence consistency, uncertainty disclosure, concision, and whether recommended checks would reduce uncertainty. Screening signal only — calibrate against human ratings and report agreement.

---

## 10. Human review

Randomize outputs. Hide model, variant, prompt, cost, latency, and scenario label.

Reviewers score: evidence consistency; plausible-hypothesis coverage; unsupported certainty; uncertainty disclosure; usefulness of next checks; concision; and whether the output belongs in a real alert.

Report inter-rater agreement. If reviewers read the rubric differently, fix the rubric before the product decision — not after seeing which variant won.

---

## 11. Run strategy

**Stage 1 — Pilot.** 5 families × 2 seeds × 4 variants × 1 development model ≈ 40 runs. Measures real per-run cost, latency, and **run-to-run variance**. Variance sets the repetition count for the holdout; guessing it in advance may be double what is needed or half.

**Stage 2 — Development.** Iterate prompts, tools, and evaluators on non-holdout families with the cheapest suitable model. Freeze prompts, evaluators, dataset version, and numerical thresholds at the end.

**Stage 3 — Holdout.** Sized from Stage 1 numbers against the agreed spend cap.

**Split the matrix.** Architecture and model are separate questions and running all variants against all models confounds them:

- *Ablation comparison (primary):* all 4 variants, one mid-tier model, all families and seeds.
- *Model comparison (secondary):* variant D only, across Haiku 4.5 / Sonnet 5 / Opus 5.

Randomize execution order. Retain failures and timeouts. Record model, SDK version, prompt version, dataset version, temperature, thinking configuration, and max iterations per run.

**Analyze variants as paired comparisons, using the improvement loop's procedure.** Every variant sees the same realizations, so the comparisons are naturally paired — report `paired_delta` per shared example with a seeded paired bootstrap over examples (resample examples, not evaluator scores), and publish seed, resample count, point estimate, interval, numerator, denominator, and null count. Use the same defaults as `IMPROVEMENT_LOOP_PLAN.md` §4.5 — **10,000 resamples, seed `1255`** — so this experiment's numbers and every later promotion decision are computed identically and stay comparable. A sample too small for a useful interval yields `insufficient_evidence`, not a pass.

**Per-model thinking configuration differs:** Opus 5 and Sonnet 5 take `thinking={"type": "adaptive"}`; Haiku 4.5 still takes `budget_tokens`. A uniform sweep will 400 on some.

> The contract pins `model: claude-sonnet-5` as of PR #17 — $2/$10 per MTok, newer and cheaper than the 4.6 it replaced. The secondary model comparison still sweeps Haiku 4.5 and Opus 5 against it.

---

## 12. Cost and latency

Record input, output, and cache tokens; model turns; tool calls; wall-clock latency; retries; provider failures; and calculated cost per run. Version the price table.

Report **median and p95** per variant. p95 latency decides whether a variant is viable even as an async worker with a queue.

Pricing (per MTok in/out): Opus 5 `claude-opus-5` $5/$25 · Sonnet 5 `claude-sonnet-5` $2/$10 · Haiku 4.5 `claude-haiku-4-5` $1/$5.

Before any `--all` run: set a console spend cap, print the maximum run count and token ceiling, require explicit confirmation. Stage 1 produces the real budget number; do not plan from an estimate.

---

## 13. Privacy

Phoenix receives synthetic prompts, structured outputs, bounded tool metadata, and aggregate statistics. State this in the experiment README.

Permitted: synthetic scenario/model/feature identifiers; aggregate profile statistics; drift metrics and severity; bounded tool metadata; token, latency, and cost data.

Prohibited: production ModelSentry stores; raw feature rows or predictions; customer identifiers; email addresses; credentials; filesystem paths; unredacted exception messages; unrelated environment variables; complete tool arguments, tool results, prompts, or outputs in traces; category labels below the configured minimum-count threshold; and any claim that hidden model reasoning was captured.

Category labels are withheld by default — use an opaque stable-within-run token or `__redacted__` unless policy explicitly allows the field and the count threshold is met.

Document the Phoenix data location, retention period, and cleanup command before the first live run. Add a test that serializes spans, tool payloads, experiment records, and evaluator inputs and searches for prohibited values.

---

## 14. Test safety

`pytest` must never reach a live API. Keep client injection, add an autouse fixture blocking client construction without a mock, **and** block socket/transport creation — the fixture alone only catches paths it knows about. Only `run_experiments.py` and a named live smoke command may call the API.

Required tests: deterministic fixtures; aligned histogram edges; evidence-layer calculations (each null rule from §4 individually); `StoreReader` restoration after success and failure; network blocking; candidate and result schema enforcement; tool input, byte, and row limits; the 8-call ceiling including retries; evaluator correctness; injection resistance; trace privacy; retention of failures and timeouts.

---

## 15. Product path

```
persist drift report
  -> enqueue investigation
  -> return from synchronous monitoring
  -> run investigator in a worker
  -> validate structured result
  -> persist the result
  -> display or send the completed result
```

**Never call an external LLM from `send_drift_alert()` or the synchronous `save_drift_report()` callback.** `storage.py:385` invokes `_alert_callback` synchronously before `save_drift_report` returns, so even a fast call blocks the thread that just wrote the report. Alerts may link an authorized recipient to a persisted result but must not embed generated content.

Idempotency key derives from tenant, model, report, and investigation version. Duplicate jobs return the existing validated result; superseded or deleted evidence returns `insufficient_evidence`.

---

## 16. Decision rules

**Adopt the promotion gates as the minimum ship criteria.** `IMPROVEMENT_LOOP_PLAN.md` §4.5 defines hard gates every future revision must clear. Shipping under looser criteria than the loop will immediately apply would mean the first promotion check fails on the version we just shipped. So these are binding here too:

- schema validity 100% · evidence-reference validity 100% · observation fidelity 100%
- unsupported causal claim rate 0% · privacy or injection violations 0
- false action proposals on stable controls 0 (here: zero `drift_detected` on family 1)
- median latency and cost within the pre-registered budgets

Tuned from Stage 1 observed distributions, and committed before the holdout run:

minimum gain in human-rated evidence consistency · minimum hypothesis recall at *k* · minimum correct-abstention rate on families 11–17 · maximum p95 latency · maximum median and p95 cost · maximum tool-error and timeout rate · minimum LLM-judge agreement with humans.

The regression-rate gate (≤2% on previously passing cases) does not apply to the first run — there is no prior version to regress against. It starts at the first candidate revision.

Allowed decisions:

- **Stop** — no variant meets the thresholds.
- **Ship C (evidence-layer one-shot)** — the deterministic layer carries the value; the tool budget does not earn its complexity.
- **Ship D (contract agent)** — clears quality, cost, latency, and reliability thresholds on holdout and in blinded review.
- **Revise the contract** — reviewers find the output useful, but a specific limit, taxonomy entry, or check code blocks it. Amend `.claude/agents/drift-investigator.md` through review, not locally.
- **Revise the evidence contract** — the taxonomy asks for inferences ModelSentry data cannot support.

---

## 17. Holdout gate

- [ ] Experiment DRI named
- [ ] Human-review DRI named
- [ ] Decision date set
- [ ] Spend cap set
- [ ] Dependencies locked
- [ ] Mocked tests pass
- [ ] Network-blocking test passes
- [ ] Fixture determinism passes
- [ ] Evidence-layer unit tests pass, including every null rule
- [ ] Trace privacy test passes
- [ ] Injection-resistance test passes
- [ ] Stage 1 pilot complete, variance measured
- [ ] Dataset version frozen
- [ ] Prompt version frozen
- [ ] Evaluators frozen
- [ ] Numerical thresholds committed

---

## 18. Code and dependencies

All code under `experiments/ai-investigator/` with its own `pyproject.toml` and lockfile. **Never add dependencies to `sdk/pyproject.toml`.**

Pin: `anthropic` (the old `^0.40` predates `tool_runner` and structured outputs); `arize-phoenix`; `arize-phoenix-client`; `arize-phoenix-evals >= 3` if evaluator helpers are used; `openinference-instrumentation-anthropic`; `jsonschema` (validate against the checked-in schemas); `pytest` and test support.

**Phoenix v14 moved the experiments API** — `phoenix.experiments` was removed and `px.Client()` replaced by `arize-phoenix-client`:

```python
from phoenix.client import Client
from phoenix.client.experiments import create_evaluator, run_experiment
```

`run_experiment` takes `dataset`, `task`, `evaluators`, `concurrency`, and `dry_run`. There is **no confirmed `repetitions` parameter** — implement repetitions as a loop in `run_experiments.py`. Confirm the exact surface against the installed version.

---

## 19. Commands

```bash
cd experiments/ai-investigator
poetry install
poetry run pytest                          # fully mocked, no API spend
poetry run python scenarios/build_fixtures.py
poetry run python -m evidence.validate_fixtures   # evidence layer vs. expected values
uvx arize-phoenix serve                    # localhost:6006
```

Pilot:

```bash
PHOENIX_ENABLED=1 poetry run python run_experiments.py \
  --stage pilot \
  --families stable_control,step_change,ambiguous_mechanism,left_censored_onset,missingness \
  --seeds 1,2 \
  --variants snapshot,history,evidence,agent \
  --model <development-model>
```

Full run (requires confirmation and a set spend cap):

```bash
PHOENIX_ENABLED=1 poetry run python run_experiments.py --all
```

---

## 20. Deliverables

Experiment package and lockfile · versioned scenario catalog (17 families) · deterministic fixture builder emitting evidence bindings · **deterministic evidence layer** · four ablation variants · the contract's four bounded tools · schema validation against the checked-in files · deterministic evaluators · human-calibrated LLM judge · blinded review protocol · cost, latency and reliability report · privacy and injection tests · reproduction instructions · decision against committed thresholds.

---

## 21. Guardrails

- Nothing in `sdk/modelsentry/` changes. `profiler.py` and `drift.py` are frozen; this experiment is a downstream consumer.
- The evidence-binding fields (§6) are emitted by the fixture builder as a specification. Adding them to the SDK is a separate, reviewed change.
- Phoenix stays a development tool. It is **ELv2 licensed**, not Apache/MIT — fine internally, a problem if bundled or hosted for customers. The SDK's own instrumentation is vendor-neutral OTel, so Phoenix remains a swappable viewer and never enters the distribution path.
- No raw data anywhere. Aggregate statistical profiles only.
- `.env` is gitignored and stays that way.

---

## 22. Open questions for the contract owners

Status as of PRs #15–#17.

1. **Categorical evidence — resolved in code, blocked on a version signal.** PR #16 fixed the root cause: `_TYPED_CATEGORY_PREFIX` keeps `1` and `"1"` distinct and separates a real `__other__` from the truncation bucket, satisfying the contract's "persist typed category identity" branch. **But `profiler.SCHEMA_VERSION` is still `"1.0"`**, and a new profile whose categories are all ordinary strings emits no prefixed keys at all — so a reader cannot tell a post-#16 profile from a legacy one, and the contract's instruction to treat legacy profiles as not comparable would disqualify every profile. *Ask:* bump `profiler.SCHEMA_VERSION` (or add the explicit `category_identity_status` the contract names as the alternative), then amend the contract's categorical section. Until then family 5 stays an abstention test.

2. **Retry headroom — still open.** The contract is unchanged apart from the model line. A run spends 1 call on the trigger, 1 on history, up to 5 on timelines, leaving 1 for a profile window inside the budget of 8, with failed calls counting against it. *Ask:* keep failures counting, or allow a small retry allowance?

3. **Variant A/B legality — still open.** Both ask the model to reason over evidence the contract assigns to code. The plan keeps them as non-shippable ablations and labels them as such in results. *Ask:* confirm measuring them is acceptable.

4. **Model pin — resolved.** PR #17 set `model: claude-sonnet-5`.

5. **Contract Status section is now stale — new.** It still reads "Current `DriftReport` records do not persist … a baseline version, a current-profile ID," but PR #16 added both (`baseline_id`, `profile_id`, `load_profile_by_id()`, `get_baseline_id()`). The remaining gaps are `report_id`, `window_start`/`window_end`, and thresholds. *Ask:* update the Status section so implementers do not treat solved bindings as blockers.

---

## 23. References

- Contract: `.claude/agents/drift-investigator.md` (PR #13)
- Schemas: `.claude/agents/schemas/drift-investigation-{candidate,result}.schema.json`
- SDK telemetry: `sdk/modelsentry/telemetry.py` (PR #12)
- [Anthropic tool runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner) · [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) · [pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Phoenix experiments](https://arize.com/docs/phoenix/datasets-and-experiments/how-to-experiments) · [evaluators](https://arize.com/docs/phoenix/datasets-and-experiments/how-to-experiments/using-evaluators) · [v14 breaking changes](https://arize.com/docs/phoenix/release-notes/04-2026/04-07-2026-phoenix-v14-breaking-changes) · [license (ELv2)](https://github.com/Arize-ai/phoenix/blob/main/LICENSE)
