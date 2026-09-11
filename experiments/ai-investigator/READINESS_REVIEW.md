# Pre-build readiness review

> **Status:** proposals awaiting sign-off. Nothing here is decided.
>
> **Lifecycle:** this document is temporary. Once each item below is Agreed or Rejected, the agreed
> ones fold into `PLAN.md` in one commit and this file is deleted. `PLAN.md` stays the settled
> spec; this is the round of decisions that gets it there.
>
> **How to review:** set Status on each row, add a note where you disagree. Items 1–3 block writing
> code. Items 4–8 block the Stage 1 pilot. Items 9–13 are small and mostly mechanical.

## Verdict

**Ready to start building, not ready to run.** The specification is strong on contracts, safety,
evaluation methodology, and failure handling. What is missing is the layer that turns a spec into
two people working on Monday — chiefly that **nobody owns the prompts**, plus a handful of
undefined evaluation parameters.

Readiness: **7/10**.

---

## Blocking before code

### 1. Prompt authorship — split it from ground truth

**Recommendation.** One of us compresses the contract into the prompt; the *other* writes the
scenario ground truth. Write **one parameterized prompt, not four**: a shared core (taxonomy,
abstention rules, citation discipline, never-claim-cause) plus a per-variant block stating what
evidence is present. Target 60–100 lines against the contract's 523.

**Why.** `PLAN.md` mentions iterating, freezing, and versioning prompts but never says who writes
them or from what — and there is no prompt artifact in the §20 deliverables. The contract is a
specification, not a system prompt. The split also removes the risk in item 7: whoever decides what
"good" looks like should not be the person who taught the model to produce it.

One prompt rather than four is methodological. If the prompts differ more than the evidence does,
the ablation measures prompt quality instead of architecture and A-vs-D tells us nothing.

**Status:** proposed · **Owner:** TBD

### 2. Name the DRIs now

**Recommendation.** Experiment DRI is whoever did *not* author the contract. Human-review DRI is the
same person in this plan and in `IMPROVEMENT_LOOP_PLAN.md`, which currently keeps a separate list.

**Why.** The spec's author should not grade their own spec. `PLAN.md` defers all four TBDs to "before
the holdout," but sequencing needs an owner from day one.

**Status:** proposed · **Owner:** TBD

### 3. Build order

**Recommendation.** Evidence layer → fixture builder → tools + `StoreReader` → prompts → variants →
evaluators → runner. The first three are fully specified and unblocked, so one of us starts there
immediately while the other writes prompts and ground truth in parallel. They converge at variants.

**Why.** Unstated in `PLAN.md`. Two people and six workstreams need an explicit order.

**Status:** proposed

---

## Blocking before the Stage 1 pilot

### 4. `hypothesis_recall_at_k`: k = 3 primary, k = 1 secondary

**Recommendation.** Gate on recall@3. Report recall@1 as a diagnostic only.

**Why.** `k` is currently undefined. The contract caps hypotheses at 3 and deliberately permits
several retained ones for ambiguous evidence, so penalizing rank *within* three contradicts the
design intent.

**Status:** proposed

### 5. Human-review rubric: 5-point scale, anchors at 1/3/5, Krippendorff α ≥ 0.67

**Recommendation.** Write anchors for 1, 3, and 5 only. Use ordinal Krippendorff's α with a floor of
0.67. Run a calibration round on 10 shared examples first; if α is below the floor, fix the anchors
and repeat *before* anyone sees real outputs.

**Why.** §10 lists seven scoring dimensions but defines no scale, no anchors, no agreement statistic,
and no threshold. Full five-anchor rubrics cost more to write without improving agreement.

**Status:** proposed

### 6. Judge with Opus 5, not Sonnet 5

**Recommendation.** Judge with `claude-opus-5`. Calibrate on ~50 human-rated outputs. If judge-human
agreement falls below the human-human α, demote the judge to reporting-only.

**Why.** The contract pins Sonnet 5 for the investigator; judging with the same model introduces
self-preference bias on the primary screening signal. Judging is short, so the cost difference is
negligible.

**Status:** proposed

### 7. Ground truth derives from the generator, never from outputs

**Recommendation.** Labels come from the known state transitions in the scenario generators, reviewed
by the prompt author. Never label by reading what the model produced.

**Why.** We know exactly what was injected. Labeling from outputs launders model behavior into
ground truth.

**Status:** proposed

### 8. The three open §22 items

**8a — Categorical: bump `profiler.SCHEMA_VERSION` to `"1.1"`.** Smallest change that works, no new
field, readers gate on it cleanly. Ship it as its own small SDK PR before the experiment. *Note this
touches `profiler.py`, which our own rules freeze without explicit instruction — it needs a
deliberate decision, not a drive-by commit.* Without it, PR #16's typed-category fix is unusable
(nothing distinguishes a fixed profile from a legacy one) and family 5 stays an abstention test
permanently.

**8b — Retry headroom: keep it strict for v1.** The contract already counts failures against the 8.
Adding an allowance now is a knob added without evidence. Family 17 will measure what strictness
actually costs; revisit after the pilot with numbers.

**8c — Variants A/B: keep them, labeled non-conforming.** They are one-shot and cheap, and they are
the only way to price the deterministic evidence layer, which is the most expensive thing we are
building.

**Status:** 8a proposed · 8b proposed · 8c proposed

---

## Small and mostly mechanical

| # | Item | Recommendation |
|---|---|---|
| 9 | Variant B token budget | Run `count_tokens` on one 24-window payload **before building B**. Under 60k tokens: build as designed. Over: redefine B as an 8-window subset and say so. Far over: drop B and let A→C bracket the value of history. |
| 10 | Family 1 is a weak control | Keep it — it tests the deterministic short-circuit — but move the false-positive gate to families 13 and 14, where the model actually runs. Currently the gate is nearly free to pass. |
| 11 | `get_trigger_event()` thresholds | Assert `thresholds is None` in the tool test. Thresholds are never persisted, so the field is always null here. |
| 12 | Phoenix version | Pin `arize-phoenix>=14,<15` with a matching `arize-phoenix-client`; commit the lockfile with exact resolved versions. |
| 13 | Python 3.11.9 compatibility | Dry-run `poetry add arize-phoenix` this week. A dependency conflict found now is an afternoon; found in week three it is a rework. |

---

## Open concerns

These are risks to manage, not decisions to make.

**The evidence layer is a fork in waiting.** We are building logic that must eventually live in the
SDK. If the experiment's version and the SDK's later version diverge, the validation does not
transfer to the shipped product. *Mitigation:* write it as a standalone module with zero
experiment-specific imports so it can move over verbatim.

**Two-person blinding is weak.** Whoever runs the sweep sees every variant, model, and cost.
*Mitigation:* make it mechanical — the runner writes outputs to hashed filenames with a key file
neither reviewer opens until scores are submitted.

**Passing means "not disqualified," not "useful."** The ground truth is synthetic, generated from a
state machine we wrote, so the plausible-hypothesis sets encode *our* beliefs about what aggregate
evidence implies. A model can score well here and still be unhelpful on real drift. This is inherent
and cannot be fixed before launch. State the conclusion conditionally in the final report and treat
beta users as the real evidence.

**Contract amendments are a multi-day loop.** If the pilot shows the 8-call budget is too tight,
that is a PR cycle rather than a config change. Agree a fast path for amendments discovered during
the pilot, or we will be tempted to work around the contract instead of fixing it.

**Cost and wall-clock are genuinely unknown** until Stage 1 reports. That is by design and it is the
right call — but no delivery date should be committed before those 40 runs finish.

---

## Biggest single risk

The deterministic evidence layer is the largest work item, it is the thing ModelSentry does not yet
have, and if it is wrong then every variant is wrong in the same direction and the ablation
comparison measures nothing. Build it first, test it against hand-computed expected values, and keep
an LLM away from it until its unit tests pass.
