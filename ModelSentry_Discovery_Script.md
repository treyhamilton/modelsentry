# ModelSentry Customer Discovery Interview Script
# Phase 0 Validation — Version 1.0 | May 2026

---

## BEFORE THE INTERVIEW

### Who to talk to
Your target is a Senior Data Scientist or ML Engineer at a 50–200 person SaaS company
who has at least 2 models in production. Secondary targets: Head of Data or ML Lead.

**Where to find them:**
- Your existing professional network (start here — warmest conversations)
- LinkedIn (search: "Senior Data Scientist" + SaaS company 51–200 employees)
- Twitter/X ML community
- Slack communities: MLOps Community, Locally Optimistic, Data Talks Club
- Reddit: r/MachineLearning, r/datascience

### How to ask for the conversation
Keep the ask short and non-pitchy:

> "Hey [Name] — I'm exploring a problem I've personally experienced as a data scientist:
> ML models degrading silently in production before anyone notices. I'm doing 10
> conversations with practitioners to understand if this is a widespread pain or just
> something I ran into. Would you have 20 minutes to share your experience? No pitch,
> just listening."

### Before each call, note:
- [ ] Company name and size
- [ ] Interviewee title
- [ ] How many models they likely have in production (check LinkedIn/company blog)
- [ ] Any public info about their ML stack

### Interview logistics
- Length: 20–30 minutes (never go over without asking)
- Record if permitted ("Mind if I record for notes? Won't be shared.")
- Take notes in real time — especially exact phrases they use
- Never interrupt a story to ask the next question

---

## THE SCRIPT

### Opening (2 minutes)

"Thanks so much for making time. Quick context: I'm exploring a problem in ML model
monitoring and I'm doing 10 conversations before building anything. I want to hear
about your actual experience — no slides, no pitch, just your perspective.

There are no right or wrong answers. The most valuable thing you can do is tell me
about real situations you've been in, not what you think I want to hear.

Sound good? Let's start with some background on your setup."

---

### Part 1 — Context Setting (3 minutes)

These questions establish the baseline. Keep them quick.

**Q1.** "Walk me through your current ML setup — how many models do you have running
in production right now, roughly?"

*Listen for: number of models, types (classification, regression, recommendation),
how long they've been running, who owns them.*

**Q2.** "What does your team look like — is it just you, or are there other data
scientists and ML engineers involved?"

*Listen for: team size, whether monitoring is one person's job or shared,
whether there's a dedicated MLOps role.*

**Q3.** "What kinds of decisions are your models actually driving in the business?"

*Listen for: business criticality. A model driving pricing decisions is very
different from one generating weekly reports. Higher stakes = more pain.*

---

### Part 2 — The Core Pain (10 minutes)

This is the most important section. Let them talk. Ask follow-up questions.
Do not rush past a story to get to the next question.

**Q4.** "Tell me about the last time one of your production models had a problem.
What happened?"

*This is the most important question in the interview. Wait for the full story.
If they say "nothing has gone wrong," probe gently:*

> "What about a time when model performance wasn't what the business expected?
> Or a time when you weren't sure if the model was still working well?"

*Listen for: what the failure was, how long it went undetected, how they found out,
what the business impact was, how they fixed it, how they felt about it.*

**Q5.** "How did you find out the model had a problem? Who told you, or how did
you discover it?"

*Listen for: was it proactive (they caught it) or reactive (someone complained)?
The more reactive, the more painful the problem.*

**Q6.** "How long had the problem been happening before you discovered it?"

*Listen for: days, weeks, months. Longer = more acute pain.*

**Q7.** "Walk me through what you actually do today to check on your models'
health. What does that look like week to week?"

*Listen for: manual queries, scheduled jobs, dashboards, nothing at all.
Manual and ad-hoc = strong signal.*

**Q8.** "How much time would you say you spend per week on model monitoring and
health checking across all your models?"

*Listen for: time estimate and emotional tone. "Too much" or "not enough" are
both signals. "I don't really do it regularly" is a very strong signal.*

---

### Part 3 — Current Solutions (5 minutes)

**Q9.** "What tools or processes do you currently use to monitor your models in
production? Anything at all — even if it's just a spreadsheet or a cron job."

*Listen for: Evidently, MLflow, custom scripts, nothing, Datadog, Grafana.
Whatever they mention, ask how they like it.*

**Q10.** "What do you wish those tools did better, or what's still missing?"

*Listen for: specific frustrations. These become your product requirements.*

**Q11.** "Have you ever evaluated any dedicated model monitoring tools?
What did you look at and why didn't you adopt them?"

*Listen for: Arize, WhyLabs, Fiddler. If they evaluated and didn't adopt,
the reason is gold — price, complexity, doesn't solve the right problem.*

---

### Part 4 — The Hypothetical (5 minutes)

**This is where you introduce the concept — carefully. Do not pitch. Frame it as
a hypothesis you want to validate.**

"Based on what I've been hearing from people, I'm exploring a tool that would
automatically monitor your production models for drift and degradation — so you'd
get an early warning before the business notices something is wrong. It would work
by having a lightweight SDK in your pipeline that computes statistical profiles
locally — your raw data never leaves your environment — and then alerts you when
something starts shifting.

**Q12.** "Does that describe something you'd actually find useful, or does it solve
a problem that isn't really painful enough to matter?"

*Listen for: genuine interest vs. polite interest. "That would be really useful"
said flatly is different from "oh god yes, we needed that three months ago."*

**Q13.** "What would you need to see from a tool like that before you'd trust it
with your production models?"

*Listen for: accuracy concerns, setup complexity concerns, data privacy concerns,
integration concerns. These are objections you need to address.*

**Q14.** "If something like this existed today and it worked well, what would make
you choose it over just building your own monitoring?"

*Listen for: time savings, reliability, ongoing maintenance burden.
"I don't have time to build and maintain it" is the answer you want.*

**Q15.** "What would you expect to pay for something like this per month —
if it genuinely solved the monitoring problem for all your production models?"

*DO NOT anchor them with a number first. Let them name a price range.
Note their response exactly. This is pricing validation data.*

---

### Part 5 — Wrap Up (2 minutes)

**Q16.** "Is there anything about your model monitoring situation that you think
I should understand that we haven't talked about yet?"

*Always ask this. The most interesting things often come out here.*

**Q17.** "Is there anyone else on your team or in your network who deals with this
problem that you think I should talk to?"

*Ask for referrals every single time. This is how you get from 10 interviews
to 30 interviews without cold outreach.*

"This has been incredibly helpful. I'll be sharing what I learn across all 10
conversations in a few weeks — would it be useful if I sent you a summary of
the common patterns? And if I do build something, would you be open to being
an early tester?"

*Note their response. Anyone who says yes to early testing is a potential
beta user.*

---

## AFTER THE INTERVIEW

### Immediately after (within 10 minutes)
Fill out the scoring sheet below while the conversation is fresh.
Add any exact quotes you remember — word-for-word is best.

### The gate question
After all 10 interviews, tally your scores. If 3 or more interviews score 4+
on the "Would pay" question, proceed to Phase 1 (build).

If fewer than 3 score 4+, do 5 more interviews before deciding.

---

## INTERVIEW SCORING SHEET

**Interview #:** _______ **Date:** _______ **Duration:** _______

**Interviewee:**
- Name: _______________________________________
- Title: _______________________________________
- Company: ____________________________________
- Company size: ________________________________
- Models in production: _________________________

---

### Key findings

**Models in production:** _______
**Current monitoring approach:** _______________________________________________
**Tools currently used:** ______________________________________________________
**Time spent on monitoring per week:** _________________________________________

**Did they have a model failure story? (circle):** YES / NO

If yes — how long did it go undetected? _______________________________________
If yes — how did they find out? ________________________________________________
If yes — what was the business impact? _________________________________________

**Best quote from this interview:**
_______________________________________________________________________________
_______________________________________________________________________________

---

### Scoring (1–5 scale, 5 = strongest signal)

| Signal | Score (1–5) | Notes |
|---|---|---|
| Severity of pain (how bad is the monitoring problem?) | | |
| Frequency (how often does degradation happen?) | | |
| Current solution inadequacy (how bad are existing tools?) | | |
| Willingness to pay (did they name a price or express budget?) | | |
| Urgency (would they adopt today if it existed?) | | |
| **TOTAL (max 25)** | | |

**Price they named (if any):** $_______/month

**Interested in beta access? (circle):** YES / NO / MAYBE

**Referrals given:**
1. ____________________________________________________________________________
2. ____________________________________________________________________________

---

### Interview classification

After scoring, classify this interview as one of three outcomes:

**STRONG SIGNAL (total 18+):**
Real, acute pain. Would likely pay. Wants it now.

**MODERATE SIGNAL (total 12–17):**
Pain exists but may not be acute enough to drive action.
Explore what would make it more urgent.

**WEAK SIGNAL (total below 12):**
Pain is not there, or is too abstract. Note why and adjust target persona if needed.

**This interview classification:** STRONG / MODERATE / WEAK

---

## AGGREGATE TRACKING SHEET

After all interviews, complete this summary:

| # | Title | Company Size | Models | Pain Score | Would Pay | Beta? |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

**Interviews with score 18+:** _______
**Interviews where they named a price:** _______
**Average price named:** $_______
**Total beta interest:** _______
**Total referrals collected:** _______

---

## COMMON PATTERNS TO WATCH FOR

### Green flags (proceed to build)
- "We've had models break silently and didn't know for weeks"
- "I check manually but I know I'm missing things"
- "We tried [tool] but it was too complex / expensive"
- Unprompted mention of specific dollar amounts they'd pay
- "Can I be a beta user?" asked without prompting

### Yellow flags (dig deeper before deciding)
- "It would be nice but I'm not sure it's a priority"
- "Our models are pretty stable, we don't see much drift"
- "We have something that works for now"
- Only vague interest in the hypothetical

### Red flags (rethink the target persona)
- "We don't really have models in production yet"
- "Our data team handles all of that" (no direct pain)
- "That's a solved problem — we use [tool] and it works great"
- Difficulty naming any monitoring pain at all

---

## THINGS TO NEVER DO IN THESE INTERVIEWS

- **Never pitch.** The moment you start selling, they stop telling the truth.
- **Never lead the witness.** Don't say "Does it frustrate you when models break?" Say "Tell me about a time when..."
- **Never interrupt a story.** If they're telling you about a painful model failure, let them finish even if it takes 5 minutes.
- **Never ask hypothetical future questions before you've heard about real past pain.** Earn the right to the hypothetical by listening first.
- **Never anchor a price.** If you say "$99 a month, does that sound reasonable?" you've contaminated the data.
- **Never dismiss a "no."** A strong "this isn't painful enough" from 3 interviews is more valuable than 10 polite "yes that sounds interesting" responses.

---

## INTERVIEW REQUEST TEMPLATES

### LinkedIn DM (cold)
"Hi [Name] — saw your work on [specific thing from their profile/posts].
I'm doing research on ML model monitoring in production — specifically the
gap between when a model starts degrading and when the team notices.
Would you have 20 minutes to share your experience? No pitch, pure learning.
Happy to share what I hear across all conversations in return."

### Email (warm — someone you know)
"Hey [Name] — hope you're well. I'm spending the next few weeks talking to
data scientists about model monitoring in production. Specifically I want to
understand how teams catch drift and degradation before it causes business
problems. Given your work at [company], I think you'd have a really useful
perspective. Any chance you have 20 minutes this week or next?
No slides, no pitch — just conversation."

### Twitter/X DM
"Hey — I'm doing customer discovery around ML model monitoring (specifically
catching drift before it impacts the business). Would you be up for a 20-min
conversation? Trying to understand the real pain before building anything.
No pitch involved."

---

*ModelSentry Customer Discovery Script v1.0 | Phase 0 Validation*
*Conduct 10 interviews. Gate: 3+ score 18+ → proceed to Phase 1.*
