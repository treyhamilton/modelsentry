# ModelSentry Demo

Self-contained demo for sales calls and personal QA. Three modes:

| Mode | Use it for |
|---|---|
| `fast` | Live demos with prospects — keypress controls drift |
| `slow` | Unattended QA — drift introduced across models on a schedule |
| `walkthrough` | First time showing someone — prints the integration code, then runs fast mode |

All three modes spin up the real `modelsentry serve` dashboard at
`http://127.0.0.1:8080` and fire real email alerts to `getmodelsentry@gmail.com`
when drift is detected.

---

## Setup (one time)

```bash
cd demos
cp .env.example .env
# edit .env and fill in SMTP_USER and SMTP_PASSWORD
pip install -r requirements.txt
```

For the SMTP credentials, generate a Gmail App Password at
[myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
(requires 2FA enabled). Use the 16-character password, not the account password.

---

## Run

```bash
# Live sales demo — interactive
python demo.py fast

# Autonomous demo — runs for 25 minutes by default
python demo.py slow
python demo.py slow --minutes 5    # quick test

# First-time pitch — annotated code walkthrough → fast mode
python demo.py walkthrough
```

Press `Ctrl+C` to stop at any time. The dashboard server is terminated automatically.

---

## Fast mode keypress map

```
1 → trigger drift on  churn-v3
2 → trigger drift on  lead-score-v2
3 → trigger drift on  fraud-detect-v4
r → reset all models to baseline
q → quit
```

Drift is detected ~10 seconds after a keypress (5 batches × 10 rows = 1 profile
window of 50). Each drift event fires one email alert; the alert re-arms when
the model returns to stable.

---

## The 3 simulated models

| Model | Type | Drift story |
|---|---|---|
| `churn-v3` | binary classification | New marketing campaign brought in younger, lower-paying customers |
| `lead-score-v2` | regression (0-100) | New paid acquisition channel, lower-quality leads |
| `fraud-detect-v4` | binary classification | International expansion changed the risk profile |

Feature distributions are realistic for a 50-200 person SaaS company.

---

## Architecture

The demo uses ModelSentry's public API:

- `ms.init(profile_handler=demo_handler)` — custom handler that saves the profile,
  computes drift against baseline, saves the drift report, and triggers the alert
- `@ms.monitor(model_id=...)` — decorates 3 predict functions, one per model
- `AlertConfig` + `send_drift_alert` from `modelsentry.alerts` — email delivery
- `modelsentry serve` spawned as a subprocess — serves the dashboard

Profiles are stored in `~/.modelsentry/{model_id}/` (the SDK's default location).
