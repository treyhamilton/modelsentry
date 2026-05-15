# ModelSentry

**Early warning when your production ML models start degrading.**

Your model went live. Stakeholders loved it. Everyone moved on. Eight weeks later,
an upstream data change shifted three key features. The model kept scoring — confidently,
silently, wrongly. Six weeks of bad predictions before anyone noticed.

ModelSentry catches that on day one.

---

## Install

```bash
pip install modelsentry
```

Python 3.11+. No Docker. No cloud account. No infrastructure.

---

## Quickstart

**Step 1 — Wrap your predict function**

```python
import modelsentry as ms

ms.init(model_id="churn-v3", profile_window=500)

@ms.monitor()
def predict(features_df):
    return model.predict(features_df)
```

One decorator. No changes to your model logic.

**Step 2 — Open the dashboard**

```bash
modelsentry serve \
  --model churn-v3 \
  --alert-email you@company.com
```

Open [http://localhost:8080](http://localhost:8080). PSI and KS drift scores update
as predictions roll in. You get an email the moment drift crosses your threshold.

---

## What you see

The dashboard always shows monitoring state — not just when something breaks:

- **Model health** — green / yellow / red per model
- **Prediction volume** — total monitored since install
- **Feature distributions** — current vs. baseline charts for every feature
- **Drift scores** — PSI and KS per feature, color-coded by severity
- **Last updated** — auto-refreshes every 60 seconds, proving the system is alive
- **Alert history** — recent drift events with timestamps and severity
- **All systems nominal** — explicitly shown when no drift is detected

---

## How it works

1. **Profile** — `@ms.monitor()` intercepts your predict function and builds
   statistical profiles of input features and output distributions locally.
2. **Detect** — profiles are compared against a baseline using PSI (Population
   Stability Index) and KS tests. Severity levels: `stable` / `warning` / `critical`.
3. **Alert** — when drift crosses threshold, an email is sent directly from your
   machine via SMTP. No cloud backend required.

---

## Email alerts

```bash
modelsentry serve \
  --model churn-v3 \
  --alert-email you@company.com \
  --smtp-host smtp.gmail.com \
  --smtp-port 587 \
  --smtp-user you@gmail.com \
  --smtp-password "your-app-password"
```

Gmail users: generate an [App Password](https://support.google.com/accounts/answer/185833)
rather than using your account password.

---

## Privacy

**Your data never leaves your machine.**

The SDK computes statistical profiles locally. Only anonymized summaries
(feature distributions, drift scores) are written to `~/.modelsentry/`.
Raw feature values and raw predictions are never written to disk, never
transmitted over a network, and never seen by ModelSentry.

This is a design constraint, not just a policy — the architecture makes
raw data transmission impossible.

---

## Who this is for

ModelSentry is built for data scientists at 50–200 person SaaS companies who:

- Have 3–8 models in production
- Are monitoring via manual ad-hoc queries, or not at all
- Need early warning before stakeholders notice model failures
- Can't justify a $500–$2,500/month enterprise monitoring platform

---

## Beta program

ModelSentry is in active beta. If you're running models in production and want
early access, we'd love to have you.

**Beta users get Pro features free forever.**

Sign up at [getmodelsentry.com](https://getmodelsentry.com) or open an issue
here on GitHub to introduce yourself and tell us what you're monitoring.

Found a bug? [Open an issue](https://github.com/treyhamilton/modelsentry/issues).
Have feedback on the dashboard or alert format? Same place. We read every one.

---

## Links

- **Website:** [getmodelsentry.com](https://getmodelsentry.com)
- **PyPI:** [pypi.org/project/modelsentry](https://pypi.org/project/modelsentry)
- **Issues:** [github.com/treyhamilton/modelsentry/issues](https://github.com/treyhamilton/modelsentry/issues)

---

## License

MIT


## Internal documentation

Product strategy, PRDs, and operational runbooks are maintained in a private
location outside this repository. Contributors who need access should contact
getmodelsentry@gmail.com.
