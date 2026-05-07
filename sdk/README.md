# ModelSentry

Early warning when your production ML models start degrading.

ModelSentry computes statistical profiles of your model's inputs and outputs
locally — nothing leaves your machine. When feature distributions drift from
baseline, you get an alert before your stakeholders notice.

```
PSI 0.83 ↑  income     CRITICAL
PSI 2.43 ↑  age        CRITICAL
PSI 0.01    tenure     stable
PSI 0.00    country    stable
```

---

## Install

```bash
pip install modelsentry
```

Requires Python 3.11+. No Docker. No cloud account. No infrastructure.

---

## Quickstart

```python
import modelsentry as ms

ms.init(model_id="churn-v3", profile_window=500)

@ms.monitor()
def predict(features_df):
    return model.predict(features_df)
```

Then start the local dashboard:

```bash
modelsentry serve --model churn-v3 --alert-email you@company.com
```

Open [http://localhost:8080](http://localhost:8080) to see live drift scores,
feature distributions vs. baseline, and alert history. The dashboard updates
every 60 seconds and shows an explicit "all systems nominal" state when no
drift is detected — so you always know the system is alive.

---

## How it works

1. **Profile** — the `@ms.monitor()` decorator intercepts your predict function
   and builds statistical profiles of input features and output distributions.
2. **Detect** — profiles are compared against a baseline using PSI (Population
   Stability Index) and KS tests. Severity: stable / warning / critical.
3. **Alert** — when drift crosses threshold, ModelSentry sends an email directly
   from your machine via SMTP. No cloud backend required.

Raw feature values and raw predictions are never written to disk or transmitted.
Only anonymized statistical summaries are stored in `~/.modelsentry/`.

---

## Configuration

### Email alerts via CLI

```bash
modelsentry serve \
  --model churn-v3 \
  --alert-email you@company.com \
  --smtp-host smtp.gmail.com \
  --smtp-port 587 \
  --smtp-user you@gmail.com \
  --smtp-password "your-app-password"
```

### Profile handler (advanced)

```python
ms.init(
    model_id="churn-v3",
    profile_window=500,
    profile_handler=lambda profile: print(f"New profile: {profile}"),
)
```

---

## Dashboard

The local dashboard at `localhost:8080` shows:

- Model health overview (green / yellow / red)
- Prediction volume — total monitored since install
- Per-feature distribution charts vs. baseline
- PSI and KS drift scores, color-coded by severity
- Last updated timestamp (auto-refreshes every 60 seconds)
- Alert history with timestamps
- Explicit "all systems nominal" state when no drift is detected

---

## Links

- **Website:** [getmodelsentry.com](https://getmodelsentry.com)
- **GitHub:** [github.com/treyhamilton/modelsentry](https://github.com/treyhamilton/modelsentry)

---

## License

MIT
