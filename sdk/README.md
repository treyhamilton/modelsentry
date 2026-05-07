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

ModelSentry runs as two independent processes — you don't need both running at the
same time.

**In your model (SDK side)**

`@ms.monitor()` captures inputs and outputs on every `predict()` call. Every 500
predictions (configurable via `profile_window`), it computes a statistical profile
on a background thread and saves it to `~/.modelsentry/{model_id}/`. The first
profile is automatically saved as the baseline. Your predict function is not blocked
— monitoring overhead is under 1ms.

**In your browser (dashboard side)**

`modelsentry serve` reads those profile files whenever you open it. You don't need
the dashboard running continuously — just open it when you want to check in, or
after receiving a drift alert email. It auto-refreshes every 60 seconds.

**Storage:** roughly 5–20 KB per profile. At 10,000 predictions/day with
`profile_window=500`, that's ~20 profiles/day — around 200–400 KB/day.

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

### Custom storage location

```python
ms.init(
    model_id="churn-v3",
    profile_window=500,
    storage_path="/data/modelsentry",
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
