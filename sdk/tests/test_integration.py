"""End-to-end integration test for ModelSentry.

Proves Definition of Done item #5: the full flow from @ms.monitor() through
storage, drift detection, email alert, and dashboard API endpoints works as a
single coherent system. No new production code — only this test file.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import modelsentry as ms
from modelsentry import server, storage
from modelsentry.alerts import AlertConfig
from modelsentry.alerts import send_drift_alert as real_send_drift_alert
from modelsentry.drift import detect_drift
from modelsentry.monitor import flush, shutdown


MODEL_ID = "churn-v3"
PROFILE_WINDOW = 20  # small for fast tests; real default is 100
RECIPIENT = "alex@example.com"
SMTP_USER = "sender@example.com"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Patch STORAGE_ROOT so ~/.modelsentry is never touched."""
    monkeypatch.setattr(storage, "STORAGE_ROOT", tmp_path)
    yield tmp_path
    storage.set_alert_callback(None)


@pytest.fixture
def dashboard_file(tmp_path):
    path = tmp_path / "dashboard.html"
    path.write_text(
        "<html><body>ModelSentry Dashboard</body></html>", encoding="utf-8"
    )
    return path


@pytest.fixture(autouse=True)
def _shutdown_sdk():
    """Always clean up monitor.py global state after each test."""
    yield
    try:
        shutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Synthetic data + model helpers
# ---------------------------------------------------------------------------


def _baseline_features(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, size=n),
            "income": rng.normal(50_000, 12_000, size=n),
            "country": rng.choice(["US", "CA", "UK"], size=n),
        }
    )


def _drifted_features(n: int = 200, seed: int = 99) -> pd.DataFrame:
    """Deliberate large shift on age and income — guarantees critical drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(45, 95, size=n),
            "income": rng.normal(85_000, 12_000, size=n),
            "country": rng.choice(["US", "CA", "UK"], size=n),
        }
    )


def _train_model(features: pd.DataFrame) -> LogisticRegression:
    rng = np.random.default_rng(42)
    X = features[["age", "income"]].to_numpy()
    y = rng.integers(0, 2, size=len(features))
    return LogisticRegression(max_iter=200).fit(X, y)


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------


def test_full_flow_install_to_alert(tmp_storage, dashboard_file):
    """End-to-end: ms.monitor → storage → drift → alert → dashboard API.

    Reads as a 12-phase narrative — each phase proves one layer integrates
    correctly with the next.
    """

    # ------------------------------------------------------------------
    # Phase 0: Setup — alert config + FastAPI app wired with TestClient
    # ------------------------------------------------------------------
    alert_config = AlertConfig(
        recipient_email=RECIPIENT,
        smtp_user=SMTP_USER,
        smtp_password="app-password",
        min_severity="warning",
    )
    app = server.create_app(dashboard_path=dashboard_file, alert_config=alert_config)
    client = TestClient(app)

    # ------------------------------------------------------------------
    # Phase 1: ms.init() — handler routes computed profiles to local storage.
    # Timestamps are made unique because two flushes within the same wall-clock
    # second would collide on storage's per-second filename granularity (a
    # test-only artifact; real profile_windows are minutes apart).
    # ------------------------------------------------------------------
    profile_count = 0

    def save_handler(prof, mid: str) -> None:
        nonlocal profile_count
        ts = f"2026-05-06T10-00-{profile_count:02d}"
        storage.save_profile(prof, mid, timestamp=ts)
        profile_count += 1

    ms.init(
        api_key="test-key",
        model_id=MODEL_ID,
        profile_window=PROFILE_WINDOW,
        profile_handler=save_handler,
    )

    # ------------------------------------------------------------------
    # Phase 2: Train a real LogisticRegression model on baseline data
    # ------------------------------------------------------------------
    baseline_pool = _baseline_features(n=200, seed=0)
    model = _train_model(baseline_pool)

    # ------------------------------------------------------------------
    # Phase 3: Wrap predict with @ms.monitor() — captures DataFrame inputs
    # ------------------------------------------------------------------
    @ms.monitor()
    def predict(features_df: pd.DataFrame) -> np.ndarray:
        return model.predict(features_df[["age", "income"]].to_numpy())

    # ------------------------------------------------------------------
    # Phase 4: Baseline batch — 20 single-row predictions trigger one window
    # ------------------------------------------------------------------
    for i in range(PROFILE_WINDOW):
        out = predict(baseline_pool.iloc[[i]])
        assert out.shape == (1,)

    # ------------------------------------------------------------------
    # Phase 5: flush() forces profile computation; verify saved + promote to baseline
    # ------------------------------------------------------------------
    flush(MODEL_ID)
    saved = storage.load_profiles(MODEL_ID, limit=1)
    assert len(saved) == 1
    baseline_profile = saved[0]
    assert baseline_profile.n_rows == PROFILE_WINDOW
    assert "age" in baseline_profile.feature_profiles
    assert "country" in baseline_profile.feature_profiles
    storage.save_baseline(baseline_profile, MODEL_ID)
    assert storage.load_baseline(MODEL_ID) is not None

    # ------------------------------------------------------------------
    # Phase 6: Drifted batch — same predict function, shifted distribution
    # ------------------------------------------------------------------
    drifted_pool = _drifted_features(n=200, seed=99)
    for i in range(PROFILE_WINDOW):
        predict(drifted_pool.iloc[[i]])

    # ------------------------------------------------------------------
    # Phase 7: flush() saves the second profile
    # ------------------------------------------------------------------
    flush(MODEL_ID)
    all_profiles = storage.load_profiles(MODEL_ID, limit=10)
    assert len(all_profiles) == 2
    current_profile = all_profiles[0]  # newest first
    assert current_profile.n_rows == PROFILE_WINDOW

    # ------------------------------------------------------------------
    # Phase 8: detect_drift produces a DriftReport with warning/critical severity
    # ------------------------------------------------------------------
    report = detect_drift(baseline_profile, current_profile)
    assert report.overall_severity in ("warning", "critical")
    severities = {r.severity for r in report.feature_results.values()}
    assert "critical" in severities or "warning" in severities

    # ------------------------------------------------------------------
    # Phase 9 + 10: Save drift report — alert callback fires.
    # SMTP mocked at the socket boundary; send_drift_alert spied via wraps.
    # ------------------------------------------------------------------
    smtp_instance = MagicMock()
    smtp_cm = MagicMock()
    smtp_cm.__enter__ = lambda s: smtp_instance
    smtp_cm.__exit__ = MagicMock(return_value=False)

    with patch("smtplib.SMTP", return_value=smtp_cm) as mock_smtp_cls, patch(
        "modelsentry.server.send_drift_alert", wraps=real_send_drift_alert
    ) as alert_spy:
        storage.save_drift_report(report, MODEL_ID)

    alert_spy.assert_called_once()
    call_args = alert_spy.call_args[0]
    assert call_args[1] == MODEL_ID
    assert call_args[2] is alert_config

    mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587)
    smtp_instance.starttls.assert_called_once()
    smtp_instance.login.assert_called_once_with(SMTP_USER, "app-password")
    smtp_instance.sendmail.assert_called_once()
    sender, recipients, raw_msg = smtp_instance.sendmail.call_args[0]
    assert sender == SMTP_USER
    assert recipients == [RECIPIENT]
    assert MODEL_ID in raw_msg

    # ------------------------------------------------------------------
    # Phase 11: Dashboard API — every endpoint reflects the full state
    # ------------------------------------------------------------------
    r = client.get("/api/models")
    assert r.status_code == 200
    assert MODEL_ID in r.json()["models"]

    r = client.get(f"/api/models/{MODEL_ID}/status")
    assert r.status_code == 200
    body = r.json()
    assert body["overall_severity"] == report.overall_severity
    assert body["baseline_exists"] is True
    # baseline + 2 saved profiles, each PROFILE_WINDOW rows. The baseline
    # counts toward the total (see storage.get_prediction_count docstring).
    assert body["prediction_count"] == PROFILE_WINDOW * 3
    assert body["last_updated"] is not None

    r = client.get(f"/api/models/{MODEL_ID}/features")
    assert r.status_code == 200
    body = r.json()
    assert body["has_baseline"] is True
    assert body["has_current"] is True
    feat_names = {f["name"] for f in body["features"]}
    assert {"age", "income", "country"}.issubset(feat_names)
    age_view = next(f for f in body["features"] if f["name"] == "age")
    assert age_view["severity"] in ("warning", "critical")
    assert age_view["psi"] is None or math.isfinite(age_view["psi"])

    r = client.get(f"/api/models/{MODEL_ID}/drift")
    assert r.status_code == 200
    drift_body = r.json()
    assert len(drift_body) == 1
    assert drift_body[0]["overall_severity"] == report.overall_severity
    assert drift_body[0]["detected_at"] is not None

    # ------------------------------------------------------------------
    # Phase 12: Dashboard HTML served at GET /
    # ------------------------------------------------------------------
    r = client.get("/")
    assert r.status_code == 200
    assert "ModelSentry Dashboard" in r.text
