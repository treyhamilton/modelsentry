"""Tests for modelsentry.server (FastAPI local dashboard)."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from modelsentry import server, storage
from modelsentry.drift import DriftReport, FeatureDriftResult, detect_drift
from modelsentry.profiler import profile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Patch storage.STORAGE_ROOT so tests never touch ~/.modelsentry/."""
    monkeypatch.setattr(storage, "STORAGE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def dashboard_file(tmp_path):
    """Write a tmp dashboard HTML file and return its path."""
    path = tmp_path / "dashboard.html"
    path.write_text("<html><body>dashboard</body></html>", encoding="utf-8")
    return path


@pytest.fixture
def client(tmp_storage, dashboard_file):
    """FastAPI TestClient bound to a freshly built app pointing at the tmp dashboard."""
    app = server.create_app(dashboard_path=dashboard_file)
    return TestClient(app)


@pytest.fixture
def client_no_dashboard(tmp_storage, tmp_path):
    """Client whose dashboard_path does not exist on disk."""
    missing = tmp_path / "does_not_exist.html"
    app = server.create_app(dashboard_path=missing)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_profile(seed: int = 42, n_rows: int = 100):
    """Build a real Profile via profiler.profile()."""
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n_rows),
            "income": rng.normal(50000, 15000, size=n_rows),
            "country": rng.choice(["US", "CA", "UK"], size=n_rows),
        }
    )
    preds = rng.normal(size=n_rows)
    return profile(features, preds)


def _make_drift_report():
    """Build a real DriftReport via detect_drift()."""
    baseline = _make_profile(seed=42)
    edges = {
        name: fp.distribution.bin_edges
        for name, fp in baseline.feature_profiles.items()
        if fp.distribution is not None
    }
    rng = np.random.default_rng(99)
    current_features = pd.DataFrame(
        {
            "age": rng.integers(25, 90, size=100),
            "income": rng.normal(60000, 15000, size=100),
            "country": rng.choice(["US", "CA", "UK"], size=100),
        }
    )
    current_preds = rng.normal(size=100)
    current = profile(current_features, current_preds, baseline_edges=edges)
    return detect_drift(baseline, current), baseline, current


# ---------------------------------------------------------------------------
# /health and /
# ---------------------------------------------------------------------------


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_localhost_binding_constant():
    assert server.HOST == "127.0.0.1"
    assert server.DEFAULT_PORT == 8080


def test_dashboard_html_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "dashboard" in resp.text


def test_dashboard_html_404_when_missing(client_no_dashboard):
    resp = client_no_dashboard.get("/")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------


def test_list_models_empty(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == {"models": []}


def test_list_models_after_save(client):
    storage.save_baseline(_make_profile(), "churn-v3")
    storage.save_baseline(_make_profile(), "fraud-v1")

    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == {"models": ["churn-v3", "fraud-v1"]}


# ---------------------------------------------------------------------------
# /api/models/{id}/status
# ---------------------------------------------------------------------------


def test_status_404_when_model_unknown(client):
    resp = client.get("/api/models/unknown/status")
    assert resp.status_code == 404


def test_status_unknown_severity_no_drift(client):
    storage.save_baseline(_make_profile(), "m1")
    storage.save_profile(_make_profile(seed=2), "m1", timestamp="2026-05-06T10-00-00")

    resp = client.get("/api/models/m1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_severity"] == "unknown"
    assert body["model_id"] == "m1"


def test_status_severity_from_drift_report(client):
    report, baseline, _current = _make_drift_report()
    storage.save_baseline(baseline, "m1")
    storage.save_drift_report(report, "m1", timestamp="2026-05-06T11-00-00")

    resp = client.get("/api/models/m1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_severity"] == report.overall_severity


def test_status_baseline_exists_flag(client):
    # Save a profile but no baseline
    storage.save_profile(_make_profile(), "m1", timestamp="2026-05-06T10-00-00")

    resp = client.get("/api/models/m1/status")
    assert resp.status_code == 200
    assert resp.json()["baseline_exists"] is False

    # Now add baseline and re-check
    storage.save_baseline(_make_profile(), "m1")
    resp = client.get("/api/models/m1/status")
    assert resp.json()["baseline_exists"] is True


def test_status_prediction_count(client):
    storage.save_baseline(_make_profile(n_rows=100), "m1")
    storage.save_profile(_make_profile(n_rows=50, seed=2), "m1", timestamp="2026-05-06T10-00-00")
    storage.save_profile(_make_profile(n_rows=75, seed=3), "m1", timestamp="2026-05-06T11-00-00")

    resp = client.get("/api/models/m1/status")
    assert resp.json()["prediction_count"] == 225


def test_status_last_updated_iso_format(client):
    storage.save_profile(_make_profile(), "m1", timestamp="2026-05-06T10-00-00")

    body = client.get("/api/models/m1/status").json()
    assert body["last_updated"] is not None
    parsed = datetime.fromisoformat(body["last_updated"])
    assert parsed.tzinfo is not None


def test_status_last_updated_none_when_only_baseline(client):
    """Baseline alone is not proof of life — last_updated should be None."""
    storage.save_baseline(_make_profile(), "m1")
    body = client.get("/api/models/m1/status").json()
    assert body["last_updated"] is None


# ---------------------------------------------------------------------------
# /api/models/{id}/profiles
# ---------------------------------------------------------------------------


def test_profiles_endpoint_returns_list(client):
    prof = _make_profile()
    storage.save_profile(prof, "m1", timestamp="2026-05-06T10-00-00")

    resp = client.get("/api/models/m1/profiles")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["n_rows"] == prof.n_rows
    assert "feature_profiles" in body[0]
    assert "age" in body[0]["feature_profiles"]


def test_profiles_endpoint_limit(client):
    storage.save_profile(_make_profile(seed=1), "m1", timestamp="2026-05-06T10-00-00")
    storage.save_profile(_make_profile(seed=2), "m1", timestamp="2026-05-06T11-00-00")
    storage.save_profile(_make_profile(seed=3), "m1", timestamp="2026-05-06T12-00-00")

    resp = client.get("/api/models/m1/profiles?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_profiles_endpoint_404_when_no_data(client):
    resp = client.get("/api/models/empty/profiles")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/models/{id}/drift
# ---------------------------------------------------------------------------


def test_drift_endpoint_returns_list(client):
    report, baseline, _current = _make_drift_report()
    storage.save_baseline(baseline, "m1")
    storage.save_drift_report(report, "m1", timestamp="2026-05-06T11-00-00")

    resp = client.get("/api/models/m1/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["overall_severity"] == report.overall_severity
    assert "feature_results" in body[0]


def test_drift_endpoint_nan_psi_serialized_as_null(client):
    """A FeatureDriftResult with psi=NaN must surface as JSON null, not a NaN token."""
    nan_result = FeatureDriftResult(
        name="oddball",
        dtype="numeric",
        severity="warning",
        psi=float("nan"),
        ks_statistic=None,
        ks_p_value=None,
        notes=("dtype mismatch",),
    )
    report = DriftReport(
        schema_version="1.0",
        overall_severity="warning",
        feature_results={"oddball": nan_result},
        missing_in_current=(),
        missing_in_baseline=(),
    )
    storage.save_baseline(_make_profile(), "m1")
    storage.save_drift_report(report, "m1", timestamp="2026-05-06T11-00-00")

    resp = client.get("/api/models/m1/drift")
    assert resp.status_code == 200
    body = resp.json()
    # JSON null deserializes back to None — not float('nan')
    assert body[0]["feature_results"]["oddball"]["psi"] is None
    # Sanity: response body should not contain the literal string "NaN"
    assert "NaN" not in resp.text


def test_drift_endpoint_404_when_no_data(client):
    resp = client.get("/api/models/empty/drift")
    assert resp.status_code == 404


def test_drift_endpoint_includes_detected_at(client):
    """Each drift report in the /drift response includes a detected_at ISO timestamp."""
    report, baseline, _current = _make_drift_report()
    storage.save_baseline(baseline, "m1")
    storage.save_drift_report(report, "m1", timestamp="2026-05-06T11-00-00")

    resp = client.get("/api/models/m1/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert "detected_at" in body[0]
    assert body[0]["detected_at"] is not None
    # Must be parseable as an ISO datetime
    from datetime import datetime
    parsed = datetime.fromisoformat(body[0]["detected_at"])
    assert parsed.tzinfo is not None


# ---------------------------------------------------------------------------
# /api/models/{id}/features
# ---------------------------------------------------------------------------


def test_features_endpoint_combines_data(client):
    report, baseline, current = _make_drift_report()
    storage.save_baseline(baseline, "m1")
    storage.save_profile(current, "m1", timestamp="2026-05-06T11-00-00")
    storage.save_drift_report(report, "m1", timestamp="2026-05-06T11-00-00")

    resp = client.get("/api/models/m1/features")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "m1"
    assert body["has_baseline"] is True
    assert body["has_current"] is True

    feature_names = [f["name"] for f in body["features"]]
    assert "age" in feature_names
    assert "country" in feature_names

    # numeric feature should have distributions and severity from drift report
    age_view = next(f for f in body["features"] if f["name"] == "age")
    assert age_view["dtype"] == "numeric"
    assert age_view["severity"] in ("stable", "warning", "critical")
    assert age_view["baseline_distribution"] is not None
    assert age_view["current_distribution"] is not None
    assert isinstance(age_view["baseline_distribution"]["bin_edges"], list)

    # categorical feature should carry value_counts, no distribution
    country_view = next(f for f in body["features"] if f["name"] == "country")
    assert country_view["dtype"] == "categorical"
    assert country_view["baseline_distribution"] is None
    assert country_view["baseline_value_counts"] is not None
    assert country_view["current_value_counts"] is not None


def test_features_endpoint_severity_unknown_without_drift_report(client):
    """When no drift report exists, severity falls back to 'unknown' and psi is None."""
    _report, baseline, current = _make_drift_report()
    storage.save_baseline(baseline, "m1")
    storage.save_profile(current, "m1", timestamp="2026-05-06T11-00-00")
    # Deliberately do NOT save a drift report

    resp = client.get("/api/models/m1/features")
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_baseline"] is True
    assert body["has_current"] is True
    for feat in body["features"]:
        assert feat["severity"] == "unknown"
        assert feat["psi"] is None
        assert feat["ks_p_value"] is None


def test_features_endpoint_no_baseline(client):
    """has_baseline=False when no baseline saved; current features still surfaced."""
    storage.save_profile(_make_profile(), "m1", timestamp="2026-05-06T10-00-00")
    resp = client.get("/api/models/m1/features")
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_baseline"] is False
    assert body["has_current"] is True
    assert len(body["features"]) > 0
    for feat in body["features"]:
        assert feat["baseline_distribution"] is None
        assert feat["baseline_value_counts"] is None


def test_features_endpoint_404_when_model_unknown(client):
    resp = client.get("/api/models/unknown/features")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(client):
    """model_id containing path-traversal characters must be rejected by FastAPI."""
    # Two ways to express path traversal — both must 404 or 422, never 200
    resp = client.get("/api/models/..%2Fetc/status")
    assert resp.status_code in (404, 422)

    # Spaces and slashes also fail the regex
    resp = client.get("/api/models/has%20space/status")
    assert resp.status_code == 422


def test_invalid_model_id_format_rejected(client):
    """Non-conforming model_ids return 422 from the FastAPI validator."""
    resp = client.get("/api/models/-leading-dash/status")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# NaN psi roundtrip via Pydantic
# ---------------------------------------------------------------------------


def test_pydantic_nan_validator_directly():
    """The FeatureDriftResultModel field_validator converts NaN floats to None."""
    m = server.FeatureDriftResultModel(
        name="x",
        dtype="numeric",
        severity="warning",
        psi=float("nan"),
        ks_statistic=None,
        ks_p_value=None,
        notes=[],
    )
    assert m.psi is None

    m2 = server.FeatureDriftResultModel(
        name="x",
        dtype="numeric",
        severity="stable",
        psi=0.05,
        ks_statistic=0.1,
        ks_p_value=0.5,
        notes=[],
    )
    assert m2.psi == 0.05
    assert math.isfinite(m2.psi)
