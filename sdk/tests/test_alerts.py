"""Tests for modelsentry.alerts — email alert module."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modelsentry.alerts import AlertConfig, _build_body, _build_subject, send_drift_alert
from modelsentry.drift import DriftReport, FeatureDriftResult, detect_drift
from modelsentry.profiler import profile


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_profile(seed: int = 42, n_rows: int = 100):
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


def _make_warning_report() -> DriftReport:
    """Real DriftReport with at least one warning feature."""
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
    return detect_drift(baseline, current)


def _make_stable_report() -> DriftReport:
    """DriftReport where all features are stable."""
    baseline = _make_profile(seed=1)
    edges = {
        name: fp.distribution.bin_edges
        for name, fp in baseline.feature_profiles.items()
        if fp.distribution is not None
    }
    current = profile(
        pd.DataFrame(
            {
                "age": np.random.default_rng(1).integers(18, 80, size=100),
                "income": np.random.default_rng(1).normal(50000, 15000, size=100),
                "country": np.random.default_rng(1).choice(["US", "CA", "UK"], size=100),
            }
        ),
        np.random.default_rng(1).normal(size=100),
        baseline_edges=edges,
    )
    report = detect_drift(baseline, current)
    # Force stable for test clarity
    return DriftReport(
        schema_version=report.schema_version,
        overall_severity="stable",
        feature_results=report.feature_results,
        missing_in_current=report.missing_in_current,
        missing_in_baseline=report.missing_in_baseline,
    )


def _make_critical_report() -> DriftReport:
    """DriftReport with overall_severity=critical."""
    r = _make_warning_report()
    return DriftReport(
        schema_version=r.schema_version,
        overall_severity="critical",
        feature_results=r.feature_results,
        missing_in_current=r.missing_in_current,
        missing_in_baseline=r.missing_in_baseline,
    )


def _base_config(**overrides) -> AlertConfig:
    defaults = dict(
        recipient_email="alex@example.com",
        smtp_user="sender@example.com",
        smtp_password="secret",
    )
    defaults.update(overrides)
    return AlertConfig(**defaults)


# ---------------------------------------------------------------------------
# SMTP mock helper
# ---------------------------------------------------------------------------


@contextmanager
def mock_smtp():
    with patch("smtplib.SMTP") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value.__enter__ = lambda s: instance
        mock_cls.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_cls, instance


# ---------------------------------------------------------------------------
# AlertConfig defaults
# ---------------------------------------------------------------------------


def test_alertconfig_defaults() -> None:
    cfg = AlertConfig(recipient_email="x@example.com")
    assert cfg.smtp_host == "smtp.gmail.com"
    assert cfg.smtp_port == 587
    assert cfg.min_severity == "warning"
    assert cfg.model_id_filter is None
    assert cfg.from_email is None


def test_alertconfig_from_email_defaults_to_smtp_user() -> None:
    """from_email=None means send_drift_alert uses smtp_user as sender."""
    report = _make_warning_report()
    cfg = _base_config(smtp_user="sender@example.com", from_email=None)
    with mock_smtp() as (_cls, instance):
        send_drift_alert(report, "m1", cfg)
    call_args = instance.sendmail.call_args
    assert call_args[0][0] == "sender@example.com"


# ---------------------------------------------------------------------------
# Success / severity threshold
# ---------------------------------------------------------------------------


def test_send_drift_alert_success_returns_true() -> None:
    report = _make_warning_report()
    with mock_smtp():
        result = send_drift_alert(report, "churn-v3", _base_config())
    assert result is True


def test_send_drift_alert_stable_does_not_send() -> None:
    report = _make_stable_report()
    with mock_smtp() as (_cls, instance):
        result = send_drift_alert(report, "m1", _base_config(min_severity="warning"))
    assert result is False
    instance.sendmail.assert_not_called()


def test_send_drift_alert_warning_meets_warning_threshold() -> None:
    report = DriftReport(
        schema_version="1.0",
        overall_severity="warning",
        feature_results={},
        missing_in_current=(),
        missing_in_baseline=(),
    )
    with mock_smtp():
        assert send_drift_alert(report, "m1", _base_config(min_severity="warning")) is True


def test_send_drift_alert_warning_blocked_by_critical_threshold() -> None:
    report = DriftReport(
        schema_version="1.0",
        overall_severity="warning",
        feature_results={},
        missing_in_current=(),
        missing_in_baseline=(),
    )
    with mock_smtp() as (_cls, instance):
        result = send_drift_alert(report, "m1", _base_config(min_severity="critical"))
    assert result is False
    instance.sendmail.assert_not_called()


def test_send_drift_alert_critical_meets_critical_threshold() -> None:
    report = _make_critical_report()
    with mock_smtp():
        assert send_drift_alert(report, "m1", _base_config(min_severity="critical")) is True


def test_send_drift_alert_critical_meets_warning_threshold() -> None:
    report = _make_critical_report()
    with mock_smtp():
        assert send_drift_alert(report, "m1", _base_config(min_severity="warning")) is True


# ---------------------------------------------------------------------------
# model_id_filter
# ---------------------------------------------------------------------------


def test_model_id_filter_match_sends() -> None:
    report = _make_warning_report()
    cfg = _base_config(model_id_filter=["churn-v3", "fraud-v1"])
    with mock_smtp():
        assert send_drift_alert(report, "churn-v3", cfg) is True


def test_model_id_filter_no_match_suppresses() -> None:
    report = _make_warning_report()
    cfg = _base_config(model_id_filter=["fraud-v1"])
    with mock_smtp() as (_cls, instance):
        result = send_drift_alert(report, "churn-v3", cfg)
    assert result is False
    instance.sendmail.assert_not_called()


def test_model_id_filter_none_sends_all() -> None:
    report = _make_warning_report()
    cfg = _base_config(model_id_filter=None)
    with mock_smtp():
        assert send_drift_alert(report, "any-model-id", cfg) is True


# ---------------------------------------------------------------------------
# SMTP failure handling
# ---------------------------------------------------------------------------


def test_smtp_exception_returns_false() -> None:
    report = _make_warning_report()
    with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("refused")):
        result = send_drift_alert(report, "m1", _base_config())
    assert result is False


def test_send_drift_alert_never_raises() -> None:
    report = _make_warning_report()
    with patch("smtplib.SMTP", side_effect=RuntimeError("unexpected")):
        result = send_drift_alert(report, "m1", _base_config())
    assert result is False


# ---------------------------------------------------------------------------
# Email content
# ---------------------------------------------------------------------------


def test_email_subject_format() -> None:
    subject = _build_subject("churn-v3", "critical")
    assert "churn-v3" in subject
    assert "critical" in subject
    assert subject == "ModelSentry Alert: churn-v3 — critical drift detected"


def test_email_body_contains_feature_names_and_psi() -> None:
    result = FeatureDriftResult(
        name="age",
        dtype="numeric",
        severity="critical",
        psi=0.8753,
        ks_statistic=0.45,
        ks_p_value=0.001,
        notes=(),
    )
    report = DriftReport(
        schema_version="1.0",
        overall_severity="critical",
        feature_results={"age": result},
        missing_in_current=(),
        missing_in_baseline=(),
    )
    body = _build_body(report, "churn-v3")
    assert "age" in body
    assert "0.8753" in body
    assert "churn-v3" in body
    assert "critical" in body


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_send_drift_alert_logs_success(caplog) -> None:
    report = _make_warning_report()
    with mock_smtp():
        with caplog.at_level(logging.INFO, logger="modelsentry.alerts"):
            send_drift_alert(report, "churn-v3", _base_config())
    assert any("churn-v3" in r.message for r in caplog.records)


def test_send_drift_alert_logs_failure(caplog) -> None:
    report = _make_warning_report()
    with patch("smtplib.SMTP", side_effect=OSError("timeout")):
        with caplog.at_level(logging.WARNING, logger="modelsentry.alerts"):
            send_drift_alert(report, "churn-v3", _base_config())
    assert any("churn-v3" in r.message for r in caplog.records)
