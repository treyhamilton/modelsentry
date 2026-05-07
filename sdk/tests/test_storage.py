"""Tests for modelsentry.storage."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelsentry.drift import FeatureDriftResult, detect_drift
from modelsentry.profiler import profile
import modelsentry.storage as storage_module
from modelsentry.storage import (
    get_last_updated,
    get_prediction_count,
    list_models,
    load_baseline,
    load_drift_reports,
    load_drift_reports_with_timestamps,
    load_profiles,
    save_baseline,
    save_drift_report,
    save_profile,
    set_alert_callback,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Patch STORAGE_ROOT to tmp_path for test isolation and clear alert callback."""
    monkeypatch.setattr(storage_module, "STORAGE_ROOT", tmp_path)
    yield tmp_path
    set_alert_callback(None)


def make_profile(n_rows: int = 100, seed: int = 42) -> tuple:
    """Build a real Profile using profiler.profile().

    Returns (profile, features_df, predictions_array).
    """
    rng = np.random.default_rng(seed)
    features_df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n_rows),
            "income": rng.normal(50000, 15000, size=n_rows),
            "country": rng.choice(["US", "CA", "UK"], size=n_rows),
        }
    )
    predictions = rng.normal(size=n_rows)
    prof = profile(features_df, predictions)
    return prof, features_df, predictions


def make_drift_report() -> tuple:
    """Build a real DriftReport using detect_drift().

    Returns (drift_report, baseline_profile, current_profile).
    """
    baseline, base_features, base_preds = make_profile(n_rows=100, seed=42)

    # Shift distribution slightly for current
    rng = np.random.default_rng(99)
    current_features = pd.DataFrame(
        {
            "age": rng.integers(25, 90, size=100),
            "income": rng.normal(60000, 15000, size=100),
            "country": rng.choice(["US", "CA", "UK"], size=100),
        }
    )
    # Use baseline_edges for fair comparison
    edges_map = {
        name: baseline.feature_profiles[name].distribution.bin_edges
        for name in baseline.feature_profiles
        if baseline.feature_profiles[name].distribution is not None
    }
    current_preds = rng.normal(size=100)
    current = profile(
        current_features,
        current_preds,
        baseline_edges=edges_map,
    )

    report = detect_drift(baseline, current)
    return report, baseline, current


# ---------------------------------------------------------------------------
# Test: save_profile / load_profiles
# ---------------------------------------------------------------------------


def test_save_profile_creates_file(tmp_storage):
    """File appears at correct path under profiles/."""
    prof, _, _ = make_profile()
    model_id = "test-model"
    path = save_profile(prof, model_id, timestamp="2026-05-06T10-00-00")

    assert path.exists()
    assert path.name == "2026-05-06T10-00-00.json"
    assert path.parent.name == "profiles"
    assert path.parent.parent.name == model_id


def test_profile_roundtrip(tmp_storage):
    """load_profiles()[0] == original Profile."""
    prof, _, _ = make_profile()
    model_id = "test-model"
    save_profile(prof, model_id, timestamp="2026-05-06T10-00-00")

    loaded = load_profiles(model_id)
    assert len(loaded) == 1
    assert loaded[0] == prof


def test_profile_roundtrip_numeric_stats(tmp_storage):
    """NumericStats fields survive exactly."""
    prof, _, _ = make_profile()
    model_id = "test-model"
    save_profile(prof, model_id)

    loaded = load_profiles(model_id)[0]
    age_prof = loaded.feature_profiles["age"]
    age_stats = age_prof.numeric_stats

    orig_age_prof = prof.feature_profiles["age"]
    orig_stats = orig_age_prof.numeric_stats

    assert age_stats.mean == orig_stats.mean
    assert age_stats.std == orig_stats.std
    assert age_stats.min == orig_stats.min
    assert age_stats.max == orig_stats.max
    assert age_stats.p25 == orig_stats.p25
    assert age_stats.p50 == orig_stats.p50
    assert age_stats.p75 == orig_stats.p75


def test_profile_roundtrip_distribution_tuples(tmp_storage):
    """bin_edges and bin_counts are tuple (not list) after load."""
    prof, _, _ = make_profile()
    model_id = "test-model"
    save_profile(prof, model_id)

    loaded = load_profiles(model_id)[0]
    age_dist = loaded.feature_profiles["age"].distribution

    assert isinstance(age_dist.bin_edges, tuple)
    assert isinstance(age_dist.bin_counts, tuple)
    assert all(isinstance(e, float) for e in age_dist.bin_edges)
    assert all(isinstance(c, int) for c in age_dist.bin_counts)


def test_profile_roundtrip_categorical(tmp_storage):
    """value_counts dict survives."""
    prof, _, _ = make_profile()
    model_id = "test-model"
    save_profile(prof, model_id)

    loaded = load_profiles(model_id)[0]
    country_counts = loaded.feature_profiles["country"].value_counts
    orig_counts = prof.feature_profiles["country"].value_counts

    assert country_counts == orig_counts


def test_load_profiles_newest_first(tmp_storage):
    """Two saved profiles return newest first."""
    prof1, _, _ = make_profile(seed=10)
    prof2, _, _ = make_profile(seed=20)

    model_id = "test-model"
    save_profile(prof1, model_id, timestamp="2026-05-06T10-00-00")
    save_profile(prof2, model_id, timestamp="2026-05-06T12-00-00")

    loaded = load_profiles(model_id)
    assert len(loaded) == 2
    # Second one (seed=20) should come first (newest)
    assert loaded[0].n_rows == prof2.n_rows
    assert loaded[1].n_rows == prof1.n_rows


def test_load_profiles_limit(tmp_storage):
    """limit=1 returns only one item."""
    prof1, _, _ = make_profile(seed=10)
    prof2, _, _ = make_profile(seed=20)

    model_id = "test-model"
    save_profile(prof1, model_id, timestamp="2026-05-06T10-00-00")
    save_profile(prof2, model_id, timestamp="2026-05-06T12-00-00")

    loaded = load_profiles(model_id, limit=1)
    assert len(loaded) == 1
    assert loaded[0].n_rows == prof2.n_rows


def test_load_profiles_empty_dir(tmp_storage):
    """returns [] when no files exist."""
    model_id = "nonexistent-model"
    loaded = load_profiles(model_id)
    assert loaded == []


# ---------------------------------------------------------------------------
# Test: save_baseline / load_baseline
# ---------------------------------------------------------------------------


def test_save_baseline_creates_file(tmp_storage):
    """baseline.json exists at correct path."""
    prof, _, _ = make_profile()
    model_id = "test-model"
    path = save_baseline(prof, model_id)

    assert path.exists()
    assert path.name == "baseline.json"
    assert path.parent.name == model_id


def test_baseline_roundtrip(tmp_storage):
    """load_baseline() == original Profile."""
    prof, _, _ = make_profile()
    model_id = "test-model"
    save_baseline(prof, model_id)

    loaded = load_baseline(model_id)
    assert loaded is not None
    assert loaded == prof


def test_load_baseline_missing(tmp_storage):
    """returns None when no baseline.json exists."""
    model_id = "nonexistent-model"
    loaded = load_baseline(model_id)
    assert loaded is None


# ---------------------------------------------------------------------------
# Test: save_drift_report / load_drift_reports
# ---------------------------------------------------------------------------


def test_save_drift_report_creates_file(tmp_storage):
    """File appears under drift_reports/."""
    report, _, _ = make_drift_report()
    model_id = "test-model"
    path = save_drift_report(report, model_id, timestamp="2026-05-06T15-00-00")

    assert path.exists()
    assert path.name == "2026-05-06T15-00-00.json"
    assert path.parent.name == "drift_reports"
    assert path.parent.parent.name == model_id


def test_drift_report_roundtrip_stable(tmp_storage):
    """All fields match for a stable report."""
    report, _, _ = make_drift_report()
    model_id = "test-model"
    save_drift_report(report, model_id)

    loaded = load_drift_reports(model_id)
    assert len(loaded) == 1

    # Compare non-NaN fields
    loaded_report = loaded[0]
    assert loaded_report.schema_version == report.schema_version
    assert loaded_report.overall_severity == report.overall_severity

    # Check feature results
    assert set(loaded_report.feature_results.keys()) == set(report.feature_results.keys())
    for feat_name in report.feature_results:
        orig = report.feature_results[feat_name]
        loaded_feat = loaded_report.feature_results[feat_name]

        assert loaded_feat.name == orig.name
        assert loaded_feat.dtype == orig.dtype
        assert loaded_feat.severity == orig.severity
        if math.isfinite(orig.psi):
            assert loaded_feat.psi == orig.psi
        assert loaded_feat.ks_statistic == orig.ks_statistic
        assert loaded_feat.ks_p_value == orig.ks_p_value
        assert loaded_feat.notes == orig.notes


def test_drift_report_roundtrip_nan_psi(tmp_storage):
    """A FeatureDriftResult with psi=float('nan') survives roundtrip."""
    # Create a report with a NaN psi manually
    from modelsentry.drift import DriftReport

    nan_result = FeatureDriftResult(
        name="test_feature",
        dtype="numeric",
        severity="warning",
        psi=float("nan"),
        ks_statistic=None,
        ks_p_value=None,
        notes=("test note",),
    )
    report = DriftReport(
        schema_version="1.0",
        overall_severity="warning",
        feature_results={"test_feature": nan_result},
        missing_in_current=(),
        missing_in_baseline=(),
    )

    model_id = "test-model"
    save_drift_report(report, model_id)

    loaded = load_drift_reports(model_id)
    assert len(loaded) == 1
    loaded_report = loaded[0]

    # Verify file was written and loaded
    assert "test_feature" in loaded_report.feature_results
    loaded_result = loaded_report.feature_results["test_feature"]

    # Verify NaN psi survives
    assert math.isnan(loaded_result.psi)
    assert loaded_result.name == "test_feature"
    assert loaded_result.dtype == "numeric"
    assert loaded_result.severity == "warning"


def test_drift_report_tuples(tmp_storage):
    """notes, missing_in_current, missing_in_baseline are tuple after load."""
    report, _, _ = make_drift_report()
    model_id = "test-model"
    save_drift_report(report, model_id)

    loaded = load_drift_reports(model_id)[0]

    assert isinstance(loaded.missing_in_current, tuple)
    assert isinstance(loaded.missing_in_baseline, tuple)

    for result in loaded.feature_results.values():
        assert isinstance(result.notes, tuple)


def test_load_drift_reports_limit(tmp_storage):
    """limit=1 returns only one report."""
    report1, _, _ = make_drift_report()
    report2, _, _ = make_drift_report()

    model_id = "test-model"
    save_drift_report(report1, model_id, timestamp="2026-05-06T10-00-00")
    save_drift_report(report2, model_id, timestamp="2026-05-06T12-00-00")

    loaded = load_drift_reports(model_id, limit=1)
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Test: get_prediction_count
# ---------------------------------------------------------------------------


def test_get_prediction_count_zero(tmp_storage):
    """returns 0 when no data exists."""
    model_id = "nonexistent-model"
    count = get_prediction_count(model_id)
    assert count == 0


def test_get_prediction_count_sums_all(tmp_storage):
    """baseline.n_rows + profile.n_rows totalled correctly."""
    prof1, _, _ = make_profile(n_rows=100, seed=1)
    prof2, _, _ = make_profile(n_rows=50, seed=2)
    prof3, _, _ = make_profile(n_rows=75, seed=3)

    model_id = "test-model"
    save_baseline(prof1, model_id)
    save_profile(prof2, model_id, timestamp="2026-05-06T10-00-00")
    save_profile(prof3, model_id, timestamp="2026-05-06T11-00-00")

    total = get_prediction_count(model_id)
    # baseline (100) + profile1 (50) + profile2 (75) = 225
    assert total == 225


# ---------------------------------------------------------------------------
# Test: Corrupted file handling
# ---------------------------------------------------------------------------


def test_corrupted_profile_skipped(tmp_storage):
    """write 'not json' to profiles/ file; load_profiles returns [] not crash."""
    model_id = "test-model"
    prof, _, _ = make_profile()
    save_profile(prof, model_id, timestamp="2026-05-06T10-00-00")

    # Write a corrupted file
    profiles_dir = tmp_storage / model_id / "profiles"
    corrupted_path = profiles_dir / "2026-05-06T11-00-00.json"
    corrupted_path.write_text("not valid json", encoding="utf-8")

    # load_profiles should skip it and return only the valid one
    loaded = load_profiles(model_id)
    assert len(loaded) == 1
    assert loaded[0] == prof


def test_corrupted_baseline_returns_none(tmp_storage):
    """write 'not json' to baseline.json; load_baseline returns None."""
    model_id = "test-model"
    prof, _, _ = make_profile()
    save_baseline(prof, model_id)

    # Overwrite with corrupted data
    baseline_path = tmp_storage / model_id / "baseline.json"
    baseline_path.write_text("not valid json", encoding="utf-8")

    # load_baseline should return None
    loaded = load_baseline(model_id)
    assert loaded is None


# ---------------------------------------------------------------------------
# Test: Directory creation
# ---------------------------------------------------------------------------


def test_dirs_created_automatically(tmp_storage):
    """save_profile works with no pre-existing dirs."""
    prof, _, _ = make_profile()
    model_id = "brand-new-model"

    # No directories should exist yet
    model_dir = tmp_storage / model_id
    assert not model_dir.exists()

    # save_profile should create them
    path = save_profile(prof, model_id)

    assert path.exists()
    assert (tmp_storage / model_id / "profiles").exists()
    assert (tmp_storage / model_id / "drift_reports").exists()


# ---------------------------------------------------------------------------
# Test: Timestamp handling
# ---------------------------------------------------------------------------


def test_timestamp_string_accepted(tmp_storage):
    """save_profile and save_drift_report accept an ISO string timestamp."""
    prof, _, _ = make_profile()
    report, _, _ = make_drift_report()
    model_id = "test-model"

    # ISO string timestamps
    iso_ts = "2026-05-06T14:30:45"

    path_prof = save_profile(prof, model_id, timestamp=iso_ts)
    assert path_prof.exists()
    assert "2026-05-06T14-30-45.json" in str(path_prof)

    path_report = save_drift_report(report, model_id, timestamp=iso_ts)
    assert path_report.exists()
    assert "2026-05-06T14-30-45.json" in str(path_report)


# ---------------------------------------------------------------------------
# Test: list_models / get_last_updated
# ---------------------------------------------------------------------------


def test_list_models_empty(tmp_path, monkeypatch):
    """list_models returns [] when STORAGE_ROOT does not exist."""
    import modelsentry.storage as storage_module

    # Point at a non-existent dir
    monkeypatch.setattr(storage_module, "STORAGE_ROOT", tmp_path / "absent")
    assert list_models() == []


def test_list_models_returns_directory_names(tmp_storage):
    """list_models returns sorted directory names; ignores files."""
    prof, _, _ = make_profile()
    save_baseline(prof, "model-b")
    save_baseline(prof, "model-a")
    save_baseline(prof, "model-c")

    # Drop a stray file at STORAGE_ROOT — must be ignored
    (tmp_storage / "stray.txt").write_text("not a model dir")

    assert list_models() == ["model-a", "model-b", "model-c"]


def test_get_last_updated_none_when_empty(tmp_storage):
    """get_last_updated returns None when no profile or drift files exist."""
    # Even with a baseline only, get_last_updated should return None
    # (baseline.json is intentionally excluded — it's not proof of ongoing activity)
    prof, _, _ = make_profile()
    save_baseline(prof, "test-model")
    assert get_last_updated("test-model") is None
    assert get_last_updated("nonexistent-model") is None


def test_get_last_updated_picks_newest_across_profiles_and_drift(tmp_storage):
    """get_last_updated picks the newest mtime across both directories."""
    import os
    import time
    from datetime import datetime, timezone

    prof, _, _ = make_profile()
    report, _, _ = make_drift_report()
    model_id = "test-model"

    # Save profile, then drift report — drift is newer
    profile_path = save_profile(prof, model_id, timestamp="2026-05-06T10-00-00")
    time.sleep(0.05)
    drift_path = save_drift_report(report, model_id, timestamp="2026-05-06T11-00-00")

    # Force drift mtime to be definitively later than the profile mtime
    # (in case filesystem mtime resolution masks the time.sleep above).
    profile_mtime = profile_path.stat().st_mtime
    os.utime(drift_path, (profile_mtime + 10, profile_mtime + 10))

    last = get_last_updated(model_id)
    assert last is not None
    assert last.tzinfo is timezone.utc
    # The mtime returned should match the drift file's mtime
    assert abs(last.timestamp() - drift_path.stat().st_mtime) < 0.001
    # And it should be later than the profile file's mtime
    assert last.timestamp() > profile_mtime


# ---------------------------------------------------------------------------
# Test: detected_at timestamp in saved drift reports
# ---------------------------------------------------------------------------


def test_save_drift_report_embeds_detected_at(tmp_storage):
    """save_drift_report embeds a detected_at ISO timestamp in the saved JSON."""
    report, _, _ = make_drift_report()
    path = save_drift_report(report, "test-model", timestamp="2026-05-06T11-00-00")

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "detected_at" in raw
    # Must be a parseable ISO timestamp
    parsed = datetime.fromisoformat(raw["detected_at"])
    assert parsed.tzinfo is not None


def test_load_drift_reports_with_timestamps_returns_datetimes(tmp_storage):
    """load_drift_reports_with_timestamps returns (datetime, DriftReport) tuples."""
    report, _, _ = make_drift_report()
    save_drift_report(report, "test-model", timestamp="2026-05-06T10-00-00")
    save_drift_report(report, "test-model", timestamp="2026-05-06T11-00-00")

    entries = load_drift_reports_with_timestamps("test-model")
    assert len(entries) == 2

    for ts, r in entries:
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None
        assert r.schema_version == report.schema_version
        assert r.overall_severity == report.overall_severity

    # Newest first
    assert entries[0][0] >= entries[1][0]


# ---------------------------------------------------------------------------
# Alert callback
# ---------------------------------------------------------------------------


def test_set_alert_callback_called_after_save(tmp_storage) -> None:
    report, _, _ = make_drift_report()
    calls: list[tuple] = []
    set_alert_callback(lambda r, mid: calls.append((r, mid)))
    save_drift_report(report, "m1")
    assert len(calls) == 1
    assert calls[0][1] == "m1"
    assert calls[0][0].overall_severity == report.overall_severity


def test_set_alert_callback_none_does_not_fire(tmp_storage) -> None:
    report, _, _ = make_drift_report()
    calls: list[tuple] = []
    set_alert_callback(lambda r, mid: calls.append((r, mid)))
    set_alert_callback(None)
    save_drift_report(report, "m1")
    assert calls == []


def test_alert_callback_exception_does_not_block_save(tmp_storage) -> None:
    report, _, _ = make_drift_report()

    def bad_callback(r, mid):
        raise RuntimeError("boom")

    set_alert_callback(bad_callback)
    path = save_drift_report(report, "m1")
    assert path.exists()


def test_alert_callback_isolation(tmp_storage) -> None:
    """set_alert_callback(None) leaves no state for the next test."""
    set_alert_callback(lambda r, mid: None)
    set_alert_callback(None)
    assert storage_module._alert_callback is None
