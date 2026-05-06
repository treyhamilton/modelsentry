"""Tests for modelsentry.drift."""
from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
import pytest

from modelsentry.drift import (
    DriftReport,
    FeatureDriftResult,
    detect_drift,
)
from modelsentry.profiler import (
    Distribution,
    FeatureProfile,
    PredictionProfile,
    Profile,
)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _empty_pp() -> PredictionProfile:
    return PredictionProfile(
        task_type="regression",
        count=0,
        null_count=0,
        null_rate=0.0,
        numeric_stats=None,
        distribution=None,
        class_counts=None,
    )


def _numeric_fp(
    name: str, edges: Iterable[float], counts: Iterable[int]
) -> FeatureProfile:
    edges_t = tuple(float(e) for e in edges)
    counts_t = tuple(int(c) for c in counts)
    return FeatureProfile(
        name=name,
        dtype="numeric",
        count=sum(counts_t),
        null_count=0,
        null_rate=0.0,
        cardinality=len(counts_t),
        numeric_stats=None,
        distribution=Distribution(bin_edges=edges_t, bin_counts=counts_t),
        value_counts=None,
    )


def _categorical_fp(name: str, value_counts: dict[str, int]) -> FeatureProfile:
    return FeatureProfile(
        name=name,
        dtype="categorical",
        count=sum(value_counts.values()),
        null_count=0,
        null_rate=0.0,
        cardinality=len(value_counts),
        numeric_stats=None,
        distribution=None,
        value_counts=dict(value_counts),
    )


def _empty_numeric_fp(name: str) -> FeatureProfile:
    return FeatureProfile(
        name=name,
        dtype="numeric",
        count=0,
        null_count=10,
        null_rate=1.0,
        cardinality=0,
        numeric_stats=None,
        distribution=None,
        value_counts=None,
    )


def _make_profile(features: dict[str, FeatureProfile]) -> Profile:
    n = max((f.count for f in features.values()), default=0)
    return Profile(
        schema_version="1.0",
        n_rows=n,
        feature_profiles=features,
        prediction_profile=_empty_pp(),
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_profiles_are_stable() -> None:
    edges = tuple(np.linspace(0.0, 10.0, 11))
    p = _make_profile(
        {
            "x": _numeric_fp("x", edges, [100] * 10),
            "g": _categorical_fp("g", {"A": 50, "B": 50}),
        }
    )
    report = detect_drift(p, p)
    assert report.overall_severity == "stable"
    for r in report.feature_results.values():
        assert r.severity == "stable"
        assert r.psi == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Numeric drift
# ---------------------------------------------------------------------------


def test_heavy_numeric_shift_is_critical() -> None:
    edges = tuple(np.linspace(0.0, 10.0, 11))
    baseline = _make_profile({"x": _numeric_fp("x", edges, [1000] * 10)})
    current = _make_profile(
        {"x": _numeric_fp("x", edges, [100, 100, 100, 100, 100, 100, 100, 100, 100, 9100])}
    )
    report = detect_drift(baseline, current)
    r = report.feature_results["x"]
    assert r.severity == "critical"
    assert r.psi >= 0.25
    assert r.ks_p_value is not None and r.ks_p_value < 0.05
    assert report.overall_severity == "critical"


def test_mild_numeric_shift_is_warning() -> None:
    edges = tuple(np.linspace(0.0, 10.0, 11))
    baseline = _make_profile({"x": _numeric_fp("x", edges, [1000] * 10)})
    current = _make_profile(
        {"x": _numeric_fp("x", edges, [200, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1800])}
    )
    report = detect_drift(baseline, current)
    r = report.feature_results["x"]
    assert 0.1 <= r.psi < 0.25
    assert r.severity == "warning"


def test_ks_driven_warning() -> None:
    """Tiny PSI but large N → KS p < 0.05 escalates to warning."""
    edges = tuple(np.linspace(0.0, 10.0, 11))
    baseline = _make_profile({"x": _numeric_fp("x", edges, [10000] * 10)})
    current = _make_profile(
        {"x": _numeric_fp("x", edges, [9000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 11000])}
    )
    report = detect_drift(baseline, current)
    r = report.feature_results["x"]
    assert r.psi < 0.1
    assert r.ks_p_value is not None and r.ks_p_value < 0.05
    assert r.severity == "warning"


def test_bin_edge_mismatch_records_note_and_skips_ks() -> None:
    edges_a = tuple(np.linspace(0.0, 10.0, 11))
    edges_b = tuple(np.linspace(0.0, 20.0, 11))
    baseline = _make_profile({"x": _numeric_fp("x", edges_a, [100] * 10)})
    current = _make_profile({"x": _numeric_fp("x", edges_b, [100] * 10)})
    report = detect_drift(baseline, current)
    r = report.feature_results["x"]
    assert r.severity == "warning"
    assert r.ks_statistic is None
    assert r.ks_p_value is None
    assert any("bin edges mismatch" in n for n in r.notes)


# ---------------------------------------------------------------------------
# Categorical drift
# ---------------------------------------------------------------------------


def test_identical_categorical_is_stable() -> None:
    p = _make_profile({"g": _categorical_fp("g", {"A": 50, "B": 50})})
    report = detect_drift(p, p)
    assert report.feature_results["g"].severity == "stable"


def test_disjoint_categorical_is_critical() -> None:
    baseline = _make_profile({"g": _categorical_fp("g", {"A": 100, "B": 100})})
    current = _make_profile({"g": _categorical_fp("g", {"C": 100, "D": 100})})
    report = detect_drift(baseline, current)
    r = report.feature_results["g"]
    assert r.severity == "critical"
    assert r.psi >= 0.25
    assert r.ks_statistic is None  # no KS for categorical


def test_new_category_in_current_aligned_via_union() -> None:
    baseline = _make_profile({"g": _categorical_fp("g", {"A": 80, "B": 20})})
    current = _make_profile({"g": _categorical_fp("g", {"A": 60, "B": 20, "C": 20})})
    report = detect_drift(baseline, current)
    r = report.feature_results["g"]
    assert np.isfinite(r.psi)
    assert r.psi > 0.0


# ---------------------------------------------------------------------------
# Schema mismatches
# ---------------------------------------------------------------------------


def test_feature_in_baseline_only() -> None:
    edges = tuple(np.linspace(0.0, 1.0, 11))
    baseline = _make_profile(
        {
            "x": _numeric_fp("x", edges, [10] * 10),
            "y": _numeric_fp("y", edges, [10] * 10),
        }
    )
    current = _make_profile({"x": _numeric_fp("x", edges, [10] * 10)})
    report = detect_drift(baseline, current)
    assert report.missing_in_current == ("y",)
    assert "y" not in report.feature_results


def test_feature_in_current_only() -> None:
    edges = tuple(np.linspace(0.0, 1.0, 11))
    baseline = _make_profile({"x": _numeric_fp("x", edges, [10] * 10)})
    current = _make_profile(
        {
            "x": _numeric_fp("x", edges, [10] * 10),
            "z": _numeric_fp("z", edges, [10] * 10),
        }
    )
    report = detect_drift(baseline, current)
    assert report.missing_in_baseline == ("z",)
    assert "z" not in report.feature_results


def test_dtype_mismatch_is_critical() -> None:
    edges = tuple(np.linspace(0.0, 1.0, 11))
    baseline = _make_profile({"x": _numeric_fp("x", edges, [10] * 10)})
    current = _make_profile({"x": _categorical_fp("x", {"A": 100})})
    report = detect_drift(baseline, current)
    r = report.feature_results["x"]
    assert r.severity == "critical"
    assert any("dtype mismatch" in n for n in r.notes)


# ---------------------------------------------------------------------------
# Missing distributions
# ---------------------------------------------------------------------------


def test_missing_baseline_distribution_is_stable_with_note() -> None:
    edges = tuple(np.linspace(0.0, 1.0, 11))
    baseline = _make_profile({"x": _empty_numeric_fp("x")})
    current = _make_profile({"x": _numeric_fp("x", edges, [10] * 10)})
    report = detect_drift(baseline, current)
    r = report.feature_results["x"]
    assert r.severity == "stable"
    assert any("baseline" in n for n in r.notes)


def test_missing_current_distribution_is_warning_with_note() -> None:
    edges = tuple(np.linspace(0.0, 1.0, 11))
    baseline = _make_profile({"x": _numeric_fp("x", edges, [10] * 10)})
    current = _make_profile({"x": _empty_numeric_fp("x")})
    report = detect_drift(baseline, current)
    r = report.feature_results["x"]
    assert r.severity == "warning"
    assert any("current" in n for n in r.notes)


# ---------------------------------------------------------------------------
# Threshold parameters
# ---------------------------------------------------------------------------


def test_custom_thresholds_reclassify_boundary_cases() -> None:
    edges = tuple(np.linspace(0.0, 10.0, 11))
    baseline = _make_profile({"x": _numeric_fp("x", edges, [1000] * 10)})
    current = _make_profile(
        {"x": _numeric_fp("x", edges, [200, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1800])}
    )
    # PSI ≈ 0.176, which is warning at defaults but critical at psi_critical=0.15.
    default = detect_drift(baseline, current)
    tightened = detect_drift(baseline, current, psi_warning=0.05, psi_critical=0.15)
    assert default.feature_results["x"].severity == "warning"
    assert tightened.feature_results["x"].severity == "critical"


def test_invalid_thresholds_raise() -> None:
    p = _make_profile({})
    with pytest.raises(ValueError, match="psi_warning"):
        detect_drift(p, p, psi_warning=0.3, psi_critical=0.2)
    with pytest.raises(ValueError, match="ks_alpha"):
        detect_drift(p, p, ks_alpha=0.0)


# ---------------------------------------------------------------------------
# Overall severity
# ---------------------------------------------------------------------------


def test_overall_severity_is_max() -> None:
    edges = tuple(np.linspace(0.0, 10.0, 11))
    baseline = _make_profile(
        {
            "stable_x": _numeric_fp("stable_x", edges, [100] * 10),
            "warn_x": _numeric_fp("warn_x", edges, [1000] * 10),
            "crit_x": _numeric_fp("crit_x", edges, [1000] * 10),
        }
    )
    current = _make_profile(
        {
            "stable_x": _numeric_fp("stable_x", edges, [100] * 10),
            "warn_x": _numeric_fp(
                "warn_x", edges, [200, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1800]
            ),
            "crit_x": _numeric_fp(
                "crit_x", edges, [100, 100, 100, 100, 100, 100, 100, 100, 100, 9100]
            ),
        }
    )
    report = detect_drift(baseline, current)
    assert report.feature_results["stable_x"].severity == "stable"
    assert report.feature_results["warn_x"].severity == "warning"
    assert report.feature_results["crit_x"].severity == "critical"
    assert report.overall_severity == "critical"


def test_empty_profiles_are_stable() -> None:
    p = _make_profile({})
    report = detect_drift(p, p)
    assert report.overall_severity == "stable"
    assert report.feature_results == {}
    assert report.missing_in_current == ()
    assert report.missing_in_baseline == ()


# ---------------------------------------------------------------------------
# Privacy invariants
# ---------------------------------------------------------------------------


def test_drift_report_is_frozen() -> None:
    p = _make_profile({})
    report = detect_drift(p, p)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.overall_severity = "critical"  # type: ignore[misc]


def test_drift_report_asdict_only_primitives() -> None:
    edges = tuple(np.linspace(0.0, 10.0, 11))
    baseline = _make_profile(
        {
            "x": _numeric_fp("x", edges, [100] * 10),
            "g": _categorical_fp("g", {"A": 50, "B": 50}),
        }
    )
    current = _make_profile(
        {
            "x": _numeric_fp("x", edges, [50, 100, 100, 100, 100, 100, 100, 100, 100, 150]),
            "g": _categorical_fp("g", {"A": 30, "B": 70}),
        }
    )
    report = detect_drift(baseline, current)
    d = dataclasses.asdict(report)

    def _walk(obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str)
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
        else:
            assert isinstance(obj, (int, float, str, bool, type(None))), (
                f"unexpected type {type(obj)}"
            )

    _walk(d)
