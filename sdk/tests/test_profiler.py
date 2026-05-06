"""Tests for modelsentry.profiler."""
from __future__ import annotations

import copy
import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

from modelsentry.profiler import (
    Distribution,
    FeatureProfile,
    NumericStats,
    PredictionProfile,
    Profile,
    compute_psi,
    profile,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _mixed_frame(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n),
            "income": rng.normal(50000, 15000, size=n),
            "country": rng.choice(["US", "CA", "UK", "DE"], size=n),
            "tier": rng.choice(["free", "pro", "enterprise"], size=n),
        }
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_profile_regression_happy_path() -> None:
    df = _mixed_frame()
    preds = np.random.default_rng(0).normal(size=len(df))
    p = profile(df, preds)
    assert p.schema_version == "1.0"
    assert p.n_rows == 1000
    assert set(p.feature_profiles) == {"age", "income", "country", "tier"}
    assert p.feature_profiles["age"].dtype == "numeric"
    assert p.feature_profiles["country"].dtype == "categorical"
    assert p.prediction_profile.task_type == "regression"
    assert p.prediction_profile.numeric_stats is not None
    assert p.prediction_profile.distribution is not None
    assert p.prediction_profile.class_counts is None
    age = p.feature_profiles["age"]
    assert age.count == 1000
    assert age.null_count == 0
    assert age.numeric_stats is not None
    assert age.distribution is not None
    assert len(age.distribution.bin_counts) == 10
    assert sum(age.distribution.bin_counts) == 1000


def test_profile_classification_happy_path() -> None:
    df = _mixed_frame(n=500)
    preds = np.random.default_rng(1).choice(["yes", "no"], size=500)
    p = profile(df, preds)
    pp = p.prediction_profile
    assert pp.task_type == "classification"
    assert pp.numeric_stats is None
    assert pp.distribution is None
    assert pp.class_counts is not None
    assert sum(pp.class_counts.values()) == 500
    assert set(pp.class_counts) == {"yes", "no"}


# ---------------------------------------------------------------------------
# PSI math
# ---------------------------------------------------------------------------


def test_psi_identical_distributions_is_zero() -> None:
    counts = np.array([100, 200, 300, 400, 500])
    assert compute_psi(counts, counts) == pytest.approx(0.0, abs=1e-9)


def test_psi_disjoint_distributions_is_critical() -> None:
    expected = np.array([1000, 0, 0, 0])
    actual = np.array([0, 0, 0, 1000])
    psi = compute_psi(expected, actual)
    assert np.isfinite(psi)
    assert psi > 0.25


def test_psi_zero_bin_handling_finite() -> None:
    expected = np.array([100, 0, 0])
    actual = np.array([0, 0, 100])
    psi = compute_psi(expected, actual)
    assert np.isfinite(psi)
    assert psi > 0.0


def test_psi_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_psi(np.array([1, 2, 3]), np.array([1, 2]))


def test_psi_zero_total_raises() -> None:
    with pytest.raises(ValueError, match="positive total"):
        compute_psi(np.array([0, 0, 0]), np.array([1, 2, 3]))


def test_psi_epsilon_parameter_changes_score() -> None:
    expected = np.array([500, 500])
    actual = np.array([1000, 0])
    score_default = compute_psi(expected, actual)
    score_loose = compute_psi(expected, actual, epsilon=1e-2)
    assert score_default > score_loose > 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_dataframe_raises() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        profile(pd.DataFrame({"a": []}), np.array([]))


def test_length_mismatch_raises() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="length mismatch"):
        profile(df, np.array([1, 2]))


def test_n_bins_must_be_positive() -> None:
    df = _mixed_frame(n=10)
    preds = np.zeros(10)
    with pytest.raises(ValueError, match="n_bins"):
        profile(df, preds, n_bins=0)


def test_all_null_numeric_column() -> None:
    df = pd.DataFrame({"x": [np.nan] * 5, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    preds = np.zeros(5)
    p = profile(df, preds)
    fp = p.feature_profiles["x"]
    assert fp.count == 0
    assert fp.null_count == 5
    assert fp.null_rate == 1.0
    assert fp.numeric_stats is None
    assert fp.distribution is None
    assert fp.cardinality == 0


def test_constant_numeric_column() -> None:
    df = pd.DataFrame({"x": [7.0] * 100})
    preds = np.zeros(100)
    p = profile(df, preds)
    fp = p.feature_profiles["x"]
    assert fp.numeric_stats is not None
    assert fp.numeric_stats.std == pytest.approx(0.0)
    assert fp.distribution is not None
    assert len(fp.distribution.bin_counts) == 1
    assert fp.distribution.bin_counts[0] == 100
    assert fp.cardinality == 1


def test_categorical_top_n_truncation() -> None:
    df = pd.DataFrame({"id": [f"u{i}" for i in range(100)]})
    preds = np.zeros(100)
    p = profile(df, preds, top_n_categories=10)
    fp = p.feature_profiles["id"]
    assert fp.value_counts is not None
    assert "__other__" in fp.value_counts
    assert sum(fp.value_counts.values()) == fp.count
    assert fp.cardinality == 100
    assert len(fp.value_counts) == 11  # top 10 + __other__


# ---------------------------------------------------------------------------
# Privacy invariants
# ---------------------------------------------------------------------------


def test_no_raw_values_in_pickled_profile() -> None:
    """High-cardinality categorical values must not leak via pickle."""
    sentinels = ["SENTINEL_USER_X8K2", "SENTINEL_USER_Q9M3"]
    high_freq = (
        ["common_a"] * 30
        + ["common_b"] * 25
        + ["common_c"] * 20
        + ["common_d"] * 15
        + ["common_e"] * 8
    )
    df = pd.DataFrame({"username": high_freq + sentinels})
    preds = np.zeros(len(df))
    p = profile(df, preds, top_n_categories=5)
    blob = pickle.dumps(p)
    for s in sentinels:
        assert s.encode() not in blob, f"{s} leaked into pickled profile"
    assert b"common_a" in blob  # frequent values intentionally retained
    cloned = copy.deepcopy(p)
    assert cloned == p


def test_profile_asdict_only_primitives() -> None:
    """asdict output must contain no ndarray, Series, or DataFrame instances."""
    df = _mixed_frame(n=50)
    preds = np.zeros(50)
    p = profile(df, preds)
    d = dataclasses.asdict(p)
    blob = repr(d)
    assert "ndarray" not in blob
    assert "Series" not in blob
    assert "DataFrame" not in blob

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


def test_profile_is_frozen() -> None:
    df = _mixed_frame(n=10)
    preds = np.zeros(10)
    p = profile(df, preds)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.n_rows = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_profile_is_reproducible() -> None:
    df = _mixed_frame(n=100, seed=7)
    preds = np.linspace(0.0, 1.0, 100)
    p1 = profile(df, preds)
    p2 = profile(df, preds)
    assert p1 == p2


# ---------------------------------------------------------------------------
# Baseline edges reuse
# ---------------------------------------------------------------------------


def test_baseline_edges_reuse() -> None:
    df_baseline = pd.DataFrame({"x": np.linspace(0.0, 100.0, 200)})
    df_current = pd.DataFrame({"x": np.linspace(50.0, 150.0, 200)})
    preds = np.zeros(200)

    p_baseline = profile(df_baseline, preds)
    edges = {"x": p_baseline.feature_profiles["x"].distribution.bin_edges}
    p_current = profile(df_current, preds, baseline_edges=edges)

    assert (
        p_current.feature_profiles["x"].distribution.bin_edges
        == p_baseline.feature_profiles["x"].distribution.bin_edges
    )
    counts = p_current.feature_profiles["x"].distribution.bin_counts
    assert sum(counts) == 200
    psi = compute_psi(
        np.array(p_baseline.feature_profiles["x"].distribution.bin_counts),
        np.array(counts),
    )
    assert psi > 0.25  # meaningful drift detected
