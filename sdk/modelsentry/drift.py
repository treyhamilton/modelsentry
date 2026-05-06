"""Drift detection between two Profile objects.

Compares baseline and current Profile statistics, returning a DriftReport that
classifies each shared feature as stable / warning / critical based on PSI
thresholds and (for numeric features) a KS test.

This module never sees raw data. All computation operates on the aggregate
profile statistics (bin counts, value counts) emitted by profiler.profile().

KS test note: scipy.stats.ks_2samp requires raw 1-D samples, but Profiles only
carry binned histograms. We reconstruct synthetic samples by emitting ``count``
copies of each bin midpoint. Within-bin variance is collapsed, so the KS
statistic is approximate at one bin width — acceptable for POC scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy import stats

from modelsentry.profiler import (
    Distribution,
    FeatureProfile,
    Profile,
    compute_psi,
)

Severity = Literal["stable", "warning", "critical"]
SCHEMA_VERSION = "1.0"
DEFAULT_PSI_WARNING = 0.1
DEFAULT_PSI_CRITICAL = 0.25
DEFAULT_KS_ALPHA = 0.05
SEVERITY_RANK: dict[Severity, int] = {"stable": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class FeatureDriftResult:
    """Per-feature drift result."""

    name: str
    dtype: Literal["numeric", "categorical"]
    severity: Severity
    psi: float
    ks_statistic: float | None
    ks_p_value: float | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DriftReport:
    """Top-level drift report comparing baseline → current."""

    schema_version: str
    overall_severity: Severity
    feature_results: dict[str, FeatureDriftResult]
    missing_in_current: tuple[str, ...]
    missing_in_baseline: tuple[str, ...]


def detect_drift(
    baseline: Profile,
    current: Profile,
    *,
    psi_warning: float = DEFAULT_PSI_WARNING,
    psi_critical: float = DEFAULT_PSI_CRITICAL,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> DriftReport:
    """Compare two profiles and emit a per-feature + overall drift report.

    Args:
        baseline: Reference Profile (e.g. from training data or a stable window).
        current: Current Profile (e.g. from the latest production window).
            Numeric features should have been profiled with ``baseline_edges``
            taken from the baseline so PSI bin alignment is correct.
        psi_warning: PSI threshold; ``psi < psi_warning`` is stable.
        psi_critical: PSI threshold; ``psi >= psi_critical`` is critical.
        ks_alpha: KS p-value threshold for declaring drift on numeric features.

    Returns:
        DriftReport with overall_severity, per-feature results, and the
        symmetric difference of feature names between the two profiles.

    Raises:
        ValueError: thresholds invalid.
    """
    _validate_thresholds(psi_warning, psi_critical, ks_alpha)

    baseline_keys = set(baseline.feature_profiles)
    current_keys = set(current.feature_profiles)
    shared = baseline_keys & current_keys
    missing_in_current = tuple(sorted(baseline_keys - current_keys))
    missing_in_baseline = tuple(sorted(current_keys - baseline_keys))

    feature_results: dict[str, FeatureDriftResult] = {}
    for name in sorted(shared):
        feature_results[name] = _compare_features(
            baseline.feature_profiles[name],
            current.feature_profiles[name],
            psi_warning,
            psi_critical,
            ks_alpha,
        )

    overall = _max_severity(r.severity for r in feature_results.values())

    return DriftReport(
        schema_version=SCHEMA_VERSION,
        overall_severity=overall,
        feature_results=feature_results,
        missing_in_current=missing_in_current,
        missing_in_baseline=missing_in_baseline,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_thresholds(
    psi_warning: float, psi_critical: float, ks_alpha: float
) -> None:
    """Check threshold parameters are sane."""
    if not (0.0 < psi_warning < psi_critical):
        raise ValueError(
            f"psi_warning ({psi_warning}) must be > 0 and < "
            f"psi_critical ({psi_critical})"
        )
    if not (0.0 < ks_alpha <= 1.0):
        raise ValueError(f"ks_alpha must be in (0, 1], got {ks_alpha}")


def _compare_features(
    baseline_fp: FeatureProfile,
    current_fp: FeatureProfile,
    psi_warning: float,
    psi_critical: float,
    ks_alpha: float,
) -> FeatureDriftResult:
    """Dispatch on dtype; handle dtype mismatch up front."""
    if baseline_fp.dtype != current_fp.dtype:
        return FeatureDriftResult(
            name=baseline_fp.name,
            dtype=baseline_fp.dtype,
            severity="critical",
            psi=float("nan"),
            ks_statistic=None,
            ks_p_value=None,
            notes=(
                f"dtype mismatch: baseline={baseline_fp.dtype} "
                f"current={current_fp.dtype}",
            ),
        )
    if baseline_fp.dtype == "numeric":
        return _compare_numeric(
            baseline_fp, current_fp, psi_warning, psi_critical, ks_alpha
        )
    return _compare_categorical(
        baseline_fp, current_fp, psi_warning, psi_critical
    )


def _compare_numeric(
    b: FeatureProfile,
    c: FeatureProfile,
    psi_warning: float,
    psi_critical: float,
    ks_alpha: float,
) -> FeatureDriftResult:
    """Compare numeric features via PSI + KS."""
    if b.distribution is None:
        return _missing_data_result(b.name, "numeric", "baseline")
    if c.distribution is None:
        return _missing_data_result(b.name, "numeric", "current")
    notes: list[str] = []
    if b.distribution.bin_edges != c.distribution.bin_edges:
        notes.append(
            "bin edges mismatch — current was not built with baseline_edges"
        )
    expected = np.asarray(b.distribution.bin_counts, dtype=np.int64)
    actual = np.asarray(c.distribution.bin_counts, dtype=np.int64)
    if expected.shape != actual.shape:
        return FeatureDriftResult(
            name=b.name,
            dtype="numeric",
            severity="warning",
            psi=float("nan"),
            ks_statistic=None,
            ks_p_value=None,
            notes=(*notes, "bin count length mismatch — KS skipped"),
        )
    psi = compute_psi(expected, actual)
    if notes:
        # Bin edges differ → PSI compares non-comparable bins. Surface the score
        # as informational only and force severity to warning so a misleading
        # number can't drive a stable/critical classification.
        return FeatureDriftResult(
            name=b.name,
            dtype="numeric",
            severity="warning",
            psi=psi,
            ks_statistic=None,
            ks_p_value=None,
            notes=tuple(notes),
        )
    ks_stat, ks_p = _ks_test(b.distribution, c.distribution)
    severity = _classify_severity(
        psi, ks_p, psi_warning, psi_critical, ks_alpha
    )
    return FeatureDriftResult(
        name=b.name,
        dtype="numeric",
        severity=severity,
        psi=psi,
        ks_statistic=ks_stat,
        ks_p_value=ks_p,
        notes=tuple(notes),
    )


def _compare_categorical(
    b: FeatureProfile,
    c: FeatureProfile,
    psi_warning: float,
    psi_critical: float,
) -> FeatureDriftResult:
    """Compare categorical features via PSI on aligned value_counts."""
    if b.value_counts is None:
        return _missing_data_result(b.name, "categorical", "baseline")
    if c.value_counts is None:
        return _missing_data_result(b.name, "categorical", "current")
    expected, actual = _align_categorical_counts(b.value_counts, c.value_counts)
    psi = compute_psi(expected, actual)
    severity = _classify_severity(psi, None, psi_warning, psi_critical, 1.0)
    return FeatureDriftResult(
        name=b.name,
        dtype="categorical",
        severity=severity,
        psi=psi,
        ks_statistic=None,
        ks_p_value=None,
        notes=(),
    )


def _missing_data_result(
    name: str,
    dtype: Literal["numeric", "categorical"],
    side: Literal["baseline", "current"],
) -> FeatureDriftResult:
    """Result for a feature where one side has no observed data."""
    if side == "baseline":
        return FeatureDriftResult(
            name=name,
            dtype=dtype,
            severity="stable",
            psi=0.0,
            ks_statistic=None,
            ks_p_value=None,
            notes=("no data in baseline",),
        )
    return FeatureDriftResult(
        name=name,
        dtype=dtype,
        severity="warning",
        psi=float("nan"),
        ks_statistic=None,
        ks_p_value=None,
        notes=("no data in current",),
    )


def _ks_test(
    baseline: Distribution, current: Distribution
) -> tuple[float | None, float | None]:
    """KS test via reconstructed samples; ``(None, None)`` if either is empty."""
    baseline_samples = _reconstruct_samples(baseline)
    current_samples = _reconstruct_samples(current)
    if baseline_samples.size == 0 or current_samples.size == 0:
        return None, None
    result = stats.ks_2samp(baseline_samples, current_samples)
    return float(result.statistic), float(result.pvalue)


def _reconstruct_samples(distribution: Distribution) -> np.ndarray:
    """Reconstruct synthetic samples from a binned histogram for KS testing.

    Each bin emits ``count`` copies of its midpoint. Within-bin variance is
    discarded; the resulting KS statistic is approximate at one bin width.
    """
    edges = np.asarray(distribution.bin_edges, dtype=np.float64)
    counts = np.asarray(distribution.bin_counts, dtype=np.int64)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    return np.repeat(midpoints, counts)


def _align_categorical_counts(
    baseline: dict[str, int], current: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Align two value_counts dicts into matching count vectors."""
    keys = sorted(set(baseline) | set(current))
    expected = np.array([baseline.get(k, 0) for k in keys], dtype=np.int64)
    actual = np.array([current.get(k, 0) for k in keys], dtype=np.int64)
    return expected, actual


def _classify_severity(
    psi: float,
    ks_p: float | None,
    psi_warning: float,
    psi_critical: float,
    ks_alpha: float,
) -> Severity:
    """Map PSI + optional KS p-value to a 3-level severity."""
    if not np.isfinite(psi):
        return "warning"
    if psi >= psi_critical:
        return "critical"
    if psi >= psi_warning:
        return "warning"
    if ks_p is not None and ks_p < ks_alpha:
        return "warning"
    return "stable"


def _max_severity(severities: Iterable[Severity]) -> Severity:
    """Return the highest-rank severity from an iterable; 'stable' if empty."""
    max_rank = -1
    max_sev: Severity = "stable"
    for s in severities:
        rank = SEVERITY_RANK[s]
        if rank > max_rank:
            max_rank = rank
            max_sev = s
    return max_sev
