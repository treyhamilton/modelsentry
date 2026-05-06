"""Statistical profiling of features and predictions.

Computes aggregate-only Profile objects from raw input data. Raw values are never
retained in returned objects — only scalar stats, bin edges/counts, and capped
top-N category maps. This module enforces the architectural rule that raw data
never leaves the customer environment.

Note: integer-typed predictions are treated as regression in the POC. Real-world
detection of low-cardinality classification targets stored as ints is deferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"
DEFAULT_N_BINS = 10
DEFAULT_TOP_N_CATEGORIES = 50
OTHER_BUCKET = "__other__"


@dataclass(frozen=True)
class NumericStats:
    """Scalar summary statistics for a numeric column."""

    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float


@dataclass(frozen=True)
class Distribution:
    """Binned histogram. Shared edges across baseline and current enable PSI."""

    bin_edges: tuple[float, ...]
    bin_counts: tuple[int, ...]


@dataclass(frozen=True)
class FeatureProfile:
    """Aggregate profile of a single feature column."""

    name: str
    dtype: Literal["numeric", "categorical"]
    count: int
    null_count: int
    null_rate: float
    cardinality: int
    numeric_stats: NumericStats | None
    distribution: Distribution | None
    value_counts: dict[str, int] | None


@dataclass(frozen=True)
class PredictionProfile:
    """Aggregate profile of model predictions."""

    task_type: Literal["regression", "classification"]
    count: int
    null_count: int
    null_rate: float
    numeric_stats: NumericStats | None
    distribution: Distribution | None
    class_counts: dict[str, int] | None


@dataclass(frozen=True)
class Profile:
    """Complete statistical profile — features + predictions."""

    schema_version: str
    n_rows: int
    feature_profiles: dict[str, FeatureProfile]
    prediction_profile: PredictionProfile


def profile(
    features: pd.DataFrame,
    predictions: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    top_n_categories: int = DEFAULT_TOP_N_CATEGORIES,
    baseline_edges: dict[str, tuple[float, ...]] | None = None,
) -> Profile:
    """Compute a privacy-preserving statistical profile.

    Raw values are not retained in the returned Profile.

    Args:
        features: DataFrame of input features (each column profiled independently).
        predictions: 1-D array of model predictions; numeric → regression,
            object/string/bool → classification.
        n_bins: Histogram resolution for numeric columns and regression preds.
        top_n_categories: Max distinct categorical values retained per column;
            tail aggregated as ``__other__``.
        baseline_edges: Optional pre-existing per-feature bin edges. When provided,
            the resulting Distribution objects reuse those edges so the profile is
            directly comparable to the baseline (PSI requires shared edges).

    Returns:
        Profile object containing only aggregate statistics.

    Raises:
        ValueError: empty DataFrame, mismatched lengths, or n_bins < 1.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if features.empty:
        raise ValueError("features must contain at least one row")
    predictions = np.asarray(predictions)
    if len(predictions) != len(features):
        raise ValueError(
            f"length mismatch: features has {len(features)} rows, "
            f"predictions has {len(predictions)}"
        )

    edges_map = baseline_edges or {}
    feature_profiles: dict[str, FeatureProfile] = {}
    for col in features.columns:
        name = str(col)
        series = features[col]
        if _classify_dtype(series) == "numeric":
            feature_profiles[name] = _profile_numeric_feature(
                name, series, n_bins, edges_map.get(name)
            )
        else:
            feature_profiles[name] = _profile_categorical_feature(
                name, series, top_n_categories
            )

    prediction_profile = _profile_predictions(predictions, n_bins)

    return Profile(
        schema_version=SCHEMA_VERSION,
        n_rows=len(features),
        feature_profiles=feature_profiles,
        prediction_profile=prediction_profile,
    )


def compute_psi(
    expected_counts: np.ndarray,
    actual_counts: np.ndarray,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index between two binned distributions.

    Both arrays must be the same length. Zero-bin proportions are clipped to
    ``epsilon`` after normalization to avoid log(0).

    Args:
        expected_counts: Bin counts from baseline distribution.
        actual_counts: Bin counts from current distribution.
        epsilon: Floor on bin proportions to prevent log(0). Default 1e-4.

    Returns:
        PSI score (0 = identical; >0.1 warning; >0.25 critical by convention).

    Raises:
        ValueError: shape mismatch or non-positive total counts.
    """
    expected = np.asarray(expected_counts, dtype=np.float64)
    actual = np.asarray(actual_counts, dtype=np.float64)
    if expected.shape != actual.shape:
        raise ValueError(
            f"count arrays must match shape, got {expected.shape} vs {actual.shape}"
        )
    e_total = expected.sum()
    a_total = actual.sum()
    if e_total <= 0 or a_total <= 0:
        raise ValueError("count arrays must have positive total")
    e = np.maximum(expected / e_total, epsilon)
    a = np.maximum(actual / a_total, epsilon)
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _classify_dtype(series: pd.Series) -> Literal["numeric", "categorical"]:
    """Numeric iff numpy numeric dtype and not boolean."""
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _classify_predictions(
    predictions: np.ndarray,
) -> Literal["regression", "classification"]:
    """Regression iff numeric (non-bool) dtype; else classification."""
    if predictions.dtype == bool:
        return "classification"
    if predictions.dtype.kind in "fcui":
        return "regression"
    return "classification"


def _compute_numeric_stats(values: np.ndarray) -> NumericStats:
    """Mean/std/min/max + quartiles. Caller ensures ``values`` is non-empty."""
    return NumericStats(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=0)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        p25=float(np.quantile(values, 0.25)),
        p50=float(np.quantile(values, 0.50)),
        p75=float(np.quantile(values, 0.75)),
    )


def _compute_numeric_bins(
    values: np.ndarray,
    n_bins: int,
    edges: tuple[float, ...] | None,
) -> Distribution:
    """Build histogram. Reuses provided edges when supplied, else equal-width."""
    if edges is not None:
        edges_arr = np.asarray(edges, dtype=np.float64)
        clipped = np.clip(values, edges_arr[0], edges_arr[-1])
        counts, _ = np.histogram(clipped, bins=edges_arr)
        return Distribution(
            bin_edges=tuple(float(e) for e in edges_arr),
            bin_counts=tuple(int(c) for c in counts),
        )
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        return Distribution(
            bin_edges=(vmin, vmin + 1e-9),
            bin_counts=(int(values.size),),
        )
    counts, edge_arr = np.histogram(values, bins=n_bins, range=(vmin, vmax))
    return Distribution(
        bin_edges=tuple(float(e) for e in edge_arr),
        bin_counts=tuple(int(c) for c in counts),
    )


def _profile_numeric_feature(
    name: str,
    series: pd.Series,
    n_bins: int,
    edges: tuple[float, ...] | None,
) -> FeatureProfile:
    """Profile a numeric column."""
    null_count = int(series.isna().sum())
    total = len(series)
    cleaned = series.dropna().to_numpy()
    if cleaned.size == 0:
        return FeatureProfile(
            name=name,
            dtype="numeric",
            count=0,
            null_count=null_count,
            null_rate=1.0,
            cardinality=0,
            numeric_stats=None,
            distribution=None,
            value_counts=None,
        )
    return FeatureProfile(
        name=name,
        dtype="numeric",
        count=int(cleaned.size),
        null_count=null_count,
        null_rate=null_count / total,
        cardinality=int(np.unique(cleaned).size),
        numeric_stats=_compute_numeric_stats(cleaned),
        distribution=_compute_numeric_bins(cleaned, n_bins, edges),
        value_counts=None,
    )


def _profile_categorical_feature(
    name: str,
    series: pd.Series,
    top_n: int,
) -> FeatureProfile:
    """Profile a categorical column with capped top-N value counts."""
    null_count = int(series.isna().sum())
    total = len(series)
    cleaned = series.dropna()
    count = int(cleaned.size)
    if count == 0:
        return FeatureProfile(
            name=name,
            dtype="categorical",
            count=0,
            null_count=null_count,
            null_rate=1.0,
            cardinality=0,
            numeric_stats=None,
            distribution=None,
            value_counts=None,
        )
    return FeatureProfile(
        name=name,
        dtype="categorical",
        count=count,
        null_count=null_count,
        null_rate=null_count / total,
        cardinality=int(cleaned.nunique()),
        numeric_stats=None,
        distribution=None,
        value_counts=_truncate_value_counts(cleaned.value_counts(), top_n),
    )


def _truncate_value_counts(counts: pd.Series, top_n: int) -> dict[str, int]:
    """Keep top_n categories; collapse the tail into ``__other__``."""
    head = counts.head(top_n)
    result = {str(k): int(v) for k, v in head.items()}
    if len(counts) > top_n:
        tail_total = int(counts.iloc[top_n:].sum())
        if tail_total > 0:
            result[OTHER_BUCKET] = tail_total
    return result


def _profile_predictions(
    predictions: np.ndarray, n_bins: int
) -> PredictionProfile:
    """Profile prediction array as regression or classification."""
    task = _classify_predictions(predictions)
    total = predictions.size
    if task == "regression":
        mask = ~pd.isna(predictions)
        cleaned = predictions[mask].astype(np.float64)
        null_count = int(total - cleaned.size)
        if cleaned.size == 0:
            return PredictionProfile(
                task_type="regression",
                count=0,
                null_count=null_count,
                null_rate=1.0,
                numeric_stats=None,
                distribution=None,
                class_counts=None,
            )
        return PredictionProfile(
            task_type="regression",
            count=int(cleaned.size),
            null_count=null_count,
            null_rate=null_count / total if total else 0.0,
            numeric_stats=_compute_numeric_stats(cleaned),
            distribution=_compute_numeric_bins(cleaned, n_bins, None),
            class_counts=None,
        )
    series = pd.Series(predictions)
    null_count = int(series.isna().sum())
    cleaned = series.dropna()
    if cleaned.size == 0:
        return PredictionProfile(
            task_type="classification",
            count=0,
            null_count=null_count,
            null_rate=1.0,
            numeric_stats=None,
            distribution=None,
            class_counts=None,
        )
    return PredictionProfile(
        task_type="classification",
        count=int(cleaned.size),
        null_count=null_count,
        null_rate=null_count / total if total else 0.0,
        numeric_stats=None,
        distribution=None,
        class_counts={str(k): int(v) for k, v in cleaned.value_counts().items()},
    )
