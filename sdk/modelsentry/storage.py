"""Local JSON storage for Profile and DriftReport objects.

Persists aggregate-only statistical objects to ~/.modelsentry/{model_id}/.
Raw feature values and raw predictions are never written to disk — only the
summary statistics produced by profiler.py and drift.py.

Directory layout:
    ~/.modelsentry/{model_id}/
        baseline.json
        profiles/
            2026-05-06T14-00-00.json
        drift_reports/
            2026-05-06T15-00-00.json
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelsentry.drift import DriftReport, FeatureDriftResult
from modelsentry.profiler import (
    Distribution,
    FeatureProfile,
    NumericStats,
    PredictionProfile,
    Profile,
)

_log = logging.getLogger(__name__)

STORAGE_ROOT: Path = Path.home() / ".modelsentry"
_PROFILES_DIR = "profiles"
_DRIFT_DIR = "drift_reports"
_BASELINE_FILE = "baseline.json"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def _model_dir(model_id: str) -> Path:
    return STORAGE_ROOT / model_id


def _ensure_dirs(model_id: str) -> None:
    base = _model_dir(model_id)
    (base / _PROFILES_DIR).mkdir(parents=True, exist_ok=True)
    (base / _DRIFT_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Timestamp / filename helpers
# ---------------------------------------------------------------------------


def _to_filename(ts: datetime | str | None) -> str:
    """Return an ISO-8601 timestamp string safe for use as a filename.

    Colons are replaced with hyphens so the result is valid on all filesystems.
    If ``ts`` is None, the current UTC time is used.
    """
    if ts is None:
        ts = datetime.now(timezone.utc)
    if isinstance(ts, datetime):
        iso = ts.isoformat(timespec="seconds")
    else:
        iso = str(ts)
    return iso.replace(":", "-") + ".json"


# ---------------------------------------------------------------------------
# Profile serialization
# ---------------------------------------------------------------------------


def _numeric_stats_to_dict(s: NumericStats) -> dict[str, float]:
    return {
        "mean": s.mean,
        "std": s.std,
        "min": s.min,
        "max": s.max,
        "p25": s.p25,
        "p50": s.p50,
        "p75": s.p75,
    }


def _distribution_to_dict(d: Distribution) -> dict[str, list]:
    return {
        "bin_edges": list(d.bin_edges),
        "bin_counts": list(d.bin_counts),
    }


def _feature_profile_to_dict(fp: FeatureProfile) -> dict[str, Any]:
    return {
        "name": fp.name,
        "dtype": fp.dtype,
        "count": fp.count,
        "null_count": fp.null_count,
        "null_rate": fp.null_rate,
        "cardinality": fp.cardinality,
        "numeric_stats": _numeric_stats_to_dict(fp.numeric_stats) if fp.numeric_stats else None,
        "distribution": _distribution_to_dict(fp.distribution) if fp.distribution else None,
        "value_counts": fp.value_counts,
    }


def _prediction_profile_to_dict(pp: PredictionProfile) -> dict[str, Any]:
    return {
        "task_type": pp.task_type,
        "count": pp.count,
        "null_count": pp.null_count,
        "null_rate": pp.null_rate,
        "numeric_stats": _numeric_stats_to_dict(pp.numeric_stats) if pp.numeric_stats else None,
        "distribution": _distribution_to_dict(pp.distribution) if pp.distribution else None,
        "class_counts": pp.class_counts,
    }


def _profile_to_dict(p: Profile) -> dict[str, Any]:
    return {
        "schema_version": p.schema_version,
        "n_rows": p.n_rows,
        "feature_profiles": {k: _feature_profile_to_dict(v) for k, v in p.feature_profiles.items()},
        "prediction_profile": _prediction_profile_to_dict(p.prediction_profile),
    }


def _dict_to_numeric_stats(d: dict) -> NumericStats:
    return NumericStats(
        mean=d["mean"],
        std=d["std"],
        min=d["min"],
        max=d["max"],
        p25=d["p25"],
        p50=d["p50"],
        p75=d["p75"],
    )


def _dict_to_distribution(d: dict) -> Distribution:
    return Distribution(
        bin_edges=tuple(d["bin_edges"]),
        bin_counts=tuple(d["bin_counts"]),
    )


def _dict_to_feature_profile(d: dict) -> FeatureProfile:
    ns_raw = d.get("numeric_stats")
    dist_raw = d.get("distribution")
    return FeatureProfile(
        name=d["name"],
        dtype=d["dtype"],
        count=d["count"],
        null_count=d["null_count"],
        null_rate=d["null_rate"],
        cardinality=d["cardinality"],
        numeric_stats=_dict_to_numeric_stats(ns_raw) if ns_raw is not None else None,
        distribution=_dict_to_distribution(dist_raw) if dist_raw is not None else None,
        value_counts=d.get("value_counts"),
    )


def _dict_to_prediction_profile(d: dict) -> PredictionProfile:
    ns_raw = d.get("numeric_stats")
    dist_raw = d.get("distribution")
    return PredictionProfile(
        task_type=d["task_type"],
        count=d["count"],
        null_count=d["null_count"],
        null_rate=d["null_rate"],
        numeric_stats=_dict_to_numeric_stats(ns_raw) if ns_raw is not None else None,
        distribution=_dict_to_distribution(dist_raw) if dist_raw is not None else None,
        class_counts=d.get("class_counts"),
    )


def _dict_to_profile(d: dict) -> Profile:
    return Profile(
        schema_version=d["schema_version"],
        n_rows=d["n_rows"],
        feature_profiles={k: _dict_to_feature_profile(v) for k, v in d["feature_profiles"].items()},
        prediction_profile=_dict_to_prediction_profile(d["prediction_profile"]),
    )


# ---------------------------------------------------------------------------
# DriftReport serialization
# ---------------------------------------------------------------------------


def _feature_drift_result_to_dict(r: FeatureDriftResult) -> dict[str, Any]:
    return {
        "name": r.name,
        "dtype": r.dtype,
        "severity": r.severity,
        # float("nan") is not valid JSON — serialize as null, restore on load
        "psi": r.psi if math.isfinite(r.psi) else None,
        "ks_statistic": r.ks_statistic,
        "ks_p_value": r.ks_p_value,
        "notes": list(r.notes),
    }


def _drift_report_to_dict(r: DriftReport) -> dict[str, Any]:
    return {
        "schema_version": r.schema_version,
        "overall_severity": r.overall_severity,
        "feature_results": {
            k: _feature_drift_result_to_dict(v) for k, v in r.feature_results.items()
        },
        "missing_in_current": list(r.missing_in_current),
        "missing_in_baseline": list(r.missing_in_baseline),
    }


def _dict_to_feature_drift_result(d: dict) -> FeatureDriftResult:
    raw_psi = d.get("psi")
    return FeatureDriftResult(
        name=d["name"],
        dtype=d["dtype"],
        severity=d["severity"],
        psi=float("nan") if raw_psi is None else float(raw_psi),
        ks_statistic=d.get("ks_statistic"),
        ks_p_value=d.get("ks_p_value"),
        notes=tuple(d.get("notes", [])),
    )


def _dict_to_drift_report(d: dict) -> DriftReport:
    return DriftReport(
        schema_version=d["schema_version"],
        overall_severity=d["overall_severity"],
        feature_results={
            k: _dict_to_feature_drift_result(v) for k, v in d["feature_results"].items()
        },
        missing_in_current=tuple(d.get("missing_in_current", [])),
        missing_in_baseline=tuple(d.get("missing_in_baseline", [])),
    )


# ---------------------------------------------------------------------------
# Safe file loader
# ---------------------------------------------------------------------------


def _load_json_file(path: Path, label: str) -> dict | None:
    """Read and parse a JSON file; log a warning and return None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        _log.warning("Skipping corrupted %s at %s: %s", label, path, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_profile(
    profile: Profile,
    model_id: str,
    timestamp: datetime | str | None = None,
) -> Path:
    """Persist a Profile object as JSON under ~/.modelsentry/{model_id}/profiles/.

    Args:
        profile: The Profile to persist (aggregate statistics only — no raw data).
        model_id: Identifier for the model being monitored.
        timestamp: Filename timestamp. Defaults to current UTC time.

    Returns:
        Path to the written JSON file.
    """
    _ensure_dirs(model_id)
    path = _model_dir(model_id) / _PROFILES_DIR / _to_filename(timestamp)
    path.write_text(json.dumps(_profile_to_dict(profile), indent=2), encoding="utf-8")
    return path


def load_profiles(
    model_id: str,
    limit: int | None = None,
) -> list[Profile]:
    """Load recent profiles for a model, newest first.

    Corrupted or unreadable files are skipped with a warning.

    Args:
        model_id: Identifier for the model.
        limit: Maximum number of profiles to return. None returns all.

    Returns:
        List of Profile objects sorted newest-first.
    """
    profiles_dir = _model_dir(model_id) / _PROFILES_DIR
    if not profiles_dir.exists():
        return []

    paths = sorted(profiles_dir.glob("*.json"), reverse=True)
    results: list[Profile] = []
    for path in paths:
        if limit is not None and len(results) >= limit:
            break
        raw = _load_json_file(path, "profile")
        if raw is None:
            continue
        try:
            results.append(_dict_to_profile(raw))
        except (KeyError, TypeError, ValueError) as exc:
            _log.warning("Skipping malformed profile at %s: %s", path, exc)
    return results


def save_baseline(profile: Profile, model_id: str) -> Path:
    """Persist a baseline Profile as JSON under ~/.modelsentry/{model_id}/baseline.json.

    Overwrites any existing baseline.

    Args:
        profile: The baseline Profile (aggregate statistics only — no raw data).
        model_id: Identifier for the model being monitored.

    Returns:
        Path to the written JSON file.
    """
    _ensure_dirs(model_id)
    path = _model_dir(model_id) / _BASELINE_FILE
    path.write_text(json.dumps(_profile_to_dict(profile), indent=2), encoding="utf-8")
    return path


def load_baseline(model_id: str) -> Profile | None:
    """Load the baseline Profile for a model.

    Args:
        model_id: Identifier for the model.

    Returns:
        Profile if baseline.json exists and is valid; None otherwise.
    """
    path = _model_dir(model_id) / _BASELINE_FILE
    if not path.exists():
        return None
    raw = _load_json_file(path, "baseline")
    if raw is None:
        return None
    try:
        return _dict_to_profile(raw)
    except (KeyError, TypeError, ValueError) as exc:
        _log.warning("Skipping malformed baseline at %s: %s", path, exc)
        return None


def save_drift_report(
    report: DriftReport,
    model_id: str,
    timestamp: datetime | str | None = None,
) -> Path:
    """Persist a DriftReport as JSON under ~/.modelsentry/{model_id}/drift_reports/.

    Args:
        report: The DriftReport to persist.
        model_id: Identifier for the model being monitored.
        timestamp: Filename timestamp. Defaults to current UTC time.

    Returns:
        Path to the written JSON file.
    """
    _ensure_dirs(model_id)
    path = _model_dir(model_id) / _DRIFT_DIR / _to_filename(timestamp)
    data = _drift_report_to_dict(report)
    data["detected_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_drift_reports(
    model_id: str,
    limit: int | None = None,
) -> list[DriftReport]:
    """Load recent drift reports for a model, newest first.

    Corrupted or unreadable files are skipped with a warning.

    Args:
        model_id: Identifier for the model.
        limit: Maximum number of reports to return. None returns all.

    Returns:
        List of DriftReport objects sorted newest-first.
    """
    drift_dir = _model_dir(model_id) / _DRIFT_DIR
    if not drift_dir.exists():
        return []

    paths = sorted(drift_dir.glob("*.json"), reverse=True)
    results: list[DriftReport] = []
    for path in paths:
        if limit is not None and len(results) >= limit:
            break
        raw = _load_json_file(path, "drift report")
        if raw is None:
            continue
        try:
            results.append(_dict_to_drift_report(raw))
        except (KeyError, TypeError, ValueError) as exc:
            _log.warning("Skipping malformed drift report at %s: %s", path, exc)
    return results


def load_drift_reports_with_timestamps(
    model_id: str,
    limit: int | None = None,
) -> list[tuple[datetime, DriftReport]]:
    """Load recent drift reports paired with their detection timestamps, newest first.

    ``detected_at`` is read from the ``detected_at`` field embedded in the JSON
    at save time. Falls back to file mtime for reports saved before that field
    was introduced.

    Corrupted or unreadable files are skipped with a warning.

    Args:
        model_id: Identifier for the model.
        limit: Maximum number of entries to return. None returns all.

    Returns:
        List of ``(detected_at, DriftReport)`` tuples sorted newest-first.
    """
    drift_dir = _model_dir(model_id) / _DRIFT_DIR
    if not drift_dir.exists():
        return []

    paths = sorted(drift_dir.glob("*.json"), reverse=True)
    results: list[tuple[datetime, DriftReport]] = []
    for path in paths:
        if limit is not None and len(results) >= limit:
            break
        raw = _load_json_file(path, "drift report")
        if raw is None:
            continue
        try:
            report = _dict_to_drift_report(raw)
            detected_at_str: str | None = raw.get("detected_at")
            if detected_at_str:
                detected_at = datetime.fromisoformat(detected_at_str)
            else:
                detected_at = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                )
            results.append((detected_at, report))
        except (KeyError, TypeError, ValueError) as exc:
            _log.warning("Skipping malformed drift report at %s: %s", path, exc)
    return results


def get_prediction_count(model_id: str) -> int:
    """Return the total number of predictions monitored for a model.

    Sums ``n_rows`` from baseline.json (if present) and all files in profiles/.
    Returns 0 if no data exists.

    Note: baseline and rolling profiles are independent storage paths. Callers
    are responsible for not double-counting — if the same prediction window was
    passed to both ``save_baseline`` and ``save_profile``, those rows will be
    counted twice here.

    Args:
        model_id: Identifier for the model.

    Returns:
        Total prediction count across all stored profiles.
    """
    total = 0

    baseline = load_baseline(model_id)
    if baseline is not None:
        total += baseline.n_rows

    profiles_dir = _model_dir(model_id) / _PROFILES_DIR
    if profiles_dir.exists():
        for path in profiles_dir.glob("*.json"):
            raw = _load_json_file(path, "profile")
            if raw is None:
                continue
            try:
                total += int(raw["n_rows"])
            except (KeyError, TypeError, ValueError) as exc:
                _log.warning("Could not read n_rows from %s: %s", path, exc)

    return total


def list_models() -> list[str]:
    """Return all model_ids that have a directory under STORAGE_ROOT.

    Only directories are returned (loose files at the root are ignored).
    Result is sorted alphabetically.

    Returns:
        Sorted list of model_id strings; empty list if STORAGE_ROOT is missing.
    """
    if not STORAGE_ROOT.exists():
        return []
    return sorted(p.name for p in STORAGE_ROOT.iterdir() if p.is_dir())


def get_last_updated(model_id: str) -> datetime | None:
    """Return the most recent mtime across profiles/ and drift_reports/ for a model.

    Used as the dashboard's proof-of-life timestamp — proves monitoring is
    actively running, not just that a baseline was once recorded. ``baseline.json``
    is therefore intentionally excluded.

    Args:
        model_id: Identifier for the model.

    Returns:
        UTC datetime of the most recently modified profile or drift report file,
        or None if no such files exist for this model.
    """
    candidates: list[Path] = []
    profiles_dir = _model_dir(model_id) / _PROFILES_DIR
    drift_dir = _model_dir(model_id) / _DRIFT_DIR
    if profiles_dir.exists():
        candidates.extend(profiles_dir.glob("*.json"))
    if drift_dir.exists():
        candidates.extend(drift_dir.glob("*.json"))
    if not candidates:
        return None
    latest_mtime = max(p.stat().st_mtime for p in candidates)
    return datetime.fromtimestamp(latest_mtime, tz=timezone.utc)
