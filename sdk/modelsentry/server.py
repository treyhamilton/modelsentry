"""FastAPI local dashboard server.

Reads Profile and DriftReport objects from storage.py and serves them via JSON
endpoints to the dashboard frontend. All routes are designed to run on the
customer's machine bound to 127.0.0.1 only — never exposed to the network.

uvicorn (the ASGI server that actually binds the port) is invoked from
modelsentry.cli; this module only defines the FastAPI app.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, field_validator

from modelsentry import storage
from modelsentry.drift import DriftReport

HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
DEFAULT_DASHBOARD_PATH = (
    Path(__file__).resolve().parents[2] / "dashboard" / "index.html"
)


# ---------------------------------------------------------------------------
# Top-level response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ModelsListResponse(BaseModel):
    models: list[str]


class ModelStatusResponse(BaseModel):
    model_id: str
    overall_severity: Literal["stable", "warning", "critical", "unknown"]
    last_updated: str | None
    prediction_count: int
    baseline_exists: bool


# ---------------------------------------------------------------------------
# Profile mirror models
# ---------------------------------------------------------------------------


class NumericStatsModel(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float

    model_config = ConfigDict(from_attributes=True)


class DistributionModel(BaseModel):
    bin_edges: list[float]
    bin_counts: list[int]

    model_config = ConfigDict(from_attributes=True)


class FeatureProfileModel(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical"]
    count: int
    null_count: int
    null_rate: float
    cardinality: int
    numeric_stats: NumericStatsModel | None = None
    distribution: DistributionModel | None = None
    value_counts: dict[str, int] | None = None

    model_config = ConfigDict(from_attributes=True)


class PredictionProfileModel(BaseModel):
    task_type: Literal["regression", "classification"]
    count: int
    null_count: int
    null_rate: float
    numeric_stats: NumericStatsModel | None = None
    distribution: DistributionModel | None = None
    class_counts: dict[str, int] | None = None

    model_config = ConfigDict(from_attributes=True)


class ProfileModel(BaseModel):
    schema_version: str
    n_rows: int
    feature_profiles: dict[str, FeatureProfileModel]
    prediction_profile: PredictionProfileModel

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# DriftReport mirror models
# ---------------------------------------------------------------------------


class FeatureDriftResultModel(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical"]
    severity: Literal["stable", "warning", "critical"]
    psi: float | None
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    notes: list[str]

    model_config = ConfigDict(from_attributes=True)

    @field_validator("psi", mode="before")
    @classmethod
    def _nan_to_none(cls, v: float | None) -> float | None:
        # NaN is not valid JSON. Surface as null so the frontend can display "n/a".
        if v is None:
            return None
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v


class DriftReportModel(BaseModel):
    schema_version: str
    overall_severity: Literal["stable", "warning", "critical"]
    feature_results: dict[str, FeatureDriftResultModel]
    missing_in_current: list[str]
    missing_in_baseline: list[str]
    detected_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# /features view
# ---------------------------------------------------------------------------


class FeatureView(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical"]
    severity: Literal["stable", "warning", "critical", "unknown"]
    psi: float | None
    ks_p_value: float | None
    baseline_distribution: DistributionModel | None
    current_distribution: DistributionModel | None
    baseline_value_counts: dict[str, int] | None
    current_value_counts: dict[str, int] | None


class FeaturesResponse(BaseModel):
    model_id: str
    has_baseline: bool
    has_current: bool
    last_updated: str | None
    features: list[FeatureView]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drift_to_model(ts: datetime, r: DriftReport) -> DriftReportModel:
    return DriftReportModel(
        schema_version=r.schema_version,
        overall_severity=r.overall_severity,
        feature_results={
            k: FeatureDriftResultModel.model_validate(v)
            for k, v in r.feature_results.items()
        },
        missing_in_current=list(r.missing_in_current),
        missing_in_baseline=list(r.missing_in_baseline),
        detected_at=ts.isoformat(),
    )


def _require_known_model(model_id: str) -> None:
    if model_id not in storage.list_models():
        raise HTTPException(status_code=404, detail=f"unknown model_id: {model_id}")


def _last_updated_iso(model_id: str) -> str | None:
    ts = storage.get_last_updated(model_id)
    return ts.isoformat() if ts is not None else None


def _compute_overall_severity(
    model_id: str,
) -> Literal["stable", "warning", "critical", "unknown"]:
    reports = storage.load_drift_reports(model_id, limit=1)
    if not reports:
        return "unknown"
    return reports[0].overall_severity


def _normalize_psi(psi: float) -> float | None:
    return None if not math.isfinite(psi) else psi


def _build_features_view(model_id: str) -> FeaturesResponse:
    baseline = storage.load_baseline(model_id)
    current_list = storage.load_profiles(model_id, limit=1)
    current = current_list[0] if current_list else None
    report_list = storage.load_drift_reports(model_id, limit=1)
    report = report_list[0] if report_list else None

    feature_names: set[str] = set()
    if baseline is not None:
        feature_names |= set(baseline.feature_profiles)
    if current is not None:
        feature_names |= set(current.feature_profiles)

    features: list[FeatureView] = []
    for name in sorted(feature_names):
        b_fp = baseline.feature_profiles.get(name) if baseline else None
        c_fp = current.feature_profiles.get(name) if current else None
        drift_result = report.feature_results.get(name) if report else None

        # dtype: prefer drift_result, fall back to whichever profile carries it.
        if drift_result is not None:
            dtype = drift_result.dtype
        elif c_fp is not None:
            dtype = c_fp.dtype
        elif b_fp is not None:
            dtype = b_fp.dtype
        else:
            continue

        if drift_result is not None:
            severity = drift_result.severity
            psi = _normalize_psi(drift_result.psi)
            ks_p = drift_result.ks_p_value
        else:
            severity = "unknown"
            psi = None
            ks_p = None

        features.append(
            FeatureView(
                name=name,
                dtype=dtype,
                severity=severity,
                psi=psi,
                ks_p_value=ks_p,
                baseline_distribution=DistributionModel.model_validate(b_fp.distribution)
                if b_fp and b_fp.distribution
                else None,
                current_distribution=DistributionModel.model_validate(c_fp.distribution)
                if c_fp and c_fp.distribution
                else None,
                baseline_value_counts=dict(b_fp.value_counts) if b_fp and b_fp.value_counts else None,
                current_value_counts=dict(c_fp.value_counts) if c_fp and c_fp.value_counts else None,
            )
        )

    return FeaturesResponse(
        model_id=model_id,
        has_baseline=baseline is not None,
        has_current=current is not None,
        last_updated=_last_updated_iso(model_id),
        features=features,
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(dashboard_path: Path = DEFAULT_DASHBOARD_PATH) -> FastAPI:
    """Build the FastAPI app for the local dashboard server.

    Args:
        dashboard_path: Path to the static index.html served at GET /.
            Configurable so cli.py and tests can override the default.

    Returns:
        FastAPI app instance. Caller is responsible for binding it via uvicorn
        to ``HOST`` only (never 0.0.0.0).
    """
    app = FastAPI(title="ModelSentry Local Dashboard", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not dashboard_path.exists():
            raise HTTPException(status_code=404, detail="dashboard not found")
        return FileResponse(dashboard_path, media_type="text/html")

    @app.get("/api/models", response_model=ModelsListResponse)
    def list_models() -> ModelsListResponse:
        return ModelsListResponse(models=storage.list_models())

    @app.get(
        "/api/models/{model_id}/status",
        response_model=ModelStatusResponse,
    )
    def model_status(
        model_id: str = PathParam(..., pattern=MODEL_ID_PATTERN),
    ) -> ModelStatusResponse:
        _require_known_model(model_id)
        return ModelStatusResponse(
            model_id=model_id,
            overall_severity=_compute_overall_severity(model_id),
            last_updated=_last_updated_iso(model_id),
            prediction_count=storage.get_prediction_count(model_id),
            baseline_exists=storage.load_baseline(model_id) is not None,
        )

    @app.get(
        "/api/models/{model_id}/profiles",
        response_model=list[ProfileModel],
    )
    def list_profiles(
        model_id: str = PathParam(..., pattern=MODEL_ID_PATTERN),
        limit: int = Query(10, ge=1, le=1000),
    ) -> list[ProfileModel]:
        _require_known_model(model_id)
        return [
            ProfileModel.model_validate(p)
            for p in storage.load_profiles(model_id, limit=limit)
        ]

    @app.get(
        "/api/models/{model_id}/drift",
        response_model=list[DriftReportModel],
    )
    def list_drift(
        model_id: str = PathParam(..., pattern=MODEL_ID_PATTERN),
        limit: int = Query(10, ge=1, le=1000),
    ) -> list[DriftReportModel]:
        _require_known_model(model_id)
        return [
            _drift_to_model(ts, r)
            for ts, r in storage.load_drift_reports_with_timestamps(model_id, limit=limit)
        ]

    @app.get(
        "/api/models/{model_id}/features",
        response_model=FeaturesResponse,
    )
    def features_view(
        model_id: str = PathParam(..., pattern=MODEL_ID_PATTERN),
    ) -> FeaturesResponse:
        _require_known_model(model_id)
        return _build_features_view(model_id)

    return app


app = create_app()
