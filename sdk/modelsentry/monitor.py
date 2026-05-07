"""Framework-agnostic instrumentation for ML predict functions.

Provides ``init()`` to configure the SDK and ``@monitor()`` to decorate any
predict-like callable. The decorator captures (features, predictions) per call
into a per-model buffer, and once the buffer reaches ``profile_window`` calls,
profile computation runs on a daemon worker thread off the predict critical
path. Monitoring failures never affect the wrapped function's return value.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

import modelsentry.storage as _storage
from modelsentry.profiler import Profile, profile

F = TypeVar("F", bound=Callable[..., Any])
ProfileHandler = Callable[[Profile, str], None]

DEFAULT_PROFILE_WINDOW = 500


@dataclass
class _SdkConfig:
    api_key: str
    model_id: str
    profile_window: int
    profile_handler: ProfileHandler
    logger: logging.Logger
    storage_path: Path | None


@dataclass
class _BufferedCall:
    features: object
    predictions: object


_config: _SdkConfig | None = None
_buffers: dict[str, list[_BufferedCall]] = {}
_latest_profiles: dict[str, Profile] = {}
_buffer_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_uninit_warning_emitted = False


def init(
    *,
    api_key: str = "",
    model_id: str,
    profile_window: int = DEFAULT_PROFILE_WINDOW,
    profile_handler: ProfileHandler | None = None,
    storage_path: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Initialize the ModelSentry SDK.

    Must be called before the first decorated call. Re-calling shuts down any
    existing worker and replaces the configuration.

    Args:
        api_key: Cloud API key. Optional in Phase 1 (local-only). Reserved for
            Phase 2 cloud transmission.
        model_id: Default model identifier for monitored functions.
        profile_window: Buffered calls before a profile is computed. Default 500.
        profile_handler: Callback invoked with (profile, model_id) after each
            profile is computed. Defaults to saving profiles to storage_path and
            auto-saving the baseline on first profile.
        storage_path: Directory for profile storage. Defaults to ~/.modelsentry/.
            Override for testing or custom storage locations.
        logger: Custom logger; defaults to logging.getLogger("modelsentry").
    """
    if profile_window < 1:
        raise ValueError(f"profile_window must be >= 1, got {profile_window}")
    global _config, _executor, _uninit_warning_emitted
    if _executor is not None:
        shutdown()
    log = logger or logging.getLogger("modelsentry")
    resolved_path: Path | None = Path(storage_path) if storage_path is not None else None
    if resolved_path is not None:
        _storage.STORAGE_ROOT = resolved_path
    handler = profile_handler if profile_handler is not None else _default_storage_handler
    _config = _SdkConfig(
        api_key=api_key,
        model_id=model_id,
        profile_window=profile_window,
        profile_handler=handler,
        logger=log,
        storage_path=resolved_path,
    )
    _executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="modelsentry-worker"
    )
    _uninit_warning_emitted = False


def monitor(*, model_id: str | None = None) -> Callable[[F], F]:
    """Decorator that captures (features, predictions) on each call.

    The wrapped function's signature, return value, and exception behavior are
    preserved exactly. Monitoring failures are logged and never propagate.

    Args:
        model_id: Override the default model_id from init().
    """

    def decorator(fn: F) -> F:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                _safe_capture(model_id, args, kwargs, result)
                return result

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            _safe_capture(model_id, args, kwargs, result)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def flush(model_id: str | None = None) -> None:
    """Profile any buffered calls now and wait for results.

    Drains the entire worker queue (including futures submitted by prior
    capture calls) by relying on the single-worker FIFO ordering of the
    executor — a sentinel task waits behind all in-flight work.

    Args:
        model_id: If given, only flush that model. Otherwise flush all.
    """
    if _executor is None:
        return
    with _buffer_lock:
        if model_id is None:
            batches = [(mid, buf) for mid, buf in _buffers.items() if buf]
            _buffers.clear()
        else:
            buf = _buffers.get(model_id, [])
            batches = [(model_id, buf)] if buf else []
            _buffers[model_id] = []
    for mid, b in batches:
        _executor.submit(_compute_and_dispatch, b, mid)
    # Drain: with max_workers=1, this sentinel runs only after every prior task.
    sentinel = _executor.submit(lambda: None)
    sentinel.result(timeout=10.0)


def shutdown(timeout: float = 5.0) -> None:
    """Flush buffers, drain the worker, and reset SDK state.

    Safe to call multiple times. Required at end of tests to avoid leaking
    worker threads across the suite.
    """
    global _executor, _config
    if _executor is None:
        return
    try:
        flush()
    except Exception:
        pass
    _executor.shutdown(wait=True, cancel_futures=False)
    _executor = None
    _config = None
    _buffers.clear()
    _latest_profiles.clear()


def get_latest_profile(model_id: str | None = None) -> Profile | None:
    """Return the most recently computed Profile for a model_id."""
    if _config is None:
        return None
    mid = model_id or _config.model_id
    return _latest_profiles.get(mid)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_capture(
    override_model_id: str | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> None:
    """Run capture under a blanket try/except so monitoring never raises."""
    global _uninit_warning_emitted
    try:
        if _config is None:
            if not _uninit_warning_emitted:
                logging.getLogger("modelsentry").warning(
                    "modelsentry.init() not called; monitoring disabled"
                )
                _uninit_warning_emitted = True
            return
        mid = override_model_id or _config.model_id
        features = _extract_features(args, kwargs)
        if features is None:
            return
        _capture_call(mid, features, result)
    except Exception:
        if _config is not None:
            _config.logger.exception(
                "modelsentry: capture failed (non-fatal)"
            )


def _extract_features(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """First positional arg, else common kwarg names."""
    if args:
        return args[0]
    for name in ("features", "X", "x"):
        if name in kwargs:
            return kwargs[name]
    return None


def _capture_call(model_id: str, features: object, predictions: object) -> None:
    """Append to per-model buffer; submit batch to worker when full."""
    assert _config is not None
    assert _executor is not None
    batch_to_profile: list[_BufferedCall] | None = None
    with _buffer_lock:
        buf = _buffers.setdefault(model_id, [])
        buf.append(_BufferedCall(features, predictions))
        if len(buf) >= _config.profile_window:
            batch_to_profile = buf
            _buffers[model_id] = []
    if batch_to_profile is not None:
        _executor.submit(_compute_and_dispatch, batch_to_profile, model_id)


def _compute_and_dispatch(
    batch: list[_BufferedCall], model_id: str
) -> None:
    """Worker-thread function: coerce inputs, profile, dispatch handler."""
    if _config is None:
        return
    try:
        df = _coerce_features([c.features for c in batch])
        preds = _coerce_predictions([c.predictions for c in batch])
        prof = profile(df, preds)
        _latest_profiles[model_id] = prof
    except Exception:
        _config.logger.exception(
            "modelsentry: profile computation failed for model_id=%s", model_id
        )
        return
    try:
        _config.profile_handler(prof, model_id)
    except Exception:
        _config.logger.exception(
            "modelsentry: profile_handler raised for model_id=%s", model_id
        )


def _coerce_features(features_list: list[object]) -> pd.DataFrame:
    """Concatenate a list of feature batches into one DataFrame."""
    frames: list[pd.DataFrame] = []
    for f in features_list:
        if isinstance(f, pd.DataFrame):
            frames.append(f)
        elif isinstance(f, dict):
            frames.append(pd.DataFrame([f]))
        else:
            arr = np.asarray(f)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                # sklearn convention: 1-D input is one sample × N features
                arr = arr.reshape(1, -1)
            frames.append(pd.DataFrame(arr))
    return pd.concat(frames, ignore_index=True)


def _coerce_predictions(predictions_list: list[object]) -> np.ndarray:
    """Concatenate predictions into a single 1-D array."""
    chunks: list[np.ndarray] = []
    for p in predictions_list:
        arr = np.asarray(p)
        if arr.ndim == 2:
            arr = np.argmax(arr, axis=1)
        elif arr.ndim > 2:
            raise ValueError(f"unexpected prediction shape: {arr.shape}")
        chunks.append(np.atleast_1d(arr).ravel())
    return np.concatenate(chunks)


def _default_storage_handler(prof: Profile, model_id: str) -> None:
    """Default handler: persist profile to ~/.modelsentry/ and auto-save baseline."""
    try:
        _storage.save_profile(prof, model_id)
    except Exception:
        if _config is not None:
            _config.logger.exception(
                "modelsentry: failed to save profile for model_id=%s", model_id
            )
        return
    try:
        if _storage.load_baseline(model_id) is None:
            _storage.save_baseline(prof, model_id)
    except Exception:
        if _config is not None:
            _config.logger.exception(
                "modelsentry: failed to save baseline for model_id=%s", model_id
            )


def _default_profile_handler(prof: Profile, model_id: str) -> None:
    """Legacy handler: one-line INFO log summarizing the profile."""
    if _config is None:
        return
    feature_names = list(prof.feature_profiles)
    _config.logger.info(
        "modelsentry profile: model_id=%s n_rows=%d task=%s features=%s",
        model_id,
        prof.n_rows,
        prof.prediction_profile.task_type,
        feature_names,
    )
