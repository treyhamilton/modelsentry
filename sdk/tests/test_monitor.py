"""Tests for modelsentry.monitor."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable

import numpy as np
import pandas as pd
import pytest

import modelsentry as ms
from modelsentry.monitor import (
    flush,
    get_latest_profile,
    shutdown,
)
from modelsentry.profiler import Profile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sdk() -> Callable[..., list]:
    """Factory fixture: returns a function that initializes the SDK and the
    list that collected profiles will accumulate into. Auto-shutdown."""
    captured: list = []

    def _init(
        profile_window: int = 10,
        model_id: str = "test-model",
        handler: Callable[[Profile, str], None] | None = None,
    ) -> list:
        actual_handler = handler or (
            lambda p, mid: captured.append((p, mid))
        )
        ms.init(
            api_key="test-key",
            model_id=model_id,
            profile_window=profile_window,
            profile_handler=actual_handler,
        )
        return captured

    yield _init
    shutdown()


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})


# ---------------------------------------------------------------------------
# Decorator semantics
# ---------------------------------------------------------------------------


def test_decorator_preserves_return_value(sdk, small_df):
    sdk()

    @ms.monitor()
    def predict(X):
        return np.array([10.0, 20.0])

    result = predict(small_df)
    np.testing.assert_array_equal(result, [10.0, 20.0])


def test_decorator_propagates_user_exceptions(sdk, small_df):
    sdk()

    @ms.monitor()
    def predict(X):
        raise RuntimeError("user error")

    with pytest.raises(RuntimeError, match="user error"):
        predict(small_df)


def test_decorator_preserves_function_metadata(sdk):
    sdk()

    @ms.monitor()
    def predict(X):
        """Original docstring."""
        return X

    assert predict.__name__ == "predict"
    assert predict.__doc__ == "Original docstring."


# ---------------------------------------------------------------------------
# Capture and profiling
# ---------------------------------------------------------------------------


def test_profile_after_window_calls(sdk, small_df):
    captured = sdk(profile_window=10)

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    for _ in range(10):
        predict(small_df)
    flush()
    assert len(captured) == 1
    prof, mid = captured[0]
    assert mid == "test-model"
    assert prof.n_rows == 20  # 10 calls × 2 rows each


def test_buffer_drained_between_batches(sdk, small_df):
    captured = sdk(profile_window=10)

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    for _ in range(15):
        predict(small_df)
    flush()
    assert len(captured) == 2
    assert captured[0][0].n_rows == 20
    assert captured[1][0].n_rows == 10


def test_flush_partial_buffer(sdk, small_df):
    captured = sdk(profile_window=100)

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    for _ in range(5):
        predict(small_df)
    flush()
    assert len(captured) == 1
    assert captured[0][0].n_rows == 10


def test_get_latest_profile(sdk, small_df):
    sdk(profile_window=5)

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    assert get_latest_profile() is None
    for _ in range(5):
        predict(small_df)
    flush()
    p = get_latest_profile()
    assert p is not None
    assert p.n_rows == 10


# ---------------------------------------------------------------------------
# Multi-model
# ---------------------------------------------------------------------------


def test_multi_model_buffers_independently(sdk, small_df):
    captured = sdk(profile_window=5)

    @ms.monitor(model_id="model-a")
    def predict_a(X):
        return np.zeros(len(X))

    @ms.monitor(model_id="model-b")
    def predict_b(X):
        return np.ones(len(X))

    for _ in range(5):
        predict_a(small_df)
    for _ in range(5):
        predict_b(small_df)
    flush()
    model_ids = sorted(mid for _, mid in captured)
    assert model_ids == ["model-a", "model-b"]


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def test_dataframe_features(sdk, small_df):
    captured = sdk(profile_window=1)

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    predict(small_df)
    flush()
    p = captured[0][0]
    assert set(p.feature_profiles) == {"a", "b"}


def test_ndarray_features(sdk):
    captured = sdk(profile_window=1)

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    predict(arr)
    flush()
    p = captured[0][0]
    assert len(p.feature_profiles) == 3
    assert p.n_rows == 2


def test_2d_predictions_argmax(sdk):
    captured = sdk(profile_window=1)

    @ms.monitor()
    def predict(X):
        n = len(X)
        probs = np.zeros((n, 3))
        probs[:, 1] = 1.0
        return probs

    df = pd.DataFrame({"a": np.arange(5.0)})
    predict(df)
    flush()
    p = captured[0][0]
    pp = p.prediction_profile
    # argmax axis=1 → all 1s, treated as regression (int dtype) per POC convention.
    assert pp.task_type == "regression"
    assert pp.numeric_stats is not None
    assert pp.numeric_stats.min == 1.0
    assert pp.numeric_stats.max == 1.0


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


def test_profile_handler_failure_isolated(sdk, small_df):
    handler_calls = []

    def boom(p, mid):
        handler_calls.append(1)
        raise RuntimeError("handler boom")

    sdk(profile_window=2, handler=boom)

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    for _ in range(6):
        result = predict(small_df)
        np.testing.assert_array_equal(result, np.zeros(2))
    flush()
    assert len(handler_calls) == 3  # all 3 batches handled despite raises


def test_profile_failure_does_not_crash_predict(sdk, small_df):
    sdk(profile_window=2)

    @ms.monitor()
    def predict(X):
        # 3 predictions for a 2-row input → length mismatch in profile()
        return np.array([1.0, 2.0, 3.0])

    r1 = predict(small_df)
    r2 = predict(small_df)
    flush()
    np.testing.assert_array_equal(r1, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(r2, [1.0, 2.0, 3.0])


def test_decorator_works_without_init(small_df):
    shutdown()  # ensure no prior init

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    result = predict(small_df)
    np.testing.assert_array_equal(result, np.zeros(2))
    assert get_latest_profile() is None


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_async_predict(sdk, small_df):
    captured = sdk(profile_window=3)

    @ms.monitor()
    async def predict(X):
        return np.zeros(len(X))

    async def run():
        for _ in range(3):
            await predict(small_df)

    asyncio.run(run())
    flush()
    assert len(captured) == 1
    assert captured[0][0].n_rows == 6


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------


def test_thread_safety(sdk, small_df):
    captured = sdk(profile_window=100)

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    def worker():
        for _ in range(50):
            predict(small_df)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flush()
    # 200 calls, profile_window=100 → exactly 2 profiles
    assert len(captured) == 2
    total_rows = sum(p.n_rows for p, _ in captured)
    assert total_rows == 400  # 200 calls × 2 rows


# ---------------------------------------------------------------------------
# Overhead (soft timing assertion)
# ---------------------------------------------------------------------------


def test_overhead_below_soft_threshold(sdk, small_df):
    sdk(profile_window=1_000_000)  # don't trigger any profile during measurement

    @ms.monitor()
    def predict(X):
        return np.zeros(len(X))

    raw = predict.__wrapped__

    # Warmup
    for _ in range(100):
        predict(small_df)
        raw(small_df)

    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        predict(small_df)
    instrumented = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        raw(small_df)
    bare = time.perf_counter() - t0

    overhead_us = max(0.0, (instrumented - bare) / n) * 1e6
    print(f"\nmonitor overhead: {overhead_us:.1f}us per call")
    # Spec says <5ms; this is a soft regression guard at 10x.
    assert overhead_us < 50_000.0
