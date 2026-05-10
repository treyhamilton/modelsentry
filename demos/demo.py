"""ModelSentry sales-call demo.

Three modes:
  fast         — interactive keypress-driven demo for live sales calls
  slow         — autonomous ~25-minute schedule for personal QA
  walkthrough  — print annotated integration code, then enter fast mode

Usage:
  python demos/demo.py fast
  python demos/demo.py slow [--minutes N]
  python demos/demo.py walkthrough

Requires demos/.env populated with SMTP_USER and SMTP_PASSWORD (see .env.example).
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import modelsentry as ms
import modelsentry.storage as storage
from modelsentry.alerts import AlertConfig, send_drift_alert
from modelsentry.drift import detect_drift
from modelsentry.profiler import profile

from models import GENERATORS, MODEL_IDS

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning,
    module="scipy")

# ---------------------------------------------------------------------------
# Constants & ANSI colors
# ---------------------------------------------------------------------------

PROFILE_WINDOW = 1000         # 500 left p95 PSI for natural variance close to the 0.10 warning threshold; 1000 keeps stable runs at p95 ≈ 0.04
BATCH_SIZE = 100              # rows synthesized per cycle, fanned out as individual predict() calls; with sim_interval=1.0s a 1000-row profile flushes every ~10s
DEMO_PORT = 8080
RECIPIENT = "getmodelsentry@gmail.com"

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MUTED = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Shared state — read by both main and worker threads
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_model_states: dict[str, str] = {mid: "baseline" for mid in MODEL_IDS}
_stop_event = threading.Event()
_server_proc: subprocess.Popen | None = None
_alert_config: AlertConfig | None = None

_SEVERITY_RANK: dict[str, int] = {"stable": 0, "warning": 1, "critical": 2}
_last_alert_severity: dict[str, str] = {mid: "stable" for mid in MODEL_IDS}

# Demo-owned per-model raw buffers. Drift detection runs against these — NOT
# the SDK's automatic profiles, which can't pass baseline_edges back through
# the public profile_handler interface and therefore always trip the
# bin-edges-mismatch check in detect_drift.
_demo_buffers: dict[str, list[tuple[pd.DataFrame, Any]]] = {mid: [] for mid in MODEL_IDS}
_demo_buffer_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SDK profile_handler — intentionally a no-op for the demo
# ---------------------------------------------------------------------------

def demo_handler(profile_obj, model_id: str) -> None:
    # The SDK computes a Profile after every profile_window calls, but doesn't
    # pass baseline_edges to profile() — so its histograms never align with the
    # baseline's. We swallow the SDK's profile and run our own detection in
    # _compute_and_save_drift, where we control bin alignment.
    return


def _async_send_alert(report, model_id: str) -> None:
    # SMTP on a daemon thread so a slow TLS/login can't stall the simulation.
    if _alert_config is None:
        return
    threading.Thread(
        target=send_drift_alert,
        args=(report, model_id, _alert_config),
        daemon=True,
        name=f"modelsentry-alert-{model_id}",
    ).start()


def _maybe_alert(report, model_id: str) -> None:
    # Fire one email per escalation: stable→warning, stable→critical,
    # warning→critical. Repeats at the same level are suppressed; reset
    # (or drift settling back to stable) clears the cooldown.
    prev = _last_alert_severity[model_id]
    sev = report.overall_severity
    if _SEVERITY_RANK[sev] > _SEVERITY_RANK[prev]:
        _async_send_alert(report, model_id)
    _last_alert_severity[model_id] = sev


def _compute_and_save_drift(
    rows: list[tuple[pd.DataFrame, Any]], model_id: str
) -> None:
    # Profile the demo's raw buffer using the baseline's bin edges so PSI is
    # apples-to-apples. Save the profile + drift report, then evaluate alert.
    df = pd.concat([r for r, _ in rows], ignore_index=True)
    preds = np.asarray([p for _, p in rows])
    baseline = storage.load_baseline(model_id)
    edges = (
        {
            name: fp.distribution.bin_edges
            for name, fp in baseline.feature_profiles.items()
            if fp.distribution is not None
        }
        if baseline is not None
        else None
    )
    current_profile = profile(df, preds, baseline_edges=edges)
    storage.save_profile(current_profile, model_id)
    if baseline is None:
        # Belt-and-braces: pre-seed should have created a baseline already.
        storage.save_baseline(current_profile, model_id)
        return
    report = detect_drift(baseline, current_profile)
    storage.save_drift_report(report, model_id)
    _maybe_alert(report, model_id)


def _reset_demo_state() -> None:
    # Wipe ~/.modelsentry/<model_id>/ so every demo run starts from the same
    # blank state. Without this, a leftover baseline from a previous session
    # makes the first comparison score against unrelated old data, producing
    # false WARNING readings before any drift has been triggered.
    storage_root = Path.home() / ".modelsentry"
    for mid in MODEL_IDS:
        model_dir = storage_root / mid
        if model_dir.exists():
            shutil.rmtree(model_dir)


# ---------------------------------------------------------------------------
# Decorated predict functions — one per model_id
# ---------------------------------------------------------------------------

@ms.monitor(model_id="churn-v3")
def predict_churn(features):
    # Real model would compute predictions here.
    # We piggyback the synthetic predictions through a closure (set per-call below).
    return _last_preds["churn-v3"]


@ms.monitor(model_id="lead-score-v2")
def predict_lead(features):
    return _last_preds["lead-score-v2"]


@ms.monitor(model_id="fraud-detect-v4")
def predict_fraud(features):
    return _last_preds["fraud-detect-v4"]


_last_preds: dict[str, object] = {mid: None for mid in MODEL_IDS}

PREDICTORS: dict[str, Callable] = {
    "churn-v3":        predict_churn,
    "lead-score-v2":   predict_lead,
    "fraud-detect-v4": predict_fraud,
}


def _generate_one_batch(model_id: str) -> None:
    """Generate one batch, feed each row to @ms.monitor() (showcase) and to
    the demo's own buffer (drift detection)."""
    with _state_lock:
        state = _model_states[model_id]
    features, preds = GENERATORS[model_id](state, BATCH_SIZE)
    new_rows: list[tuple[pd.DataFrame, Any]] = []
    for i in range(len(features)):
        row = features.iloc[[i]]
        pred = preds[i]
        _last_preds[model_id] = pred
        PREDICTORS[model_id](row)
        new_rows.append((row, pred))
    flush_rows: list[tuple[pd.DataFrame, Any]] | None = None
    with _demo_buffer_lock:
        _demo_buffers[model_id].extend(new_rows)
        if len(_demo_buffers[model_id]) >= PROFILE_WINDOW:
            flush_rows = _demo_buffers[model_id]
            _demo_buffers[model_id] = []
    if flush_rows is not None:
        _compute_and_save_drift(flush_rows, model_id)


# ---------------------------------------------------------------------------
# Setup: load env, configure alerts, init SDK, spawn server
# ---------------------------------------------------------------------------

def _setup() -> None:
    global _alert_config
    load_dotenv(Path(__file__).parent / ".env")

    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    if not smtp_user or not smtp_pass:
        print(f"{RED}ERROR{RESET}: SMTP_USER and SMTP_PASSWORD must be set in demos/.env")
        print(f"{MUTED}       Copy demos/.env.example → demos/.env and fill in.{RESET}")
        sys.exit(1)

    _alert_config = AlertConfig(
        recipient_email=RECIPIENT,
        smtp_host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(os.environ.get("SMTP_PORT", "587")),
        smtp_user=smtp_user,
        smtp_password=smtp_pass,
    )

    _reset_demo_state()
    _preseed_baselines()

    # Note: NOT calling storage.set_alert_callback. That hook fires on every
    # save_drift_report regardless of cooldown — the demo dispatches alerts
    # itself via _maybe_alert so escalation gating actually works.

    for mid in MODEL_IDS:
        ms.init(
            model_id=mid,
            profile_window=PROFILE_WINDOW,
            profile_handler=demo_handler,
        )


def _preseed_baselines() -> None:
    # Generate one full window of baseline-state data per model and save the
    # resulting Profile as the baseline. Doing this before the simulation
    # starts means the very first sim flush at t≈10s already has a baseline
    # to compare against — no UNKNOWN gap on the dashboard during warmup.
    for mid in MODEL_IDS:
        feats, preds = GENERATORS[mid]("baseline", PROFILE_WINDOW)
        storage.save_baseline(profile(feats, preds), mid)


_DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"

_SERVER_SCRIPT = (
    "import uvicorn, sys; "
    "from modelsentry.server import HOST, create_app; "
    "from pathlib import Path; "
    "app = create_app(dashboard_path=Path(sys.argv[1])); "
    f"uvicorn.run(app, host=HOST, port={DEMO_PORT}, log_level='warning')"
)


def _spawn_server() -> None:
    """Spawn the dashboard server as a subprocess with the correct dashboard path."""
    global _server_proc
    cmd = [sys.executable, "-c", _SERVER_SCRIPT, str(_DASHBOARD_PATH)]
    print(f"{MUTED}Starting dashboard server on port {DEMO_PORT}…{RESET}")
    _server_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for /health (max ~10s)
    url = f"http://127.0.0.1:{DEMO_PORT}/health"
    for _ in range(50):
        try:
            with urllib.request.urlopen(url, timeout=0.5) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    # Browser is opened later by the run_* mode, after baselines + first
    # drift report are on disk — otherwise the dashboard's first paint
    # shows UNKNOWN for ~30s, which reads as broken on a sales call.


def _shutdown() -> None:
    """Graceful teardown — terminate server, shutdown SDK."""
    _stop_event.set()
    if _server_proc is not None:
        try:
            _server_proc.terminate()
            _server_proc.wait(timeout=3)
        except Exception:
            try:
                _server_proc.kill()
            except Exception:
                pass
    try:
        ms.shutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Background simulation loop (used by fast mode)
# ---------------------------------------------------------------------------

def _simulation_loop(interval_s: float = 1.0) -> None:
    while not _stop_event.is_set():
        for mid in MODEL_IDS:
            try:
                _generate_one_batch(mid)
            except Exception as e:
                print(f"{RED}simulation error for {mid}: {e}{RESET}")
        _stop_event.wait(interval_s)


# ---------------------------------------------------------------------------
# Mode 1: fast — interactive
# ---------------------------------------------------------------------------

_FAST_MENU = f"""
{BOLD}ModelSentry Live Demo{RESET}

Dashboard: {CYAN}http://127.0.0.1:{DEMO_PORT}{RESET}
Alerts to: {CYAN}{RECIPIENT}{RESET}

Press a key:
  {BOLD}1{RESET}  WARNING drift   on  churn-v3
  {BOLD}2{RESET}  CRITICAL drift  on  churn-v3
  {BOLD}3{RESET}  WARNING drift   on  lead-score-v2
  {BOLD}4{RESET}  CRITICAL drift  on  lead-score-v2
  {BOLD}5{RESET}  WARNING drift   on  fraud-detect-v4
  {BOLD}6{RESET}  CRITICAL drift  on  fraud-detect-v4
  {BOLD}r{RESET}  reset all models to stable baseline
  {BOLD}q{RESET}  quit
"""

_KEYPRESS_MAP: dict[str, tuple[str, str]] = {
    "1": ("churn-v3",        "warning"),
    "2": ("churn-v3",        "critical"),
    "3": ("lead-score-v2",   "warning"),
    "4": ("lead-score-v2",   "critical"),
    "5": ("fraud-detect-v4", "warning"),
    "6": ("fraud-detect-v4", "critical"),
}

_STATE_COLORS = {"baseline": GREEN, "warning": YELLOW, "critical": RED}
_STATE_LABELS = {"baseline": "stable", "warning": "WARNING drift", "critical": "CRITICAL drift"}


def _set_state(model_id: str, new_state: str) -> None:
    with _state_lock:
        _model_states[model_id] = new_state
    color = _STATE_COLORS[new_state]
    label = _STATE_LABELS[new_state]
    eta = "back to nominal" if new_state == "baseline" else "alert in ~10s"
    print(f"  → {model_id}: {color}{label}{RESET}  ({MUTED}{eta}{RESET})")


def _reset_all_to_baseline() -> None:
    # Clear demo buffers AND alert cooldown so the next profile is computed
    # purely from fresh baseline rows and a future drift can re-fire alerts.
    print(f"\n{MUTED}Resetting all models to stable…{RESET}")
    with _demo_buffer_lock:
        for mid in MODEL_IDS:
            _demo_buffers[mid] = []
    for mid in MODEL_IDS:
        _last_alert_severity[mid] = "stable"
        _set_state(mid, "baseline")


def run_fast() -> None:
    print(_FAST_MENU)
    print(f"{MUTED}Seeding baselines (warming up ~10s)…{RESET}")

    sim_thread = threading.Thread(target=_simulation_loop, daemon=True)
    sim_thread.start()

    # Pre-seed baselines were saved synchronously in _setup. We still wait
    # for the first sim flush at ~10s (PROFILE_WINDOW=1000 / BATCH_SIZE=100 per
    # cycle / sim_interval=1.0s) so the dashboard has at least one drift report
    # (STABLE green) before the user can press a key.
    time.sleep(12)
    print(f"{GREEN}✓ Baselines established. Demo is live.{RESET}\n")
    webbrowser.open(f"http://127.0.0.1:{DEMO_PORT}")

    while not _stop_event.is_set():
        try:
            ch = click.getchar(echo=False)
        except (KeyboardInterrupt, EOFError):
            break

        if ch == "q":
            break
        elif ch == "r":
            _reset_all_to_baseline()
        elif ch in _KEYPRESS_MAP:
            mid, new_state = _KEYPRESS_MAP[ch]
            _set_state(mid, new_state)
        else:
            print(f"  {MUTED}(unknown key '{ch}' — try 1–6, r, q){RESET}")

    print(f"\n{MUTED}Stopping demo…{RESET}")


# ---------------------------------------------------------------------------
# Mode 2: slow — autonomous schedule
# ---------------------------------------------------------------------------

def run_slow(total_minutes: int) -> None:
    """Autonomous schedule. Default 25 min, scales by --minutes."""
    print(f"{BOLD}ModelSentry Slow Demo{RESET} — {total_minutes} minutes total\n")
    print(f"Dashboard: {CYAN}http://127.0.0.1:{DEMO_PORT}{RESET}")
    print(f"Alerts to: {CYAN}{RECIPIENT}{RESET}\n")

    # 5 phases evenly spaced
    phase_seconds = (total_minutes * 60) / 5

    schedule = [
        ("baseline established",                       {}),
        ("WARNING drift on churn-v3",                  {"churn-v3":        "warning"}),
        ("CRITICAL drift on churn-v3",                 {"churn-v3":        "critical"}),
        ("CRITICAL drift on lead-score-v2",            {"lead-score-v2":   "critical"}),
        ("CRITICAL drift on fraud-detect-v4 + reset",  {"fraud-detect-v4": "critical", **{mid: "baseline" for mid in MODEL_IDS if mid != "fraud-detect-v4"}}),
    ]

    sim_thread = threading.Thread(target=lambda: _simulation_loop(interval_s=3.0), daemon=True)
    sim_thread.start()

    start = time.time()
    for phase_idx, (description, transitions) in enumerate(schedule):
        if _stop_event.is_set():
            break
        elapsed = int(time.time() - start)
        ts = f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"
        for mid, new_state in transitions.items():
            with _state_lock:
                _model_states[mid] = new_state
        color = RED if "drift" in description and "reset" not in description else GREEN
        print(f"{ts} {color}{description}{RESET}")
        if phase_idx < len(schedule) - 1:
            _stop_event.wait(phase_seconds)
            if phase_idx == 0:
                # First stable phase done — drift reports are on disk, dashboard
                # will paint GREEN STABLE on first load.
                webbrowser.open(f"http://127.0.0.1:{DEMO_PORT}")

    # Hold the final phase for the remainder
    _stop_event.wait(phase_seconds)
    print(f"\n{GREEN}✓ Slow demo complete.{RESET}")


# ---------------------------------------------------------------------------
# Mode 3: walkthrough — print integration code, then run fast mode
# ---------------------------------------------------------------------------

_WALKTHROUGH = f"""
{BOLD}{CYAN}╭─ ModelSentry — 4-line integration ──────────────────────────╮{RESET}

{MUTED}# 1. Install (30 seconds){RESET}
{GREEN}$ pip install modelsentry{RESET}

{MUTED}# 2. In your model code: configure once at startup{RESET}
{CYAN}import modelsentry as ms{RESET}
{CYAN}ms.init(model_id="churn-v3"){RESET}

{MUTED}# 3. Decorate your existing predict function — that's it{RESET}
{CYAN}@ms.monitor(){RESET}
{CYAN}def predict(features_df):{RESET}
{CYAN}    return model.predict(features_df){RESET}

{MUTED}# 4. In another terminal: open the live dashboard + email alerts{RESET}
{GREEN}$ modelsentry serve --model churn-v3 --alert-email you@company.com{RESET}

{BOLD}{CYAN}╰──────────────────────────────────────────────────────────────╯{RESET}

  {MUTED}• Profiles save automatically to ~/.modelsentry/{RESET}
  {MUTED}• Baseline auto-detected from the first profile{RESET}
  {MUTED}• Raw feature values never leave your machine{RESET}
  {MUTED}• Email alert fires the moment drift crosses threshold{RESET}
"""


def run_walkthrough() -> None:
    print(_WALKTHROUGH)
    input(f"\n  {BOLD}Press Enter to launch the live demo →{RESET} ")
    print()
    run_fast()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _signal_handler(_signum, _frame) -> None:
    _stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="modelsentry-demo",
        description="ModelSentry live demo — fast / slow / walkthrough modes.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("fast", help="Interactive keypress-driven demo for sales calls")
    slow = sub.add_parser("slow", help="Autonomous schedule for personal QA")
    slow.add_argument("--minutes", type=int, default=25, help="Total runtime (default: 25)")
    sub.add_parser("walkthrough", help="Print integration code, then fast mode")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    _setup()
    _spawn_server()

    try:
        if args.mode == "fast":
            run_fast()
        elif args.mode == "slow":
            run_slow(args.minutes)
        elif args.mode == "walkthrough":
            run_walkthrough()
    finally:
        _shutdown()
        print(f"{GREEN}✓ Demo stopped cleanly.{RESET}")


if __name__ == "__main__":
    main()
