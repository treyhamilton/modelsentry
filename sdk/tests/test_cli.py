"""Tests for modelsentry.cli — Click entry point."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from modelsentry import __version__
from modelsentry.cli import cli
from modelsentry.server import HOST, DEFAULT_PORT, app as server_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Context manager to mock blocking/side-effect dependencies
# ---------------------------------------------------------------------------


@contextmanager
def mock_serve():
    """Patch uvicorn.run, webbrowser.open, and threading.Timer for all serve tests."""
    with patch("uvicorn.run") as mock_uv, \
         patch("webbrowser.open") as mock_wb, \
         patch("threading.Timer") as mock_timer:
        mock_timer_instance = MagicMock()
        mock_timer.return_value = mock_timer_instance
        yield mock_uv, mock_wb, mock_timer


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_serve_requires_model(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["serve"])
    assert result.exit_code == 2
    assert "Missing option" in result.output or "Error" in result.output


def test_serve_default_port(runner: CliRunner) -> None:
    with mock_serve() as (mock_uv, _wb, _timer):
        result = runner.invoke(cli, ["serve", "--model", "churn-v3"])
    assert result.exit_code == 0
    mock_uv.assert_called_once()
    _, kwargs = mock_uv.call_args
    assert kwargs.get("port", mock_uv.call_args[0][2] if len(mock_uv.call_args[0]) > 2 else None) == DEFAULT_PORT or mock_uv.call_args[1].get("port") == DEFAULT_PORT


def test_serve_custom_port(runner: CliRunner) -> None:
    with mock_serve() as (mock_uv, _wb, _timer):
        result = runner.invoke(cli, ["serve", "--model", "churn-v3", "--port", "9000"])
    assert result.exit_code == 0
    mock_uv.assert_called_once()
    assert mock_uv.call_args[1]["port"] == 9000


def test_serve_host_always_localhost(runner: CliRunner) -> None:
    with mock_serve() as (mock_uv, _wb, _timer):
        result = runner.invoke(cli, ["serve", "--model", "churn-v3"])
    assert result.exit_code == 0
    assert mock_uv.call_args[1]["host"] == "127.0.0.1"


def test_serve_no_host_flag(runner: CliRunner) -> None:
    with mock_serve():
        result = runner.invoke(cli, ["serve", "--model", "churn-v3", "--host", "0.0.0.0"])
    assert result.exit_code == 2


def test_serve_startup_message_contains_url(runner: CliRunner) -> None:
    with mock_serve():
        result = runner.invoke(cli, ["serve", "--model", "churn-v3"])
    assert result.exit_code == 0
    assert f"http://{HOST}:{DEFAULT_PORT}" in result.output


def test_serve_startup_message_contains_models(runner: CliRunner) -> None:
    with mock_serve():
        result = runner.invoke(cli, ["serve", "--model", "churn-v3"])
    assert result.exit_code == 0
    assert "churn-v3" in result.output


def test_serve_multiple_models_in_message(runner: CliRunner) -> None:
    with mock_serve():
        result = runner.invoke(cli, ["serve", "--model", "churn-v3", "--model", "fraud-v1"])
    assert result.exit_code == 0
    assert "churn-v3" in result.output
    assert "fraud-v1" in result.output


def test_serve_shutdown_message(runner: CliRunner) -> None:
    with mock_serve() as (mock_uv, _wb, _timer):
        mock_uv.side_effect = KeyboardInterrupt
        result = runner.invoke(cli, ["serve", "--model", "churn-v3"])
    assert result.exit_code == 0
    assert "stopped" in result.output.lower()


def test_serve_opens_browser(runner: CliRunner) -> None:
    with mock_serve() as (_uv, _wb, mock_timer):
        result = runner.invoke(cli, ["serve", "--model", "churn-v3"])
    assert result.exit_code == 0
    mock_timer.assert_called_once()
    delay, func, args = mock_timer.call_args[0]
    assert delay == 1.5
    assert f"http://{HOST}:{DEFAULT_PORT}" in args[0]


def test_serve_app_passed_to_uvicorn(runner: CliRunner) -> None:
    with mock_serve() as (mock_uv, _wb, _timer):
        result = runner.invoke(cli, ["serve", "--model", "churn-v3"])
    assert result.exit_code == 0
    mock_uv.assert_called_once()
    first_positional = mock_uv.call_args[0][0]
    assert first_positional is server_app


def test_help_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0


def test_serve_help_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
