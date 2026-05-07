"""CLI entry point for ModelSentry.

Exposes the ``modelsentry`` command group with a ``serve`` subcommand that
starts the local FastAPI dashboard server bound to 127.0.0.1 only.
"""
from __future__ import annotations

import sys
import threading
import webbrowser

import click
import uvicorn

from modelsentry import __version__
from modelsentry.alerts import AlertConfig
from modelsentry.server import HOST, DEFAULT_PORT, create_app


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


@click.group()
def cli() -> None:
    """ModelSentry — early warning for production ML model degradation."""


@cli.command()
@click.option(
    "--model",
    "-m",
    multiple=True,
    required=True,
    help="Model ID to monitor. Repeat to monitor multiple models.",
)
@click.option(
    "--port",
    "-p",
    default=DEFAULT_PORT,
    show_default=True,
    type=int,
    help="Port for the local dashboard server.",
)
@click.option(
    "--alert-email",
    default=None,
    help="Email address to receive drift alerts.",
)
@click.option(
    "--smtp-host",
    default="smtp.gmail.com",
    show_default=True,
    help="SMTP server hostname.",
)
@click.option(
    "--smtp-port",
    default=587,
    show_default=True,
    type=int,
    help="SMTP port (587 for STARTTLS, 465 for SSL).",
)
@click.option(
    "--smtp-user",
    default=None,
    help="SMTP login username. Defaults to --alert-email if omitted.",
)
@click.option(
    "--smtp-password",
    default=None,
    help="SMTP password or app password.",
)
@click.option(
    "--profile-window",
    default=500,
    show_default=True,
    type=int,
    help="Predictions per profile window (informational — set in ms.init()).",
)
def serve(
    model: tuple[str, ...],
    port: int,
    alert_email: str | None,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str | None,
    smtp_password: str | None,
    profile_window: int,
) -> None:
    """Start the local dashboard server at http://127.0.0.1:PORT."""
    url = f"http://{HOST}:{port}"
    model_list = ", ".join(model)
    click.echo(f"\nModelSentry {__version__}\n")
    click.echo(f"  Monitoring:    {model_list}")
    click.echo(f"  Dashboard:     {url}")
    click.echo(f"  Profile window: {profile_window} predictions")

    alert_config: AlertConfig | None = None
    if alert_email:
        alert_config = AlertConfig(
            recipient_email=alert_email,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user or alert_email,
            smtp_password=smtp_password or "",
        )
        click.echo(f"  Alerts →     {alert_email} via {smtp_host}:{smtp_port}")

    click.echo("\n  Press Ctrl+C to stop.\n")
    threading.Timer(1.5, _open_browser, [url]).start()
    app = create_app(alert_config=alert_config)
    try:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    click.echo("\nModelSentry stopped.")
    sys.exit(0)
