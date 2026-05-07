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
from modelsentry.server import HOST, DEFAULT_PORT, app


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
def serve(model: tuple[str, ...], port: int) -> None:
    """Start the local dashboard server at http://127.0.0.1:PORT."""
    url = f"http://{HOST}:{port}"
    model_list = ", ".join(model)
    click.echo(f"\nModelSentry {__version__}\n")
    click.echo(f"  Monitoring:  {model_list}")
    click.echo(f"  Dashboard:   {url}")
    click.echo("\n  Press Ctrl+C to stop.\n")
    threading.Timer(1.5, _open_browser, [url]).start()
    try:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    click.echo("\nModelSentry stopped.")
    sys.exit(0)
