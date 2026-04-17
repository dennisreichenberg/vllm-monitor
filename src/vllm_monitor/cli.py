"""CLI entry point for vllm-monitor."""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(
    name="vllm-monitor",
    help="Real-time TUI dashboard for monitoring a vLLM server.",
    add_completion=False,
)


@app.command()
def monitor(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Base URL of the vLLM server."),
    interval: float = typer.Option(2.0, "--interval", "-i", help="Polling interval in seconds.", min=0.5),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="VLLM_API_KEY", help="API key for the vLLM server."),
) -> None:
    """Launch the vLLM health monitor TUI dashboard.

    \b
    Examples:
      vllm-monitor
      vllm-monitor --url http://localhost:8000
      vllm-monitor --url http://my-server:8000 --interval 1
      vllm-monitor --url http://my-server:8000 --api-key mytoken
    """
    from .app import VllmMonitorApp
    from .metrics import MetricsPoller

    poller = MetricsPoller(base_url=url, api_key=api_key, interval=interval)
    dashboard = VllmMonitorApp(poller=poller, interval=interval)
    dashboard.run()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
