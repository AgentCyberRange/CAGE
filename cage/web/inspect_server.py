"""Internal web inspector server entrypoint used by managed Cage processes."""

from __future__ import annotations

import argparse
import socket
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path

import click


def serve_inspector(
    path: str,
    *,
    port: int | None,
    host: str | None,
    no_open: bool | None,
    auth_token: str = "",
) -> None:
    """Run the foreground web inspector for a resolved artifact root."""

    from cage.config import WebInspectorAuthConfig, load_repo_config
    from cage.web.app import create_app

    repo_config = load_repo_config(Path.cwd())
    web_config = repo_config.web_inspector
    host = host if host is not None else web_config.host
    port = port if port is not None else web_config.port
    open_browser = web_config.open_browser if no_open is None else not no_open
    auth = web_config.auth
    if auth_token:
        auth = WebInspectorAuthConfig(enabled=True, token=auth_token)

    root = Path(path).resolve()
    try:
        app = create_app(root, auth=auth, ui=web_config.ui)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if port == 0:
        # No fixed port configured — fall back to an ephemeral one.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
    else:
        from cage.web.inspect_board import (
            describe_pid,
            find_listening_pid,
        )

        # Single-port policy: never relocate to another port. Fail loudly only
        # when a *live* listener actually holds the port — a port left in
        # TIME_WAIT by a just-stopped inspector reports "not free" but has no
        # listener, and Werkzeug's app.run (SO_REUSEADDR) binds through it. So
        # gate on an actual PID, not on raw port_is_free, or a stop+start would
        # spuriously fail right after every restart.
        holder = find_listening_pid(port, host)
        if holder:
            raise click.ClickException(
                f"Inspector port {port} on {host} is already in use "
                f"(PID {holder}: {describe_pid(holder)}).\n"
                f"All Cage inspectors share the single port set in "
                f"config/cage.yml, so only one runs at a time.\n"
                f"Stop the running inspector (Ctrl+C it, or `cage inspect stop`) "
                f"and retry."
            )

    url = f"http://{host}:{port}"
    click.echo(f"Cage Inspector: {url}")
    click.echo(f"Scanning: {root}")
    click.echo("Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    # Warm the run-summary cache off the request path. The first scan of a
    # NAS-backed .cage_runs tree is expensive; doing it in a daemon thread means
    # browser polls hit a warm cache instead of paying the cold walk inline.
    from cage.web.warmer import start_cache_warmer

    start_cache_warmer(root)

    # threaded=True: the inspector serves concurrent requests. A run page over
    # NAS-backed .cage_runs can take several seconds to render (100+ trials);
    # without threading that single slow request blocks the browser's parallel
    # asset/`/api/runs` poll requests, so the page appears to never load.
    app.run(host=host, port=port, debug=False, threaded=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the internal argparse surface for managed inspector subprocesses."""

    parser = argparse.ArgumentParser(
        prog="python -m cage.web.inspect_server",
        description="Run Cage's web inspector server.",
    )
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--no-open", action="store_true", default=None)
    parser.add_argument("--auth-token", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse internal inspector-server arguments and start the server."""

    args = build_arg_parser().parse_args(argv)
    serve_inspector(
        args.path,
        port=args.port,
        host=args.host,
        no_open=args.no_open,
        auth_token=args.auth_token,
    )


if __name__ == "__main__":
    main()
