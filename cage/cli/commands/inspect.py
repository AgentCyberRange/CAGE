"""Web inspector Cage CLI commands."""

from __future__ import annotations

from pathlib import Path

import click


class InspectGroup(click.Group):
    """Click group that keeps ``cage inspect PATH`` as foreground inspect."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            ctx.meta["inspect_path_args"] = [args[0]]
            return super().parse_args(ctx, args[1:])
        return super().parse_args(ctx, args)


@click.group(
    name="inspect",
    cls=InspectGroup,
    invoke_without_command=True,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--port", default=None, type=int, help="Port to serve on (0 = auto)")
@click.option("--host", default=None, help="Host to bind to")
@click.option(
    "--no-open",
    is_flag=True,
    default=None,
    help="Don't auto-open browser",
)
@click.pass_context
def inspect(
    ctx: click.Context,
    port: int | None,
    host: str | None,
    no_open: bool | None,
) -> None:
    """Launch web inspector to browse experiment runs.

    \b
    Examples:
      cage inspect                                    # scan examples/ (default)
      cage inspect examples/<benchmark>               # scan a specific project
    \b
    Every inspector shares the single port from config/cage.yml. If that port is
    already in use this fails instead of relocating to a random port; stop the
    running inspector first.
    """

    if ctx.invoked_subcommand is not None:
        return
    path_args = list(ctx.meta.get("inspect_path_args") or ctx.args)
    if len(path_args) > 1:
        raise click.UsageError("cage inspect accepts at most one PATH")
    path = path_args[0] if path_args else "examples"
    if not Path(path).exists():
        raise click.BadParameter(f"Path {path!r} does not exist")
    from cage.web.inspect_server import serve_inspector

    serve_inspector(path, port=port, host=host, no_open=no_open)


@inspect.command("start")
@click.argument("path", default="examples", required=False, type=click.Path(exists=True))
def inspect_start(path: str) -> None:
    """Start or reuse a managed inspector board."""

    from cage.config import load_repo_config
    from cage.web.inspect_board import ensure_inspector_board

    repo_config = load_repo_config(Path.cwd())
    try:
        info = ensure_inspector_board(
            Path(path),
            repo_config.web_inspector,
            mode="on",
            interactive=True,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    state = "started" if info.started else "already running"
    click.echo(f"Inspector board {state}: {info.url}")
    click.echo(f"PID: {info.pid}")
    click.echo(f"Root: {info.root}")
    if info.log_path is not None:
        click.echo(f"Log: {info.log_path}")
    click.echo(f"Stop: cage inspect stop {info.root}")


@inspect.command("status")
@click.argument("path", default="examples", required=False, type=click.Path(exists=True))
def inspect_status(path: str) -> None:
    """Show whether the shared inspector port is serving anything.

    Status is keyed on the single port from config/cage.yml, so it sees a
    foreground ``cage inspect``, the managed board, or anything else bound there.
    """

    from cage.config import load_repo_config
    from cage.web.inspect_board import (
        describe_pid,
        find_listening_pid,
        inspect_board_status,
        port_is_free,
    )

    web = load_repo_config(Path.cwd()).web_inspector
    host = web.host or "127.0.0.1"
    port = int(web.port or 0)

    if port and not port_is_free(host, port):
        pid = find_listening_pid(port, host) or 0
        click.echo(f"Inspector running on port {port}: http://{host}:{port}")
        if pid:
            click.echo(f"PID: {pid}")
            description = describe_pid(pid)
            if description:
                click.echo(f"Process: {description}")
        info = inspect_board_status(Path(path))
        if info.alive and info.log_path is not None:
            click.echo(f"Log: {info.log_path}")
        click.echo(f"Stop: cage inspect stop {path}")
    elif port:
        click.echo(f"No inspector running on port {port} (from config/cage.yml).")
    else:
        click.echo("No fixed inspector port configured in config/cage.yml.")


@inspect.command("stop")
@click.argument("path", default="examples", required=False, type=click.Path(exists=True))
def inspect_stop(path: str) -> None:
    """Stop the managed inspector board for PATH."""

    from cage.web.inspect_board import stop_inspector_board

    if stop_inspector_board(Path(path)):
        click.echo("Inspector board stopped")
    else:
        click.echo("Inspector board was not running")
