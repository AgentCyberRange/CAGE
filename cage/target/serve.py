"""Internal target_server runner used by Cage runtime processes."""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path


def _exit_when_orphaned(
    parent_pid: int, *, interval: float = 3.0, grace: float = 10.0
) -> None:
    """Self-terminate once cage (``parent_pid``) is no longer our parent.

    The embedded target_server is launched with ``start_new_session=True`` so a
    terminal Ctrl+C can't kill it before cage runs its controlled teardown. The
    flip side: an *ungraceful* parent death (SIGKILL, OOM, ``kill -9``) leaves
    this server orphaned — a never-self-exiting service holding a port — and
    that is exactly how ``cage targets-check`` accumulated piles of leftover
    processes.

    This watchdog closes the gap portably: no Linux-only ``PR_SET_PDEATHSIG``
    (which fires on the *spawning thread's* death and is a latent footgun) and
    no signal that SIGKILL/OOM could outrun. ``parent_pid`` is passed in by the
    spawner (cage's own pid) rather than read here, so a cage death *during* our
    multi-second startup is still caught: the first tick simply sees
    ``getppid()`` no longer equal to ``parent_pid`` (we've been reparented to
    init/a subreaper). On detection, ask the server to shut down gracefully,
    then hard-exit if it lingers.
    """
    import signal
    import time

    while True:
        time.sleep(interval)
        if os.getppid() == parent_pid:
            continue
        # Reparented → the cage runtime that owned us is gone. SIGTERM ourselves
        # so uvicorn runs its graceful shutdown (lifespan teardown), then force
        # exit if that hangs.
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except OSError:
            pass
        time.sleep(grace)
        os._exit(0)


def run_target_server(
    *,
    host: str,
    port: int,
    benchmark_root: str = "",
    namespace: str = "default",
    adapters: Sequence[str] = (),
    external_token: str = "",
    parent_pid: int = 0,
) -> None:
    """Run the target_server FastAPI app after applying process-local config.

    When ``parent_pid`` is set (embedded runs spawned by cage pass their own
    pid), a daemon watchdog terminates this process if that parent dies, so an
    ungraceful cage exit can't leave the server orphaned. Standalone ``cage
    serve`` leaves it 0 and runs until explicitly stopped.
    """

    try:
        import uvicorn
    except ImportError as exc:
        print(f"Error: {exc}")
        print("Install with: pip install -e .")
        raise SystemExit(1) from exc

    if namespace:
        os.environ["TARGET_SERVER_NAMESPACE"] = namespace
    if external_token:
        os.environ["TARGET_SERVER_EXTERNAL_TOKEN"] = external_token
    if benchmark_root:
        root = Path(benchmark_root).expanduser().resolve()
        if not root.is_dir():
            print(f"Error: benchmark root not found: {root}")
            raise SystemExit(1)
        os.environ["TARGET_SERVER_BENCHMARK_SOURCES_JSON"] = json.dumps(
            [{"adapter_kind": "challenge_json", "root": str(root)}],
        )
    if adapters:
        resolved: list[str] = []
        for spec in adapters:
            path_str, _, class_name = spec.partition(":")
            path = Path(path_str).expanduser().resolve()
            if not path.is_file():
                print(f"Error: adapter module not found: {path}")
                raise SystemExit(1)
            resolved.append(f"{path}:{class_name}" if class_name else str(path))
        os.environ["TARGET_SERVER_ADAPTER_MODULES"] = ",".join(resolved)

    print(f"cage target_server: http://{host}:{port}")
    if benchmark_root:
        print(f"benchmark_root: {benchmark_root}")
    print(f"namespace: {namespace}")
    if adapters:
        print(f"extra adapters: {len(adapters)}")
        for spec in adapters:
            print(f"  - {spec}")
    if external_token:
        print("external audience: ENABLED (bearer token required for non-loopback)")
    else:
        print("external audience: disabled (all callers treated as internal)")
    if parent_pid > 1:
        print(f"parent watchdog: exit if cage pid {parent_pid} dies")
        threading.Thread(
            target=_exit_when_orphaned,
            args=(parent_pid,),
            name="cage-parent-watchdog",
            daemon=True,
        ).start()
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "cage.target.server.challenge_server:app",
        host=host,
        port=port,
        log_level="info",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the internal argparse surface for target_server subprocesses."""

    parser = argparse.ArgumentParser(
        prog="python -m cage.target.serve",
        description="Run Cage's target_server process.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--benchmark-root", default="")
    parser.add_argument("--namespace", default="default")
    parser.add_argument(
        "--adapter",
        dest="adapters",
        action="append",
        default=[],
        help=(
            "Load an extra benchmark adapter. Format: "
            "'path/to/module.py:ClassName'. Repeatable."
        ),
    )
    parser.add_argument("--external-token", default="")
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=0,
        help=(
            "Exit automatically if this pid (the spawning cage process) dies. "
            "Used by cage for embedded per-run servers so an ungraceful cage "
            "exit leaves no orphaned target_server. 0 disables the watchdog."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse internal target_server arguments and run the server."""

    args = build_arg_parser().parse_args(argv)
    run_target_server(
        host=args.host,
        port=args.port,
        benchmark_root=args.benchmark_root,
        namespace=args.namespace,
        adapters=tuple(args.adapters),
        external_token=args.external_token,
        parent_pid=args.parent_pid,
    )


if __name__ == "__main__":
    main()
