"""Managed web inspector board helpers."""

from __future__ import annotations

import base64
import json
import logging
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import psutil

from cage.config import WebInspectorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InspectorBoardInfo:
    """Current or newly-started managed inspector board."""

    enabled: bool
    root: Path
    url: str = ""
    host: str = ""
    port: int = 0
    pid: int = 0
    log_path: Path | None = None
    registry_path: Path | None = None
    started: bool = False
    alive: bool = False
    message: str = ""
    auth_token: str = ""


@dataclass(frozen=True)
class InspectorUrlVariant:
    """Copyable URLs for one reachable view of a managed inspector board."""

    label: str
    base_url: str
    run_url: str
    dashboard_url: str


def registry_path(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    return root_path / ".cage" / "inspect.json"


def run_page_url(base_url: str, root: str | Path, run_dir: str | Path) -> str:
    root_path = Path(root).resolve()
    run_path = Path(run_dir).resolve()
    agent_label = run_path.parent.name
    run_id = run_path.name
    # Canonical layout: <benchmark>/.cage_runs/<agent>:<model>:<mode>/<run_id>.
    benchmark = run_path.parent.parent.parent.name
    label_parts = agent_label.split(":")
    model = label_parts[1] if len(label_parts) >= 2 else ""
    try:
        rel = str(run_path.relative_to(root_path))
    except ValueError:
        rel = str(run_path)
    else:
        if (
            run_path.parent.parent.name == ".cage_runs"
            and benchmark
            and model
            and "/" not in agent_label
            and "/" not in run_id
        ):
            readable = (
                f"{quote(benchmark, safe=':@._-')}/"
                f"{quote(model, safe=':@._-')}/"
                f"{quote(run_id, safe=':@._-')}"
            )
            return f"{base_url.rstrip('/')}/{readable}"
    encoded = base64.urlsafe_b64encode(rel.encode()).decode()
    return f"{base_url.rstrip('/')}/run/{encoded}"


def run_dashboard_url(base_url: str, root: str | Path, run_dir: str | Path) -> str:
    return f"{run_page_url(base_url, root, run_dir)}/dashboard"


def run_url_variants(
    board: InspectorBoardInfo,
    root: str | Path,
    run_dir: str | Path,
    *,
    lan_ips: list[str] | tuple[str, ...] | None = None,
) -> list[InspectorUrlVariant]:
    """Return copyable run/dashboard URLs for the board's actual bind address."""

    port = int(getattr(board, "port", 0) or _port_from_url(board.url) or 0)
    if not getattr(board, "enabled", False) or not port:
        return []
    host = (getattr(board, "host", "") or _host_from_url(board.url) or "127.0.0.1").strip()
    token = getattr(board, "auth_token", "") or _token_from_url(board.url)

    addresses: list[tuple[str, str]] = []
    normalized = host.lower()
    if normalized in {"0.0.0.0", "::"}:
        # ``0.0.0.0`` / ``::`` is a bind-only wildcard: a browser can never
        # connect to it, so it must NOT be offered as a clickable URL. Lead with
        # the LAN address(es) — the URL a remote operator actually opens — and
        # keep loopback only as the same-host fallback.
        discovered = list(lan_ips) if lan_ips is not None else discover_lan_ipv4_addresses()
        for ip in discovered:
            if _is_lan_ipv4(ip):
                addresses.append((f"network {ip}", ip))
        addresses.append(("local", "127.0.0.1"))
    elif normalized in {"127.0.0.1", "localhost", "::1"}:
        addresses.append(("local", host))
    else:
        addresses.append((f"host {host}", host))

    variants: list[InspectorUrlVariant] = []
    seen: set[str] = set()
    for label, address in addresses:
        base_url = f"http://{address}:{port}"
        if base_url in seen:
            continue
        seen.add(base_url)
        run_url = _with_token_query(run_page_url(base_url, root, run_dir), token)
        variants.append(
            InspectorUrlVariant(
                label=label,
                base_url=base_url,
                run_url=run_url,
                dashboard_url=_with_token_query(
                    run_dashboard_url(base_url, root, run_dir),
                    token,
                ),
            )
        )
    return variants


def discover_lan_ipv4_addresses() -> list[str]:
    """Return stable non-loopback IPv4 addresses for URL display."""

    addresses: list[str] = []
    for _name, entries in psutil.net_if_addrs().items():
        for entry in entries:
            if getattr(entry, "family", None) != socket.AF_INET:
                continue
            address = str(getattr(entry, "address", "") or "").strip()
            if _is_lan_ipv4(address):
                addresses.append(address)
    return sorted(dict.fromkeys(addresses))


def pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def port_is_free(host: str, port: int) -> bool:
    """True if ``port`` can be bound on ``host`` right now.

    No ``SO_REUSEADDR`` so an active listener (e.g. another project's board on
    the shared ``cage.yml`` port) is correctly reported as busy.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def find_listening_pid(port: int, host: str | None = None) -> int | None:
    """Best-effort PID of the process LISTENing on *port*.

    Used by ``cage inspect status`` and the single-port reuse path. Returns
    ``None`` when nothing listens or the owning PID is not visible to this user.
    The *host* argument is accepted for symmetry but ignored: a LISTEN socket on
    ``0.0.0.0`` and one on ``127.0.0.1`` both answer for the shared port.
    """
    if port <= 0:
        return None
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, psutil.Error, PermissionError):
        connections = []
    for conn in connections:
        laddr = getattr(conn, "laddr", None)
        if (
            laddr
            and getattr(laddr, "port", None) == port
            and conn.status == psutil.CONN_LISTEN
            and conn.pid
        ):
            return int(conn.pid)
    # The global table can hide PIDs owned by other users; scan our own
    # processes so a same-user foreground ``cage inspect`` is still found.
    for proc in psutil.process_iter(["pid"]):
        try:
            for conn in proc.net_connections(kind="inet"):
                laddr = getattr(conn, "laddr", None)
                if (
                    laddr
                    and getattr(laddr, "port", None) == port
                    and conn.status == psutil.CONN_LISTEN
                ):
                    return int(proc.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.Error):
            continue
    return None


def describe_pid(pid: int) -> str:
    """Human-readable command line for *pid*; ``""`` when the process is gone."""
    if pid <= 0:
        return ""
    try:
        proc = psutil.Process(pid)
        return " ".join(proc.cmdline()) or proc.name()
    except psutil.Error:
        return ""


def ensure_inspector_board(
    root: str | Path,
    web_config: WebInspectorConfig,
    *,
    mode: str = "auto",
    interactive: bool = True,
) -> InspectorBoardInfo:
    """Start or reuse a managed inspector board for *root*.

    ``auto`` starts only for interactive runs and when repo config allows
    ``board.auto_start_on_run``. ``on`` starts regardless of TTY. ``off`` never
    starts. Foreground ``cage inspect`` remains separate from this managed path.
    """
    root_path = Path(root).resolve()
    reg = registry_path(root_path)
    log_path = root_path / ".cage" / "inspect.log"
    board_cfg = web_config.board

    if mode == "off" or not board_cfg.enabled:
        return InspectorBoardInfo(
            enabled=False,
            root=root_path,
            log_path=log_path,
            registry_path=reg,
            message="disabled",
        )
    if mode == "auto" and (not interactive or not board_cfg.auto_start_on_run):
        return InspectorBoardInfo(
            enabled=False,
            root=root_path,
            log_path=log_path,
            registry_path=reg,
            message="auto skipped",
        )

    existing = _read_registry(reg)
    if existing is not None:
        pid = int(existing.get("pid") or 0)
        recorded_port = int(existing.get("port") or 0)
        configured_port = int(web_config.port or 0)
        # Reuse a cached managed board only when it still matches the *current*
        # config. A registry left over from before a fixed port was pinned (or
        # from a different configured port) records a stale, once-ephemeral port;
        # blindly reusing it shadows the configured single port forever — e.g.
        # keeps printing :44809 long after config pins :7777. When a fixed port
        # is configured and the cached board sits on a different one, fall through
        # and re-resolve: start_inspector_board then reuses the live listener on
        # the fixed port (single-port policy). port==0 (pure ephemeral mode)
        # keeps the old root+pid reuse behaviour unchanged.
        port_matches = configured_port == 0 or recorded_port == configured_port
        same_root = Path(str(existing.get("root", ""))).resolve() == root_path
        if same_root and port_matches and is_pid_alive(pid):
            return _info_from_registry(existing, reg, started=False, alive=True)

    return start_inspector_board(root_path, web_config, reg=reg, log_path=log_path)


def start_inspector_board(
    root: str | Path,
    web_config: WebInspectorConfig,
    *,
    reg: Path | None = None,
    log_path: Path | None = None,
) -> InspectorBoardInfo:
    root_path = Path(root).resolve()
    reg = reg or registry_path(root_path)
    log_path = log_path or (root_path / ".cage" / "inspect.log")
    host = web_config.host or "127.0.0.1"
    auth_token = ""
    if getattr(web_config.auth, "enabled", False):
        auth_token = str(getattr(web_config.auth, "token", "") or "")
    port = int(web_config.port or 0)
    if port == 0:
        # No fixed port configured — fall back to an ephemeral one.
        port = pick_free_port(host)
    elif not port_is_free(host, port) and (existing_pid := find_listening_pid(port, host)):
        # Single-port policy: every Cage inspector shares the one port from
        # config/cage.yml. A LIVE listener on it means an inspector is already
        # serving this host:port, so reuse it instead of silently relocating to
        # a random port (which produced confusing, ever-changing URLs). No
        # second process is spawned.
        #
        # Note the ``find_listening_pid`` guard: a port can be unbindable
        # *without* a live listener — e.g. a TIME_WAIT socket left by a board we
        # just stopped (``cage inspect stop`` then ``start``). Werkzeug binds
        # with SO_REUSEADDR, so in that case we fall through and start normally
        # rather than "reusing" a dead port and leaving 7777 unserved.
        url = f"http://{host}:{port}"
        logger.info(
            "inspector port %d already in use (pid %s); reusing it for %s "
            "instead of starting a second board",
            port, existing_pid, root_path.name,
        )
        return InspectorBoardInfo(
            enabled=True,
            root=root_path,
            url=url,
            host=host,
            port=port,
            pid=existing_pid,
            log_path=log_path,
            registry_path=reg,
            started=False,
            alive=True,
            auth_token=auth_token,
            message="reused existing inspector on the configured port",
        )
    url = f"http://{host}:{port}"

    reg.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "cage.web.inspect_server",
        str(root_path),
        "--host",
        host,
        "--port",
        str(port),
        "--no-open",
    ]
    if auth_token:
        argv.extend(["--auth-token", auth_token])
    log_file = log_path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv is fully controlled.
            argv,
            cwd=str(Path.cwd()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log_file.close()

    if _process_exited_after_start(proc):
        try:
            reg.unlink()
        except OSError:
            pass
        return InspectorBoardInfo(
            enabled=False,
            root=root_path,
            host=host,
            port=port,
            log_path=log_path,
            registry_path=reg,
            started=False,
            alive=False,
            message=f"inspector board failed to start; see {log_path}",
        )

    data = {
        "pid": int(proc.pid),
        "root": str(root_path),
        "host": host,
        "port": port,
        "url": url,
        "auth_token": auth_token,
        "log_path": str(log_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_registry(reg, data)
    return _info_from_registry(data, reg, started=True, alive=True)


def _process_exited_after_start(proc: Any, *, delay_s: float = 0.2) -> bool:
    poll = getattr(proc, "poll", None)
    if not callable(poll):
        return False
    if poll() is not None:
        return True
    time.sleep(delay_s)
    return poll() is not None


def inspect_board_status(root: str | Path) -> InspectorBoardInfo:
    root_path = Path(root).resolve()
    reg = registry_path(root_path)
    data = _read_registry(reg)
    log_path = root_path / ".cage" / "inspect.log"
    if data is None:
        return InspectorBoardInfo(
            enabled=False,
            root=root_path,
            log_path=log_path,
            registry_path=reg,
            message="no managed board registry",
        )
    pid = int(data.get("pid") or 0)
    return _info_from_registry(
        data,
        reg,
        started=False,
        alive=is_pid_alive(pid),
    )


def stop_inspector_board(root: str | Path) -> bool:
    root_path = Path(root).resolve()
    reg = registry_path(root_path)
    data = _read_registry(reg)
    if data is None:
        return False
    pid = int(data.get("pid") or 0)
    if is_pid_alive(pid):
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        except psutil.Error:
            pass
    try:
        reg.unlink()
    except OSError:
        pass
    return True


def _read_registry(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_registry(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _info_from_registry(
    data: dict[str, Any],
    reg: Path,
    *,
    started: bool,
    alive: bool,
) -> InspectorBoardInfo:
    root = Path(str(data.get("root") or reg.parent.parent)).resolve()
    log_raw = data.get("log_path")
    return InspectorBoardInfo(
        enabled=True,
        root=root,
        url=str(data.get("url") or ""),
        host=str(data.get("host") or ""),
        port=int(data.get("port") or 0),
        pid=int(data.get("pid") or 0),
        log_path=Path(str(log_raw)) if log_raw else None,
        registry_path=reg,
        started=started,
        alive=alive,
        auth_token=str(data.get("auth_token") or _token_from_url(str(data.get("url") or ""))),
    )


def _host_from_url(url: str) -> str:
    if not url:
        return ""
    return urlsplit(url).hostname or ""


def _token_from_url(url: str) -> str:
    if not url:
        return ""
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    return str(query.get("token") or "")


def _port_from_url(url: str) -> int:
    if not url:
        return 0
    try:
        return int(urlsplit(url).port or 0)
    except ValueError:
        return 0


def _with_token_query(base_url: str, token: str) -> str:
    if not token:
        return base_url
    split = urlsplit(base_url)
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    query["token"] = token
    return urlunsplit(
        (
            split.scheme,
            split.netloc,
            split.path,
            urlencode(query),
            split.fragment,
        )
    )


def _is_lan_ipv4(address: str) -> bool:
    if not address:
        return False
    if address.startswith(("127.", "169.254.")):
        return False
    return address != "0.0.0.0"
