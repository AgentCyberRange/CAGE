from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from cage.cli import main
from cage.config import WebInspectorConfig


def test_run_page_url_prefers_readable_agent_and_run_id(tmp_path: Path) -> None:
    from cage.web.inspect_board import run_page_url

    root = tmp_path / "bench"
    run_dir = root / ".cage_runs" / "agent-a:model:stateless" / "run-1"
    run_dir.mkdir(parents=True)

    url = run_page_url("http://127.0.0.1:7777", root, run_dir)

    assert url == "http://127.0.0.1:7777/bench/model/run-1"


def test_run_dashboard_url_extends_readable_run_page_url(tmp_path: Path) -> None:
    from cage.web.inspect_board import run_dashboard_url

    root = tmp_path / "bench"
    run_dir = root / ".cage_runs" / "agent-a:model:stateless" / "run-1"
    run_dir.mkdir(parents=True)

    url = run_dashboard_url("http://127.0.0.1:7777", root, run_dir)

    assert url == "http://127.0.0.1:7777/bench/model/run-1/dashboard"


def test_run_url_variants_expand_wildcard_bind_to_lan_then_loopback(
    tmp_path: Path,
) -> None:
    from cage.web.inspect_board import InspectorBoardInfo, run_url_variants

    root = tmp_path / "bench"
    run_dir = root / ".cage_runs" / "agent-a:model:stateless" / "run-1"
    run_dir.mkdir(parents=True)
    info = InspectorBoardInfo(
        enabled=True,
        root=root,
        url="http://0.0.0.0:7777",
        host="0.0.0.0",
        port=7777,
    )

    variants = run_url_variants(info, root, run_dir, lan_ips=["10.1.2.3"])

    # A wildcard 0.0.0.0 bind must NOT yield a clickable URL (a browser cannot
    # connect to it). The connectable LAN address leads; loopback is the
    # same-host fallback.
    assert [(item.label, item.base_url) for item in variants] == [
        ("network 10.1.2.3", "http://10.1.2.3:7777"),
        ("local", "http://127.0.0.1:7777"),
    ]
    assert [item.run_url for item in variants] == [
        "http://10.1.2.3:7777/bench/model/run-1",
        "http://127.0.0.1:7777/bench/model/run-1",
    ]
    assert [item.dashboard_url for item in variants] == [
        f"{item.run_url}/dashboard" for item in variants
    ]


def test_run_url_variants_keep_loopback_honest_when_board_is_loopback(
    tmp_path: Path,
) -> None:
    from cage.web.inspect_board import InspectorBoardInfo, run_url_variants

    root = tmp_path / "bench"
    run_dir = root / ".cage_runs" / "agent-a:model:stateless" / "run-1"
    run_dir.mkdir(parents=True)
    info = InspectorBoardInfo(
        enabled=True,
        root=root,
        url="http://127.0.0.1:7777",
        host="127.0.0.1",
        port=7777,
    )

    variants = run_url_variants(info, root, run_dir, lan_ips=["10.1.2.3"])

    assert [(item.label, item.base_url) for item in variants] == [
        ("local", "http://127.0.0.1:7777"),
    ]


def test_run_dashboard_url_falls_back_to_encoded_for_noncanonical_paths(
    tmp_path: Path,
) -> None:
    from cage.web.inspect_board import run_dashboard_url, run_page_url

    root = tmp_path / "bench"
    run_dir = root / "custom-run-dir"
    run_dir.mkdir(parents=True)

    page_url = run_page_url("http://127.0.0.1:7777", root, run_dir)
    url = run_dashboard_url("http://127.0.0.1:7777", root, run_dir)

    assert page_url.startswith("http://127.0.0.1:7777/run/")
    assert not page_url.endswith("/dashboard")
    assert url.startswith("http://127.0.0.1:7777/run/")
    assert url.endswith("/dashboard")
    assert url == f"{page_url}/dashboard"
    assert "custom-run-dir" not in url


def test_ensure_inspector_board_reuses_alive_registry(tmp_path: Path, monkeypatch) -> None:
    from cage.web.inspect_board import ensure_inspector_board, registry_path

    root = tmp_path / "bench"
    root.mkdir()
    registry = registry_path(root)
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "pid": 12345,
                "root": str(root.resolve()),
                "host": "127.0.0.1",
                "port": 7777,
                "url": "http://127.0.0.1:7777",
                "log_path": str(root / ".cage" / "inspect.log"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("cage.web.inspect_board.is_pid_alive", lambda _pid: True)

    info = ensure_inspector_board(
        root,
        WebInspectorConfig(port=7777),
        mode="on",
        interactive=True,
    )

    assert info.started is False
    assert info.url == "http://127.0.0.1:7777"
    assert info.pid == 12345


def test_ensure_inspector_board_ignores_registry_on_stale_port(
    tmp_path: Path, monkeypatch,
) -> None:
    # Regression: a registry left over from before a fixed port was pinned records
    # a once-ephemeral port (e.g. 44809). With config now pinning 7777, that cached
    # board must NOT be reused — it would shadow the single port forever. The board
    # must fall through to the single-port reuse path and land on 7777.
    from cage.web.inspect_board import ensure_inspector_board, registry_path

    root = tmp_path / "bench"
    root.mkdir()
    registry = registry_path(root)
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "pid": 99999,
                "root": str(root.resolve()),
                "host": "0.0.0.0",
                "port": 44809,  # stale ephemeral port from before 7777 was pinned
                "url": "http://0.0.0.0:44809",
                "log_path": str(root / ".cage" / "inspect.log"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("cage.web.inspect_board.is_pid_alive", lambda _pid: True)
    # 7777 is held by a live standing inspector → single-port reuse should win.
    monkeypatch.setattr("cage.web.inspect_board.port_is_free", lambda _h, _p: False)
    monkeypatch.setattr(
        "cage.web.inspect_board.find_listening_pid", lambda _p, _h=None: 63571
    )

    info = ensure_inspector_board(
        root,
        WebInspectorConfig(host="0.0.0.0", port=7777),
        mode="on",
        interactive=True,
    )

    assert info.started is False  # reused the standing 7777 board, no new spawn
    assert info.port == 7777
    assert info.pid == 63571
    assert info.url == "http://0.0.0.0:7777"


def test_ensure_inspector_board_starts_background_process(
    tmp_path: Path, monkeypatch,
) -> None:
    from cage.web.inspect_board import ensure_inspector_board, registry_path

    root = tmp_path / "bench"
    root.mkdir()
    calls: dict[str, object] = {}

    class FakePopen:
        pid = 2468

        def __init__(self, argv, **kwargs):
            calls["argv"] = argv
            calls["kwargs"] = kwargs

    monkeypatch.setattr("cage.web.inspect_board.pick_free_port", lambda _host: 8765)
    monkeypatch.setattr("subprocess.Popen", FakePopen)

    info = ensure_inspector_board(
        root,
        WebInspectorConfig(port=0, open_browser=True),
        mode="on",
        interactive=True,
    )

    assert info.started is True
    assert info.pid == 2468
    assert info.url == "http://0.0.0.0:8765"
    assert info.auth_token == ""
    argv = calls["argv"]
    assert argv[:3] == [sys.executable, "-m", "cage.web.inspect_server"]
    assert str(root.resolve()) in argv
    assert "--host" in argv
    assert "0.0.0.0" in argv
    assert "--port" in argv
    assert "8765" in argv
    assert "--no-open" in argv
    assert "--auth-token" not in argv
    saved = json.loads(registry_path(root).read_text(encoding="utf-8"))
    assert saved["pid"] == 2468
    assert saved["root"] == str(root.resolve())
    assert saved["auth_token"] == ""


def test_start_inspector_board_reuses_when_configured_port_busy(
    tmp_path: Path, monkeypatch,
) -> None:
    # Single-port policy: a fixed cage.yml port (e.g. 7777) already in use must
    # be REUSED, not silently relocated to a random free port. No second
    # inspector process is spawned, and the URL keeps pointing at the one port.
    from cage.web.inspect_board import ensure_inspector_board

    root = tmp_path / "bench"
    root.mkdir()
    spawned: dict[str, object] = {}

    class FakePopen:
        pid = 4321

        def __init__(self, argv, **kwargs):
            spawned["argv"] = argv

    monkeypatch.setattr("cage.web.inspect_board.port_is_free", lambda _h, _p: False)
    monkeypatch.setattr(
        "cage.web.inspect_board.find_listening_pid", lambda _p, _h=None: 9999,
    )
    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    info = ensure_inspector_board(
        root,
        WebInspectorConfig(host="0.0.0.0", port=7777),
        mode="on",
        interactive=True,
    )

    assert info.started is False
    assert info.alive is True
    assert info.port == 7777
    assert info.url == "http://0.0.0.0:7777"
    assert info.pid == 9999
    assert "argv" not in spawned  # no second inspector was started


def test_start_inspector_board_spawns_when_port_busy_without_live_listener(
    tmp_path: Path, monkeypatch,
) -> None:
    # `cage inspect stop` then `start` leaves a TIME_WAIT socket: the port is
    # unbindable (port_is_free False) but nothing is LISTENing on it. The board
    # must START (Werkzeug binds with SO_REUSEADDR), not "reuse" a dead port and
    # leave the configured port unserved.
    from cage.web.inspect_board import ensure_inspector_board, registry_path

    root = tmp_path / "bench"
    root.mkdir()
    spawned: dict[str, object] = {}

    class FakePopen:
        pid = 7654

        def __init__(self, argv, **kwargs):
            spawned["argv"] = argv

    monkeypatch.setattr("cage.web.inspect_board.port_is_free", lambda _h, _p: False)
    monkeypatch.setattr(
        "cage.web.inspect_board.find_listening_pid", lambda _p, _h=None: None,
    )
    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    info = ensure_inspector_board(
        root,
        WebInspectorConfig(host="0.0.0.0", port=7777),
        mode="on",
        interactive=True,
    )

    assert info.started is True
    assert info.pid == 7654
    assert info.port == 7777  # kept the configured port, did not relocate
    assert info.url == "http://0.0.0.0:7777"
    assert "argv" in spawned  # a real board was started
    saved = json.loads(registry_path(root).read_text(encoding="utf-8"))
    assert saved["pid"] == 7654


def test_ensure_inspector_board_does_not_register_failed_start(
    tmp_path: Path, monkeypatch,
) -> None:
    from cage.web.inspect_board import ensure_inspector_board, registry_path

    root = tmp_path / "bench"
    root.mkdir()
    registry = registry_path(root)
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "pid": 1234,
                "root": str(root.resolve()),
                "host": "0.0.0.0",
                "port": 7777,
                "url": "http://0.0.0.0:7777",
                "log_path": str(root / ".cage" / "inspect.log"),
            }
        ),
        encoding="utf-8",
    )

    class FailedPopen:
        pid = 2468

        def __init__(self, argv, **kwargs):
            pass

        def poll(self):
            return 1

    monkeypatch.setattr("cage.web.inspect_board.is_pid_alive", lambda _pid: False)
    # Free port → reach the spawn path (this test is about failed-start cleanup,
    # not the single-port reuse branch).
    monkeypatch.setattr("cage.web.inspect_board.port_is_free", lambda _h, _p: True)
    monkeypatch.setattr("subprocess.Popen", FailedPopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    info = ensure_inspector_board(
        root,
        WebInspectorConfig(host="0.0.0.0", port=7777),
        mode="on",
        interactive=True,
    )

    assert info.enabled is False
    assert info.started is False
    assert info.alive is False
    assert info.url == ""
    assert "failed to start" in info.message
    assert not registry.exists()


def test_inspector_board_public_bind_does_not_generate_auth_when_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cage.web.inspect_board import ensure_inspector_board, registry_path

    root = tmp_path / "bench"
    root.mkdir()
    calls: dict[str, object] = {}

    class FakePopen:
        pid = 2468

        def __init__(self, argv, **kwargs):
            calls["argv"] = argv
            calls["kwargs"] = kwargs

    monkeypatch.setattr("cage.web.inspect_board.pick_free_port", lambda _host: 8765)
    monkeypatch.setattr("subprocess.Popen", FakePopen)

    info = ensure_inspector_board(
        root,
        WebInspectorConfig(host="0.0.0.0", port=0),
        mode="on",
        interactive=True,
    )

    argv = calls["argv"]
    assert info.url == "http://0.0.0.0:8765"
    assert info.auth_token == ""
    assert "--auth-token" not in argv
    saved = json.loads(registry_path(root).read_text(encoding="utf-8"))
    assert saved["auth_token"] == ""


def test_ensure_inspector_board_auto_skips_non_interactive(tmp_path: Path) -> None:
    from cage.web.inspect_board import ensure_inspector_board

    root = tmp_path / "bench"
    root.mkdir()

    info = ensure_inspector_board(
        root,
        WebInspectorConfig(port=7777),
        mode="auto",
        interactive=False,
    )

    assert info.enabled is False
    assert info.started is False
    assert info.url == ""


def test_inspect_start_status_stop_commands_use_managed_board(
    tmp_path: Path, monkeypatch,
) -> None:
    start_calls: list[Path] = []
    stop_calls: list[Path] = []

    monkeypatch.setattr(
        "cage.web.inspect_board.ensure_inspector_board",
        lambda root, web_config, mode, interactive: (
            start_calls.append(Path(root))
            or SimpleNamespace(
                enabled=True,
                started=True,
                url="http://127.0.0.1:7777",
                pid=123,
                root=Path(root),
                log_path=Path(root) / ".cage" / "inspect.log",
            )
        ),
    )
    monkeypatch.setattr(
        "cage.web.inspect_board.stop_inspector_board",
        lambda root: stop_calls.append(Path(root)) or True,
    )
    # status is port-based: it reports whoever holds the single shared port.
    monkeypatch.setattr(
        "cage.config.load_repo_config",
        lambda _root=None: SimpleNamespace(
            web_inspector=SimpleNamespace(host="127.0.0.1", port=7777),
        ),
    )
    monkeypatch.setattr("cage.web.inspect_board.port_is_free", lambda _h, _p: False)
    monkeypatch.setattr(
        "cage.web.inspect_board.find_listening_pid", lambda _p, _h=None: 123,
    )
    monkeypatch.setattr(
        "cage.web.inspect_board.describe_pid",
        lambda _pid: "python -m cage.web.inspect_server",
    )
    monkeypatch.setattr(
        "cage.web.inspect_board.inspect_board_status",
        lambda root: SimpleNamespace(alive=False, log_path=None, root=Path(root)),
    )

    runner = CliRunner()
    start = runner.invoke(main, ["inspect", "start", str(tmp_path)])
    status = runner.invoke(main, ["inspect", "status", str(tmp_path)])
    stop = runner.invoke(main, ["inspect", "stop", str(tmp_path)])

    assert start.exit_code == 0, start.output
    assert "Inspector board started" in start.output
    assert "http://127.0.0.1:7777" in start.output
    assert status.exit_code == 0, status.output
    assert "running on port 7777" in status.output
    assert "PID: 123" in status.output
    assert stop.exit_code == 0, stop.output
    assert "Inspector board stopped" in stop.output
    assert start_calls == [tmp_path]
    assert stop_calls == [tmp_path]


def test_repo_config_reads_managed_board_defaults(tmp_path: Path) -> None:
    from cage.config import WebInspectorBoardConfig, load_repo_config

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cage.yml").write_text(
        """
web_inspector:
  host: 127.0.0.1
  port: 7777
  open_browser: false
  board:
    enabled: true
    auto_start_on_run: false
""",
        encoding="utf-8",
    )

    config = load_repo_config(tmp_path)

    assert isinstance(config.web_inspector.board, WebInspectorBoardConfig)
    assert config.web_inspector.board.enabled is True
    assert config.web_inspector.board.auto_start_on_run is False
