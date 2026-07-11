"""Tests for cage.sandbox.containers."""

from __future__ import annotations

import socket
from pathlib import Path

from cage.sandbox import containers as container_module
from cage.sandbox.containers import Container, resolve_extra_hosts_for_url
from cage.sandbox.exec import ExecResult
from cage.sandbox.state import (
    STATE_TRANSFER_TIMEOUT_SECONDS,
    StateSnapshot,
    restore_state,
)


class TestContainer:
    def test_resolve_extra_hosts_for_url_uses_host_dns(self, monkeypatch) -> None:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *args, **kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.10.10.8", 0)),
            ],
        )

        assert resolve_extra_hosts_for_url("https://internal-model.example.com/v1") == {
            "internal-model.example.com": "10.10.10.8",
        }

    def test_resolve_extra_hosts_for_url_skips_ip_literal(self) -> None:
        assert resolve_extra_hosts_for_url("http://10.1.2.3:8000/v1") == {}

    def test_start_uses_configured_network_mode(self) -> None:
        container = Container(
            name="cage-test-network",
            image="cage/claude-code:pentestenv",
            network_mode="host",
        )

        recorded: list[list[str]] = []

        def fake_run_local(cmd: list[str], *, timeout: float | None = None) -> ExecResult:
            recorded.append(cmd)
            return ExecResult(
                command=" ".join(cmd),
                stdout="container-id\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
            )

        container._run_local = fake_run_local  # type: ignore[method-assign]

        container.start()

        assert recorded == [[
            "docker",
            "run",
            "-d",
            "--name",
            "cage-test-network",
            "--network",
            "host",
            "-e",
            "HTTP_PROXY=",
            "-e",
            "HTTPS_PROXY=",
            "-e",
            "ALL_PROXY=",
            "-e",
            "http_proxy=",
            "-e",
            "https_proxy=",
            "-e",
            "all_proxy=",
            "--add-host",
            "host.docker.internal:host-gateway",
            "cage/claude-code:pentestenv",
            "sleep",
            "infinity",
        ]]
        assert container.is_running is True
        container_module._ACTIVE_CONTAINER_NAMES.discard(container.name)
        container._started = False

    def test_start_mounts_colon_host_paths_with_mount_flag(self) -> None:
        container = Container(
            name="cage-test-volume-colon",
            image="cage/claude-code:pentestenv",
            volumes={
                "/var/cage_runs/agent:model:stateless/run-fixed/trials/trial-one/proxy": (
                    "/var/lib/cage/proxy"
                ),
            },
        )

        recorded: list[list[str]] = []

        def fake_run_local(cmd: list[str], *, timeout: float | None = None) -> ExecResult:
            recorded.append(cmd)
            return ExecResult(
                command=" ".join(cmd),
                stdout="container-id\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
            )

        container._run_local = fake_run_local  # type: ignore[method-assign]
        container._wait_for_container_ready = lambda: None  # type: ignore[method-assign]

        container.start()

        run_cmd = recorded[0]
        assert "-v" not in run_cmd
        mount_index = run_cmd.index("--mount")
        assert run_cmd[mount_index + 1] == (
            "type=bind,"
            "source=/var/cage_runs/agent:model:stateless/run-fixed/trials/trial-one/proxy,"
            "target=/var/lib/cage/proxy"
        )
        container_module._ACTIVE_CONTAINER_NAMES.discard(container.name)
        container._started = False

    def test_start_preserves_readonly_volume_mode_with_mount_flag(self) -> None:
        container = Container(
            name="cage-test-volume-readonly",
            image="cage/claude-code:pentestenv",
            volumes={"/tmp/plugin-marketplace": "/opt/cage-plugins/plugin-marketplace:ro"},
        )

        recorded: list[list[str]] = []

        def fake_run_local(cmd: list[str], *, timeout: float | None = None) -> ExecResult:
            recorded.append(cmd)
            return ExecResult(
                command=" ".join(cmd),
                stdout="container-id\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
            )

        container._run_local = fake_run_local  # type: ignore[method-assign]
        container._wait_for_container_ready = lambda: None  # type: ignore[method-assign]

        container.start()

        run_cmd = recorded[0]
        mount_index = run_cmd.index("--mount")
        assert run_cmd[mount_index + 1] == (
            "type=bind,"
            "source=/tmp/plugin-marketplace,"
            "target=/opt/cage-plugins/plugin-marketplace,"
            "readonly"
        )
        container_module._ACTIVE_CONTAINER_NAMES.discard(container.name)
        container._started = False

    def test_start_passes_supplemental_groups(self) -> None:
        container = Container(
            name="cage-test-group-add",
            image="cage/claude-code:pentestenv",
            group_add=["1002"],
        )

        recorded: list[list[str]] = []

        def fake_run_local(cmd: list[str], *, timeout: float | None = None) -> ExecResult:
            recorded.append(cmd)
            return ExecResult(
                command=" ".join(cmd),
                stdout="container-id\n",
                stderr="",
                exit_code=0,
                duration_ms=1,
            )

        container._run_local = fake_run_local  # type: ignore[method-assign]
        container._wait_for_container_ready = lambda: None  # type: ignore[method-assign]

        container.start()

        run_cmd = recorded[0]
        assert "--group-add" in run_cmd
        group_index = run_cmd.index("--group-add")
        assert run_cmd[group_index + 1] == "1002"
        container_module._ACTIVE_CONTAINER_NAMES.discard(container.name)
        container._started = False

    def test_sync_runtime_network_connects_and_switches_networks(self) -> None:
        container = Container(
            name="cage-test-network-sync",
            image="cage/claude-code:pentestenv",
            network_mode=None,
        )
        container._started = True

        recorded: list[list[str]] = []

        def fake_run_local(cmd: list[str], *, timeout: float | None = None) -> ExecResult:
            recorded.append(cmd)
            return ExecResult(
                command=" ".join(cmd),
                stdout="",
                stderr="",
                exit_code=0,
                duration_ms=1,
            )

        container._run_local = fake_run_local  # type: ignore[method-assign]

        container.sync_runtime_network("runtime-net-a")
        container.sync_runtime_network("runtime-net-b")
        container.sync_runtime_network(None)

        assert recorded == [
            ["docker", "network", "connect", "runtime-net-a", "cage-test-network-sync"],
            ["docker", "network", "disconnect", "-f", "runtime-net-a", "cage-test-network-sync"],
            ["docker", "network", "connect", "runtime-net-b", "cage-test-network-sync"],
            ["docker", "network", "disconnect", "-f", "runtime-net-b", "cage-test-network-sync"],
        ]

    def test_copy_to_accepts_custom_timeout(self) -> None:
        container = Container(
            name="cage-test-copy-to",
            image="cage/claude-code:pentestenv",
        )

        recorded: list[tuple[list[str], float | None]] = []

        def fake_run_local(cmd: list[str], *, timeout: float | None = None) -> ExecResult:
            recorded.append((cmd, timeout))
            return ExecResult(
                command=" ".join(cmd),
                stdout="",
                stderr="",
                exit_code=0,
                duration_ms=1,
            )

        container._run_local = fake_run_local  # type: ignore[method-assign]

        container.copy_to("/tmp/state", "/home/agent/.claude", timeout=120.0)

        assert recorded == [
            (
                ["docker", "cp", "/tmp/state", "cage-test-copy-to:/home/agent/.claude"],
                120.0,
            )
        ]


class TestStateRestore:
    def test_restore_state_uses_extended_copy_timeout(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        state_dir = snapshot_dir / ".claude"
        state_dir.mkdir()
        (state_dir / "MEMORY.md").write_text("hello", encoding="utf-8")

        snapshot = StateSnapshot(
            snapshot_dir=snapshot_dir,
            state_paths=(".claude",),
            timestamp_ms=0,
        )

        class _Container:
            def __init__(self) -> None:
                self.exec_calls: list[str] = []
                self.copy_calls: list[tuple[str, str, float | None]] = []

            def exec(self, command: str):
                self.exec_calls.append(command)
                return ExecResult(command=command, stdout="", stderr="", exit_code=0, duration_ms=1)

            def copy_to(self, host_path: str, container_path: str, timeout: float | None = None):
                self.copy_calls.append((host_path, container_path, timeout))
                return ExecResult(command="", stdout="", stderr="", exit_code=0, duration_ms=1)

        container = _Container()

        restore_state(
            container,  # type: ignore[arg-type]
            snapshot=snapshot,
            home_dir="/home/agent",
        )

        assert container.exec_calls == ["mkdir -p /home/agent"]
        assert container.copy_calls == [
            (str(state_dir), "/home/agent/.claude", STATE_TRANSFER_TIMEOUT_SECONDS)
        ]
