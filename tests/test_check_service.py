"""Tests for cage.target.services.check.service — unit tests (no Docker required)."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cage.target.services.check.service import (
    CheckServiceHandle,
    has_builtin_check,
    needs_check_container,
    start_check_service,
)


@dataclass
class _BenchCaps:
    """Minimal stand-in declaring the benchmark live-check capability flags."""

    needs_check_service: bool = False
    uses_builtin_check: bool = False
    needs_submit_service: bool = False


# A benchmark that needs a check container (e.g. nyu_ctf / autopenbench)
_CHECK_CONTAINER = _BenchCaps(needs_check_service=True)
# A benchmark with a built-in check (e.g. cvebench)
_BUILTIN_CHECK = _BenchCaps(uses_builtin_check=True)
# A benchmark needing no live-check services (e.g. strongreject)
_NO_SERVICES = _BenchCaps()


class TestNeedsCheckContainer:
    def test_declared_enabled(self):
        assert needs_check_container(_CHECK_CONTAINER, True) is True

    def test_builtin_check_does_not_need_container(self):
        assert needs_check_container(_BUILTIN_CHECK, True) is False

    def test_declared_but_disabled(self):
        assert needs_check_container(_CHECK_CONTAINER, False) is False

    def test_other_benchmark(self):
        assert needs_check_container(_NO_SERVICES, True) is False

    def test_other_benchmark_disabled(self):
        assert needs_check_container(_NO_SERVICES, False) is False


class TestHasBuiltinCheck:
    def test_declared_enabled(self):
        assert has_builtin_check(_BUILTIN_CHECK, True) is True

    def test_declared_disabled(self):
        assert has_builtin_check(_BUILTIN_CHECK, False) is False

    def test_check_container_is_not_builtin(self):
        assert has_builtin_check(_CHECK_CONTAINER, True) is False

    def test_no_services_is_not_builtin(self):
        assert has_builtin_check(_NO_SERVICES, True) is False


class TestCheckServiceHandle:
    def test_stop_removes_container_only(self):
        handle = CheckServiceHandle(
            container_name="test-ctr",
            network_name="test-net",
            artifact_path=Path("/tmp/test"),
        )
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=0)
            handle.stop()
            assert mock_docker.call_count == 1
            cmd = mock_docker.call_args[0][0]
            assert cmd == ["docker", "rm", "-f", "test-ctr"]

    def test_stop_handles_failure_gracefully(self):
        handle = CheckServiceHandle(
            container_name="test-ctr",
            network_name="test-net",
            artifact_path=Path("/tmp/test"),
        )
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=1, stderr="not found")
            # Should not raise
            handle.stop()


class TestStartCheckService:
    def test_requires_network_name(self, tmp_path):
        """The orchestrator must provide the network for the check container."""
        artifact_dir = tmp_path / "artifacts"
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            with pytest.raises(ValueError, match="network_name is required"):
                start_check_service(
                    question_id="q1",
                    expected_answer="flag{test}",
                    max_checks=3,
                    benchmark="nyu_ctf",
                    trial_artifact_dir=artifact_dir,
                    run_id="run-1",
                    trial_id="trial-1",
                    network_name=None,
                )
        mock_docker.assert_not_called()

    def test_uses_existing_network(self, tmp_path):
        """When network_name is provided, no new network is created."""
        artifact_dir = tmp_path / "artifacts"
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=0)
            handle = start_check_service(
                question_id="q1",
                expected_answer="flag{test}",
                max_checks=3,
                benchmark="autopenbench",
                trial_artifact_dir=artifact_dir,
                run_id="run-1",
                trial_id="trial-1",
                network_name="existing-net",
            )
        assert handle.network_name == "existing-net"
        # Only docker run, no network create
        assert mock_docker.call_count == 1
        run_cmd = mock_docker.call_args[0][0]
        assert "--network" in run_cmd
        net_idx = run_cmd.index("--network")
        assert run_cmd[net_idx + 1] == "existing-net"

    def test_docker_run_has_network_alias_check(self, tmp_path):
        """The check container is started with --network-alias check."""
        artifact_dir = tmp_path / "artifacts"
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=0)
            start_check_service(
                question_id="q1",
                expected_answer="flag{test}",
                max_checks=3,
                benchmark="nyu_ctf",
                trial_artifact_dir=artifact_dir,
                run_id="r1",
                trial_id="t1",
                network_name="some-net",
            )
        run_cmd = mock_docker.call_args[0][0]
        assert "--network-alias" in run_cmd
        alias_idx = run_cmd.index("--network-alias")
        assert run_cmd[alias_idx + 1] == "check"

    def test_docker_run_mounts_check_server_script(self, tmp_path):
        """The check server source is mounted into the plain Python image."""
        artifact_dir = tmp_path / "artifacts"
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=0)
            start_check_service(
                question_id="q1",
                expected_answer="flag{test}",
                max_checks=3,
                benchmark="nyu_ctf",
                trial_artifact_dir=artifact_dir,
                run_id="r1",
                trial_id="t1",
                network_name="net1",
            )

        run_cmd = mock_docker.call_args[0][0]
        mount_specs = [
            run_cmd[i + 1]
            for i, token in enumerate(run_cmd)
            if token == "--mount" and i + 1 < len(run_cmd)
        ]
        assert any(
            "target=/check_server.py" in spec and "readonly" in spec
            for spec in mount_specs
        )
        assert run_cmd[-3:] == ["python:3.11-slim", "python", "/check_server.py"]

    def test_docker_run_mounts_writable_artifact_dir(self, tmp_path):
        """The check server must be able to write live_checks.jsonl."""
        artifact_dir = tmp_path / "artifacts"
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=0)
            start_check_service(
                question_id="q1",
                expected_answer="flag{test}",
                max_checks=3,
                benchmark="nyu_ctf",
                trial_artifact_dir=artifact_dir,
                run_id="r1",
                trial_id="t1",
                network_name="net1",
            )

        run_cmd = mock_docker.call_args[0][0]
        mount_specs = [
            run_cmd[i + 1]
            for i, token in enumerate(run_cmd)
            if token == "--mount" and i + 1 < len(run_cmd)
        ]
        artifact_mount = next(spec for spec in mount_specs if "target=/artifacts" in spec)
        assert "readonly" not in artifact_mount

    def test_docker_run_passes_env_vars(self, tmp_path):
        """Required env vars are passed to the container."""
        artifact_dir = tmp_path / "artifacts"
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=0)
            start_check_service(
                question_id="q1",
                expected_answer="flag{test}",
                max_checks=5,
                benchmark="nyu_ctf",
                trial_artifact_dir=artifact_dir,
                run_id="r1",
                trial_id="t1",
                network_name="net1",
            )
        run_cmd = mock_docker.call_args[0][0]
        env_pairs = []
        for i, token in enumerate(run_cmd):
            if token == "-e" and i + 1 < len(run_cmd):
                env_pairs.append(run_cmd[i + 1])

        assert any(p.startswith("QUESTION_ID=q1") for p in env_pairs)
        assert any(p.startswith("MAX_CHECKS=5") for p in env_pairs)
        assert any(p.startswith("BENCHMARK=nyu_ctf") for p in env_pairs)
        assert any(p.startswith("ANSWER_SHA256=") for p in env_pairs)
        assert any(p.startswith("ARTIFACT_PATH=/artifacts") for p in env_pairs)

    def test_docker_run_failure_does_not_remove_external_network(self, tmp_path):
        """Network lifecycle is owned by the orchestrator, not check_service."""
        artifact_dir = tmp_path / "artifacts"
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=1, stderr="container start failed")
            with pytest.raises(RuntimeError, match="container start failed"):
                start_check_service(
                    question_id="q1",
                    expected_answer="flag{test}",
                    max_checks=3,
                    benchmark="nyu_ctf",
                    trial_artifact_dir=artifact_dir,
                    run_id="r1",
                    trial_id="t1",
                    network_name="external-net",
                )
        assert mock_docker.call_count == 1
        run_cmd = mock_docker.call_args[0][0]
        assert run_cmd[:3] == ["docker", "run", "-d"]

    def test_artifact_dir_created(self, tmp_path):
        """The artifact directory is created if it doesn't exist."""
        artifact_dir = tmp_path / "nested" / "artifacts"
        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=0)
            start_check_service(
                question_id="q1",
                expected_answer="flag{test}",
                max_checks=3,
                benchmark="nyu_ctf",
                trial_artifact_dir=artifact_dir,
                run_id="r1",
                trial_id="t1",
                network_name="net1",
            )
        assert artifact_dir.exists()

    def test_answer_is_hashed_not_plaintext(self, tmp_path):
        """The expected answer is hashed before being passed as env var."""
        artifact_dir = tmp_path / "artifacts"
        from cage.target.services.check.server import normalize_answer, sha256_hex
        expected_hash = sha256_hex(normalize_answer("flag{secret}"))

        with patch("cage.target.services.check.service._run_docker") as mock_docker:
            mock_docker.return_value = MagicMock(exit_code=0)
            start_check_service(
                question_id="q1",
                expected_answer="flag{secret}",
                max_checks=3,
                benchmark="nyu_ctf",
                trial_artifact_dir=artifact_dir,
                run_id="r1",
                trial_id="t1",
                network_name="net1",
            )
        run_cmd = mock_docker.call_args[0][0]
        env_pairs = []
        for i, token in enumerate(run_cmd):
            if token == "-e" and i + 1 < len(run_cmd):
                env_pairs.append(run_cmd[i + 1])

        hash_var = next(p for p in env_pairs if p.startswith("ANSWER_SHA256="))
        actual_hash = hash_var.split("=", 1)[1]
        assert actual_hash == expected_hash
        # Placket text not present in command
        assert "flag{secret}" not in " ".join(run_cmd)
