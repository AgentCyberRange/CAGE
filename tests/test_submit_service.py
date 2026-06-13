"""Tests for cage.target.services.submit.service — host-side submit lifecycle."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cage.target.services.submit.server import normalize_answer, sha256_hex
from cage.target.services.submit.service import (
    SUBMIT_CLIENT_PATH,
    SUBMIT_SERVER_PATH,
    SubmitServiceHandle,
    needs_submit_service,
    start_submit_service,
)


def _container() -> MagicMock:
    container = MagicMock()
    container.name = "agent-ctr"
    container.copy_to.return_value = SimpleNamespace(exit_code=0, stdout="", stderr="")
    container.exec.return_value = SimpleNamespace(exit_code=0, stdout="", stderr="")
    return container


def _process() -> MagicMock:
    process = MagicMock()
    process.stdin = MagicMock()
    process.stdout.readline.return_value = "READY\n"
    process.poll.return_value = None
    process.wait.return_value = 0
    return process


class TestNeedsSubmitService:
    def test_declared_enabled(self):
        assert needs_submit_service(SimpleNamespace(needs_submit_service=True), True) is True

    def test_not_declared(self):
        assert needs_submit_service(SimpleNamespace(needs_submit_service=False), True) is False

    def test_disabled(self):
        assert needs_submit_service(SimpleNamespace(needs_submit_service=True), False) is False


class TestStartSubmitService:
    def test_copies_server_and_client_and_sets_permissions(self, tmp_path):
        container = _container()
        process = _process()

        with patch("cage.target.services.submit.service.subprocess.Popen", return_value=process):
            start_submit_service(
                container=container,
                question_id="q1",
                expected_answer="flag{secret}",
                max_checks=3,
                benchmark="nyu_ctf",
                trial_artifact_dir=tmp_path,
                trial_id="trial-q1",
            )

        copied_destinations = [call.args[1] for call in container.copy_to.call_args_list]
        assert SUBMIT_SERVER_PATH in copied_destinations
        assert SUBMIT_CLIENT_PATH in copied_destinations

        setup_commands = "\n".join(call.args[0] for call in container.exec.call_args_list)
        assert "chown root:root /opt/cage-submit" in setup_commands
        assert "chmod 0700 /opt/cage-submit" in setup_commands
        assert f"chown root:root {SUBMIT_SERVER_PATH}" in setup_commands
        assert f"chmod 0500 {SUBMIT_SERVER_PATH}" in setup_commands
        assert f"chown root:root {SUBMIT_CLIENT_PATH}" in setup_commands
        assert f"chmod 0755 {SUBMIT_CLIENT_PATH}" in setup_commands

    def test_starts_server_as_root_with_stdin_open(self, tmp_path):
        container = _container()
        process = _process()

        with patch(
            "cage.target.services.submit.service.subprocess.Popen",
            return_value=process,
        ) as mock_popen:
            start_submit_service(
                container=container,
                question_id="q1",
                expected_answer="flag{secret}",
                max_checks=3,
                benchmark="nyu_ctf",
                trial_artifact_dir=tmp_path,
                trial_id="trial-q1",
            )

        cmd = mock_popen.call_args.args[0]
        assert cmd == [
            "docker",
            "exec",
            "-i",
            "-u",
            "root",
            "agent-ctr",
            "python3",
            SUBMIT_SERVER_PATH,
        ]
        assert process.stdin.close.call_count == 0

    def test_initializes_server_with_answer_hash_not_plaintext(self, tmp_path):
        container = _container()
        process = _process()
        expected_hash = sha256_hex(normalize_answer("flag{secret}"))

        with patch("cage.target.services.submit.service.subprocess.Popen", return_value=process):
            start_submit_service(
                container=container,
                question_id="q1",
                expected_answer="flag{secret}",
                max_checks=5,
                benchmark="nyu_ctf",
                trial_artifact_dir=tmp_path,
                trial_id="trial-q1",
                container_artifact_path="/var/lib/cage/live_checks.jsonl",
            )

        init_payload = process.stdin.write.call_args.args[0]
        assert "flag{secret}" not in init_payload
        init = json.loads(init_payload)
        assert init["question_id"] == "q1"
        assert init["answer_sha256"] == expected_hash
        assert init["max_checks"] == 5
        assert init["benchmark"] == "nyu_ctf"
        assert init["artifact_path"] == "/var/lib/cage/live_checks.jsonl"

    def test_does_not_put_plaintext_answer_in_docker_command(self, tmp_path):
        container = _container()
        process = _process()

        with patch(
            "cage.target.services.submit.service.subprocess.Popen",
            return_value=process,
        ) as mock_popen:
            start_submit_service(
                container=container,
                question_id="q1",
                expected_answer="flag{secret}",
                max_checks=3,
                benchmark="nyu_ctf",
                trial_artifact_dir=tmp_path,
                trial_id="trial-q1",
            )

        assert "flag{secret}" not in " ".join(mock_popen.call_args.args[0])

    def test_raises_when_server_does_not_report_ready(self, tmp_path):
        container = _container()
        process = _process()
        process.stdout.readline.return_value = "boom\n"

        with patch("cage.target.services.submit.service.subprocess.Popen", return_value=process):
            try:
                start_submit_service(
                    container=container,
                    question_id="q1",
                    expected_answer="flag{secret}",
                    max_checks=3,
                    benchmark="nyu_ctf",
                    trial_artifact_dir=tmp_path,
                    trial_id="trial-q1",
                )
            except RuntimeError as exc:
                assert "did not become ready" in str(exc)
            else:
                raise AssertionError("expected RuntimeError")

        cleanup_commands = "\n".join(call.args[0] for call in container.exec.call_args_list)
        assert "rm -rf /run/cage-submit /opt/cage-submit" in cleanup_commands
        assert f"rm -f {SUBMIT_CLIENT_PATH}" in cleanup_commands


class TestSubmitServiceHandle:
    def test_stop_closes_stdin_and_cleans_container_files(self):
        container = _container()
        process = _process()
        handle = SubmitServiceHandle(
            process=process,
            container=container,
            trial_id="trial-q1",
        )

        handle.stop()

        process.stdin.close.assert_called_once()
        process.terminate.assert_called_once()
        process.wait.assert_called()
        cleanup_commands = "\n".join(call.args[0] for call in container.exec.call_args_list)
        assert "rm -rf /run/cage-submit /opt/cage-submit" in cleanup_commands
        assert f"rm -f {SUBMIT_CLIENT_PATH}" in cleanup_commands
