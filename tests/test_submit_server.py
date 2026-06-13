"""Tests for cage.target.services.submit.server."""

from __future__ import annotations

import json
import socket
import threading

from cage.target.services.submit.server import (
    SubmitState,
    build_unix_server,
    normalize_answer,
    sha256_hex,
)


def _make_state(
    *,
    question_id: str = "q1",
    answer: str = "flag{test}",
    max_checks: int = 3,
    benchmark: str = "nyu_ctf",
    artifact_path: str = "",
) -> SubmitState:
    return SubmitState(
        question_id=question_id,
        answer_sha256=sha256_hex(normalize_answer(answer)),
        max_checks=max_checks,
        benchmark=benchmark,
        artifact_path=artifact_path,
    )


class TestSubmitState:
    def test_accepts_correct_answer(self):
        state = _make_state(answer="flag{test}")

        result = state.submit("flag{test}")

        assert result["correct"] is True
        assert result["message"] == "Answer accepted."
        assert result["checks_remaining"] == 2

    def test_rejects_wrong_answer(self):
        state = _make_state(answer="flag{test}")

        result = state.submit("flag{wrong}")

        assert result["correct"] is False
        assert result["message"] == "Answer not accepted."
        assert result["checks_remaining"] == 2

    def test_limits_checks(self):
        state = _make_state(answer="flag{test}", max_checks=1)

        state.submit("flag{wrong}")
        result = state.submit("flag{test}")

        assert result["correct"] is False
        assert result["checks_remaining"] == 0
        assert "No checks remaining" in result["message"]

    def test_normalizes_answer(self):
        state = _make_state(answer="flag{test}")

        result = state.submit("  flag{test}  ")

        assert result["correct"] is True

    def test_artifact_contains_hash_not_plaintext(self, tmp_path):
        log_path = tmp_path / "live_checks.jsonl"
        state = _make_state(answer="flag{secret}", artifact_path=str(log_path))

        state.submit("flag{secret}")

        content = log_path.read_text(encoding="utf-8")
        assert "flag{secret}" not in content
        entry = json.loads(content)
        assert entry["question_id"] == "q1"
        assert entry["source"] == "container-submit"
        assert entry["correct"] is True
        assert entry["answer_hash"].startswith("sha256:")


class TestSubmitUnixSocket:
    def test_handles_submit_request_over_unix_socket(self, tmp_path):
        socket_path = tmp_path / "submit.sock"
        state = _make_state(answer="flag{socket}")
        server = build_unix_server(str(socket_path), state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                client.sendall(b'{"answer": "flag{socket}"}\n')
                payload = client.recv(4096)

            result = json.loads(payload.decode("utf-8"))
            assert result["correct"] is True
            assert result["checks_remaining"] == 2
        finally:
            server.shutdown()
            server.server_close()

    def test_rejects_invalid_json_over_unix_socket(self, tmp_path):
        socket_path = tmp_path / "submit.sock"
        server = build_unix_server(str(socket_path), _make_state())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                client.sendall(b"not-json\n")
                payload = client.recv(4096)

            result = json.loads(payload.decode("utf-8"))
            assert result["correct"] is False
            assert "invalid JSON" in result["message"]
        finally:
            server.shutdown()
            server.server_close()
