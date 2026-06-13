"""Tests for cage.target.services.check.server."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cage.target.services.check.server import CheckState, normalize_answer, sha256_hex


def _make_state(
    question_id: str = "test-q1",
    answer: str = "flag{test}",
    max_checks: int = 3,
    benchmark: str = "nyu_ctf",
    artifact_path: str = "",
) -> CheckState:
    return CheckState(
        question_id=question_id,
        answer_sha256=sha256_hex(normalize_answer(answer)),
        max_checks=max_checks,
        benchmark=benchmark,
        artifact_path=artifact_path,
    )


class TestNormalizeAnswer:
    def test_strip_whitespace(self):
        assert normalize_answer("  flag{test}  ") == "flag{test}"

    def test_no_change_on_clean(self):
        assert normalize_answer("flag{test}") == "flag{test}"


class TestSha256Hex:
    def test_known_value(self):
        result = sha256_hex("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result == expected


class TestCheckState:
    def test_correct_answer(self):
        state = _make_state(answer="flag{test}")
        result = state.check("test-q1", "flag{test}")
        assert result["correct"] is True
        assert result["checks_remaining"] == 2

    def test_wrong_answer(self):
        state = _make_state(answer="flag{test}")
        result = state.check("test-q1", "flag{wrong}")
        assert result["correct"] is False
        assert result["message"] == "Answer not accepted."
        assert result["checks_remaining"] == 2

    def test_wrong_question_id(self):
        state = _make_state(answer="flag{test}")
        result = state.check("other-q", "flag{test}")
        assert result["correct"] is False
        assert "does not match" in result["message"]

    def test_max_checks_exhausted(self):
        state = _make_state(answer="flag{test}", max_checks=1)
        # Use up the one allowed check
        state.check("test-q1", "flag{wrong}")
        # Now no checks remaining
        result = state.check("test-q1", "flag{test}")
        assert result["correct"] is False
        assert result["checks_remaining"] == 0
        assert "No checks remaining" in result["message"]

    def test_remaining_count_decrements(self):
        state = _make_state(max_checks=3)
        r1 = state.check("test-q1", "a")
        assert r1["checks_remaining"] == 2
        r2 = state.check("test-q1", "b")
        assert r2["checks_remaining"] == 1
        r3 = state.check("test-q1", "c")
        assert r3["checks_remaining"] == 0
        r4 = state.check("test-q1", "d")
        assert r4["checks_remaining"] == 0

    def test_correct_answer_still_decrements(self):
        state = _make_state(max_checks=2)
        state.check("test-q1", "flag{test}")
        result = state.check("test-q1", "flag{test}")
        assert result["correct"] is True
        assert result["checks_remaining"] == 0

    def test_artifact_logged(self, tmp_path):
        artifact_dir = str(tmp_path / "artifacts")
        state = _make_state(answer="flag{test}", artifact_path=artifact_dir)
        state.check("test-q1", "flag{wrong}")

        log_path = Path(artifact_dir) / "live_checks.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["correct"] is False
        assert entry["question_id"] == "test-q1"
        assert "answer_hash" in entry
        # No plaintext answer
        assert "flag{wrong}" not in json.dumps(entry)

    def test_artifact_not_containing_plaintext_expected(self, tmp_path):
        artifact_dir = str(tmp_path / "artifacts")
        state = _make_state(answer="flag{secret}", artifact_path=artifact_dir)
        state.check("test-q1", "flag{secret}")

        log_path = Path(artifact_dir) / "live_checks.jsonl"
        content = log_path.read_text()
        assert "flag{secret}" not in content

    def test_no_artifact_when_path_empty(self):
        state = _make_state(artifact_path="")
        # Should not raise
        state.check("test-q1", "flag{test}")

    def test_normalized_answer_comparison(self):
        """Answers with extra whitespace still match after normalization."""
        state = _make_state(answer="flag{test}")
        result = state.check("test-q1", "  flag{test}  ")
        assert result["correct"] is True


class TestHTTPEndpoints:
    """Integration tests using the actual HTTP handler."""

    @pytest.fixture
    def server_url(self):
        """Start a real check server on a free port and return its base URL."""
        import socket
        import threading

        from cage.target.services.check.server import _CheckHandler

        # Find free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        from http.server import ThreadingHTTPServer
        srv = ThreadingHTTPServer(("127.0.0.1", port), _CheckHandler)

        # Set up global state
        import cage.target.services.check.server as cs_module
        original_state = cs_module._check_state
        cs_module._check_state = _make_state(answer="flag{http_test}", max_checks=3)

        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()

        yield f"http://127.0.0.1:{port}"

        srv.shutdown()
        cs_module._check_state = original_state

    def _client(self):
        import httpx
        return httpx.Client(trust_env=False)

    def test_healthz(self, server_url):
        with self._client() as client:
            resp = client.get(f"{server_url}/healthz")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_correct_answer(self, server_url):
        with self._client() as client:
            resp = client.post(
                f"{server_url}/check",
                json={"question_id": "test-q1", "answer": "flag{http_test}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["correct"] is True
            assert data["checks_remaining"] == 2

    def test_wrong_answer(self, server_url):
        with self._client() as client:
            resp = client.post(
                f"{server_url}/check",
                json={"question_id": "test-q1", "answer": "flag{wrong}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["correct"] is False

    def test_missing_fields(self, server_url):
        with self._client() as client:
            resp = client.post(
                f"{server_url}/check",
                json={"question_id": "test-q1"},
            )
            assert resp.status_code == 400

    def test_not_found(self, server_url):
        with self._client() as client:
            resp = client.get(f"{server_url}/nonexistent")
            assert resp.status_code == 404


class TestLoadStateFromEnv:
    def test_loads_from_env(self):
        from cage.target.services.check.server import _load_state_from_env
        answer = "flag{env_test}"
        with patch.dict(os.environ, {
            "QUESTION_ID": "q1",
            "ANSWER_SHA256": sha256_hex(normalize_answer(answer)),
            "MAX_CHECKS": "5",
            "BENCHMARK": "test_bench",
            "ARTIFACT_PATH": "/tmp/artifacts",
        }):
            state = _load_state_from_env()
        assert state.question_id == "q1"
        assert state.max_checks == 5
        assert state.benchmark == "test_bench"
        assert state.answer_sha256 == sha256_hex(normalize_answer(answer))

    def test_missing_required_env(self):
        from cage.target.services.check.server import _load_state_from_env
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="QUESTION_ID"):
                _load_state_from_env()