"""Lightweight HTTP check server for live answer verification.

Runs inside a Docker container alongside the agent. The agent sends POST
requests to ``http://check:8080/check`` with a candidate answer; the server
compares it against the expected answer hash and returns a boolean verdict.

Environment variables:
  QUESTION_ID   — the current challenge/sample ID (must match the request)
  ANSWER_SHA256 — SHA-256 hex digest of the normalised expected answer
  MAX_CHECKS    — maximum number of check attempts (default 3)
  BENCHMARK     — benchmark name for artifact logging
  ARTIFACT_PATH — directory path to write live_checks.jsonl (optional)

The server never exposes the correct answer.  It only returns whether the
submitted answer matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """Normalise an answer for comparison: strip whitespace."""
    return answer.strip()


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Check logic (separable for testing)
# ---------------------------------------------------------------------------

class CheckState:
    """Mutable check state: tracks remaining calls and writes artifacts."""

    def __init__(
        self,
        question_id: str,
        answer_sha256: str,
        max_checks: int,
        benchmark: str = "",
        artifact_path: str = "",
    ) -> None:
        self.question_id = question_id
        self.answer_sha256 = answer_sha256
        self.max_checks = max_checks
        self.checks_remaining = max_checks
        self.benchmark = benchmark
        self.artifact_path = artifact_path

    def check(self, question_id: str, answer: str) -> dict[str, Any]:
        """Evaluate a check request. Returns the JSON response body."""
        if self.checks_remaining <= 0:
            return {
                "correct": False,
                "message": "No checks remaining.",
                "checks_remaining": 0,
            }

        self.checks_remaining -= 1

        # question_id must match exactly
        if question_id != self.question_id:
            self._log_artifact(question_id, answer, correct=False)
            return {
                "correct": False,
                "message": "Question ID does not match current challenge.",
                "checks_remaining": self.checks_remaining,
            }

        norm = normalize_answer(answer)
        candidate_hash = sha256_hex(norm)
        correct = candidate_hash == self.answer_sha256

        result = {
            "correct": correct,
            "message": "Answer accepted." if correct else "Answer not accepted.",
            "checks_remaining": self.checks_remaining,
        }

        self._log_artifact(question_id, answer, correct=correct)
        return result

    def _log_artifact(self, question_id: str, answer: str, *, correct: bool) -> None:
        if not self.artifact_path:
            return
        try:
            artifact_dir = Path(self.artifact_path)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts_ms": int(time.time() * 1000),
                "benchmark": self.benchmark,
                "question_id": question_id,
                "correct": correct,
                "checks_remaining": self.checks_remaining,
                "answer_hash": f"sha256:{sha256_hex(normalize_answer(answer))}",
                "source": "container-check",
            }
            with open(artifact_dir / "live_checks.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # artifact logging is best-effort


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_check_state: CheckState | None = None


class _CheckHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for /healthz and /check."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send_json({"status": "ok"}, HTTPStatus.OK)
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/check":
            self._handle_check()
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _handle_check(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 65536:
            self._send_json({"error": "payload too large"}, HTTPStatus.BAD_REQUEST)
            return

        raw = self.rfile.read(content_length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return

        question_id = body.get("question_id", "")
        answer = body.get("answer", "")

        if not question_id or not answer:
            self._send_json(
                {"error": "question_id and answer are required"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        if len(answer) > 4096:
            self._send_json(
                {"error": "answer too long (max 4096 chars)"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        state = _check_state
        if state is None:
            self._send_json({"error": "server not configured"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        result = state.check(question_id, answer)
        self._send_json(result, HTTPStatus.OK)

    def _send_json(self, data: dict[str, Any], status: int) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress default stderr logging."""
        pass


# ---------------------------------------------------------------------------
# CLI entry point — intended to run inside a container via ``python -m``
# ---------------------------------------------------------------------------

def _load_state_from_env() -> CheckState:
    question_id = os.environ.get("QUESTION_ID", "")
    answer_sha256 = os.environ.get("ANSWER_SHA256", "")
    max_checks = int(os.environ.get("MAX_CHECKS", "3"))
    benchmark = os.environ.get("BENCHMARK", "")
    artifact_path = os.environ.get("ARTIFACT_PATH", "")

    if not question_id or not answer_sha256:
        raise ValueError("QUESTION_ID and ANSWER_SHA256 environment variables are required")

    return CheckState(
        question_id=question_id,
        answer_sha256=answer_sha256,
        max_checks=max_checks,
        benchmark=benchmark,
        artifact_path=artifact_path,
    )


def main() -> None:
    global _check_state  # noqa: PLW0603
    _check_state = _load_state_from_env()

    port = 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), _CheckHandler)
    print(f"check-server listening on 0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()