"""In-container submit server for live flag checks.

This module is copied into the agent runtime container and run as root. It
listens on a Unix domain socket and checks submitted candidate flags against a
single expected-answer hash for the current trial.
"""

from __future__ import annotations

import grp
import hashlib
import json
import os
import pwd
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_SOCKET_PATH = "/run/cage-submit/submit.sock"
MAX_PAYLOAD_BYTES = 65536
MAX_ANSWER_CHARS = 4096


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    return answer.strip()


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SubmitState:
    """Mutable submit state for one trial."""

    def __init__(
        self,
        *,
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
        self._lock = threading.Lock()

    def submit(self, answer: str) -> dict[str, Any]:
        """Check a submitted answer and return a result dictionary."""
        if len(answer) > MAX_ANSWER_CHARS:
            return {
                "correct": False,
                "message": f"Answer too long (max {MAX_ANSWER_CHARS} chars).",
                "checks_remaining": self.checks_remaining,
            }

        with self._lock:
            if self.checks_remaining <= 0:
                return {
                    "correct": False,
                    "message": "No checks remaining.",
                    "checks_remaining": 0,
                }

            self.checks_remaining -= 1
            candidate_hash = sha256_hex(normalize_answer(answer))
            correct = candidate_hash == self.answer_sha256
            result = {
                "correct": correct,
                "message": "Answer accepted." if correct else "Answer not accepted.",
                "checks_remaining": self.checks_remaining,
            }
            self._log_artifact(candidate_hash, correct=correct)
            return result

    def _log_artifact(self, candidate_hash: str, *, correct: bool) -> None:
        if not self.artifact_path:
            return
        try:
            artifact_file = Path(self.artifact_path)
            artifact_file.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts_ms": int(time.time() * 1000),
                "benchmark": self.benchmark,
                "question_id": self.question_id,
                "correct": correct,
                "checks_remaining": self.checks_remaining,
                "answer_hash": f"sha256:{candidate_hash}",
                "source": "container-submit",
            }
            with artifact_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


class _SubmitHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_PAYLOAD_BYTES + 1)
        if len(raw) > MAX_PAYLOAD_BYTES:
            self._send(
                {
                    "correct": False,
                    "message": "payload too large",
                    "checks_remaining": self.server.state.checks_remaining,
                }
            )
            return

        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(
                {
                    "correct": False,
                    "message": "invalid JSON",
                    "checks_remaining": self.server.state.checks_remaining,
                }
            )
            return

        answer = body.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            self._send(
                {
                    "correct": False,
                    "message": "candidate flag is required",
                    "checks_remaining": self.server.state.checks_remaining,
                }
            )
            return

        self._send(self.server.state.submit(answer))

    def _send(self, data: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n")


class SubmitUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Threaded Unix socket server carrying a SubmitState."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: str, state: SubmitState) -> None:
        self.socket_path = socket_path
        self.state = state
        Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        super().__init__(socket_path, _SubmitHandler)

    def server_close(self) -> None:
        super().server_close()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


def build_unix_server(socket_path: str, state: SubmitState) -> SubmitUnixServer:
    """Build a submit Unix socket server for tests and the CLI entry point."""
    return SubmitUnixServer(socket_path, state)


def _agent_gid() -> int | None:
    try:
        return pwd.getpwnam("agent").pw_gid
    except KeyError:
        try:
            return grp.getgrnam("agent").gr_gid
        except KeyError:
            return None


def _secure_socket_permissions(socket_path: str) -> None:
    gid = _agent_gid()
    socket_dir = Path(socket_path).parent
    if gid is not None and hasattr(os, "chown"):
        os.chown(socket_dir, 0, gid)
        os.chown(socket_path, 0, gid)
    os.chmod(socket_dir, 0o710)
    os.chmod(socket_path, 0o660 if gid is not None else 0o600)


def _load_state_from_stdin() -> SubmitState:
    line = sys.stdin.readline()
    if not line:
        raise ValueError("submit server init payload is required on stdin")
    payload = json.loads(line)
    question_id = str(payload.get("question_id", "")).strip()
    answer_sha256 = str(payload.get("answer_sha256", "")).strip()
    max_checks = int(payload.get("max_checks", 3))
    benchmark = str(payload.get("benchmark", ""))
    artifact_path = str(payload.get("artifact_path", ""))
    if not question_id or not answer_sha256:
        raise ValueError("question_id and answer_sha256 are required")
    return SubmitState(
        question_id=question_id,
        answer_sha256=answer_sha256,
        max_checks=max_checks,
        benchmark=benchmark,
        artifact_path=artifact_path,
    )


def _watch_stdin_for_shutdown(server: SubmitUnixServer) -> None:
    try:
        for line in sys.stdin:
            if line.strip().upper() == "STOP":
                break
    finally:
        server.shutdown()


def main() -> None:
    state = _load_state_from_stdin()
    socket_path = os.environ.get("CAGE_SUBMIT_SOCKET", DEFAULT_SOCKET_PATH)
    server = build_unix_server(socket_path, state)
    _secure_socket_permissions(socket_path)
    watcher = threading.Thread(target=_watch_stdin_for_shutdown, args=(server,), daemon=True)
    watcher.start()
    print("READY", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
