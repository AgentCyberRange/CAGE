"""Tests for the in-container submit CLI client."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cage.target.services.submit.client import format_result, submit_candidate
from cage.target.services.submit.server import (
    SubmitState,
    build_unix_server,
    normalize_answer,
    sha256_hex,
)


def _state(answer: str = "flag{client}") -> SubmitState:
    return SubmitState(
        question_id="q1",
        answer_sha256=sha256_hex(normalize_answer(answer)),
        max_checks=3,
    )


@pytest.fixture
def submit_socket(tmp_path):
    socket_path = tmp_path / "submit.sock"
    server = build_unix_server(str(socket_path), _state())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield str(socket_path)
    finally:
        server.shutdown()
        server.server_close()


def test_submit_candidate_returns_server_result(submit_socket):
    result = submit_candidate("flag{client}", socket_path=submit_socket)

    assert result["correct"] is True
    assert result["checks_remaining"] == 2


def test_submit_client_is_directly_executable():
    client_path = (
        Path(__file__).resolve().parents[1]
        / "cage"
        / "target"
        / "services"
        / "submit"
        / "client.py"
    )

    assert client_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3")


def test_submit_candidate_rejects_empty_answer(submit_socket):
    with pytest.raises(ValueError, match="candidate flag"):
        submit_candidate("", socket_path=submit_socket)


def test_submit_candidate_reports_unavailable_service(tmp_path):
    missing_socket = tmp_path / "missing.sock"

    with pytest.raises(ConnectionError, match="submit service is not available"):
        submit_candidate("flag{x}", socket_path=str(missing_socket))


def test_format_result_for_correct_answer():
    result = {"correct": True, "checks_remaining": 1}

    assert format_result(result) == "Correct. Checks remaining: 1"


def test_format_result_for_wrong_answer():
    result = {"correct": False, "message": "Answer not accepted.", "checks_remaining": 2}

    assert format_result(result) == "Incorrect. Checks remaining: 2"


def test_format_result_for_no_checks_remaining():
    result = {"correct": False, "message": "No checks remaining.", "checks_remaining": 0}

    assert format_result(result) == "No checks remaining. Checks remaining: 0"
