"""Target teardown outcome contracts.

These tests keep target cleanup from drifting back into a best-effort
fire-and-forget call.  The orchestrator needs a small, explicit outcome object
so ResourceLedger can distinguish "confirmed released" from "cleanup failed"
or "cleanup was requested but not proven".
"""

from __future__ import annotations

import logging
from typing import Any

from cage.target.client import (
    BackendStrategy,
    ChallengeClient,
    ChallengeClientConfig,
)


class _OutcomeBackend(BackendStrategy):
    """Backend stub that can prove success, fail, or stay legacy/unknown."""

    def __init__(
        self,
        config: ChallengeClientConfig,
        logger: logging.Logger,
        *,
        teardown_return: object = True,
        teardown_error: Exception | None = None,
    ):
        super().__init__(config, logger)
        self.teardown_return = teardown_return
        self.teardown_error = teardown_error
        self.calls: list[dict[str, Any]] = []

    def initialize(
        self,
        challenge_id: str,
        metadata: dict[str, Any],
        force_recreate: bool = False,
        runtime_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "initialize": challenge_id,
                "force_recreate": force_recreate,
                "runtime_args": dict(runtime_args or {}),
            }
        )
        return {
            "id": challenge_id,
            "type": "dynamic",
            "run_id": f"{challenge_id}-run",
            "project_name": f"project-{challenge_id}",
            "network_name": f"net-{challenge_id}",
            "network_subnet": "10.201.1.0/26",
            "network_gateway": "10.201.1.1",
            "scoring": {},
            "debug": {},
            "services": {},
        }

    def teardown(self, challenge_id: str, run_id: str | None = None) -> object:
        self.calls.append({"teardown": challenge_id, "run_id": run_id})
        if self.teardown_error is not None:
            raise self.teardown_error
        return self.teardown_return

    def validate_connectivity(self, challenge_id: str, record: dict[str, Any]) -> bool:
        return True

    def handle_crash(self, challenge_id: str, observation: str) -> tuple[str, bool]:
        return observation, True


def _client_with_backend(backend: _OutcomeBackend) -> ChallengeClient:
    client = ChallengeClient(
        backend.config,
        logger=logging.getLogger("test_target_teardown"),
    )
    client.backend = backend
    return client


def test_finish_challenge_returns_released_outcome_and_clears_runtime_state() -> None:
    cfg = ChallengeClientConfig(challenges={"web-1": {"benchmark_family": "web"}})
    backend = _OutcomeBackend(cfg, logging.getLogger("test_target_teardown"))
    client = _client_with_backend(backend)

    launched = client.get_challenge_data("web-1")
    launched_run_id = launched["runtime"]["run_id"]
    result = client.finish_challenge("web-1")

    assert backend.calls[-1] == {"teardown": "web-1", "run_id": launched_run_id}
    assert result.challenge_id == "web-1"
    assert result.run_id == launched_run_id
    assert result.requested is True
    assert result.succeeded is True
    assert result.status == "released"
    assert result.error is None
    assert client.challenges["web-1"]["target_status"] == "stopped"
    assert client.challenges["web-1"]["runtime"] == {}


def test_finish_challenge_reports_cleanup_failure_without_raising() -> None:
    cfg = ChallengeClientConfig(challenges={"web-1": {"benchmark_family": "web"}})
    backend = _OutcomeBackend(
        cfg,
        logging.getLogger("test_target_teardown"),
        teardown_error=RuntimeError("target delete failed"),
    )
    client = _client_with_backend(backend)
    launched = client.get_challenge_data("web-1")
    launched_run_id = launched["runtime"]["run_id"]

    result = client.finish_challenge("web-1")

    assert backend.calls[-1] == {"teardown": "web-1", "run_id": launched_run_id}
    assert result.challenge_id == "web-1"
    assert result.run_id == launched_run_id
    assert result.requested is True
    assert result.succeeded is False
    assert result.status == "cleanup_failed"
    assert "target delete failed" in (result.error or "")
    assert client.challenges["web-1"]["target_status"] == "stopped"


def test_finish_challenge_preserves_unknown_outcome_for_legacy_backends() -> None:
    cfg = ChallengeClientConfig(challenges={"web-1": {"benchmark_family": "web"}})
    backend = _OutcomeBackend(
        cfg,
        logging.getLogger("test_target_teardown"),
        teardown_return=None,
    )
    client = _client_with_backend(backend)
    launched = client.get_challenge_data("web-1")
    launched_run_id = launched["runtime"]["run_id"]

    result = client.finish_challenge("web-1")

    assert result.challenge_id == "web-1"
    assert result.run_id == launched_run_id
    assert result.requested is True
    assert result.succeeded is None
    assert result.status == "cleanup_requested"
    assert result.error is None
