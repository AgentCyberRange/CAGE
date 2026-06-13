"""Check service manager — lifecycle for Docker-based live-check containers.

For benchmarks that declare ``needs_check_service``, a lightweight ``check``
container runs alongside the agent in the same Docker network.  The agent
submits candidate answers to ``http://check:8080/check`` and receives a boolean
verdict.

This module handles:
  - Starting the check container on a Docker network provided by the orchestrator
  - Mounting the lightweight check server into the Python runtime image
  - Stopping the check container
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cage.sandbox.exec import ExecResult
from cage.target.services.check.server import normalize_answer, sha256_hex

if TYPE_CHECKING:
    from cage.benchmarks import Benchmark

logger = logging.getLogger(__name__)

# The image used for the check container.  ``cage/check-server:latest`` should
# be built from a minimal Dockerfile that installs httpx and runs
# ``python -m cage.target.services.check.server``.  For the initial implementation we
# fall back to running the module inline via ``python:3.11-slim``.
CHECK_SERVER_IMAGE = "python:3.11-slim"


def _run_docker(cmd: list[str], *, timeout: float = 30.0) -> ExecResult:
    """Run a local Docker CLI command and return an ExecResult."""
    started = int(time.time() * 1000)
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = f"Timed out after {timeout}s"
        exit_code = -1

    ended = int(time.time() * 1000)
    return ExecResult(
        command=" ".join(cmd),
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_ms=ended - started,
    )


@dataclass
class CheckServiceHandle:
    """Handle for a running check container and its associated resources."""

    container_name: str
    network_name: str
    artifact_path: Path

    def stop(self) -> None:
        """Stop and remove the check container."""
        result = _run_docker(["docker", "rm", "-f", self.container_name], timeout=15.0)
        if result.exit_code != 0:
            logger.warning(
                "Failed to remove check container %s: %s",
                self.container_name, result.stderr[:200],
            )
        else:
            logger.info("Check container removed: %s", self.container_name)


def _check_server_script_path() -> Path:
    """Return the host path to the script mounted into the check container."""
    return Path(__file__).with_name("server.py").resolve()


def start_check_service(
    *,
    question_id: str,
    expected_answer: str,
    max_checks: int,
    benchmark: str,
    trial_artifact_dir: Path,
    run_id: str = "",
    trial_id: str = "",
    network_name: str | None = None,
    image: str = CHECK_SERVER_IMAGE,
) -> CheckServiceHandle:
    """Start a check container for a trial.

    Args:
        question_id: The challenge/sample ID that must match in check requests.
        expected_answer: The expected answer (plaintext); will be hashed internally.
        max_checks: Maximum number of check attempts allowed.
        benchmark: Benchmark name for artifact logging.
        trial_artifact_dir: Host-side directory for live_checks.jsonl.
        run_id: Run identifier for container/network naming.
        trial_id: Trial identifier for container/network naming.
        network_name: Existing Docker network to join. The orchestrator creates
            a dedicated check network when no runtime network exists.
        image: Docker image to use for the check container.

    Returns:
        A CheckServiceHandle that must be stopped when the trial ends.
    """
    if not network_name:
        raise ValueError("network_name is required to start the check service")

    container_name = f"cage-check-{run_id}-{trial_id}"
    answer_hash = sha256_hex(normalize_answer(expected_answer))

    # Ensure artifact dir exists on host
    artifact_host_dir = str(trial_artifact_dir.resolve())
    Path(artifact_host_dir).mkdir(parents=True, exist_ok=True)
    check_server_host_path = str(_check_server_script_path())

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", container_name,
        "--network", network_name,
        "--network-alias", "check",
        "-e", f"QUESTION_ID={question_id}",
        "-e", f"ANSWER_SHA256={answer_hash}",
        "-e", f"MAX_CHECKS={max_checks}",
        "-e", f"BENCHMARK={benchmark}",
        "-e", "ARTIFACT_PATH=/artifacts",
        "--mount",
        f"type=bind,source={artifact_host_dir},target=/artifacts",
        "--mount",
        f"type=bind,source={check_server_host_path},target=/check_server.py,readonly",
    ]

    cmd.extend([
        image,
        "python", "/check_server.py",
    ])

    result = _run_docker(cmd, timeout=60.0)
    if result.exit_code != 0:
        raise RuntimeError(
            f"Failed to start check container {container_name}: {result.stderr[:500]}"
        )

    logger.info(
        "Check container started: %s (network=%s, alias=check, q=%s, max_checks=%d)",
        container_name, network_name, question_id, max_checks,
    )

    return CheckServiceHandle(
        container_name=container_name,
        network_name=network_name,
        artifact_path=trial_artifact_dir,
    )


# ---------------------------------------------------------------------------
# Benchmark → check behaviour mapping (declarative; the benchmark decides)
# ---------------------------------------------------------------------------


def needs_check_container(benchmark: Benchmark, live_check_enabled: bool) -> bool:
    """Return True if this benchmark + config requires a check container."""
    return live_check_enabled and benchmark.needs_check_service


def has_builtin_check(benchmark: Benchmark, live_check_enabled: bool) -> bool:
    """Return True if this benchmark has built-in checking (prompt-only change)."""
    return live_check_enabled and benchmark.uses_builtin_check
