"""Lifecycle manager for in-container submit services."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cage.target.services.submit.server import normalize_answer, sha256_hex

if TYPE_CHECKING:
    from cage.benchmarks import Benchmark

logger = logging.getLogger(__name__)

SUBMIT_SERVER_PATH = "/opt/cage-submit/submit_server.py"
SUBMIT_CLIENT_PATH = "/usr/local/bin/submit"
SUBMIT_SOCKET_PATH = "/run/cage-submit/submit.sock"


def _submit_server_host_path() -> Path:
    return Path(__file__).with_name("server.py").resolve()


def _submit_client_host_path() -> Path:
    return Path(__file__).with_name("client.py").resolve()


def needs_submit_service(benchmark: Benchmark, live_check_enabled: bool) -> bool:
    """Return True when this benchmark should use in-container submit."""
    return live_check_enabled and benchmark.needs_submit_service


@dataclass
class SubmitServiceHandle:
    """Handle for a running in-container submit daemon."""

    process: subprocess.Popen[str]
    container: Any
    trial_id: str

    def stop(self) -> None:
        """Stop the submit daemon and remove trial-local submit files."""
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass

        try:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)
        except Exception:
            pass

        _cleanup_submit_files(self.container, self.trial_id)


def start_submit_service(
    *,
    container: Any,
    question_id: str,
    expected_answer: str,
    max_checks: int,
    benchmark: str,
    trial_artifact_dir: Path,
    trial_id: str,
    container_artifact_path: str = "",
) -> SubmitServiceHandle:
    """Install and start a root-owned submit daemon inside an agent container."""
    trial_artifact_dir.mkdir(parents=True, exist_ok=True)

    _install_submit_files(container)

    cmd = [
        "docker",
        "exec",
        "-i",
        "-u",
        "root",
        container.name,
        "python3",
        SUBMIT_SERVER_PATH,
    ]
    process = subprocess.Popen(
        cmd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    init = {
        "question_id": question_id,
        "answer_sha256": sha256_hex(normalize_answer(expected_answer)),
        "max_checks": max_checks,
        "benchmark": benchmark,
        "artifact_path": container_artifact_path,
    }
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("submit service failed to open stdin/stdout pipes")
    process.stdin.write(json.dumps(init, ensure_ascii=False) + "\n")
    process.stdin.flush()

    ready = process.stdout.readline().strip()
    if ready != "READY":
        _terminate_failed_process(process)
        _cleanup_submit_files(container, trial_id)
        raise RuntimeError(f"submit service did not become ready: {ready}")

    logger.info("Submit service started for trial %s (q=%s)", trial_id, question_id)
    return SubmitServiceHandle(process=process, container=container, trial_id=trial_id)


def _install_submit_files(container: Any) -> None:
    server_host_path = str(_submit_server_host_path())
    client_host_path = str(_submit_client_host_path())

    container.exec(
        "rm -rf /opt/cage-submit /run/cage-submit && mkdir -p /opt/cage-submit",
        timeout=10.0,
    )
    server_copy = container.copy_to(server_host_path, SUBMIT_SERVER_PATH)
    if getattr(server_copy, "exit_code", 0) != 0:
        raise RuntimeError(
            f"failed to copy submit server: {getattr(server_copy, 'stderr', '')[:500]}"
        )
    client_copy = container.copy_to(client_host_path, SUBMIT_CLIENT_PATH)
    if getattr(client_copy, "exit_code", 0) != 0:
        raise RuntimeError(
            f"failed to copy submit client: {getattr(client_copy, 'stderr', '')[:500]}"
        )

    result = container.exec(
        " && ".join(
            [
                "chown root:root /opt/cage-submit",
                "chmod 0700 /opt/cage-submit",
                f"chown root:root {SUBMIT_SERVER_PATH}",
                f"chmod 0500 {SUBMIT_SERVER_PATH}",
                f"chown root:root {SUBMIT_CLIENT_PATH}",
                f"chmod 0755 {SUBMIT_CLIENT_PATH}",
            ]
        ),
        timeout=10.0,
    )
    if getattr(result, "exit_code", 0) != 0:
        raise RuntimeError(f"failed to secure submit files: {getattr(result, 'stderr', '')[:500]}")


def _terminate_failed_process(process: subprocess.Popen[str]) -> None:
    try:
        if process.stdin:
            process.stdin.close()
    except Exception:
        pass
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5.0)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _cleanup_submit_files(container: Any, trial_id: str) -> None:
    try:
        container.exec(
            f"rm -rf /run/cage-submit /opt/cage-submit && rm -f {SUBMIT_CLIENT_PATH}",
            timeout=10.0,
        )
    except Exception as exc:
        logger.warning("Failed to clean submit service files for %s: %s", trial_id, exc)
