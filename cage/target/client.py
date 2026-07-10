"""Challenge lifecycle client with pluggable backends.

The host-side client that talks to the target_server target service (either the
embedded ``cage serve`` running locally, or a remote deployment).
``ChallengeClient.get_challenge_data`` launches a target, ``finish_challenge``
tears it down. Two backends:

  - ``RemoteBackend`` — HTTP to a target_server server (the supported path).
  - ``LocalBackend``  — placeholder that shells out to ``docker compose``
    directly without target_server; currently incomplete and unused by the
    bundled benchmarks.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from cage.target.scope import normalize_target_scope, resolve_target_scope

# Guarded imports for optional dependencies
try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    from sshtunnel import SSHTunnelForwarder
except ImportError:
    SSHTunnelForwarder = None  # type: ignore[assignment,misc]


# ==========================================
# Configuration classes
# ==========================================

@dataclass
class SSHConfig:
    """SSH jump host configuration.

    All fields default to empty; SSH tunnelling is opt-in. Users that need
    tunnelling must set ``jump_host``/``jump_user``/``ssh_key_path`` in
    ``project.yml`` under ``target.ssh:``.
    """

    jump_host: str = ""
    jump_user: str = ""
    ssh_key_path: str = ""
    remote_bind_address: str = "127.0.0.1"
    remote_bind_port: int = 8000


@dataclass
class ChallengeClientConfig:
    """Environment configuration."""
    # Challenge data (populated externally via ChallengeJsonAdapter)
    challenges: Dict[str, Any] = field(default_factory=dict)

    # Mode: 'local' | 'remote'
    run_mode: str = "remote"

    # Local mode configuration
    network_name: str = "cage_net"
    local_port_range: range = field(default_factory=lambda: range(20000, 30000))

    # Remote mode configuration — defaults to the embedded ``cage serve`` running
    # locally. Override via ``target.server_url`` in project.yml to point at a remote
    # target_server instance.
    server_url: str = "http://127.0.0.1:8000"
    use_ssh_tunnel: bool = False
    ssh_config: SSHConfig = field(default_factory=SSHConfig)

    # Internal container access only (Alias/inner-IP:Port). Host-published external
    # access (True) is forbidden — config resolution raises on it (see
    # experiment.py) so a target is never reachable from the host.
    use_external_access: bool = False

    # Agent-side host IP for the docker bridge. Empty by default — set this
    # explicitly when the agent container must reach a specific gateway IP
    # rather than ``host.docker.internal``.
    host_ip_for_agent: str = ""

    # Cage-side run identifier. Passed to the target_server server on every
    # ``/launch/<chal>`` so the server stamps ``cage.run_id=<run_id>`` on
    # every container and network it creates for this run. Used by
    # ``close()`` to drive the one-shot ``DELETE /run/<run_id>`` fast path.
    cage_run_id: str = ""

    # HTTP timeout for the synchronous ``GET /launch/<challenge>`` request.
    # The server-side launch can include compose build, compose up, and
    # readiness waits, so cage run must raise this when it raises those caps.
    launch_timeout_s: float = 300.0


@dataclass(frozen=True)
class TargetTeardownResult:
    """Outcome of one target teardown request.

    The old client API treated teardown as a fire-and-forget side effect. That
    was not enough for canonical ResourceLedger records: a ledger needs to know
    whether a target was actually released, failed cleanup, or merely had a
    cleanup request issued by a legacy backend that cannot prove the result.

    ``succeeded`` uses three values on purpose:
    - ``True`` means the backend confirmed deletion/release.
    - ``False`` means the backend attempted cleanup and observed a failure.
    - ``None`` means cleanup was requested but the backend did not report a
      proof either way, so callers must keep the status as
      ``cleanup_requested`` instead of guessing.
    """

    challenge_id: str
    run_id: str | None = None
    requested: bool = True
    succeeded: bool | None = None
    error: str | None = None
    backend: str = ""
    # Container logs captured server-side just before purge (audit trail).
    # Excluded from eq/hash so the frozen dataclass stays hashable despite
    # holding a list.
    container_logs: list = field(default_factory=list, compare=False, hash=False)

    @property
    def status(self) -> str:
        """ResourceLedger status implied by this teardown outcome."""
        if self.succeeded is True:
            return "released"
        if self.succeeded is False:
            return "cleanup_failed"
        return "cleanup_requested"

    def with_context(
        self,
        *,
        challenge_id: str,
        run_id: str | None,
        backend: str,
    ) -> "TargetTeardownResult":
        """Fill missing context from the caller without changing evidence."""
        return replace(
            self,
            challenge_id=self.challenge_id or challenge_id,
            run_id=self.run_id if self.run_id not in (None, "") else run_id,
            backend=self.backend or backend,
        )


def _coerce_teardown_result(
    raw_result: object,
    *,
    challenge_id: str,
    run_id: str | None,
    backend: str,
) -> TargetTeardownResult:
    """Normalize backend teardown return values into the explicit contract.

    Backend implementations may already return ``TargetTeardownResult``. Older
    in-tree or out-of-tree implementations may return ``None`` because teardown
    used to be side-effect only. Boolean returns are accepted as the smallest
    useful proof: ``True`` means released, ``False`` means failed.
    """
    if isinstance(raw_result, TargetTeardownResult):
        return raw_result.with_context(
            challenge_id=challenge_id,
            run_id=run_id,
            backend=backend,
        )
    if isinstance(raw_result, bool):
        return TargetTeardownResult(
            challenge_id=challenge_id,
            run_id=run_id,
            requested=True,
            succeeded=raw_result,
            error=None if raw_result else "backend returned False",
            backend=backend,
        )
    if raw_result is None:
        return TargetTeardownResult(
            challenge_id=challenge_id,
            run_id=run_id,
            requested=True,
            succeeded=None,
            backend=backend,
        )
    return TargetTeardownResult(
        challenge_id=challenge_id,
        run_id=run_id,
        requested=True,
        succeeded=None,
        error=f"unrecognized teardown result type: {type(raw_result).__name__}",
        backend=backend,
    )


# ==========================================
# Backend interface
# ==========================================

class BackendStrategy(ABC):
    def __init__(self, config: ChallengeClientConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def initialize(
        self,
        challenge_id: str,
        metadata: Dict,
        force_recreate: bool = False,
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def teardown(self, challenge_id: str, run_id: str | None = None) -> TargetTeardownResult:
        pass

    @abstractmethod
    def validate_connectivity(self, challenge_id: str, record: Dict) -> bool:
        pass

    @abstractmethod
    def handle_crash(self, challenge_id: str, observation: str) -> tuple[str, bool]:
        pass

    def cleanup(self):
        """Cleanup on exit."""
        pass


# ==========================================
# Implementation 1: Local Backend
# ==========================================

class LocalBackend(BackendStrategy):
    def __init__(self, config: ChallengeClientConfig, logger: logging.Logger):
        super().__init__(config, logger)
        self._ensure_network()

    def initialize(
        self,
        challenge_id: str,
        metadata: Dict,
        force_recreate: bool = False,
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del force_recreate
        del runtime_args
        work_dir = Path(metadata["full_path"])

        # Static challenge
        if not (work_dir / "docker-compose.yml").exists():
            self.logger.info(f"[Local] Static challenge detected, skipping Docker setup. {work_dir}")
            return {
                "id": challenge_id,
                "type": "static",
                "work_dir": str(work_dir),
                "files": metadata.get("files", [])
            }

        # Dynamic challenge
        self.logger.info(f"[Local] Starting Docker for {challenge_id}...")
        return self._start_docker(challenge_id, work_dir)

    def teardown(self, challenge_id: str, run_id: str | None = None) -> TargetTeardownResult:
        del run_id
        project_name = f"cage_net_{challenge_id}"
        self.logger.info(f"[Local] Stopping {project_name}...")
        try:
            result = subprocess.run(
                ["docker", "compose", "-p", project_name, "down", "-v"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            return TargetTeardownResult(
                challenge_id=challenge_id,
                requested=True,
                succeeded=False,
                error=str(exc),
                backend="local",
            )
        return TargetTeardownResult(
            challenge_id=challenge_id,
            requested=True,
            succeeded=result.returncode == 0,
            error=None if result.returncode == 0 else f"docker compose down exited {result.returncode}",
            backend="local",
        )

    def validate_connectivity(self, challenge_id: str, record: Dict) -> bool:
        if record['type'] == 'static':
            return True
        for svc in record.get('services', {}).values():
            if not self._check_socket('127.0.0.1', svc['port']):
                return False
        return True

    def handle_crash(self, challenge_id: str, observation: str) -> tuple[str, bool]:
        self.teardown(challenge_id)
        return f"{observation}\n[System] Local container crashed.", False

    def _start_docker(self, challenge_id: str, work_dir: Path) -> Dict:
        project_name = f"cage_net_{challenge_id}"
        cmd = ["docker", "compose", "-p", project_name, "up", "-d"]
        subprocess.run(cmd, cwd=work_dir, check=True)

        services_info = {}
        # TODO: inspect containers for port mappings
        services_info["target"] = {"host": "127.0.0.1", "port": 20000}

        return {
            "id": challenge_id,
            "type": "dynamic",
            "project_name": project_name,
            "services": services_info
        }

    def _ensure_network(self):
        result = subprocess.run(
            ["docker", "network", "inspect", self.config.network_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            subprocess.run(
                ["docker", "network", "create", self.config.network_name],
                capture_output=True, text=True,
            )
            self.logger.info(f"[Local] Created network {self.config.network_name}")

    def _check_socket(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, int(port))) == 0


# ==========================================
# Implementation 2: Remote Backend (target_server HTTP API)
# ==========================================

class RemoteBackend(BackendStrategy):
    def __init__(self, config: ChallengeClientConfig, logger: logging.Logger):
        super().__init__(config, logger)
        self.tunnel: Optional[Any] = None
        self.api_base_url = config.server_url

        # SSH tunnel for API control flow
        if config.use_ssh_tunnel:
            self._start_control_tunnel()

        # Service tunnel pool: { chal_id: [tunnel_obj, ...] }
        self.service_tunnels: Dict[str, list] = {}

    def _start_control_tunnel(self):
        if SSHTunnelForwarder is None:
            raise ImportError(
                "sshtunnel is required for SSH tunnel mode. "
                "Install with: pip install sshtunnel"
            )
        ssh_cfg = self.config.ssh_config
        self.logger.info(f"[SSH] Opening control tunnel via {ssh_cfg.jump_user}@{ssh_cfg.jump_host}...")

        self.tunnel = SSHTunnelForwarder(
            (ssh_cfg.jump_host, 22),
            ssh_username=ssh_cfg.jump_user,
            ssh_pkey=ssh_cfg.ssh_key_path,
            remote_bind_address=(ssh_cfg.remote_bind_address, ssh_cfg.remote_bind_port),
            local_bind_address=('0.0.0.0', 0)
        )
        self.tunnel.start()
        self.api_base_url = f"http://127.0.0.1:{self.tunnel.local_bind_port}"
        self.logger.info(f"[SSH] Tunnel established! API mapped to {self.api_base_url}")

    def initialize(
        self,
        challenge_id: str,
        metadata: Dict,
        force_recreate: bool = False,
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if requests is None:
            raise ImportError(
                "requests is required for remote backend. "
                "Install with: pip install requests"
            )

        url = f"{self.api_base_url}/launch/{challenge_id}"

        params: Dict[str, str] = {}
        if force_recreate:
            params["force_recreate"] = "true"
        parallel_mode = str((runtime_args or {}).get("parallel_mode", "") or "").strip().lower()
        if parallel_mode:
            params["parallel_mode"] = parallel_mode
        target_scope = normalize_target_scope((runtime_args or {}).get("target_scope"))
        if target_scope:
            params["target_scope"] = target_scope
        network_mode = str((runtime_args or {}).get("network_mode", "") or "").strip().lower()
        if network_mode:
            params["network_mode"] = network_mode
        exposure_mode = str((runtime_args or {}).get("exposure_mode", "") or "").strip().lower()
        if exposure_mode:
            params["exposure_mode"] = exposure_mode
        if self.config.cage_run_id:
            params["cage_run_id"] = self.config.cage_run_id
        if not params:
            params = None  # type: ignore[assignment]

        resp = requests.get(url, params=params, timeout=self.config.launch_timeout_s)
        if not resp.ok:
            # ``resp.raise_for_status()`` drops the response body, which is
            # where ``_launch_challenge_impl`` puts the actual root cause
            # (docker compose stderr, subnet pool exhausted, health probe
            # failures, …). The server now returns a structured detail
            # ``{error, project_name, containers}`` where ``containers`` holds
            # the full per-container logs. Pull the logs out as structured data
            # (attached to the exception for the orchestrator to persist) and
            # keep only a concise one-liner in the error string — otherwise the
            # verbose log dump gets smashed into a truncated string and the real
            # cause is lost.
            body = (resp.text or "").strip()
            concise = body
            container_logs = None
            try:
                detail = resp.json().get("detail")
                if isinstance(detail, dict):
                    container_logs = detail.get("containers")
                    concise = str(detail.get("error") or "").strip() or body
                    proj = detail.get("project_name")
                    if proj:
                        concise = f"{concise} (project={proj})"
            except Exception:
                pass
            if len(concise) > 2000:
                concise = concise[:2000] + "...(truncated)"
            err = requests.HTTPError(
                f"{resp.status_code} {resp.reason} for url: {resp.url}"
                + (f"\nbody: {concise}" if concise else ""),
                response=resp,
            )
            if container_logs is not None:
                # Carried out-of-band so get_challenge_data can persist it
                # without parsing the (concise) error string.
                err.target_container_logs = container_logs
            raise err
        data = resp.json()

        status = data.get("status")

        # Static challenge
        if status == "static":
            return {
                "id": challenge_id,
                "type": "static",
                "files": metadata.get("files", []),
                "work_dir": str(metadata.get("full_path"))
            }

        # Dynamic challenge
        if status in ["launched", "reused", "recreated"]:
            raw_services = data.get("services", [])
            processed_services = self._process_services(challenge_id, raw_services)

            return {
                "id": challenge_id,
                "type": "dynamic",
                "status": status,
                "run_id": data.get("run_id"),
                "project_name": data.get("project_name"),
                "network_name": data.get("network_name"),
                "network_subnet": data.get("network_subnet"),
                "network_gateway": data.get("network_gateway"),
                "scoring": data.get("scoring", {}),
                "debug": dict(data.get("debug", {}) or {}),
                "services": processed_services
            }

        raise RuntimeError(f"Unknown status from target_server: {status}")

    def _process_services(self, chal_id: str, raw_services: List[Dict]) -> Dict:
        """Process service info from target_server, optionally setting up SSH tunnels."""
        processed = {}

        # Clear old tunnels for this challenge
        self._clear_service_tunnels(chal_id)
        current_tunnels = []

        for svc in raw_services:
            svc_name = svc["service_name"]
            processed_service = dict(svc)
            inner_host = processed_service.get("inner_host") or processed_service.get("inner_ip") or processed_service.get("alias")
            inner_port = processed_service.get("inner_port")
            if inner_port is None:
                inner_port = processed_service.get("internal_port")
            external_host = processed_service.get("external_host") or processed_service.get("ip")
            external_port = processed_service.get("external_port")

            if self.config.use_external_access:
                remote_host = external_host
                remote_port = external_port
                final_host = remote_host
                final_port = remote_port

                if remote_port:
                    if self.config.use_ssh_tunnel:
                        if SSHTunnelForwarder is None:
                            raise ImportError("sshtunnel required for SSH tunnel mode")
                        try:
                            svc_tunnel = SSHTunnelForwarder(
                                (self.config.ssh_config.jump_host, 22),
                                ssh_username=self.config.ssh_config.jump_user,
                                ssh_pkey=self.config.ssh_config.ssh_key_path,
                                remote_bind_address=(remote_host, remote_port),
                                local_bind_address=('0.0.0.0', 0)
                            )
                            svc_tunnel.start()
                            current_tunnels.append(svc_tunnel)

                            final_host = self.config.host_ip_for_agent
                            final_port = svc_tunnel.local_bind_port

                            self.logger.info(
                                f"[SSH] Forwarding Service {svc_name}: "
                                f"0.0.0.0:{final_port} (Agent use {final_host}) "
                                f"-> {remote_host}:{remote_port}"
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to tunnel service {svc_name}: {e}")
                    processed_service["external_host"] = final_host
                    processed_service["external_port"] = final_port
                processed_service["host"] = final_host
                processed_service["port"] = final_port
                if final_host and final_port is not None:
                    processed_service["url"] = f"http://{final_host}:{final_port}"
                    processed_service["netcat"] = f"nc {final_host} {final_port}"
                processed[svc_name] = processed_service
            else:
                # Internal mode: return alias
                final_host = inner_host
                final_port = inner_port
                processed_service["host"] = final_host
                processed_service["port"] = final_port
                if final_host and final_port is not None:
                    processed_service["url"] = f"http://{final_host}:{final_port}"
                    processed_service["netcat"] = f"nc {final_host} {final_port}"
                processed[svc_name] = processed_service

        if current_tunnels:
            self.service_tunnels[chal_id] = current_tunnels

        return processed

    def teardown(self, challenge_id: str, run_id: Optional[str] = None) -> TargetTeardownResult:
        # Stop the challenge via target_server HTTP API.
        requested = False
        succeeded: bool | None = None
        error: str | None = None
        container_logs: list = []
        try:
            if requests is None:
                error = "requests is required for remote backend teardown"
                succeeded = False
            else:
                params = {"run_id": run_id} if run_id else None
                requested = True
                resp = requests.delete(
                    f"{self.api_base_url}/launch/{challenge_id}",
                    params=params, timeout=10
                )
                if getattr(resp, "ok", False):
                    succeeded = True
                    try:
                        payload = resp.json()
                        if isinstance(payload, dict):
                            container_logs = payload.get("container_logs") or []
                    except Exception:
                        pass
                else:
                    status_code = getattr(resp, "status_code", "unknown")
                    reason = getattr(resp, "reason", "")
                    text = (getattr(resp, "text", "") or "").strip()
                    snippet = text if len(text) <= 1000 else text[:1000] + "...(truncated)"
                    error = f"{status_code} {reason}".strip()
                    if snippet:
                        error = f"{error}: {snippet}"
                    succeeded = False
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            requested = True
            succeeded = False
            error = str(exc)

        # Clear service tunnels for this challenge
        self._clear_service_tunnels(challenge_id)
        return TargetTeardownResult(
            challenge_id=challenge_id,
            run_id=run_id,
            requested=requested,
            succeeded=succeeded,
            error=error,
            backend="remote",
            container_logs=container_logs,
        )

    def _clear_service_tunnels(self, chal_id: str):
        if chal_id in self.service_tunnels:
            for t in self.service_tunnels[chal_id]:
                t.stop()
            del self.service_tunnels[chal_id]

    def validate_connectivity(self, challenge_id: str, record: Dict) -> bool:
        if record['type'] == 'static':
            return True
        if not self.config.use_external_access:
            return True

        for svc in record.get('services', {}).values():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            try:
                res = s.connect_ex((svc['host'], int(svc['port'])))
                if res != 0:
                    return False
            finally:
                s.close()
        return True

    def handle_crash(self, challenge_id: str, observation: str) -> tuple[str, bool]:
        return observation, True

    def cleanup(self):
        if self.tunnel:
            self.tunnel.stop()
        for tunnels in self.service_tunnels.values():
            for t in tunnels:
                t.stop()


# ==========================================
# ChallengeClient main controller
# ==========================================

class ChallengeClient:
    def __init__(
        self,
        config: ChallengeClientConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config or ChallengeClientConfig()
        self.logger = logger or logging.getLogger("ChallengeClient")

        # Challenge metadata (populated externally via config.challenges)
        self.challenges = dict(self.config.challenges)

        # Runtime state cache. Protected by ``_cache_lock`` so the client can
        # be shared across threads — concurrent get/refresh/teardown on
        # *different* challenges is now safe. Concurrent operations on the
        # *same* challenge are still serialised but never corrupt the dicts.
        self._runtime_cache: Dict[str, Dict[str, Any]] = {}
        self._runtime_args_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()

        # Initialize backend
        if self.config.run_mode == "local":
            self.logger.debug("Initializing LOCAL backend (Docker)...")
            self.backend = LocalBackend(self.config, self.logger)
        else:
            self.logger.debug(f"Initializing REMOTE backend ({self.config.server_url})...")
            self.backend = RemoteBackend(self.config, self.logger)

    def get_challenge_data(
        self,
        challenge_id: str,
        auto_init: bool = True,
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        if challenge_id not in self.challenges:
            raise ValueError(f"Challenge {challenge_id} not found.")

        with self._cache_lock:
            meta = self.challenges[challenge_id]
            resolved_runtime_args = self._resolve_runtime_args(challenge_id, runtime_args)

            # Already running — return cached
            if challenge_id in self._runtime_cache:
                return self._apply_runtime_record(
                    challenge_id, self._runtime_cache[challenge_id], meta,
                )

        # backend.initialize may take seconds; do it outside the lock to keep
        # operations on *other* challenges responsive.
        record: Dict[str, Any] | None = None
        init_error: Exception | None = None
        if auto_init:
            try:
                if resolved_runtime_args:
                    record = self.backend.initialize(
                        challenge_id,
                        meta,
                        runtime_args=resolved_runtime_args,
                    )
                else:
                    record = self.backend.initialize(challenge_id, meta)
            except Exception as exc:
                init_error = exc

        with self._cache_lock:
            if record is not None:
                return self._apply_runtime_record(challenge_id, record, meta)
            if init_error is not None:
                meta["target_status"] = "stopped"
                meta["target_info"] = {}
                meta["runtime"] = {}
                # Surface the response body / exception text so the
                # orchestrator can fold it into the trial's
                # termination_detail. Without this the frontend only
                # sees "check the log file", and operators have to
                # ssh in to find the real cause (e.g. dependency mysql
                # unhealthy). Cap at 4KB so a runaway docker compose
                # log doesn't bloat meta.json.
                err_text = str(init_error)
                meta["target_init_error"] = (
                    err_text if len(err_text) <= 4000
                    else err_text[:4000] + "...(truncated)"
                )
                # Structured container logs carried out-of-band on the
                # exception (see RemoteBackend launch-failure path). Surfaced
                # so trial_runner can persist them as an audit artifact rather
                # than losing them to the concise error string.
                logs = getattr(init_error, "target_container_logs", None)
                if logs:
                    meta["target_container_logs"] = logs
                self.logger.error(f"Init failed: {init_error}")
            self.challenges[challenge_id] |= meta
            return meta

    def refresh_challenge_data(
        self,
        challenge_id: str,
        force_recreate: bool = False,
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        if challenge_id not in self.challenges:
            raise ValueError(f"Challenge {challenge_id} not found.")

        with self._cache_lock:
            meta = self.challenges[challenge_id]
            resolved_runtime_args = self._resolve_runtime_args(challenge_id, runtime_args)
            target_scope = resolve_target_scope(
                chal_data=meta, runtime_args=resolved_runtime_args,
            )
            if (
                force_recreate
                and target_scope == "per_agent"
                and challenge_id in self._runtime_cache
            ):
                self._backend_teardown_for_record(challenge_id)
                self._clear_runtime_state(challenge_id, drop_runtime_args=False)
                force_recreate = False

        try:
            if resolved_runtime_args:
                record = self.backend.initialize(
                    challenge_id,
                    meta,
                    force_recreate=force_recreate,
                    runtime_args=resolved_runtime_args,
                )
            else:
                record = self.backend.initialize(
                    challenge_id,
                    meta,
                    force_recreate=force_recreate,
                )
        except Exception as e:
            with self._cache_lock:
                self._runtime_cache.pop(challenge_id, None)
                meta["target_status"] = "stopped"
                meta["target_info"] = {}
                meta["runtime"] = {}
            self.logger.error(f"Refresh failed for {challenge_id}: {e}")
            raise
        with self._cache_lock:
            return self._apply_runtime_record(challenge_id, record, meta)

    def finish_challenge(self, challenge_id: str) -> TargetTeardownResult:
        """Release resources after challenge completion."""
        return self.teardown(challenge_id)

    def teardown(self, challenge_id: str) -> TargetTeardownResult:
        with self._cache_lock:
            record = self._runtime_cache.get(challenge_id, {}) or {}
            run_id = record.get("run_id")
        try:
            result = self._backend_teardown_for_record(challenge_id)
        except Exception as e:
            self.logger.warning(f"Teardown backend failed for {challenge_id}: {e}")
            result = TargetTeardownResult(
                challenge_id=challenge_id,
                run_id=str(run_id) if run_id not in (None, "") else None,
                requested=True,
                succeeded=False,
                error=str(e),
                backend=type(self.backend).__name__,
            )

        with self._cache_lock:
            self._clear_runtime_state(challenge_id, drop_runtime_args=True)
        return result

    def close(self, *, delete_run: bool = True):
        """Close the manager, cleaning up local resources.

        Iterates every cached runtime record and issues ``DELETE
        /launch/<chal_id>?run_id=<run_id>`` per instance — one DELETE = one
        target. The previous batch ``DELETE /run/<id>`` endpoint was removed
        because it amplified cleanup bugs (one stale call could nuke every
        in-flight trial). Orphans not in our cache are handled by the
        orchestrator-side label sweep in ``local_cleanup.sweep_run``.

        ``delete_run`` is kept as a parameter name for back-compat but no
        longer changes whether targets are torn down — they always are, via
        the per-instance loop below. Per-trial finally blocks can still pass
        ``delete_run=False`` to skip the loop and only release local backend
        session state.
        """
        if delete_run:
            with self._cache_lock:
                active_ids = list(self._runtime_cache.keys())
            for cid in active_ids:
                try:
                    self.teardown(cid)
                except Exception as e:
                    self.logger.warning(f"Close teardown failed for {cid}: {e}")
        try:
            self.backend.cleanup()
        except Exception as e:
            self.logger.warning(f"Backend cleanup failed: {e}")

    def _apply_runtime_record(self, challenge_id: str, record: Dict, meta: Optional[Dict] = None) -> Dict:
        result = meta or self.challenges[challenge_id]
        self._runtime_cache[challenge_id] = record

        result["target_status"] = "running" if record["type"] == "dynamic" else "static"
        result["target_info"] = deepcopy(record.get("services", {}))
        result["runtime"] = self._build_runtime_metadata(record, result)
        # Clear any stale launch-failure text from a previous trial so a
        # retry that succeeds isn't reported with the old error body.
        result.pop("target_init_error", None)

        if record["type"] == "static":
            result["message"] = f"Files at {record['work_dir']}"
        else:
            result["message"] = "Service Started."
            if self.config.run_mode == "remote" and self.config.use_ssh_tunnel:
                result["message"] += " (SSH Tunneled to Localhost)"

        self.challenges[challenge_id] |= result
        return result

    def _build_runtime_metadata(self, record: Dict, challenge: Dict) -> Dict[str, Any]:
        source_fields = dict(challenge.get("source_fields", {}) or {})
        scoring = dict(record.get("scoring", {}) or source_fields.get("runtime_scoring", {}) or {})
        debug = dict(record.get("debug", {}) or {})
        network_debug = dict(debug.get("network", {}) or {})
        runtime: Dict[str, Any] = {
            "run_id": record.get("run_id"),
            "project_name": record.get("project_name"),
            "network_name": record.get("network_name") or self.config.network_name,
            "network_subnet": record.get("network_subnet") or network_debug.get("subnet"),
            "network_gateway": record.get("network_gateway") or network_debug.get("gateway"),
            "scoring": scoring,
            "debug": debug,
        }
        return runtime

    def _backend_teardown_for_record(self, challenge_id: str) -> TargetTeardownResult:
        record = self._runtime_cache.get(challenge_id, {}) or {}
        run_id = record.get("run_id")
        run_id_str = str(run_id) if run_id not in (None, "") else None
        try:
            raw_result = self.backend.teardown(challenge_id, run_id=run_id_str)
        except TypeError:
            raw_result = self.backend.teardown(challenge_id)
        return _coerce_teardown_result(
            raw_result,
            challenge_id=challenge_id,
            run_id=run_id_str,
            backend=type(self.backend).__name__,
        )

    def _clear_runtime_state(self, challenge_id: str, drop_runtime_args: bool) -> None:
        if challenge_id in self._runtime_cache:
            del self._runtime_cache[challenge_id]
        runtime_args_cache = getattr(self, "_runtime_args_cache", None)
        if drop_runtime_args and runtime_args_cache is not None and challenge_id in runtime_args_cache:
            del runtime_args_cache[challenge_id]

        if challenge_id in self.challenges:
            try:
                self.challenges[challenge_id]["target_status"] = "stopped"
                self.challenges[challenge_id]["target_info"] = {}
                self.challenges[challenge_id]["runtime"] = {}
            except Exception:
                pass

    def remember_runtime_args(
        self,
        challenge_id: str,
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._cache_lock:
            cache = getattr(self, "_runtime_args_cache", None)
            if cache is None:
                cache = {}
                self._runtime_args_cache = cache
            cache[challenge_id] = deepcopy(dict(runtime_args or {}))

    def _resolve_runtime_args(
        self,
        challenge_id: str,
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cache = getattr(self, "_runtime_args_cache", None)
        if cache is None:
            cache = {}
            self._runtime_args_cache = cache

        if runtime_args is not None:
            resolved = deepcopy(dict(runtime_args))
            cache[challenge_id] = resolved
            return deepcopy(resolved)

        return deepcopy(cache.get(challenge_id, {}))
