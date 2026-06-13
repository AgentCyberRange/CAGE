"""Health/readiness probes: container state, port openness, HTTP, inner-network reachability."""
from __future__ import annotations

import logging
import socket
import time
from typing import Any, Dict, List, Optional

import requests

from cage.target.server.network_admin import list_project_containers
from cage.target.server.schemas import ServiceInfo
from cage.target.server.server_state import (
    HEALTH_POLL_INTERVAL_S,
    HEALTH_TIMEOUT_S,
    INSTANCE_HEALTH_TIMEOUT_S,
    PORT_OPEN_STABILITY_CHECKS,
    get_docker_client,
    get_running_instance,
)

logger = logging.getLogger(__name__)


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is open and accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def wait_for_containers_running(
    project_name: str,
    required_services: Optional[List[str]],
    timeout_s: float,
    poll_interval_s: float,
) -> Optional[str]:
    """Wait until all required services have at least one container in 'running' state.

    Returns None on success, otherwise an error string.
    """
    deadline = time.time() + timeout_s
    required = set(required_services or [])

    while time.time() < deadline:
        containers = list_project_containers(project_name)
        if not containers:
            time.sleep(poll_interval_s)
            continue

        if not required:
            for c in containers:
                try:
                    c.reload()
                    state = (c.attrs or {}).get("State", {})
                    if state.get("Status") == "running":
                        return None
                except Exception:
                    continue
            time.sleep(poll_interval_s)
            continue

        running_by_service: Dict[str, bool] = {svc: False for svc in required}
        seen_by_service: Dict[str, bool] = {svc: False for svc in required}

        for c in containers:
            labels = c.labels or {}
            svc = labels.get("com.docker.compose.service")
            if svc not in required:
                continue
            seen_by_service[svc] = True
            try:
                c.reload()
                state = (c.attrs or {}).get("State", {})
                if state.get("Status") == "running":
                    running_by_service[svc] = True
            except Exception:
                pass

        missing = sorted([s for s, seen in seen_by_service.items() if not seen])
        not_running = sorted([s for s, ok in running_by_service.items() if not ok])
        if not missing and not not_running:
            return None

        time.sleep(poll_interval_s)

    if required:
        return f"timeout waiting for services to run: {sorted(required)}"
    return "timeout waiting for any container to run"


def wait_for_services_healthy(
    project_name: str,
    required_services: Optional[List[str]],
    timeout_s: float,
    poll_interval_s: float,
) -> Optional[str]:
    """Wait until all required services are ready according to Docker health checks.

    - If a service has at least one running container with a Docker health status,
      require one of them to become `healthy`.
    - If a service has no Docker health check, fall back to requiring a running container.
    - Returns None on success, otherwise an error string.
    """
    required = set(required_services or [])
    if not required:
        return None

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        containers = list_project_containers(project_name)
        if not containers:
            time.sleep(poll_interval_s)
            continue

        seen_by_service: Dict[str, bool] = {svc: False for svc in required}
        ready_by_service: Dict[str, bool] = {svc: False for svc in required}
        mode_by_service: Dict[str, str] = {}

        for c in containers:
            labels = c.labels or {}
            svc = labels.get("com.docker.compose.service")
            if svc not in required:
                continue
            seen_by_service[svc] = True
            try:
                c.reload()
            except Exception:
                continue

            state = (c.attrs or {}).get("State", {}) or {}
            if state.get("Status") != "running":
                mode_by_service.setdefault(svc, "running")
                continue

            health = (state.get("Health", {}) or {}).get("Status")
            if health:
                mode_by_service[svc] = "healthy"
                if health == "healthy":
                    ready_by_service[svc] = True
            else:
                mode_by_service.setdefault(svc, "running")
                ready_by_service[svc] = True

        missing = sorted([svc for svc, seen in seen_by_service.items() if not seen])
        waiting = sorted([svc for svc, ready in ready_by_service.items() if not ready])
        if not missing and not waiting:
            return None

        time.sleep(poll_interval_s)

    if required:
        return f"timeout waiting for services to become healthy: {sorted(required)}"
    return None


def _wait_for_ports_stably_open(
    services: List[ServiceInfo],
    timeout_s: float,
    poll_interval_s: float,
    required_stable_checks: int,
) -> Optional[str]:
    pending: Dict[str, Dict[str, int]] = {
        svc.service_name: {"port": int(svc.external_port), "stable": 0}
        for svc in services
        if svc.external_port is not None
    }
    if not pending:
        return None

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for svc_name, state in list(pending.items()):
            port = state["port"]
            if is_port_open("127.0.0.1", port, timeout=2.0):
                state["stable"] += 1
                if state["stable"] >= required_stable_checks:
                    pending.pop(svc_name, None)
            else:
                if state["stable"]:
                    logger.info(
                        "[PortCheck] Port %s (service %s) flapped after %s consecutive successes",
                        port,
                        svc_name,
                        state["stable"],
                    )
                state["stable"] = 0

        if not pending:
            return None
        time.sleep(poll_interval_s)

    parts = [f"{svc}:{state['port']}" for svc, state in sorted(pending.items())]
    return f"timeout waiting for ports to open: {', '.join(parts)}"


def wait_for_ports_open(
    services: List[ServiceInfo],
    timeout_s: float,
    poll_interval_s: float,
) -> Optional[str]:
    """Wait until all services with external_port become reachable via localhost."""
    return _wait_for_ports_stably_open(
        services,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        required_stable_checks=PORT_OPEN_STABILITY_CHECKS,
    )


def probe_inner_service(
    *,
    network_name: str,
    service: ServiceInfo,
    http_path: str = "/",
    timeout_s: float = 5.0,
) -> bool:
    """Probe a service for liveness via host-side or docker-exec connectivity check."""
    if not service.inner_ip or service.inner_port is None:
        return False

    protocol = str(service.protocol or "tcp").strip().lower() or "tcp"

    ext_port = getattr(service, "external_port", None)
    if ext_port is not None:
        if protocol == "http":
            try:
                resp = requests.get(
                    f"http://127.0.0.1:{ext_port}{http_path}",
                    timeout=min(timeout_s, 3.0),
                    allow_redirects=False,
                )
                return resp.status_code >= 100
            except requests.RequestException:
                return False
        else:
            return is_port_open("127.0.0.1", int(ext_port), timeout=min(timeout_s, 3.0))

    try:
        containers = get_docker_client().containers.list(
            filters={"network": network_name, "status": "running"},
        )
        exec_container = None
        for c in containers:
            svc_label = (c.labels or {}).get("com.docker.compose.service", "")
            if svc_label != service.service_name:
                exec_container = c
                break
        if exec_container is None and containers:
            exec_container = containers[0]

        if exec_container is not None:
            ip = service.inner_ip
            port = service.inner_port
            if protocol == "udp":
                probe_cmd = (
                    f"(echo -n | nc -u -w1 {ip} {port}) "
                    f"2>/dev/null; exit 0"
                )
                exit_code, _ = exec_container.exec_run(
                    ["sh", "-c", probe_cmd], demux=False,
                )
            else:
                exit_code, _ = exec_container.exec_run(
                    ["bash", "-c", f"echo > /dev/tcp/{ip}/{port}"],
                    demux=False,
                )
                if exit_code == 127:
                    nc_cmd = f"(echo | nc -w2 {ip} {port}) 2>/dev/null"
                    exit_code, _ = exec_container.exec_run(
                        ["sh", "-c", nc_cmd], demux=False,
                    )
            if exit_code == 0:
                return True
            if exit_code == 127:
                logger.info(
                    "[InnerProbe] No networking tools in container for %s, using host TCP fallback",
                    service.service_name,
                )
            else:
                logger.info(
                    "[InnerProbe] docker exec probe failed for %s on %s:%s (exit=%s)",
                    service.service_name, ip, port, exit_code,
                )
                return False
    except Exception as e:
        logger.info("[InnerProbe] docker exec probe unavailable: %s", e)

    return is_port_open(service.inner_ip, int(service.inner_port), timeout=min(timeout_s, 3.0))


def wait_for_inner_services_ready(
    *,
    services: List[ServiceInfo],
    network_name: str,
    timeout_s: float,
    poll_interval_s: float,
    http_path: str = "/",
) -> Optional[str]:
    pending: Dict[str, Dict[str, Any]] = {}
    missing_inner_ip: List[str] = []
    for svc in services:
        if svc.inner_port is None:
            continue
        if not svc.inner_ip:
            missing_inner_ip.append(svc.service_name)
            continue
        pending[svc.service_name] = {"service": svc, "stable": 0}

    if missing_inner_ip:
        return f"missing inner IPs for services: {', '.join(sorted(missing_inner_ip))}"
    if not pending:
        return None

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for svc_name, state in list(pending.items()):
            svc = state["service"]
            if probe_inner_service(
                network_name=network_name,
                service=svc,
                http_path=http_path,
                timeout_s=max(8.0, poll_interval_s + 5.0),
            ):
                state["stable"] += 1
                if state["stable"] >= PORT_OPEN_STABILITY_CHECKS:
                    pending.pop(svc_name, None)
                continue

            if state["stable"]:
                logger.info(
                    "[InnerProbe] Service %s flapped after %s consecutive successes",
                    svc_name,
                    state["stable"],
                )
            state["stable"] = 0

        if not pending:
            return None
        time.sleep(poll_interval_s)

    parts = [
        f"{svc_name}:{state['service'].inner_ip}:{state['service'].inner_port}"
        for svc_name, state in sorted(pending.items())
    ]
    return f"timeout waiting for inner services to become reachable: {', '.join(parts)}"


def wait_for_http_ready(
    services: List[ServiceInfo],
    timeout_s: float,
    poll_interval_s: float,
    path: str = "/",
) -> Optional[str]:
    """For HTTP services, wait until localhost responds for a stability window."""
    pending: Dict[str, Dict[str, Any]] = {
        svc.service_name: {"port": int(svc.external_port), "stable": 0}
        for svc in services
        if svc.external_port is not None
    }
    if not pending:
        return None

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for svc_name, state in list(pending.items()):
            port = state["port"]
            try:
                resp = requests.get(
                    f"http://127.0.0.1:{port}{path}",
                    timeout=2.0,
                    allow_redirects=False,
                )
                if resp.status_code >= 100:
                    state["stable"] += 1
                    if state["stable"] >= PORT_OPEN_STABILITY_CHECKS:
                        pending.pop(svc_name, None)
                    continue
            except requests.RequestException:
                pass

            if state["stable"]:
                logger.info(
                    "[HttpCheck] Service %s on port %s flapped after %s consecutive HTTP successes",
                    svc_name,
                    port,
                    state["stable"],
                )
            state["stable"] = 0

        if not pending:
            return None
        time.sleep(poll_interval_s)

    parts = [f"{svc}:{state['port']}" for svc, state in sorted(pending.items())]
    return f"timeout waiting for HTTP readiness: {', '.join(parts)}"


def is_instance_healthy(chal_id: str) -> bool:
    """Determine instance health from container state, Docker healthcheck, and inner probe."""
    info = get_running_instance(chal_id)
    if info is None:
        return False
    health_timeout_s = max(HEALTH_TIMEOUT_S, INSTANCE_HEALTH_TIMEOUT_S)

    project_name = info.get("project_name")
    services = list(info.get("services", []) or [])
    public_service_names = set(info.get("public_services", []) or [])
    if public_service_names:
        filtered_services = [
            svc for svc in services if getattr(svc, "service_name", None) in public_service_names
        ]
        if filtered_services:
            services = filtered_services
    if not services:
        if not project_name:
            logger.warning(f"[HealthCheck] No services and no project_name recorded for {chal_id}")
            return False
        containers = list_project_containers(project_name)
        if not containers:
            logger.warning(f"[HealthCheck] No containers found for {chal_id} ({project_name})")
            return False
        for c in containers:
            try:
                c.reload()
                state = (c.attrs or {}).get("State", {})
                if state.get("Status") == "running":
                    return True
            except Exception:
                continue
        logger.warning(f"[HealthCheck] Containers exist but none running for {chal_id} ({project_name})")
        return False

    service_names = [svc.service_name for svc in services if getattr(svc, "service_name", None)]
    if project_name and service_names:
        health_err = wait_for_services_healthy(
            project_name=project_name,
            required_services=service_names,
            timeout_s=health_timeout_s,
            poll_interval_s=HEALTH_POLL_INTERVAL_S,
        )
        if health_err is not None:
            logger.warning(f"[HealthCheck] {chal_id} failed container health check: {health_err}")
            return False

    network_name = str(info.get("network_name", "") or "").strip()
    services_with_inner_targets = [
        svc
        for svc in services
        if getattr(svc, "inner_ip", None) and getattr(svc, "inner_port", None) is not None
    ]
    if network_name and services_with_inner_targets:
        inner_err = wait_for_inner_services_ready(
            services=services_with_inner_targets,
            network_name=network_name,
            timeout_s=health_timeout_s,
            poll_interval_s=HEALTH_POLL_INTERVAL_S,
        )
        if inner_err is not None:
            logger.warning(f"[HealthCheck] {chal_id} failed inner probe: {inner_err}")
            return False
        return True

    ports_err = _wait_for_ports_stably_open(
        services,
        timeout_s=health_timeout_s,
        poll_interval_s=HEALTH_POLL_INTERVAL_S,
        required_stable_checks=PORT_OPEN_STABILITY_CHECKS,
    )
    if ports_err is not None:
        logger.warning(f"[HealthCheck] {chal_id} failed stable port check: {ports_err}")
        return False
    return True
