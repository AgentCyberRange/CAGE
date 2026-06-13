"""Target readiness/exploitation probes for ``cage targets-check``.

Launch-and-probe over a target instance (HTTP readiness, post-exploitation and
pentest-remote ingress checks). Consumed by :mod:`cage.target.check`.
"""
from __future__ import annotations

import logging
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InstanceResult:
    sample_id: str
    instance_idx: int
    cage_run_id: str
    project_name: str = ""
    network_name: str = ""
    network_subnet: str = ""
    services_running: int = 0
    services_total: int = 0
    duration_s: float = 0.0
    passed: bool = False
    error: str = ""
    notes: list[str] = field(default_factory=list)


def _target_id(sample: dict[str, Any]) -> str:
    """Return the id target_server should launch for a sample.

    Benchmarks may fan a single target out into multiple prompt/hint variants
    such as ``pb-demo-l0`` and ``pb-demo-l1``. The target_server only knows the
    underlying challenge id, so target checks must launch ``challenge_id`` once
    instead of treating prompt variants as independent target stacks.
    """

    return str(sample.get("challenge_id") or sample.get("id") or "?")


class _LaunchHTTPError(RuntimeError):
    """HTTP error from target_server with the server-side detail attached.

    ``urllib.error.HTTPError`` carries the raw body but ``str(exc)`` is only
    ``HTTP Error 500: Internal Server Error`` — useless for diagnosing why a
    launch failed. We unwrap the body (FastAPI puts the actionable info into
    ``detail`` — either the docker compose stderr from ``launch_workflow.py``
    line 631, or the structured readiness-failure dict from line 687) and
    expose it on ``self.detail``.
    """

    def __init__(self, code: int, detail: Any, raw_body: str):
        self.code = code
        self.detail = detail
        self.raw_body = raw_body
        super().__init__(self._format())

    def _format(self) -> str:
        if isinstance(self.detail, str) and self.detail:
            return f"HTTP {self.code}: {self.detail}"
        if isinstance(self.detail, dict):
            # Surface the most useful fields from the readiness-failure dict
            # (cage/target/server/launch_workflow.py:672-678) without
            # dumping the whole containers blob.
            err = self.detail.get("error")
            services = self.detail.get("public_services") or []
            containers = self.detail.get("containers") or []
            head = f"HTTP {self.code}: {err}" if err else f"HTTP {self.code}"
            if services:
                head += f"  (public_services={services})"
            if containers:
                # Each container summary has name+state — show non-running ones.
                bad = [
                    f"{c.get('service') or c.get('name')}={c.get('state')}"
                    for c in containers
                    if isinstance(c, dict) and c.get("state") not in {"running", "exited"}
                ]
                if bad:
                    head += f"  not-running={bad}"
            return head
        if self.raw_body:
            return f"HTTP {self.code}: {self.raw_body[:400]}"
        return f"HTTP {self.code}"


def _http_get_json(url: str, timeout: float | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    try:
        # timeout=None → wait as long as the server takes. /launch is bounded
        # by server-side TARGET_SERVER_COMPOSE_UP_TIMEOUT_S + TARGET_SERVER_STARTUP_TIMEOUT_S;
        # a client-side cap on top of that is just a footgun.
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = ""
        detail: Any = None
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    detail = parsed.get("detail", parsed)
                else:
                    detail = parsed
            except json.JSONDecodeError:
                detail = raw
        raise _LaunchHTTPError(exc.code, detail, raw) from exc


def _http_delete(url: str, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        # 404 is fine — instance already gone.
        if exc.code != 404:
            raise


def _docker_inspect_network(network: str) -> dict[str, Any]:
    if not network:
        return {}
    proc = subprocess.run(
        ["docker", "network", "inspect", network],
        capture_output=True, text=True, timeout=15.0, check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        arr = json.loads(proc.stdout)
        return arr[0] if isinstance(arr, list) and arr else {}
    except json.JSONDecodeError:
        return {}


def _compose_ps(project_name: str) -> list[dict[str, Any]]:
    if not project_name:
        return []
    proc = subprocess.run(
        ["docker", "compose", "-p", project_name, "ps", "--format", "json"],
        capture_output=True, text=True, timeout=30.0, check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    rows: list[dict[str, Any]] = []
    text = proc.stdout.strip()
    # docker compose emits either a JSON array OR one object per line.
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
        except json.JSONDecodeError:
            return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _is_http_ready(code: str) -> bool:
    """A response counts as 'service is up' when HTTP returned anything 2xx/3xx/4xx.

    ``000`` (curl: couldn't connect / timed out) and ``5xx`` mean the service
    isn't actually serving yet.
    """
    return bool(code) and code[0] in {"2", "3", "4"}


_PROBE_IMAGE = "python:3.11-alpine"


def _probe_from_network(
    *,
    network: str,
    targets: list[tuple[str, int]],
    deadline: float,
    per_attempt_timeout: float = 5.0,
    wait_for_all: bool = True,
) -> dict[str, str]:
    """Probe each ``(host, port)`` from a temp container on ``network``.

    Returns a dict ``{ "host:port": "<final_http_code>" }``. A status whose
    first digit is 2/3/4 means the service answered; ``000`` means we never
    got a response before ``deadline``. Polling is exponential-backoff up to
    5s between attempts.

    The probe runs from inside the docker network, exactly the way the
    agent will hit the target — host-port mappings, NAT, and the host's
    /etc/hosts are irrelevant.
    """
    if not network or not targets:
        return {}
    results: dict[str, str] = {f"{h}:{p}": "000" for h, p in targets}
    if time.monotonic() >= deadline:
        return results

    # One ``docker run`` per attempt instead of per target — container startup
    # dominates HTTP timeout. ``--pull never`` keeps this check read-only with
    # respect to the image cache; missing probe images fail fast as 000.
    script = """
import ssl
import sys
import urllib.error
import urllib.request

timeout = float(sys.argv[1])
context = ssl._create_unverified_context()
for target in sys.argv[2:]:
    host, port = target.rsplit(":", 1)
    code = "000"
    try:
        req = urllib.request.Request(f"http://{host}:{port}/", method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            code = str(resp.getcode())
    except urllib.error.HTTPError as exc:
        code = str(exc.code)
    except Exception:
        code = "000"
    print(f"{target} {code}", flush=True)
""".strip()
    target_args = [f"{h}:{p}" for h, p in targets]

    delay = 1.0
    while time.monotonic() < deadline:
        try:
            proc = subprocess.run(
                [
                    "docker", "run", "--rm", "--pull", "never",
                    "--network", network,
                    "--entrypoint", "python3", _PROBE_IMAGE,
                    "-c", script, str(per_attempt_timeout), *target_args,
                ],
                capture_output=True, text=True,
                timeout=per_attempt_timeout * len(targets) + 30.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("probe container error on %s: %s", network, exc)
            time.sleep(min(delay, max(0.5, deadline - time.monotonic())))
            delay = min(delay * 1.5, 5.0)
            continue

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            logger.warning(
                "probe container failed on %s with exit=%s: %s",
                network,
                proc.returncode,
                detail[:300],
            )
            if "No such image" in detail or "pull access denied" in detail:
                return results
            time.sleep(min(delay, max(0.5, deadline - time.monotonic())))
            delay = min(delay * 1.5, 5.0)
            continue

        for line in (proc.stdout or "").splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            key, code = parts
            if key in results:
                results[key] = code

        if all(_is_http_ready(c) for c in results.values()):
            return results
        if not wait_for_all and any(_is_http_ready(c) for c in results.values()):
            return results
        time.sleep(min(delay, max(0.5, deadline - time.monotonic())))
        delay = min(delay * 1.5, 5.0)
    return results


def _ps_running_summary(ps: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
    running = 0
    bad: list[str] = []
    for row in ps:
        state = str(row.get("State") or row.get("Status") or "").lower()
        name = row.get("Service") or row.get("Name") or "?"
        if "running" in state or "up" in state:
            running += 1
        else:
            bad.append(f"{name}={state or 'unknown'}")
    return running, len(ps), bad


def _ingress_targets_for_pentest_remote(
    target_data: dict[str, Any], sample: dict[str, Any],
) -> list[tuple[str, str, int]]:
    """Return ``[(service_name, agent_facing_host, port), ...]`` to probe.

    Only the services the agent actually attacks count — for SIYUCMS that's
    ``web`` (the application). Evaluator / scoring sidecars listen too but
    aren't the agent's target, so skipping them keeps the PASS criterion
    honest. Falls back to all services with an internal_port if the sample
    doesn't declare ``application_service_keys``.
    """
    services = target_data.get("services") or []
    by_name = {str(s.get("service_name") or ""): s for s in services}

    source_fields = sample.get("source_fields") or {}
    # The orchestrator strips source_fields out; fall back to the raw
    # challenge.json keys we stored on the sample.
    keys = sample.get("application_service_keys") or source_fields.get(
        "application_service_keys"
    ) or []
    keys = [str(k) for k in keys if k]
    if not keys:
        keys = [
            str(s.get("service_name"))
            for s in services
            if s.get("internal_port") and s.get("service_name")
        ]

    out: list[tuple[str, str, int]] = []
    for key in keys:
        svc = by_name.get(key)
        if not isinstance(svc, dict):
            continue
        port = svc.get("internal_port")
        if not port:
            continue
        # ``inner_host`` is the qualified alias target_server publishes on the
        # shared run network. That's what the agent will resolve once the
        # orchestrator's isolation bridge aliases it back to the bare
        # service name — but for this check we use the qualified name
        # directly so we don't have to replicate the bridge.
        host = str(svc.get("inner_host") or svc.get("alias") or key)
        out.append((key, host, int(port)))
    return out


def _ingress_targets_for_post_exp(
    target_data: dict[str, Any], sample: dict[str, Any],
) -> list[tuple[str, str, int]]:
    """Return the entry-network hosts the agent must be able to reach.

    For ``compose_project_local`` networks (post-exp), services on the
    agent_network are reachable by their plain compose service name —
    that's what an agent inside the container sees with ``getent hosts
    1_report``.
    """
    services = target_data.get("services") or []
    out: list[tuple[str, str, int]] = []
    for svc in services:
        svc_name = str(svc.get("service_name") or "")
        if not svc_name:
            continue
        # We only want services actually attached to the agent's network.
        # target_server doesn't expose per-service network membership directly
        # in services[]; rely on the project-local-network convention plus
        # the fact that post-exp samples use project-local networks
        # uniformly. The probe will just fail-fast on services not on the
        # network — that's correct.
        port = svc.get("internal_port")
        if not port:
            continue
        out.append((svc_name, svc_name, int(port)))
    return out


def _sample_list_field(sample: dict[str, Any], key: str) -> list[str]:
    for container in (
        sample,
        sample.get("metadata") or {},
        sample.get("source_fields") or {},
    ):
        if not isinstance(container, dict):
            continue
        raw = container.get(key)
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip()]
            if values:
                return values
    return []


def _required_post_exp_probe_names(sample: dict[str, Any]) -> list[str]:
    explicit = _sample_list_field(sample, "readiness_probe_services")
    if explicit:
        return explicit

    marker_services = _sample_list_field(sample, "marker_services")
    if marker_services:
        return marker_services[:1]

    entry_services = _sample_list_field(sample, "entry_services")
    if entry_services:
        return entry_services[:1]
    return []


def _probe_post_exploitation(
    target_data: dict[str, Any],
    result: InstanceResult,
    sample: dict[str, Any],
    readiness_timeout: float,
) -> None:
    project_name = result.project_name
    network_name = result.network_name

    ps = _compose_ps(project_name)
    running, total, bad = _ps_running_summary(ps)
    result.services_running = running
    result.services_total = total
    if not ps:
        result.error = f"docker compose -p {project_name} ps returned nothing"
        return
    if bad:
        result.error = "non-running services: " + ", ".join(bad)
        return

    net = _docker_inspect_network(network_name)
    ipam_cfg = ((net.get("IPAM") or {}).get("Config") or [])
    if not ipam_cfg:
        result.error = f"agent network {network_name} has no IPAM config"
        return
    result.network_subnet = str(ipam_cfg[0].get("Subnet") or "")

    # In-network reachability check — only services on agent_network will
    # actually answer. We don't pre-filter; the probe will surface which
    # ones do/don't, which is itself useful info.
    targets = _ingress_targets_for_post_exp(target_data, sample)
    if not targets:
        result.passed = True
        return
    targets_by_service = {svc_name: (svc_name, host, port) for svc_name, host, port in targets}
    required_names = _required_post_exp_probe_names(sample)
    required_targets: list[tuple[str, str, int]]
    if required_names:
        required_targets = [
            targets_by_service[name]
            for name in required_names
            if name in targets_by_service
        ]
        missing = [name for name in required_names if name not in targets_by_service]
        if missing:
            result.error = "required entry service missing target port: " + ", ".join(missing)
            return
    else:
        required_targets = targets

    deadline = time.monotonic() + readiness_timeout
    codes = _probe_from_network(
        network=network_name,
        targets=[(h, p) for _, h, p in required_targets],
        deadline=deadline,
        wait_for_all=True,
    )
    reachable, unreachable = [], []
    for svc_name, host, port in required_targets:
        code = codes.get(f"{host}:{port}", "000")
        line = f"{svc_name} {host}:{port} -> {code}"
        (reachable if _is_http_ready(code) else unreachable).append((svc_name, line))
    for _, line in reachable:
        result.notes.append(line)
    for _, line in unreachable:
        result.notes.append("unreachable: " + line)

    if unreachable:
        result.error = (
            "required entry service unreachable over HTTP/HTTPS: "
            + ", ".join(svc_name for svc_name, _line in unreachable)
        )
        return
    result.passed = True


def _probe_pentest_remote(
    target_data: dict[str, Any],
    result: InstanceResult,
    sample: dict[str, Any],
    readiness_timeout: float,
) -> None:
    ps = _compose_ps(result.project_name)
    running, total, bad = _ps_running_summary(ps)
    result.services_running = running
    result.services_total = total
    if not ps:
        result.error = f"docker compose -p {result.project_name} ps returned nothing"
        return
    if bad:
        result.error = "non-running services: " + ", ".join(bad)
        return

    network_name = result.network_name
    targets = _ingress_targets_for_pentest_remote(target_data, sample)
    if not targets:
        result.error = "no agent-facing services declared (application_service_keys empty)"
        return

    deadline = time.monotonic() + readiness_timeout
    codes = _probe_from_network(
        network=network_name,
        targets=[(h, p) for _, h, p in targets],
        deadline=deadline,
    )
    all_ok = True
    for svc_name, host, port in targets:
        code = codes.get(f"{host}:{port}", "000")
        line = f"{svc_name} http://{host}:{port}/ -> {code}"
        if _is_http_ready(code):
            result.notes.append(line)
        else:
            result.notes.append("unreachable: " + line)
            all_ok = False

    if not all_ok:
        # The agent will fail with "can't curl http://web:80" if we let this
        # pass. Fail-fast here so the operator notices BEFORE burning a run.
        result.error = "agent-facing service unreachable from inside the network"
        return
    result.passed = True


def _probe_generic(target_data: dict[str, Any], result: InstanceResult) -> None:
    ps = _compose_ps(result.project_name)
    running, total, bad = _ps_running_summary(ps)
    result.services_running = running
    result.services_total = total
    if not ps:
        result.error = f"docker compose -p {result.project_name} ps returned nothing"
        return
    if bad:
        result.error = "non-running services: " + ", ".join(bad)
        return
    result.passed = True


def _probe_instance(
    target_data: dict[str, Any],
    sample: dict[str, Any],
    result: InstanceResult,
    readiness_timeout: float,
) -> None:
    profile = str(sample.get("task_profile") or "")
    if profile == "post_exploitation":
        _probe_post_exploitation(target_data, result, sample, readiness_timeout)
    elif profile == "pentest_remote":
        _probe_pentest_remote(target_data, result, sample, readiness_timeout)
    else:
        _probe_generic(target_data, result)


def _launch_and_probe(
    *,
    server_url: str,
    sample: dict[str, Any],
    instance_idx: int,
    cage_run_id: str,
    readiness_timeout: float,
) -> InstanceResult:
    sample_id = _target_id(sample)
    result = InstanceResult(
        sample_id=sample_id, instance_idx=instance_idx, cage_run_id=cage_run_id,
    )
    qs = urllib.parse.urlencode({
        "target_scope": "per_agent",
        "cage_run_id": cage_run_id,
    })
    url = f"{server_url}/launch/{urllib.parse.quote(sample_id, safe='')}?{qs}"
    t0 = time.monotonic()
    try:
        logger.info("launch %s instance %d (run_id=%s)", sample_id, instance_idx, cage_run_id)
        target_data = _http_get_json(url)
        # Valid statuses from target_server: launched / recreated / reused / static.
        status = str(target_data.get("status") or "")
        if status not in {"launched", "recreated", "reused", "static"}:
            result.error = f"launch status={status!r}"
            return result
        result.project_name = str(target_data.get("project_name") or "")
        result.network_name = str(target_data.get("network_name") or "")
        _probe_instance(target_data, sample, result, readiness_timeout)
    except _LaunchHTTPError as exc:
        # ``_LaunchHTTPError.__str__`` already pulls the FastAPI ``detail`` field
        # (docker compose stderr or readiness-failure dict) out of the body —
        # no need to repeat the type prefix.
        result.error = str(exc)
        result.passed = False
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.passed = False
    finally:
        result.duration_s = time.monotonic() - t0
    return result
