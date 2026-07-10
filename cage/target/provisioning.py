"""Target runtime construction: embedded target_server, isolation networks, attach.

Owns the pure (stateless) target-provisioning slice used by the trial runtime:
launching the embedded target_server subprocess, building target/runtime config,
constructing and attaching per-trial agent isolation networks, injecting target
stack metadata into samples, and ephemeral check networks. These are plain
functions over config/sample/container inputs — they hold no run-level mutable
state (the active-client registry and teardown live with the run conductor).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cage.contracts.sample_keys import (
    SAMPLE_NETWORK_NAME_KEY,
    SAMPLE_NETWORK_SUBNET_KEY,
    SAMPLE_RUNTIME_ARGS_KEY,
    SAMPLE_TARGET_INFO_KEY,
)
from cage.benchmarks import Benchmark
from cage.contracts.runtime_state import RUNTIME_STATE_KEY, RuntimeState
from cage.sandbox.naming import (
    _safe_docker_name_component,
)
from cage.sandbox.containers import Container
from cage.target.client import (
    ChallengeClientConfig,
    SSHConfig,
)
from cage.target.server.network_alloc import (
    allocate_cage_trial_subnet,
    release_cage_trial_subnet,
)

if TYPE_CHECKING:
    from cage.experiment.engine.run_context import ExperimentRun


logger = logging.getLogger("cage.target.provisioning")


@dataclass
class EmbeddedTargetServer:
    """A per-run target_server server subprocess managed by the orchestrator."""

    process: subprocess.Popen
    port: int
    namespace: str
    server_url: str

    def stop(self, timeout: float = 5.0) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "embedded target_server did not exit on SIGTERM; sending SIGKILL"
                )
                self.process.kill()
                self.process.wait(timeout=2)
        except Exception as exc:
            logger.warning("embedded target_server stop failed: %s", exc)

def discover_benchmark_root(benchmark: Any) -> Path | None:
    """Best-effort: read ``benchmark.benchmark_root`` if the benchmark exposes one."""
    root = getattr(benchmark, "benchmark_root", None)
    if root is None:
        return None
    try:
        resolved = Path(root).expanduser().resolve()
    except (TypeError, ValueError):
        return None
    return resolved if resolved.is_dir() else None

def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

def _namespace_from_run_id(run_id: str) -> str:
    """Convert ``run_id`` into a target_server namespace (alnum + underscore only)."""
    safe = re.sub(r"[^a-z0-9_]+", "_", run_id.lower()).strip("_")
    return safe or "run"

def spawn_embedded_target_server(
    *,
    run_id: str,
    benchmark_root: Path | None,
    log_path: Path | None = None,
    ready_timeout: float = 120.0,
    extra_env: dict[str, str] | None = None,
) -> EmbeddedTargetServer:
    """Launch a per-run target_server subprocess and wait for ``/openapi.json``.

    If ``log_path`` is provided, the child's stdout+stderr are written there
    (parent dirs are created as needed). Otherwise they are discarded; real
    runs pass a per-run path so 500s from ``/launch/...`` are attributable.

    ``extra_env`` is layered on top of the parent's environment. Callers use
    it to forward timeout knobs such as ``TARGET_SERVER_COMPOSE_UP_TIMEOUT_S``
    and ``TARGET_SERVER_STARTUP_TIMEOUT_S`` without mutating ``os.environ``.
    """
    port = _pick_free_port()
    namespace = _namespace_from_run_id(run_id)
    cmd = [
        sys.executable, "-m", "cage.target.serve",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--namespace", namespace,
        # Embedded servers run with start_new_session=True (so a terminal Ctrl+C
        # can't kill them before controlled teardown). Pass our own pid so the
        # server's watchdog self-exits if cage dies ungracefully (SIGKILL/OOM),
        # leaving nothing orphaned — this is what piled up under repeated
        # targets-check runs. Passing the pid (vs the child reading getppid)
        # also catches a cage death during the server's multi-second startup.
        "--parent-pid", str(os.getpid()),
    ]
    if benchmark_root is not None:
        cmd += ["--benchmark-root", str(benchmark_root)]
    env = os.environ.copy()
    # Scope the server's namespace startup-GC to THIS benchmark's runs. Without
    # it, ``default_cage_runs_roots()`` scans every ``.cage_runs/`` under the repo
    # — including huge sibling runs (e.g. cybergym trials with 100k+-file
    # ``workspace/`` trees) — so the GC can take 30s+ and push the server past the
    # readiness deadline. The embedded server's namespace is per-run, so its own
    # benchmark root is the only relevant liveness source. An explicit
    # ``CAGE_RUNS_ROOT`` in the environment still wins.
    if benchmark_root is not None:
        env.setdefault("CAGE_RUNS_ROOT", str((benchmark_root / ".cage_runs").resolve()))
    if extra_env:
        env.update(extra_env)
    logger.debug(
        "embedded target_server: spawning port=%d namespace=%s root=%s log=%s",
        port, namespace, benchmark_root, log_path,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab", buffering=0)
        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                # New process group so SIGTERM/SIGINT delivered to cage don't
                # kill the child before we've had a chance to clean up.
                start_new_session=True,
            )
        finally:
            # Parent doesn't need its handle once Popen has dup'd the fd
            # into the child; close immediately so we don't leak a fd per run.
            log_fh.close()
    else:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    server_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + ready_timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"embedded target_server exited early (code={process.returncode}) "
                f"before becoming ready"
            )
        try:
            with urllib.request.urlopen(f"{server_url}/openapi.json", timeout=1.0) as resp:
                if resp.status == 200:
                    return EmbeddedTargetServer(
                        process=process, port=port,
                        namespace=namespace, server_url=server_url,
                    )
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.3)
    # Ready-probe timed out — stop the subprocess and raise.
    try:
        process.terminate()
        process.wait(timeout=2)
    except Exception:
        pass
    raise TimeoutError(
        f"embedded target_server did not become ready within {ready_timeout:.0f}s: "
        f"{last_error}"
    )

def target_challenge_id(sample: dict[str, Any], fallback: str = "") -> str:
    """Resolve the target challenge id from a sample.

    Sample ``id`` and target ``challenge_id`` are intentionally different concepts:
    a single target challenge can produce multiple samples (per-variant, per-seed,
    etc.), so sample ``id`` must be unique per trial, while the target_server server is
    indexed by the underlying challenge identity. Benchmarks should set
    ``challenge_id`` explicitly when they expand challenges into multiple
    samples; otherwise we fall back to ``id`` for compatibility.
    """
    cid = sample.get("challenge_id")
    if cid:
        return str(cid)
    return str(sample.get("id") or fallback or "")

def target_runtime_args(
    run: ExperimentRun,
    sample: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build the ``runtime_args`` dict passed to ``ChallengeClient.get_challenge_data``.

    Resolution order (later wins):
      1. ``TargetConfig.target_scope`` / ``TargetConfig.parallel_mode``
      2. ``sample["runtime_args"]`` — lets benchmarks override per-sample
         (e.g. one variant wants alias mode while others want network mode).

    Empty values are dropped so the server can fall back to its own
    universal defaults: ``parallel_mode="network"`` (gates IPAM injection
    on compose runtime networks) and ``target_scope="per_challenge"``
    (one compose stack reused across passes). Benchmarks that need other
    values must declare them explicitly on the challenge or in
    ``runtime_args``.
    """
    args: dict[str, str] = {}
    target_scope = str(run.target.target_scope or "").strip()
    if target_scope:
        args["target_scope"] = target_scope
    parallel_mode = str(run.target.parallel_mode or "").strip()
    if parallel_mode:
        args["parallel_mode"] = parallel_mode
    network_mode = str(run.target.network_mode or "").strip()
    if network_mode:
        args["network_mode"] = network_mode
    exposure_mode = str(run.target.exposure_mode or "").strip()
    if exposure_mode:
        args["exposure_mode"] = exposure_mode
    if sample is not None:
        per_sample = sample.get(SAMPLE_RUNTIME_ARGS_KEY)
        if isinstance(per_sample, dict):
            for key in ("target_scope", "parallel_mode", "network_mode", "exposure_mode"):
                value = str(per_sample.get(key, "") or "").strip()
                if value:
                    args[key] = value
    return args

def target_server_timeout_env(
    target_config: Any,
    *,
    benchmark_id: str = "",
) -> dict[str, str]:
    """Environment overrides for the embedded target_server used by ``cage run``."""

    env: dict[str, str] = {"TARGET_SERVER_BUILD_IF_MISSING": "0"}
    if benchmark_id:
        env["CAGE_BENCHMARK_ID"] = benchmark_id
    compose_up_timeout = getattr(target_config, "compose_up_timeout", None)
    if compose_up_timeout is not None:
        env["TARGET_SERVER_COMPOSE_UP_TIMEOUT_S"] = str(float(compose_up_timeout))
    startup_timeout = getattr(target_config, "startup_timeout", None)
    if startup_timeout is not None:
        env["TARGET_SERVER_STARTUP_TIMEOUT_S"] = str(float(startup_timeout))
    return env

def target_launch_request_timeout_s(target_config: Any) -> float:
    """Client-side timeout for ``GET /launch``.

    The endpoint is synchronous. A cold launch can spend time in compose build,
    compose up, and several readiness probes; the client timeout must exceed
    the server-side caps or it will abort first.
    """

    compose_up_timeout = getattr(target_config, "compose_up_timeout", None)
    startup_timeout = getattr(target_config, "startup_timeout", None)
    if compose_up_timeout is None and startup_timeout is None:
        return 300.0

    compose_s = float(compose_up_timeout if compose_up_timeout is not None else 1200.0)
    startup_s = float(startup_timeout if startup_timeout is not None else 120.0)
    return max(300.0, (2.0 * compose_s) + (3.0 * startup_s) + 60.0)

def challenges_from_benchmark(run: ExperimentRun) -> dict[str, Any]:
    """Pull the challenge dict from a benchmark that exposes its own ChallengeClient.

    Some benchmarks load and normalize their
    challenge metadata during ``setup()`` and stash it on a ``challenge_client``
    attribute. When that's available, the orchestrator should use those
    challenges rather than re-discovering them from a hardcoded path that
    may not match the benchmark's actual ``benchmark_root`` (which is often
    set in project.yml to an absolute path elsewhere on disk).
    """
    bench = run.benchmark
    bench_cm = getattr(bench, "challenge_client", None)
    if bench_cm is None:
        return {}
    challenges = getattr(bench_cm, "challenges", None)
    if isinstance(challenges, dict):
        return dict(challenges)
    return {}

def build_target_config(
    run: ExperimentRun,
    cage_runs: Path,
    *,
    run_id: str = "",
) -> ChallengeClientConfig | None:
    """Build ChallengeClientConfig from ExperimentRun, or None if targets are disabled.

    Resolves challenges in this priority order:
      1. ``run.benchmark.challenge_client.challenges`` — benchmark-owned (preferred).
      2. ``<project_dir>/datasets/*.json`` — legacy convention.
    """
    if not run.target.enabled:
        return None

    challenges = challenges_from_benchmark(run)
    if not challenges:
        for challenge_json in Path(cage_runs.parent / "datasets").glob("*.json"):
            with open(challenge_json, "r") as f:
                challenges = json.load(f)

    return ChallengeClientConfig(
        challenges=challenges,
        run_mode=run.target.run_mode,
        server_url=run.target.server_url,
        use_ssh_tunnel=run.target.use_ssh_tunnel,
        ssh_config=SSHConfig(
            jump_host=run.target.jump_host,
            jump_user=run.target.jump_user,
            ssh_key_path=run.target.ssh_key_path,
            remote_bind_address=run.target.remote_bind_address,
            remote_bind_port=run.target.remote_bind_port,
        ),
        use_external_access=run.target.use_external_access,
        host_ip_for_agent=run.target.host_ip_for_agent,
        network_name=run.target.network_name,
        launch_timeout_s=target_launch_request_timeout_s(run.target),
        cage_run_id=run_id,
    )

def inject_ctf_info(sample: dict[str, Any], target_data: dict[str, Any]) -> None:
    """Inject target stack information into the benchmark sample.

    Always injects whatever the server returned that may be useful to the
    prompt template:

    - ``network_name``  — set whenever the server returned one.
    - ``target_info``   — set whenever the server returned non-empty service
      info, regardless of ``parallel_mode``. Network-mode targets often
      still expose a fixed ``target`` alias; dropping it forced every
      network-mode benchmark to also embed the subnet → service-name
      mapping in its prompt by hand.
    - ``network_subnet`` — set only in network mode, since that's where the
      agent is supposed to discover services by scanning the subnet.
    """
    runtime = target_data.get("runtime", {})
    target_info = target_data.get("target_info", {})

    network_name = runtime.get("network_name")
    if network_name:
        sample[SAMPLE_NETWORK_NAME_KEY] = network_name

    project_name = runtime.get("project_name")
    run_id = runtime.get("run_id")
    if project_name or run_id:
        runtime_state: RuntimeState = sample.setdefault(RUNTIME_STATE_KEY, {})
        if isinstance(runtime_state, dict):
            if project_name:
                runtime_state["project_name"] = project_name
            if run_id:
                runtime_state["run_id"] = run_id

    scoring = runtime.get("scoring")
    if isinstance(scoring, dict) and scoring:
        metadata = sample.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["runtime_scoring"] = dict(scoring)
        runtime_state = sample.setdefault(RUNTIME_STATE_KEY, {})
        if isinstance(runtime_state, dict):
            runtime_state["scoring"] = dict(scoring)

    if target_info:
        sample[SAMPLE_TARGET_INFO_KEY] = target_info

    debug = runtime.get("debug", {})
    parallel_mode = debug.get("parallel_mode", "") if isinstance(debug, dict) else ""
    if parallel_mode == "network":
        network_subnet = runtime.get("network_subnet")
        if network_subnet:
            sample[SAMPLE_NETWORK_SUBNET_KEY] = network_subnet

@dataclass
class AgentIsolationNetwork:
    """A cage-owned docker bridge created per trial.

    Only the agent and the *intended* target service containers are attached
    (one network connect per public service, aliased to the service name).
    Internal services (db, secrets_init, cache, …) on target_server's compose
    project networks stay unreachable regardless of how the adapter
    configured its own networking.

    The agent's pre-existing default bridge connection is kept for outbound
    HTTPS to the upstream model proxy (``host.docker.internal``).

    ``subnet`` is the cidr we asked docker to use (from the cage-trial pool).
    ``None`` means we fell back to docker's default address pool — log line
    in the builder explains why.
    """

    name: str
    connected_targets: list[str]
    subnet: str | None = None

    def teardown(self) -> bool:
        """Disconnect public targets, remove the bridge, and return rm success.

        The boolean return is the daemon-side cleanup fact: ``True`` means the
        bridge was removed or was already gone, ``False`` means Docker did not
        confirm removal. Callers use this to write accurate ResourceLedger
        cleanup status instead of guessing from the absence of an exception.
        """

        for container_name in list(self.connected_targets):
            try:
                subprocess.run(
                    ["docker", "network", "disconnect", "-f", self.name, container_name],
                    text=True,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=30.0,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to disconnect %s from %s: %s",
                    container_name, self.name, exc,
                )
        rm_succeeded = False
        try:
            result = subprocess.run(
                ["docker", "network", "rm", self.name],
                text=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30.0,
            )
            stderr_lower = (result.stderr or "").lower()
            if (
                result.returncode == 0
                or "no such network" in stderr_lower
                or "not found" in stderr_lower
            ):
                rm_succeeded = True
            else:
                logger.warning(
                    "Failed to remove %s: %s", self.name, result.stderr[:200],
                )
        except Exception as exc:
            logger.warning("docker network rm %s raised: %s", self.name, exc)
        # Only release the /26 back to the pool when the daemon-side network
        # is actually gone. Releasing while the network still exists would
        # let the next ``allocate_cage_trial_subnet`` pick a candidate that
        # overlaps the (still alive) network → ``_collect_used_docker_subnets``
        # would skip it correctly, but in the limit (all reservations
        # released, but networks still alive) the pool reports "exhausted"
        # for the wrong reason. Better to leak the reservation slot — the
        # next ``cleanup_orphan_networks`` sweep removes the network and
        # releases the slot via the IPAM-config code path.
        if rm_succeeded:
            release_cage_trial_subnet(self.subnet)
        elif self.subnet:
            logger.warning(
                "Leaking cage-trial subnet reservation %s because docker network "
                "rm of %s did not confirm removal; orphan reclaim will recover it.",
                self.subnet,
                self.name,
            )
        return rm_succeeded

def public_target_services(target_data: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``[(service_name, container_name), ...]`` for public targets.

    "Public" = the service has ``external_port`` set in the target_server launch
    response (i.e. the adapter intentionally exposed it). Internal services
    that the agent should not reach (db, secrets_init, cache, …) have no
    external_port and are filtered out here.

    Container names come from ``runtime.debug.network.services``, which
    target_server builds by inspecting the per-project compose network. If that
    section is missing (legacy server, local backend), we can't enforce
    isolation and return an empty list — caller falls back to the
    server-returned network.
    """
    target_info = target_data.get("target_info", {}) or {}
    public_names = {
        name
        for name, svc in target_info.items()
        if isinstance(svc, dict) and svc.get("external_port")
    }
    if not public_names:
        return []
    debug = (target_data.get("runtime", {}) or {}).get("debug", {}) or {}
    network_block = debug.get("network", {}) or {}
    service_to_container: dict[str, str] = {}
    for entry in network_block.get("services", []) or []:
        if not isinstance(entry, dict):
            continue
        svc_name = entry.get("service_name")
        cn = entry.get("container_name")
        if isinstance(svc_name, str) and isinstance(cn, str) and svc_name and cn:
            service_to_container[svc_name] = cn
    return [
        (name, service_to_container[name])
        for name in sorted(public_names)
        if name in service_to_container
    ]

def build_agent_isolation_network(
    trial_id: str,
    target_data: dict[str, Any],
    agent_container_name: str | None = None,
) -> AgentIsolationNetwork | None:
    """Create a cage-owned bridge with only public target services attached.

    The network name is keyed by ``(trial_id, agent_container_name)`` so that
    multiple agents running the same trial in parallel each get their own
    isolation bridge — otherwise they race to create the same network and the
    losers fall back to the shared run network, where parallel target stacks
    each register the same service alias (``web``, …) and DNS becomes
    nondeterministic.

    Returns ``None`` when prerequisites aren't met (no public services,
    or no known container names), in which case the orchestrator falls back
    to attaching the agent to the server-returned network as before.
    """
    targets = public_target_services(target_data)
    if not targets:
        return None

    name_parts = ["cage-trial", _safe_docker_name_component(trial_id)]
    if agent_container_name:
        agent_suffix = hashlib.sha1(
            agent_container_name.encode("utf-8")
        ).hexdigest()[:8]
        name_parts.append(agent_suffix)
    network_name = "-".join(name_parts)

    # Allocate a /26 (default) from the cage-trial pool BEFORE invoking
    # ``docker network create`` so docker doesn't auto-grab a full /16
    # from its default address pool. Pool exhaustion is logged and we
    # fall back to docker's pool — a leaked /16 is better than a failed
    # trial.
    #
    # Race tolerance (reviewer M1): if ``docker network create`` rejects
    # the /26 with a "overlaps" / "already in use" complaint — meaning a
    # peer process took the slot between our ``_collect_used_docker_subnets``
    # and this create — release the reservation and retry once with a
    # fresh allocation. Persistent conflict (twice in a row) falls back
    # to the server-returned network.
    def _alloc_one() -> str | None:
        try:
            return allocate_cage_trial_subnet(network_name)
        except RuntimeError as exc:
            logger.warning(
                "Cage isolation: cage-trial subnet pool exhausted (%s); falling "
                "back to docker's default address pool for %s",
                exc,
                network_name,
            )
            return None

    subnet_cidr: str | None = _alloc_one()
    create = None
    # Track every cidr we tried so the retry path doesn't keep the failed
    # ones reserved (preventing the allocator from re-picking them).
    failed_cidrs: list[str] = []
    # Stable label set so a peer cage server can auto-reclaim this bridge
    # if our orchestrator dies before teardown (network_admin's
    # cleanup_orphan_networks scans cage-trial-* networks tagged with
    # ``cage.network.kind=cage-trial`` and reclaims those whose owner PID
    # is no longer alive).
    base_labels = [
        "--label", "cage.network.kind=cage-trial",
        "--label", f"cage.network.owner-pid={os.getpid()}",
    ]
    for attempt in range(2):
        create_argv = ["docker", "network", "create", "--driver", "bridge"]
        if subnet_cidr:
            create_argv += ["--subnet", subnet_cidr]
        create_argv += list(base_labels)
        if subnet_cidr:
            create_argv += ["--label", f"cage.network.subnet={subnet_cidr}"]
        create_argv.append(network_name)
        create = subprocess.run(
            create_argv,
            text=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30.0,
        )
        if create.returncode == 0:
            break
        stderr_lower = (create.stderr or "").lower()
        is_subnet_race = subnet_cidr and (
            "overlaps" in stderr_lower
            or "already in use" in stderr_lower
            or "pool" in stderr_lower
        )
        if attempt == 0 and is_subnet_race:
            logger.info(
                "Cage isolation: %s race-conflicted (%s); retrying with fresh /26",
                subnet_cidr, (create.stderr or "")[:200],
            )
            # Keep the failed cidr in our process-local reserved set so
            # ``_alloc_one`` skips it on the next pick (otherwise the
            # deterministic seed yields the same /26).
            failed_cidrs.append(subnet_cidr)
            subnet_cidr = _alloc_one()
            continue
        break

    # Release ALL failed cidrs back to the pool (they're not in use by us).
    for stale in failed_cidrs:
        release_cage_trial_subnet(stale)

    if create is None or create.returncode != 0:
        msg = (create.stderr if create is not None else "no result")[:200]
        logger.warning(
            "Cage isolation: failed to create %s (%s); "
            "falling back to server-returned network",
            network_name, msg,
        )
        release_cage_trial_subnet(subnet_cidr)
        return None

    connected: list[str] = []
    for service_name, container_name in targets:
        result = subprocess.run(
            [
                "docker", "network", "connect",
                "--alias", service_name,
                network_name, container_name,
            ],
            text=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30.0,
        )
        if result.returncode != 0:
            logger.warning(
                "Cage isolation: failed to connect %s as %s on %s: %s",
                container_name, service_name, network_name, result.stderr[:200],
            )
            continue
        connected.append(container_name)

    if not connected:
        subprocess.run(
            ["docker", "network", "rm", network_name],
            text=True, stdin=subprocess.DEVNULL,
            capture_output=True, timeout=30.0,
        )
        release_cage_trial_subnet(subnet_cidr)
        return None

    logger.info(
        "Cage isolation: created %s (subnet=%s) with %d public target(s): %s",
        network_name,
        subnet_cidr or "docker-default",
        len(connected),
        ", ".join(name for name, _ in targets if _ in connected),
    )
    return AgentIsolationNetwork(
        name=network_name,
        connected_targets=connected,
        subnet=subnet_cidr,
    )

def attach_agent_to_target(
    container: Container,
    trial_id: str,
    target_data: dict[str, Any],
    server_network: str | None,
    isolation_policy: str,
) -> tuple[str | None, AgentIsolationNetwork | None]:
    """Resolve which docker network the agent should join, honouring the
    configured isolation policy.

    Returns ``(network_name, isolation)`` where:

    - ``network_name`` is what was passed to ``container.sync_runtime_network``;
      ``None`` if no attachment was performed.
    - ``isolation`` is a teardown handle for the cage-private bridge, or
      ``None`` if we fell back to the server-returned network.
    """
    if container.network_mode == "host":
        if isolation_policy == "per_trial_bridge":
            logger.warning(
                "Cage isolation requested but agent container uses host "
                "networking — no docker namespace isolation is possible.",
            )
        return None, None

    if isolation_policy == "per_trial_bridge":
        isolation = build_agent_isolation_network(
            trial_id, target_data, getattr(container, "name", None),
        )
        if isolation is not None:
            container.sync_runtime_network(isolation.name)
            return isolation.name, isolation
        if server_network:
            logger.warning(
                "Cage isolation: could not enumerate public target containers "
                "for trial %s; falling back to server network %s",
                trial_id, server_network,
            )

    if server_network:
        container.sync_runtime_network(server_network)
        return server_network, None
    return None, None

def _active_runtime_network(container: Container, fallback: Any = None) -> str | None:
    """Return the runtime network currently associated with the agent container."""
    network_name = getattr(container, "_runtime_network_name", None)
    if isinstance(network_name, str) and network_name.strip():
        return network_name.strip()
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    return None

def capture_trial_check_done(
    *,
    benchmark: Benchmark,
    container: Container,
    sample: dict[str, Any],
    trial_dir: Path,
    trial_id: str,
    agent_output_dir: Path | None = None,
) -> Path | None:
    """Run the benchmark scorer's live ``gather`` and save the raw evidence.

    Returns the artifact path, or ``None`` when the scorer gathers no live
    evidence (returns empty string). The artifact keeps its
    ``check_done_output.txt`` name — that is the path ``ScoringContext``
    reads back post-trial.
    """
    try:
        from cage.scoring import GatherRuntime

        output = benchmark.scorer().gather(
            GatherRuntime(
                sample=sample,
                container=container,
                agent_output_dir=agent_output_dir,
            )
        )
    except Exception as exc:
        output = f"gather failed: {exc}"
        logger.warning("scorer.gather capture failed for trial %s: %s", trial_id, exc)

    if not output:
        return None

    runtime_dir = trial_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output_path = runtime_dir / "check_done_output.txt"
    output_path.write_text(str(output or ""), encoding="utf-8")
    logger.info("Saved check_done output for trial %s: %s", trial_id, output_path)
    return output_path

def create_check_network(*, run_id: str, trial_id: str) -> str:
    """Create an isolated bridge network for a check container."""
    network_name = (
        f"problem-check-net-"
        f"{_safe_docker_name_component(run_id)}-"
        f"{_safe_docker_name_component(trial_id)}"
    )
    result = subprocess.run(
        ["docker", "network", "create", "--driver", "bridge", network_name],
        text=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30.0,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create check network {network_name}: {result.stderr[:500]}"
        )
    logger.info("Created check network for trial %s: %s", trial_id, network_name)
    return network_name

def remove_check_network(network_name: str) -> None:
    """Remove an isolated check network created by the orchestrator."""
    result = subprocess.run(
        ["docker", "network", "rm", network_name],
        text=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30.0,
    )
    if result.returncode != 0:
        logger.warning(
            "Failed to remove check network %s: %s",
            network_name,
            result.stderr[:200],
        )
    else:
        logger.info("Removed check network: %s", network_name)
