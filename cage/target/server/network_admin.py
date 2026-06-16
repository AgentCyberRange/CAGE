"""Docker network/container administration helpers."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from docker.errors import APIError, NotFound

from cage.target.server.network_alloc import (
    allocate_agent_home_subnet,
    release_agent_home_subnet,
    release_cage_trial_subnet,
    release_reserved_project_local_subnet,
)
from cage.target.server.server_state import (
    DOCKER_NETWORK,
    NETWORK_REMOVE_RETRY_INTERVAL_S,
    NETWORK_REMOVE_RETRY_TIMEOUT_S,
    ORPHAN_NETWORK_MIN_AGE_S,
    TARGET_SERVER_NAMESPACE,
    get_docker_client,
)

try:  # docker-py IPAM helpers — used to size the agent_home network.
    from docker.types import IPAMConfig, IPAMPool
    _IPAM_SUPPORT = True
except Exception:  # pragma: no cover — docker-py too old in CI
    IPAMConfig = None  # type: ignore[assignment]
    IPAMPool = None  # type: ignore[assignment]
    _IPAM_SUPPORT = False

logger = logging.getLogger(__name__)


# Shared prefix every parent ``DOCKER_NETWORK`` carries (regardless of
# namespace). Used by ``cleanup_orphan_networks`` to enumerate other
# namespaces' parent home networks for liveness-based reclamation.
_CAGE_BENCH_NETWORK_PREFIX = "cage_bench_"

# Prefix the per-trial isolation networks carry (see
# ``cage/target/provisioning.py::build_agent_isolation_network``). Reclaimed
# by ``cleanup_orphan_networks`` when the owner PID has died — guards
# against trial-runner crashes leaving network leaks.
_CAGE_TRIAL_NETWORK_PREFIX = "cage-trial-"


def _server_network_labels(role: str) -> dict[str, str]:
    return {
        "cage.target.namespace": TARGET_SERVER_NAMESPACE,
        "cage.target.network": DOCKER_NETWORK,
        "cage.target.role": role,
        # Owner PID for liveness-based orphan reclamation. A peer cage
        # server scanning orphans treats this PID as the canonical hint
        # for whether the owner is still alive (combined with the
        # namespace-cmdline scan as belt-and-suspenders).
        "cage.target.pid": str(os.getpid()),
    }


def _is_namespace_alive(namespace: str, pid_hint: Optional[str] = None) -> bool:
    """Return True if some live process is the cage-server owning ``namespace``.

    Two-step liveness check:
      1. If ``pid_hint`` parses as int and ``/proc/<pid>`` exists with the
         expected ``cage`` cmdline, treat as alive (fast path; survives
         server restarts that re-use the same namespace label only if the
         old PID is reused).
      2. Fall back to scanning ``/proc/*/cmdline`` for any process whose
         args contain ``--namespace`` immediately followed by the target
         namespace string, or ``--namespace=<ns>``.

    Returns False on read errors / non-Linux hosts — orphan removal is
    additionally gated by ``ORPHAN_NETWORK_MIN_AGE_S``, so a false negative
    here just defers reclamation, never deletes a live peer's resources.
    """
    if not namespace:
        return False
    proc_root = Path("/proc")
    if not proc_root.exists():
        return False  # not a procfs host — bail safely

    namespace_bytes = namespace.encode("utf-8", errors="replace")
    flag_kv_bytes = f"--namespace={namespace}".encode("utf-8", errors="replace")

    def _cmdline_matches(pid_dir: Path) -> bool:
        try:
            raw = (pid_dir / "cmdline").read_bytes()
        except (OSError, PermissionError):
            return False
        if not raw:
            return False
        # cmdline is a NUL-separated argv list with a trailing NUL.
        argv = raw.split(b"\x00")
        # Cage server processes contain the internal target_server module,
        # the old ``cage.cli serve`` module form, or the ``cage`` console
        # script entrypoint in argv. Without this filter
        # we'd false-positive on any process whose args happen to mention
        # the namespace string (e.g. logs, docker shell-outs).
        joined = b" ".join(argv)
        module_markers = (b"cage.target.serve", b"cage.cli")
        if (
            not any(marker in joined for marker in module_markers)
            and b"/cage " not in (b" " + joined + b" ")
        ):
            # The ``/cage `` form catches the installed console script
            # (``/.../bin/cage``); the leading/trailing spaces avoid
            # matching unrelated paths containing ``cage`` as a substring.
            if not any(arg.endswith(b"/cage") or arg == b"cage" for arg in argv):
                return False
        for i, arg in enumerate(argv):
            if arg == flag_kv_bytes:
                return True
            if arg == b"--namespace" and i + 1 < len(argv) and argv[i + 1] == namespace_bytes:
                return True
        return False

    # Fast path: trust the PID label if its cmdline still matches.
    if pid_hint:
        try:
            pid_int = int(str(pid_hint).strip())
        except (TypeError, ValueError):
            pid_int = None
        if pid_int and pid_int > 0:
            pid_dir = proc_root / str(pid_int)
            if pid_dir.exists() and _cmdline_matches(pid_dir):
                return True
            # PID hint stale (process died, or PID reused for unrelated
            # program). Fall through to full scan rather than declaring
            # dead — another server in the same namespace may exist.

    try:
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            if _cmdline_matches(entry):
                return True
    except OSError:
        return False
    return False


def _is_pid_alive(pid_hint: Any) -> bool:
    """Return True iff /proc/<pid_hint> exists.

    Used by the cage-trial reclamation path: orchestrator stamps its PID
    onto each isolation network at create time, and ``cleanup_orphan_networks``
    treats a network as orphaned when its owner PID is no longer in
    ``/proc``. No cmdline check here — cage-trial networks are uniquely
    namespaced by name + label, so PID-death is enough.

    Returns False on non-procfs hosts or invalid hints.
    """
    try:
        pid_int = int(str(pid_hint).strip())
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    return (Path("/proc") / str(pid_int)).exists()


def _parse_docker_created(raw: Any) -> Optional[datetime]:
    """Parse docker's ``Created`` ISO-8601 timestamp.

    Returns ``None`` on any parse failure — the caller falls back to
    treating the network as old-enough-to-evaluate (paired with an
    explicit age threshold check separately).
    """
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Docker emits e.g. ``2026-05-17T00:09:07.282784732Z`` and sometimes
    # high-precision fractional seconds beyond what ``fromisoformat``
    # accepts pre-3.11. Trim to microseconds + replace trailing Z.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, frac = text.split(".", 1)
        # Split fraction from timezone suffix (``+00:00`` or ``Z``-replaced).
        for tz_sep in ("+", "-"):
            if tz_sep in frac:
                idx = frac.index(tz_sep)
                tz = frac[idx:]
                frac = frac[:idx]
                break
        else:
            tz = ""
        # Truncate sub-microsecond digits.
        frac = frac[:6]
        text = f"{head}.{frac}{tz}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _network_age_seconds(attrs: dict[str, Any]) -> Optional[float]:
    parsed = _parse_docker_created(attrs.get("Created"))
    if parsed is None:
        return None
    return (datetime.now(tz=timezone.utc) - parsed).total_seconds()


def list_project_containers(project_name: str):
    """Return all containers (any state) for a docker compose project."""
    try:
        return get_docker_client().containers.list(
            all=True,
            filters={"label": f"com.docker.compose.project={project_name}"},
        )
    except Exception as e:
        logger.warning(f"[Docker] Failed to list containers for project {project_name}: {e}")
        return []


def summarize_project_containers(project_name: str, max_logs_tail: int = 40) -> List[dict]:
    summaries: List[dict] = []
    for c in list_project_containers(project_name):
        try:
            c.reload()
            state = (c.attrs or {}).get("State", {}) if hasattr(c, "attrs") else {}
            status = state.get("Status") or getattr(c, "status", None)
            exit_code = state.get("ExitCode")
            summary = {
                "name": c.name,
                "service": (c.labels or {}).get("com.docker.compose.service"),
                "status": status,
                "exit_code": exit_code,
                "error": state.get("Error"),
            }

            if status not in ("running",) and max_logs_tail > 0:
                try:
                    raw = c.logs(tail=max_logs_tail)
                    if isinstance(raw, (bytes, bytearray)):
                        summary["logs_tail"] = raw.decode("utf-8", errors="replace")
                    else:
                        summary["logs_tail"] = str(raw)
                except Exception:
                    pass

            if summary.get("logs_tail") and len(summary["logs_tail"]) > 4000:
                summary["logs_tail"] = summary["logs_tail"][-4000:]

            summaries.append(summary)
        except Exception as e:
            summaries.append({"name": getattr(c, "name", "<unknown>"), "error": f"failed to summarize: {e}"})
    return summaries


def capture_project_logs(
    project_name: str, *, max_chars_per_container: int = 50_000
) -> List[dict]:
    """Capture logs for *every* container in a compose project, for audit.

    Unlike ``summarize_project_containers`` (40-line tail, only non-running
    containers), this grabs the full log of every container regardless of
    state — so a crash-looping failure (e.g. a JVM cgroup-v2 NPE) and a clean
    successful teardown both leave a complete, auditable record. Each
    container's log is tail-capped to ``max_chars_per_container`` to bound
    memory and on-disk artifact size.

    MUST be called BEFORE the project is purged — once containers are removed
    their logs are gone for good.
    """
    out: List[dict] = []
    for c in list_project_containers(project_name):
        entry: dict = {"name": getattr(c, "name", "<unknown>")}
        try:
            c.reload()
            state = (c.attrs or {}).get("State", {}) if hasattr(c, "attrs") else {}
            entry["service"] = (c.labels or {}).get("com.docker.compose.service")
            entry["status"] = state.get("Status") or getattr(c, "status", None)
            entry["exit_code"] = state.get("ExitCode")
            entry["error"] = state.get("Error") or None
            try:
                raw = c.logs(timestamps=True)
                text = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, (bytes, bytearray))
                    else str(raw)
                )
            except Exception as exc:
                text = f"<failed to read logs: {exc}>"
            if len(text) > max_chars_per_container:
                text = "...(head truncated)...\n" + text[-max_chars_per_container:]
            entry["logs"] = text
        except Exception as exc:
            entry["error"] = f"failed to capture: {exc}"
        out.append(entry)
    return out


def resolve_service_inner_ips(project_name: str, network_name: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for container in list_project_containers(project_name):
        labels = getattr(container, "labels", {}) or {}
        service_name = labels.get("com.docker.compose.service")
        if not service_name:
            continue
        try:
            container.reload()
            networks = ((container.attrs or {}).get("NetworkSettings", {}) or {}).get("Networks", {}) or {}
            network_info = networks.get(network_name, {}) or {}
            ip_address = str(network_info.get("IPAddress", "") or "").strip()
            if ip_address:
                mapping[service_name] = ip_address
        except Exception as e:
            logger.warning("Failed to resolve inner IP for %s/%s: %s", project_name, service_name, e)
    return mapping


def _create_agent_home_network() -> None:
    """Create DOCKER_NETWORK with an explicit /24 from the cage agent_home pool.

    Falls back to docker's default address pool when:
      - docker-py is too old to expose ``IPAMConfig`` / ``IPAMPool``, or
      - the pool is fully occupied (logged, then continues — better to
        create a /16 from docker default than to crash the server).

    The chosen subnet is stamped into the network labels so peer
    ``cleanup_orphan_networks`` can release it back to the pool when the
    namespace dies without graceful shutdown.

    Race tolerance (reviewer M1): if ``client.networks.create`` raises a
    "Pool overlaps with other one" / 409 — meaning the daemon admitted
    a competing network between our ``_collect_used_docker_subnets`` and
    the create call — release our reservation and retry once with a
    fresh allocation. A persistent conflict (twice in a row) propagates.
    """
    client = get_docker_client()
    max_attempts = 2 if _IPAM_SUPPORT else 1
    last_exc: Exception | None = None
    # Same retry pattern as orchestrator.build_agent_isolation_network:
    # keep failed cidrs reserved during the loop so the deterministic
    # allocator picks a different /24 on the second attempt; release them
    # at the end.
    failed_cidrs: list[str] = []
    for attempt in range(max_attempts):
        chosen_cidr: str | None = None
        ipam = None
        if _IPAM_SUPPORT:
            try:
                chosen_cidr = allocate_agent_home_subnet(DOCKER_NETWORK)
                ipam = IPAMConfig(pool_configs=[IPAMPool(subnet=chosen_cidr)])
            except RuntimeError as exc:
                logger.warning(
                    "agent_home pool fully occupied; falling back to docker default "
                    "address pool for %s: %s",
                    DOCKER_NETWORK,
                    exc,
                )
        labels = _server_network_labels("agent_home")
        if chosen_cidr:
            labels["cage.target.subnet"] = chosen_cidr
        try:
            if ipam is not None:
                client.networks.create(
                    DOCKER_NETWORK,
                    driver="bridge",
                    ipam=ipam,
                    labels=labels,
                )
            else:
                client.networks.create(
                    DOCKER_NETWORK,
                    driver="bridge",
                    labels=labels,
                )
            logger.info(
                "Created network %r (subnet=%s)",
                DOCKER_NETWORK,
                chosen_cidr or "docker-default",
            )
            for stale in failed_cidrs:
                release_agent_home_subnet(stale)
            return
        except APIError as exc:
            last_exc = exc
            msg = str(exc).lower()
            if attempt + 1 < max_attempts and chosen_cidr and ("overlap" in msg or "already" in msg):
                logger.info(
                    "agent_home subnet %s conflicts with a peer-created network "
                    "(daemon: %s); retrying with fresh allocation",
                    chosen_cidr,
                    str(exc)[:200],
                )
                # Keep failed cidr reserved during retry so the next pick
                # skips it.
                failed_cidrs.append(chosen_cidr)
                continue
            release_agent_home_subnet(chosen_cidr)
            for stale in failed_cidrs:
                release_agent_home_subnet(stale)
            raise
        except Exception:
            release_agent_home_subnet(chosen_cidr)
            for stale in failed_cidrs:
                release_agent_home_subnet(stale)
            raise
    # All attempts exhausted; release any reservations and re-raise.
    for stale in failed_cidrs:
        release_agent_home_subnet(stale)
    if last_exc is not None:
        raise last_exc


def ensure_docker_network() -> None:
    """Claim exclusive ownership of DOCKER_NETWORK for this server process.

    Policy (option A, strict fail-fast):
      - Network does not exist         → create it (with /24 ipam, see
        ``_create_agent_home_network``).
      - Network exists and is empty    → leftover from previous crash;
        delete + recreate so the recreated copy carries our PID label +
        comes from the managed pool.
      - Network exists with containers → another server with the same
        TARGET_SERVER_NAMESPACE is (or was) running. Refuse to start.
    """
    client = get_docker_client()
    try:
        network = client.networks.get(DOCKER_NETWORK)
    except NotFound:
        _create_agent_home_network()
        return

    try:
        network.reload()
    except Exception as exc:
        raise RuntimeError(
            f"Docker network '{DOCKER_NETWORK}' exists but inspect failed: {exc}. "
            f"Refusing to start."
        ) from exc

    containers = network.attrs.get("Containers", {}) or {}
    if containers:
        names = sorted((c or {}).get("Name", "<unknown>") for c in containers.values())
        raise RuntimeError(
            f"Docker network '{DOCKER_NETWORK}' already has {len(containers)} "
            f"attached container(s): {names}. Another Challenge server with "
            f"TARGET_SERVER_NAMESPACE may be running. Refusing to start.\n"
            f"If you are certain no other server is running, remove it manually: "
            f"docker network rm {DOCKER_NETWORK}"
        )

    logger.info(
        f"Docker network '{DOCKER_NETWORK}' exists but is empty "
        f"(leftover from previous run); removing and recreating."
    )
    # Release the old subnet back to the pool if it was in the cage range.
    old_labels = network.attrs.get("Labels", {}) or {}
    old_subnet = str(old_labels.get("cage.target.subnet") or "").strip()
    try:
        network.remove()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to remove stale network '{DOCKER_NETWORK}': {exc}"
        ) from exc
    if old_subnet:
        release_agent_home_subnet(old_subnet)
    _create_agent_home_network()


def self_heal_docker_network() -> None:
    """Idempotent self-heal: ensure DOCKER_NETWORK exists on the host.

    Called before every compose-up so that if the docker daemon racy-loses
    the network mid-run (observed under heavy concurrent compose load) or
    something external ``docker network rm``'s it, the next launch
    transparently recreates it instead of cascading "external network not
    found" failures for the rest of the run.

    Unlike ``ensure_docker_network`` (the strict startup gate), this never
    raises on "exists with containers" — that's the normal state mid-run.
    """
    client = get_docker_client()
    try:
        client.networks.get(DOCKER_NETWORK)
        return
    except NotFound:
        pass
    try:
        _create_agent_home_network()
        logger.warning(
            "self-heal: re-created missing DOCKER_NETWORK %r "
            "(removed out-of-band — daemon race or external rm)",
            DOCKER_NETWORK,
        )
    except APIError as exc:
        # Lost a race with another launch thread that just created it.
        msg = str(exc).lower()
        if "already exists" in msg or "conflict" in msg:
            return
        raise


def _classify_cage_bench_network(name: str, role_label: str) -> Optional[str]:
    """Bucket a docker network for orphan-reclamation purposes.

    Returns one of:
      - ``"own_runtime"``  — child runtime network of THIS namespace
      - ``"own_home"``     — parent agent_home of THIS namespace (managed
                             elsewhere; cleanup never touches it directly)
      - ``"peer_runtime"`` — child runtime of another namespace's parent
      - ``"peer_home"``    — parent agent_home of another namespace
      - ``None``           — not a cage-managed network, skip

    The role-label hint takes precedence when present so we don't false-
    categorize benchmarks whose name happens to contain ``_runtime`` in
    the middle (the substring match was the previous heuristic).
    """
    if not name.startswith(_CAGE_BENCH_NETWORK_PREFIX):
        return None

    role = role_label.strip().lower()
    is_runtime = role == "runtime" or "_runtime" in name
    is_own = name == DOCKER_NETWORK or name.startswith(f"{DOCKER_NETWORK}_")

    if is_own:
        return "own_runtime" if is_runtime else "own_home"
    return "peer_runtime" if is_runtime else "peer_home"


def cleanup_orphan_networks() -> None:
    """Remove orphan Docker networks from previous (or peer) server runs.

    Three-tier reclamation:

    1. **Own namespace** — runtime networks under
       ``<DOCKER_NETWORK>_<id>_runtime...`` are removed when empty.
       Pre-existing behaviour, preserved.

    2. **Peer namespaces** — parent home networks (``cage_bench_<other_ns>``)
       and their runtime children are reclaimed when:

         a. zero active container endpoints,
         b. ``Created`` more than ``ORPHAN_NETWORK_MIN_AGE_S`` ago (default
            120s) — avoids deleting a peer server's *just-created* home
            before its compose-up attaches the first container, and
         c. the owning namespace has no live target_server process with
            ``--namespace <ns>`` (cross-checked against the network's
            ``cage.target.pid`` label as a fast hint).

    3. **Cage-trial isolation networks** — per-trial isolation
       bridges named ``cage-trial-<trial>-<agent_hash>`` (created by
       ``cage/target/provisioning.py::build_agent_isolation_network``) carry a
       ``cage.network.owner-pid`` label. Reclaimed when:

         a. zero active container endpoints,
         b. age > ``ORPHAN_NETWORK_MIN_AGE_S``, and
         c. the owner PID is not in ``/proc`` (the trial runner died before
            ``AgentIsolationNetwork.teardown`` ran).

       Closes the orchestrator-side /16-or-/26 leak path (mirror of the
       target_server-side parent home reclamation in (2)).
    """
    client = get_docker_client()
    removed_own_runtime = 0
    removed_peer_runtime = 0
    removed_peer_home = 0
    removed_cage_trial = 0
    skipped_too_young: list[tuple[str, float]] = []
    skipped_live_peer: list[str] = []
    try:
        all_networks = client.networks.list()
    except Exception as exc:
        logger.warning(f"cleanup_orphan_networks: failed to list networks: {exc}")
        return

    for network in all_networks:
        name = getattr(network, "name", "") or ""
        is_cage_bench = name.startswith(_CAGE_BENCH_NETWORK_PREFIX)
        is_cage_trial = name.startswith(_CAGE_TRIAL_NETWORK_PREFIX)
        if not (is_cage_bench or is_cage_trial):
            continue
        try:
            network.reload()
            attrs = network.attrs or {}
            container_count = len(attrs.get("Containers", {}) or {})
            labels = attrs.get("Labels", {}) or {}
        except NotFound:
            continue
        except Exception as exc:
            logger.warning(f"cleanup_orphan_networks: reload failed for {name}: {exc}")
            continue
        if container_count > 0:
            continue

        # cage-trial-* (orchestrator-owned isolation bridges) — gated on
        # owner-PID liveness + age.
        if is_cage_trial:
            kind_label = str(labels.get("cage.network.kind") or "").strip().lower()
            if kind_label != "cage-trial":
                # No cage label → user-created network with our prefix;
                # leave it alone.
                continue
            age = _network_age_seconds(attrs)
            if age is None or age < ORPHAN_NETWORK_MIN_AGE_S:
                skipped_too_young.append((name, age or -1.0))
                continue
            owner_pid = labels.get("cage.network.owner-pid")
            if _is_pid_alive(owner_pid):
                skipped_live_peer.append(name)
                continue
            try:
                network.remove()
                removed_cage_trial += 1
                # Release the /26 back to the cage-trial pool. IPAM-config
                # subnet is the source of truth; the ``cage.network.subnet``
                # label is informational (could drift if docker auto-extended).
                for cfg in (attrs.get("IPAM", {}).get("Config") or []):
                    subnet = (cfg or {}).get("Subnet")
                    if subnet:
                        release_cage_trial_subnet(subnet)
            except NotFound:
                continue
            except Exception as exc:
                logger.warning(f"cleanup_orphan_networks: failed to remove {name}: {exc}")
            continue

        # cage_bench_* — own-runtime / peer reclamation (existing logic).
        role_label = str(labels.get("cage.target.role") or "").strip()
        kind = _classify_cage_bench_network(name, role_label)
        if kind is None or kind == "own_home":
            # ``own_home`` is owned by ensure_docker_network /
            # remove_own_docker_network; never reclaim it here.
            continue

        if kind in ("peer_home", "peer_runtime"):
            # Age gate first — even a freshly-created peer home with
            # ``--restart=unless-stopped`` containers can briefly appear
            # empty during compose-up.
            age = _network_age_seconds(attrs)
            if age is None or age < ORPHAN_NETWORK_MIN_AGE_S:
                skipped_too_young.append((name, age or -1.0))
                continue
            ns = str(labels.get("cage.target.namespace") or "").strip()
            if not ns:
                # No namespace label → can't safely identify the owner.
                # Skip to avoid clobbering anything user-created that
                # happens to share the cage_bench_* name prefix.
                continue
            if _is_namespace_alive(ns, labels.get("cage.target.pid")):
                skipped_live_peer.append(name)
                continue

        try:
            network.remove()
            if kind == "own_runtime":
                removed_own_runtime += 1
            elif kind == "peer_runtime":
                removed_peer_runtime += 1
            elif kind == "peer_home":
                removed_peer_home += 1
            # Clear THIS process's in-memory reservation set for the
            # subnet, when we happen to be the cage server that
            # originally reserved it. For peer networks the original
            # reserver was a different process whose reservation set
            # died with it — the discard() is a harmless no-op in that
            # case. The cross-process gate that actually prevents reuse
            # is ``_collect_used_docker_subnets`` (queries the docker
            # daemon, sees the now-removed network drop off), not this
            # in-memory bookkeeping.
            for cfg in (attrs.get("IPAM", {}).get("Config") or []):
                subnet = (cfg or {}).get("Subnet")
                if not subnet:
                    continue
                if kind == "peer_home":
                    # peer_home came from this server's AGENT_HOME pool
                    # if the peer was a sibling cage server — discard
                    # locally; harmless no-op for cross-namespace peers.
                    release_agent_home_subnet(subnet)
                else:
                    # peer_runtime or own_runtime: project-local pool
                    # (172.31.0.0/16 by default). Same caveat applies.
                    try:
                        release_reserved_project_local_subnet(subnet)
                    except Exception:
                        pass
        except NotFound:
            continue
        except Exception as exc:
            logger.warning(f"cleanup_orphan_networks: failed to remove {name}: {exc}")

    if (
        removed_own_runtime
        or removed_peer_home
        or removed_peer_runtime
        or removed_cage_trial
    ):
        logger.info(
            "cleanup_orphan_networks: removed %d own-runtime / %d peer-home / "
            "%d peer-runtime / %d cage-trial networks (own=%s)",
            removed_own_runtime,
            removed_peer_home,
            removed_peer_runtime,
            removed_cage_trial,
            DOCKER_NETWORK,
        )
    else:
        logger.info("cleanup_orphan_networks: no orphan networks found")
    if skipped_too_young:
        logger.debug(
            "cleanup_orphan_networks: skipped %d networks younger than %.0fs",
            len(skipped_too_young),
            ORPHAN_NETWORK_MIN_AGE_S,
        )
    if skipped_live_peer:
        logger.debug(
            "cleanup_orphan_networks: skipped %d peer networks with live owners",
            len(skipped_live_peer),
        )


def cleanup_orphan_volumes() -> None:
    """Remove orphan named volumes belonging to **this** target_server namespace.

    Volumes are stamped with ``cage.target.namespace=<ns>`` at
    compose-render time (see ``network_alloc._stamp_cage_run_id_labels``).
    The filter on that label is the **only** path: we deliberately do not
    fall back to a broader label like ``cage.component=target`` because
    that would let one namespace's GC sweep a peer namespace's volumes
    on a shared docker daemon. If the namespace label is missing on a
    volume (legacy artifact, pre-this-change), it is left alone.

    A volume is "orphan" if docker accepts ``volume.remove(force=True)``.
    Docker refuses to remove a volume still in use, surfacing
    ``APIError(409)`` — we treat that as "still alive, skip". Other
    errors are logged but never raise.
    """
    client = get_docker_client()
    namespace_label = f"cage.target.namespace={TARGET_SERVER_NAMESPACE}"
    try:
        volumes = client.volumes.list(filters={"label": namespace_label})
    except Exception as exc:
        # Listing failed — bail safely. Do NOT fall back to a broader
        # filter; that's how a transient docker daemon hiccup could
        # become "namespace A wipes namespace B's volumes". The next
        # server startup will retry.
        logger.warning(
            "cleanup_orphan_volumes: docker list failed (%s); skipping this cycle",
            exc,
        )
        return

    removed = 0
    skipped_in_use = 0
    for volume in volumes:
        name = getattr(volume, "name", "") or ""
        if not name:
            continue
        # Belt-and-suspenders: confirm the namespace label is exactly
        # ours before removing. The list filter above already guarantees
        # this, but a buggy docker version once returned mis-filtered
        # results — verifying inline costs nothing.
        try:
            attrs = volume.attrs or {}
            labels = attrs.get("Labels") or {}
        except Exception:
            labels = {}
        labelled_ns = str(labels.get("cage.target.namespace") or "").strip()
        if labelled_ns != TARGET_SERVER_NAMESPACE:
            continue
        try:
            volume.remove(force=True)
            removed += 1
        except APIError as exc:
            # Prefer the structured status code over substring match —
            # docker daemon localization can change the error string.
            if getattr(exc, "status_code", None) == 409:
                skipped_in_use += 1
                continue
            msg = str(exc).lower()
            if "in use" in msg:
                skipped_in_use += 1
                continue
            logger.warning(f"cleanup_orphan_volumes: failed to remove {name}: {exc}")
        except NotFound:
            continue
        except Exception as exc:
            logger.warning(f"cleanup_orphan_volumes: failed to remove {name}: {exc}")

    if removed or skipped_in_use:
        logger.info(
            f"cleanup_orphan_volumes: removed {removed}, "
            f"skipped {skipped_in_use} in-use volume(s) for {DOCKER_NETWORK}"
        )
    else:
        logger.info("cleanup_orphan_volumes: no orphan volumes found")


def _has_active_endpoints_error(error: Exception) -> bool:
    return "active endpoints" in str(error).lower()


def remove_network_with_retry(
    network: Any,
    *,
    timeout_s: float = NETWORK_REMOVE_RETRY_TIMEOUT_S,
    poll_interval_s: float = NETWORK_REMOVE_RETRY_INTERVAL_S,
) -> None:
    deadline = time.time() + max(timeout_s, 0.0)
    while True:
        try:
            network.remove()
            logger.info(f"Removed network: {network.name}")
            return
        except NotFound:
            logger.info("Network already removed: %s", getattr(network, "name", "<unknown>"))
            return
        except Exception as e:
            if not _has_active_endpoints_error(e):
                raise
            if time.time() >= deadline:
                raise
            logger.info(
                "Network %s still has active endpoints; waiting %.1fs before retry",
                getattr(network, "name", "<unknown>"),
                poll_interval_s,
            )
            time.sleep(poll_interval_s)


def remove_own_docker_network() -> None:
    """Force-remove the server-owned DOCKER_NETWORK on shutdown."""
    client = get_docker_client()
    try:
        network = client.networks.get(DOCKER_NETWORK)
    except NotFound:
        return
    except Exception as exc:
        logger.warning(f"Failed to look up {DOCKER_NETWORK} on shutdown: {exc}")
        return
    own_subnet: str | None = None
    try:
        network.reload()
        labels = network.attrs.get("Labels", {}) or {}
        own_subnet = str(labels.get("cage.target.subnet") or "").strip() or None
        for cid in list((network.attrs.get("Containers", {}) or {}).keys()):
            try:
                network.disconnect(cid, force=True)
            except Exception as exc:
                logger.warning(
                    f"Failed to force-disconnect {cid} from {DOCKER_NETWORK}: {exc}"
                )
    except Exception as exc:
        logger.warning(f"Failed to inspect {DOCKER_NETWORK} before removal: {exc}")
    rm_succeeded = False
    try:
        network.remove()
        rm_succeeded = True
        logger.info(f"Removed own docker network '{DOCKER_NETWORK}' on shutdown")
    except NotFound:
        rm_succeeded = True  # already gone — treat as success for release.
    except Exception as exc:
        logger.warning(f"Failed to remove {DOCKER_NETWORK} on shutdown: {exc}")
    # Only release the /24 reservation if the docker network is actually
    # gone. If remove failed we leak the slot rather than risking the next
    # ``ensure_docker_network`` picking the same /24 and getting a daemon
    # conflict; ``cleanup_orphan_networks`` will reclaim the slot when the
    # network is eventually removed (own_home is preserved by classify,
    # but a peer cage server sweeping the host would catch our leak after
    # this server exits).
    if rm_succeeded:
        release_agent_home_subnet(own_subnet)
