"""Local docker sweep by ``cage.run_id`` label.

Belt-and-suspenders for the graceful in-band ``DELETE /run/<id>`` path:
when the server-side cleanup didn't run (server crashed, network broken,
``kill -9``, or a bug inside ``_cleanup_instance_impl``), this fallback
walks every docker container / network / named volume labelled with the
run id and force-removes them. Idempotent.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    run_id: str
    containers_removed: int = 0
    networks_removed: int = 0
    volumes_removed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def removed_anything(self) -> bool:
        return (
            self.containers_removed > 0
            or self.networks_removed > 0
            or self.volumes_removed > 0
        )


def sweep_docker_resources(
    run_id: str,
    *,
    containers: tuple[str, ...] = (),
    networks: tuple[str, ...] = (),
    volumes: tuple[str, ...] = (),
    docker_timeout: float = 60.0,
) -> SweepResult:
    """Force-remove explicitly named Docker resources for one run.

    This is the ResourceLedger counterpart to :func:`sweep_run`. ``sweep_run``
    discovers resources through Docker labels; this helper trusts a canonical
    ledger that already recorded concrete Docker container, network, and volume
    identifiers. It intentionally ignores free-form ``cleanup_action`` strings:
    the caller supplies structured resource ids, and this function maps them to
    a small allowlist of Docker commands.

    Each resource is removed independently so one stale or already-removed id
    does not prevent cleanup of the rest of the ledger. Successful removals are
    counted in the returned :class:`SweepResult`; failures are summarized in
    ``errors`` for dry-run/apply reports.
    """

    result = SweepResult(run_id=run_id)
    if not run_id:
        return result

    for container in _dedupe_ids(containers):
        rc, err = _docker_rm(
            ["docker", "rm", "-f", "-v", container],
            timeout=docker_timeout,
        )
        if rc == 0:
            result.containers_removed += 1
        else:
            result.errors.append(f"container {container[:40]}: {err[:200]}")

    for network in _dedupe_ids(networks):
        rc, err = _docker_rm(
            ["docker", "network", "rm", network],
            timeout=docker_timeout,
        )
        if rc == 0:
            result.networks_removed += 1
        else:
            result.errors.append(f"network {network[:40]}: {err[:200]}")

    for volume in _dedupe_ids(volumes):
        rc, err = _docker_rm(
            ["docker", "volume", "rm", "-f", volume],
            timeout=docker_timeout,
        )
        if rc == 0:
            result.volumes_removed += 1
        else:
            result.errors.append(f"volume {volume[:40]}: {err[:200]}")

    return result


def _dedupe_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return non-empty Docker identifiers in first-seen order."""

    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


_VALID_COMPONENTS = frozenset({"agent", "target"})


def sweep_run(
    run_id: str,
    *,
    components: tuple[str, ...] = ("agent", "target"),
    namespace: str | None = None,
    docker_timeout: float = 60.0,
) -> SweepResult:
    """Force-remove docker resources labelled ``cage.run_id=<run_id>``.

    ``components`` restricts the sweep — ``"agent"`` only removes cage-owned
    agent containers; ``"target"`` removes target_server-owned target
    containers, networks, and named volumes. The default removes both.
    Unknown component names are silently dropped.

    ``namespace`` restricts the **target** sweep to one
    ``cage.target.namespace`` label. When set, target
    containers/networks/volumes are filtered by both ``cage.run_id`` AND
    the namespace label; cross-namespace target resources sharing the
    same run_id are left alone. Agent containers don't carry a
    namespace label (they're orchestrator-side, not server-side), so
    the agent sweep ignores ``namespace`` — same agent containers
    always belong to the same logical run regardless of which target
    server hosted them. Pass ``None`` to sweep all namespaces (the
    default; backwards-compatible).

    Container removal uses ``docker rm -f -v`` so anonymous volumes
    attached to those containers also get removed. Named volumes declared
    in the compose file are reclaimed via a separate ``docker volume rm``
    pass that filters on the same ``cage.run_id`` + ``cage.component=target``
    (+ optional namespace) labels stamped at compose-render time
    (see ``network_alloc._stamp_cage_run_id_labels``).
    """
    result = SweepResult(run_id=run_id)
    if not run_id:
        return result

    wanted = {c for c in components if c in _VALID_COMPONENTS}
    if not wanted:
        return result

    run_filter = ["--filter", f"label=cage.run_id={run_id}"]
    namespace_filter: list[str] = []
    if namespace:
        namespace_filter = ["--filter", f"label=cage.target.namespace={namespace}"]
    for component in sorted(wanted):
        # The namespace filter only applies to target-side resources.
        # Agent containers belong to the orchestrator and don't carry a
        # target_server.namespace label; applying the filter would make
        # the agent branch return zero matches.
        per_component_extra = namespace_filter if component == "target" else []
        component_filter = run_filter + [
            "--filter", f"label=cage.component={component}",
        ] + per_component_extra
        # Containers
        ids = _docker_ls(["docker", "ps", "-aq", *component_filter])
        if ids:
            rc, err = _docker_rm(
                ["docker", "rm", "-f", "-v", *ids], timeout=docker_timeout,
            )
            if rc == 0:
                result.containers_removed += len(ids)
            else:
                result.errors.append(f"{component} containers: {err[:200]}")
        # Networks + named volumes (agents don't own any; only target stacks do)
        if component == "target":
            ids = _docker_ls(["docker", "network", "ls", "-q", *component_filter])
            for nid in ids:
                rc, err = _docker_rm(
                    ["docker", "network", "rm", nid], timeout=docker_timeout,
                )
                if rc == 0:
                    result.networks_removed += 1
                else:
                    result.errors.append(f"network {nid[:12]}: {err[:200]}")
            # Named volumes: ``docker volume ls -q`` accepts the same
            # ``--filter label=`` form. ``docker volume rm -f`` skips the
            # "still in use" check; the container removal above has
            # already detached everything labelled with the same run_id.
            vol_names = _docker_ls(["docker", "volume", "ls", "-q", *component_filter])
            for vname in vol_names:
                rc, err = _docker_rm(
                    ["docker", "volume", "rm", "-f", vname], timeout=docker_timeout,
                )
                if rc == 0:
                    result.volumes_removed += 1
                else:
                    result.errors.append(f"volume {vname[:40]}: {err[:200]}")

    return result


def _docker_ls(cmd: list[str]) -> list[str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("docker ls failed: %s", exc)
        return []
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


def _docker_rm(cmd: list[str], *, timeout: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, proc.stderr or ""
