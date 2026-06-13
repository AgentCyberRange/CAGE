"""``cage gc`` — reclaim docker resources from non-running runs.

The host accumulates docker resources whenever a ``cage run`` exits
abnormally before its finally-blocks fire: SIGKILL, OOMKill, host
reboot, a crash inside the orchestrator's own teardown. The in-band
``DELETE /launch/{id}`` path never runs in those cases, so containers
labelled ``cage.run_id=<rid>`` survive — same for the networks and
named volumes the target stack created.

This module decides which run_ids are eligible for reclamation and
hands the actual ``docker rm`` work to ``local_cleanup.sweep_run``.

Eligibility = liveness signal says the run isn't ticking. ``.cage_runs/``
is the single source of truth (mirrors what the Web inspector uses for
its "running" badge).

Safety properties:
  * ``.cage_runs/`` is never written. Run artifacts are immortal from
    this module's perspective.
  * Default is dry-run. Actual removal requires ``apply=True``.
  * Namespace scoping (when set) restricts the sweep to one cage
    target_server namespace — composes cleanly with main-branch's
    peer-aware orphan reclamation.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.resources import ResourceLedgerReader, ResourceLedgerWriter
from cage.experiment.model import ResourceRecord
from cage.gc.plan import (
    COUNT_KEYS,
    ResourceCleanupPlan,
    build_resource_cleanup_plan,
)
from cage.experiment.engine.live.liveness import (
    RunLiveness,
    is_run_running,
    iter_known_run_ids,
    locate_run_dir,
)
from cage.target.local_cleanup import (
    SweepResult,
    sweep_docker_resources,
    sweep_run,
)

logger = logging.getLogger(__name__)


DECISION_ALIVE = "alive"
DECISION_DEAD = "dead"
DECISION_ORPHAN = "orphan"


@dataclass
class GcDecision:
    run_id: str
    decision: str  # one of DECISION_*
    reason: str
    run_dir: Path | None
    container_count: int = 0
    network_count: int = 0
    volume_count: int = 0
    swept: SweepResult | None = None

    @property
    def has_resources(self) -> bool:
        return (
            self.container_count > 0
            or self.network_count > 0
            or self.volume_count > 0
        )


@dataclass
class GcReport:
    decisions: list[GcDecision] = field(default_factory=list)
    applied: bool = False

    def summary(self) -> dict:
        alive = sum(1 for d in self.decisions if d.decision == DECISION_ALIVE)
        dead = sum(1 for d in self.decisions if d.decision == DECISION_DEAD)
        orphan = sum(1 for d in self.decisions if d.decision == DECISION_ORPHAN)
        containers = sum((d.swept.containers_removed if d.swept else 0) for d in self.decisions)
        networks = sum((d.swept.networks_removed if d.swept else 0) for d in self.decisions)
        volumes = sum((d.swept.volumes_removed if d.swept else 0) for d in self.decisions)
        return {
            "applied": self.applied,
            "alive": alive,
            "dead": dead,
            "orphan": orphan,
            "removed": {
                "containers": containers,
                "networks": networks,
                "volumes": volumes,
            },
        }


_AUTODISCOVER_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "dist", "build", ".idea", ".vscode",
    "target", ".cache", "worktrees",
})
_AUTODISCOVER_MAX_DEPTH = 6
_AUTODISCOVER_MAX_ANCESTOR_HOPS = 8


def _find_cage_anchor(cwd: Path) -> Path:
    """Walk up from ``cwd`` to find a cage source tree root.

    Returns the first ancestor that contains ``cage/cli/main.py`` (the
    package CLI marker of the cage repo) — that directory is the natural
    place to anchor a downward ``.cage_runs/`` scan because benchmark
    examples live under ``examples/<name>/.cage_runs/``. Falls back to
    ``cwd`` when no such ancestor is found within
    :data:`_AUTODISCOVER_MAX_ANCESTOR_HOPS` levels, so an arbitrary
    working dir still gets a chance at downward discovery.
    """
    here = cwd
    for _ in range(_AUTODISCOVER_MAX_ANCESTOR_HOPS):
        if (here / "cage" / "cli" / "main.py").is_file():
            return here
        if here.parent == here:
            break
        here = here.parent
    return cwd


def _scan_for_cage_runs(start: Path, *, max_depth: int) -> list[Path]:
    """Bounded depth-first scan for ``.cage_runs/`` directories.

    Skips noise dirs (vcs, build artifacts, virtualenvs, worktrees) and
    other hidden subdirectories. Does not recurse into a discovered
    ``.cage_runs/`` (the dir itself is the answer; its children are run
    artifacts, not more roots). ``followlinks=False`` keeps the walk
    bounded on hosts with symlinked artifact stores.
    """
    found: list[Path] = []
    try:
        start_resolved = start.resolve()
    except OSError:
        return found
    start_depth = len(start_resolved.parts)
    try:
        for dirpath, dirnames, _filenames in os.walk(
            start_resolved, topdown=True, followlinks=False,
        ):
            current = Path(dirpath)
            depth = len(current.parts) - start_depth
            if ".cage_runs" in dirnames:
                found.append(current / ".cage_runs")
            if depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if d != ".cage_runs"
                and d not in _AUTODISCOVER_SKIP_DIRS
                and not d.startswith(".")
            ]
    except OSError as exc:
        logger.warning("gc: auto-discovery walk failed under %s: %s", start_resolved, exc)
    return found


def default_cage_runs_roots(*, cwd: Path | None = None) -> list[Path]:
    """Search roots for ``.cage_runs/`` directories.

    Resolution order:
      1. ``$CAGE_RUNS_ROOT`` if set (exact path; takes precedence so
         power users can pin the root regardless of cwd).
      2. Walk **up** from ``cwd`` looking for a cage source tree
         (``cage/cli/main.py`` ancestor); fall back to ``cwd`` if no anchor
         is found.
      3. Bounded depth-first scan **down** from the anchor for any
         ``.cage_runs/`` directories. Skips noise dirs (vcs, build
         artifacts, virtualenvs, worktrees) and other hidden subdirs.
         Stops at depth :data:`_AUTODISCOVER_MAX_DEPTH`.

    The walk is the same shape the Web inspector does — both should
    answer "where are runs?" identically so ``cage gc`` doesn't
    require the user to ``cd`` into the right place. Returns a
    deduplicated list preserving discovery order.
    """
    cwd = (cwd or Path.cwd()).resolve()
    env_root = os.getenv("CAGE_RUNS_ROOT", "").strip()
    if env_root:
        return [Path(env_root)]

    anchor = _find_cage_anchor(cwd)
    discovered = _scan_for_cage_runs(anchor, max_depth=_AUTODISCOVER_MAX_DEPTH)
    # Also include cwd-rooted scan when anchor walked us elsewhere — covers
    # users who run from a sibling repo with their own .cage_runs/.
    if anchor != cwd:
        discovered.extend(_scan_for_cage_runs(cwd, max_depth=_AUTODISCOVER_MAX_DEPTH))

    seen: set[Path] = set()
    roots: list[Path] = []
    for path in discovered:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def collect_docker_run_ids(
    *,
    namespace: str | None = None,
    docker_timeout: float = 30.0,
) -> dict[str, dict[str, int]]:
    """Group every cage-labelled docker resource by ``cage.run_id``.

    Returns a dict ``{run_id: {"containers": N, "networks": N,
    "volumes": N}}``. Resources without a ``cage.run_id`` label are
    skipped (those belong to running runs that haven't crashed —
    nothing for GC to do there).

    ``namespace`` restricts to one ``cage.target.namespace`` —
    use to keep ``cage gc`` from touching another cage server's
    resources on a shared host.
    """
    out: dict[str, dict[str, int]] = {}

    base_filter = ["--filter", "label=cage.run_id"]
    if namespace:
        base_filter += ["--filter", f"label=cage.target.namespace={namespace}"]

    def _list_labels(kind: str) -> list[tuple[str, str]]:
        if kind == "container":
            cmd = ["docker", "ps", "-aq", *base_filter]
        elif kind == "network":
            cmd = ["docker", "network", "ls", "-q", *base_filter]
        elif kind == "volume":
            cmd = ["docker", "volume", "ls", "-q", *base_filter]
        else:
            return []
        ids = _docker_ls(cmd, docker_timeout)
        if not ids:
            return []
        # Inspect to pull the cage.run_id label per resource. Inspect is
        # a single subprocess call regardless of ID count.
        if kind == "container":
            inspect_cmd = ["docker", "inspect", "--format", "{{.Id}}|{{index .Config.Labels \"cage.run_id\"}}", *ids]
        elif kind == "network":
            inspect_cmd = ["docker", "network", "inspect", "--format", "{{.Id}}|{{index .Labels \"cage.run_id\"}}", *ids]
        else:  # volume
            inspect_cmd = ["docker", "volume", "inspect", "--format", "{{.Name}}|{{index .Labels \"cage.run_id\"}}", *ids]
        out_lines = _docker_ls(inspect_cmd, docker_timeout)
        results: list[tuple[str, str]] = []
        for line in out_lines:
            if "|" not in line:
                continue
            rid_token = line.split("|", 1)[1].strip()
            if rid_token:
                results.append((line.split("|", 1)[0].strip(), rid_token))
        return results

    container_pairs = _list_labels("container")
    network_pairs = _list_labels("network")
    volume_pairs = _list_labels("volume")

    for _id, rid in container_pairs:
        out.setdefault(rid, {"containers": 0, "networks": 0, "volumes": 0})["containers"] += 1
    for _id, rid in network_pairs:
        out.setdefault(rid, {"containers": 0, "networks": 0, "volumes": 0})["networks"] += 1
    for _id, rid in volume_pairs:
        out.setdefault(rid, {"containers": 0, "networks": 0, "volumes": 0})["volumes"] += 1

    return out


def collect_ledger_resource_counts(
    search_roots: list[Path],
    *,
    namespace: str | None = None,
) -> dict[str, dict[str, int]]:
    """Group unreleased canonical resource-ledger entries by run id.

    ``resources.jsonl`` is append-only: the same ``resource_id`` may have a
    ``created`` line, a ``started`` line, and later a ``released`` line. GC
    decisions must therefore count only the latest ledger record for each
    resource. Released resources are omitted; ``cleanup_failed`` and any other
    non-released status remain visible because they still need operator
    attention or a future cleanup executor.

    Namespace filtering is intentionally compatibility-friendly. If a record
    has explicit ``metadata.labels["cage.target.namespace"]`` and it
    does not match ``namespace``, it is skipped. Older agent-container ledger
    records without that metadata are kept so ``--namespace`` does not hide
    otherwise visible canonical resources.
    """

    grouped: dict[str, dict[str, int]] = {}
    for _rid, run_dir in iter_known_run_ids(search_roots):
        plan = _resource_cleanup_plan_for_gc(run_dir, namespace=namespace)
        if not plan.has_resources():
            continue
        grouped[run_dir.name] = plan.counts()
    return grouped


def _latest_resource_records_for_gc(run_dir: Path) -> dict[str, ResourceRecord]:
    """Return latest resource records for one run using canonical artifacts.

    New runs have a complete ``ExperimentArtifactReadSnapshot`` that includes
    the resource ledger, event log, trial records, and run record in one read
    model. GC prefers that shared reader so cleanup decisions use the same
    durable fact source as score/inspect/resume migrations. Historical or
    partially written runs may lack spec/plan/record files, so this function
    falls back to the resource-ledger reader instead of hiding cleanup work.
    """

    snapshot = ExperimentArtifactReader(run_dir).try_load_snapshot()
    if snapshot is None:
        return ResourceLedgerReader(run_dir).latest_by_resource_id()

    latest: dict[str, ResourceRecord] = {}
    for record in snapshot.resources:
        latest[record.resource_id] = record
    return latest


def collect_gc_resource_counts(
    *,
    namespace: str | None = None,
    search_roots: list[Path] | None = None,
) -> dict[str, dict[str, int]]:
    """Collect resource counts from Docker labels and canonical ledgers.

    Docker is the live source for resources that still carry labels. The
    canonical ledger fills gaps when Docker enumeration is unavailable, labels
    were lost, or a run recorded resources before a crash. When both sources
    report the same run and resource type, keep the larger count to avoid the
    common double-counting case where the ledger and Docker are describing the
    same container.
    """

    docker_counts = collect_docker_run_ids(namespace=namespace)
    ledger_counts = collect_ledger_resource_counts(search_roots or [], namespace=namespace)
    return _merge_resource_counts(docker_counts, ledger_counts)


def _empty_resource_counts() -> dict[str, int]:
    """Return a fresh zero-count mapping in the GC JSON shape."""

    return {key: 0 for key in COUNT_KEYS}


def _merge_resource_counts(
    *groups: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Merge run-id resource counts without double-counting known overlaps."""

    merged: dict[str, dict[str, int]] = {}
    for group in groups:
        for run_id, counts in group.items():
            out = merged.setdefault(run_id, _empty_resource_counts())
            for key in COUNT_KEYS:
                out[key] = max(out[key], int(counts.get(key, 0) or 0))
    return merged


def _resource_cleanup_plan_for_gc(
    run_dir: Path,
    *,
    namespace: str | None = None,
) -> ResourceCleanupPlan:
    """Build the canonical cleanup plan for one run directory."""

    return build_resource_cleanup_plan(
        _latest_resource_records_for_gc(run_dir).values(),
        namespace=namespace,
    )


def _record_ledger_cleanup_outcomes(
    run_dir: Path,
    *,
    records_by_key: dict[str, tuple[ResourceRecord, ...]],
    sweep: SweepResult,
) -> None:
    """Append cleanup outcome records after an explicit ledger Docker sweep.

    ``SweepResult`` is aggregate-level, while ``resources.jsonl`` is
    resource-level. Until the low-level sweeper reports per-resource results,
    Cage records successes in first-seen ledger order up to the removed count
    for each bucket; the remainder are marked ``cleanup_failed`` with the
    available Docker error summary. The append-only ledger then shows that GC
    attempted cleanup instead of leaving stale ``started`` records as the latest
    durable truth.
    """

    removed_by_key = {
        "containers": int(sweep.containers_removed or 0),
        "networks": int(sweep.networks_removed or 0),
        "volumes": int(sweep.volumes_removed or 0),
    }
    errors = "; ".join(str(error) for error in (sweep.errors or []) if str(error))
    timestamp = datetime.now(timezone.utc).isoformat()
    writer = ResourceLedgerWriter(run_dir)
    for key, records in records_by_key.items():
        removed_count = removed_by_key.get(key, 0)
        for index, record in enumerate(records):
            released = index < removed_count
            writer.append_resource(
                run_id=record.run_id,
                resource_id=record.resource_id,
                kind=record.kind,
                provider=record.provider,
                external_id=record.external_id,
                status="released" if released else "cleanup_failed",
                cleanup_action=record.cleanup_action,
                timestamp=timestamp,
                trial_id=record.trial_id,
                metadata=record.metadata,
                cleanup_error=(
                    None
                    if released
                    else errors or "gc explicit docker cleanup did not confirm removal"
                ),
            )


def _combine_sweep_results(run_id: str, *results: SweepResult | None) -> SweepResult:
    """Merge label-based and ledger-based cleanup results for one run."""

    combined = SweepResult(run_id=run_id)
    for result in results:
        if result is None:
            continue
        combined.containers_removed += result.containers_removed
        combined.networks_removed += result.networks_removed
        combined.volumes_removed += result.volumes_removed
        combined.errors.extend(result.errors)
    return combined


def decide_run(rid: str, *, search_roots: list[Path]) -> tuple[str, str, Path | None]:
    """Classify a run_id as alive / dead / orphan.

    Returns ``(decision, reason, run_dir or None)``.

    Safety: when ``search_roots`` is empty we can't tell whether a
    missing run dir means "this run was never recorded here" or "we
    just didn't look in the right place". A naive ORPHAN verdict would
    let ``--apply`` nuke every running run on a host where the user
    invoked ``cage gc`` from the wrong cwd. We bail out as ALIVE
    instead — sweeping does nothing, dry-run output makes the cause
    obvious.
    """
    if not search_roots:
        return (
            DECISION_ALIVE,
            "no .cage_runs/ search roots resolved — refusing to classify "
            "(set CAGE_RUNS_ROOT or pass --root to enable GC)",
            None,
        )
    run_dir = locate_run_dir(rid, search_roots=search_roots)
    if run_dir is None:
        return DECISION_ORPHAN, "no .cage_runs/<rid>/ directory", None
    verdict: RunLiveness = is_run_running(run_dir)
    if verdict.running:
        return DECISION_ALIVE, verdict.reason, run_dir
    return DECISION_DEAD, verdict.reason, run_dir


def gc_run(
    rid: str,
    resource_counts: dict[str, int],
    *,
    search_roots: list[Path],
    apply: bool,
    namespace: str | None = None,
) -> GcDecision:
    """Classify one run_id and (optionally) sweep its resources.

    When ``namespace`` is set, the sweep is restricted to target-side
    resources carrying ``cage.target.namespace=<namespace>``.
    This is the same scoping ``collect_docker_run_ids`` uses for
    enumeration, so end-to-end behaviour stays consistent: only the
    namespace that owns a run_id can reclaim its target resources.
    """
    decision, reason, run_dir = decide_run(rid, search_roots=search_roots)
    out = GcDecision(
        run_id=rid,
        decision=decision,
        reason=reason,
        run_dir=run_dir,
        container_count=resource_counts.get("containers", 0),
        network_count=resource_counts.get("networks", 0),
        volume_count=resource_counts.get("volumes", 0),
    )
    if decision == DECISION_ALIVE:
        return out
    if not out.has_resources:
        return out
    if apply:
        ledger_sweep: SweepResult | None = None
        if run_dir is not None:
            cleanup_plan = _resource_cleanup_plan_for_gc(run_dir, namespace=namespace)
            ledger_records = cleanup_plan.cleanup_records_by_key()
            if any(ledger_records.values()):
                ledger_ids = cleanup_plan.docker_ids()
                ledger_sweep = sweep_docker_resources(
                    rid,
                    containers=ledger_ids["containers"],
                    networks=ledger_ids["networks"],
                    volumes=ledger_ids["volumes"],
                )
                _record_ledger_cleanup_outcomes(
                    run_dir,
                    records_by_key=ledger_records,
                    sweep=ledger_sweep,
                )
        label_sweep = sweep_run(rid, components=("agent", "target"), namespace=namespace)
        out.swept = _combine_sweep_results(rid, ledger_sweep, label_sweep)
    return out


def gc_all(
    *,
    namespace: str | None = None,
    apply: bool = False,
    search_roots: list[Path] | None = None,
) -> GcReport:
    """Enumerate docker resources + classify owners + (optionally) sweep.

    Default behaviour is **dry-run**: report which run_ids would be
    swept but make no changes.

    ``namespace`` is propagated through enumeration AND sweep so that
    running ``gc_all(namespace="A")`` on a host with multiple cage
    target_server namespaces only touches namespace A's resources.
    """
    roots = search_roots if search_roots is not None else default_cage_runs_roots()
    grouped = collect_gc_resource_counts(namespace=namespace, search_roots=roots)

    report = GcReport(applied=apply)
    for rid, counts in sorted(grouped.items()):
        decision = gc_run(rid, counts, search_roots=roots, apply=apply, namespace=namespace)
        report.decisions.append(decision)
    return report


def _docker_ls(cmd: list[str], timeout: float) -> list[str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("gc: docker command failed: %s", exc)
        return []
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
