"""Parallel target smoke test for ``cage targets-check``.

Selects target samples, launches them and reports readiness; the probe
implementations live in :mod:`cage.target.check_probes`.
"""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import uuid
from cage.benchmarks import normalize_sample_id, sample_id_matches
from cage.target.build import (
    build_benchmark_targets,
    load_benchmark_from_project,
    print_build_summary,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from cage.target.check_probes import (
    InstanceResult,
    _launch_and_probe,
    _target_id,
)

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


_LEVEL_SUFFIX_RE = re.compile(r"^(?P<base>.+)-l(?P<level>\d+)$", re.IGNORECASE)


@dataclass
class CheckSummary:
    total: int
    passed: int
    failed: int
    results: list[InstanceResult]


def _requested_id_aliases(requested: str) -> list[str]:
    """Return ids accepted for target-build selection.

    Old run scripts often encoded prompt/hint levels in sample ids. For target
    builds the level does not matter, so ``pb-demo-L0`` may select
    ``pb-demo`` when there is no exact prompt-level sample.
    """

    value = str(requested or "").strip()
    if not value:
        return []
    aliases = [value]
    match = _LEVEL_SUFFIX_RE.match(value)
    if match:
        base = match.group("base").strip()
        if base:
            aliases.append(base)
    return aliases


def _sample_matches_requested_id(sample: dict[str, Any], requested: str) -> bool:
    return sample_id_matches(sample, _requested_id_aliases(requested))


def _select_target_samples(
    samples: list[dict[str, Any]],
    *,
    only: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter samples by CLI ids and de-duplicate by launchable target id."""

    selected = list(samples)
    if only:
        wanted = [str(item).strip() for item in only if str(item).strip()]
        all_ids = sorted({_target_id(s) for s in samples})
        unknown = sorted(
            requested for requested in wanted
            if not any(_sample_matches_requested_id(sample, requested) for sample in samples)
        )
        if unknown:
            raise ValueError(
                f"--only ids not in project: {unknown}. Known target ids: {all_ids}"
            )
        selected = [
            sample for sample in samples
            if any(_sample_matches_requested_id(sample, requested) for requested in wanted)
        ]

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in selected:
        target_id = _target_id(sample)
        key = normalize_sample_id(target_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def _target_matches_for_request(
    samples: list[dict[str, Any]],
    requested: str,
) -> list[str]:
    matches: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        if not _sample_matches_requested_id(sample, requested):
            continue
        target_id = _target_id(sample)
        key = normalize_sample_id(target_id)
        if key in seen:
            continue
        seen.add(key)
        matches.append(target_id)
    return matches


def _print_target_check_plan(
    samples: list[dict[str, Any]],
    *,
    only: list[str] | None,
    parallel: int,
    samples_parallel: int | None,
    readiness_timeout: float,
    build: bool,
) -> None:
    print("Target readiness check", flush=True)
    print("Mode: target launch/readiness only; no agent or model will run.", flush=True)
    if build:
        print("Launch build: benchmark build hook runs before target launch.", flush=True)
    else:
        print(
            "Launch build: disabled; use cage benchmark build before target checks",
            flush=True,
        )
    print(
        f"Readiness probe: wait up to {readiness_timeout:g}s for in-network HTTP/HTTPS readiness.",
        flush=True,
    )
    print(
        f"Parallelism: {parallel} instance(s) per target; samples_parallel={samples_parallel or 'all'}.",
        flush=True,
    )
    if only:
        for requested in only:
            targets = _target_matches_for_request(samples, requested)
            if not targets:
                continue
            print(f"Sample: {requested} -> target {', '.join(targets)}", flush=True)
    else:
        print(f"Targets selected: {len(_select_target_samples(samples))}", flush=True)
    print(flush=True)


def _run_docker(args: list[str], *, timeout: float = 60.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _compose_down(project_name: str) -> None:
    """``docker compose -p <project> down -v --remove-orphans`` (best effort).

    Run after the target_server server has been shut down — this catches
    containers/networks/volumes that the DELETE /run endpoint missed (e.g.
    when launch failed mid-flight and never registered with the server).
    """
    if not project_name:
        return
    rc, out, err = _run_docker(
        ["docker", "compose", "-p", project_name,
         "down", "-v", "--remove-orphans", "--timeout", "10"],
        timeout=120.0,
    )
    if rc != 0 and err.strip():
        logger.info("compose down %s: %s", project_name, err.strip().splitlines()[-1])


def _sweep_by_cage_run_id(run_ids: list[str]) -> None:
    """Force-remove docker resources still tagged ``cage.run_id=<rid>``.

    target_server stamps every container/network it creates with this label
    (see ``cage/target/server/network_alloc.py::_stamp_cage_run_id_labels``).
    Volumes inherit the project_name prefix, which ``docker compose down -v``
    already handled in the prior step — but iterate them here too for any
    that escaped (e.g. anonymous volumes left dangling).
    """
    for rid in run_ids:
        if not rid:
            continue
        # Narrow by component=target so this sweep never touches the shared
        # namespace network (DOCKER_NETWORK), agent containers, or anything
        # else that happens to carry cage.run_id but isn't ours to nuke.
        # Mirrors local_cleanup.sweep_run's filter pair.
        run_flt = f"label=cage.run_id={rid}"
        target_flt = "label=cage.component=target"
        # Containers
        rc, out, _ = _run_docker(
            ["docker", "ps", "-aq", "--filter", run_flt, "--filter", target_flt],
            timeout=10.0,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids:
            _run_docker(["docker", "rm", "-f", "-v", *ids], timeout=60.0)
        # Networks
        rc, out, _ = _run_docker(
            ["docker", "network", "ls", "-q", "--filter", run_flt, "--filter", target_flt],
            timeout=10.0,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        for nid in ids:
            _run_docker(["docker", "network", "rm", nid], timeout=30.0)
        # Volumes (only some adapters label these — best effort)
        rc, out, _ = _run_docker(
            ["docker", "volume", "ls", "-q", "--filter", run_flt, "--filter", target_flt],
            timeout=10.0,
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        if ids:
            _run_docker(["docker", "volume", "rm", "-f", *ids], timeout=60.0)


def _sweep_by_compose_project(project_name: str) -> None:
    if not project_name:
        return
    project_flt = f"label=com.docker.compose.project={project_name}"

    rc, out, _ = _run_docker(
        ["docker", "ps", "-aq", "--filter", project_flt],
        timeout=10.0,
    )
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    if ids:
        _run_docker(["docker", "rm", "-f", "-v", *ids], timeout=90.0)

    rc, out, _ = _run_docker(
        ["docker", "network", "ls", "-q", "--filter", project_flt],
        timeout=10.0,
    )
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    for nid in ids:
        _run_docker(["docker", "network", "rm", nid], timeout=30.0)

    rc, out, _ = _run_docker(
        ["docker", "volume", "ls", "-q", "--filter", project_flt],
        timeout=10.0,
    )
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    if ids:
        _run_docker(["docker", "volume", "rm", "-f", *ids], timeout=60.0)


def _teardown_launched_targets(
    *,
    launched_run_ids: list[str],
    project_names: list[str],
) -> None:
    """Remove target docker resources while leaving the embedded server alive."""
    for project in sorted({p for p in project_names if p}):
        _compose_down(project)
        _sweep_by_compose_project(project)
    _sweep_by_cage_run_id(launched_run_ids)


def _full_teardown(
    *,
    server_url: str,
    embedded: Any,
    launched_run_ids: list[str],
    project_names: list[str],
) -> None:
    """Layered teardown — every layer is best-effort, none short-circuit.

    Order: shut the server down first, then nuke whatever docker resources
    are still tagged with our run ids. The in-band per-run DELETE step is
    gone (its server endpoint was removed alongside the batch teardown
    cleanup-cross-talk fix); compose-down + label sweep are enough here.
    """
    # 1. Shut the embedded server down so step 2 can act without races.
    if embedded is not None:
        try:
            embedded.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedded target_server stop failed: %s", exc)

    # 2. Out-of-band cleanup — catches anything the prior steps missed.
    _teardown_launched_targets(
        launched_run_ids=launched_run_ids,
        project_names=project_names,
    )


def check_targets(
    project_path: Path,
    *,
    parallel: int = 1,
    samples_parallel: int | None = None,
    limit: int | None = None,
    keep: bool = False,
    only: list[str] | None = None,
    readiness_timeout: float = 120.0,
    compose_up_timeout: float | None = None,
    startup_timeout: float | None = None,
    build: bool = False,
) -> CheckSummary:
    """Spin up + verify all targets declared by ``project_path``.

    For each sample, launches ``parallel`` independent instances
    concurrently, probes each, then tears them down (unless ``keep`` is set).
    Samples themselves run concurrently in batches of ``samples_parallel``
    (default: ``len(samples)`` — fully parallel). Always spawns an embedded
    target_server subprocess for the lifetime of the call — there's no
    external/remote server option, matching ``cage run``'s per-run-embedded
    model. Target launch itself never builds images; pass ``build=True`` to
    run the benchmark-owned build hook before target launch.
    """
    # ``benchmark.setup()`` instantiates a ChallengeClient with the (empty,
    # placeholder) server_url and logs that — which is misleading here
    # because we immediately override server_url with the embedded
    # target_server a few lines below. Silence the noise.
    challenge_client_logger = logging.getLogger("ChallengeClient")
    prev_level = challenge_client_logger.level
    challenge_client_logger.setLevel(logging.WARNING)
    try:
        benchmark = load_benchmark_from_project(project_path)
    finally:
        challenge_client_logger.setLevel(prev_level)

    source_samples = list(benchmark.iter_samples_limited(limit))
    _print_target_check_plan(
        source_samples,
        only=only,
        parallel=parallel,
        samples_parallel=samples_parallel,
        readiness_timeout=readiness_timeout,
        build=build,
    )
    samples = _select_target_samples(source_samples, only=only)
    if not samples:
        return CheckSummary(total=0, passed=0, failed=0, results=[])

    if build:
        summary = build_benchmark_targets(
            project_path,
            limit=limit,
            only=list(only or []) or None,
            max_workers=max(1, int(samples_parallel or 1)),
            dry_run=False,
        )
        print_build_summary(summary)
        if summary.failed:
            raise RuntimeError(
                f"benchmark build failed for {summary.failed} target(s); "
                "target check was not started"
            )

    # Lazy import — only this command needs the embedded target_server
    # helpers, and pulling them in at module load drags in their docker
    # dependencies.
    from cage.target.provisioning import (
        discover_benchmark_root,
        spawn_embedded_target_server,
    )
    bench_root = discover_benchmark_root(benchmark)
    # Capture the embedded server's stdout+stderr to disk. /launch failures
    # surface their detail in the HTTP body (see ``_LaunchHTTPError``), but
    # earlier-stage failures (compose plan resolution, port allocation) and
    # the running-state of the server itself only show up in its own logger.
    # Without this file, those errors land in DEVNULL.
    tcheck_run_id = f"tcheck-{uuid.uuid4().hex[:8]}"
    server_log_dir = Path.cwd() / ".cage_runs"
    server_log_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = server_log_dir / f"target_server-{tcheck_run_id}.log"
    # Forward CLI-level server timeouts so users don't have to ``export``
    # TARGET_SERVER_* env vars. Target launch itself never builds images; the
    # optional ``--build`` path above uses benchmark-owned build hooks first.
    extra_env: dict[str, str] = {"TARGET_SERVER_BUILD_IF_MISSING": "0"}
    if compose_up_timeout is not None:
        extra_env["TARGET_SERVER_COMPOSE_UP_TIMEOUT_S"] = str(compose_up_timeout)
    if startup_timeout is not None:
        extra_env["TARGET_SERVER_STARTUP_TIMEOUT_S"] = str(startup_timeout)
    embedded = spawn_embedded_target_server(
        run_id=tcheck_run_id,
        benchmark_root=bench_root,
        log_path=server_log_path,
        extra_env=extra_env or None,
    )
    server_url = embedded.server_url
    logger.info("embedded target_server: %s  log=%s", server_url, server_log_path)

    base_run_id = f"tcheck_{uuid.uuid4().hex[:8]}"
    all_results: list[InstanceResult] = []
    launched_run_ids: list[str] = []
    project_names: list[str] = []
    # Print + result accumulation happens from many threads; one lock is
    # enough since contention is tiny (a few prints per sample).
    print_lock = threading.Lock()

    def _process_sample(sample: dict[str, Any]) -> list[InstanceResult]:
        sample_id = _target_id(sample)
        with print_lock:
            print(f"\n=== target={sample_id}  parallel={parallel}  launching ===", flush=True)
        jobs: list[tuple[int, str]] = [
            (idx, f"{base_run_id}_{_safe_id(sample_id)}_{idx}")
            for idx in range(parallel)
        ]
        sample_run_ids = [rid for _, rid in jobs]
        sample_project_names: list[str] = []
        with print_lock:
            launched_run_ids.extend(sample_run_ids)

        try:
            with ThreadPoolExecutor(
                max_workers=parallel, thread_name_prefix=f"probe-{_safe_id(sample_id)}"
            ) as pool:
                futures = [
                    pool.submit(
                        _launch_and_probe,
                        server_url=server_url,
                        sample=sample,
                        instance_idx=idx,
                        cage_run_id=rid,
                        readiness_timeout=readiness_timeout,
                    )
                    for idx, rid in jobs
                ]
                sample_results = [f.result() for f in as_completed(futures)]
            sample_results.sort(key=lambda r: r.instance_idx)
            with print_lock:
                print(f"\n=== target={sample_id}  result ===", flush=True)
                for r in sample_results:
                    _print_one(r)
                    if r.project_name:
                        project_names.append(r.project_name)
                        sample_project_names.append(r.project_name)
            return sample_results
        finally:
            if not keep:
                _teardown_launched_targets(
                    launched_run_ids=sample_run_ids,
                    project_names=sample_project_names,
                )

    # Fully parallel by default — each sample is its own compose project,
    # they don't share docker network or evaluator container. Limit only if
    # the host can't take 10 simultaneous Java microservice builds.
    if samples_parallel is None or samples_parallel < 1:
        samples_parallel = max(1, len(samples))

    try:
        with ThreadPoolExecutor(
            max_workers=samples_parallel, thread_name_prefix="sample"
        ) as outer_pool:
            sample_futures = [outer_pool.submit(_process_sample, s) for s in samples]
            for fut in as_completed(sample_futures):
                all_results.extend(fut.result())
    finally:
        if not keep:
            _full_teardown(
                server_url=server_url,
                embedded=embedded,
                launched_run_ids=launched_run_ids,
                project_names=project_names,
            )
        elif embedded is not None:
            # ``--keep`` still tears the server down, but leaves the targets up.
            try:
                embedded.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("embedded target_server stop failed: %s", exc)

    passed = sum(1 for r in all_results if r.passed)
    return CheckSummary(
        total=len(all_results),
        passed=passed,
        failed=len(all_results) - passed,
        results=all_results,
    )


def _safe_id(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def _print_one(r: InstanceResult) -> None:
    status = "PASS" if r.passed else "FAIL"
    head = (
        f"  [{status}] instance={r.instance_idx} "
        f"project={r.project_name or '-'} "
        f"services={r.services_running}/{r.services_total} "
        f"duration={r.duration_s:.1f}s"
    )
    print(head)
    if r.network_subnet:
        print(f"         network={r.network_name} subnet={r.network_subnet}")
    for note in r.notes:
        print(f"         {note}")
    if r.error:
        # Multi-line errors (compose stderr is several KB) are unreadable on
        # one line — split, indent, and trim the bulky boilerplate (compose
        # progress chatter) so the actionable last few lines stay visible.
        err = r.error
        lines = [ln for ln in err.splitlines() if ln.strip()]
        if len(lines) <= 1:
            print(f"         error: {err}")
        else:
            print("         error:")
            keep = lines if len(lines) <= 20 else (
                lines[:3] + ["         ... (trimmed) ..."] + lines[-12:]
            )
            for ln in keep:
                print(f"           {ln}")


def print_summary(summary: CheckSummary) -> None:
    print()
    print(f"Summary: {summary.passed}/{summary.total} passed, {summary.failed} failed.")
    if summary.failed:
        print()
        print("Failures:")
        for r in summary.results:
            if not r.passed:
                print(f"  {r.sample_id}/instance_{r.instance_idx}: {r.error or 'unknown'}")
