"""Garbage-collection Cage CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command()
@click.option(
    "--apply",
    "apply",
    is_flag=True,
    default=False,
    help=(
        "Actually remove resources. Without --apply, ``cage gc`` only reports "
        "what it would do (dry-run)."
    ),
)
@click.option(
    "--namespace",
    "namespace",
    default=None,
    help=(
        "Restrict the GC to one cage.target.namespace label. "
        "Use to keep ``cage gc`` from touching another cage server's "
        "resources on a shared host."
    ),
)
@click.option(
    "--run-id",
    "run_id_filter",
    default=None,
    help="Only classify and (if --apply) sweep this single run_id.",
)
@click.option(
    "--root",
    "extra_roots",
    multiple=True,
    type=click.Path(),
    help=(
        "Extra ``.cage_runs/`` search root. Default is auto-discovered: "
        "$CAGE_RUNS_ROOT takes precedence; otherwise walk up from cwd "
        "to the cage source root (any ancestor with ``cage/cli/main.py``) "
        "and scan downward for ``.cage_runs/`` directories. Repeatable."
    ),
)
def gc(
    apply: bool,
    namespace: str | None,
    run_id_filter: str | None,
    extra_roots: tuple[str, ...],
) -> None:
    """Reclaim docker resources whose owning run is no longer running.

    \b
    Walks every container / network / named volume labelled with a
    ``cage.run_id`` and decides per run_id:

      \b
      * alive  — .cage_runs/<rid>/ is actively ticking (recent
                 progress.json mtime, or fresh planned_trials.json).
                 Skipped.
      * dead   — .cage_runs/<rid>/ exists but ``completed_at`` is set,
                 or all progress files are stale. Reclaim.
      * orphan — no .cage_runs/<rid>/ at all. Reclaim.

    \b
    ``.cage_runs/`` is never modified — it remains the immortal source
    of truth for run artifacts.

    Default is **dry-run**. Add ``--apply`` to actually remove.
    """

    from cage.gc.runner import (
        DECISION_ALIVE,
        DECISION_DEAD,
        collect_gc_resource_counts,
        default_cage_runs_roots,
        gc_all,
        gc_run,
    )

    extra_paths = [Path(p) for p in extra_roots]
    auto_roots = default_cage_runs_roots()
    roots = extra_paths + auto_roots
    if not roots:
        click.echo(
            "warning: auto-discovery found no .cage_runs/ under cwd or any "
            "ancestor cage source tree. Every run_id will be treated as "
            "ALIVE (safety) so --apply is a no-op. Set $CAGE_RUNS_ROOT or "
            "pass --root <path> to enable reclamation.",
            err=True,
        )

    if run_id_filter:
        # Single-run path. Skip the full enumeration; look up only the
        # requested run_id's resource counts. ``namespace`` is plumbed
        # through to ``gc_run`` so the sweep honors the same scope as
        # enumeration — otherwise --run-id + --namespace would sweep
        # cross-namespace resources sharing the run_id (HIGH-B fix).
        all_resources = collect_gc_resource_counts(
            namespace=namespace,
            search_roots=roots,
        )
        counts = all_resources.get(
            run_id_filter,
            {"containers": 0, "networks": 0, "volumes": 0},
        )
        decision = gc_run(
            run_id_filter,
            counts,
            search_roots=roots,
            apply=apply,
            namespace=namespace,
        )
        decisions = [decision]
        applied = apply
    else:
        report = gc_all(namespace=namespace, apply=apply, search_roots=roots)
        decisions = report.decisions
        applied = report.applied

    rows = []
    swept_containers = swept_networks = swept_volumes = 0
    alive = dead = orphan = 0
    for d in decisions:
        if d.decision == DECISION_ALIVE:
            alive += 1
        elif d.decision == DECISION_DEAD:
            dead += 1
        else:
            orphan += 1
        if d.swept is not None:
            swept_containers += d.swept.containers_removed
            swept_networks += d.swept.networks_removed
            swept_volumes += d.swept.volumes_removed
        rows.append({
            "run_id": d.run_id,
            "decision": d.decision,
            "reason": d.reason,
            "resources": {
                "containers": d.container_count,
                "networks": d.network_count,
                "volumes": d.volume_count,
            },
            "removed": (
                {
                    "containers": d.swept.containers_removed,
                    "networks": d.swept.networks_removed,
                    "volumes": d.swept.volumes_removed,
                    "errors": d.swept.errors or [],
                }
                if d.swept is not None
                else None
            ),
        })

    payload = {
        "applied": applied,
        "namespace": namespace,
        "search_roots": [str(r) for r in roots],
        "counts": {
            "alive": alive,
            "dead": dead,
            "orphan": orphan,
        },
        "removed": {
            "containers": swept_containers,
            "networks": swept_networks,
            "volumes": swept_volumes,
        },
        "decisions": rows,
    }
    # ensure_ascii=False so the human-readable ``reason`` strings render their
    # punctuation (em dash, "<5m") directly instead of \uXXXX escapes. Still
    # valid UTF-8 JSON for jq / downstream parsers.
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
