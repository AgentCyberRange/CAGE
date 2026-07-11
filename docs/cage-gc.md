# `cage gc` — reclaim docker resources from non-running runs

> Reference: `cage/gc/runner.py`, `cage/experiment/engine/live/liveness.py`, `cage/cli/commands/gc.py`

> ## ⚠️ USE WITH CAUTION
>
> `cage gc --apply` **permanently deletes** docker containers, networks,
> and named volumes. Volume deletion is **irreversible**: a database
> volume from a crashed pentest run carries the only post-trial state
> for that trial, and once it's gone there is no rollback.
>
> Before you ever pass `--apply`, **run the same command without it**
> (dry-run is the default) and read the JSON output carefully. Look at
> the `decision` and `reason` fields for every run_id. If anything is
> unexpected, stop and investigate before applying.
>
> **On a host shared between multiple cage namespaces**, always pass
> `--namespace=<your_namespace>`. Omitting it means GC considers
> every cage-labelled resource on the host.

---

## What it does

`cage gc` walks every docker container / network / named volume that
carries a `cage.run_id` label, classifies each owning run as
**alive / dead / orphan**, and (with `--apply`) tears down the
resources of dead and orphan runs.

| Decision | When | Action with `--apply` |
|---|---|---|
| `alive` | `.cage_runs/<rid>/` shows recent activity (progress.json mtime within 5 min, or fresh `planned_trials.json`, or pending trials per dashboard.json) | **Skip — always preserved** |
| `dead` | `.cage_runs/<rid>/dashboard.json` has `completed_at` set, OR progress files exist but are all stale | Reclaim its docker resources |
| `orphan` | No `.cage_runs/<rid>/` directory at all (artifact dir gone or never written) | Reclaim its docker resources |

**`.cage_runs/` is never modified by GC.** It's the immortal source of
truth for run history. GC only reads it.

---

## Quick reference

```bash
# Dry-run — see what would be reclaimed. Default mode, safe.
cage gc

# Same, but namespace-scoped (use on shared hosts).
cage gc --namespace my-ns

# Same, but for one specific run only.
cage gc --run-id run-20260518100000-abc123

# Point at .cage_runs/ explicitly (auto-discovered from cwd otherwise).
cage gc --root /path/to/.cage_runs/

# Actually do the cleanup. IRREVERSIBLE.
cage gc --apply --namespace my-ns

# Server startup also calls GC automatically. Disable per-server with:
CAGE_STARTUP_GC=0 python -m cage.target.serve …
```

---

## Hard rules

### 1. Default is dry-run. `--apply` is opt-in.

Without `--apply`, `cage gc` only reports. No docker commands run.
Use this to verify the classification before committing to a removal.

There is a second safety fallback: if auto-discovery finds **no** `.cage_runs/`
under the cwd or any ancestor cage source tree (and none was passed with
`--root` / `$CAGE_RUNS_ROOT`), GC has no artifact dir to judge aliveness
against, so it treats **every** run_id as ALIVE and `--apply` becomes a no-op.
It warns on stderr rather than reclaiming resources it can't reason about.

### 2. `--namespace` is namespace-scoped, agent containers are not.

The namespace filter restricts target-side resources
(`cage.target.namespace=<ns>`). Agent containers (the ones
running the LLM agent CLI) do **not** carry namespace labels, so they
are always swept by `cage.run_id` alone whenever the run is classified
non-alive. This is intentional: agent containers belong to the
orchestrator process, not to the target_server.

In practice this means on a shared host, you cannot accidentally
delete a *peer namespace's target_stack* by running `cage gc
--namespace mine --apply`. But you would still delete the *agent
container* if it shares a run_id (which it shouldn't, unless someone
re-used a uuid).

### 3. `.cage_runs/` is read-only to GC.

GC reads `dashboard.json`, `planned_trials.json`, and
`trials/*/proxy/progress.json` mtimes to decide aliveness. It never
writes, modifies, or deletes anything under `.cage_runs/`. Your trial
artifacts (scores, trajectories, state snapshots, proxy logs) are
safe regardless of what `--apply` does to docker.

### 4. Don't move or rename `.cage_runs/<rid>/` while a run is live.

The liveness check is keyed on `.cage_runs/<rid>/` existence + mtime.
If you `mv` an active run's artifact directory mid-run, the next
`cage gc --apply` will classify it as **orphan** and reap its docker
containers. Wait for the run to finish before touching its artifacts.

### 5. There is a small race window between enumeration and sweep.

GC does two passes: enumerate docker resources, then per run_id
classify against `.cage_runs/`. Between the two, a brand-new run can
start ticking. The new run's progress.json may not yet exist, so it
could be misclassified as dead. The window is sub-second on a quiet
host, but **do not run `cage gc --apply` immediately after starting a
new experiment** — wait until the first proxy request lands (you'll
see `proxy/progress.json` appear under `.cage_runs/<new-rid>/trials/`).

---

## Output schema

`cage gc` prints a single JSON object to stdout:

```json
{
  "applied": false,
  "namespace": null,
  "search_roots": ["/data/.../.cage_runs"],
  "counts": {"alive": 1, "dead": 2, "orphan": 0},
  "removed": {"containers": 0, "networks": 0, "volumes": 0},
  "decisions": [
    {
      "run_id": "run-20260518100000-abc",
      "decision": "alive",
      "reason": "1 active trial(s), recent progress.json tick",
      "resources": {"containers": 3, "networks": 1, "volumes": 0},
      "removed": null
    },
    {
      "run_id": "run-20260518090000-xyz",
      "decision": "dead",
      "reason": "completed_at=2026-05-18T09:32:00",
      "resources": {"containers": 9, "networks": 6, "volumes": 1},
      "removed": null
    }
  ]
}
```

Fields:

- `applied` — `true` when `--apply` was passed; `false` for dry-run.
- `namespace` — the `--namespace` filter (or `null` for all namespaces).
- `counts` — per-classification totals.
- `removed` — total resources actually deleted (always all-zero in
  dry-run).
- `decisions[].reason` — short human string describing why GC
  classified that way. Use this to sanity-check before `--apply`.
- `decisions[].removed` — `null` in dry-run; populated with
  per-component counts after `--apply`.

---

## Server startup auto-GC

`python -m cage.target.serve` (which the orchestrator embeds per run) calls
`gc_all(apply=True, namespace=<own_namespace>)` after claiming
ownership of its docker network. This guarantees that a server
restarted after SIGKILL / OOM / host reboot doesn't pile up
crashed-run resources for its own namespace.

To disable:

```bash
CAGE_STARTUP_GC=0 python -m cage.target.serve …
```

Set explicitly to `0`, `false`, or `no`. Empty string is treated as
"use default" (on).

---

## Sibling commands

| Command | When to use |
|---|---|
| `cage gc` | **Periodic** / on startup. Decides aliveness from `.cage_runs/`. |
| `cage gc --run-id <rid> --apply` | **Targeted** kill -9 / OOM recovery for **one** specific rid. |

`cage gc` is the default tool. Add both `--run-id` and `--apply` when you need
to force-remove one specific run's resources.

---

## Failure recovery

If `cage gc --apply` deletes something it shouldn't:

1. **It cannot be undone.** Docker volume removal is final; there is
   no `docker volume restore` even with `--force`.
2. **`.cage_runs/` is untouched.** Your scores, trajectories, and
   state snapshots are still there. Re-run the affected trials.
3. **File a bug** if a live run was killed: `is_run_running()` should
   never return `False` for an actively ticking run. Include the
   `.cage_runs/<rid>/` listing (especially `dashboard.json` and
   any `trials/*/proxy/progress.json`) so we can reproduce.

---

## Design reference

The behaviour described above is contracted with the web inspector:
`cage.experiment.engine.live.fs_signals.dashboard_pending_count` is the same function
that the inspector uses to compute its "running" badge. The two paths
share the file by import, not by duplication, so they cannot silently
diverge. See `docs/plans/2026-05-17-cleanup-p0-design.md` for the
full design and rationale.
