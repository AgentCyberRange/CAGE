"""Resume planning, compatibility, decisions, and replay reconstruction.

Owns everything about resuming a run: read-only analysis (``analyze_resume_plan``),
manifest/trial-plan compatibility, per-trial resume decisions (rerun/replay/keep/
capped), archiving prior attempts, replaying cached results, and reconstructing
trial/summary info from planned-trial records (also used by the interrupted-run
dashboard path). Reads the durable record format from ``artifacts.record_snapshots``/``canonical_marks``.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cage.artifacts.jsonio import (
    _load_json_file,
    _write_json_file,
)
from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.record_snapshots import (
    RUN_MANIFEST_FILE,
    _build_run_manifest,
    _file_sha256,
    _json_fingerprint,
)
from cage.artifacts.run_storage import (
    PROXY_DIRNAME,
    PROXY_LOG_FILENAME,
    TASK_OUTPUT_FILENAME,
    TRIALS_DIRNAME,
    RunStorage,
    trial_meta_path,
    trial_path,
    trial_progress_path,
)
from cage.benchmarks import normalize_sample_id
from cage.config.sections import ResumeKeepIf
from cage.contracts.sample_keys import sample_pass_index
from cage.contracts.trial_status import count_trials
from cage.experiment.engine.hooks import (
    HookContext,
    default_trial_sequence,
    expand_trials_for_passk,
)
from cage.experiment.engine.run_context import ExperimentRun
from cage.experiment.engine.termination import (
    TerminationReason,
    cancelled_before_start_termination,
    user_interrupted_termination,
)
from cage.experiment.model import Trial, TrialRecord, TrialResult, TrialStatus, parse_trial_id
from cage.sandbox.exec import Timing
from cage.sandbox.naming import (
    _parse_agent_label,
)

logger = logging.getLogger("cage.experiment.engine.resume")

_DEFAULT_RESUME_RETRY_REASONS: frozenset[str] = frozenset({
    # Infra
    TerminationReason.TARGET_UNAVAILABLE.value,
    TerminationReason.TRIAL_ERROR.value,
    TerminationReason.CANCELLED_BEFORE_START.value,
    TerminationReason.OOM_KILLED.value,
    # Upstream model proxy
    TerminationReason.MODEL_QUOTA_EXHAUSTED.value,
    TerminationReason.MODEL_RATE_LIMITED.value,
    TerminationReason.MODEL_BAD_GATEWAY.value,
    TerminationReason.MODEL_AUTH_ERROR.value,
    TerminationReason.MODEL_CONTEXT_OVERFLOW.value,
    TerminationReason.MODEL_TIMEOUT.value,
    TerminationReason.MODEL_ERROR.value,
    # User-interrupted trials are typically not "I want to drop this" —
    # they're collateral damage from a Ctrl+C on the parent run.
    TerminationReason.USER_INTERRUPTED.value,
    # Legacy/coarse alias: some records store an interrupted trial as
    # status=failed, reason="interrupted" (no meta.json), instead of the
    # canonical "user_interrupted". Treat it the same — it means "didn't finish".
    "interrupted",
})

# A canonical record still marked "running"/"planned" during resume is stale by
# definition — the run is not active, so the trial never finished and its record
# was never finalized. Resume must NOT trust such a record as lifecycle truth; it
# defers to the accurate meta.json (which records interrupted/user_interrupted).
_NON_AUTHORITATIVE_RESUME_STATUS: frozenset[str] = frozenset({"planned", "running"})


def _canonical_resume_trial_id(value: object) -> str:
    """Normalize trial/sample ids for resume compatibility checks.

    Run artifacts keep the original casing on disk, but resume compatibility
    should not fail only because a dataset renamed ``pb-SIYUCMS`` to
    ``pb-siyucms``. Preserve separators so real plan-shape changes still fail.
    """

    return "/".join(normalize_sample_id(part) for part in str(value or "").split("/"))

def _resume_trial_plan_fingerprint(records: list[dict[str, Any]]) -> str:
    canonical: list[dict[str, Any]] = []
    for item in records:
        canonical.append({
            "trial_id": _canonical_resume_trial_id(item.get("trial_id")),
            "trial_index": int(item.get("trial_index") or 0),
            "trial_type": str(item.get("trial_type") or ""),
            "sample_id": _canonical_resume_trial_id(item.get("sample_id")),
        })
    return _json_fingerprint(canonical)

def _resume_trial_plans_compatible(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> bool:
    if _json_fingerprint(previous) == _json_fingerprint(current):
        return True
    return (
        _resume_trial_plan_fingerprint(previous)
        == _resume_trial_plan_fingerprint(current)
    )

def _find_resume_trial_record(
    reader: ExperimentArtifactReader, trial: Trial
) -> "TrialRecord | None":
    """Resolve the prior ``TrialRecord`` for ``trial`` with a single-file read.

    Resume only ever needs *this* trial's record (its status/termination and
    artifact refs), not the whole-run snapshot. Canonical ``trial_id`` equals
    ``Trial.id``, so the common path is an O(1) id lookup against one record
    file. The lazy scan is a legacy fallback for runs whose stored id does not
    equal the planned id (matched by task/pass/sample) — it stops at the first
    match rather than materializing every record or any ``events.jsonl``.
    """

    record = reader.trial_record_by_id(str(trial.id))
    if record is not None:
        return record
    for candidate in reader.iter_trial_records():
        if _resume_trial_record_matches(candidate, trial):
            return candidate
    return None


def _canonical_plan_resume_trials(run_dir: Path) -> list[dict[str, Any]]:
    """Project canonical ``ExperimentPlan`` trials into resume plan records.

    ``_assert_resume_compatible`` historically compared ``planned_trials.json``
    entries shaped around legacy trial ids. Canonical trial ids include the
    subject id so they can be globally unique in a future multi-subject run
    directory. Resume compatibility only needs the user task/pass order, so this
    helper deliberately compares task ids and source sample ids instead of the
    full canonical trial id.
    """

    try:
        plan = ExperimentArtifactReader(run_dir).load_plan()
    except (FileNotFoundError, OSError, ValueError):
        return []
    source_sample_by_task = {
        task.task_id: task.source_sample_id
        for task in plan.tasks
    }
    has_multiple_passes = any(trial.pass_index > 1 for trial in plan.trials)
    records: list[dict[str, Any]] = []
    for index, trial in enumerate(plan.trials):
        task_id = str(trial.task_id)
        trial_id = (
            f"{task_id}/pass_{trial.pass_index}"
            if has_multiple_passes else task_id
        )
        records.append({
            "trial_id": trial_id,
            "trial_index": index,
            "trial_type": "task",
            "sample_id": str(source_sample_by_task.get(task_id, task_id)),
        })
    return records

def _normalize_planned_trial_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in records:
        normalized.append({
            "trial_id": str(item.get("trial_id") or ""),
            "trial_index": int(item.get("trial_index") or 0),
            "trial_type": str(item.get("trial_type") or ""),
            "sample_id": str(item.get("sample_id") or ""),
        })
    return normalized

def _first_trial_id(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(none)"
    return str(records[0].get("trial_id") or "(blank)")

def _raise_resume_trial_plan_changed(
    run_dir: Path,
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> None:
    raise ResumeCompatibilityError(
        f"Refusing to resume run {run_dir.name!r}: trial plan changed.\n"
        f"Previous plan: {len(previous)} trial(s), first={_first_trial_id(previous)!r}.\n"
        f"Current plan: {len(current)} trial(s), first={_first_trial_id(current)!r}.\n"
        "Start a new run-id, or restore the original config before using --resume."
    )

def _warn_resume_manifest_drift(
    run_dir: Path,
    previous_manifest: dict[str, Any],
    current_manifest: dict[str, Any],
    *,
    warn_trial_fingerprint: bool,
) -> None:
    """Emit resume drift warnings from the legacy manifest projection.

    ``run_manifest.json`` is no longer the source of truth when a canonical
    ``ExperimentPlan`` exists, but it still carries useful diagnostics for old
    and mixed runs. Keep those diagnostics centralized so the resume decision
    path can choose its authoritative trial-plan source separately.
    """

    if (
        warn_trial_fingerprint
        and previous_manifest.get("trial_plan_fingerprint")
        != current_manifest.get("trial_plan_fingerprint")
    ):
        logger.warning(
            "resume: trial ids changed only by canonical sample-id casing "
            "for run %s; continuing with the current lowercase plan",
            run_dir.name,
        )
    if (
        previous_manifest.get("project_yml_sha256")
        != current_manifest.get("project_yml_sha256")
    ):
        logger.warning(
            "resume: project.yml changed for run %s; continuing because "
            "the trial plan is unchanged",
            run_dir.name,
        )
    if (
        previous_manifest.get("semantic_config_fingerprint")
        != current_manifest.get("semantic_config_fingerprint")
    ):
        logger.warning(
            "resume: configuration changed for run %s; continuing because "
            "the trial plan is unchanged",
            run_dir.name,
        )

def _resume_run_not_found_message(run_dir: Path) -> str:
    """Clear 'no such run' error: where we looked + which run-ids exist there."""
    runs_root = run_dir.parent  # .cage_runs/<agent>
    siblings: list[str] = []
    if runs_root.is_dir():
        siblings = sorted(p.name for p in runs_root.iterdir() if p.is_dir())
    lines = [
        f"No run with id {run_dir.name!r} to resume "
        f"(looked under {runs_root}).",
    ]
    if siblings:
        shown = ", ".join(siblings[:10])
        more = f" (+{len(siblings) - 10} more)" if len(siblings) > 10 else ""
        lines.append(f"Existing run ids for this agent: {shown}{more}")
    else:
        lines.append(
            "No previous runs exist for this agent yet — start one without "
            "--resume first."
        )
    return "\n".join(lines)


def _assert_resume_compatible(run_dir: Path, current_manifest: dict[str, Any]) -> None:
    """Fail fast if ``--resume`` would combine different trial plans."""
    if not run_dir.is_dir():
        raise ResumeCompatibilityError(_resume_run_not_found_message(run_dir))
    manifest_path = run_dir / RUN_MANIFEST_FILE
    previous_manifest = _load_json_file(manifest_path)
    current_trials = _normalize_planned_trial_records(
        current_manifest.get("planned_trials", []),
    )
    canonical_trials = _normalize_planned_trial_records(
        _canonical_plan_resume_trials(run_dir),
    )

    if canonical_trials:
        if not _resume_trial_plans_compatible(canonical_trials, current_trials):
            _raise_resume_trial_plan_changed(run_dir, canonical_trials, current_trials)
        if isinstance(previous_manifest, dict) and previous_manifest:
            manifest_trials = _normalize_planned_trial_records(
                previous_manifest.get("planned_trials", []),
            )
            manifest_matches = _resume_trial_plans_compatible(
                manifest_trials,
                current_trials,
            )
            if manifest_trials and not manifest_matches:
                logger.warning(
                    "resume: ignoring stale %s trial plan for run %s because "
                    "canonical ExperimentPlan matches the current plan",
                    RUN_MANIFEST_FILE,
                    run_dir.name,
                )
            _warn_resume_manifest_drift(
                run_dir,
                previous_manifest,
                current_manifest,
                warn_trial_fingerprint=manifest_matches,
            )
        return

    if isinstance(previous_manifest, dict) and previous_manifest:
        previous_trials = _normalize_planned_trial_records(
            previous_manifest.get("planned_trials", []),
        )
        if not _resume_trial_plans_compatible(previous_trials, current_trials):
            _raise_resume_trial_plan_changed(run_dir, previous_trials, current_trials)
        _warn_resume_manifest_drift(
            run_dir,
            previous_manifest,
            current_manifest,
            warn_trial_fingerprint=True,
        )
        return

    previous_trials = _normalize_planned_trial_records(_load_planned_trials(run_dir))
    if not previous_trials:
        raise ResumeCompatibilityError(
            f"Refusing to resume run {run_dir.name!r}: missing {RUN_MANIFEST_FILE}, "
            "planned_trials.json, and canonical ExperimentPlan artifacts, so "
            "Cage cannot verify that the current config matches the original run."
        )
    current_trials = _normalize_planned_trial_records(
        current_manifest.get("planned_trials", []),
    )
    if not _resume_trial_plans_compatible(previous_trials, current_trials):
        _raise_resume_trial_plan_changed(run_dir, previous_trials, current_trials)

    project_snapshot = run_dir / "project.yml"
    if (
        project_snapshot.exists()
        and _file_sha256(project_snapshot) != current_manifest.get("project_yml_sha256")
    ):
        logger.warning(
            "resume: project.yml changed for run %s; continuing because "
            "the trial plan is unchanged",
            run_dir.name,
        )

def _merge_interrupted_planned_trials(
    run_dir: Path,
    trial_infos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    planned_trials = _load_planned_trials(run_dir)
    if not planned_trials:
        return trial_infos

    by_id = {str(info.get("trial_id")): info for info in trial_infos if info.get("trial_id")}
    for planned in planned_trials:
        trial_id = str(planned.get("trial_id") or "")
        if not trial_id:
            continue
        meta_path = trial_meta_path(run_dir, trial_id)
        meta = _load_json_file(meta_path)
        trial_info = by_id.get(trial_id)
        if trial_info is None:
            trial_info = _trial_info_from_planned_trial(planned, meta)
            trial_infos.append(trial_info)
            by_id[trial_id] = trial_info
        else:
            _copy_missing_trial_meta(trial_info, meta)

        if trial_info.get("termination_reason"):
            continue

        termination = (
            user_interrupted_termination()
            if meta or trial_path(run_dir, trial_id).exists()
            else cancelled_before_start_termination()
        )
        trial_info.update(termination.to_metadata())
        meta_payload = {
            **_trial_info_from_planned_trial(planned, meta),
            **termination.to_metadata(),
        }
        _write_json_file(meta_path, meta_payload)

    return sorted(trial_infos, key=_trial_sort_key)

@dataclass
class _AgentResumePlan:
    """Result of a per-agent resume dry-run analysis (no side effects)."""

    agent_label: str
    run_dir: Path
    total: int
    rerun: list[tuple[str, str, str]]   # (trial_id, category, detail label)
    replay: list[tuple[str, str, str]]  # (trial_id, category, detail label)
    capped: list[tuple[str, int]]       # (trial_id, attempts) — hit max_attempts
    archives_present: int          # existing .before_resume_* siblings on disk
    max_trial: int | None = None   # execution cap active this invocation (None = off)
    considered: int = 0            # trials below the cap (== total when uncapped)

class ResumeCompatibilityError(ValueError):
    """Raised when ``--resume`` would mix incompatible run configurations."""

def analyze_resume_plan(run: ExperimentRun) -> list[_AgentResumePlan]:
    """Compute what ``cage run --resume`` *would* do, without side effects.

    Walks every configured agent the same way ``_run_single_agent`` would,
    builds the planned trial list (incl. pass@k expansion), then calls
    ``_partition_resumed_trials`` in read-only mode. No containers are
    started, no target stacks brought up, no archives created, no
    summary/dashboard files rewritten.

    Returned plans are suitable for both the CLI dry-run printer and for
    programmatic checks (e.g. integration tests asserting "this run is
    fully classified — nothing left to retry").
    """
    cage_runs = _cage_runs_root(run)
    run_id = run.run_id or ""
    plans: list[_AgentResumePlan] = []
    for agent in run.agents:
        agent_label = agent.label()
        agent_dir_name, _mode = _parse_agent_label(agent_label)
        run_dir = cage_runs / agent_dir_name / run_id
        storage = _ReadOnlyRunStorage(run_dir)

        # Build the same trial list ``_run_single_agent`` would (single-pass
        # default; pass@k expansion when configured). We don't fire hooks —
        # those are allowed side effects in real runs but we want a pure
        # snapshot here.
        bench_limit = run.sample_limit
        bench_sample_ids = run.sample_ids
        samples = list(
            run.benchmark.iter_samples_limited(bench_limit, bench_sample_ids, run.sample_slice)
        )
        hook_ctx = HookContext(
            experiment_config={"name": run.name},
            samples=samples,
            trials_completed=[],
            trials_pending=[],
            run_artifacts_dir=str(run_dir),
        )
        trials = default_trial_sequence(hook_ctx)
        trials = expand_trials_for_passk(trials, max(1, run.execution.passk))
        for index, t in enumerate(trials):
            t.index = index

        if run.resume:
            _assert_resume_compatible(
                run_dir,
                _build_run_manifest(run, agent, trials),
            )

        # Mirror the real run path's execution cap so the preview matches.
        trials_exec = _cap_trials_for_execution(trials, run.execution.max_trial)

        # One read-only classification pass feeds every row below, so the
        # preview is bit-for-bit what the real run would decide (same helper).
        decisions = _resume_decisions(
            storage, trials_exec,
            _resolve_retry_reasons(run.resume_retry_reasons),
            run.resume_max_attempts,
            run.resume_keep_if,
            run.resume_select_id_pattern,
        )
        # Each row carries the decision label so the printer can show *why*
        # (e.g. "model_error" for re-runs, "keep_if min_rounds (ran 157 ≥ 100)"
        # for salvaged replays).
        rerun_rows = [
            (t.id, d.category, d.label) for t, d in decisions if d.action == "rerun"
        ]
        replay_rows = [
            (t.id, d.category, d.label) for t, d in decisions if d.action == "replay"
        ]
        capped = [(t.id, d.attempts) for t, d in decisions if d.action == "capped"]

        archives_present = _count_archives_for_trials(run_dir, trials)

        plans.append(_AgentResumePlan(
            agent_label=agent_label,
            run_dir=run_dir,
            total=len(trials),
            rerun=rerun_rows,
            replay=replay_rows,
            capped=capped,
            archives_present=archives_present,
            max_trial=run.execution.max_trial,
            considered=len(trials_exec),
        ))
    return plans

def _cage_runs_root(run: ExperimentRun) -> Path:
    """Resolve the ``.cage_runs`` dir the same way ``run_experiment`` would."""
    benchmark_dir = getattr(run, "benchmark_dir", None)
    if benchmark_dir:
        return Path(benchmark_dir).resolve() / ".cage_runs"
    return run.project_file.resolve().parent / ".cage_runs"

class _ReadOnlyRunStorage:
    """Drop-in stand-in for :class:`RunStorage` used by the analysis path.

    Exposes only ``run_dir`` — the partition function reaches through to
    raw paths after our earlier refactor, so this is sufficient. Refusing
    to expose ``trial_dir`` ensures any future code that tries to mkdir
    via the storage object will fail loudly here rather than silently
    polluting the run directory during a dry-run.
    """

    __slots__ = ("run_dir",)

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

def _count_archives_for_trials(run_dir: Path, trials: Iterable[Trial]) -> int:
    """How many ``.before_resume_*`` archives exist for the known trial ids.

    Archive naming is deterministic — a trial whose live dir is
    ``trials/<id>`` can only have archives named ``<basename>.before_resume_*``
    in the same parent. So we iterate the known trial list (cheap, we
    already have it in memory) and ``scandir`` each unique parent at most
    once. No recursion, no descent into ``proxy/`` or other heavy subdirs.

    The earlier rglob-based implementation took 315s on a run with 300MB
    proxy.jsonl files; this version is sub-second on the same data.
    """
    trials_root = run_dir / TRIALS_DIRNAME
    if not trials_root.is_dir():
        return 0
    # Cache parent-listing so two passk siblings sharing a parent only stat once.
    parent_listings: dict[Path, list[str]] = {}
    seen_archives: set[Path] = set()
    for trial in trials:
        live = trials_root / trial.id
        parent = live.parent
        if parent not in parent_listings:
            try:
                parent_listings[parent] = [p.name for p in parent.iterdir() if p.is_dir()]
            except OSError:
                parent_listings[parent] = []
        prefix = live.name + ".before_resume_"
        for name in parent_listings[parent]:
            if name.startswith(prefix):
                seen_archives.add(parent / name)
    return len(seen_archives)

def _archive_trial_dir_before_resume(
    storage: RunStorage, trial: Trial
) -> Path | None:
    """Move ``trials/<id>/`` aside before a resume re-run.

    Why: several per-trial artifacts (``proxy.jsonl``, ``tool_calls.jsonl``,
    ``state_pre/``, ``state_post/``) are append-mode or copy-without-cleanup
    on disk. Re-running into the same directory mixes the previous attempt's
    LLM trace and snapshot files into the new one, corrupting token counts,
    ``.traj`` files, and any state diff. Renaming preserves the failure
    evidence while guaranteeing the next attempt writes into an empty dir.
    """
    # ``trial_path`` is the pure (non-mkdir) accessor; ``storage.trial_dir``
    # would create the directory as a side effect.
    trial_dir = trial_path(storage.run_dir, trial.id)
    if not trial_dir.exists() or not any(trial_dir.iterdir()):
        return None
    ts = time.strftime("%Y%m%dT%H%M%S")
    archive = trial_dir.with_name(trial_dir.name + f".before_resume_{ts}")
    suffix = 1
    while archive.exists():
        archive = trial_dir.with_name(
            trial_dir.name + f".before_resume_{ts}_{suffix}"
        )
        suffix += 1
    trial_dir.rename(archive)
    logger.info("resume: archived %s -> %s", trial_dir, archive.name)
    return archive

def _count_trial_attempts(storage: RunStorage, trial: Trial) -> int:
    """Total attempts of ``trial`` recorded on disk.

    Counts the live trial dir plus every ``<id>.before_resume_<ts>``
    archive sibling. Used to enforce ``resume_max_attempts``.
    """
    trial_dir = trial_path(storage.run_dir, trial.id)
    parent = trial_dir.parent
    if not parent.is_dir():
        return 0
    prefix = trial_dir.name + ".before_resume_"
    archives = sum(
        1 for p in parent.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    )
    live = 1 if trial_dir.exists() and any(trial_dir.iterdir()) else 0
    return archives + live

def _cap_trials_for_execution(
    trials: list[Trial], max_trial: int | None
) -> list[Trial]:
    """Restrict which planned trials run this invocation (execution filter).

    ``max_trial`` caps by ascending global ``trial.index`` (pass-major order:
    pass_1 over all samples, then pass_2, …). The full plan is still recorded
    in the manifest / planned_trials.json, so ``--resume`` stays
    plan-compatible; trials at or beyond the cap are simply left pending for a
    later (capless) invocation. ``None`` or a negative value = no cap.
    """
    if max_trial is None or max_trial < 0:
        return trials
    return [t for t in trials if t.index < max_trial]

@dataclass
class _ResumeDecision:
    """Per-trial resume verdict — the single source of truth shared by the
    real run, the ``cage score`` path, and the ``--dry-run`` preview.

    ``action`` is one of:
      - ``"rerun"``  — re-attempt the trial (goes to ``pending``);
      - ``"replay"`` — keep the prior on-disk result (goes to ``replayed``);
      - ``"capped"`` — wanted a re-run but hit ``max_attempts``; replays the
        last failed result instead.

    ``label`` is the detailed per-trial explanation (printed next to each
    trial) — e.g. ``"keep_if min_rounds (ran 157 ≥ 100)"``. ``category`` is a
    stable coarse bucket for grouping/tallying (e.g. ``"keep_if:min_rounds"``,
    ``"model_error"``, ``"completed"``) — labels embed per-trial numbers and
    would otherwise never collapse into counts.
    """

    action: str
    label: str
    category: str
    reason: str          # raw termination_reason (lower-cased; "" if none)
    meta: dict[str, Any]
    attempts: int = 0

def _resolve_retry_reasons(extra_retry_reasons: Iterable[str]) -> set[str]:
    return set(_DEFAULT_RESUME_RETRY_REASONS) | {
        str(r).strip().lower() for r in extra_retry_reasons if str(r).strip()
    }

def _resume_trial_task_and_pass(trial: Trial) -> tuple[str, int]:
    """Return the benchmark task id and pass represented by a legacy trial.

    Runtime ``Trial.id`` values are still legacy directory ids such as
    ``range1`` or ``range1/pass_2``. Canonical ``TrialRecord.trial_id`` values
    include the subject id (``agent:model:profile/range1/pass_2``), so resume
    cannot require direct string equality. The stable bridge is the benchmark
    task id plus pass index, with ``sample["pass_index"]`` taking precedence
    because pass@k expansion stores that structured value before execution.
    """

    raw_id = str(trial.id or trial.sample_id or "trial")
    task_id, pass_index = parse_trial_id(raw_id)
    structured = sample_pass_index(trial.sample)
    return task_id, structured if structured is not None else pass_index

def _resume_trial_record_matches(trial_record: TrialRecord, trial: Trial) -> bool:
    """Return whether a canonical ``TrialRecord`` describes a runtime trial.

    During migration, resume receives legacy ``Trial`` objects but canonical
    artifacts store globally unique ``TrialRecord.trial_id`` values. This
    matcher keeps all resume read paths aligned: exact id matches support
    canonical-only callers, while task/pass matches support the real
    ``_run_single_agent`` path that still schedules legacy trial ids.
    """

    if str(trial_record.trial_id) == str(trial.id):
        return True
    task_id, pass_index = _resume_trial_task_and_pass(trial)
    record_task_id = str(getattr(trial_record, "task_id", "") or "")
    try:
        record_pass_index = int(getattr(trial_record, "pass_index", 1) or 1)
    except (TypeError, ValueError):
        record_pass_index = 1
    if record_task_id == task_id and record_pass_index == pass_index:
        return True
    return (
        bool(trial.sample_id)
        and record_task_id == str(trial.sample_id)
        and record_pass_index == pass_index
    )

def _load_resume_indexed_trial_json_artifact(
    run_dir: Path,
    trial: Trial,
    kind: str,
) -> dict[str, Any]:
    """Read one canonical trial JSON artifact through ``ArtifactIndex``.

    Resume compatibility code should not guess canonical artifact paths. This
    helper keeps fallback reads tied to the recorded ``TrialRecord`` refs and
    the run-level index; historical layouts still use their explicit legacy
    paths before reaching this helper.
    """

    reader = ExperimentArtifactReader(run_dir)
    trial_record = _find_resume_trial_record(reader, trial)
    if trial_record is None:
        return {}
    for artifact in trial_record.artifacts:
        if artifact.kind != kind:
            continue
        try:
            indexed = reader.find_artifact(
                artifact_id=artifact.artifact_id,
                path=artifact.path,
                kind=artifact.kind,
            )
            if indexed is None:
                continue
            payload = _load_json_file(reader.resolve_artifact_path(indexed))
        except (FileNotFoundError, KeyError, OSError, ValueError):
            continue
        return payload if isinstance(payload, dict) else {}
    return {}

def _resume_round_count(run_dir: Path, trial: Trial) -> int | None:
    """Agent rounds the prior attempt executed = ``progress.json`` /
    ``total_requests`` (one entry per upstream LLM request). ``None`` if the
    file is missing/unreadable (e.g. the trial never produced a proxy log)."""
    prog = _load_json_file(trial_progress_path(run_dir, trial.id))
    if isinstance(prog, dict):
        val = prog.get("total_requests")
        if isinstance(val, (int, float)):
            return int(val)
    return None

def _resume_duration_s(meta: dict[str, Any]) -> float | None:
    timing = (meta or {}).get("timing") or {}
    ms = timing.get("duration_ms")
    if isinstance(ms, (int, float)) and ms > 0:
        return ms / 1000.0
    return None

def _canonical_resume_meta_from_trial_record(
    trial_record: TrialRecord,
    trial: Trial,
) -> dict[str, Any]:
    """Project a canonical ``TrialRecord`` into resume-policy metadata.

    Resume policy still evaluates the legacy ``meta.json`` shape while the
    orchestrator is being migrated. This projection keeps the policy stable but
    lets canonical-only runs participate without recreating ``meta.json`` files
    just to make resume work.
    """

    meta: dict[str, Any] = {
        "trial_id": trial_record.trial_id,
        "trial_index": trial.index,
        "trial_type": trial.type.value,
        "status": trial_record.status,
    }
    reason = (
        trial_record.termination.reason
        or trial_record.status_reason
        or ""
    )
    if reason:
        meta["termination_reason"] = reason
    if trial_record.termination.exit_code is not None:
        meta["exit_code"] = trial_record.termination.exit_code
    if trial_record.termination.signal:
        meta["termination_signal"] = trial_record.termination.signal
    if trial_record.status_reason:
        meta["status_reason"] = trial_record.status_reason
    if trial_record.started_at:
        meta["started_at"] = trial_record.started_at
    if trial_record.completed_at:
        meta["completed_at"] = trial_record.completed_at
    return meta

def _load_resume_trial_meta(storage: RunStorage, trial: Trial) -> dict[str, Any] | None:
    """Load resume-policy metadata from legacy meta or canonical records.

    Canonical ``TrialRecord`` is the preferred read boundary once it carries
    lifecycle truth. Legacy ``meta.json`` remains the compatibility fallback for
    historical runs or initial snapshots whose trial record is still only
    planned.
    """

    # One targeted record read, not a whole-run snapshot. A non-planned record
    # is authoritative; a still-"planned" record (initial snapshot) defers to a
    # legacy meta.json if one exists, then falls back to the planned record.
    trial_record = _find_resume_trial_record(
        ExperimentArtifactReader(storage.run_dir), trial
    )
    if (
        trial_record is not None
        and trial_record.status not in _NON_AUTHORITATIVE_RESUME_STATUS
    ):
        return _canonical_resume_meta_from_trial_record(trial_record, trial)

    meta_path = trial_meta_path(storage.run_dir, trial.id)
    if meta_path.is_file():
        meta_raw = _load_json_file(meta_path)
        if isinstance(meta_raw, dict):
            return meta_raw

    if trial_record is not None:
        return _canonical_resume_meta_from_trial_record(trial_record, trial)
    return None

def _load_resume_task_output(storage: RunStorage, trial: Trial) -> dict[str, Any]:
    """Load replayable task output from legacy files or canonical artifacts.

    Canonical task-output artifacts are the preferred replay source because
    they are tied to ``TrialRecord`` and ``ArtifactIndex``. Resume still accepts
    historical ``trials/<id>/task_output.json`` files for legacy-only runs.
    """

    canonical_output = _load_resume_indexed_trial_json_artifact(
        storage.run_dir,
        trial,
        "task_output",
    )
    if canonical_output:
        return canonical_output
    trial_dir = trial_path(storage.run_dir, trial.id)
    legacy_output = _load_json_file(trial_dir / TASK_OUTPUT_FILENAME)
    if isinstance(legacy_output, dict) and legacy_output:
        return legacy_output
    return {}

def _keep_if_veto_label(
    run_dir: Path,
    trial: Trial,
    meta: dict[str, Any],
    keep_if: "ResumeKeepIf",
) -> tuple[str, str] | None:
    """Return ``(category, label)`` if any ``keep_if`` threshold rescues this
    trial, else ``None``.

    Checked in declaration order (min_rounds → min_duration_s → id_matches);
    the first match wins so the label names the concrete reason.
    """
    if keep_if.min_rounds is not None:
        rounds = _resume_round_count(run_dir, trial)
        if rounds is not None and rounds >= keep_if.min_rounds:
            return (
                "keep_if:min_rounds",
                f"keep_if min_rounds (ran {rounds} ≥ {keep_if.min_rounds})",
            )
    if keep_if.min_duration_s is not None:
        dur = _resume_duration_s(meta)
        if dur is not None and dur >= keep_if.min_duration_s:
            return (
                "keep_if:min_duration_s",
                f"keep_if min_duration_s (ran {dur:.0f}s ≥ "
                f"{keep_if.min_duration_s:.0f}s)",
            )
    if keep_if.id_matches and re.search(keep_if.id_matches, trial.id):
        return (
            "keep_if:id_matches",
            f"keep_if id_matches (/{keep_if.id_matches}/)",
        )
    return None

def _decide_resume_trial(
    storage: RunStorage,
    trial: Trial,
    retry_reasons: set[str],
    max_attempts: int,
    keep_if: "ResumeKeepIf | None",
    select_re: "re.Pattern[str] | None",
) -> _ResumeDecision:
    """Classify a single trial for resume. Pure read — never mutates disk.

    Order of evaluation:
      1. Base eligibility — meta missing/unparseable, blank status, or a
         non-completed ``termination_reason`` in ``retry_reasons`` ⇒ retry
         candidate. Otherwise replay (completed, or a reason not opted in).
      2. ``select`` positive gate — if set and the id doesn't match, the trial
         is out of scope this resume ⇒ replay (``not selected``).
      3. ``keep_if`` veto — a candidate that already did enough work
         (rounds / duration / id) ⇒ replay (salvaged).
      4. ``max_attempts`` cap — a candidate that exhausted its budget ⇒
         capped (replay last failed result).
      5. Otherwise ⇒ rerun.
    """
    # Resolve the meta path directly off storage.run_dir — avoid
    # ``storage.trial_dir`` which mkdir's as a side effect. This keeps the
    # function safe for the dry-run path that must not touch disk state.
    meta_raw = _load_resume_trial_meta(storage, trial)
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    reason = ""
    if not isinstance(meta_raw, dict):
        base_label = base_cat = "(no prior meta)"
    else:
        status = str(meta.get("status") or "").strip().lower()
        reason = str(meta.get("termination_reason") or "").strip().lower()
        if not status:
            base_label = base_cat = "blank status"
        elif status == TrialStatus.COMPLETED.value:
            return _ResumeDecision("replay", "completed", "completed", reason, meta)
        elif status == TrialStatus.RUNNING.value and not reason:
            # A "running" record during resume is stale (the run isn't active)
            # and carries no outcome ⇒ the trial was interrupted mid-flight with
            # its reason never recorded ⇒ retry. A running/interrupted record
            # that DID record a reason falls through to the reason check below,
            # so legitimate stops (e.g. tool_limit) are still correctly kept.
            base_label = base_cat = "interrupted (in-flight at shutdown)"
        elif status in ("planned", TrialStatus.PENDING.value) and not reason:
            # A canonical "planned" (or "pending") record with no terminal
            # outcome means the trial was scheduled but never actually executed
            # — typically left pending by an interrupted run or a prior
            # ``max_trial`` cap. There is no on-disk result to replay, so it
            # MUST be (re)run; treating it as "not a retry reason" would silently
            # drop every still-pending trial on --resume.
            base_label = base_cat = "never started (pending)"
        elif reason in retry_reasons:
            base_label = base_cat = reason or "(no reason)"
        else:
            cat = f"{reason or '(no reason)'} (not a retry reason)"
            return _ResumeDecision("replay", cat, cat, reason, meta)

    # (2) select gate — only ids matching the positive filter may re-run.
    if select_re is not None and not select_re.search(trial.id):
        return _ResumeDecision(
            "replay", f"not selected (id !~ /{select_re.pattern}/)",
            "not selected", reason, meta,
        )

    # (3) keep_if veto — salvage trials that already did enough work.
    if keep_if is not None and not keep_if.is_empty():
        veto = _keep_if_veto_label(storage.run_dir, trial, meta, keep_if)
        if veto is not None:
            category, label = veto
            return _ResumeDecision("replay", label, category, reason, meta)

    # (4) attempt cap.
    attempts = _count_trial_attempts(storage, trial)
    if max_attempts > 0 and attempts >= max_attempts:
        return _ResumeDecision(
            "capped", f"retry-capped ({attempts} attempts)", "retry-capped",
            reason, meta, attempts,
        )

    return _ResumeDecision("rerun", base_label, base_cat, reason, meta, attempts)

def _resume_decisions(
    storage: RunStorage,
    trials: list[Trial],
    retry_reasons: set[str],
    max_attempts: int = 0,
    keep_if: "ResumeKeepIf | None" = None,
    select_id_pattern: str | None = None,
) -> list[tuple[Trial, _ResumeDecision]]:
    """Classify every trial (read-only). Shared by the run path, the dry-run
    preview, and ``_partition_resumed_trials``."""
    select_re = re.compile(select_id_pattern) if select_id_pattern else None
    return [
        (t, _decide_resume_trial(
            storage, t, retry_reasons, max_attempts, keep_if, select_re
        ))
        for t in trials
    ]

def _split_resume_decisions(
    storage: RunStorage,
    decisions: list[tuple[Trial, _ResumeDecision]],
) -> tuple[list[TrialResult], list[Trial], list[tuple[Trial, int]]]:
    """Fan a decision list into (replayed results, pending trials, capped)."""
    replayed: list[TrialResult] = []
    pending: list[Trial] = []
    capped: list[tuple[Trial, int]] = []
    for trial, d in decisions:
        if d.action == "rerun":
            pending.append(trial)
            continue
        replayed.append(_replay_trial_result_from_disk(trial, storage, d.meta))
        if d.action == "capped":
            capped.append((trial, d.attempts))
    return replayed, pending, capped

def _resume_keep_if_summary(keep_if: "ResumeKeepIf | None") -> str:
    """One-line ``keep_if`` description for the resume log header."""
    if keep_if is None or keep_if.is_empty():
        return "(none)"
    parts: list[str] = []
    if keep_if.min_rounds is not None:
        parts.append(f"min_rounds={keep_if.min_rounds}")
    if keep_if.min_duration_s is not None:
        parts.append(f"min_duration_s={keep_if.min_duration_s:g}")
    if keep_if.id_matches:
        parts.append(f"id_matches=/{keep_if.id_matches}/")
    return ", ".join(parts)

def _format_resume_decision_breakdown(
    decisions: list[tuple[Trial, _ResumeDecision]],
) -> list[str]:
    """Compact category tally (re-run vs kept) for the resume log."""
    from collections import Counter

    out: list[str] = []
    rerun = Counter(d.category for _, d in decisions if d.action == "rerun")
    kept = Counter(
        d.category for _, d in decisions if d.action in ("replay", "capped")
    )
    if rerun:
        out.append(f"re-run ({sum(rerun.values())}):")
        out += [f"  {n:4d}  {cat}" for cat, n in rerun.most_common()]
    if kept:
        out.append(f"kept ({sum(kept.values())}):")
        out += [f"  {n:4d}  {cat}" for cat, n in kept.most_common()]
    return out

def _partition_resumed_trials(
    storage: RunStorage,
    trials: list[Trial],
    extra_retry_reasons: Iterable[str] = (),
    max_attempts: int = 0,
    keep_if: "ResumeKeepIf | None" = None,
    select_id_pattern: str | None = None,
) -> tuple[list[TrialResult], list[Trial]]:
    """Split ``trials`` into (replayed-from-disk, still-to-run).

    A trial is re-attempted (added to ``pending``) iff ALL of:
      - its ``trials/<id>/meta.json`` is missing/unparseable, status blank,
        OR ``termination_reason`` is in ``_DEFAULT_RESUME_RETRY_REASONS``
        ∪ ``extra_retry_reasons``;
      - AND (when ``select_id_pattern`` is set) its id matches that regex;
      - AND no ``keep_if`` threshold salvages it (enough rounds / duration /
        id match — see :class:`ResumeKeepIf`);
      - AND the total number of attempts so far is below ``max_attempts``
        (counted as live dir + every ``.before_resume_*`` archive).

    ``max_attempts <= 0`` disables the cap, which is the default unless
    project.yml explicitly sets ``resume.max_attempts``.

    Otherwise the prior outcome (success or terminal failure) replays from
    disk — only trials whose prior attempt couldn't yield a valid result
    AND are still within the attempt budget get re-run. The per-trial verdict
    and its human-readable label come from :func:`_decide_resume_trial`.
    """
    retry_reasons = _resolve_retry_reasons(extra_retry_reasons)
    decisions = _resume_decisions(
        storage, trials, retry_reasons, max_attempts, keep_if, select_id_pattern
    )
    replayed, pending, capped = _split_resume_decisions(storage, decisions)
    for trial, n in capped:
        logger.info(
            "resume: %s exhausted retry budget (%d attempts, cap=%d); "
            "replaying last failed result. Raise resume.max_attempts to "
            "try again.",
            trial.id, n, max_attempts,
        )
    return replayed, pending

def _replay_trial_result_from_disk(
    trial: Trial,
    storage: RunStorage,
    meta: dict[str, Any],
) -> TrialResult:
    """Reconstruct a :class:`TrialResult` from a previously-persisted trial dir."""
    # Use raw path — no mkdir side effect. Safe for both the orchestrator
    # path (where the dir already exists) and the dry-run analysis path
    # (where we must not touch disk).
    trial_dir = trial_path(storage.run_dir, trial.id)
    task_output = _load_resume_task_output(storage, trial)
    timing_raw = (meta or {}).get("timing") or {}
    timing = Timing(
        started_at_ms=int(timing_raw.get("started_at_ms") or 0),
        ended_at_ms=int(timing_raw.get("ended_at_ms") or 0),
        duration_ms=int(timing_raw.get("duration_ms") or 0),
    )
    proxy_log = trial_dir / PROXY_DIRNAME / PROXY_LOG_FILENAME
    try:
        trial.status = TrialStatus(str(meta.get("status") or "").strip().lower())
    except ValueError:
        trial.status = TrialStatus.COMPLETED
    return TrialResult(
        trial_id=trial.id,
        trial_index=int(meta.get("trial_index", trial.index)),
        trial_type=str(meta.get("trial_type", trial.type.value)),
        sample_id=trial.sample_id,
        output=str(task_output.get("output", "")),
        exit_code=int(meta.get("exit_code") or 0),
        timing=timing,
        error=meta.get("error") or None,
        proxy_log=proxy_log if proxy_log.exists() else None,
        terminated_by_limit=bool(meta.get("terminated_by_limit", False)),
        metadata={"resumed": True, **{k: v for k, v in meta.items() if k != "timing"}},
    )

def _load_planned_trials(run_dir: Path) -> list[dict[str, Any]]:
    """Load resume-compatible planned trial rows for ``run_dir``.

    Legacy runs wrote ``planned_trials.json`` before execution and resume must
    continue to trust that exact file when it exists. Canonical-only runs write
    ``ExperimentPlan`` instead, so the fallback projects that plan into the
    same small row shape consumed by resume and interrupted-run repair code.
    """

    planned_path = run_dir / "planned_trials.json"
    data = _load_json_file(planned_path)
    if not isinstance(data, list):
        return _canonical_plan_resume_trials(run_dir)
    return [item for item in data if isinstance(item, dict)]

def _trial_info_from_planned_trial(
    planned: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    trial_info: dict[str, Any] = {
        "trial_id": planned.get("trial_id"),
        "trial_index": planned.get("trial_index"),
        "trial_type": planned.get("trial_type"),
        "sample_id": planned.get("sample_id"),
    }
    _copy_missing_trial_meta(trial_info, meta)
    return {key: value for key, value in trial_info.items() if value is not None}

def _copy_missing_trial_meta(trial_info: dict[str, Any], meta: dict[str, Any]) -> None:
    for key in (
        "trial_id",
        "trial_index",
        "trial_type",
        "sample_id",
        "exit_code",
        "error",
        "status",
        "termination_reason",
        "termination_detail",
        "termination_source",
    ):
        if key in meta and key not in trial_info:
            trial_info[key] = meta[key]
    timing = meta.get("timing") if isinstance(meta.get("timing"), dict) else {}
    if timing and "duration_ms" in timing and "duration_ms" not in trial_info:
        trial_info["duration_ms"] = timing["duration_ms"]

def _trial_sort_key(trial_info: dict[str, Any]) -> tuple[int, str]:
    try:
        index = int(trial_info.get("trial_index"))
    except (TypeError, ValueError):
        index = 10**9
    return index, str(trial_info.get("trial_id") or "")

def _summary_from_trial_infos(
    trial_infos: list[dict[str, Any]],
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    summary = dict(fallback)
    statuses = [str(info.get("status") or "").lower() for info in trial_infos]
    # completed / failed via the shared classifier — this picks up the
    # post-completion scoring states (``scored`` / ``not_scored``) that the
    # old literal-"completed" check silently counted as zero.
    counts = count_trials(trial_infos)
    # The resume summary keeps interrupted and cancelled as distinct keys
    # (count_trials folds both into one "interrupted" bucket), so recompute
    # that split from the raw statuses.
    interrupted = sum(1 for status in statuses if status == "interrupted")
    cancelled = sum(1 for status in statuses if status == "cancelled")
    summary.update(
        {
            "total": counts.total,
            "completed": counts.completed,
            "failed": counts.failed,
            "interrupted": interrupted,
            "cancelled": cancelled,
        }
    )
    return summary

def _resume_should_preserve_canonical_snapshot(
    storage: RunStorage,
    resume_enabled: bool,
) -> bool:
    """Return whether a resume run should keep existing canonical run truth.

    ``write_initial_snapshot`` creates a fresh ``ExperimentRecord``, planned
    ``TrialRecord`` files, an ``ArtifactIndex``, the run event log, and the
    ``ResourceLedger``. That is correct for a new run, but on ``--resume`` those
    files are precisely the prior attempt evidence used to decide replay vs
    rerun. When the snapshot is already readable, the real resume path must
    preserve it before classification so it matches the read-only dry-run path
    and so cleanup/resource history remains append-only.

    A legacy run may not have canonical artifacts yet. In that case returning
    ``False`` lets the migration path create an initial canonical projection for
    the current invocation while compatibility files continue to drive resume.
    """

    if not resume_enabled:
        return False
    return ExperimentArtifactReader(storage.run_dir).try_load_snapshot() is not None
