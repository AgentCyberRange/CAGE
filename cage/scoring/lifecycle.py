"""Trial scoring lifecycle: inline, post-run, hydrate, and aggregation.

This module owns the runtime scoring slice of a run: building a
:class:`~cage.scoring.ScoringContext` for a completed trial, scoring one trial
and persisting its scores, hydrating previously-persisted scores back into
memory, the post-run scoring pass (per-trial vs post-run scorer strategies),
and building the run summary (including pass@k aggregation).

It is a pure collaborator of the conductor: it never owns the trial lifecycle,
only the "turn trial outputs into scores and summary statistics" slice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cage.contracts.sample_keys import sample_pass_index
from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.run_storage import RunStorage
from cage.contracts.trial_status import classify_trial_status
from cage.experiment.model import TrialResult, parse_trial_id
from cage.scoring import Score, ScoringContext

if TYPE_CHECKING:
    # Type-only: keeps the scoring layer from importing benchmarks at runtime
    # (Benchmark is only used in signatures here). Breaks the scoring↔benchmarks
    # import cycle — scoring stays below benchmarks.
    from cage.benchmarks import Benchmark

logger = logging.getLogger("cage.scoring.lifecycle")


def _json_safe(obj: Any) -> Any:
    """Coerce arbitrary scorer-metadata into JSON-serialisable form."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

def _scoring_context_for_result(
    result: TrialResult,
    storage: RunStorage,
) -> ScoringContext:
    """Build the scorer context for a completed trial.

    Canonical ``TrialRecord`` refs are the preferred source for scorer inputs:
    prompt, raw proxy log, final verifier evidence, live verifier evidence, and
    metadata are all resolved through ``ArtifactIndex`` by
    ``ScoringContext.from_trial_record``. The legacy ``trial_dir`` context
    remains the fallback for older runs or partial migrations that have not
    registered task-output artifacts yet.
    """

    try:
        reader = ExperimentArtifactReader(storage.run_dir)
        for trial_record in reader.load_trial_records():
            if not _trial_record_matches_result(trial_record, result):
                continue
            ctx = ScoringContext.from_trial_record(storage.run_dir, trial_record)
            if ctx is None:
                break
            sample = result.metadata.get("sample", {})
            if not ctx.sample and isinstance(sample, dict):
                ctx.sample = sample
            return ctx
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        pass

    sample = result.metadata.get("sample", {})
    return ScoringContext(
        trial_id=result.trial_id,
        trial_index=result.trial_index,
        sample=sample if isinstance(sample, dict) else {},
        output=result.output,
        exit_code=int(result.exit_code or 0),
        trial_dir=storage.trial_dir(result.trial_id),
    )

def _score_one_trial(
    result: TrialResult,
    benchmark: Benchmark,
    storage: RunStorage,
) -> bool:
    """Score one trial and persist ``scores/<benchmark.name>.json`` to disk.

    Mutates ``result.scores`` in place so downstream consumers (dashboard
    builder, pass@k computation, etc.) see the scores. Returns True on
    success, False on any error (best-effort — must never block trial
    completion).

    Idempotent: re-running on a trial that already has scores on disk
    overwrites with fresh values. Callers that want the read-only
    "hydrate from disk" behaviour should use :func:`_hydrate_scores_from_disk`
    instead.
    """
    scorer = benchmark.scorer()
    try:
        ctx = _scoring_context_for_result(result, storage)
        scores = scorer.score(ctx)
    except Exception:
        logger.exception("Scoring failed on trial %s", result.trial_id)
        return False

    # Persist metadata too — it's the only place per-trial scorer detail
    # (successful/total counts, per-host verdicts, …) survives to disk
    # for downstream consumers (web inspector, dashboard builder, ``cage
    # score`` rerun).
    result_scores = {
        name: {
            "value": s.value,
            "answer": s.answer,
            "explanation": s.explanation,
            **({"metadata": _json_safe(s.metadata)} if s.metadata else {}),
        }
        for name, s in scores.items()
    }
    storage.save_trial_scores(result.trial_id, benchmark.name, result_scores)
    for name, score in scores.items():
        result.scores[name] = score

    logger.info(
        "trial_scored",
        extra={
            "trial_id": result.trial_id,
            "scores": {name: s.value for name, s in scores.items()},
        },
    )
    return True

def _trial_result_task_and_pass(result: TrialResult) -> tuple[str, int]:
    """Return the legacy task id/pass pair represented by a ``TrialResult``."""

    raw_id = str(result.trial_id or result.sample_id or "trial")
    task_id, pass_index = parse_trial_id(raw_id)
    sample = result.metadata.get("sample") if isinstance(result.metadata, dict) else None
    structured = sample_pass_index(sample)
    return task_id, structured if structured is not None else pass_index

def _trial_record_matches_result(record: Any, result: TrialResult) -> bool:
    """Whether a canonical ``TrialRecord`` describes a legacy ``TrialResult``."""

    if str(getattr(record, "trial_id", "")) == str(result.trial_id):
        return True
    task_id, pass_index = _trial_result_task_and_pass(result)
    record_task_id = str(getattr(record, "task_id", ""))
    if record_task_id == task_id and int(getattr(record, "pass_index", 1)) == pass_index:
        return True
    return (
        bool(result.sample_id)
        and record_task_id == str(result.sample_id)
        and int(getattr(record, "pass_index", 1)) == pass_index
    )

def _indexed_score_path_for_result(
    result: TrialResult,
    benchmark: Benchmark,
    storage: RunStorage,
) -> Path | None:
    """Resolve a trial score through canonical record refs and ``ArtifactIndex``.

    The in-memory ``TrialResult`` still carries legacy ids. Canonical
    ``TrialRecord`` ids include subject/task/pass, so this bridge matches by the
    legacy task/pass identity and only opens score files that were registered in
    the run's artifact index. Old runs without canonical artifacts fall back to
    the legacy score path in ``_hydrate_scores_from_disk``.
    """

    benchmark_name = str(getattr(benchmark, "name", "") or "")
    reader = ExperimentArtifactReader(storage.run_dir)
    try:
        trial_records = reader.load_trial_records()
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    for record in trial_records:
        if not _trial_record_matches_result(record, result):
            continue
        score_ref = getattr(record.scoring, "score_ref", None)
        scoring_id = str(getattr(record, "scoring_id", "") or "")
        if score_ref and (not scoring_id or scoring_id == benchmark_name):
            try:
                artifact = reader.find_artifact(path=score_ref, kind="trial_score")
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                artifact = None
            if artifact is not None:
                try:
                    return reader.resolve_artifact_path(artifact)
                except (KeyError, ValueError):
                    return None
        for artifact in getattr(record, "artifacts", ()):
            if getattr(artifact, "kind", "") != "trial_score":
                continue
            if benchmark_name and not (
                str(getattr(artifact, "artifact_id", "")).endswith(
                    f".score.{benchmark_name}"
                )
                or Path(str(getattr(artifact, "path", ""))).name
                == f"{benchmark_name}.json"
            ):
                continue
            try:
                indexed = reader.find_artifact(path=artifact.path, kind="trial_score")
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                indexed = None
            if indexed is None:
                continue
            try:
                return reader.resolve_artifact_path(indexed)
            except (KeyError, ValueError):
                return None
    return None

def _hydrate_scores_from_disk(
    result: TrialResult, benchmark: Benchmark, storage: RunStorage,
) -> bool:
    """Re-load a trial's previously-persisted score into ``result.scores``.

    Used by :func:`_score_trials` to make the post-run pass idempotent for
    ``strategy="per_trial"`` scorers — the score file is already on disk
    from inline scoring; we just need to put the data back in memory.

    Returns True if at least one score was loaded.
    """
    score_path = _indexed_score_path_for_result(result, benchmark, storage)
    if score_path is None:
        score_path = (
            storage.trial_dir(result.trial_id) / "scores" / f"{benchmark.name}.json"
        )
    if not score_path.is_file():
        return False
    try:
        raw = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict) or not raw:
        return False
    loaded_any = False
    for name, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        result.scores[name] = Score(
            value=float(payload.get("value") or 0.0),
            answer=str(payload.get("answer") or ""),
            explanation=str(payload.get("explanation") or ""),
            metadata=payload.get("metadata") or {},
        )
        loaded_any = True
    return loaded_any

def _score_trials(
    results: list[TrialResult],
    benchmark: Benchmark,
    storage: RunStorage,
) -> None:
    """Post-run scoring pass — runs once per agent at the end of its trials.

    Behaviour depends on the scorer's declared :attr:`Scorer.strategy`:

      * ``"per_trial"`` (default) — most trials are already scored inline
        from :func:`execute_trial`. We hydrate ``result.scores`` from
        each trial's on-disk ``scores/<name>.json`` so downstream
        consumers see them, and only re-score trials whose inline pass
        failed or never ran (resume of an old run, exception during
        inline scoring).
      * ``"post_run"`` — inline scoring is skipped; this pass scores
        every trial here, with the full :class:`list[TrialResult]`
        available so cross-trial scorers can run.
    """
    strategy = getattr(benchmark.scorer(), "strategy", "per_trial")
    for result in results:
        if result.error:
            continue
        if strategy == "per_trial":
            if _hydrate_scores_from_disk(result, benchmark, storage):
                continue
            # Inline pass didn't run (or wrote nothing) — score now as
            # the recovery path.
        _score_one_trial(result, benchmark, storage)

def _build_summary(results: list[TrialResult]) -> dict[str, Any]:
    """Build summary statistics from trial results.

    When trials repeat the same ``sample_id`` (pass@k runs), also emit a
    ``pass_at_k`` block keyed by scorer name: pass@1 is the micro-average
    over all attempts, pass@k counts a sample as solved if *any* attempt
    cleared the score threshold (>= 1.0).

    When samples carry a group key (``sample["variant"]``, ``test_type``,
    or ``metadata["group"]``), the ``pass_at_k.<scorer>.groups`` block
    additionally breaks pass@1 / pass@k down per group — used by any
    benchmark that surfaces a variant axis (e.g. zero_day / one_day).
    """
    total = len(results)
    completed = failed = 0
    for r in results:
        metadata = r.metadata if isinstance(r.metadata, dict) else {}
        bucket = classify_trial_status(
            status=metadata.get("status"),
            error=r.error,
            termination_reason=metadata.get("termination_reason"),
            default="completed",
        )
        if bucket == "completed":
            completed += 1
        elif bucket == "failed":
            failed += 1

    score_sums: dict[str, float] = {}
    score_counts: dict[str, int] = {}
    per_sample_max: dict[str, dict[str, float]] = {}  # scorer → sample_id → best score
    per_sample_count: dict[str, int] = {}             # sample_id → attempts seen
    sample_group: dict[str, str] = {}                 # sample_id → group key
    group_score_sums: dict[tuple[str, str], float] = {}   # (scorer, group) → sum
    group_score_counts: dict[tuple[str, str], int] = {}   # (scorer, group) → count
    for r in results:
        per_sample_count[r.sample_id] = per_sample_count.get(r.sample_id, 0) + 1
        group = _sample_group_key(r)
        if group:
            sample_group.setdefault(r.sample_id, group)
        for name, score in r.scores.items():
            score_sums[name] = score_sums.get(name, 0.0) + score.value
            score_counts[name] = score_counts.get(name, 0) + 1
            bucket = per_sample_max.setdefault(name, {})
            bucket[r.sample_id] = max(bucket.get(r.sample_id, float("-inf")), score.value)
            if group:
                key = (name, group)
                group_score_sums[key] = group_score_sums.get(key, 0.0) + score.value
                group_score_counts[key] = group_score_counts.get(key, 0) + 1

    mean_scores = {
        name: round(score_sums[name] / score_counts[name], 4)
        for name in score_sums
        if score_counts[name] > 0
    }

    summary: dict[str, Any] = {
        "total": total,
        "completed": completed,
        "failed": failed,
        "mean_scores": mean_scores,
    }

    k = max(per_sample_count.values(), default=0)
    if k > 1 and per_sample_max:
        threshold = 1.0
        pass_at_k: dict[str, dict[str, Any]] = {}
        for scorer, sample_max in per_sample_max.items():
            n = len(sample_max)
            if n == 0:
                continue
            any_pass = sum(1 for v in sample_max.values() if v >= threshold) / n
            block: dict[str, Any] = {
                "k": k,
                "n_samples": n,
                "pass@1": round(mean_scores.get(scorer, 0.0), 4),
                f"pass@{k}": round(any_pass, 4),
            }
            if sample_group:
                groups: dict[str, dict[str, Any]] = {}
                for sid, max_v in sample_max.items():
                    grp = sample_group.get(sid)
                    if not grp:
                        continue
                    g = groups.setdefault(grp, {"_n": 0, "_any": 0})
                    g["_n"] += 1
                    if max_v >= threshold:
                        g["_any"] += 1
                if groups:
                    block["groups"] = {
                        grp: {
                            "n_samples": g["_n"],
                            "pass@1": round(
                                group_score_sums.get((scorer, grp), 0.0)
                                / max(group_score_counts.get((scorer, grp), 1), 1),
                                4,
                            ),
                            f"pass@{k}": round(g["_any"] / g["_n"], 4) if g["_n"] else 0.0,
                        }
                        for grp, g in sorted(groups.items())
                    }
            pass_at_k[scorer] = block
        summary["pass_at_k"] = pass_at_k

    return summary

def _sample_group_key(result: TrialResult) -> str:
    """Best-effort grouping key for pass@k breakdown.

    Looks at the sample dict that the orchestrator stashes under
    ``metadata["sample"]``. We accept three field names in order of
    preference so benchmarks can opt in without subclassing:

      1. ``variant``        — the conventional variant axis (e.g. zero_day / one_day)
      2. ``test_type``      — alias for the same axis
      3. ``metadata.group`` — escape hatch for any other benchmark
    """
    sample = result.metadata.get("sample") if isinstance(result.metadata, dict) else None
    if not isinstance(sample, dict):
        return ""
    for key in ("variant", "test_type"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = sample.get("metadata")
    if isinstance(meta, dict):
        value = meta.get("group")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
