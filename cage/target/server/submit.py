"""Serve-only submission scoring — close the PULL benchmark loop.

An external agent that launched an isolated instance (``GET /launch``), attacked
it, and produced a ``final_answer/`` output submits that output here. The server
then, without ever naming a benchmark:

  1. looks up the running instance (target reachability) by ``run_id``,
  2. discovers the challenge's scorer metadata + benchmark package,
  3. reconstructs the benchmark scorer's ``sample`` from the running-instance
     registry + the discovered challenge metadata,
  4. ``scorer.gather()`` — gathers LIVE evidence against the still-running
     target (``docker cp`` markers for post-exploitation, evaluator POST for
     web) — so this must run BEFORE the instance is closed,
  5. ``scorer.score()`` — scores offline (verifier signals always; the optional
     ``LLM_judge`` signal only when a judge model is injected),
  6. persists an inspectable run under ``.cage_runs/`` so the existing cage
     inspector renders the verdict. The serve console (launch/submit) and the
     inspector (verdict) are the two halves of the same benchmark surface,
     rooted at the same ``.cage_runs`` tree.

Layer-1 clean: the benchmark is *discovered* by its configured source path and
loaded via :func:`cage.benchmarks.loader.load_benchmark_from_module` — the
framework never names a benchmark. The server hosts targets and orchestrates a
benchmark-supplied scorer; it does not know what any benchmark is.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from cage.benchmarks.loader import load_benchmark_from_module
from cage.contracts import RUNTIME_STATE_KEY
from cage.scoring import ScoringContext
from cage.scoring.scorer import GatherRuntime

logger = logging.getLogger(__name__)


class SubmissionError(Exception):
    """A submission could not be scored (unknown run_id, missing benchmark, …)."""


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a ServiceInfo object OR a plain dict (registry stores both)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def resolve_benchmark_module(challenge: dict[str, Any]) -> Path:
    """Walk up from the challenge dir to the nearest ``benchmark.py``.

    A benchmark laid out as ``examples/<name>/{benchmark.py, datasets/…}``
    resolves generically — no benchmark name is hardcoded. The challenge's
    ``full_path`` points at the challenge dir inside the dataset submodule, so
    the first ``benchmark.py`` above it is the owning benchmark's module.
    """
    full_path = str(challenge.get("full_path") or "")
    if not full_path:
        raise SubmissionError("challenge carries no full_path; cannot locate benchmark.py")
    here = Path(full_path).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "benchmark.py"
        if candidate.is_file():
            return candidate
    raise SubmissionError(f"no benchmark.py found above {full_path}")


def reconstruct_sample(
    run_id: str,
    instance: dict[str, Any],
    challenge: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the scorer's ``sample`` from the live instance + challenge metadata.

    Mirrors what the cage-run orchestrator injects before scoring
    (``inject_ctf_info`` + ``_apply_runtime_record``): ``target_info`` keyed by
    service name (the scorer reads ``target_info["evaluator"]`` for the web
    evaluator sidecar), ``runtime_state.project_name`` (the scorer resolves the
    docker compose project for ``docker cp`` markers), and ``metadata`` (the
    vulnerability / marker-stage list from the challenge's ``source_fields``).
    """
    source_fields = challenge.get("source_fields") or {}

    target_info: dict[str, Any] = {}
    for svc in instance.get("services") or []:
        name = _attr(svc, "service_name")
        if not name:
            continue
        ext_host = _attr(svc, "external_host")
        ext_port = _attr(svc, "external_port")
        target_info[str(name)] = {
            "service_name": str(name),
            "alias": _attr(svc, "alias") or str(name),
            "inner_ip": _attr(svc, "inner_ip"),
            "inner_port": _attr(svc, "inner_port"),
            "external_host": ext_host,
            "external_port": ext_port,
            # convenience aliases some verifiers read
            "host": ext_host,
            "port": ext_port,
        }

    metadata: dict[str, Any] = {
        "verifier_kind": str(
            source_fields.get("verifier_kind")
            or challenge.get("verifier_kind")
            or ""
        ),
        "vulnerabilities": list(source_fields.get("vulnerabilities") or []),
        "marker_stages": list(source_fields.get("marker_stages") or []),
        "marker_services": list(source_fields.get("marker_services") or []),
        "marker_user_path": str(source_fields.get("marker_user_path") or ""),
        "marker_root_path": str(source_fields.get("marker_root_path") or ""),
    }

    return {
        "id": challenge.get("id") or instance.get("chal_id") or run_id,
        "challenge_name": challenge.get("name") or challenge.get("challenge_name"),
        "target_info": target_info,
        RUNTIME_STATE_KEY: {
            "project_name": str(instance.get("project_name") or ""),
            "run_id": run_id,
            "scoring": instance.get("scoring") or {},
        },
        "network_name": str(instance.get("network_name") or ""),
        "metadata": metadata,
    }


def _prompt_level_to_hint(prompt_level: str) -> int:
    """Map an ``l0`` / ``l1`` / ``l2`` prompt level to a hint tier (0–2)."""
    text = str(prompt_level or "").strip().lower().lstrip("l")
    try:
        return max(0, min(2, int(text)))
    except (TypeError, ValueError):
        return 0


# Placeholder tokens the un-rendered ``task_prompt_template`` shows in place of
# the concrete, per-instance target addresses that ``task_prompt`` fills in.
_TEMPLATE_TARGET_TOKEN = "{{APPLICATION_TARGETS}}"
_TEMPLATE_ENTRY_TOKEN = "{{ENTRY_HOST}}"


def _entry_host_urls(sample: dict[str, Any]) -> list[str]:
    """The entry-host URLs ``build_prompt`` renders for a post-exploitation sample.

    Mirrors the benchmark's own ``target_info[service].inner_ip:inner_port`` →
    ``http://ip:port/`` mapping so the template masker can find those exact
    address strings in the rendered prompt and swap them for a placeholder.
    """
    target_info = sample.get("target_info") or {}
    entry_services = (sample.get("metadata") or {}).get("entry_services") or []
    urls: list[str] = []
    for service in entry_services:
        info = target_info.get(service) or {}
        if not isinstance(info, dict):
            continue
        ip = info.get("inner_ip")
        port = info.get("inner_port") or info.get("internal_port")
        if ip and port:
            urls.append(f"http://{ip}:{int(port)}/")
    return urls


def _mask_targets(prompt: str, sample: dict[str, Any]) -> str:
    """Turn a filled ``task_prompt`` into a ``task_prompt_template``.

    Replaces the concrete target address(es) the benchmark injected (the web
    ``application_targets`` and each post-exploitation entry-host URL) with named
    placeholder tokens, so the template is structurally identical to the filled
    prompt but with obvious fill-in slots for the target.
    """
    agent_input = sample.get("agent_input") or {}
    app = str(agent_input.get("application_targets") or "").strip()
    if app:
        prompt = prompt.replace(app, _TEMPLATE_TARGET_TOKEN)
    for url in _entry_host_urls(sample):
        prompt = prompt.replace(url, _TEMPLATE_ENTRY_TOKEN)
    return prompt


def render_task_prompt(
    run_id: str,
    instance: dict[str, Any],
    challenge: dict[str, Any],
    *,
    prompt_level: str = "l0",
) -> tuple[str, str]:
    """Render the agent-facing task briefing for a launched instance.

    Returns ``(task_prompt, task_prompt_template)``:

    - ``task_prompt`` — the ready-to-use briefing an external agent completes
      the task from, with the live target address(es) filled in. The serve-mode
      counterpart of what ``cage run`` injects as ``{task_instruction}``,
      produced by the benchmark's own ``build_prompt`` so it matches a
      CAGE-managed agent's briefing (task framing, target address, the
      ``final_answer`` output contract, any operator-selected hint tier).
    - ``task_prompt_template`` — the same briefing with the target address(es)
      shown as placeholder tokens, i.e. the un-filled template.

    Reconstructs the sample from the running instance (so post-exploitation entry
    hosts resolve to the instance's live inner IPs) plus the challenge's
    ``agent_input`` / ``task_profile`` from ``source_fields``.

    ``prompt_level`` (``l0``/``l1``/``l2``) is chosen by the OPERATOR at serve
    time, never by the agent — an agent cannot escalate itself to a higher-hint
    tier. Best-effort: returns ``("", "")`` if the benchmark exposes no
    ``build_prompt`` or a field is missing, so a render failure never blocks the
    loop.
    """
    try:
        benchmark = load_benchmark_from_module(resolve_benchmark_module(challenge))
    except Exception as exc:  # noqa: BLE001
        logger.warning("serve task-prompt: benchmark load failed for %s: %s", run_id, exc)
        return "", ""
    build_prompt = getattr(benchmark, "build_prompt", None)
    if not callable(build_prompt):
        return "", ""

    source_fields = challenge.get("source_fields") or {}
    sample = reconstruct_sample(run_id, instance, challenge)
    # Agent-facing inputs the prompt templates read (application_targets for web;
    # marker paths / hint tiers for post-exploitation). The operator's prompt
    # level sets the hint tier — assigned outright, never taken from the agent.
    agent_input = dict(source_fields.get("agent_input") or {})
    agent_input["hint_level"] = _prompt_level_to_hint(prompt_level)
    sample["agent_input"] = agent_input
    sample["task_profile"] = (
        challenge.get("task_profile") or source_fields.get("task_profile") or ""
    )
    # build_prompt's post-exploitation branch turns metadata.entry_services into
    # entry-host URLs from the live inner IPs already in target_info.
    metadata = dict(sample.get("metadata") or {})
    metadata["entry_services"] = list(
        source_fields.get("application_service_keys")
        or source_fields.get("entry_services")
        or []
    )
    sample["metadata"] = metadata
    try:
        task_prompt = str(build_prompt(sample) or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("serve task-prompt render failed for %s: %s", run_id, exc)
        return "", ""
    return task_prompt, _mask_targets(task_prompt, sample)


def _score_to_dict(score: Any) -> dict[str, Any]:
    """Serialize a benchmark ``Score`` (dataclass or duck-typed) to plain JSON."""
    if dataclasses.is_dataclass(score) and not isinstance(score, type):
        return dataclasses.asdict(score)
    return {
        "value": getattr(score, "value", None),
        "answer": getattr(score, "answer", ""),
        "explanation": getattr(score, "explanation", ""),
        "metadata": getattr(score, "metadata", {}) or {},
    }


def _persist_run(
    runs_root: Path,
    subject: str,
    run_id: str,
    trial_id: str,
    *,
    sample: dict[str, Any],
    output: str,
    evidence: str,
) -> tuple[Path, Path]:
    """Write the ``.cage_runs`` trial layout the scorer + inspector read.

    Returns ``(run_dir, trial_dir)``. The layout is the same
    ``.cage_runs/<subject>/<run_id>/trials/<trial_id>/`` shape a cage-run trial
    uses, so the cage inspector's legacy discovery lists it and opens the trial
    (the console launches, the inspector shows the verdict — one ``.cage_runs``
    tree). Writes ``task_output.json`` (output + sample),
    ``runtime/check_done_output.txt`` (the gathered evidence, read by
    :pyattr:`ScoringContext.check_done_output`), and ``meta.json`` (trial status
    the inspector's trial page reads). The scores file is written by the caller
    after scoring.
    """
    run_dir = runs_root / subject / run_id
    trial_dir = run_dir / "trials" / trial_id
    (trial_dir / "runtime").mkdir(parents=True, exist_ok=True)
    (trial_dir / "task_output.json").write_text(
        json.dumps({"output": output, "sample": sample}, ensure_ascii=False),
        encoding="utf-8",
    )
    (trial_dir / "runtime" / "check_done_output.txt").write_text(
        evidence or "", encoding="utf-8"
    )
    (trial_dir / "meta.json").write_text(
        json.dumps({"trial_id": trial_id, "status": "completed"}, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir, trial_dir


_EXPERIMENT_LOCKS: dict[str, threading.Lock] = {}
_EXPERIMENT_LOCKS_GUARD = threading.Lock()


def _experiment_lock(run_dir: Path) -> threading.Lock:
    """A process-local lock per experiment run dir (serializes concurrent appends)."""
    key = str(run_dir)
    with _EXPERIMENT_LOCKS_GUARD:
        lock = _EXPERIMENT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _EXPERIMENT_LOCKS[key] = lock
    return lock


def _append_trial_to_experiment(
    run_dir: Path,
    trial_dir: Path,
    *,
    experiment_run_id: str,
    trial_id: str,
    chal_id: str,
    agent_id: str,
    benchmark: str,
    scores_out: dict[str, Any],
    created_at_iso: str,
) -> None:
    """Append this scored submission as one trial to the agent's experiment.

    A serve-only session is ONE experiment per external agent (``subject`` =
    ``serve__<agent_id>``, one run): every ``POST /submit`` adds a trial. The
    inspector's detail view resolves scores only through the canonical
    ``experiment_record`` + ``artifact_index`` + ``TrialRecord`` (never an
    unindexed ``scores/*.json`` path), so this maintains that canonical set,
    MERGING the new trial into the existing plan / record / index rather than
    overwriting — the run accumulates the agent's challenges over time, agent
    runtime and trajectory externalized (those panels stay empty). Serialized by
    a per-run lock; best-effort (the caller ignores failures so a records hiccup
    never sinks a real verdict).
    """
    import cage.experiment.model as M
    from cage.artifacts.reader import ExperimentArtifactReader

    trial_rel = trial_dir.relative_to(run_dir).as_posix()  # trials/<trial_id>
    new_artifacts = tuple(
        M.ArtifactRef(
            artifact_id=f"trial.{trial_id}.score.{name.replace('/', '_')}",
            path=f"{trial_rel}/scores/{name.replace('/', '_')}.json",
            kind="trial_score",
        )
        for name in scores_out
    )
    primary_score_ref = new_artifacts[0].path if new_artifacts else ""

    with _experiment_lock(run_dir):
        reader = ExperimentArtifactReader(run_dir)

        # spec: create once, reuse thereafter
        spec_path = run_dir / "experiment_spec.json"
        if spec_path.is_file():
            spec = reader.load_spec()
        else:
            spec = M.experiment_spec_from_project_mapping(
                {"project": {"name": f"serve__{agent_id}"}},
                project_file=run_dir / "project.yml",
                base_dir=run_dir,
            )
            spec_path.write_text(M.experiment_spec_to_json(spec), encoding="utf-8")

        # plan trials: merge existing + this one (dedupe by trial_id, this wins)
        trial_plans_by_id: dict[str, Any] = {}
        try:
            for tp in reader.load_plan().trials:
                trial_plans_by_id[tp.trial_id] = tp
        except Exception:  # noqa: BLE001 — first submission, no plan yet
            pass
        trial_plans_by_id[trial_id] = M.TrialPlan(
            trial_id=trial_id, subject_id="serve", task_id=chal_id, pass_index=0
        )
        plan = M.ExperimentPlan(
            schema_version="experiment_plan.v1",
            plan_id=f"plan_{experiment_run_id}",
            source=M.PlanSource(project_file=str(run_dir / "project.yml"), benchmark_id=benchmark),
            subjects=(M.SubjectPlan(subject_id="serve", agent=agent_id, kind="serve", profile="", model=""),),
            tasks=(),
            trials=tuple(trial_plans_by_id.values()),
            controls=spec.protocol,
        )

        # artifact index: merge existing + new (dedupe by artifact_id)
        artifacts_by_id: dict[str, Any] = {}
        try:
            for art in reader.load_artifact_index().artifacts:
                artifacts_by_id[art.artifact_id] = art
        except Exception:  # noqa: BLE001
            pass
        for art in new_artifacts:
            artifacts_by_id[art.artifact_id] = art

        record = M.create_experiment_record(
            plan,
            run_id=experiment_run_id,
            created_at=created_at_iso,
            status="completed",
            trial_record_refs={t: f"trials/{t}/record.json" for t in trial_plans_by_id},
        )
        trial_record = M.TrialRecord(
            schema_version="trial_record.v1",
            trial_id=trial_id,
            run_id=experiment_run_id,
            plan_ref="experiment_plan.json",
            status="completed",
            status_reason="",
            subject_id="serve",
            task_id=chal_id,
            pass_index=0,
            artifacts=new_artifacts,
            scoring=M.TrialScoringRecord(status="scored", score_ref=primary_score_ref),
        )

        (run_dir / "experiment_plan.json").write_text(M.experiment_plan_to_json(plan), encoding="utf-8")
        (run_dir / "experiment_record.json").write_text(M.experiment_record_to_json(record), encoding="utf-8")
        (run_dir / "artifact_index.json").write_text(
            M.artifact_index_to_json(M.ArtifactIndex(artifacts=tuple(artifacts_by_id.values()))),
            encoding="utf-8",
        )
        (trial_dir / "record.json").write_text(M.trial_record_to_json(trial_record), encoding="utf-8")


def score_submission(
    run_id: str,
    agent_output_dir: str | Path,
    instance: dict[str, Any],
    challenge: dict[str, Any],
    *,
    agent_id: str = "local",
    judge: dict[str, Any] | None = None,
    runs_root: str | Path | None = None,
    output: str = "",
    now: float | None = None,
    uuid_hex: str | None = None,
) -> dict[str, Any]:
    """Score one serve-only submission against a still-running instance.

    ``run_id`` is the launched *instance* id being scored. ``agent_output_dir``
    is the unpacked submission (its ``final_answer/`` dir holds the per-vuln
    reports for web challenges; ignored for marker-only post-exploitation
    challenges, which read the live target). ``instance`` is the running-instance
    registry value; ``challenge`` is the discovered ``NormalizedChallenge`` dict.
    ``agent_id`` identifies the external agent — the serve session is ONE
    experiment per agent, and this submission becomes one trial appended to it.
    ``judge`` (optional) is the judge-model config injected into the benchmark
    for the ``LLM_judge`` signal — omit it and ``LLM_judge`` vulns report
    "no judge model configured" (verifier-only).

    Returns a verdict dict: ``{run_id, chal_id, benchmark_module, scores,
    evidence, run_dir}``. Persists an inspectable trial into the agent's
    experiment run under ``.cage_runs``.
    """
    agent_output_dir = Path(agent_output_dir)
    bench_module = resolve_benchmark_module(challenge)
    kwargs = {"judge": judge} if judge else None
    benchmark = load_benchmark_from_module(bench_module, kwargs=kwargs)
    scorer = benchmark.scorer()

    sample = reconstruct_sample(run_id, instance, challenge)

    # (4) LIVE gather — must run while the target is still up.
    runtime = GatherRuntime(
        sample=sample,
        agent_output_dir=agent_output_dir,
        container=None,  # serve-only: the agent container is never touched
    )
    evidence = scorer.gather(runtime)

    # (6) persist this submission as one trial in the AGENT's experiment run.
    # Layout ``.cage_runs/serve__<agent_id>/serve/trials/<trial_id>/`` — one
    # experiment per external agent (the depth the inspector scans; subject is a
    # single path segment mirroring a cage-run agent label). Every submission
    # appends a trial, so the run accumulates the agent's challenges over time.
    chal_id = str(instance.get("chal_id") or challenge.get("id") or run_id)
    benchmark_name = str(challenge.get("benchmark") or challenge.get("benchmark_name") or chal_id)
    subject = f"serve__{agent_id}"
    experiment_run_id = "serve"
    # trial id is unique per submission (challenge + instance + nonce) and sorts
    # by challenge in the inspector.
    hexpart = uuid_hex or uuid.uuid4().hex[:8]
    trial_id = f"{chal_id}__{run_id}_{hexpart}"
    runs_root = Path(runs_root) if runs_root is not None else (
        bench_module.parent / ".cage_runs"
    )
    run_dir, trial_dir = _persist_run(
        runs_root, subject, experiment_run_id, trial_id,
        sample=sample, output=output, evidence=evidence,
    )

    # (5) score offline
    ctx = ScoringContext(
        trial_id=trial_id,
        trial_index=0,
        sample=sample,
        output=output,
        exit_code=0,
        trial_dir=trial_dir,
        run_dir=run_dir,
        canonical_trial_id=trial_id,
    )
    scores = scorer.score(ctx)
    scores_out = {name: _score_to_dict(score) for name, score in (scores or {}).items()}

    scores_dir = trial_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in scores_out.items():
        safe = name.replace("/", "_")
        # Nest under the scorer name — the shape the inspector reads
        # (``{"<scorer>": {value, answer, explanation, metadata}}``).
        (scores_dir / f"{safe}.json").write_text(
            json.dumps({name: payload}, ensure_ascii=False), encoding="utf-8"
        )

    # Append this trial to the agent's experiment so `cage inspect` renders it.
    # Best-effort — a records failure must never sink the verdict just computed.
    stamp = time.gmtime(now if now is not None else time.time())
    try:
        _append_trial_to_experiment(
            run_dir, trial_dir,
            experiment_run_id=experiment_run_id,
            trial_id=trial_id, chal_id=chal_id, agent_id=agent_id,
            benchmark=benchmark_name,
            scores_out=scores_out,
            created_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", stamp),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("serve submission canonical records skipped: %s", exc)

    return {
        "run_id": run_id,
        "chal_id": chal_id,
        "benchmark_module": str(bench_module),
        "trial_id": trial_id,
        "scores": scores_out,
        "evidence": evidence,
        "run_dir": str(run_dir),
        "scored_at": now if now is not None else time.time(),
    }
