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
  6. writes a **serve-native submission record** under ``.cage_serve/`` (see
     :mod:`cage.target.server.serve_log`) — a target-centric log carrying the
     trial's information minus the agent half, with the agent findings persisted
     so an offline re-judge has its inputs.

Serve mode has no Cage-launched agent, so it does NOT reuse the agent-centric
``cage run`` experiment record; it has its own log. Layer-1 clean: the benchmark
is *discovered* by its configured source path and loaded via
:func:`cage.benchmarks.loader.load_benchmark_from_module` — the framework never
names a benchmark. The server hosts targets and orchestrates a benchmark-supplied
scorer; it does not know what any benchmark is.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from cage.benchmarks.loader import load_benchmark_from_module
from cage.contracts import RUNTIME_STATE_KEY
from cage.scoring import ScoringContext
from cage.scoring.scorer import GatherRuntime
from cage.target.server import serve_log

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


_JUDGE_MODEL_KEYS = ("model_id", "id", "model")


def _judge_model_ids(judge: dict[str, Any] | None) -> tuple[str, ...]:
    """Extract the judge model id(s) from a judge config for the pass signature.

    Accepts the legacy single ``model_id``/``id``/``model`` form OR a ``models:``
    list of strings/dicts (an ensemble). Returns ``()`` when no judge is
    configured — the pass is then a verifier-only pass.
    """
    if not judge:
        return ()
    models = judge.get("models")
    ids: list[str] = []
    if isinstance(models, list):
        for m in models:
            if isinstance(m, str) and m:
                ids.append(m)
            elif isinstance(m, dict):
                for k in _JUDGE_MODEL_KEYS:
                    if m.get(k):
                        ids.append(str(m[k]))
                        break
    else:
        for k in _JUDGE_MODEL_KEYS:
            if judge.get(k):
                ids.append(str(judge[k]))
                break
    return tuple(ids)


def score_submission(
    run_id: str,
    agent_output_dir: str | Path,
    instance: dict[str, Any],
    challenge: dict[str, Any],
    *,
    agent_id: str = "local",
    label: str = "",
    judge: dict[str, Any] | None = None,
    serve_root: str | Path | None = None,
    output: str = "",
    now: float | None = None,
    uuid_hex: str | None = None,
) -> dict[str, Any]:
    """Score one serve-only submission against a still-running instance.

    ``run_id`` is the launched *instance* id being scored (the target-launch id
    a verdict is meaningful relative to). ``agent_output_dir`` is the unpacked
    submission (its ``final_answer/`` dir holds the per-vuln reports for web
    challenges; ignored for marker-only post-exploitation challenges, which read
    the live target). ``instance`` is the running-instance registry value;
    ``challenge`` is the discovered ``NormalizedChallenge`` dict. ``agent_id``
    identifies the external client (self-declared). ``label`` is an optional,
    caller-supplied human name for easy lookup — it prefixes the record's
    directory and is stored on the record, but is NOT an identity (the machine
    primary key is the submission id). ``judge`` (optional) is the judge-model
    config injected into the benchmark for the ``LLM_judge`` signal.

    Writes a serve-native submission record under ``.cage_serve/<client_id>/
    <label?>__<submission_id>/`` (see :mod:`cage.target.server.serve_log`) with
    the agent findings persisted, then returns a verdict dict.
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

    # Identities. The submission id embeds the challenge + target-launch id so it
    # is traceable to the exact boot it scored; the label is a human alias only.
    chal_id = str(instance.get("chal_id") or challenge.get("id") or run_id)
    client_id = agent_id or "local"
    hexpart = uuid_hex or uuid.uuid4().hex[:8]
    submission_id = f"{chal_id}__{run_id}_{hexpart}"
    root = (
        Path(serve_root)
        if serve_root is not None
        else bench_module.parent / serve_log.SERVE_ROOT_DIRNAME
    )
    sub_dir = serve_log.submission_dir(root, client_id, submission_id, label)
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Persist the scorer-visible inputs (findings included) BEFORE score(), so the
    # LLM judge reads workspace/final_answer and an offline re-judge later has its
    # inputs. Fixes the serve bug where the upload was deleted before the judge
    # could read it.
    submission_meta = serve_log.persist_inputs(
        sub_dir,
        sample=sample,
        output=output,
        evidence=evidence,
        agent_output_dir=agent_output_dir,
    )

    # (5) score offline — trial_dir is the serve submission dir, so the scorer's
    # judge reads workspace/final_answer and writes scores/judge_io.jsonl here.
    ctx = ScoringContext(
        trial_id=submission_id,
        trial_index=0,
        sample=sample,
        output=output,
        exit_code=0,
        trial_dir=sub_dir,
        run_dir=root,
        canonical_trial_id=submission_id,
    )
    scores = scorer.score(ctx)
    scores_out = {name: _score_to_dict(score) for name, score in (scores or {}).items()}

    scores_dir = sub_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    score_refs: dict[str, str] = {}
    for name, payload in scores_out.items():
        safe = name.replace("/", "_")
        rel = f"scores/{safe}.json"
        # Nested under the scorer name — ``{"<scorer>": {value, answer, …}}``.
        (sub_dir / rel).write_text(
            json.dumps({name: payload}, ensure_ascii=False), encoding="utf-8"
        )
        score_refs[name] = rel
    canonical_score_ref = next(iter(score_refs.values()), "")

    # Timestamps + provenance.
    ts = now if now is not None else time.time()
    score_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    grader = serve_log.build_grader(challenge, sample)
    inputs_digest = serve_log.compute_inputs_digest(
        evidence, submission_meta.findings_digest
    )
    target_launch = serve_log.build_target_launch(
        launch_id=run_id, challenge_id=chal_id, instance=instance, sample=sample,
    )
    prompt_level = str(instance.get("prompt_level") or "")
    try:
        prompt_text, _ = render_task_prompt(
            run_id, instance, challenge, prompt_level=prompt_level or "l0"
        )
    except Exception:  # noqa: BLE001 — provenance nicety, never sinks a verdict
        prompt_text = ""
    prompt_served = serve_log.PromptServed(
        level=prompt_level,
        digest=serve_log.prompt_digest(prompt_text) if prompt_text else "",
    )

    judge_models = _judge_model_ids(judge)
    pass_id = serve_log.make_pass_id(judge_models, score_time)
    provenance = serve_log.PassProvenance(
        mode="serve",
        challenge_id=chal_id,
        target_launch_id=run_id,
        project_name=target_launch.project_name,
        launch_time=target_launch.launch_time,
        client_id=client_id,
        prompt_level=prompt_level,
        prompt_digest=prompt_served.digest,
        inputs_digest=inputs_digest,
        grader=grader,
        judge_models=judge_models,
        score_time=score_time,
    )
    score_pass = serve_log.ScorePass(
        pass_id=pass_id,
        judge_models=judge_models,
        score_time=score_time,
        score_ref=canonical_score_ref,
        scores=scores_out,
        provenance=provenance,
    )
    record = serve_log.ServeSubmissionRecord(
        submission_id=submission_id,
        client_id=client_id,
        challenge_id=chal_id,
        benchmark_module=str(bench_module),
        created_at=score_time,
        target_launch=target_launch,
        prompt_served=prompt_served,
        submission=dataclasses.replace(submission_meta, received_at=score_time),
        grader=grader,
        inputs_digest=inputs_digest,
        canonical_pass_id=pass_id,
        label=serve_log.sanitize_label(label),
        passes=(score_pass,),
    )
    serve_log.write_record(sub_dir, record)

    return {
        "run_id": run_id,
        "chal_id": chal_id,
        "benchmark_module": str(bench_module),
        "submission_id": submission_id,
        "trial_id": submission_id,  # back-compat alias
        "label": record.label,
        "scores": scores_out,
        "evidence": evidence,
        "run_dir": str(sub_dir),  # back-compat: the inspectable submission dir
        "submission_dir": str(sub_dir),
        "pass_id": pass_id,
        "inputs_digest": inputs_digest,
        "scored_at": ts,
    }
