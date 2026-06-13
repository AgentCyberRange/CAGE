"""Run-record snapshot builders: planned trials, run manifest, resume snapshots."""
from __future__ import annotations

import hashlib
import json
import logging
from cage.agents.base import AgentInstance
from cage.artifacts.writer import ExperimentArtifactWriter
from cage.artifacts.run_storage import RunStorage
from cage.experiment.engine.run_context import ExperimentRun
from cage.experiment.model import Trial, TrialResult
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


AGENT_HOME = "/home/agent"


RUN_MANIFEST_FILE = "run_manifest.json"


RUN_HISTORY_FILE = "run_history.json"


def _planned_trial_records(trials: list[Trial]) -> list[dict[str, Any]]:
    return [
        {
            "trial_id": trial.id,
            "trial_index": trial.index,
            "trial_type": trial.type.value,
            "sample_id": trial.sample_id,
        }
        for trial in trials
    ]


def _save_planned_trials(storage: RunStorage, trials: list[Trial]) -> None:
    """Write the legacy planned-trials compatibility artifact.

    ``ExperimentPlan`` is the canonical planned workload, but the web inspector
    and older resume paths still understand ``planned_trials.json``. Keep
    writing the file during migration, and register it in ``artifact_index`` so
    inspect/export code can tell it is a compatibility projection rather than
    the source of truth.
    """

    planned = _planned_trial_records(trials)
    planned_path = storage.run_dir / "planned_trials.json"
    planned_path.write_text(
        json.dumps(planned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        ExperimentArtifactWriter(storage.run_dir).mark_run_artifact(
            artifact_id="run.compat.planned_trials",
            path=planned_path,
            kind="compat_planned_trials",
            schema_version="planned_trials.compat.v1",
            producer="_save_planned_trials",
            replayability="compatibility",
        )
    except Exception as exc:
        logger.warning(
            "canonical planned_trials artifact registration failed for %s: %s",
            storage.run_dir,
            exc,
        )


def _json_safe_for_manifest(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _json_safe_for_manifest(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_for_manifest(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe_for_manifest(v) for v in value)
    return repr(value)


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        _json_safe_for_manifest(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _safe_attr(obj: Any, name: str, default: Any = "") -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _model_resume_snapshot(model: Any) -> dict[str, Any]:
    api_key = str(_safe_attr(model, "api_key", "") or "")
    return _json_safe_for_manifest({
        "id": _safe_attr(model, "id", ""),
        "provider": _safe_attr(model, "provider", ""),
        "model": _safe_attr(model, "model", ""),
        "base_url": _safe_attr(model, "base_url", ""),
        "auth_source": _safe_attr(model, "auth_source", ""),
        "timeout": _safe_attr(model, "timeout", ""),
        "max_retries": _safe_attr(model, "max_retries", ""),
        "extra": _safe_attr(model, "extra", {}),
        # Keep equality strict without writing the secret itself to disk.
        "api_key_sha256": hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        if api_key else "",
    })


def _agent_resume_snapshot(agent: AgentInstance) -> dict[str, Any]:
    agent_type = _safe_attr(agent, "agent_type", None)
    return _json_safe_for_manifest({
        "id": _safe_attr(agent, "id", ""),
        "label": agent.label(),
        "kind": _safe_attr(agent_type, "name", type(agent_type).__name__),
        "stateful": bool(_safe_attr(agent, "stateful", False)),
        "home": _safe_attr(agent, "home", ""),
        "session_args": list(_safe_attr(agent, "session_args", []) or []),
        "shared_paths": list(_safe_attr(agent, "shared_paths", []) or []),
        "skill": _safe_attr(agent, "skill", ""),
        "plugins": list(_safe_attr(agent, "plugins", []) or []),
        "version": _safe_attr(agent, "version", ""),
        "image": _safe_attr(agent, "image", ""),
        "effective_image": _safe_attr(agent, "effective_image", ""),
        "max_rounds": _safe_attr(agent, "max_rounds", 0),
        "max_concurrent": _safe_attr(agent, "max_concurrent", 0),
        "context_compaction_threshold": _safe_attr(
            agent, "context_compaction_threshold", "",
        ),
    })


def _dataclass_snapshot(value: Any) -> dict[str, Any]:
    try:
        return _json_safe_for_manifest(asdict(value))
    except TypeError:
        return _json_safe_for_manifest(dict(getattr(value, "__dict__", {})))


def _semantic_resume_config(run: ExperimentRun, agent: AgentInstance) -> dict[str, Any]:
    target = _dataclass_snapshot(run.target)
    # Runtime-generated by the embedded target server; it changes every run and
    # is not a user configuration field.
    target.pop("server_url", None)
    benchmark = run.benchmark
    axes_fn = getattr(benchmark, "variant_display_axes", None)
    variant_axes = axes_fn() if callable(axes_fn) else {}
    return _json_safe_for_manifest({
        "experiment": run.name,
        "benchmark": {
            "name": getattr(benchmark, "name", type(benchmark).__name__),
            "class": f"{type(benchmark).__module__}.{type(benchmark).__qualname__}",
            "sample_limit": run.sample_limit,
            "sample_ids": list(run.sample_ids) or None,
            "prompt_levels": list(variant_axes.get("prompt", ())) or None,
        },
        "agent": _agent_resume_snapshot(agent),
        "model": _model_resume_snapshot(agent.model),
        "proxy": _dataclass_snapshot(run.proxy),
        "execution": _dataclass_snapshot(run.execution),
        "target": target,
        "output": _dataclass_snapshot(run.output),
        "admission": _dataclass_snapshot(run.admission),
        "resume_policy": {
            "retry_reasons": list(run.resume_retry_reasons),
            "max_attempts": run.resume_max_attempts,
            "select_id_pattern": run.resume_select_id_pattern,
            "keep_if": _dataclass_snapshot(run.resume_keep_if),
        },
    })


def _build_run_manifest(
    run: ExperimentRun,
    agent: AgentInstance,
    trials: list[Trial],
) -> dict[str, Any]:
    planned = _planned_trial_records(trials)
    semantic_config = _semantic_resume_config(run, agent)
    project_yml_sha256 = _file_sha256(run.project_file)
    trial_plan_fingerprint = _json_fingerprint(planned)
    semantic_config_fingerprint = _json_fingerprint(semantic_config)
    manifest = {
        "schema_version": 1,
        "experiment": run.name,
        "agent_label": agent.label(),
        "project_yml_sha256": project_yml_sha256,
        "semantic_config_fingerprint": semantic_config_fingerprint,
        "trial_plan_fingerprint": trial_plan_fingerprint,
        "trial_count": len(planned),
        "semantic_config": semantic_config,
        "planned_trials": planned,
    }
    manifest["fingerprint"] = _json_fingerprint({
        "schema_version": manifest["schema_version"],
        "agent_label": manifest["agent_label"],
        "project_yml_sha256": project_yml_sha256,
        "semantic_config_fingerprint": semantic_config_fingerprint,
        "trial_plan_fingerprint": trial_plan_fingerprint,
    })
    return manifest


def _save_run_manifest(storage: RunStorage, manifest: dict[str, Any]) -> None:
    """Write the legacy resume manifest as an indexed compatibility artifact.

    ``ExperimentPlan`` and ``ExperimentRecord`` are the canonical run contracts.
    The manifest remains useful for current resume diagnostics, so keep writing
    it while making its compatibility role explicit in ``artifact_index``.
    """

    manifest_path = storage.run_dir / RUN_MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        ExperimentArtifactWriter(storage.run_dir).mark_run_artifact(
            artifact_id="run.compat.run_manifest",
            path=manifest_path,
            kind="compat_run_manifest",
            schema_version="run_manifest.compat.v1",
            producer="_save_run_manifest",
            replayability="compatibility",
        )
    except Exception as exc:
        logger.warning(
            "canonical run_manifest artifact registration failed for %s: %s",
            storage.run_dir,
            exc,
        )


def _result_terminal_status(result: TrialResult) -> str:
    status = str(result.metadata.get("status") or "").strip().lower()
    if status:
        return status
    return "completed" if result.exit_code == 0 else "failed"

