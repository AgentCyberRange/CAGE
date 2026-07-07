"""Helpers for applying CLI overrides to project.yml mappings."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


def clone_project(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a mutable deep copy of a project mapping."""
    return copy.deepcopy(raw)


def parse_set_expression(raw: str) -> tuple[str, Any]:
    """Parse ``path=value`` using YAML semantics for the value."""
    if "=" not in raw:
        raise ValueError("--set must use KEY=VALUE syntax")
    path, value_src = raw.split("=", 1)
    path = path.strip()
    if not path:
        raise ValueError("--set path cannot be empty")
    try:
        value = yaml.safe_load(value_src)
    except yaml.YAMLError as exc:
        raise ValueError(f"--set {path}: invalid YAML value: {exc}") from exc
    return path, value


def set_project_path(raw: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted mapping path in a project.yml mapping."""
    parts = [part.strip() for part in path.split(".") if part.strip()]
    if not parts:
        raise ValueError("project.yml path cannot be empty")
    cursor: Any = raw
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set {path!r}: {part!r} is not inside a mapping")
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {path!r}: {part!r} is not a mapping")
        cursor = child
    if not isinstance(cursor, dict):
        raise ValueError(f"Cannot set {path!r}: parent is not a mapping")
    cursor[parts[-1]] = value


def apply_set_expressions(raw: dict[str, Any], values: Iterable[str]) -> None:
    """Apply multiple CLI ``--set`` expressions to an experiment project mapping."""
    for item in values:
        path, value = parse_set_expression(item)
        set_project_path(raw, path, value)


def override_selected_agent_field(
    raw: dict[str, Any],
    *,
    agent_ids: Iterable[str],
    field: str,
    value: Any,
    flag: str,
) -> None:
    """Set one field for exactly one selected agent."""
    requested = [str(agent_id) for agent_id in agent_ids if str(agent_id)]
    agents = raw.get("agents", []) or []
    if not isinstance(agents, list):
        raise ValueError(f"agents must be a list before {flag} can be applied")

    candidates: list[dict[str, Any]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        if requested:
            if str(agent.get("id") or "") in requested:
                candidates.append(agent)
        else:
            candidates.append(agent)

    if len(candidates) != 1:
        hint = f"select exactly one agent with --agent before using {flag}"
        if requested:
            hint = f"--agent matched {len(candidates)} agent(s); expected exactly one"
        raise ValueError(f"Cannot apply {flag} {value!r}: {hint}")
    candidates[0][field] = value


def merge_selected_agent_params(
    raw: dict[str, Any],
    *,
    agent_ids: Iterable[str],
    params: dict[str, Any],
    flag: str = "--param",
) -> None:
    """Merge custom-agent ``params`` into exactly one selected agent.

    Creates the agent's ``params`` mapping if absent; CLI values override any
    same-named manifest / yaml defaults (the manifest applies its own defaults
    underneath at load time).
    """
    if not params:
        return
    requested = [str(agent_id) for agent_id in agent_ids if str(agent_id)]
    agents = raw.get("agents", []) or []
    if not isinstance(agents, list):
        raise ValueError(f"agents must be a list before {flag} can be applied")

    candidates: list[dict[str, Any]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        if requested:
            if str(agent.get("id") or "") in requested:
                candidates.append(agent)
        else:
            candidates.append(agent)

    if len(candidates) != 1:
        hint = f"select exactly one agent with --agent before using {flag}"
        if requested:
            hint = f"--agent matched {len(candidates)} agent(s); expected exactly one"
        raise ValueError(f"Cannot apply {flag}: {hint}")
    existing = candidates[0].get("params")
    merged: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    merged.update(params)
    candidates[0]["params"] = merged


def override_selected_agent_model(
    raw: dict[str, Any],
    *,
    agent_ids: Iterable[str],
    model_id: str,
    sources: Iterable[str] | None = None,
) -> None:
    """Set the model for one selected agent.

    With ``sources``, configure multi-source rotation:
    ``models: [{id: <model_id>, sources: [...]}]`` — ``model_id`` is the logical
    run key and the sources are registered endpoints rotated per trial. Without
    sources it sets the plain ``agents[].model = model_id``.
    """
    source_list = [str(s).strip() for s in (sources or []) if str(s).strip()]
    requested = [str(agent_id) for agent_id in agent_ids if str(agent_id)]
    agents = raw.get("agents", []) or []
    if not isinstance(agents, list):
        raise ValueError("agents must be a list before --model can be applied")

    candidates: list[dict[str, Any]] = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        if requested:
            if str(agent.get("id") or "") in requested:
                candidates.append(agent)
        else:
            candidates.append(agent)

    flag = "--model-source" if source_list else "--model"
    if len(candidates) != 1:
        hint = f"select exactly one agent with --agent before using {flag}"
        if requested:
            hint = f"--agent matched {len(candidates)} agent(s); expected exactly one"
        raise ValueError(f"Cannot apply {flag}: {hint}")
    if source_list:
        if not str(model_id or "").strip():
            raise ValueError(
                "--model-source requires --model <logical-id> (the run key the "
                "sources rotate behind)"
            )
        candidates[0]["models"] = [{"id": model_id, "sources": source_list}]
        candidates[0].pop("model", None)
    else:
        candidates[0]["model"] = model_id
        candidates[0].pop("models", None)


def _absolutize_path(value: Any, *, base_dir: Path) -> Any:
    if value is None:
        return value
    text = str(value).strip()
    if not text:
        return value
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return str(candidate.resolve())


def _normalize_path_at(raw: dict[str, Any], path: tuple[str, ...], *, base_dir: Path) -> None:
    cursor: Any = raw
    for part in path[:-1]:
        if not isinstance(cursor, dict):
            return
        cursor = cursor.get(part)
    if not isinstance(cursor, dict) or path[-1] not in cursor:
        return
    cursor[path[-1]] = _absolutize_path(cursor[path[-1]], base_dir=base_dir)


def _normalize_project_relative_paths(
    raw: dict[str, Any],
    *,
    source_project: Path,
) -> dict[str, Any]:
    """Return a copy whose project-local paths survive temp-file relocation."""
    normalized = clone_project(raw)
    base_dir = source_project.parent
    for path in (
        ("models_file",),
        ("eval", "benchmark", "module"),
        ("eval", "benchmark", "benchmark_root"),
    ):
        _normalize_path_at(normalized, path, base_dir=base_dir)
    # Custom-agent `source:` is resolved relative to the project file, so it must
    # survive the move to a temp effective-project location too (agents is a list,
    # not a dotted scalar path, so normalize each entry here).
    agents = normalized.get("agents")
    if isinstance(agents, list):
        for agent in agents:
            if isinstance(agent, dict) and agent.get("source"):
                agent["source"] = _absolutize_path(agent["source"], base_dir=base_dir)
    return normalized


def materialize_effective_project(
    source_project: Path,
    raw: dict[str, Any],
) -> Path:
    """Write an effective project file outside the source project directory."""
    source_project = source_project.resolve()
    normalized = _normalize_project_relative_paths(raw, source_project=source_project)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="cage-effective-",
        suffix=".yml",
        delete=False,
    ) as handle:
        yaml.safe_dump(normalized, handle, sort_keys=False)
        return Path(handle.name)
