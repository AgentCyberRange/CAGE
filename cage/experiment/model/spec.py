"""Declarative experiment specification — pure user intent from YAML.

``ExperimentSpec`` and its sub-specs carry no benchmark instances, no
resolved credentials, and no live runtime handles. The loader here is
side-effect free: it reads YAML and maps it to immutable dataclasses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from cage.contracts.coerce import (
    optional_float,
    optional_int,
    positive_float_or_none,
    positive_int_or_none,
)
from cage.contracts.execution import normalize_max_rounds_config
from cage.experiment.model._serde import _json_ready


@dataclass(frozen=True)
class ExperimentIdentity:
    """User-facing identity for one experiment specification.

    The identity is copied from ``project.yml::project`` and stays stable across
    planning, dry-run output, run records, and inspector display. It intentionally
    does not include runtime state such as generated run directories.
    """

    experiment_id: str
    display_name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    run_id: str = ""


@dataclass(frozen=True)
class BenchmarkReference:
    """Reference to the benchmark adapter declared by the user spec.

    The reference records module/class/root paths as data only. Loading this
    object must never import the benchmark module; imports belong to later
    compatibility or runtime layers.
    """

    id: str
    project_name: str
    module: str
    class_name: str = ""
    benchmark_root: str = ""
    package_ref: str = ""
    default_config_ref: str = ""


@dataclass(frozen=True)
class AgentModelSelection:
    """One model selected for an agent in the experiment workload."""

    model: str
    max_concurrent: int | None = None
    overrides: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AgentSelection:
    """One agent class/profile selected by ``ExperimentSpec.workload``."""

    agent: str
    kind: str
    profile: str = "stateless"
    models: tuple[AgentModelSelection, ...] = ()
    max_concurrent: int | None = None


@dataclass(frozen=True)
class TaskSelection:
    """User-specified sample/task selection before benchmark expansion."""

    samples: tuple[str, ...] = ()
    max_sample_num: int | None = None
    max_trial_num: int | None = None


@dataclass(frozen=True)
class WorkloadSpec:
    """The workload section of ``ExperimentSpec``.

    Workload is still benchmark-agnostic at this layer. Benchmark-owned axes are
    stored under ``variants`` as plain data and are not interpreted by Cage core.
    """

    subjects: tuple[AgentSelection, ...]
    task_selection: TaskSelection
    variants: Mapping[str, tuple[str, ...]]
    passk: int = 1


@dataclass(frozen=True)
class ProtocolControls:
    """Stop conditions that apply to every planned trial unless overridden."""

    # Config value space: "unlimited" | -1 (benchmark default) | N | None.
    # Absent ⇒ "unlimited" (the run is bounded by other stop conditions).
    max_rounds: int | str | None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost: float | None = None


@dataclass(frozen=True)
class SchedulerSpec:
    """Concurrency policy resolved from runtime YAML into pure data."""

    max_trials_global: int
    max_concurrent: int | None = None
    max_target_setups: int = 1


@dataclass(frozen=True)
class TimeoutSpec:
    """Timeout controls that are independent of scoring success."""

    trial_timeout_s: float
    request_timeout_s: float
    target_startup_timeout_s: float | None = None
    target_compose_timeout_s: float | None = None


@dataclass(frozen=True)
class ProxySpec:
    """Proxy configuration that can be planned without starting the proxy."""

    enabled: bool
    request_timeout_s: float
    upstream_http_proxy: str = ""


@dataclass(frozen=True)
class RuntimeSpec:
    """Runtime policy copied from the user spec without live handles."""

    scheduler: SchedulerSpec
    timeouts: TimeoutSpec
    proxy: ProxySpec
    target_enabled: bool = True
    allow_launch_build: bool = False
    inspector_start: bool = True


@dataclass(frozen=True)
class ScoringSelection:
    """Scoring selection before resolving concrete scorer implementations."""

    scorer: str = "benchmark_default"
    judge_model: str | None = None


@dataclass(frozen=True)
class ObservationSpec:
    """Human/tool observation preferences from the user spec."""

    terminal_ui: str = "plain"
    debug_log: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    """Serializable user intent loaded from ``project.yml``.

    This is not the legacy runtime config. It contains no benchmark instance,
    no resolved model credentials, no Docker state, and no mutable runtime
    handles.
    """

    schema_version: str
    project_file: Path
    base_dir: Path
    identity: ExperimentIdentity
    benchmark: BenchmarkReference
    workload: WorkloadSpec
    protocol: ProtocolControls
    runtime: RuntimeSpec
    scoring: ScoringSelection
    observation: ObservationSpec


def load_experiment_spec(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    sample_ids: tuple[str, ...] = (),
    max_sample_num: int | None = None,
    max_trial_num: int | None = None,
) -> ExperimentSpec:
    """Load ``project.yml`` into a pure ``ExperimentSpec``.

    This loader is intentionally side-effect free. It reads YAML and performs
    compatibility mapping only; it does not import the benchmark module, call
    ``Benchmark.setup()``, load model credentials, instantiate agent adapters, or
    touch Docker. CLI selection arguments are accepted here because they are
    overlays on user intent, not runtime state.
    """

    project_file = Path(path).expanduser().resolve()
    resolved_base_dir = (
        Path(base_dir).expanduser().resolve()
        if base_dir is not None
        else project_file.parent
    )
    raw = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{project_file} must contain a YAML mapping")
    return experiment_spec_from_project_mapping(
        raw,
        project_file=project_file,
        base_dir=resolved_base_dir,
        sample_ids=sample_ids,
        max_sample_num=max_sample_num,
        max_trial_num=max_trial_num,
    )


def experiment_spec_from_project_mapping(
    raw: Mapping[str, Any],
    *,
    project_file: Path,
    base_dir: Path,
    sample_ids: tuple[str, ...] = (),
    max_sample_num: int | None = None,
    max_trial_num: int | None = None,
) -> ExperimentSpec:
    """Translate the current project YAML shape into ``ExperimentSpec``.

    The mapper is a compatibility layer. It keeps existing files working while
    making the new contract explicit. All nested data is copied into immutable
    dataclasses so later runtime code cannot mutate the user's raw YAML in
    place.
    """

    project_raw = _mapping(raw.get("project"), "project")
    eval_raw = _mapping(raw.get("eval"), "eval")
    benchmark_raw = _benchmark_mapping(eval_raw)
    runtime_raw = _mapping(raw.get("runtime", raw.get("execution", {})), "runtime")
    proxy_raw = _mapping(raw.get("proxy"), "proxy")
    target_raw = _mapping(raw.get("target"), "target")
    logging_raw = _mapping(raw.get("logging"), "logging")

    project_name = str(project_raw.get("name") or project_file.stem)
    task_selection = TaskSelection(
        samples=tuple(str(item) for item in sample_ids),
        max_sample_num=(
            max_sample_num
            if max_sample_num is not None
            else optional_int(eval_raw.get("limit"))
        ),
        max_trial_num=(
            max_trial_num
            if max_trial_num is not None
            else optional_int(runtime_raw.get("max_trial"))
        ),
    )

    return ExperimentSpec(
        schema_version="experiment_spec.v1",
        project_file=project_file,
        base_dir=base_dir,
        identity=ExperimentIdentity(
            experiment_id=project_name,
            display_name=project_name,
            run_id=str(project_raw.get("run_id") or "").strip(),
            tags=_string_tuple(project_raw.get("tags")),
            description=str(project_raw.get("description") or ""),
        ),
        benchmark=BenchmarkReference(
            id=_benchmark_id(project_name),
            project_name=project_name,
            module=str(benchmark_raw.get("module") or ""),
            class_name=str(benchmark_raw.get("class") or ""),
            benchmark_root=str(benchmark_raw.get("benchmark_root") or ""),
            package_ref=str(base_dir),
            default_config_ref=str(project_file),
        ),
        workload=WorkloadSpec(
            subjects=_agent_selections(raw),
            task_selection=task_selection,
            variants=_benchmark_variants(benchmark_raw),
            passk=max(1, int(runtime_raw.get("passk", 1) or 1)),
        ),
        protocol=ProtocolControls(
            # Config space: absent/"unlimited" ⇒ unlimited; -1 ⇒ benchmark
            # default; N ⇒ N rounds. (Resolved against the benchmark sample
            # default later, in resolve_max_rounds.)
            max_rounds=normalize_max_rounds_config(
                runtime_raw.get("max_rounds"), default="unlimited"
            ),
            max_input_tokens=positive_int_or_none(runtime_raw.get("max_input_tokens")),
            max_output_tokens=positive_int_or_none(runtime_raw.get("max_output_tokens")),
            max_cost=positive_float_or_none(runtime_raw.get("max_cost")),
        ),
        runtime=RuntimeSpec(
            scheduler=SchedulerSpec(
                # 0/unset/negative = unlimited (no global cap); the engine
                # treats <=0 as "no global gate". Legacy aliases preserved.
                max_trials_global=max(0, int(
                    runtime_raw.get(
                        "max_trials_global",
                        runtime_raw.get(
                            "n_concurrent",
                            runtime_raw.get("max_running_trials", 0),
                        ),
                    )
                    or 0
                )),
                max_concurrent=_first_agent_concurrency(raw),
                max_target_setups=max(0, int(
                    runtime_raw.get(
                        "max_target_setups",
                        runtime_raw.get("max_sample_target_setups", 1),
                    )
                    or 0
                )),
            ),
            timeouts=TimeoutSpec(
                trial_timeout_s=float(
                    runtime_raw.get(
                        "timeout",
                        runtime_raw.get("timeout_seconds", 0.0),
                    )
                    or 0.0
                ),
                request_timeout_s=float(proxy_raw.get("request_timeout", 3600.0) or 3600.0),
                target_startup_timeout_s=optional_float(target_raw.get("startup_timeout")),
                target_compose_timeout_s=optional_float(target_raw.get("compose_up_timeout")),
            ),
            proxy=ProxySpec(
                enabled=bool(proxy_raw.get("enabled", True)),
                request_timeout_s=float(proxy_raw.get("request_timeout", 3600.0) or 3600.0),
                upstream_http_proxy=str(proxy_raw.get("upstream_http_proxy", "") or ""),
            ),
            target_enabled=bool(target_raw.get("enabled", True)),
            inspector_start=str(logging_raw.get("inspect_mode", "auto") or "auto").lower() != "off",
        ),
        scoring=ScoringSelection(
            scorer="benchmark_default",
            judge_model=_judge_model(raw, benchmark_raw),
        ),
        observation=ObservationSpec(
            terminal_ui=_terminal_ui(logging_raw),
            debug_log=bool(logging_raw.get("debug_file", False)),
        ),
    )


def experiment_spec_to_mapping(spec: ExperimentSpec) -> dict[str, Any]:
    """Return a JSON-ready mapping for an ``ExperimentSpec``.

    Runtime artifact writers use this to snapshot the declarative user intent
    that produced a run. The mapping keeps path provenance as strings but does
    not import benchmark code or resolve any live runtime handles.
    """

    return _json_ready(spec)


def experiment_spec_to_json(spec: ExperimentSpec, *, indent: int | None = 2) -> str:
    """Serialize an ``ExperimentSpec`` to deterministic JSON text."""

    return (
        json.dumps(
            experiment_spec_to_mapping(spec),
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    )


def narrow_spec_to_agent_run(
    spec: ExperimentSpec,
    *,
    run_id: str,
    agent_id: str,
    agent_kind: str,
    model_id: str,
    max_concurrent: int | None,
    task_ids: tuple[str, ...],
    passk: int,
    profile: str = "stateless",
    experiment_name: str | None = None,
    benchmark: BenchmarkReference | None = None,
) -> ExperimentSpec:
    """Scope a whole-experiment spec to one agent's run-directory snapshot.

    The orchestrator still schedules each agent under its own
    ``.cage_runs/<agent_label>/<run_id>`` directory, so the durable per-run
    snapshot is a single-subject view of the experiment: this agent as the
    only subject, the actually-scheduled task ids as the selection, the live
    run id stamped on the identity. Everything else — protocol controls,
    runtime spec, scoring, observation, and crucially the benchmark variant
    axes — carries over from the one parsed spec, so the snapshot can no
    longer drift from what the run actually resolved.
    """

    identity = replace(spec.identity, run_id=run_id)
    if experiment_name:
        identity = replace(
            identity, experiment_id=experiment_name, display_name=experiment_name
        )
    return replace(
        spec,
        identity=identity,
        benchmark=benchmark if benchmark is not None else spec.benchmark,
        workload=WorkloadSpec(
            subjects=(
                AgentSelection(
                    agent=agent_id,
                    kind=agent_kind,
                    profile=profile,
                    models=(
                        AgentModelSelection(
                            model=model_id, max_concurrent=max_concurrent
                        ),
                    ),
                    max_concurrent=max_concurrent,
                ),
            ),
            task_selection=TaskSelection(samples=task_ids),
            variants=spec.workload.variants,
            passk=max(1, passk),
        ),
        runtime=replace(
            spec.runtime,
            scheduler=replace(spec.runtime.scheduler, max_concurrent=max_concurrent),
        ),
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping, accepting ``None`` as an empty mapping.

    The project YAML format has many optional sections. Treating absent sections
    as empty keeps compatibility mapping straightforward while still rejecting
    malformed non-mapping sections with a useful field name.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _benchmark_mapping(eval_raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the benchmark mapping from legacy ``eval`` configuration."""

    benchmark_raw = eval_raw.get("benchmark", eval_raw)
    if isinstance(benchmark_raw, str):
        return {"module": "./benchmark.py", "name": benchmark_raw}
    return _mapping(benchmark_raw, "eval.benchmark")


def _benchmark_id(project_name: str) -> str:
    """Derive a stable benchmark id from the project name for compatibility."""

    return project_name.strip().lower().replace("-", "_")


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Normalize an optional YAML scalar/list into a tuple of strings."""

    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _benchmark_variants(benchmark_raw: Mapping[str, Any]) -> Mapping[str, tuple[str, ...]]:
    """Map legacy benchmark level lists into generic workload variants.

    Existing example configs use ``prompt_levels`` and ``hint_levels``. The new
    core contract stores these as benchmark-owned axes named ``prompt_level`` and
    ``hint_level`` without interpreting their domain meaning.
    """

    variants: dict[str, tuple[str, ...]] = {}
    for key, value in benchmark_raw.items():
        if key.endswith("_levels"):
            axis = key[: -len("s")]
            variants[axis] = tuple(str(item) for item in _list(value))
    return variants


def _list(value: Any) -> list[Any]:
    """Normalize an optional YAML scalar/list into a list for iteration."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _agent_selections(raw: Mapping[str, Any]) -> tuple[AgentSelection, ...]:
    """Map legacy ``agents`` and ``subjects`` fields into workload subjects."""

    subject_models = _subject_model_ids(raw.get("subjects", []) or [])
    selections: list[AgentSelection] = []
    for agent_raw in _list(raw.get("agents", [])):
        agent_map = _mapping(agent_raw, "agents[]")
        agent_id = str(
            agent_map.get("id")
            or agent_map.get("kind")
            or agent_map.get("agent_type")
            or ""
        )
        if not agent_id:
            continue
        kind = str(agent_map.get("kind") or agent_map.get("agent_type") or agent_id)
        models = _agent_model_selections(agent_map, subject_models)
        selections.append(
            AgentSelection(
                agent=agent_id,
                kind=kind,
                models=models,
                max_concurrent=optional_int(agent_map.get("max_concurrent")),
            )
        )
    return tuple(selections)


def _subject_model_ids(subjects_raw: Any) -> tuple[str, ...]:
    """Extract legacy top-level ``subjects`` model ids."""

    model_ids: list[str] = []
    for subject in _list(subjects_raw):
        if isinstance(subject, Mapping):
            model_id = subject.get("id") or subject.get("model")
        else:
            model_id = subject
        if model_id:
            model_ids.append(str(model_id))
    return tuple(model_ids)


def _agent_model_selections(
    agent_map: Mapping[str, Any],
    subject_models: tuple[str, ...],
) -> tuple[AgentModelSelection, ...]:
    """Resolve model ids configured on one agent entry."""

    if agent_map.get("model"):
        return (AgentModelSelection(model=str(agent_map["model"])),)
    if agent_map.get("models"):
        models: list[AgentModelSelection] = []
        for item in _list(agent_map.get("models")):
            if isinstance(item, Mapping):
                model_id = item.get("id") or item.get("model")
                overrides = {k: v for k, v in item.items() if k not in {"id", "model"}}
                max_concurrent = optional_int(item.get("max_concurrent"))
            else:
                model_id = item
                overrides = {}
                max_concurrent = None
            if model_id:
                models.append(
                    AgentModelSelection(
                        model=str(model_id),
                        max_concurrent=max_concurrent,
                        overrides=overrides,
                    )
                )
        return tuple(models)
    return tuple(AgentModelSelection(model=model_id) for model_id in subject_models)


def _first_agent_concurrency(raw: Mapping[str, Any]) -> int | None:
    """Return the first configured per-agent concurrency cap, if present."""

    for agent_raw in _list(raw.get("agents", [])):
        agent_map = _mapping(agent_raw, "agents[]")
        cap = optional_int(agent_map.get("max_concurrent"))
        if cap is not None:
            return cap
    return None


def _judge_model(raw: Mapping[str, Any], benchmark_raw: Mapping[str, Any]) -> str | None:
    """Resolve the first configured judge model id without loading model auth."""

    judge_raw = raw.get("judge")
    if isinstance(judge_raw, Mapping) and judge_raw.get("id"):
        return str(judge_raw["id"])
    benchmark_judge = benchmark_raw.get("judge")
    if isinstance(benchmark_judge, Mapping):
        models = _list(benchmark_judge.get("models"))
        if models:
            return str(models[0])
    return None


def _terminal_ui(logging_raw: Mapping[str, Any]) -> str:
    """Normalize legacy terminal UI flags into a display policy string."""

    value = logging_raw.get("terminal_ui", "plain")
    if isinstance(value, bool):
        return "plain" if value else "off"
    return str(value or "plain")
