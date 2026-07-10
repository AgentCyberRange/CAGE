"""Project configuration — loads project.yml (Snowl-compatible format).

project.yml structure:
  project:
    name: ...

  subjects:              # models being evaluated
    - id: glm-5.1-sii

  judge:                 # model for scoring (optional)
    id: llama-33-70b
    temperature: 0.0
    max_tokens: 4096

  proxy:
    enabled: true
    rewrite:
      system: |
        {{ system_raw }}
        ...extra instructions...

  eval:
    benchmark:
      module: ./benchmark.py
    limit: 10

  agents:
    - id: claude_code_baseline
      kind: claude_code
      session_timeout: 300
      home: /home/agent/workspace
      session_args: [--verbose]
      shared_paths: [/home/agent/workspace/.claude]
      skill: self-improving-agent
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from cage.agents.base import AgentInstance, get_agent_type
from cage.agents.custom import load_custom_agent
from cage.benchmarks import parse_sample_slice
from cage.benchmarks.loader import load_benchmark_from_module
from cage.config import find_repo_root, resolve_models_file
from cage.config.sections import (
    ExecutionConfig,
    JudgeConfig,
    LiveCheckConfig,
    OutputConfig,
    PollingLiveCheckConfig,
    ProxyConfig,
    ReactiveLiveCheckConfig,
    ResumeKeepIf,
    TargetConfig,
)
from cage.contracts.execution import normalize_max_rounds_config
from cage.contracts.logging import LoggingConfig
from cage.experiment.engine.hooks import load_hooks
from cage.experiment.engine.run_context import ExperimentRun
from cage.experiment.model import (
    ExperimentSpec,
    experiment_spec_from_project_mapping,
)
from cage.models import ModelConfig, load_models
from cage.sandbox.admission import AdmissionConfig


def _parse_resume_id_pattern(value: Any, field_name: str) -> str | None:
    """Validate a resume id-regex knob: must be a compilable regex string."""
    import re

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string regex, got: {value!r}")
    pattern = value.strip()
    if not pattern:
        return None
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"{field_name} is not a valid regex ({pattern!r}): {exc}")
    return pattern


def _parse_resume_nonneg(value: Any, field_name: str, kind: type) -> Any | None:
    """Validate a non-negative ``int``/``float`` resume threshold (None=unset)."""
    if value is None:
        return None
    try:
        parsed = kind(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field_name} must be a {kind.__name__}, got: {value!r}"
        )
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0, got: {parsed}")
    return parsed


def _resolve_models(raw: dict[str, Any], base_dir: Path) -> tuple[dict[str, ModelConfig], Any]:
    """Resolve the model registry. Returns ``(models, models_file)`` where
    ``models_file`` is the path used (for not-found error messages)."""
    # Load model registry. Project-local ``models_file`` is still honored for
    # explicit overrides, but the normal path is repo config:
    # ``config/cage.yml::models_file`` → ``config/models.yml``.
    models_file = raw.get("models_file")
    if models_file:
        models_path = Path(str(models_file)).expanduser()
        if not models_path.is_absolute():
            models_path = base_dir / models_path
        models_path = models_path.resolve()
    else:
        repo_root = find_repo_root(base_dir)
        if repo_root is None:
            legacy_local = next(
                (
                    candidate
                    for candidate in (base_dir / "models.yml", base_dir / "models.yaml")
                    if candidate.is_file()
                ),
                None,
            )
            models_path = (
                legacy_local.resolve()
                if legacy_local
                else resolve_models_file(repo_root=Path.cwd())
            )
        else:
            models_path = resolve_models_file(repo_root=repo_root)
        models_file = str(models_path)
    return load_models(models_path), models_file


def _resolve_judge(
    raw: dict[str, Any], models: dict[str, ModelConfig], models_file: Any
) -> JudgeConfig | None:
    """Resolve the optional judge model config."""
    judge_raw = raw.get("judge")
    if not judge_raw:
        return None
    judge_model_id = judge_raw["id"]
    if judge_model_id not in models:
        raise ValueError(f"Judge model '{judge_model_id}' not found in {models_file}")
    return JudgeConfig(
        model=models[judge_model_id],
        temperature=judge_raw.get("temperature", 0.0),
        max_tokens=judge_raw.get("max_tokens", 4096),
    )


def _resolve_proxy(raw: dict[str, Any], spec: "ExperimentSpec") -> ProxyConfig:
    """Resolve the in-container proxy config.

    Fields the declarative spec also carries are taken FROM the spec — one
    parse, so the recorded snapshot cannot drift from the running config.
    Only ``rewrite_system`` (not part of the declarative surface) is read
    from the raw mapping here.
    """
    proxy_raw = raw.get("proxy", {})
    rewrite_system = ""
    rewrite_raw = proxy_raw.get("rewrite", {})
    if isinstance(rewrite_raw, dict):
        rewrite_system = rewrite_raw.get("system", "")
    return ProxyConfig(
        enabled=spec.runtime.proxy.enabled,
        rewrite_system=rewrite_system,
        request_timeout=spec.runtime.proxy.request_timeout_s,
        upstream_http_proxy=spec.runtime.proxy.upstream_http_proxy,
    )


def _resolve_benchmark(
    raw: dict[str, Any], base_dir: Path
) -> tuple[Any, Path, int | None, slice | None]:
    """Load and set up the benchmark. Returns ``(benchmark, benchmark_dir,
    sample_limit, sample_slice)``. ``sample_limit`` (``eval.limit``) and
    ``sample_slice`` (``eval.sample_slice``) are Layer-3 run knobs, carried on
    the run (not the benchmark)."""
    eval_raw = raw.get("eval", {})
    bench_cfg = eval_raw.get("benchmark", eval_raw)
    benchmark_dir: Path
    if isinstance(bench_cfg, dict) and "module" in bench_cfg:
        module_path = base_dir / bench_cfg["module"]
        benchmark_dir = module_path.resolve().parent
        # Pass any extra keys as kwargs to Benchmark.__init__
        bench_kwargs = {
            k: v for k, v in bench_cfg.items()
            if k not in ("module", "class")
        }
        benchmark = load_benchmark_from_module(
            module_path, class_name=bench_cfg.get("class"),
            kwargs=bench_kwargs,
        )
    elif isinstance(bench_cfg, str):
        # Short form: eval.benchmark = "<dir-name>" → ./benchmark.py
        module_path = base_dir / "benchmark.py"
        benchmark_dir = module_path.resolve().parent
        benchmark = load_benchmark_from_module(module_path)
    else:
        raise ValueError("eval.benchmark must specify a module path or benchmark name")

    benchmark.setup()

    sample_limit_raw = eval_raw.get("limit")
    sample_limit = int(sample_limit_raw) if sample_limit_raw is not None else None
    sample_slice = parse_sample_slice(eval_raw.get("sample_slice"))
    return benchmark, benchmark_dir, sample_limit, sample_slice


def _resolve_agents(
    raw: dict[str, Any],
    models: dict[str, ModelConfig],
    models_file: Any,
    base_dir: Path,
) -> list[AgentInstance]:
    """Build the agent instances (expanding ``models``/``subjects`` matrices).

    An agent entry with a ``source:`` is a *custom agent* — a self-contained
    directory holding the agent's code plus an ``agent.yml`` manifest. We load
    that manifest into a :class:`CustomAgent` carrying its own source path /
    launch command / env, instead of looking up a built-in by ``kind``. The
    manifest's host paths resolve relative to ``base_dir`` (the experiment
    file's directory), like ``eval.benchmark.module``.
    """
    agent_cfgs = raw.get("agents", [])
    subjects = raw.get("subjects", [])

    agents: list[AgentInstance] = []
    for acfg in agent_cfgs:
        source = acfg.get("source")
        if source:
            at = load_custom_agent(source, base_dir, acfg.get("params"))
        else:
            kind = acfg.get("kind", acfg.get("agent_type", ""))
            at = get_agent_type(kind)

        # Determine which model(s) this agent uses
        # ``agents[].models`` is the benchmark-friendly form: one logical
        # agent class can declare the model ids it should run against.
        if "model" in acfg:
            model_id = acfg.get("model", "")
            if model_id and model_id not in models:
                raise ValueError(f"Model '{model_id}' not found in {models_file}")
            model = models[model_id] if model_id else next(iter(models.values()))
            agents.append(_build_agent_instance(acfg, at, model))
        elif acfg.get("models"):
            for model_id, source_ids, source_conc, model_overrides in (
                _agent_model_entries(acfg.get("models"))
            ):
                model_agent_cfg = {**acfg, **model_overrides}
                model_agent_cfg.pop("models", None)
                if source_ids:
                    # Case 2: this run round-robins across several registered
                    # endpoints. ``model_id`` is the logical key; the run's
                    # representative model carries that id with endpoint fields
                    # from the first source (for labels / preflight / single-key
                    # readers), and the full resolved pool drives per-trial
                    # rotation. ``source_conc`` carries any per-source
                    # concurrency caps.
                    source_models = _resolve_model_sources(
                        model_id, source_ids, models, models_file
                    )
                    logical = replace(source_models[0], id=model_id)
                    agents.append(_build_agent_instance(
                        model_agent_cfg, at, logical,
                        model_sources=source_models,
                        source_concurrency=source_conc,
                    ))
                else:
                    if model_id not in models:
                        raise ValueError(
                            f"Model '{model_id}' not found in {models_file}"
                        )
                    agents.append(
                        _build_agent_instance(model_agent_cfg, at, models[model_id])
                    )
        # If subjects are specified, expand agents x subjects.
        elif subjects:
            for subj in subjects:
                subj_id = subj if isinstance(subj, str) else subj["id"]
                if subj_id not in models:
                    raise ValueError(f"Subject '{subj_id}' not found in {models_file}")
                agents.append(_build_agent_instance(acfg, at, models[subj_id]))
        else:
            agents.append(_build_agent_instance(acfg, at, next(iter(models.values()))))
    return agents


def _resolve_execution(raw: dict[str, Any], spec: "ExperimentSpec") -> ExecutionConfig:
    """Resolve the execution/runtime config (incl. live-check)."""
    runtime_raw = raw.get("runtime", raw.get("execution", {}))
    live_check_raw = runtime_raw.get("live_check", {})
    reactive_raw = live_check_raw.get("reactive", {}) or {}
    polling_raw = live_check_raw.get("polling", {}) or {}
    live_check = LiveCheckConfig(
        enabled=live_check_raw.get("enabled", True),
        max_calls=live_check_raw.get("max_calls", 3),
        stop_on_success=live_check_raw.get("stop_on_success", True),
        reactive=ReactiveLiveCheckConfig(
            enabled=reactive_raw.get("enabled", True),
            check_on_submit=reactive_raw.get("check_on_submit", True),
            check_on_9091_call=reactive_raw.get("check_on_9091_call", True),
        ),
        polling=PollingLiveCheckConfig(
            enabled=polling_raw.get("enabled", False),
            interval_seconds=float(polling_raw.get("interval_seconds", 5.0)),
            stop_on_success=polling_raw.get("stop_on_success", True),
            confirm_polls=max(1, int(polling_raw.get("confirm_polls", 1))),
        ),
    )

    # Every field the declarative spec also carries derives FROM the spec —
    # one parse for run + snapshot (legacy aliases like max_running_trials /
    # max_sample_target_setups are honoured by the spec mapper). Fields below
    # that still read runtime_raw are the runtime-only knobs the spec does
    # not declare.
    return ExecutionConfig(
        max_trials_global=spec.runtime.scheduler.max_trials_global,
        max_target_setups=spec.runtime.scheduler.max_target_setups,
        timeout=spec.runtime.timeouts.trial_timeout_s,
        on_failure=runtime_raw.get("on_failure", "continue"),
        chunk_size=raw.get("hooks", {}).get("chunk_size"),
        agent_network_mode=runtime_raw.get("agent_network_mode", "host"),
        max_rounds=spec.protocol.max_rounds,
        max_input_tokens=spec.protocol.max_input_tokens,
        max_output_tokens=spec.protocol.max_output_tokens,
        max_cost=spec.protocol.max_cost,
        store_proxy=runtime_raw.get("store_proxy", True),
        wait_for_model=bool(runtime_raw.get("wait_for_model", False)),
        wait_timeout=float(runtime_raw.get("wait_timeout", 0.0) or 0.0),
        wait_interval=float(runtime_raw.get("wait_interval", 10.0) or 10.0),
        live_check=live_check,
        passk=spec.workload.passk,
        max_trial=spec.workload.task_selection.max_trial_num,
    )


def _resolve_logging(raw: dict[str, Any]) -> LoggingConfig:
    """Resolve the logging config."""
    log_raw = raw.get("logging", {})
    terminal_ui_raw = log_raw.get("terminal_ui", True)
    if isinstance(terminal_ui_raw, bool):
        terminal_ui_enabled = terminal_ui_raw
    else:
        terminal_ui_enabled = str(terminal_ui_raw or "").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
            "off",
            "none",
            "plain",
        }
    inspect_mode = str(log_raw.get("inspect_mode", "auto") or "auto").lower()
    if inspect_mode not in {"auto", "on", "off"}:
        raise ValueError("logging.inspect_mode must be one of: auto, on, off")
    return LoggingConfig(
        console_level=log_raw.get("level", "INFO"),
        file_level=log_raw.get("file_level", "DEBUG"),
        debug_file_enabled=log_raw.get("debug_file", False),
        terminal_ui=terminal_ui_enabled,
        inspect_mode=inspect_mode,
    )


def _resolve_target(raw: dict[str, Any]) -> TargetConfig:
    """Resolve the target / target_server config."""
    target_raw = raw.get("target", {})

    ssh_raw = target_raw.get("ssh", {})
    if "server_url" in target_raw:
        raise ValueError(
            "target.server_url is no longer a user-facing field — the orchestrator "
            "always spawns a per-run embedded target_server and sets the URL "
            "programmatically. Remove `server_url:` from project.yml."
        )
    if "embedded" in target_raw:
        raise ValueError(
            "target.embedded is no longer a user-facing field — embedded mode is "
            "always on. Remove `embedded:` from project.yml."
        )
    if target_raw.get("use_external_access"):
        raise ValueError(
            "target.use_external_access is no longer supported. Agents reach "
            "targets over the isolated docker network (their internal address), "
            "never host-published ports — this keeps targets off the host so a "
            "scanning agent can't reach them via localhost. Remove "
            "`use_external_access:` (and any SSH-tunnel settings) from project.yml."
        )
    return TargetConfig(
        enabled=target_raw.get("enabled", True),
        run_mode=target_raw.get("run_mode", "remote"),
        use_ssh_tunnel=target_raw.get("use_ssh_tunnel", False),
        jump_host=ssh_raw.get("jump_host", ""),
        jump_user=ssh_raw.get("jump_user", ""),
        ssh_key_path=ssh_raw.get("ssh_key_path", ""),
        remote_bind_address=ssh_raw.get("remote_bind_address", "127.0.0.1"),
        remote_bind_port=ssh_raw.get("remote_bind_port", 8000),
        use_external_access=False,  # forbidden above; agents use the isolated network
        host_ip_for_agent=target_raw.get("host_ip_for_agent", ""),
        network_name=target_raw.get("network_name", "cage_net"),
        startup_timeout=(
            float(target_raw["startup_timeout"])
            if target_raw.get("startup_timeout") is not None
            else None
        ),
        compose_up_timeout=(
            float(target_raw["compose_up_timeout"])
            if target_raw.get("compose_up_timeout") is not None
            else None
        ),
        target_scope=str(target_raw.get("target_scope", "per_agent") or ""),
        parallel_mode=str(target_raw.get("parallel_mode", "") or ""),
        agent_network_isolation=str(
            target_raw.get("agent_network_isolation", "per_trial_bridge")
            or "per_trial_bridge"
        ),
    )


def _resolve_admission(raw: dict[str, Any]) -> AdmissionConfig:
    """Resolve the host-level memory back-pressure (admission) config."""
    admission_raw = raw.get("admission", {}) or {}
    return AdmissionConfig(
        enabled=bool(admission_raw.get("enabled", True)),
        memory_pause_at=float(admission_raw.get("memory_pause_at", 0.80)),
        memory_resume_at=float(admission_raw.get("memory_resume_at", 0.70)),
        poll_seconds=float(admission_raw.get("poll_seconds", 3.0)),
        log_every_seconds=float(admission_raw.get("log_every_seconds", 30.0)),
    )


def _resolve_resume(
    raw: dict[str, Any],
) -> tuple[list[str], int, str | None, ResumeKeepIf]:
    """Resolve resume policy. Returns ``(retry_reasons, max_attempts,
    select_id_pattern, keep_if)``."""
    resume_raw = raw.get("resume", {}) or {}
    resume_retry_reasons_raw = resume_raw.get("retry_reasons", []) or []
    if not isinstance(resume_retry_reasons_raw, list):
        raise ValueError(
            "resume.retry_reasons must be a list of termination_reason strings"
        )
    resume_retry_reasons = [
        str(r).strip().lower() for r in resume_retry_reasons_raw if str(r).strip()
    ]
    resume_max_attempts_raw = resume_raw.get("max_attempts", 0)
    try:
        resume_max_attempts = int(resume_max_attempts_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"resume.max_attempts must be an integer, got: {resume_max_attempts_raw!r}"
        )
    if resume_max_attempts < 0:
        raise ValueError(
            f"resume.max_attempts must be >= 0, got: {resume_max_attempts}"
        )

    # resume.select — positive id-regex gate (which trials may be re-run).
    resume_select_raw = resume_raw.get("select", {}) or {}
    if not isinstance(resume_select_raw, dict):
        raise ValueError("resume.select must be a mapping (e.g. {id_matches: ...})")
    resume_select_id_pattern = _parse_resume_id_pattern(
        resume_select_raw.get("id_matches"), "resume.select.id_matches"
    )

    # resume.keep_if — veto thresholds (keep an otherwise-retryable trial).
    resume_keep_if_raw = resume_raw.get("keep_if", {}) or {}
    if not isinstance(resume_keep_if_raw, dict):
        raise ValueError(
            "resume.keep_if must be a mapping (e.g. {min_rounds: 100})"
        )
    resume_keep_if = ResumeKeepIf(
        min_rounds=_parse_resume_nonneg(
            resume_keep_if_raw.get("min_rounds"), "resume.keep_if.min_rounds", int
        ),
        min_duration_s=_parse_resume_nonneg(
            resume_keep_if_raw.get("min_duration_s"),
            "resume.keep_if.min_duration_s",
            float,
        ),
        id_matches=_parse_resume_id_pattern(
            resume_keep_if_raw.get("id_matches"), "resume.keep_if.id_matches"
        ),
    )
    return (
        resume_retry_reasons,
        resume_max_attempts,
        resume_select_id_pattern,
        resume_keep_if,
    )


def _resolve_output(raw: dict[str, Any]) -> OutputConfig:
    """Resolve the dashboard/CSV output config."""
    output_raw = raw.get("output", {})
    return OutputConfig(
        dashboard_prompt=output_raw.get("dashboard_prompt", True),
        dashboard_output=output_raw.get("dashboard_output", True),
        dashboard_reasoning=output_raw.get("dashboard_reasoning", True),
        csv_prompt=output_raw.get("csv_prompt", True),
        csv_output=output_raw.get("csv_output", True),
        csv_reasoning=output_raw.get("csv_reasoning", True),
    )


def resolve(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> ExperimentRun:
    """Resolve a ``project.yml`` into a runnable :class:`ExperimentRun`.

    Reads the declarative project file (Snowl-compatible format), wires the
    model registry, benchmark, agents, and runtime sections, and returns the
    single live run object the conductor threads through trial execution.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base_dir = Path(base_dir).expanduser() if base_dir is not None else path.parent

    project_raw = raw.get("project", {})
    name = project_raw.get("name", path.stem)

    # The declarative spec is parsed FIRST: it is the single parse of the
    # project mapping, and the resolved sections below derive their
    # overlapping fields from it (snapshot == executed config by construction).
    spec = experiment_spec_from_project_mapping(
        raw,
        project_file=path.resolve(),
        base_dir=Path(base_dir).resolve(),
    )

    models, models_file = _resolve_models(raw, base_dir)
    judge = _resolve_judge(raw, models, models_file)
    proxy = _resolve_proxy(raw, spec)
    benchmark, benchmark_dir, sample_limit, sample_slice = _resolve_benchmark(raw, base_dir)
    agents = _resolve_agents(raw, models, models_file, base_dir)
    hooks = load_hooks(raw.get("hooks", {}))
    execution = _resolve_execution(raw, spec)
    logging_config = _resolve_logging(raw)
    target_config = _resolve_target(raw)
    admission_config = _resolve_admission(raw)
    (
        resume_retry_reasons,
        resume_max_attempts,
        resume_select_id_pattern,
        resume_keep_if,
    ) = _resolve_resume(raw)
    output_config = _resolve_output(raw)

    return ExperimentRun(
        name=name,
        project_file=path.resolve(),
        spec=spec,
        benchmark_dir=benchmark_dir,
        benchmark=benchmark,
        sample_limit=sample_limit,
        sample_slice=sample_slice,
        agents=agents,
        models=models,
        judge=judge,
        hooks=hooks,
        proxy=proxy,
        execution=execution,
        logging=logging_config,
        target=target_config,
        output=output_config,
        admission=admission_config,
        run_id=str(project_raw.get("run_id") or "").strip(),
        resume_retry_reasons=resume_retry_reasons,
        resume_max_attempts=resume_max_attempts,
        resume_select_id_pattern=resume_select_id_pattern,
        resume_keep_if=resume_keep_if,
        metadata={
            **raw.get("metadata", {}),
            "preflight": raw.get("preflight", []),
        },
    )


def _build_agent_instance(
    acfg: dict[str, Any],
    agent_type: Any,
    model: ModelConfig,
    model_sources: list[ModelConfig] | None = None,
    source_concurrency: dict[str, int] | None = None,
) -> AgentInstance:
    """Build an AgentInstance from YAML config + resolved model.

    The agent type's :meth:`validate_auth` runs here so misconfigured
    credentials surface at config-load time, not after the trial loop has
    already started spending docker resources. For a multi-source run every
    source is validated, since any of them may serve a trial.
    """
    for m in (model_sources or [model]):
        agent_type.validate_auth(m)
    return AgentInstance(
        agent_type=agent_type,
        model=model,
        model_sources=list(model_sources or []),
        source_concurrency=dict(source_concurrency or {}),
        id=acfg.get("id", ""),
        home=acfg.get("home", "/home/agent/workspace"),
        session_args=acfg.get("session_args", []),
        shared_paths=acfg.get("shared_paths", []),
        skill=acfg.get("skill", ""),
        plugins=acfg.get("plugins", []),
        extra_env=acfg.get("extra_env", {}),
        version=acfg.get("version", "latest"),
        image=acfg.get("image", ""),
        # Agent absent ⇒ None (inherit the runtime budget); "unlimited" / -1 / N
        # otherwise. Resolved against runtime + sample by resolve_max_rounds().
        max_rounds=normalize_max_rounds_config(acfg.get("max_rounds")),
        # Unset ⇒ None ⇒ each agent keeps its CLI's own default compaction
        # behaviour. Cage only overrides when the user explicitly opts in.
        context_compaction_threshold=(
            None
            if acfg.get("context_compaction_threshold") is None
            else float(acfg["context_compaction_threshold"])
        ),
        max_concurrent=int(acfg.get("max_concurrent", 0) or 0),
    )


def _parse_source_entry(entry: Any) -> tuple[str, int | None]:
    """Normalize one ``sources`` entry to ``(model_id, concurrency|None)``.

    Accepts ``"glm-5.1-w4a8"``, the suffixed ``"glm-5.1-w4a8:6"`` (CLI form),
    or ``{id: glm-5.1-w4a8, concurrency: 6}`` (yaml form). ``None`` concurrency
    means "take the default even share of the run's total".
    """
    if isinstance(entry, dict):
        sid = str(entry.get("id") or entry.get("model") or "").strip()
        conc = entry.get("concurrency")
        return sid, (int(conc) if conc not in (None, "") else None)
    text = str(entry).strip()
    if ":" in text:
        head, _, tail = text.rpartition(":")
        if head.strip() and tail.strip().isdigit():
            return head.strip(), int(tail)
    return text, None


def _agent_model_entries(
    raw: Any,
) -> list[tuple[str, list[str], dict[str, int], dict[str, Any]]]:
    """Normalize ``agents[].models`` to ``(model_id, sources, source_conc, overrides)``.

    Each entry is one independent ``agent×model`` run (case 1). A dict entry may
    additionally declare ``sources`` — registered model ids the run round-robins
    across per trial (case 2, load balancing). When ``sources`` is set,
    ``model_id`` is taken from an explicit ``id`` and is a *logical key*: it
    groups the run (run dir / labels / scores) and need not itself be registered.
    A source may carry a per-source concurrency (``id:N`` or ``{id, concurrency}``);
    ``source_conc`` maps source id → that cap (absent = default even share).
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("agents[].models must be a non-empty list")
    entries: list[tuple[str, list[str], dict[str, int], dict[str, Any]]] = []
    for item in raw:
        sources: list[str] = []
        source_conc: dict[str, int] = {}
        if isinstance(item, str):
            model_id = item
            overrides: dict[str, Any] = {}
        elif isinstance(item, dict):
            raw_sources = item.get("sources") or []
            if raw_sources and not isinstance(raw_sources, list):
                raise ValueError(
                    "agents[].models[].sources must be a list of model ids"
                )
            for s in raw_sources:
                sid, conc = _parse_source_entry(s)
                if not sid:
                    continue
                sources.append(sid)
                if conc is not None:
                    source_conc[sid] = conc
            if sources and not str(item.get("id") or "").strip():
                raise ValueError(
                    "an agents[].models entry with `sources` must set an "
                    "explicit `id` — the logical model key (e.g. id: glm-5.1) "
                    "that groups the run while the sources supply endpoints"
                )
            model_id = str(item.get("id") or item.get("model") or "")
            overrides = {
                str(k): v for k, v in item.items()
                if str(k) not in {"id", "model", "sources"}
            }
        else:
            model_id = ""
            overrides = {}
        model_id = str(model_id).strip()
        if not model_id:
            raise ValueError("agents[].models entries must be model ids")
        entries.append((model_id, sources, source_conc, overrides))
    return entries


def _resolve_model_sources(
    logical_id: str,
    source_ids: list[str],
    models: dict[str, ModelConfig],
    models_file: Any,
) -> list[ModelConfig]:
    """Resolve a multi-source entry's source ids to registered ModelConfigs.

    Validates every source is registered and that the pool shares one protocol
    (the run translates uniformly, so anthropic/openai endpoints can't mix).
    """
    resolved: list[ModelConfig] = []
    for sid in source_ids:
        if sid not in models:
            raise ValueError(
                f"Model source '{sid}' for logical model '{logical_id}' not "
                f"found in {models_file}"
            )
        resolved.append(models[sid])
    protocols = {m.protocol for m in resolved}
    if len(protocols) > 1:
        raise ValueError(
            f"Model '{logical_id}' sources must share one protocol; got "
            f"{sorted(protocols)} across {source_ids}"
        )
    return resolved
