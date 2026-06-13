"""Plain terminal progress output for ``cage run``."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from cage.contracts.execution import max_rounds_config_label, resolve_max_rounds
from cage.contracts.sample_keys import SAMPLE_MAX_ROUNDS_KEY
from cage.contracts.style import should_color, style

_RUN_BANNER = (
    "   _________   ____________   ____  __  ___   __",
    "  / ____/   | / ____/ ____/  / __ \\/ / / / | / /",
    " / /   / /| |/ / __/ __/    / /_/ / / / /  |/ /",
    "/ /___/ ___ / /_/ / /___   / _, _/ /_/ / /|  /",
    "\\____/_/  |_\\____/_____/  /_/ |_|\\____/_/ |_/",
)


def format_run_banner(*, color: bool = False) -> str:
    return "\n".join(
        style(line, "cyan", "bold", enabled=color)
        for line in _RUN_BANNER
    )


def print_run_banner(*, stream: Any | None = None) -> None:
    output_stream = stream or sys.stderr
    print(
        format_run_banner(color=should_color(output_stream)),
        file=output_stream,
        flush=True,
    )
    print(file=output_stream, flush=True)


@dataclass
class RunAgentContract:
    agent_id: str = ""
    label: str = ""
    kind: str = ""
    model_id: str = ""
    provider: str = ""
    model_name: str = ""
    agent_model_name: str = ""
    image: str = ""
    max_concurrent: int | str = ""
    max_rounds: str = ""
    session_args: str = ""


@dataclass
class RunViewLink:
    label: str = ""
    base_url: str = ""
    run_url: str = ""
    dashboard_url: str = ""


@dataclass
class RunContract:
    project_name: str = ""
    benchmark_id: str = ""
    benchmark_name: str = ""
    benchmark_path: str = ""
    project_file: str = ""
    run_id: str = ""
    run_dir: str = ""
    board_url: str = ""
    run_url: str = ""
    dashboard_url: str = ""
    inspect_command: str = ""
    run_log_path: str = ""
    debug_log_path: str = ""
    view_links: list[RunViewLink] = field(default_factory=list)
    planned_trials: int = 0
    runnable_trials: int = 0
    resume_replayed_trials: int = 0
    passk: int = 1
    levels: str = "n/a"
    samples: str = "all"
    max_trials_global: int = 0  # 0 = unlimited (no global cap)
    max_target_setups: int = 1
    target_enabled: bool = False
    launch_build_policy: str = "disabled; run cage benchmark build before cage run"
    trial_timeout_s: int | float | str = "unlimited"
    request_timeout_s: int | float | str = "n/a"
    target_startup_timeout_s: int | float | str = "n/a"
    target_compose_timeout_s: int | float | str = "n/a"
    effective_max_rounds: str = "n/a"
    max_input_tokens: str = "n/a"
    max_output_tokens: str = "n/a"
    judge_max_tokens: str = "n/a"
    max_cost: str = "n/a"
    agents: list[RunAgentContract] = field(default_factory=list)

    def to_plain_text(self, *, color: bool = False) -> str:
        return format_run_parameter_text(self, color=color)


def build_run_contract(
    *,
    config: Any,
    run_id: str,
    run_dir: Any,
    board_url: str = "",
    run_url: str = "",
    dashboard_url: str = "",
    planned_trials: int,
    runnable_trials: int,
    resume_replayed_trials: int = 0,
    samples: list[dict[str, Any]],
    inspect_command: str = "",
    view_links: list[Any] | None = None,
) -> RunContract:
    benchmark = getattr(config, "benchmark", None)
    benchmark_dir = getattr(config, "benchmark_dir", None)
    axes_fn = getattr(benchmark, "variant_display_axes", None)
    variant_axes = axes_fn() if callable(axes_fn) else {}
    hint_levels = _join_hint_levels(variant_axes.get("hint"))
    prompt_levels = _join_optional_sequence(variant_axes.get("prompt"))
    levels = hint_levels or prompt_levels or "n/a"
    selected_samples = _join_optional_sequence(getattr(config, "sample_ids", ()) or None)
    if not selected_samples and len(samples) <= 5:
        selected_samples = _join_optional_sequence(
            [sample.get("id") or sample.get("sample_id") for sample in samples]
        )
    agents = [_agent_contract(agent) for agent in getattr(config, "agents", []) or []]
    metadata = getattr(config, "metadata", {}) or {}
    resolved_inspect_command = inspect_command or _default_inspect_command(
        benchmark_dir or getattr(config, "project_file", None)
    )
    return RunContract(
        project_name=str(getattr(config, "name", "") or ""),
        benchmark_id=str(metadata.get("benchmark_id", "") or ""),
        benchmark_name=str(getattr(benchmark, "name", type(benchmark).__name__) or ""),
        benchmark_path=str(benchmark_dir or ""),
        project_file=str(getattr(config, "project_file", "") or ""),
        run_id=run_id,
        run_dir=str(run_dir),
        board_url=board_url,
        run_url=run_url,
        dashboard_url=dashboard_url,
        inspect_command=resolved_inspect_command,
        run_log_path=str(Path(run_dir) / ".cage.runlog") if run_dir else "",
        debug_log_path=(
            str(Path(run_dir) / ".cage.debuglog")
            if bool(getattr(getattr(config, "logging", None), "debug_file_enabled", False))
            and run_dir
            else ""
        ),
        view_links=_coerce_view_links(view_links, run_url, dashboard_url),
        planned_trials=planned_trials,
        runnable_trials=runnable_trials,
        resume_replayed_trials=resume_replayed_trials,
        passk=max(1, int(getattr(getattr(config, "execution", None), "passk", 1) or 1)),
        levels=levels,
        samples=selected_samples or "all",
        max_trials_global=max(0, int(getattr(getattr(config, "execution", None), "max_trials_global", 0) or 0)),
        max_target_setups=int(
            getattr(getattr(config, "execution", None), "max_target_setups", 1) or 0
        ),
        target_enabled=bool(getattr(getattr(config, "target", None), "enabled", False)),
        launch_build_policy=_launch_build_policy(metadata),
        trial_timeout_s=_format_timeout(
            getattr(getattr(config, "execution", None), "timeout", 0)
        ),
        request_timeout_s=_format_timeout(
            getattr(getattr(config, "proxy", None), "request_timeout", "")
        ),
        target_startup_timeout_s=_format_timeout(
            getattr(getattr(config, "target", None), "startup_timeout", "")
        ),
        target_compose_timeout_s=_format_timeout(
            getattr(getattr(config, "target", None), "compose_up_timeout", "")
        ),
        effective_max_rounds=_effective_max_rounds(config, samples),
        max_input_tokens=_format_limit(
            getattr(getattr(config, "execution", None), "max_input_tokens", None)
        ),
        max_output_tokens=_format_limit(
            getattr(getattr(config, "execution", None), "max_output_tokens", None)
        ),
        judge_max_tokens=str(
            getattr(getattr(config, "judge", None), "max_tokens", "n/a") or "n/a"
        ),
        max_cost=_max_cost(config),
        agents=agents,
    )
















def format_run_parameter_text(contract: RunContract, *, color: bool = False) -> str:
    agent = contract.agents[0] if contract.agents else RunAgentContract()
    agent_rows = [
        ("agent", agent.agent_id or agent.label or "n/a"),
        ("agent label", agent.label or "n/a"),
        ("agent kind", agent.kind or "n/a"),
        ("model", agent.model_id or "n/a"),
        ("provider", agent.provider or "n/a"),
        ("endpoint model", agent.model_name or "n/a"),
    ]
    if agent.agent_model_name and agent.agent_model_name != agent.model_name:
        agent_rows.append(("agent model", agent.agent_model_name))
    agent_rows.extend(
        [
            ("concurrency", _format_run_concurrency(contract, agent)),
            ("image", agent.image or "n/a"),
            ("session args", agent.session_args or "none"),
        ]
    )
    grouped_rows: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Browser inspector",
            _inspector_rows(contract),
        ),
        (
            "Run selection",
            _run_selection_rows(contract),
        ),
        (
            "Target lifecycle",
            _target_lifecycle_rows(contract),
        ),
        (
            "Stop conditions",
            [
                ("max rounds", contract.effective_max_rounds or "n/a"),
                ("trial timeout", str(contract.trial_timeout_s)),
                ("request timeout", str(contract.request_timeout_s)),
                (
                    "target timeout",
                    (
                        f"startup={contract.target_startup_timeout_s}, "
                        f"compose={contract.target_compose_timeout_s}"
                    ),
                ),
                ("max input tokens", contract.max_input_tokens or "n/a"),
                ("max output tokens", contract.max_output_tokens or "n/a"),
                ("max cost", contract.max_cost or "n/a"),
                ("stop run", "Ctrl-C"),
            ],
        ),
        (
            "Agent / model",
            agent_rows,
        ),
        (
            "Logs",
            _log_rows(contract),
        ),
    ]
    if contract.effective_max_rounds == "0":
        grouped_rows.insert(
            4,
            (
                "Zero-round mode",
                [
                    (
                        "target setup",
                        (
                            "enabled; targets still launch and tear down"
                            if contract.target_enabled
                            else "disabled"
                        ),
                    ),
                    ("agent/proxy/model", "skipped; no model calls"),
                ],
            ),
        )
    rows = [row for _heading, section_rows in grouped_rows for row in section_rows]
    width = max(len(label) for label, _value in rows)
    lines: list[str] = [
        style(
            "Pre-flight checks passed. Cage is entering the benchmark run.",
            "green",
            "bold",
            enabled=color,
        ),
        style(
            "Open a browser URL below for live trials, logs, artifacts, and scores.",
            "blue",
            enabled=color,
        ),
    ]
    for heading, section_rows in grouped_rows:
        lines.append("")
        lines.append(style(heading, "cyan", "bold", enabled=color))
        lines.extend(
            (
                f"{style(label.rjust(width), 'dim', enabled=color)}: "
                f"{_style_row_value(label, value, color=color)}"
            )
            for label, value in section_rows
        )
    return "\n".join(lines)


def _style_row_value(label: str, value: str, *, color: bool) -> str:
    if not color:
        return value
    normalized = label.strip().lower()
    if normalized.endswith("url"):
        return style(value, "blue", "bold", enabled=True)
    if normalized in {"benchmark", "run id", "samples", "trials to run"}:
        return style(value, "green", "bold", enabled=True)
    if normalized in {"max cost", "max input tokens", "max output tokens", "max rounds"}:
        return style(value, "yellow", enabled=True)
    if normalized in {"debug log", "run log", "run dir"}:
        return style(value, "dim", enabled=True)
    return value


def _inspector_rows(contract: RunContract) -> list[tuple[str, str]]:
    urls = _inspector_browser_urls(contract)
    if not urls:
        return [("web", "disabled")]
    rows: list[tuple[str, str]] = [
        ("open browser", "live trials, logs, artifacts, and scores"),
    ]
    for url in urls:
        host = _host_from_url(url)
        if host in {"127.0.0.1", "localhost", "::1"}:
            label = "local url"
        elif host in {"0.0.0.0", "::"}:
            label = "bind url"
        else:
            label = "network url"
        rows.append((label, url))
    return rows


def _run_selection_rows(contract: RunContract) -> list[tuple[str, str]]:
    levels = contract.levels or ""
    samples = contract.samples or ""
    benchmark_label = (
        contract.benchmark_id
        or contract.benchmark_name
        or contract.project_name
        or "n/a"
    )
    rows = [
        ("benchmark", benchmark_label),
    ]
    if (
        contract.benchmark_id
        and contract.benchmark_name
        and contract.benchmark_name != contract.benchmark_id
    ):
        rows.append(("suite", contract.benchmark_name))
    runnable_detail = (
        f"{contract.runnable_trials} runnable after filters, caps, and resume replay"
        if contract.resume_replayed_trials
        else f"{contract.runnable_trials} runnable after filters and caps"
    )
    rows.extend([
        ("project", contract.project_name or "n/a"),
        ("run id", contract.run_id or "n/a"),
        ("samples", samples if samples and samples != "n/a" else "all selected samples"),
        ("planned trials", f"{contract.planned_trials} total before filters and caps"),
        ("trials to run", runnable_detail),
        ("pass@k attempts", f"{contract.passk} per sample"),
        ("prompt/hint levels", levels if levels and levels != "n/a" else "benchmark default"),
    ])
    if contract.resume_replayed_trials:
        rows.append(("resume replayed", f"{contract.resume_replayed_trials} kept from disk"))
    return rows


def _target_lifecycle_rows(contract: RunContract) -> list[tuple[str, str]]:
    return [
        ("target server", "enabled" if contract.target_enabled else "disabled"),
        ("launch build", contract.launch_build_policy),
        ("target setup cap", f"max_target_setups={contract.max_target_setups}"),
    ]


def _log_rows(contract: RunContract) -> list[tuple[str, str]]:
    run_dir = str(contract.run_dir or "")
    run_log = str(contract.run_log_path or "")
    debug_log = str(contract.debug_log_path or "")
    if not run_log and run_dir:
        run_log = str(Path(run_dir) / ".cage.runlog")
    debug_value = debug_log or "disabled; set logging.debug_file: true in project.yml"
    return [
        ("terminal output", "summary plus one progress line; details stay in web/logs"),
        ("run dir", run_dir or "n/a"),
        ("run log", run_log or "n/a"),
        ("debug log", debug_value),
        ("proxy details", "written to run log; debug file follows project.yml logging"),
    ]


def browser_urls_from_summary(summary: dict[str, Any]) -> list[str]:
    """Inspector browser URL(s) for a finished run, from its summary dict.

    The conductor stashes the managed board's ``board_url`` / ``run_url`` /
    ``view_links`` on the run summary. Rebuild a minimal contract from them and
    reuse the launch-banner URL selection (local, bind, one network) so the
    end-of-run reminder shows exactly the URLs the operator saw at start.
    """
    view_links = [
        RunViewLink(
            label=str(item.get("label", "") or ""),
            base_url=str(item.get("base_url", "") or ""),
            run_url=str(item.get("run_url", "") or ""),
            dashboard_url=str(item.get("dashboard_url", "") or ""),
        )
        for item in (summary.get("view_links") or [])
        if isinstance(item, dict)
    ]
    contract = RunContract(
        board_url=str(summary.get("board_url", "") or ""),
        run_url=str(summary.get("run_url", "") or ""),
        dashboard_url=str(summary.get("dashboard_url", "") or ""),
        view_links=view_links,
    )
    return _inspector_browser_urls(contract)


def _inspector_browser_urls(contract: RunContract) -> list[str]:
    raw_urls: list[str] = []
    for link in contract.view_links:
        run_url = str(getattr(link, "run_url", "") or "")
        if run_url:
            raw_urls.append(run_url)
            continue
        base_url = str(getattr(link, "base_url", "") or "")
        if base_url:
            raw_urls.append(
                _url_on_base(
                    contract.run_url or contract.dashboard_url or base_url,
                    base_url,
                )
            )
    if not raw_urls:
        raw_urls.extend(
            url
            for url in (
                contract.run_url,
                contract.dashboard_url,
                contract.board_url,
            )
            if url
        )
    result: list[str] = []
    seen: set[str] = set()
    for url in raw_urls:
        display_url = str(url or "")
        if display_url and display_url not in seen:
            seen.add(display_url)
            result.append(display_url)
    return _limit_inspector_urls(result)


def _url_on_base(url: str, base_url: str) -> str:
    split = urlsplit(url)
    base = urlsplit(base_url)
    if not split.scheme or not split.netloc or not base.scheme or not base.netloc:
        return url or base_url
    return urlunsplit((base.scheme, base.netloc, split.path, split.query, split.fragment))


def _limit_inspector_urls(urls: list[str]) -> list[str]:
    """Keep the terminal hint compact and *connectable*: one network URL first
    (what a remote operator opens), then the loopback fallback.

    A wildcard ``0.0.0.0`` / ``::`` bind address is dropped entirely — a browser
    cannot connect to it, so showing it as a URL only misleads operators into
    copying a link that refuses to connect.
    """
    selected: dict[str, str] = {}
    for url in urls:
        host = _host_from_url(url)
        if host in {"0.0.0.0", "::"}:
            continue
        if host in {"127.0.0.1", "localhost", "::1"}:
            key = "local"
        else:
            key = "network"
        selected.setdefault(key, url)
    return [
        selected[key]
        for key in ("network", "local")
        if key in selected
    ]


def _coerce_view_links(
    links: list[Any] | None,
    run_url: str,
    dashboard_url: str,
) -> list[RunViewLink]:
    if links:
        coerced: list[RunViewLink] = []
        for item in links:
            coerced.append(
                RunViewLink(
                    label=_normalize_view_label(str(getattr(item, "label", "") or "")),
                    base_url=str(getattr(item, "base_url", "") or ""),
                    run_url=str(getattr(item, "run_url", "") or ""),
                    dashboard_url=str(getattr(item, "dashboard_url", "") or ""),
                )
            )
        return coerced
    if run_url or dashboard_url:
        return [
            RunViewLink(
                base_url=_base_from_url(run_url or dashboard_url),
                run_url=run_url or "",
                dashboard_url=dashboard_url or "",
            )
        ]
    return []


def _normalize_view_label(label: str) -> str:
    normalized = label.strip()
    lowered = normalized.lower()
    if lowered in {"localhost", "loopback"}:
        return "local"
    if lowered == "bind address":
        return "bind"
    if lowered.startswith("lan "):
        return f"network {normalized[4:]}"
    return normalized


def _default_inspect_command(path: Any) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    if candidate.is_file():
        candidate = candidate.parent
    return f"cage inspect {_display_path(candidate)}"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _format_run_concurrency(contract: RunContract, agent: RunAgentContract) -> str:
    # The scheduling unit is the trial (one sample × one pass attempt). Two caps
    # gate how many run AT ONCE: this agent's own max_concurrent, and the global
    # max_trials_global shared across all agents. The effective in-flight count is
    # the smaller of the two — lead with it so the banner isn't read as "16".
    try:
        per_agent = int(agent.max_concurrent or 0)
    except (TypeError, ValueError):
        per_agent = 0  # 'n/a' / unset → no per-agent cap to display
    g = int(contract.max_trials_global or 0)
    global_cap = g if g > 0 else "unlimited"
    if per_agent > 0 and g > 0:
        effective: object = min(per_agent, g)
    elif per_agent > 0:
        effective = per_agent
    elif g > 0:
        effective = g
    else:
        effective = "unlimited"
    return (
        f"{effective} trials in flight "
        f"(max_concurrent={per_agent or 'n/a'} per agent, "
        f"max_trials_global={global_cap}, "
        f"max_target_setups={contract.max_target_setups})"
    )


def _agent_contract(agent: Any) -> RunAgentContract:
    agent_type = getattr(agent, "agent_type", None)
    model = getattr(agent, "model", None)
    kind = str(getattr(agent_type, "name", type(agent_type).__name__) or "")
    endpoint_model_name = str(getattr(model, "model", "") or "")
    agent_model_name = endpoint_model_name
    if hasattr(model, "model_name_for_agent"):
        agent_model_name = str(model.model_name_for_agent(kind) or endpoint_model_name)
    label_fn = getattr(agent, "label", None)
    label = label_fn() if callable(label_fn) else ""
    session_args = " ".join(str(arg) for arg in (getattr(agent, "session_args", []) or []))
    return RunAgentContract(
        agent_id=str(getattr(agent, "id", "") or ""),
        label=label,
        kind=kind,
        model_id=str(getattr(model, "id", "") or ""),
        provider=str(getattr(model, "provider", "") or ""),
        model_name=endpoint_model_name,
        agent_model_name=agent_model_name,
        image=str(
            getattr(agent, "effective_image", "")
            or getattr(agent, "image", "")
            or getattr(agent_type, "default_image", "")
            or ""
        ),
        max_concurrent=getattr(agent, "max_concurrent", "") or "n/a",
        max_rounds=max_rounds_config_label(getattr(agent, "max_rounds", None)),
        session_args=session_args,
    )


def _join_optional_sequence(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        items = list(value)
    except TypeError:
        return str(value)
    return ",".join(str(item) for item in items if item is not None and str(item))


def _join_hint_levels(value: Any) -> str:
    raw = _join_optional_sequence(value)
    if not raw:
        return ""
    levels: list[str] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        levels.append(token if token.startswith("l") else f"l{token}")
    return ",".join(levels)


def _format_timeout(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number <= 0:
        return "unlimited"
    if number.is_integer():
        return f"{int(number)}s"
    return f"{number:.1f}s"


def _format_limit(value: Any) -> str:
    if value in (None, ""):
        return "unlimited"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number <= 0:
        return "unlimited"
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _effective_max_rounds(config: Any, samples: list[dict[str, Any]]) -> str:
    """Banner label for the round budget; ``-1`` (nothing set) shows "unlimited"."""

    values: set[int] = set()
    execution = getattr(config, "execution", None)
    for agent in list(getattr(config, "agents", []) or [object()]):
        for sample in samples or [{}]:
            resolved = resolve_max_rounds(
                getattr(agent, "max_rounds", None),
                getattr(execution, "max_rounds", "unlimited"),
                sample.get(SAMPLE_MAX_ROUNDS_KEY) if isinstance(sample, dict) else None,
            )
            if resolved >= 0:
                values.add(resolved)
    if not values:
        return "unlimited"
    labels = [str(value) for value in sorted(values)]
    return labels[0] if len(labels) == 1 else "mixed: " + ", ".join(labels)


def _max_cost(config: Any) -> str:
    execution = getattr(config, "execution", None)
    for attr in ("max_cost_usd", "cost_limit_usd", "max_cost"):
        value = getattr(execution, attr, None)
        if value not in (None, ""):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return str(value)
            if number <= 0:
                return "unlimited"
            return f"${number:.2f}"
    return "unlimited"


def _launch_build_policy(metadata: dict[str, Any]) -> str:
    policy = str((metadata or {}).get("launch_build") or "disabled").strip()
    if policy == "benchmark-hook":
        return "benchmark hook ran before target launch"
    return "disabled; run cage benchmark build before cage run"














def _base_from_url(url: str) -> str:
    parsed = urlsplit(str(url or ""))
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _host_from_url(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""
