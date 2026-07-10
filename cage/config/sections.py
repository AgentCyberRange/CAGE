"""Parsed configuration sections for an experiment.

These are the cohesive value objects ``project.yml`` deserializes into — proxy,
target, execution, live-check, output, judge, and resume policy. They are plain
dataclasses with no engine dependencies, so both the loader
(:mod:`cage.config.experiment`) and the live run context
(:class:`cage.experiment.engine.run_context.ExperimentRun`) can hold them
without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cage.models import ModelConfig


@dataclass
class JudgeConfig:
    """Judge model configuration for scoring."""

    model: ModelConfig
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class ProxyConfig:
    """Proxy configuration."""

    enabled: bool = True
    rewrite_system: str = ""  # template with {{ system_raw }}
    request_timeout: float = 3600.0  # upstream request timeout in seconds
    upstream_http_proxy: str = ""  # HTTP proxy used by the in-container upstream proxy


@dataclass
class TargetConfig:
    """Target / target_server server configuration.

    Drives the lifecycle of the target stack each trial talks to: how the
    agent attaches to the runtime network, SSH tunnelling, and per-experiment
    isolation knobs.

    Note: target_server is **always** spawned per-run by the orchestrator
    (``spawn_embedded_target_server``). Its URL is an internal runtime value
    and is not configurable from project.yml — it's set programmatically on
    the ChallengeClient after the embedded subprocess starts.
    """

    enabled: bool = True
    run_mode: str = "remote"  # 'local' | 'remote'
    # ``server_url`` is set at runtime by the orchestrator from the embedded
    # target_server's chosen port; do not write to it from yaml.
    server_url: str = ""
    use_ssh_tunnel: bool = False
    # SSH tunnelling defaults are empty — opt in via project.yml ``target.ssh:``.
    jump_host: str = ""
    jump_user: str = ""
    ssh_key_path: str = ""
    remote_bind_address: str = "127.0.0.1"
    remote_bind_port: int = 8000
    # Forbidden if set true (config resolution raises). Agents reach targets over
    # the isolated docker network, never host-published ports. Kept False-only so
    # the field still resolves for the client that reads it.
    use_external_access: bool = False
    # Empty default — set this explicitly when the agent must reach a specific
    # gateway IP instead of ``host.docker.internal``.
    host_ip_for_agent: str = ""
    network_name: str = "cage_net"
    # Embedded target_server launch limits. ``None`` means keep the server
    # default; project.yml can raise these for large compose stacks.
    startup_timeout: float | None = None
    compose_up_timeout: float | None = None
    # Per-experiment isolation policy. Passed through to target_server via the
    # ChallengeClient runtime args. Empty string = let the server pick the
    # default for the benchmark family.
    #   target_scope:  "per_agent" (default) | "per_challenge" | ""
    #   parallel_mode: ""           (default — server picks "network" for
    #                                compose-network benchmarks, "alias" otherwise)
    #                | "network"    (each agent gets its own L3 subnet)
    #                | "alias"      (agents share a network, route by alias)
    target_scope: str = "per_agent"
    parallel_mode: str = ""
    # Network-namespace isolation between the agent container and the
    # challenge's *internal* services (databases, secrets initialisers,
    # caches, …).
    #   "per_trial_bridge" (default): cage creates a per-trial bridge,
    #       connects only services flagged public by target_server
    #       (``external_port`` set), and attaches the agent only there.
    #       Internal services on the compose project's own networks stay
    #       unreachable regardless of how the adapter configured them.
    #   "trust_server": legacy behaviour — agent attaches to the network
    #       name returned by target_server as-is.
    agent_network_isolation: str = "per_trial_bridge"


@dataclass
class ReactiveLiveCheckConfig:
    """Configuration for agent-triggered live checks."""

    enabled: bool = True
    check_on_submit: bool = True
    check_on_9091_call: bool = True


@dataclass
class PollingLiveCheckConfig:
    """Configuration for orchestrator-driven live check polling."""

    enabled: bool = False
    interval_seconds: float = 5.0
    stop_on_success: bool = True
    # Require this many CONSECUTIVE successful polls before locking in
    # a live-success verdict. Defends against transient validator flips
    # (target slowdowns, restarts, network jitter) that briefly mark the
    # exploit as "successful" while the agent has done nothing.
    # 1 = legacy single-poll lock-in (back-compat). 2-3 is the audited
    # audited default for spurious-trigger resistance. Benchmarks can also override per-verdict
    # via :meth:`Benchmark.live_check_confirm_polls`.
    confirm_polls: int = 1


@dataclass
class LiveCheckConfig:
    """Configuration for in-container live answer checking."""

    enabled: bool = False
    max_calls: int = 3
    stop_on_success: bool = True
    reactive: ReactiveLiveCheckConfig = field(default_factory=ReactiveLiveCheckConfig)
    polling: PollingLiveCheckConfig = field(default_factory=PollingLiveCheckConfig)


@dataclass
class ExecutionConfig:
    """Execution parameters."""

    # Global cap on simultaneously in-flight trials across the whole run.
    # 0 = unlimited (no global cap); the per-agent ``max_concurrent`` still
    # applies and is the usual control.
    max_trials_global: int = 0
    # Max concurrent target stack launches/readiness waits. 1 serializes heavy
    # docker compose setup while still allowing already-ready agents to run.
    # 0 disables this separate setup cap.
    max_target_setups: int = 1
    timeout: float = 0.0  # per-trial timeout in seconds (0 = unlimited)
    on_failure: str = "continue"
    chunk_size: int | None = None
    agent_network_mode: str | None = "host"
    # Config space: "unlimited" (the default; no round cap) | -1 (use the
    # benchmark sample default) | 0 (no rounds) | N. Resolved by
    # resolve_max_rounds(); an unlimited budget needs another stop condition.
    max_rounds: int | str | None = "unlimited"
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost: float | None = None
    store_proxy: bool = False
    # Poll self-hosted (vllm/sglang) model endpoints until they answer before
    # starting — for remotely-launched servers with unknown boot time. The CLI
    # ``--wait-for-model`` flag overrides this. ``wait_timeout`` of 0 waits
    # indefinitely.
    wait_for_model: bool = False
    wait_timeout: float = 0.0
    wait_interval: float = 10.0
    live_check: LiveCheckConfig = field(default_factory=LiveCheckConfig)
    # Number of independent attempts per sample. >1 turns the run into a
    # pass@k evaluation: the trial sequence is replayed k times in
    # ``[pass1 over all samples, pass2 over all samples, ...]`` order, so a
    # full pass completes before the next pass begins (even under
    # parallelism — passes are batched).
    passk: int = 1
    # Per-invocation execution cap: run only trials whose global index is
    # ``< max_trial`` this run, leaving the rest pending. Trials are ordered
    # pass-major (pass_1 over all samples, then pass_2, …), so e.g. with 24
    # samples ``max_trial=48`` runs pass_1+pass_2 and defers pass_3. Unlike
    # lowering ``passk``, this does NOT shrink the recorded trial plan, so
    # ``--resume`` stays plan-compatible and a later capless run finishes the
    # remainder. ``None`` = no cap. Also handy as a quota throttle.
    max_trial: int | None = None


@dataclass
class OutputConfig:
    """Controls what goes into dashboard.json and results.csv.

    All flags default to True (full output). Set to False in project.yml
    ``output:`` section to exclude heavy fields.

    Example project.yml::

        output:
          dashboard_prompt: false
          dashboard_output: false
          dashboard_reasoning: false
          csv_prompt: false
          csv_output: false
          csv_reasoning: false
    """

    dashboard_prompt: bool = True
    dashboard_output: bool = True
    dashboard_reasoning: bool = True
    csv_prompt: bool = True
    csv_output: bool = True
    csv_reasoning: bool = True


@dataclass
class ResumeKeepIf:
    """Veto thresholds for ``--resume``: a trial that is otherwise eligible to
    be re-run (failed with a retry reason / missing result) is instead KEPT
    (its prior result replays from disk) when **any** of these match its
    on-disk evidence. Empty = no veto (legacy behaviour).

    Configured under ``resume.keep_if`` in project.yml. The intent is
    "this attempt already did enough work — don't throw it away just because
    it ended on a retryable error":

        resume:
          keep_if:
            min_rounds: 100        # ran >= 100 agent rounds (progress.json)
            min_duration_s: 1800   # ran >= 30 min wall-clock (meta.timing)
            id_matches: "range-8"  # trial_id matches this regex
    """

    min_rounds: int | None = None
    min_duration_s: float | None = None
    id_matches: str | None = None

    def is_empty(self) -> bool:
        return (
            self.min_rounds is None
            and self.min_duration_s is None
            and not self.id_matches
        )
