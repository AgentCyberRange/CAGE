"""Generic adapter for a user-authored *custom agent* (manifest-driven).

One ``CustomAgent`` instance carries one parsed ``agent.yml``. It implements
the ``AgentType`` contract entirely from manifest data:

* :meth:`setup_container` copies the agent's self-contained source dir into the
  container (``docker cp``, the same pattern Cage uses for ``sidecar.py``) —
  no image rebuild when you edit the code, and the process stays sealed in
  Docker (nothing bind-mounted).
* :meth:`build_launch_command` fills ``{token}`` placeholders in the manifest's
  command from the resolved model + proxy + prompt, so Cage controls every run
  parameter (model name, endpoint, key, round budget, …) and switching models
  needs no edit to the command.
* :meth:`env_vars` fills the manifest's ``env`` map the same way.
* :meth:`parse_output` reads the answer per the manifest's ``output`` spec.

This is the only Python in the framework for custom agents — it knows no agent
name. Each actual agent is pure config + its own code.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from cage.agents.base import AgentType
from cage.agents.base.output import failure_banner
from cage.agents.base.resources import AgentContainerResources
from cage.agents.custom.manifest import CustomManifest, load_manifest
from cage.models import ModelConfig
from cage.sandbox.exec import ExecResult

# The agent's own code is copied here (outside the workspace, so the per-trial
# reset_directory never wipes it). The benchmark workspace stays at _WORKSPACE.
_SRC_DIR = "/opt/cage-agent/src"
_WORKSPACE = "/home/agent/workspace"
_TOKEN = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


def load_custom_agent(
    source: str, base_dir: Any, params: dict[str, Any] | None = None,
) -> "CustomAgent":
    """Build a CustomAgent from the manifest at ``source`` (rel. to base_dir).

    ``params`` (from the experiment yaml's ``agents[].params`` / ``cage run
    --param``) override the manifest's own ``params`` defaults.
    """
    return CustomAgent(load_manifest(source, base_dir), params=params)


class CustomAgent(AgentType):
    """A custom agent: copy its source in, run its templated command, read stdout."""

    dockerfile = ""

    def __init__(
        self, manifest: CustomManifest, params: dict[str, Any] | None = None,
    ) -> None:
        self.manifest = manifest
        # Instance attrs shadow the AgentType class attrs.
        self.name = manifest.name
        self.default_image = manifest.image
        self.state_paths = list(manifest.state_paths)
        # User placeholders: manifest defaults < experiment-yaml / CLI overrides.
        # Cage's reserved tokens overlay these in _base_tokens, so cage always wins.
        self.params: dict[str, str] = {
            **manifest.params,
            **{str(k): str(v) for k, v in (params or {}).items()},
        }

    # -- install: nothing; deps live in the manifest's image ----------------- #

    def version_command(self) -> str:
        # Non-empty, non-"unknown" output makes the runner skip CLI install.
        return "echo custom-agent"

    def install_command(self, version: str = "latest") -> str:
        return "true"

    def container_resources(
        self, *, home_dir: str, model: ModelConfig,
    ) -> AgentContainerResources:
        """Pre-start Docker resources — only the manifest's ``privileged`` flag.

        A custom agent that runs its own Docker daemon inside the container
        (Docker-in-Docker, e.g. the Cairn orchestrator) declares
        ``privileged: true`` in its manifest; Cage launches the trial container
        with ``--privileged`` so the inner daemon can start.
        """
        del home_dir, model
        return AgentContainerResources(privileged=self.manifest.privileged)

    # -- token substitution -------------------------------------------------- #

    def _base_tokens(
        self, *, model: ModelConfig, max_rounds: int, proxy_url: str,
    ) -> dict[str, str]:
        """The run-parameter surface Cage exposes to the command/env."""
        protocol = (model.protocol or "openai").lower()
        if proxy_url and protocol != "anthropic":
            base_url = proxy_url.rstrip("/") + "/v1"  # OpenAI-compatible base
        else:
            base_url = proxy_url
        # User-defined params first; cage's reserved tokens overlay them (win).
        tokens: dict[str, str] = dict(self.params)
        tokens.update({
            "model_name": model.model_name_for_agent(self.name) or model.model or "",
            "base_url": base_url,
            "api_key": model.api_key or "",
            "max_rounds": str(int(max_rounds)),
            "workspace_dir": _WORKSPACE,
        })
        # {model.<field>} passthrough so a model entry can drive per-model knobs.
        for fld in ("model", "base_url", "api_key", "provider", "timeout", "max_retries"):
            tokens[f"model.{fld}"] = str(getattr(model, fld, "") or "")
        extra = getattr(model, "extra", None)
        if isinstance(extra, dict):
            for key, val in extra.items():
                tokens[f"model.extra.{key}"] = "" if val is None else str(val)
        return tokens

    def _subst(self, template: str, tokens: dict[str, str]) -> str:
        def repl(match: re.Match) -> str:
            key = match.group(1)
            if key not in tokens:
                raise ValueError(
                    f"custom agent {self.name!r}: unknown placeholder {{{key}}} in "
                    f"manifest; available: {sorted(tokens)}"
                )
            return tokens[key]

        return _TOKEN.sub(repl, template)

    # -- launch / env / output ---------------------------------------------- #

    def build_launch_command(
        self, prompt: str, *, model: ModelConfig, max_rounds: int = -1, proxy_url: str = "",
    ) -> str:
        tokens = self._base_tokens(model=model, max_rounds=max_rounds, proxy_url=proxy_url)
        tokens["task_instruction"] = shlex.quote(prompt)  # safe to drop into argv
        cmd = self._subst(self.manifest.command, tokens)
        wd = self.manifest.workdir
        workdir = _SRC_DIR if wd in (".", "") else f"{_SRC_DIR}/{wd.lstrip('/')}"
        inner = f"cd {shlex.quote(workdir)} && {cmd}"
        # Wrap in a subshell. The trial runner prepends Cage's env exports
        # (``OPENAI_BASE_URL=… CAGE_TRACE=1 … <launch_command>``); a launch command
        # that *began* with ``cd`` would consume those exports — shell applies a
        # ``VAR=val`` prefix only to the single following command (the ``cd``), so
        # the agent process after ``&&`` would run WITHOUT them (no proxy URL, no
        # api key, no trace). Running cd+command inside ``bash -c`` makes the
        # exports apply to ``bash``, which the agent process then inherits.
        return f"bash -c {shlex.quote(inner)}"

    def env_vars(
        self,
        *,
        proxy_url: str,
        model: ModelConfig,
        container: Any = None,
        home_dir: str = "/home/agent",
        workspace_dir: str = "",
        max_rounds: int = -1,
        context_compaction_threshold: float | None = None,
    ) -> dict[str, str]:
        del container, home_dir, workspace_dir, context_compaction_threshold
        tokens = self._base_tokens(model=model, max_rounds=max_rounds, proxy_url=proxy_url)
        # Enable the base image's LangChain hook, which stamps the current
        # LangGraph node onto each model request as X-Cage-* headers (the proxy
        # records them in proxy.jsonl). Harmless no-op for non-LangChain agents.
        env = {"CAGE_TRACE": "1"}
        env.update({k: self._subst(v, tokens) for k, v in self.manifest.env.items()})
        return env

    def parse_output(self, result: ExecResult) -> str:
        banner = failure_banner(result)
        if banner is not None:
            return banner
        out = result.stdout.strip()
        spec = self.manifest.output
        if spec.get("type") == "json_field":
            try:
                obj = json.loads(out)
                value = obj.get(spec["field"]) if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                value = None
            if value is not None:
                return value if isinstance(value, str) else json.dumps(value)
        return out or result.stdout[:2000]

    # -- code delivery ------------------------------------------------------- #

    def setup_container(
        self, container: Any, *, home_dir: str, model: ModelConfig | None = None,
    ) -> None:
        """Copy the agent's self-contained source into the container.

        Runs once per container. The source path is on ``self`` (this agent was
        built from its manifest), so no per-trial threading is needed. Lands at
        :data:`_SRC_DIR`, outside the workspace, owned by the agent user.
        """
        del home_dir, model
        container.exec(f"mkdir -p {_SRC_DIR}", timeout=10.0)
        container.copy_to(self.manifest.source_dir.rstrip("/") + "/.", _SRC_DIR)
        container.exec("chown -R agent:agent /opt/cage-agent", timeout=30.0)
