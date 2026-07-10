"""Load + validate a custom-agent manifest (``agent.yml``).

A *custom agent* is a self-contained directory holding the agent's own code
plus an ``agent.yml`` describing how Cage runs it. The manifest is the entire
contract: a launch command (with ``{token}`` placeholders Cage fills), the
runtime image, optional env / output / workdir. No Python is implemented
against the framework — one generic :class:`~cage.agents.custom.agent.CustomAgent`
interprets any manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CustomManifest:
    """Parsed ``agent.yml`` plus the resolved absolute source directory."""

    source_dir: str  # absolute host path to the agent's self-contained dir
    name: str
    image: str
    command: str
    workdir: str = "."
    output: dict[str, Any] = field(default_factory=lambda: {"type": "stdout"})
    env: dict[str, str] = field(default_factory=dict)
    state_paths: list[str] = field(default_factory=list)
    # Launch the trial container with ``--privileged``. Custom agents that run
    # their own Docker daemon inside the container (Docker-in-Docker) set this.
    privileged: bool = False
    # Author-declared default values for the manifest's own ``{placeholders}``.
    # The experiment YAML's ``agents[].params`` and ``cage run --param`` override
    # these. The Cage-controlled tokens (prompt/model/base_url/api_key/
    # max_rounds/workspace/model.*) are reserved and always win.
    params: dict[str, str] = field(default_factory=dict)
    # Optional build recipe so ``cage agent build --agent <name>`` is the SINGLE
    # build entry point even for a custom agent whose image needs more than one
    # ``docker build`` (e.g. a multi-image Docker-in-Docker bake). Declare
    # ``build: {script: <path-relative-to-repo-root>}``; the build command runs
    # that script instead of the default single-image build.
    build: dict[str, str] = field(default_factory=dict)


# Reserved {tokens} Cage fills (see cage/agents/custom/agent.py::_base_tokens and
# build_launch_command). A manifest `param` may NOT reuse one of these names, nor
# the `model.` prefix: Cage owns these cross-cutting concepts and there is exactly
# ONE canonical way to set each — rounds via `max_rounds` / `--max-rounds`, model
# via the model config, etc. A param that duplicates one is a second, confusing
# knob for the same thing, so it's rejected at load time instead of silently lost.
RESERVED_TOKEN_NAMES = frozenset({
    "task_instruction", "model_name", "base_url", "api_key",
    "max_rounds", "workspace_dir",
})


def _reject_reserved_params(params: dict[str, str], manifest_path: Path) -> None:
    clashing = sorted(
        k for k in params if k in RESERVED_TOKEN_NAMES or k.startswith("model.")
    )
    if clashing:
        raise ValueError(
            f"{manifest_path}: params may not reuse Cage-reserved token name(s) "
            f"{clashing}. Cage fills these ({', '.join(sorted(RESERVED_TOKEN_NAMES))}, "
            f"model.*) — use the reserved {{token}} directly in `command` (e.g. map "
            f"rounds with {{max_rounds}}); keep `params` for your agent's own knobs."
        )


def load_manifest(source: str, base_dir: Path) -> CustomManifest:
    """Resolve ``source`` (relative to ``base_dir``) and parse its ``agent.yml``.

    Fails fast at config-load time (not mid-trial) on a missing dir, missing
    manifest, or missing required fields.
    """
    src = Path(source).expanduser()
    if not src.is_absolute():
        src = base_dir / src
    src = src.resolve()
    if not src.is_dir():
        raise ValueError(
            f"custom agent source is not a directory: {src} (declared {source!r})"
        )

    manifest_path = src / "agent.yml"
    if not manifest_path.is_file():
        alt = src / "agent.yaml"
        if alt.is_file():
            manifest_path = alt
        else:
            raise ValueError(
                f"custom agent manifest not found: expected {src}/agent.yml"
            )

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path}: top level must be a mapping")
    for required in ("image", "command"):
        if not raw.get(required):
            raise ValueError(f"{manifest_path}: missing required field '{required}'")

    params = {str(k): str(v) for k, v in (raw.get("params") or {}).items()}
    _reject_reserved_params(params, manifest_path)

    return CustomManifest(
        source_dir=str(src),
        name=str(raw.get("name") or src.name),
        image=str(raw["image"]),
        command=str(raw["command"]),
        workdir=str(raw.get("workdir") or "."),
        output=_normalize_output(raw.get("output", "stdout"), manifest_path),
        env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
        state_paths=[str(p) for p in (raw.get("state_paths") or [])],
        params=params,
        privileged=bool(raw.get("privileged", False)),
        build={str(k): str(v) for k, v in (raw.get("build") or {}).items()},
    )


def _normalize_output(output: Any, manifest_path: Path) -> dict[str, Any]:
    """Normalize the ``output`` spec. Supported: ``stdout`` | ``{json_field: X}``."""
    if isinstance(output, str):
        if output != "stdout":
            raise ValueError(
                f"{manifest_path}: output string must be 'stdout', got {output!r}"
            )
        return {"type": "stdout"}
    if isinstance(output, dict):
        if "json_field" in output:
            return {"type": "json_field", "field": str(output["json_field"])}
        raise ValueError(
            f"{manifest_path}: unsupported output spec {output!r}; "
            "use 'stdout' or {json_field: <key>}"
        )
    raise ValueError(
        f"{manifest_path}: output must be 'stdout' or a mapping, got {type(output).__name__}"
    )
