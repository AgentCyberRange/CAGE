"""Model registry loading from Cage configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from cage.contracts.coerce import optional_float
from cage.contracts.env import expand_env_refs
from cage.models.endpoint import ModelConfig

_KNOWN_FIELDS = frozenset((
    "provider", "model", "base_url", "api_key", "api_keys", "auth_source",
    "timeout", "max_retries", "input_cost_per_1m", "output_cost_per_1m",
    "extra_headers", "agent_model_names",
    # Context-window capabilities, promoted out of ``extra`` to typed fields.
    # ``context_window_size`` is an accepted alias for ``max_context_size``.
    "max_context_size", "reserved_context_size", "context_window_size",
))


def _optional_int(value: object) -> int | None:
    """Coerce a declared token count to a positive int, or None if unset."""

    if value is None:
        return None
    n = int(value)
    if n <= 0:
        raise ValueError(f"context size must be a positive integer, got {value!r}")
    return n


def load_models(path: str | Path) -> dict[str, ModelConfig]:
    """Load model endpoint declarations from ``models.yml``.

    The loader accepts either a top-level ``models:`` mapping or a direct
    mapping of model ids to endpoint fields. Credential pools are normalized so
    older single-key consumers can keep reading ``api_key`` while trial-level
    scheduling can rotate over ``api_keys``.
    """

    raw = expand_env_refs(yaml.safe_load(Path(path).read_text()))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid models file: {path}")

    models_section = raw.get("models", raw)
    result: dict[str, ModelConfig] = {}

    for model_id, cfg in models_section.items():
        if not isinstance(cfg, dict):
            continue
        key_pool = [
            str(k).strip()
            for k in (cfg.get("api_keys") or [])
            if str(k).strip()
        ]
        explicit_key = str(cfg.get("api_key", "") or "").strip()
        effective_key = explicit_key or (key_pool[0] if key_pool else "")
        if "max_concurrent" in cfg:
            raise ValueError(
                f"config/models.yml::{model_id} sets ``max_concurrent`` - this "
                "field has moved to ``agents.<i>.max_concurrent`` in "
                "project.yml. Each agent declares its own concurrency budget "
                "against the model it uses; concurrency is no longer a "
                "property of the model id."
            )
        agent_model_names = cfg.get("agent_model_names") or {}
        if not isinstance(agent_model_names, dict):
            raise ValueError(
                f"config/models.yml::{model_id}.agent_model_names must be a mapping"
            )
        result[model_id] = ModelConfig(
            id=model_id,
            provider=cfg.get("provider", "anthropic"),
            model=cfg.get("model", model_id),
            agent_model_names={
                str(k): str(v)
                for k, v in agent_model_names.items()
                if str(k).strip() and str(v).strip()
            },
            base_url=cfg.get("base_url", ""),
            api_key=effective_key,
            api_keys=key_pool,
            auth_source=str(cfg.get("auth_source", "") or "").strip(),
            timeout=cfg.get("timeout", 360),
            max_retries=cfg.get("max_retries", 2),
            input_cost_per_1m=optional_float(cfg.get("input_cost_per_1m")),
            output_cost_per_1m=optional_float(cfg.get("output_cost_per_1m")),
            extra_headers={
                str(k): str(v)
                for k, v in (cfg.get("extra_headers") or {}).items()
            },
            max_context_size=_optional_int(
                cfg.get("max_context_size", cfg.get("context_window_size"))
            ),
            reserved_context_size=_optional_int(cfg.get("reserved_context_size")),
            extra={k: v for k, v in cfg.items() if k not in _KNOWN_FIELDS},
        )

    return result
