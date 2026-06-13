"""Shared helpers for the challenge launch/cleanup workflow.

Small docker/env utilities used by :mod:`launch`, :mod:`cleanup` and
:mod:`network_debug`. Kept here as the dependency-free leaf of the workflow.
"""
from __future__ import annotations


from cage.target.server.server_state import TARGET_SERVER_NAMESPACE
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Dict, Optional

def pydantic_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_internal_port(port_def: Any) -> Optional[int]:
    try:
        if isinstance(port_def, int):
            return port_def
        if isinstance(port_def, str):
            return int(port_def.split(':')[-1].split('/')[0])
        return None
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return None


def ensure_docker_cli_config_dir(env: Dict[str, str]) -> None:
    docker_config = env.get("DOCKER_CONFIG", "").strip()
    if not docker_config:
        docker_config = f"/tmp/cage-bench-docker-config-{TARGET_SERVER_NAMESPACE}"
        env["DOCKER_CONFIG"] = docker_config
    Path(docker_config).mkdir(parents=True, exist_ok=True)


def load_env_file_vars(env_file: str | Path | None) -> Dict[str, str]:
    if not env_file:
        return {}
    path = Path(env_file)
    if not path.exists():
        return {}

    result: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result
