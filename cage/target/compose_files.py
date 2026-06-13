"""Compose-file plumbing: load/merge a compose stack, expand env, absolutize paths, inject external networks.

``load_compose_stack`` and ``expand_compose_env_values`` are the public
compose-reading surface for benchmark authors (Layer 2): a benchmark that
ships docker-compose targets can resolve the same merged/expanded view of a
stack that the target server materializes, without reaching into
``cage.target.server`` internals. The underscore helpers below are
implementation detail shared with :mod:`cage.target.server.launch_runtime`.
"""
from __future__ import annotations

import yaml
from cage.target.adapters.base import LaunchSpec
from pathlib import Path
from typing import Any


def load_compose_stack(compose_files: list[str], compose_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Load and deep-merge a compose stack (including ``include:`` files)."""
    merged: dict[str, Any] = {}
    compose_env = compose_env or {}
    for compose_file in compose_files:
        current = _load_compose_file(Path(compose_file), compose_env=compose_env, seen_paths=set())
        merged = _merge_compose_dicts(merged, current)
    return merged


def _merge_compose_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_compose_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _inject_external_network(
    service_config: dict[str, Any],
    docker_network: str,
    *,
    aliases: list[str] | None,
    project_directory: Path,
    compose_env: dict[str, str],
) -> None:
    if _service_uses_network_mode(
        service_config,
        project_directory=project_directory,
        compose_env=compose_env,
    ):
        return
    networks = service_config.setdefault("networks", {})
    if isinstance(networks, list):
        networks = {name: {} for name in networks}
        service_config["networks"] = networks
    network_config = networks.get(docker_network)
    if not isinstance(network_config, dict):
        network_config = {}
        networks[docker_network] = network_config

    desired_aliases = [alias for alias in (aliases or []) if alias]
    if desired_aliases:
        existing_aliases = list(network_config.get("aliases", []) or [])
        for alias in desired_aliases:
            if alias not in existing_aliases:
                existing_aliases.append(alias)
        network_config["aliases"] = existing_aliases


def _compose_project_directory(spec: LaunchSpec) -> Path:
    if spec.compose_files:
        return Path(spec.compose_files[0]).resolve().parent
    return Path(spec.working_directory).resolve()


def _absolutize_compose_paths(config: dict[str, Any], project_directory: Path, compose_env: dict[str, str] | None = None) -> None:
    compose_env = compose_env or {}
    services_config = config.get("services", {}) or {}
    for service_config in services_config.values():
        extends = service_config.get("extends")
        if isinstance(extends, dict):
            extends_file = extends.get("file")
            if isinstance(extends_file, str):
                extends["file"] = _resolve_compose_path(extends_file, project_directory, compose_env)

        build = service_config.get("build")
        if isinstance(build, str):
            service_config["build"] = _resolve_compose_path(build, project_directory, compose_env)
        elif isinstance(build, dict):
            context = build.get("context")
            if isinstance(context, str):
                build["context"] = _resolve_compose_path(context, project_directory, compose_env)
            dockerfile = build.get("dockerfile")
            if isinstance(dockerfile, str) and not Path(dockerfile).is_absolute():
                context_root = Path(build.get("context", project_directory))
                build["dockerfile"] = str((context_root / dockerfile).resolve())

        volumes = service_config.get("volumes", []) or []
        service_config["volumes"] = [_absolutize_volume(volume, project_directory, compose_env) for volume in volumes]


def _resolve_compose_path(raw_path: str, project_directory: Path, compose_env: dict[str, str] | None = None) -> str:
    expanded_path = _expand_compose_env(raw_path, compose_env or {})
    path = Path(expanded_path).expanduser()
    if path.is_absolute():
        return str(path)
    return str((project_directory / path).resolve())


def _absolutize_volume(volume: Any, project_directory: Path, compose_env: dict[str, str]) -> Any:
    if isinstance(volume, str):
        parts = volume.split(":")
        if len(parts) < 2:
            return volume
        source = parts[0]
        if _looks_like_bind_source(source):
            parts[0] = _resolve_compose_path(source, project_directory, compose_env)
            return ":".join(parts)
        return volume

    if isinstance(volume, dict):
        source = volume.get("source")
        volume_type = volume.get("type")
        if isinstance(source, str) and (volume_type == "bind" or _looks_like_bind_source(source)):
            volume = dict(volume)
            volume["source"] = _resolve_compose_path(source, project_directory, compose_env)
        return volume

    return volume


def _looks_like_bind_source(source: str) -> bool:
    return source.startswith(".") or source.startswith("~") or "/" in source


def _service_uses_network_mode(
    service_config: dict[str, Any],
    *,
    project_directory: Path,
    compose_env: dict[str, str],
) -> bool:
    if service_config.get("network_mode"):
        return True

    extends = service_config.get("extends")
    if not isinstance(extends, dict):
        return False

    ext_file = extends.get("file")
    ext_service = extends.get("service")
    if not isinstance(ext_file, str) or not isinstance(ext_service, str):
        return False

    ext_path = Path(_resolve_compose_path(ext_file, project_directory, compose_env))
    if not ext_path.exists():
        return False

    ext_config = _load_compose_file(ext_path, compose_env, seen_paths=set())
    ext_service_config = ((ext_config.get("services", {}) or {}).get(ext_service, {}) or {})
    return _service_uses_network_mode(
        ext_service_config,
        project_directory=ext_path.parent,
        compose_env=compose_env,
    )


def _expand_compose_env(value: str, compose_env: dict[str, str]) -> str:
    pieces: list[str] = []
    cursor = 0
    while cursor < len(value):
        start = value.find("${", cursor)
        if start == -1:
            pieces.append(value[cursor:])
            break
        if start > cursor:
            pieces.append(value[cursor:start])
        expanded, cursor = _expand_compose_expr(value, start, compose_env)
        pieces.append(expanded)
    return "".join(pieces)


def _expand_compose_expr(value: str, start: int, compose_env: dict[str, str]) -> tuple[str, int]:
    cursor = start + 2
    depth = 1
    while cursor < len(value) and depth > 0:
        if value.startswith("${", cursor):
            depth += 1
            cursor += 2
            continue
        if value[cursor] == "}":
            depth -= 1
        cursor += 1

    expression = value[start + 2 : cursor - 1]
    if ":-" in expression:
        var_name, default_value = expression.split(":-", 1)
        var_name = var_name.strip()
        if compose_env.get(var_name):
            return compose_env[var_name], cursor
        return _expand_compose_env(default_value, compose_env), cursor
    if ":?" in expression:
        var_name, _error_message = expression.split(":?", 1)
        var_name = var_name.strip()
        if compose_env.get(var_name):
            return compose_env[var_name], cursor
        return f"${{{expression}}}", cursor

    var_name = expression.strip()
    return compose_env.get(var_name, f"${{{expression}}}"), cursor


def expand_compose_env_values(value: Any, compose_env: dict[str, str]) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in a compose tree."""
    if isinstance(value, str):
        return _expand_compose_env(value, compose_env)
    if isinstance(value, list):
        return [expand_compose_env_values(item, compose_env) for item in value]
    if isinstance(value, dict):
        return {
            key: expand_compose_env_values(item, compose_env)
            for key, item in value.items()
        }
    return value


def _load_compose_file(path: Path, compose_env: dict[str, str], seen_paths: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen_paths:
        return {}
    seen_paths.add(resolved)

    with open(resolved, "r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}

    merged: dict[str, Any] = {}
    include_entries = current.get("include", []) or []
    if isinstance(include_entries, (str, dict)):
        include_entries = [include_entries]

    for entry in include_entries:
        include_path = None
        if isinstance(entry, str):
            include_path = Path(_resolve_compose_path(entry, resolved.parent, compose_env))
        elif isinstance(entry, dict):
            raw_path = entry.get("path")
            if isinstance(raw_path, str):
                include_path = Path(_resolve_compose_path(raw_path, resolved.parent, compose_env))
        if include_path is not None:
            merged = _merge_compose_dicts(merged, _load_compose_file(include_path, compose_env, seen_paths))

    current = dict(current)
    current.pop("include", None)
    merged = _merge_compose_dicts(merged, current)
    return merged

