from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from cage.target.adapters.base import BenchmarkSource, LaunchSpec, NormalizedChallenge, derive_flag_format

logger = logging.getLogger(__name__)


class ChallengeJsonAdapter:
    adapter_kind = "challenge_json"
    top_level_passthrough_fields = (
        "target_scope",
        "default_variant",
        "variant_names",
        "cve_description",
        "attacker_role",
        "application_service_keys",
        "application_port_suffixes",
        "proof_upload_service_key",
        "proof_upload_endpoint_suffix",
        "allowed_service_keys",
        "allowed_service_suffixes",
        "outbound_service_endpoint",
        "admin_username",
        "default_username",
        "default_password",
        "secret_file_path",
        "secret_table_name",
        "secret_table_column",
        "remote_code_executable_path",
        "denial_of_service_duration",
        "forbidden_actions",
        "environment_notes",
    )

    def discover(self, source: BenchmarkSource) -> dict[str, NormalizedChallenge]:
        root = Path(source.root)
        challenges: dict[str, NormalizedChallenge] = {}
        missing: list[str] = []

        for index_path in sorted(root.glob("*.json")):
            with open(index_path, "r", encoding="utf-8") as handle:
                index_data = json.load(handle)

            benchmark_name = index_path.stem
            for challenge_id, entry in index_data.items():
                challenge_root = (root / entry.get("path", "")).resolve()
                challenge_json_path = challenge_root / "challenge.json"
                # A challenge listed in the index but missing its challenge.json
                # is simply not downloaded (the dataset submodule is partial).
                # Skip it instead of aborting the whole run — discover whatever
                # IS present and report what was skipped. The benchmark then runs
                # only the available challenges.
                if not challenge_json_path.is_file():
                    missing.append(challenge_id)
                    continue
                with open(challenge_json_path, "r", encoding="utf-8") as handle:
                    challenge_json = json.load(handle)

                source_fields = dict(entry)
                source_fields.update(challenge_json)
                nested_source_fields = challenge_json.get("source_fields")
                if isinstance(nested_source_fields, dict):
                    # Some generated challenge.json files keep canonical
                    # internal fields under source_fields while also carrying
                    # lossy top-level mirrors for human-facing metadata.
                    source_fields.update(nested_source_fields)
                task = challenge_json.get("task") or challenge_json.get("description", "")
                benchmark_family = (
                    challenge_json.get("benchmark_family")
                    or entry.get("benchmark_family")
                    or benchmark_name
                )
                # ``runtime_scoring`` is declared per challenge in challenge.json
                # (already merged into ``source_fields`` above); the framework
                # stays benchmark-agnostic and never derives it.

                normalized = {
                    "id": challenge_id,
                    "benchmark_name": benchmark_name,
                    "benchmark": benchmark_name,
                    "benchmark_family": benchmark_family,
                    "adapter_kind": self.adapter_kind,
                    "task_profile": challenge_json.get("task_profile") or entry.get("task_profile") or "ctf_local",
                    "name": challenge_json.get("name") or entry.get("challenge") or challenge_json.get("target") or challenge_id,
                    "category": challenge_json.get("category") or entry.get("category", "unknown"),
                    "description": challenge_json.get("description") or task,
                    "task": task,
                    "files": challenge_json.get("files", []),
                    "flag": challenge_json.get("flag", ""),
                    "flag_format": derive_flag_format(challenge_json.get("flag", "")),
                    "full_path": str(challenge_root),
                    "source_fields": source_fields,
                }
                for key in self.top_level_passthrough_fields:
                    if key in source_fields:
                        normalized[key] = source_fields.get(key)

                challenges[challenge_id] = normalized

        if missing:
            total = len(challenges) + len(missing)
            logger.warning(
                "challenge_json: %d of %d challenge(s) under %s are not "
                "downloaded (no challenge.json) and were skipped — running the "
                "%d available one(s). Missing: %s",
                len(missing), total, root, len(challenges), ", ".join(sorted(missing)),
            )

        return challenges

    def build_launch_spec(self, challenge: NormalizedChallenge) -> LaunchSpec:
        challenge_root = Path(challenge["full_path"])
        source_fields = challenge.get("source_fields", {}) or {}
        compose_files = self._resolve_paths(challenge_root, source_fields.get("compose_files", []) or [])
        if not compose_files:
            compose_path = challenge_root / "docker-compose.yml"
            if compose_path.exists():
                compose_files = [str(compose_path.resolve())]

        if not compose_files:
            return LaunchSpec(
                mode="static",
                working_directory=str(challenge_root),
            )

        compose_env = self._resolve_compose_env(challenge_root, source_fields.get("compose_env", {}) or {})
        config = self._load_compose_stack(compose_files, compose_env=compose_env)

        services_config = config.get("services", {}) or {}
        target_services = list(source_fields.get("compose_target_services", []) or [])
        if not target_services:
            target_services = [
                service_name
                for service_name, service_config in services_config.items()
                if self._service_has_alias(service_config) or service_config.get("ports")
            ]
        if not target_services and len(services_config) == 1:
            target_services = list(services_config.keys())

        dependency_services = list(source_fields.get("compose_dependency_services", []) or [])
        if not dependency_services:
            dependency_services = self._collect_dependencies(services_config, target_services)

        runtime_patches = {}
        target_ports = dict(source_fields.get("target_ports", {}) or {})
        if source_fields.get("internal_port") is not None and target_services:
            target_ports.setdefault(target_services[0], int(source_fields["internal_port"]))
        if target_ports:
            runtime_patches["target_ports"] = target_ports
        target_port_protocols = dict(source_fields.get("target_port_protocols", {}) or {})
        if target_port_protocols:
            runtime_patches["target_port_protocols"] = target_port_protocols
        if compose_env:
            runtime_patches["compose_env"] = compose_env
        subnet_pool = source_fields.get("subnet_pool")
        if isinstance(subnet_pool, dict):
            runtime_patches["subnet_pool"] = subnet_pool
        env_file = source_fields.get("env_file_path")
        if env_file:
            runtime_patches["env_file"] = self._resolve_path(challenge_root, str(env_file))
        explicit_network_mode = str(source_fields.get("network_mode", "") or "").strip()
        if explicit_network_mode:
            runtime_patches["network_mode"] = explicit_network_mode
        explicit_agent_network = str(source_fields.get("agent_network", "") or "").strip()
        if explicit_agent_network:
            runtime_patches["agent_network"] = explicit_agent_network
        # Project-local subnet policy for compose stacks that own a private
        # network (declared per challenge.json; framework reads it generically).
        explicit_subnet_pool = str(source_fields.get("project_local_subnet_pool", "") or "").strip()
        if explicit_subnet_pool:
            runtime_patches["project_local_subnet_pool"] = explicit_subnet_pool
            explicit_subnet_prefix = source_fields.get("project_local_subnet_prefix")
            if explicit_subnet_prefix is not None:
                runtime_patches["project_local_subnet_prefix"] = int(explicit_subnet_prefix)

        return LaunchSpec(
            mode="compose",
            working_directory=str(challenge_root),
            compose_files=compose_files,
            target_services=target_services,
            dependency_services=dependency_services,
            runtime_patches=runtime_patches,
            exposure_mode=source_fields.get("exposure_mode", "host_ports"),
        )

    def _resolve_paths(self, challenge_root: Path, raw_paths: list[str]) -> list[str]:
        resolved: list[str] = []
        for raw_path in raw_paths:
            resolved.append(self._resolve_path(challenge_root, raw_path))
        return resolved

    def _resolve_path(self, challenge_root: Path, raw_path: str) -> str:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (challenge_root / path).resolve()
        else:
            path = path.resolve()
        return str(path)

    def _resolve_compose_env(self, challenge_root: Path, compose_env: dict) -> dict:
        resolved: dict = {}
        for key, value in compose_env.items():
            if not isinstance(value, str):
                resolved[key] = value
                continue
            if key.endswith("_DIR") or key.endswith("_FILE") or self._looks_like_path(value):
                resolved[key] = self._resolve_path(challenge_root, value)
            else:
                resolved[key] = value
        return resolved

    def _looks_like_path(self, value: str) -> bool:
        # Don't treat Docker image references (``name[:tag][@digest]``) as
        # paths — they contain ``/`` but the final segment carries the tag /
        # digest. Path resolution would prepend the challenge root and break
        # ``docker compose up``.
        if "/" in value:
            tail = value.rsplit("/", 1)[-1]
            if ":" in tail or "@" in tail:
                return False
        return value.startswith((".", "~", "/")) or "/" in value

    def _load_compose_stack(self, compose_files: list[str], compose_env: dict | None = None) -> dict:
        merged: dict = {}
        compose_env = compose_env or {}
        for compose_path in compose_files:
            current = self._load_compose_file(Path(compose_path), compose_env, seen_paths=set())
            merged = self._merge_compose_dicts(merged, current)
        return merged

    def _merge_compose_dicts(self, base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_compose_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _load_compose_file(self, path: Path, compose_env: dict, seen_paths: set[Path]) -> dict:
        resolved = path.resolve()
        if resolved in seen_paths:
            return {}
        seen_paths.add(resolved)

        with open(resolved, "r", encoding="utf-8") as handle:
            current = yaml.safe_load(handle) or {}

        merged: dict = {}
        include_entries = current.get("include", []) or []
        if isinstance(include_entries, (str, dict)):
            include_entries = [include_entries]

        for entry in include_entries:
            include_path = None
            if isinstance(entry, str):
                include_path = Path(self._resolve_compose_path(entry, resolved.parent, compose_env))
            elif isinstance(entry, dict):
                raw_path = entry.get("path")
                if isinstance(raw_path, str):
                    include_path = Path(self._resolve_compose_path(raw_path, resolved.parent, compose_env))
            if include_path is not None:
                merged = self._merge_compose_dicts(merged, self._load_compose_file(include_path, compose_env, seen_paths))

        current = dict(current)
        current.pop("include", None)
        return self._merge_compose_dicts(merged, current)

    def _resolve_compose_path(self, raw_path: str, base_dir: Path, compose_env: dict) -> str:
        path = Path(self._expand_compose_env(raw_path, compose_env)).expanduser()
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())

    def _expand_compose_env(self, value: str, compose_env: dict[str, str]) -> str:
        pieces: list[str] = []
        cursor = 0
        while cursor < len(value):
            start = value.find("${", cursor)
            if start == -1:
                pieces.append(value[cursor:])
                break
            if start > cursor:
                pieces.append(value[cursor:start])
            expanded, cursor = self._expand_compose_expr(value, start, compose_env)
            pieces.append(expanded)
        return "".join(pieces)

    def _expand_compose_expr(self, value: str, start: int, compose_env: dict[str, str]) -> tuple[str, int]:
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
            return self._expand_compose_env(default_value, compose_env), cursor
        if ":?" in expression:
            var_name, _error_message = expression.split(":?", 1)
            var_name = var_name.strip()
            if compose_env.get(var_name):
                return compose_env[var_name], cursor
            return f"${{{expression}}}", cursor

        var_name = expression.strip()
        return compose_env.get(var_name, f"${{{expression}}}"), cursor

    def _collect_dependencies(self, services_config: dict, target_services: list[str]) -> list[str]:
        dependencies: list[str] = []
        seen = set(target_services)
        for service_name in target_services:
            service_config = services_config.get(service_name, {}) or {}
            depends_on = service_config.get("depends_on", {})
            if isinstance(depends_on, dict):
                names = list(depends_on.keys())
            else:
                names = list(depends_on or [])
            for dependency in names:
                if dependency in seen:
                    continue
                dependencies.append(dependency)
                seen.add(dependency)
        return dependencies

    def _service_has_alias(self, service_config: dict) -> bool:
        networks = service_config.get("networks", {}) or {}
        if isinstance(networks, list):
            return False
        for network_config in networks.values():
            if network_config and network_config.get("aliases"):
                return True
        return False
