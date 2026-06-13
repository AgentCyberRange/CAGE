from __future__ import annotations

from pathlib import Path

import pytest

from cage.benchmarks.prompt_contract import render_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTH_MARKER = "authorized pentest"

_PROFILE_DIRS: dict[str, Path] = {
    "ctfbench": REPO_ROOT / "examples" / "nyu" / "prompts",
    "autopenbench": REPO_ROOT / "examples" / "autopenbench" / "prompts",
    "cvebench/zero_day": REPO_ROOT / "examples" / "cvebench" / "prompts" / "zero_day",
    "cvebench/one_day": REPO_ROOT / "examples" / "cvebench" / "prompts" / "one_day",
}


def _instance_context(profile: str) -> dict[str, object]:
    if profile == "ctfbench":
        return {
            "instance_data": {
                "name": "Collision Course",
                "category": "crypto",
                "description": "Recover the password.",
                "flag_format": "flag{...}",
                "files": ["handout.zip"],
                "check_supported": True,
                "target_info": {
                    "target": {
                        "host": "127.0.0.1",
                        "port": 31337,
                        "inner_host": "target",
                        "inner_port": 31337,
                    }
                },
            }
        }

    if profile == "autopenbench":
        return {
            "instance_data": {
                "task": "Find the target in {target_subnet} and retrieve the secret.",
                "runtime": {"network_subnet": "172.31.12.0/24"},
                "check_supported": True,
            }
        }

    if profile.startswith("cvebench/"):
        return {
            "instance_data": {
                "service_names": {"target": "target"},
                "application_service_keys": ["target"],
                "application_port_suffixes": [":9090"],
                "allowed_service_keys": ["target"],
                "allowed_service_suffixes": [":9090"],
                "proof_upload_service_key": "target",
                "proof_upload_endpoint_suffix": ":9091/upload",
                "runtime_scoring_service_key": "target",
                "runtime_scoring_endpoint_suffix": ":9091/done",
                "outbound_service_endpoint": "http://target:8080/outbound",
                "cve_description": "The target is vulnerable to unauthenticated RCE.",
            }
        }

    raise AssertionError(f"Unexpected profile: {profile}")


@pytest.mark.parametrize("profile", list(_PROFILE_DIRS))
def test_system_prompt_marks_requests_as_authorized_pentest(profile: str) -> None:
    prompt = render_prompt(
        template_dir=_PROFILE_DIRS[profile],
        template_type="system",
        command_docs="submit: submit a candidate flag",
        skill_descriptions="web_rce: web exploitation helper",
    )

    assert AUTH_MARKER in prompt.lower()


@pytest.mark.parametrize("profile", list(_PROFILE_DIRS))
def test_instance_prompt_marks_requests_as_authorized_pentest(profile: str) -> None:
    prompt = render_prompt(
        template_dir=_PROFILE_DIRS[profile],
        template_type="instance",
        **_instance_context(profile),
    )

    assert AUTH_MARKER in prompt.lower()


def test_render_prompt_prefers_plain_jinja_template_name(tmp_path: Path) -> None:
    (tmp_path / "instance.j2").write_text(
        "preferred jinja template renders {{ value }}",
        encoding="utf-8",
    )
    (tmp_path / "instance_template.txt").write_text(
        "legacy fallback template renders {{ value }}",
        encoding="utf-8",
    )

    assert render_prompt(tmp_path, value="template") == "preferred jinja template renders template"


def test_render_prompt_keeps_legacy_template_name_compatibility(tmp_path: Path) -> None:
    (tmp_path / "instance_template.txt").write_text(
        "legacy fallback template renders {{ value }}",
        encoding="utf-8",
    )

    assert render_prompt(tmp_path, value="template") == "legacy fallback template renders template"
