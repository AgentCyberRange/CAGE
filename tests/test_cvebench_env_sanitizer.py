"""Regression tests for the cvebench .env sanitizer.

The challenge's source ``.env`` is used to drive the *target* docker-compose
stack and carries answer-revealing fields like ``CVE=CVE-2023-37999``.
The sanitizer must strip everything not on the explicit whitelist before
the .env reaches ``/home/agent/workspace`` — otherwise the zero-day
variant of every CVEBench challenge is trivially solvable by
``cat /home/agent/workspace/.env``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_CVEBENCH = Path(__file__).resolve().parents[1] / "examples" / "cvebench" / "benchmark.py"


@pytest.fixture(scope="module")
def cvebench_module():
    """Load cvebench benchmark.py as a module without running its ChallengeClient setup."""
    spec = importlib.util.spec_from_file_location("_cage_cvebench_test", str(_CVEBENCH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_env(tmp_path: Path, body: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return p


def test_strips_cve_keys(cvebench_module, tmp_path):
    body = "CVE=CVE-2023-37999\nCVE_LOWER=cve-2023-37999\nADMIN_USERNAME=admin\n"
    env_path = _write_env(tmp_path, body)
    out = cvebench_module._sanitize_env_for_agent(env_path)
    assert "CVE-2023-37999" not in out
    assert "cve-2023-37999" not in out
    assert "ADMIN_USERNAME=admin" in out


def test_preserves_operational_keys(cvebench_module, tmp_path):
    body = (
        "APPLICATION_URL=target:9090\n"
        "DEFAULT_PASSWORD=password\n"
        "PROOF_UPLOADING_URL=target:9091/upload\n"
        "REMOTE_CODE_EXECUTABLE_PATH=/tmp/pwn\n"
    )
    env_path = _write_env(tmp_path, body)
    out = cvebench_module._sanitize_env_for_agent(env_path)
    assert "APPLICATION_URL=target:9090" in out
    assert "DEFAULT_PASSWORD=password" in out
    assert "PROOF_UPLOADING_URL=target:9091/upload" in out
    assert "REMOTE_CODE_EXECUTABLE_PATH=/tmp/pwn" in out


def test_drops_unknown_keys_by_default(cvebench_module, tmp_path):
    """Whitelist semantics: any new field is dropped unless explicitly allowed.

    This is the security property — challenge authors adding a leaky key
    in the future won't accidentally expose it to the agent.
    """
    body = "ADMIN_USERNAME=admin\nSOLUTION_HINT=use_path_traversal\nFLAG=secret\n"
    env_path = _write_env(tmp_path, body)
    out = cvebench_module._sanitize_env_for_agent(env_path)
    assert "ADMIN_USERNAME=admin" in out
    assert "SOLUTION_HINT" not in out
    assert "FLAG" not in out
    assert "secret" not in out


def test_passes_through_comments_and_blank_lines(cvebench_module, tmp_path):
    body = "# operational config\n\nADMIN_USERNAME=admin\n# end\n"
    env_path = _write_env(tmp_path, body)
    out = cvebench_module._sanitize_env_for_agent(env_path)
    assert "# operational config" in out
    assert "# end" in out
    assert "ADMIN_USERNAME=admin" in out


def test_real_cvebench_env_fixture_is_leak_free(cvebench_module, tmp_path):
    """Smoke test against a real CVEBench .env if the dataset is available.

    Skips when the dataset isn't on disk (e.g. CI without the heavy fixture).
    """
    real_env = Path(
        "/data/pxd-team/workspace/fyh/cage/examples/cvebench/datasets/cvebench/"
        "critical/challenges/CVE-2023-37999/.env"
    )
    if not real_env.is_file():
        pytest.skip(f"dataset not present at {real_env}")
    out = cvebench_module._sanitize_env_for_agent(real_env)
    # Both the upper and lower case CVE references must be gone.
    assert "CVE-2023-37999" not in out
    assert "cve-2023-37999" not in out
    # Pattern check: no CVE-YYYY-NNNN style left anywhere.
    import re
    assert re.search(r"CVE-\d{4}-\d+", out, re.IGNORECASE) is None
