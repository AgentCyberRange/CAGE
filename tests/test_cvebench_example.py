from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml


def _load_cvebench_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "cvebench"
        / "benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("cvebench_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RecordingContainer:
    def __init__(self) -> None:
        self.exec_commands: list[str] = []
        self.writes: dict[str, str] = {}
        self.copies: list[tuple[str, str]] = []

    def exec(self, command: str, *, timeout=None):
        self.exec_commands.append(command)
        return SimpleNamespace(exit_code=0, stdout='{"status": "success"}', stderr="")

    def write_file(self, path: str, content: str) -> None:
        self.writes[path] = content

    def copy_to(self, source: str, dest: str) -> None:
        self.copies.append((source, dest))


def test_cvebench_loads_variant_samples_from_ctf_manager_dataset():
    module = _load_cvebench_module()
    benchmark = module.CVEBench()

    benchmark.setup()
    samples = [s for s in benchmark.iter_samples() if s["challenge_name"] == "CVE-2023-37999"]
    sample = next(s for s in samples if s["test_type"] == "zero_day")

    assert benchmark.ctf_manager is not None
    assert sample["benchmark"] == "cvebench"
    assert sample["category"] == "critical"
    assert sample["challenge_id"] == "cvb-CVE-2023-37999"
    assert sample["challenge_name"] == "CVE-2023-37999"
    assert sample["variant"] == "zero_day"
    assert sample["test_type"] == "zero_day"
    assert sample["id"] == "cvb-CVE-2023-37999-zero_day"
    assert sample["metadata"]["variant_names"] == ["zero_day", "one_day"]
    assert sample["metadata"]["internal_port"] == 9090
    assert sample["metadata"]["service_ports"] == {"target": 9090}
    assert sample["metadata"]["runtime_scoring"] == {
        "kind": "http_poll",
        "service": "target",
        "port": 9091,
        "path": "/done",
    }
    assert sample["challenge_dir"].endswith("cvebench/critical/challenges/CVE-2023-37999")
    assert "Privilege Escalation" in sample["cve_description"]
    assert {s["test_type"] for s in samples} == {"zero_day", "one_day"}


def test_cvebench_benchmark_script_imports_from_example_dir():
    repo_root = Path(__file__).resolve().parents[1]
    benchmark_dir = repo_root / "examples" / "cvebench"

    result = subprocess.run(
        ["python3", "benchmark.py"],
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cvebench_prepare_trial_copies_env_file_only():
    module = _load_cvebench_module()
    benchmark = module.CVEBench()
    benchmark.setup()
    sample = next(
        s for s in benchmark.iter_samples() if s["id"] == "cvb-CVE-2023-37999-zero_day"
    )
    container = _RecordingContainer()

    benchmark.prepare_trial(container, sample, "/workspace")

    assert "/workspace/MISSION.md" not in container.writes
    assert "/workspace/target.json" not in container.writes
    assert [dest for _, dest in container.copies] == ["/workspace/.env"]


def test_cvebench_build_prompt_renders_variant_prompt_profile():
    module = _load_cvebench_module()
    benchmark = module.CVEBench()
    benchmark.setup()
    sample = next(
        s for s in benchmark.iter_samples() if s["id"] == "cvb-CVE-2023-37999-one_day"
    )

    prompt = benchmark.build_prompt(sample)

    assert prompt.startswith("# MISSION BRIEF")
    assert "known to have the following vulnerability" in prompt
    assert sample["cve_description"] in prompt
    assert "target:9090" in prompt
    assert "target:9091/upload" in prompt
    assert "target:9091/done" in prompt
    assert "OPERATIONAL GUIDANCE" in prompt
    assert "check:8080/check" not in prompt


def test_cvebench_tool_check_done_queries_scoring_url_from_agent_container():
    module = _load_cvebench_module()
    benchmark = module.CVEBench(benchmark_root="/does/not/matter")
    container = _RecordingContainer()
    sample = {
        "metadata": {
            "runtime_scoring": {
                "service": "check",
                "port": 9091,
                "path": "/done",
            }
        }
    }

    from cage.scoring import GatherRuntime

    output = benchmark.scorer().gather(GatherRuntime(sample=sample, container=container))

    assert output == '{"status": "success"}'
    assert container.exec_commands == [
        (
            "env -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy "
            "-u ALL_PROXY -u all_proxy curl -fsS --max-time 60 http://check:9091/done"
        )
    ]


def test_cvebench_score_parses_saved_check_done_output(tmp_path):
    module = _load_cvebench_module()
    benchmark = module.CVEBench(benchmark_root="/does/not/matter")
    sample = {"id": "cvb-demo", "category": "critical", "challenge_name": "CVE-2024-DEMO"}
    trial_dir = tmp_path / "trials" / "trial-demo"
    runtime_dir = trial_dir / "runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "check_done_output.txt").write_text(
        "check_done returned:\n" + json.dumps({"status": "ok", "message": "RCE"}),
        encoding="utf-8",
    )

    from cage.scoring import ScoringContext

    context = ScoringContext(
        trial_id="trial-demo",
        trial_index=0,
        sample=sample,
        trial_dir=trial_dir,
    )

    scores = benchmark.scorer().score(context)

    assert scores["cvebench"].value == 1.0
    assert scores["cvebench"].metadata["scorable"] is True
    assert scores["cvebench"].metadata["matched"] is True
    assert scores["cvebench"].metadata["check_done_output"] == {
        "status": "ok",
        "message": "RCE",
    }


def test_cvebench_score_prefers_live_success_verdict(tmp_path):
    module = _load_cvebench_module()
    benchmark = module.CVEBench(benchmark_root="/does/not/matter")
    sample = {"id": "cvb-demo", "category": "critical", "challenge_name": "CVE-2024-DEMO"}
    trial_dir = tmp_path / "trials" / "trial-demo"
    runtime_dir = trial_dir / "runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "live_success.json").write_text(
        json.dumps(
            {
                "success": True,
                "mode": "reactive",
                "source": "agent_9091_triggered_check_done",
            }
        ),
        encoding="utf-8",
    )
    from cage.scoring import ScoringContext

    context = ScoringContext(
        trial_id="trial-demo",
        trial_index=0,
        sample=sample,
        trial_dir=trial_dir,
    )

    scores = benchmark.scorer().score(context)

    assert scores["cvebench"].value == 1.0
    assert scores["cvebench"].metadata["matched"] is True
    assert scores["cvebench"].metadata["score_source"] == "live_success"
    assert scores["cvebench"].metadata["live_success"]["source"] == (
        "agent_9091_triggered_check_done"
    )


def test_cvebench_project_config_has_no_shared_runtime_hooks():
    project_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "cvebench"
        / "default_cvebench.yml"
    )

    raw = yaml.safe_load(project_path.read_text())

    assert raw["project"]["name"] == "cvebench-cage"
    assert raw["eval"]["benchmark"]["module"] == "./benchmark.py"
    assert raw["eval"]["benchmark"]["class"] == "CVEBench"
    assert raw["eval"]["limit"] == 1
    assert "hooks" not in raw
