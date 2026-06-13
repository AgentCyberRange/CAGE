from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from cage.artifacts.writer import ExperimentArtifactWriter
from cage.benchmarks.registry import BenchmarkSpec, list_benchmarks, resolve_benchmark
from cage.cli import main
from cage.experiment.model import build_experiment_plan, load_experiment_spec
from cage.experiment.engine.overlays import override_selected_agent_model
from cage.target.check import _print_target_check_plan, _select_target_samples, _target_id

ROOT = Path(__file__).resolve().parents[1]
CLI_MAIN = importlib.import_module("cage.cli.main")
BENCHMARK_CLI = importlib.import_module("cage.cli.commands.benchmark")
SCORE_CLI = importlib.import_module("cage.cli.commands.score")
RUN_CLI = importlib.import_module("cage.cli.commands.run")


def _wait_config(*models):
    return SimpleNamespace(
        agents=[SimpleNamespace(model=models[0], model_sources=list(models))]
    )


def test_wait_for_model_skips_non_local_providers(monkeypatch, capsys):
    from cage.models import ModelConfig

    polled: list[str] = []
    monkeypatch.setattr(
        RUN_CLI, "_model_endpoint_reachable",
        lambda url, **k: polled.append(url) or True,
    )
    # anthropic + openai SaaS: never polled, returns immediately.
    saas = ModelConfig(id="m", provider="anthropic", model="x", base_url="https://api")
    RUN_CLI._wait_for_model_endpoints(_wait_config(saas), timeout=5, interval=0)
    assert polled == []
    assert "no self-hosted" in capsys.readouterr().out


def test_wait_for_model_polls_local_until_up(monkeypatch):
    from cage.models import ModelConfig

    calls = {"n": 0}

    def fake(url, **k):
        calls["n"] += 1
        return calls["n"] >= 3  # down twice, then up

    monkeypatch.setattr(RUN_CLI, "_model_endpoint_reachable", fake)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    vllm = ModelConfig(id="glm", provider="vllm", model="g", base_url="http://h/v1")
    RUN_CLI._wait_for_model_endpoints(_wait_config(vllm), timeout=0, interval=1)
    assert calls["n"] == 3


def test_wait_for_model_times_out(monkeypatch):
    import click

    from cage.models import ModelConfig

    monkeypatch.setattr(RUN_CLI, "_model_endpoint_reachable", lambda url, **k: False)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    vllm = ModelConfig(id="glm", provider="vllm", model="g", base_url="http://h/v1")
    with pytest.raises(click.UsageError, match="timed out"):
        RUN_CLI._wait_for_model_endpoints(_wait_config(vllm), timeout=1, interval=2)


def test_builtin_registry_lists_release_facing_benchmarks_without_aliases() -> None:
    ids = [spec.id for spec in list_benchmarks()]

    assert ids == [
        "web_exploit_bench",
        "post_exploit_bench",
        "autopenbench",
        "cvebench",
        "cybergym",
        "arvo",
        "hackworld",
        "nyu_ctf",
        "strongreject",
    ]
    assert all(spec.aliases == () for spec in list_benchmarks())
    assert resolve_benchmark("web_exploit_bench").id == "web_exploit_bench"


def test_registry_ignores_local_run_configs(tmp_path, monkeypatch) -> None:
    """``local*.yml`` (git-ignored private run configs) never register.

    Regression: a ``local_*.yml`` that declares the same ``registration.id`` as
    the committed ``default_*.yml`` must not register a second (or hijacking)
    entry, nor make ``cage benchmark list`` print the git-ignored local path as
    the project file users are told to run.
    """
    from cage.benchmarks import registry as registry_mod

    bench = tmp_path / "examples" / "demo"
    bench.mkdir(parents=True)
    (bench / "default_demo.yml").write_text(
        "registration:\n  id: demo\n  display_name: Demo\n  order: 1\n",
        encoding="utf-8",
    )
    (bench / "local_my_run.yml").write_text(
        "registration:\n  id: demo\n  display_name: Demo\n  order: 1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(registry_mod, "EXAMPLES_DIR", tmp_path / "examples")
    monkeypatch.setattr(registry_mod, "REPO_ROOT", tmp_path)
    registry_mod._registered_specs.cache_clear()
    try:
        specs = registry_mod.list_benchmarks()
        assert [s.id for s in specs] == ["demo"]
        assert specs[0].project_file == Path("examples/demo/default_demo.yml")
    finally:
        registry_mod._registered_specs.cache_clear()


def test_benchmark_list_and_show_include_benchmark_owned_options() -> None:
    runner = CliRunner()

    listed = runner.invoke(main, ["benchmark", "list"])
    assert listed.exit_code == 0, listed.output
    assert "web_exploit_bench" in listed.output
    assert "post_exploit_bench" in listed.output
    assert "cvebench" in listed.output
    assert "strongreject" in listed.output
    assert "Default project" in listed.output
    assert "Default config" not in listed.output
    assert "Run help: cage run <id> --help" in listed.output
    assert "Run by ID: cage run <id> ..." in listed.output

    web = runner.invoke(main, ["benchmark", "show", "web_exploit_bench"])
    assert web.exit_code == 0, web.output
    assert "Web Exploit Bench" in web.output
    assert "Benchmark ID: web_exploit_bench" in web.output
    assert "Default project:" in web.output
    assert "Project name: web-exploit-bench (used in run metadata/artifacts)" in web.output
    assert "Benchmark root:" in web.output
    assert "Samples:" in web.output
    assert "Sample IDs:" in web.output
    assert "target builds collapse level ids to the underlying target" in web.output
    assert "Runtime defaults:" in web.output
    assert "timeout: 7200" in web.output
    # ``runtime.max_rounds: -1`` defers to the benchmark/sample default; the help
    # surface renders the resolved number (150 for the web tasks), not the sentinel.
    assert "max_rounds: 150 (benchmark default; project.yml sets max_rounds: -1 to defer)" in web.output
    assert "max_input_tokens: unset (unlimited)" in web.output
    assert "max_output_tokens: unset (unlimited)" in web.output
    assert "max_cost: unset (unlimited)" in web.output
    assert "Concurrency:" in web.output
    assert "runtime.max_trials_global caps in-flight trials across the whole run" in web.output
    assert "(default unlimited)" in web.output
    assert "--max-concurrent to cap one selected agent/model" in web.output
    assert "Target defaults:" in web.output
    assert "Agents / models:" in web.output
    assert "codex (max_concurrent: 3): gpt-5.5" in web.output
    assert "codex" in web.output
    assert "gpt-5.5" in web.output
    assert "claude_code" in web.output
    assert "claude-opus-sub" in web.output
    assert "claude-opus" in web.output
    assert "deepseek-v4-pro" in web.output
    assert "glm-5.1" in web.output
    assert "qwen_code" in web.output
    assert "qwen3.7-max" in web.output
    assert "kimi_code" in web.output
    assert "kimi-k2.6-cli" in web.output
    assert "Aliases:" not in web.output
    assert "codex_gpt55" not in web.output
    assert "--prompt-level" in web.output
    assert "Recommended workflow:" in web.output
    assert "Step 1. Check config and prompt rendering" in web.output
    assert "Step 2. Build or verify benchmark targets" in web.output
    assert "Step 3. Run one smoke trial" in web.output
    assert "--prompt-level l0" in web.output

    post = runner.invoke(main, ["benchmark", "show", "post_exploit_bench"])
    assert post.exit_code == 0, post.output
    assert "Post Exploit Bench" in post.output
    assert "Samples:" in post.output
    assert "Sample IDs:" in post.output
    assert "Agents / models:" in post.output
    assert (
        "qwen_code (max_concurrent: 8; "
        "model caps: qwen3.6-max-preview=8, qwen3.7-max=2)"
    ) in post.output
    assert "--prompt-level" in post.output
    assert "--prompt-level l0" in post.output


def test_run_benchmark_help_prints_benchmark_configuration_surface() -> None:
    result = CliRunner().invoke(main, ["run", "web_exploit_bench", "--help"])

    assert result.exit_code == 0, result.output
    assert "Web Exploit Bench" in result.output
    assert "Benchmark ID: web_exploit_bench" in result.output
    assert "Default project:" in result.output
    assert "Samples:" in result.output
    assert "Runtime defaults:" in result.output
    assert "Target defaults:" in result.output
    assert "Agents / models:" in result.output
    assert "Benchmark options:" in result.output
    assert "--prompt-level" in result.output
    assert "Recommended workflow:" in result.output
    assert "Step 1. Check config and prompt rendering" in result.output
    assert "Step 2. Build or verify benchmark targets" in result.output
    assert "Step 3. Run one smoke trial" in result.output
    assert "cage run web_exploit_bench --agent <agent> --model <model-id>" in result.output
    assert "--max-concurrent 1" in result.output
    assert "--max-workers" not in result.output
    assert "cage benchmark check web_exploit_bench --sample <sample_id>" in result.output
    assert "cage benchmark build web_exploit_bench --max-concurrent 4" in result.output


def test_run_benchmark_help_works_after_benchmark_owned_options() -> None:
    result = CliRunner().invoke(
        main,
        ["run", "web_exploit_bench", "--prompt-level", "l0", "--help"],
    )

    assert result.exit_code == 0, result.output
    assert "Web Exploit Bench" in result.output
    assert "Benchmark options:" in result.output
    assert "--prompt-level" in result.output


def test_generic_run_help_points_users_to_benchmark_specific_help() -> None:
    result = CliRunner().invoke(main, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert "PROJECT_OR_BENCHMARK" in result.output
    assert "CAGE RUN" in result.output
    assert "Step 1. Choose a benchmark" in result.output
    assert "web_exploit_bench" in result.output
    assert "post_exploit_bench" in result.output
    assert "Step 2. Inspect benchmark-specific run help" in result.output
    assert "cage run <id> --help" in result.output
    assert "Step 3. Check and build benchmark targets" in result.output
    assert "cage benchmark build <id> --sample <sample_id>" in result.output
    assert "Step 4. Run one smoke trial" in result.output
    assert "Step 5. Scale up or resume" in result.output
    assert "--agent" in result.output
    assert "--model" in result.output
    assert "--max-sample-num" in result.output
    assert "--max-trial-num" in result.output
    assert "--max-concurrent" in result.output
    assert "--allow-launch-build" in result.output
    assert "--max-workers" not in result.output
    assert "cage benchmark build <id> --max-concurrent 4" in result.output
    assert "--limit" not in result.output
    assert "--max-trial " not in result.output
    assert "--inspect" not in result.output
    assert "--no-terminal-ui" not in result.output
    assert "--debug-log" not in result.output


def test_run_rejects_removed_run_options(tmp_path: Path) -> None:
    project_file = _write_demo_project(tmp_path)
    runner = CliRunner()

    for args in (
        ["--limit", "1"],
        ["--max-trial", "1"],
        ["--max-workers", "1"],
    ):
        result = runner.invoke(main, ["run", str(project_file), *args])
        assert result.exit_code != 0, result.output
        assert "Unknown run option" in result.output


def test_model_override_collapses_agent_model_matrix() -> None:
    raw = {
        "agents": [
            {
                "id": "claude_code",
                "kind": "claude_code",
                "models": ["claude-opus", "glm-5.1"],
            }
        ]
    }

    override_selected_agent_model(
        raw,
        agent_ids=("claude_code",),
        model_id="deepseek-v4-pro",
    )

    assert raw["agents"][0]["model"] == "deepseek-v4-pro"
    assert "models" not in raw["agents"][0]


def test_run_benchmark_id_applies_project_yml_overlays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    models_file = tmp_path / "models.override.yml"
    models_file.write_text(
        """
models:
  base:
    provider: openai
    model: base-model
    api_key: dummy
  override:
    provider: openai
    model: override-model
    api_key: dummy
""",
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=("demo",),
    )

    def fake_resolve_benchmark(value: str) -> BenchmarkSpec:
        assert value == "demo_bench"
        return spec

    captured: dict[str, object] = {}

    def fake_run_experiment(config, **_kwargs):
        captured["config"] = config
        return {
            "experiment": config.name,
            "run_id": config.run_id or "run-demo",
            "agents": {
                config.agents[0].label(): {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "run_dir": str(tmp_path / ".cage_runs" / "demo"),
                }
            },
        }

    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", fake_resolve_benchmark)
    monkeypatch.setattr("cage.experiment.engine.conductor.run_experiment", fake_run_experiment)

    result = CliRunner().invoke(
        main,
        [
            "run",
            "demo_bench",
            "--agent",
            "demo_agent",
            "--model",
            "override",
            "--max-concurrent",
            "4",
            "--models",
            str(models_file),
            "--upstream-proxy",
            "http://10.1.2.146:7890",
            "--timeout",
            "123",
            "--passk",
            "3",
            "--max-sample-num",
            "5",
            "--max-trial-num",
            "7",
            "--max-rounds",
            "9",
            "--max-input-tokens",
            "100000",
            "--max-output-tokens",
            "20000",
            "--max-cost",
            "12.34",
            "--force",
            "--set",
            "eval.benchmark.levels=[0,1,2]",
            "--demo-mode",
            "fast",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.project_file.parent != project_file.parent
    assert config.project_file.name.startswith("cage-effective-")
    assert not config.project_file.exists()
    assert not list(project_file.parent.glob(".cage-effective-*.yml"))
    # Run output belongs under the benchmark's source dir (next to its
    # project.yml), not the cwd the operator happened to launch from.
    assert config.benchmark_dir == project_file.parent.resolve()
    assert config.benchmark.mode == "fast"
    assert config.benchmark.levels == [0, 1, 2]
    assert config.proxy.upstream_http_proxy == "http://10.1.2.146:7890"
    assert config.execution.timeout == 123
    assert config.execution.max_trials_global == 1
    assert config.execution.max_trial == 7
    assert config.execution.passk == 3
    assert config.execution.max_rounds == 9
    assert config.execution.max_input_tokens == 100000
    assert config.execution.max_output_tokens == 20000
    assert config.execution.max_cost == 12.34
    assert config.force is True
    assert config.metadata["benchmark_id"] == "demo_bench"
    assert config.sample_limit == 5
    assert [agent.id for agent in config.agents] == ["demo_agent"]
    assert config.agents[0].model.id == "override"
    assert config.agents[0].max_concurrent == 4
    assert "Experiment:" not in result.output
    assert "View with: cage inspect" not in result.output
    assert "View with: cage show" not in result.output


def test_run_resume_applies_max_concurrent_without_agent_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    _add_second_demo_agent(project_file)
    captured: dict[str, object] = {}

    def fake_run_experiment(config, **_kwargs):
        captured["config"] = config
        return {
            "experiment": config.name,
            "run_id": config.run_id,
            "agents": {},
        }

    monkeypatch.setattr("cage.experiment.engine.conductor.run_experiment", fake_run_experiment)

    result = CliRunner().invoke(
        main,
        [
            "run",
            str(project_file),
            "--run-id",
            "run-fixed",
            "--resume",
            "--max-concurrent",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.resume is True
    assert config.run_id == "run-fixed"
    assert {agent.id: agent.max_concurrent for agent in config.agents} == {
        "demo_agent": 1,
        "demo_agent_2": 1,
    }


def test_run_allow_launch_build_invokes_benchmark_build_hook(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=(),
    )
    captured: dict[str, object] = {}

    def fake_run_benchmark_image_build(project_path: Path, **kwargs):
        captured["build_project_path"] = project_path
        captured["build_kwargs"] = kwargs

    def fake_run_experiment(config, **_kwargs):
        captured["config"] = config
        return {
            "experiment": config.name,
            "run_id": "run-demo",
            "agents": {
                config.agents[0].label(): {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "run_dir": str(tmp_path / ".cage_runs" / "demo"),
                }
            },
        }

    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda _value: spec)
    monkeypatch.setattr(
        "cage.cli.commands._project_prep.run_benchmark_image_build",
        fake_run_benchmark_image_build,
        raising=False,
    )
    monkeypatch.setattr("cage.experiment.engine.conductor.run_experiment", fake_run_experiment)

    result = CliRunner().invoke(
        main,
        [
            "run",
            "demo_bench",
            "--allow-launch-build",
            "--sample",
            "DEMO-1",
            "--max-sample-num",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["build_project_path"] == project_file
    assert captured["build_kwargs"] == {
        "limit": 3,
        "only": ("DEMO-1",),
        "max_workers": 1,
        "dry_run": False,
    }
    assert captured["config"].metadata["launch_build"] == "benchmark-hook"


def test_benchmark_owned_repeatable_options_accept_commas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=(),
    )
    captured: dict[str, object] = {}

    def fake_run_experiment(config, **_kwargs):
        captured["config"] = config
        return {
            "experiment": config.name,
            "run_id": "run-demo",
            "agents": {
                config.agents[0].label(): {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "run_dir": str(tmp_path / ".cage_runs" / "demo"),
                }
            },
        }

    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda _value: spec)
    monkeypatch.setattr("cage.experiment.engine.conductor.run_experiment", fake_run_experiment)

    result = CliRunner().invoke(main, ["run", "demo_bench", "--demo-level", "0,2"])

    assert result.exit_code == 0, result.output
    assert captured["config"].benchmark.levels == [0, 2]


def test_run_force_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    project_file = _write_demo_project(tmp_path)
    _add_second_demo_agent(project_file)

    result = CliRunner().invoke(
        main,
        [
            "run",
            str(project_file),
            "--run-id",
            "run-fixed",
            "--resume",
            "--force",
            "--max-concurrent",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "--force and --resume are mutually exclusive" in result.output
    assert "Cannot apply --max-concurrent" not in result.output


def test_run_rejects_unknown_benchmark_owned_option(tmp_path: Path, monkeypatch) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=(),
    )
    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda _value: spec)

    result = CliRunner().invoke(main, ["run", "demo_bench", "--not-a-real-option", "x"])

    assert result.exit_code != 0
    assert "Unknown run option" in result.output


def test_benchmark_build_resolves_default_project_and_builds_images_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=("demo",),
    )
    captured: dict[str, object] = {}

    def fake_build_targets(project_path: Path, **kwargs):
        captured["project_path"] = project_path
        captured["kwargs"] = kwargs
        return object()

    def fail_check_targets(*_args, **_kwargs):
        raise AssertionError("benchmark build must not launch target checks")

    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda value: spec)
    monkeypatch.setattr(
        "cage.cli.commands._project_prep.run_benchmark_image_build",
        fake_build_targets,
        raising=False,
    )
    monkeypatch.setattr("cage.target.check.check_targets", fail_check_targets)

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "build",
            "demo_bench",
            "--sample",
            "DEMO-1",
            "--only",
            "demo-1",
            "--limit",
            "4",
            "--max-concurrent",
            "4",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["project_path"] == project_file
    assert captured["kwargs"] == {
        "limit": 4,
        "only": ("DEMO-1", "demo-1"),
        "max_workers": 4,
        "dry_run": True,
        "rebuild": False,
    }


def test_benchmark_build_applies_set_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import yaml

    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=(),
    )
    captured: dict[str, object] = {}

    def fake_build_targets(project_path: Path, **kwargs):
        captured["project_path"] = project_path
        captured["raw"] = yaml.safe_load(project_path.read_text(encoding="utf-8"))
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda value: spec)
    monkeypatch.setattr(
        "cage.cli.commands._project_prep.run_benchmark_image_build",
        fake_build_targets,
        raising=False,
    )

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "build",
            "demo_bench",
            "--set",
            "eval.benchmark.levels=[0,1,2]",
            "--sample",
            "DEMO-1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["project_path"] != project_file
    assert Path(captured["project_path"]).name.startswith("cage-effective-")
    assert not Path(captured["project_path"]).exists()
    assert captured["raw"]["eval"]["benchmark"]["levels"] == [0, 1, 2]
    assert captured["kwargs"] == {
        "limit": None,
        "only": ("DEMO-1",),
        "max_workers": 1,
        "dry_run": True,
        "rebuild": False,
    }


def test_benchmark_build_accepts_singular_max_worker_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=(),
    )
    captured: dict[str, object] = {}

    def fake_build_targets(project_path: Path, **kwargs):
        captured["project_path"] = project_path
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda value: spec)
    monkeypatch.setattr(
        "cage.cli.commands._project_prep.run_benchmark_image_build",
        fake_build_targets,
        raising=False,
    )

    result = CliRunner().invoke(
        main,
        ["benchmark", "build", "demo_bench", "--max-concurrent", "8"],
    )

    assert result.exit_code == 0, result.output
    assert captured["project_path"] == project_file
    assert captured["kwargs"] == {
        "limit": None,
        "only": (),
        "max_workers": 8,
        "dry_run": False,
        "rebuild": False,
    }


def test_benchmark_build_help_is_build_only() -> None:
    result = CliRunner().invoke(main, ["benchmark", "build", "--help"])

    assert result.exit_code == 0, result.output
    assert "--sample" in result.output
    assert "--limit" in result.output
    assert "--max-concurrent" in result.output
    assert "--set" in result.output
    assert "--dry-run" in result.output
    assert "--no-build" not in result.output
    assert "--build / --no-build" not in result.output
    assert "--parallel" not in result.output
    assert "--samples-parallel" not in result.output
    assert "--keep" not in result.output
    assert "--readiness-timeout" not in result.output


def test_target_build_selects_canonical_challenge_id_for_prompt_level_sample() -> None:
    samples = [
        {"id": "pb-siyucms-l0", "challenge_id": "pb-siyucms", "content": "demo"},
        {"id": "pb-siyucms-l1", "challenge_id": "pb-siyucms", "content": "demo"},
        {"id": "pb-other-l0", "challenge_id": "pb-other", "content": "demo"},
    ]

    selected = _select_target_samples(samples, only=["pb-SIYUCMS-L0"])

    assert len(selected) == 1
    assert _target_id(selected[0]) == "pb-siyucms"


def test_target_build_plan_explains_sample_mapping_and_no_build(capsys) -> None:
    samples = [
        {"id": "pb-siyucms-l0", "challenge_id": "pb-siyucms", "content": "demo"},
    ]

    _print_target_check_plan(
        samples,
        only=["pb-SIYUCMS-L0"],
        parallel=1,
        samples_parallel=None,
        readiness_timeout=120.0,
        build=False,
    )

    output = capsys.readouterr().out
    assert "Target readiness check" in output
    assert "no agent or model will run" in output
    assert "Launch build: disabled; use cage benchmark build before target checks" in output
    assert "Readiness probe: wait up to 120s for in-network HTTP/HTTPS readiness" in output
    assert "Sample: pb-SIYUCMS-L0 -> target pb-siyucms" in output


def test_benchmark_check_writes_prompt_artifacts_without_html_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    out_dir = tmp_path / "checks"
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=("demo",),
    )
    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda _value: spec)

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "check",
            "demo_bench",
            "--sample",
            "demo-1",
            "--agent",
            "demo_agent",
            "--demo-level",
            "0",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Benchmark: demo_bench" in result.output
    assert "Prompt artifacts directory:" in result.output
    assert "Prompt preview (first 12 lines):" in result.output
    assert "Rendered prompt file:" in result.output
    assert "--- PROMPT" not in result.output
    assert "report.html" not in result.output
    assert not (out_dir / "report.html").exists()
    prompt_file = out_dir / "prompts" / "demo-1" / "prompt.md"
    assert prompt_file.exists()
    assert "Demo prompt for demo-1" in prompt_file.read_text(encoding="utf-8")
    assert (out_dir / "check.json").exists()


def test_benchmark_check_filters_agents_before_auth_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    models_file = tmp_path / "models.yml"
    models_file.write_text(
        f"""
models:
  base:
    provider: openai
    model: base-model
    api_key: dummy
  blocked:
    provider: anthropic
    model: claude-opus
    auth_source: {tmp_path / "missing-claude-auth"}
""",
        encoding="utf-8",
    )
    text = project_file.read_text(encoding="utf-8")
    project_file.write_text(
        text.replace(
            """agents:
  - id: demo_agent
    kind: codex
    model: base
""",
            """agents:
  - id: demo_agent
    kind: codex
    model: base
  - id: blocked_claude
    kind: claude_code
    model: blocked
""",
        ),
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=(),
    )
    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda _value: spec)

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "check",
            "demo_bench",
            "--agent",
            "demo_agent",
            "--sample",
            "demo-1",
            "--out",
            str(tmp_path / "checks"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "agents: 1 (demo_agent:base:stateless)" in result.output
    assert "blocked_claude" not in result.output


def test_targets_check_does_not_validate_agent_auth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    models_file = tmp_path / "models.yml"
    models_file.write_text(
        f"""
models:
  base:
    provider: openai
    model: base-model
    api_key: dummy
  blocked:
    provider: anthropic
    model: claude-opus
    auth_source: {tmp_path / "missing-claude-auth"}
""",
        encoding="utf-8",
    )
    text = project_file.read_text(encoding="utf-8")
    project_file.write_text(
        text.replace(
            """agents:
  - id: demo_agent
    kind: codex
    model: base
""",
            """agents:
  - id: demo_agent
    kind: codex
    model: base
  - id: blocked_claude
    kind: claude_code
    model: blocked
""",
        ),
        encoding="utf-8",
    )

    from cage.target import check as targets_check_module

    class FakeEmbedded:
        server_url = "http://127.0.0.1:1"

        def stop(self) -> None:
            return None

    def fake_launch_and_probe(**kwargs):
        sample = kwargs["sample"]
        return targets_check_module.InstanceResult(
            sample_id=targets_check_module._target_id(sample),
            instance_idx=kwargs["instance_idx"],
            cage_run_id=kwargs["cage_run_id"],
            project_name="demo",
            services_running=1,
            services_total=1,
            passed=True,
        )

    monkeypatch.setattr(
        "cage.experiment.engine.conductor.discover_benchmark_root",
        lambda _benchmark: tmp_path,
    )
    monkeypatch.setattr(
        "cage.experiment.engine.conductor.spawn_embedded_target_server",
        lambda **_kwargs: FakeEmbedded(),
    )
    monkeypatch.setattr(
        targets_check_module,
        "_launch_and_probe",
        fake_launch_and_probe,
    )
    monkeypatch.setattr(
        targets_check_module,
        "_full_teardown",
        lambda **_kwargs: None,
    )

    summary = targets_check_module.check_targets(
        project_file,
        only=["demo-1"],
        samples_parallel=1,
        build=False,
    )

    assert summary.total == 1
    assert summary.passed == 1
    assert summary.failed == 0


def test_targets_check_build_uses_benchmark_hook_not_target_server_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cage.benchmarks import BenchmarkBuildSummary
    project_file = _write_demo_project(tmp_path)

    from cage.target import check as targets_check_module

    captured: dict[str, object] = {}

    class FakeEmbedded:
        server_url = "http://127.0.0.1:1"

        def stop(self) -> None:
            return None

    def fake_build_benchmark_targets(project_path: Path, **kwargs):
        captured["build_project_path"] = project_path
        captured["build_kwargs"] = kwargs
        return BenchmarkBuildSummary(total=0, built=0, skipped=0, failed=0, results=[])

    def fake_spawn_embedded_target_server(**kwargs):
        captured["server_env"] = kwargs["extra_env"]
        return FakeEmbedded()

    def fake_launch_and_probe(**kwargs):
        sample = kwargs["sample"]
        return targets_check_module.InstanceResult(
            sample_id=targets_check_module._target_id(sample),
            instance_idx=kwargs["instance_idx"],
            cage_run_id=kwargs["cage_run_id"],
            project_name="demo",
            services_running=1,
            services_total=1,
            passed=True,
        )

    monkeypatch.setattr(
        targets_check_module,
        "build_benchmark_targets",
        fake_build_benchmark_targets,
        raising=False,
    )
    monkeypatch.setattr(
        "cage.target.provisioning.discover_benchmark_root",
        lambda _benchmark: tmp_path,
    )
    monkeypatch.setattr(
        "cage.target.provisioning.spawn_embedded_target_server",
        fake_spawn_embedded_target_server,
    )
    monkeypatch.setattr(
        targets_check_module,
        "_launch_and_probe",
        fake_launch_and_probe,
    )
    monkeypatch.setattr(
        targets_check_module,
        "_full_teardown",
        lambda **_kwargs: None,
    )

    summary = targets_check_module.check_targets(
        project_file,
        only=["demo-1"],
        samples_parallel=1,
        build=True,
    )

    assert summary.total == 1
    assert summary.passed == 1
    assert captured["build_project_path"] == project_file
    assert captured["build_kwargs"] == {
        "limit": None,
        "only": ["demo-1"],
        "max_workers": 1,
        "dry_run": False,
    }
    assert captured["server_env"] == {"TARGET_SERVER_BUILD_IF_MISSING": "0"}


def test_targets_check_tears_down_each_sample_before_next_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    benchmark_py = tmp_path / "benchmark.py"
    benchmark_py.write_text(
        benchmark_py.read_text(encoding="utf-8").replace(
            'yield {"id": "demo-1", "content": "demo"}',
            'yield {"id": "demo-1", "content": "demo"}\n'
            '        yield {"id": "demo-2", "content": "demo"}',
        ),
        encoding="utf-8",
    )

    from cage.target import check as targets_check_module

    events: list[str] = []

    class FakeEmbedded:
        server_url = "http://127.0.0.1:1"

        def stop(self) -> None:
            events.append("stop-server")

    def fake_launch_and_probe(**kwargs):
        sample_id = targets_check_module._target_id(kwargs["sample"])
        events.append(f"launch:{sample_id}")
        return targets_check_module.InstanceResult(
            sample_id=sample_id,
            instance_idx=kwargs["instance_idx"],
            cage_run_id=kwargs["cage_run_id"],
            project_name=f"project-{sample_id}",
            services_running=1,
            services_total=1,
            passed=True,
        )

    def fake_teardown_targets(**kwargs):
        projects = ",".join(kwargs["project_names"])
        events.append(f"teardown:{projects}")

    monkeypatch.setattr(
        "cage.experiment.engine.conductor.discover_benchmark_root",
        lambda _benchmark: tmp_path,
    )
    monkeypatch.setattr(
        "cage.experiment.engine.conductor.spawn_embedded_target_server",
        lambda **_kwargs: FakeEmbedded(),
    )
    monkeypatch.setattr(
        targets_check_module,
        "_launch_and_probe",
        fake_launch_and_probe,
    )
    monkeypatch.setattr(
        targets_check_module,
        "_teardown_launched_targets",
        fake_teardown_targets,
        raising=False,
    )

    summary = targets_check_module.check_targets(
        project_file,
        samples_parallel=1,
        build=False,
    )

    assert summary.total == 2
    assert events[:4] == [
        "launch:demo-1",
        "teardown:project-demo-1",
        "launch:demo-2",
        "teardown:project-demo-2",
    ]


def test_targets_check_teardown_sweeps_compose_project_labels(monkeypatch) -> None:
    from cage.target import check as targets_check_module

    calls: list[list[str]] = []

    def fake_run_docker(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[:3] == ["docker", "ps", "-aq"]:
            return 0, "container-1\n", ""
        if cmd[:3] == ["docker", "network", "ls"]:
            return 0, "network-1\n", ""
        if cmd[:3] == ["docker", "volume", "ls"]:
            return 0, "volume-1\n", ""
        return 0, "", ""

    monkeypatch.setattr(targets_check_module, "_run_docker", fake_run_docker)

    targets_check_module._teardown_launched_targets(
        launched_run_ids=[],
        project_names=["demo-project"],
    )

    project_filter = "label=com.docker.compose.project=demo-project"
    assert ["docker", "ps", "-aq", "--filter", project_filter] in calls
    assert ["docker", "rm", "-f", "-v", "container-1"] in calls
    assert ["docker", "network", "ls", "-q", "--filter", project_filter] in calls
    assert ["docker", "network", "rm", "network-1"] in calls
    assert ["docker", "volume", "ls", "-q", "--filter", project_filter] in calls
    assert ["docker", "volume", "rm", "-f", "volume-1"] in calls


def test_targets_check_probe_uses_local_python_image_without_pull(monkeypatch) -> None:
    from types import SimpleNamespace

    from cage.target import check_probes as targets_check_module

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="web:80 200\n", stderr="", returncode=0)

    monkeypatch.setattr(targets_check_module.subprocess, "run", fake_run)

    codes = targets_check_module._probe_from_network(
        network="demo-network",
        targets=[("web", 80)],
        deadline=targets_check_module.time.monotonic() + 10,
    )

    assert codes == {"web:80": "200"}
    assert calls
    cmd = calls[0]
    assert "--pull" in cmd
    assert "never" in cmd
    assert "python:3.11-alpine" in cmd
    assert "curlimages/curl:latest" not in cmd
    assert cmd[cmd.index("--entrypoint") + 1] == "python3"


def test_targets_check_probe_uses_head_request(monkeypatch) -> None:
    from types import SimpleNamespace

    from cage.target import check_probes as targets_check_module

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="web:80 200\n", stderr="", returncode=0)

    monkeypatch.setattr(targets_check_module.subprocess, "run", fake_run)

    codes = targets_check_module._probe_from_network(
        network="demo-network",
        targets=[("web", 80)],
        deadline=targets_check_module.time.monotonic() + 10,
    )

    assert codes == {"web:80": "200"}
    assert calls
    script = calls[0][calls[0].index("-c") + 1]
    assert 'method="HEAD"' in script


def test_targets_check_probe_allows_self_signed_https_redirects(monkeypatch) -> None:
    from types import SimpleNamespace

    from cage.target import check_probes as targets_check_module

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout="cpanel:2086 200\n", stderr="", returncode=0)

    monkeypatch.setattr(targets_check_module.subprocess, "run", fake_run)

    codes = targets_check_module._probe_from_network(
        network="demo-network",
        targets=[("cpanel", 2086)],
        deadline=targets_check_module.time.monotonic() + 10,
    )

    assert codes == {"cpanel:2086": "200"}
    assert calls
    script = calls[0][calls[0].index("-c") + 1]
    assert "ssl._create_unverified_context()" in script
    assert "context=context" in script


def test_targets_check_probe_can_return_after_any_ready(monkeypatch) -> None:
    from types import SimpleNamespace

    from cage.target import check_probes as targets_check_module

    def fake_run(cmd, **_kwargs):
        return SimpleNamespace(
            stdout="web:80 200\ndb:5432 000\n",
            stderr="",
            returncode=0,
        )

    def fail_sleep(_seconds):
        raise AssertionError("probe should return before sleeping when any target is ready")

    monkeypatch.setattr(targets_check_module.subprocess, "run", fake_run)
    monkeypatch.setattr(targets_check_module.time, "sleep", fail_sleep)

    codes = targets_check_module._probe_from_network(
        network="demo-network",
        targets=[("web", 80), ("db", 5432)],
        deadline=targets_check_module.time.monotonic() + 10,
        wait_for_all=False,
    )

    assert codes == {"web:80": "200", "db:5432": "000"}


def test_post_targets_check_requires_declared_real_entry_service(monkeypatch) -> None:
    from cage.target import check_probes as targets_check_module

    result = targets_check_module.InstanceResult(
        sample_id="pb-postexp-range-2",
        instance_idx=0,
        cage_run_id="tcheck_demo",
        project_name="demo-project",
        network_name="demo-network",
    )
    sample = {
        "id": "pb-postexp-range-2",
        "source_fields": {
            "entry_services": ["1_langflow", "2_not_vulnerable"],
            "marker_services": ["1_langflow", "3_packagekit"],
        },
    }
    target_data = {
        "services": [
            {"service_name": "1_langflow", "internal_port": 7860},
            {"service_name": "2_not_vulnerable", "internal_port": 80},
        ],
        "network_name": "demo-network",
    }

    def fake_probe_from_network(**kwargs):
        assert kwargs["targets"] == [("1_langflow", 7860)]
        assert kwargs["wait_for_all"] is True
        return {"1_langflow:7860": "000"}

    monkeypatch.setattr(
        targets_check_module,
        "_compose_ps",
        lambda _project: [
            {"Service": "1_langflow", "State": "running"},
            {"Service": "2_not_vulnerable", "State": "running"},
        ],
    )
    monkeypatch.setattr(
        targets_check_module,
        "_docker_inspect_network",
        lambda _network: {"IPAM": {"Config": [{"Subnet": "172.80.1.0/24"}]}},
    )
    monkeypatch.setattr(
        targets_check_module,
        "_probe_from_network",
        fake_probe_from_network,
    )

    targets_check_module._probe_post_exploitation(
        target_data,
        result,
        sample,
        readiness_timeout=120.0,
    )

    assert result.passed is False
    assert result.error == "required entry service unreachable over HTTP/HTTPS: 1_langflow"
    assert result.notes == ["unreachable: 1_langflow 1_langflow:7860 -> 000"]


def test_benchmark_check_matches_sample_ids_case_insensitively(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=(),
    )
    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda _value: spec)

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "check",
            "demo_bench",
            "--sample",
            "DEMO-1",
            "--out",
            str(tmp_path / "checks"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "OK demo-1" in result.output


def test_run_dry_run_matches_sample_ids_case_insensitively(tmp_path: Path) -> None:
    project_file = _write_demo_project(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "run",
            str(project_file),
            "--sample",
            "DEMO-1",
            "--run-id",
            "case-sample",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Planned trials:    1" in result.output
    assert "demo-1" in result.output
    assert "Runtime:" in result.output
    assert "max_trials_global:  1" in result.output
    assert "passk:              1" in result.output


def test_run_dry_run_with_explicit_samples_does_not_import_benchmark(
    tmp_path: Path,
) -> None:
    project_file = _write_demo_project(tmp_path)
    (tmp_path / "benchmark.py").write_text(
        'raise RuntimeError("benchmark module imported during dry-run")\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "run",
            str(project_file),
            "--sample",
            "DEMO-1",
            "--run-id",
            "contract-dry-run",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DRY-RUN: cage run plan" in result.output
    assert "Run ID:              contract-dry-run" in result.output
    assert "Planned trials:    1" in result.output
    assert "demo-1" in result.output
    assert "benchmark module imported during dry-run" not in result.output


def test_benchmark_check_show_prompt_prints_full_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=("demo",),
    )
    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda _value: spec)

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "check",
            "demo_bench",
            "--sample",
            "demo-1",
            "--out",
            str(tmp_path / "checks"),
            "--show-prompt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--- PROMPT demo-1 BEGIN ---" in result.output
    assert "Demo prompt for demo-1" in result.output
    assert "--- PROMPT demo-1 END ---" in result.output


def test_benchmark_check_reports_load_errors_without_traceback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=("demo",),
    )
    monkeypatch.setattr("cage.benchmarks.registry.resolve_benchmark", lambda _value: spec)

    def fail_load(_project_file: Path):
        raise FileNotFoundError("/tmp/missing/challenge.json")

    monkeypatch.setattr("cage.config.experiment.resolve", fail_load)

    result = CliRunner().invoke(main, ["benchmark", "check", "demo_bench"])

    assert result.exit_code != 0
    assert "Unable to load benchmark config for demo_bench" in result.output
    assert "challenge.json" in result.output
    assert "Traceback" not in result.output


def test_run_help_names_project_or_benchmark() -> None:
    result = CliRunner().invoke(main, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert "PROJECT_OR_BENCHMARK" in result.output
    assert "Usage: main run [OPTIONS] PROJECT_FILE" not in result.output


def test_top_level_help_exposes_only_public_command_groups() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    command_names = {
        line.strip().split()[0]
        for line in result.output.splitlines()
        if line.startswith("  ") and line.strip() and not line.strip().startswith("-")
    }
    assert {"agent", "benchmark", "run", "inspect"}.issubset(command_names)
    for hidden in [
        "agents",
        "build",
        "check",
        "cleanup",
        "debug",
        "targets-check",
        "target-debug",
        "serve",
        "dashboard",
        "show",
    ]:
        assert hidden not in command_names


def test_removed_top_level_legacy_commands_are_not_callable() -> None:
    runner = CliRunner()

    for command in [
        "agents",
        "build",
        "check",
        "cleanup",
        "debug",
        "targets-check",
        "target-debug",
        "serve",
        "dashboard",
        "show",
    ]:
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code != 0, command
        assert "No such command" in result.output


def test_agent_group_lists_agent_commands() -> None:
    result = CliRunner().invoke(main, ["agent", "--help"])

    assert result.exit_code == 0, result.output
    assert "list" in result.output
    assert "build" in result.output
    assert "debug" in result.output


def test_agent_list_shows_release_agents_without_internal_state_paths() -> None:
    result = CliRunner().invoke(main, ["agent", "list"])

    assert result.exit_code == 0, result.output
    assert "Agent runtimes" in result.output
    assert "codex" in result.output
    assert "claude_code" in result.output
    assert "qwen_code" in result.output
    assert "kimi_code" in result.output
    assert "cage/codex:latest" in result.output
    assert "cage/codex:pentestenv" in result.output
    assert "Use with: cage run <benchmark> --agent <runner> --model <model-id>" in result.output
    assert "Stable --agent values: codex, claude_code, qwen_code, kimi_code" in result.output
    assert "Models come from config/models.yml" in result.output
    assert "Container images" in result.output
    assert "state_paths" not in result.output
    assert "hermes" not in result.output


def test_agent_list_all_includes_experimental_agents() -> None:
    result = CliRunner().invoke(main, ["agent", "list", "--all"])

    assert result.exit_code == 0, result.output
    assert "hermes" in result.output
    assert "experimental" in result.output


def test_score_benchmark_id_uses_registered_project_and_current_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_file = _write_demo_project(tmp_path)
    spec = BenchmarkSpec(
        id="demo_bench",
        display_name="DemoBench",
        description="Demo benchmark",
        project_file=project_file,
        aliases=("demo",),
    )
    monkeypatch.setattr(SCORE_CLI, "resolve_benchmark", lambda _value: spec)
    monkeypatch.chdir(tmp_path)

    trial_dir = (
        tmp_path
        / ".cage_runs"
        / "demo_agent:base:stateless"
        / "smoke-1"
        / "trials"
        / "demo-1"
    )
    trial_dir.mkdir(parents=True)
    (trial_dir / "task_output.json").write_text(
        """
{
  "trial_id": "demo-1",
  "trial_index": 0,
  "sample": {"id": "demo-1", "content": "demo"},
  "output": "done",
  "exit_code": 0
}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["score", "demo_bench", "--run-id", "smoke-1"],
    )

    assert result.exit_code == 0, result.output
    assert "Benchmark: demo_bench" in result.output
    assert "Run dirs: 1" in result.output
    assert "Found 1 trials" in result.output
    assert "demo=1.00" in result.output
    assert (trial_dir / "scores" / "demo.json").exists()


def test_score_run_dir_uses_saved_project_context(tmp_path: Path) -> None:
    project_file = _write_demo_project(tmp_path)
    run_dir = tmp_path / "copied-run"
    trial_dir = run_dir / "trials" / "demo-1"
    trial_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        f"project_file: {project_file}\n",
        encoding="utf-8",
    )
    (trial_dir / "task_output.json").write_text(
        """
{
  "trial_id": "demo-1",
  "trial_index": 0,
  "sample": {"id": "demo-1", "content": "demo"},
  "output": "done",
  "exit_code": 0
}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["score", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "Scoring run:" in result.output
    assert "Project context:" in result.output
    assert "Scorers: demo" in result.output
    assert "demo=1.00" in result.output
    assert (trial_dir / "scores" / "demo.json").exists()


def test_score_run_dir_discovers_trials_from_canonical_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = _write_demo_project(project_dir)
    spec = load_experiment_spec(project_file, sample_ids=("demo-1",))
    plan = build_experiment_plan(spec)
    run_dir = tmp_path / ".cage_runs" / "demo_agent:base:stateless" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_dir = run_dir / "canonical_outputs" / "demo-1-pass_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "task_output.json").write_text(
        """
{
  "trial_id": "demo-1/pass_1",
  "trial_index": 0,
  "sample": {"id": "demo-1", "content": "demo"},
  "output": "canonical output",
  "exit_code": 0
}
""",
        encoding="utf-8",
    )
    writer.mark_trial_artifact(
        plan.trials[0].trial_id,
        artifact_id="trial.output",
        path=trial_dir / "task_output.json",
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    stray_dir = run_dir / "trials" / "stray"
    stray_dir.mkdir(parents=True)
    (stray_dir / "task_output.json").write_text(
        """
{
  "trial_id": "stray",
  "trial_index": 99,
  "sample": {"id": "stray", "content": "legacy"},
  "output": "stray legacy output",
  "exit_code": 0
}
""",
        encoding="utf-8",
    )
    scorer_file = tmp_path / "score_canonical.py"
    scorer_file.write_text(
        """
from cage.scoring import Scorer
from cage.scoring import Score


class CanonicalScorer(Scorer):
    name = "canonical"

    def score(self, ctx):
        return {
            "canonical": Score(
                value=1.0 if ctx.output == "canonical output" else 0.0,
                answer=ctx.trial_id,
            )
        }
""",
        encoding="utf-8",
    )
    from cage.artifacts.reader import ExperimentArtifactReader

    original_load_snapshot = ExperimentArtifactReader.load_snapshot
    load_snapshot_calls: list[Path] = []

    def load_snapshot_spy(self: ExperimentArtifactReader):
        load_snapshot_calls.append(self.run_dir)
        return original_load_snapshot(self)

    monkeypatch.setattr(ExperimentArtifactReader, "load_snapshot", load_snapshot_spy)

    result = CliRunner().invoke(
        main,
        ["score", str(run_dir), "--scorer", str(scorer_file)],
    )

    assert result.exit_code == 0, result.output
    assert load_snapshot_calls == [run_dir.resolve()]
    assert "Found 1 trials" in result.output
    assert "demo-1: canonical=1.00" in result.output
    expected_score_ref = f"scores/trials/{plan.trials[0].trial_id}/canonical.json"
    assert (run_dir / expected_score_ref).exists()
    assert not (trial_dir / "scores" / "canonical.json").exists()
    assert not (stray_dir / "scores" / "canonical.json").exists()
    record = json.loads((run_dir / "experiment_record.json").read_text(encoding="utf-8"))
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((run_dir / trial_ref).read_text(encoding="utf-8"))
    assert trial_record["scoring"]["status"] == "scored"
    assert trial_record["scoring"]["score_ref"] == expected_score_ref
    assert trial_record["scoring_id"] == "canonical"
    assert record["score_summary"]["status"] == "scored"
    assert record["score_summary"]["summary_ref"] == "scores/summary.json"
    score_summary = json.loads(
        (run_dir / "scores" / "summary.json").read_text(encoding="utf-8")
    )
    assert score_summary["scores"]["canonical"] == {"count": 1, "mean": 1.0}
    artifact_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    summary_artifacts = [
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "scores/summary.json"
    ]
    trial_score_artifacts = [
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == expected_score_ref
    ]
    assert summary_artifacts
    assert summary_artifacts[0]["kind"] == "score_summary"
    assert trial_score_artifacts
    assert trial_score_artifacts[0]["kind"] == "trial_score"


def test_score_run_dir_resolves_project_snapshot_relative_to_original_project_parent(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project-src"
    project_dir.mkdir()
    project_file = _write_demo_project(project_dir)
    run_dir = tmp_path / "copied-run"
    trial_dir = run_dir / "trials" / "demo-1"
    trial_dir.mkdir(parents=True)
    (run_dir / "project.yml").write_text(
        project_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        f"project_file: {project_dir / '.cage-effective-deleted.yml'}\n",
        encoding="utf-8",
    )
    (trial_dir / "task_output.json").write_text(
        """
{
  "trial_id": "demo-1",
  "trial_index": 0,
  "sample": {"id": "demo-1", "content": "demo"},
  "output": "done",
  "exit_code": 0
}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["score", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "Project context:" in result.output
    assert "demo=1.00" in result.output
    assert (trial_dir / "scores" / "demo.json").exists()


def _write_demo_project(tmp_path: Path) -> Path:
    (tmp_path / "benchmark.py").write_text(
        """
from typing import Any, Iterator

from cage.benchmarks import Benchmark, BenchmarkOption
from cage.scoring import Scorer
from cage.scoring import Score


class DemoBench(Benchmark):
    name = "demo"

    def __init__(self, mode: str = "", levels: list[int] | None = None) -> None:
        self.mode = mode
        self.levels = levels or []

    def cli_options(self) -> list[BenchmarkOption]:
        return [
            BenchmarkOption(
                flag="--demo-mode",
                config_path="eval.benchmark.mode",
                choices=("fast", "slow"),
                help="Demo mode.",
            ),
            BenchmarkOption(
                flag="--demo-level",
                config_path="eval.benchmark.levels",
                choices=("0", "1", "2"),
                multiple=True,
                value_type="int",
                help="Demo levels.",
            )
        ]

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        yield {"id": "demo-1", "content": "demo"}

    def prepare_trial(self, container, sample, workspace_dir):
        return None

    def build_prompt(self, sample):
        return f"Demo prompt for {sample['id']} with content {sample['content']}."

    def scorer(self) -> Scorer:
        return DemoScorer()


class DemoScorer(Scorer):
    name = "demo"

    def score(self, ctx):
        return {"demo": Score(value=1.0)}
""",
        encoding="utf-8",
    )
    (tmp_path / "models.yml").write_text(
        """
models:
  base:
    provider: openai
    model: base-model
    api_key: dummy
""",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        """
project:
  name: demo-project
models_file: models.yml
eval:
  benchmark:
    module: ./benchmark.py
    class: DemoBench
agents:
  - id: demo_agent
    kind: codex
    model: base
runtime:
  timeout: 10
  max_trials_global: 1
  passk: 1
  max_input_tokens:
  max_output_tokens:
  max_cost:
""",
        encoding="utf-8",
    )
    return project_file


def _add_second_demo_agent(project_file: Path) -> None:
    text = project_file.read_text(encoding="utf-8")
    project_file.write_text(
        text.replace(
            """agents:
  - id: demo_agent
    kind: codex
    model: base
""",
            """agents:
  - id: demo_agent
    kind: codex
    model: base
  - id: demo_agent_2
    kind: codex
    model: base
""",
        ),
        encoding="utf-8",
    )
