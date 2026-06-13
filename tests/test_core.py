"""Tests for cage core modules."""
import shutil
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
import yaml

from cage.agents.base import AgentInstance, get_agent_type
from cage.benchmarks import Benchmark
from cage.config.experiment import LiveCheckConfig, resolve
from cage.experiment.engine.hooks import HookContext, default_trial_sequence
from cage.models import ModelConfig, load_models
from cage.scoring import Scorer, ScoringContext
from cage.experiment.model import Trial, TrialType
from cage.scoring import Score

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples" / "strongreject"
CONFIG_MODELS = ROOT / "config" / "models.example.yml"


def _materialized_strongreject_project(tmp_path: Path) -> Path:
    import cage.agents.claude_code  # noqa: F401

    repo = tmp_path / "repo"
    project_dir = repo / "examples" / "strongreject"
    shutil.copytree(
        EXAMPLES,
        project_dir,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    config_dir = repo / "config"
    config_dir.mkdir(parents=True)
    shutil.copyfile(ROOT / "config" / "cage.yml", config_dir / "cage.yml")
    shutil.copyfile(CONFIG_MODELS, config_dir / "models.yml")
    return project_dir / "project.yml"


class TestModelConfig:
    def test_protocol_anthropic(self):
        m = ModelConfig(id="x", provider="anthropic", model="claude-3")
        assert m.protocol == "anthropic"

    def test_protocol_vllm(self):
        m = ModelConfig(id="x", provider="vllm", model="llama")
        assert m.protocol == "openai"

    def test_protocol_openai(self):
        m = ModelConfig(id="x", provider="openai", model="gpt-4")
        assert m.protocol == "openai"

    def test_needs_translation(self):
        m = ModelConfig(id="x", provider="vllm", model="llama")
        assert m.needs_translation("anthropic") is True
        assert m.needs_translation("openai") is False

    def test_load_models(self):
        models = load_models(CONFIG_MODELS)
        assert "openai-example" in models
        assert "vllm-example" in models
        assert models["openai-example"].provider == "openai"
        assert models["openai-example"].model == "gpt-5.5"
        assert models["openai-example"].timeout == 360
        assert models["vllm-example"].provider == "vllm"

    def test_api_key_direct(self):
        m = ModelConfig(id="x", provider="vllm", model="llama", api_key="sk-test")
        assert m.api_key == "sk-test"

    def test_default_timeout(self):
        m = ModelConfig(id="x", provider="vllm", model="llama")
        assert m.timeout == 360
        assert m.max_retries == 2

    def test_api_key_pool_scalar(self):
        m = ModelConfig(id="x", provider="vllm", model="llama", api_key="sk-a")
        assert m.api_key_pool == ["sk-a"]

    def test_api_key_pool_empty(self):
        m = ModelConfig(id="x", provider="vllm", model="llama")
        assert m.api_key_pool == []

    def test_api_keys_pool_load(self, tmp_path):
        path = tmp_path / "models.yml"
        path.write_text(
            "models:\n"
            "  pooled:\n"
            "    provider: openai\n"
            "    model: foo\n"
            "    api_keys:\n"
            "      - sk-a\n"
            "      - sk-b\n"
        )
        m = load_models(path)["pooled"]
        # ``api_key`` is populated with the first pool entry so single-key
        # readers (preflight, judge) keep working; the full pool is preserved.
        assert m.api_key == "sk-a"
        assert m.api_key_pool == ["sk-a", "sk-b"]

    def test_api_keys_explicit_scalar_takes_precedence(self, tmp_path):
        path = tmp_path / "models.yml"
        path.write_text(
            "models:\n"
            "  pooled:\n"
            "    provider: openai\n"
            "    model: foo\n"
            "    api_key: sk-explicit\n"
            "    api_keys:\n"
            "      - sk-a\n"
            "      - sk-b\n"
        )
        m = load_models(path)["pooled"]
        assert m.api_key == "sk-explicit"
        assert m.api_key_pool == ["sk-a", "sk-b"]

    def test_model_for_trial_rotation(self):
        from cage.experiment.engine.trial_runner import _model_for_trial

        m = ModelConfig(
            id="x", provider="openai", model="foo",
            api_keys=["sk-a", "sk-b"], api_key="sk-a",
        )
        # Round-robin by trial index; only ``api_key`` changes.
        assert _model_for_trial(m, 0).api_key == "sk-a"
        assert _model_for_trial(m, 1).api_key == "sk-b"
        assert _model_for_trial(m, 2).api_key == "sk-a"
        assert _model_for_trial(m, 3).api_key == "sk-b"
        assert _model_for_trial(m, 1).model == "foo"

    def test_model_for_trial_single_key_noop(self):
        from cage.experiment.engine.trial_runner import _model_for_trial

        m = ModelConfig(id="x", provider="openai", model="foo", api_key="sk-a")
        assert _model_for_trial(m, 5) is m


class _DemoBenchmark(Benchmark):
    name = "demo"

    def __init__(self, samples=None):
        self._samples = samples or [
            {"id": "t0", "content": "task 0"},
            {"id": "t1", "content": "task 1"},
            {"id": "t2", "content": "task 2"},
        ]

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        yield from self._samples

    def prepare_trial(self, container, sample, workspace_dir):
        container.write_file(
            f"{workspace_dir}/note.md",
            str(sample.get("content", "")),
        )

    def build_prompt(self, sample):
        return "Read note.md and follow the instructions."

    def scorer(self) -> Scorer:
        return _DummyScorer()


class _DummyScorer(Scorer):
    name = "dummy"

    def score(self, ctx: ScoringContext) -> dict[str, Score]:
        return {"dummy": Score(value=1.0)}


class TestBenchmark:
    def test_iter_samples(self):
        b = _DemoBenchmark()
        samples = list(b.iter_samples())
        assert len(samples) == 3
        assert samples[0]["id"] == "t0"

    def test_iter_samples_limited(self):
        b = _DemoBenchmark(
            [{"id": f"t{i}", "content": f"c{i}"} for i in range(10)]
        )
        limited = list(b.iter_samples_limited(limit=3))
        assert len(limited) == 3

    def test_iter_samples_slice(self):
        from cage.benchmarks import parse_sample_slice

        b = _DemoBenchmark(
            [{"id": f"t{i}", "content": f"c{i}"} for i in range(20)]
        )
        ids = lambda gen: [s["id"] for s in gen]
        assert ids(b.iter_samples_limited(slice_spec=parse_sample_slice(":3"))) == [
            "t0", "t1", "t2",
        ]
        assert ids(b.iter_samples_limited(slice_spec=parse_sample_slice("-3:"))) == [
            "t17", "t18", "t19",
        ]
        assert ids(b.iter_samples_limited(slice_spec=parse_sample_slice("-3:-1"))) == [
            "t17", "t18",
        ]
        assert ids(b.iter_samples_limited(slice_spec=parse_sample_slice("::10"))) == [
            "t0", "t10",
        ]
        # slice then limit
        assert ids(
            b.iter_samples_limited(limit=2, slice_spec=parse_sample_slice("5:"))
        ) == ["t5", "t6"]
        # id filter applied before slice
        assert ids(
            b.iter_samples_limited(
                sample_ids=["t1", "t2", "t3"],
                slice_spec=parse_sample_slice(":2"),
            )
        ) == ["t1", "t2"]

    def test_parse_sample_slice_forms(self):
        from cage.benchmarks import parse_sample_slice as ps

        lst = list(range(20))
        assert lst[ps(":5")] == lst[:5]
        assert lst[ps("5:10:2")] == lst[5:10:2]
        assert lst[ps("5")] == [lst[5]]
        assert lst[ps("-1")] == [lst[-1]]
        assert ps("") is None and ps(None) is None
        for bad in ("a:b", "1:2:3:4", "::0"):
            with pytest.raises(ValueError):
                ps(bad)

    def test_score(self):
        b = _DemoBenchmark()
        ctx = ScoringContext(trial_id="t0", trial_index=0, sample={"id": "t0", "content": "task"}, output="output")
        scores = b.scorer().score(ctx)
        assert "dummy" in scores
        assert scores["dummy"].value == 1.0

    def test_prepare_trial(self):
        b = _DemoBenchmark()
        container = MagicMock()
        sample = {"id": "t0", "content": "test content"}
        b.prepare_trial(container, sample, "/workspace")
        container.write_file.assert_called_once_with("/workspace/note.md", "test content")

    def test_build_prompt(self):
        b = _DemoBenchmark()
        prompt = b.build_prompt({"id": "t0", "content": "x"})
        assert "note.md" in prompt

    def test_setup_teardown_optional(self):
        b = _DemoBenchmark()
        b.setup()
        b.teardown()

    def test_strongreject_real_dataset(self):
        """Load the real StrongReject benchmark from CSV dataset."""
        import sys
        sys.path.insert(0, str(EXAMPLES))
        try:
            from benchmark import StrongReject
            b = StrongReject()
            b.setup()
            samples = list(b.iter_samples())
            assert len(samples) > 0
            assert all("id" in s for s in samples)
            assert all("content" in s for s in samples)
            assert all("forbidden_prompt" in s for s in samples)
            assert all("category" in s for s in samples)
            print(f"\nLoaded {len(samples)} StrongReject samples")
            print(f"Sample 0: id={samples[0]['id']}, category={samples[0]['category']}")
        finally:
            sys.path.remove(str(EXAMPLES))

    def test_strongreject_keyword_score(self):
        """Test keyword-based scoring fallback."""
        import sys
        sys.path.insert(0, str(EXAMPLES))
        try:
            from benchmark import StrongReject
            b = StrongReject()
            b.setup()
            b._judge = None  # force keyword scoring (no LLM judge)
            sample = {"id": "sr_001", "content": "test", "forbidden_prompt": "test"}
            scorer = b.scorer()

            # Refusal → score 0.0
            ctx = ScoringContext(trial_id="t0", trial_index=0, sample=sample, output="I cannot help with that")
            scores = scorer.score(ctx)
            assert scores["strongreject"].value == 0.0
            assert scores["strongreject"].answer == "refused"

            # Compliance → score 1.0
            ctx = ScoringContext(trial_id="t1", trial_index=1, sample=sample, output="Sure, here is the code...")
            scores = scorer.score(ctx)
            assert scores["strongreject"].value == 1.0
            assert scores["strongreject"].answer == "complied"
        finally:
            sys.path.remove(str(EXAMPLES))


class TestAgentType:
    def test_claude_code_registered(self):
        import cage.agents.claude_code  # noqa: F401
        agent = get_agent_type("claude_code")
        assert agent.name == "claude_code"
        assert ".claude" in agent.state_paths

    def test_codex_registered(self):
        import cage.agents.codex  # noqa: F401
        agent = get_agent_type("codex")
        assert agent.name == "codex"

    def test_unknown_agent_type(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_agent_type("nonexistent_agent")

    def test_agent_instance_stateful_via_shared_paths(self):
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        model = ModelConfig(id="test-model", provider="anthropic", model="claude-3")
        inst = AgentInstance(
            agent_type=at, model=model,
            shared_paths=["/home/agent/.claude"],
        )
        assert inst.stateful is True
        assert "stateful" in inst.label()

    def test_agent_instance_stateless(self):
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        model = ModelConfig(id="m", provider="anthropic", model="x")
        inst = AgentInstance(agent_type=at, model=model)
        assert inst.stateful is False
        assert "stateless" in inst.label()

    def test_agent_instance_id_in_label(self):
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        model = ModelConfig(id="glm", provider="vllm", model="GLM-5.1-sii")
        inst = AgentInstance(agent_type=at, model=model, id="my_baseline")
        assert inst.label().startswith("my_baseline:")

    def test_launch_command_uses_model_field(self):
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        model = ModelConfig(id="glm", provider="vllm", model="GLM-5.1-sii")
        cmd = at.build_launch_command("test prompt", model=model)
        assert "GLM-5.1-sii" in cmd

    def test_agent_instance_max_rounds(self):
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        model = ModelConfig(id="m", provider="vllm", model="x")
        inst = AgentInstance(agent_type=at, model=model, max_rounds=7)
        assert inst.max_rounds == 7

    def test_agent_instance_plugins_default_empty(self):
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        model = ModelConfig(id="m", provider="vllm", model="x")
        inst = AgentInstance(agent_type=at, model=model)
        assert inst.plugins == []

    def test_agent_instance_plugins_set(self):
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        model = ModelConfig(id="m", provider="vllm", model="x")
        inst = AgentInstance(agent_type=at, model=model, plugins=["openviking-memory"])
        assert inst.plugins == ["openviking-memory"]

    def test_claude_code_plugin_installed_detection(self):
        """_plugin_installed returns False for a mock container with no state."""
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        container = MagicMock()
        container.exec.return_value = MagicMock(exit_code=1)
        assert at._plugin_installed(
            container, name="openviking-memory", home_dir="/home/agent",
        ) is False
        container.exec.assert_called_once()

    def test_codex_plugin_installed_detection(self):
        """_plugin_installed returns True if grep matches in config.toml."""
        import cage.agents.codex  # noqa: F401
        at = get_agent_type("codex")
        container = MagicMock()
        container.exec.return_value = MagicMock(exit_code=0)
        assert at._plugin_installed(
            container, name="openviking-memory", home_dir="/home/agent",
        ) is True

    def test_install_plugin_skips_when_installed(self):
        """install_plugin should not call _do_install if already installed."""
        import cage.agents.claude_code  # noqa: F401
        at = get_agent_type("claude_code")
        container = MagicMock()
        # Simulate already installed
        container.exec.return_value = MagicMock(exit_code=0)
        at.install_plugin(container, name="openviking-memory", home_dir="/home/agent")
        # Only the detection grep should have been called, not marketplace add
        assert container.exec.call_count == 1


class TestProjectConfig:
    def test_load_project_yml(self, tmp_path: Path):
        config = resolve(_materialized_strongreject_project(tmp_path))
        assert config.name == "strongreject-cage"
        assert config.project_file.name == "project.yml"
        assert config.benchmark_dir == config.project_file.parent
        assert config.benchmark.name == "strongreject"
        assert config.proxy.enabled is True
        assert "{{ system_raw }}" in config.proxy.rewrite_system
        assert config.judge is not None
        assert config.judge.model.id == "openai-example"
        assert config.judge.temperature == 0.0

    def test_project_logging_terminal_ui_flag(self, tmp_path: Path):
        project_yml = _materialized_strongreject_project(tmp_path)
        source = yaml.safe_load(project_yml.read_text(encoding="utf-8"))
        source["logging"] = {"terminal_ui": False, "inspect_mode": "off"}
        project_yml.write_text(yaml.safe_dump(source), encoding="utf-8")

        config = resolve(project_yml)

        assert config.logging.terminal_ui is False
        assert config.logging.inspect_mode == "off"

    def test_agents_expanded_with_subjects(self, tmp_path: Path):
        config = resolve(_materialized_strongreject_project(tmp_path))
        # 2 agents × 1 subject = 2 agent instances
        assert len(config.agents) == 2
        for agent in config.agents:
            assert agent.model.id == "openai-example"

    def test_agent_models_list_expands_agent_by_model(self, tmp_path: Path):
        project_yml = _materialized_strongreject_project(tmp_path)
        source = yaml.safe_load(project_yml.read_text(encoding="utf-8"))
        source.pop("subjects", None)
        source["agents"] = [
            {
                "id": "claude_code",
                "kind": "claude_code",
                "home": "/home/agent/workspace",
                "models": [
                    "openai-example",
                    {"id": "vllm-example", "max_concurrent": 1},
                ],
                "max_concurrent": 2,
            }
        ]
        project_yml.write_text(yaml.safe_dump(source), encoding="utf-8")

        config = resolve(project_yml)

        assert [agent.id for agent in config.agents] == ["claude_code", "claude_code"]
        assert [agent.model.id for agent in config.agents] == [
            "openai-example",
            "vllm-example",
        ]
        assert [agent.max_concurrent for agent in config.agents] == [2, 1]

    def test_agent_model_multi_source_rotation(self, tmp_path: Path):
        project_yml = _materialized_strongreject_project(tmp_path)
        source = yaml.safe_load(project_yml.read_text(encoding="utf-8"))
        source.pop("subjects", None)
        source["agents"] = [
            {
                "id": "cc",
                "kind": "claude_code",
                "home": "/home/agent/workspace",
                "models": [
                    {
                        "id": "glm-logical",
                        "sources": ["openai-example", "vllm-example"],
                    },
                ],
            }
        ]
        project_yml.write_text(yaml.safe_dump(source), encoding="utf-8")

        config = resolve(project_yml)

        # One run, keyed by the logical id; sources resolved behind it.
        assert len(config.agents) == 1
        agent = config.agents[0]
        assert agent.model.id == "glm-logical"
        assert agent.label() == "cc:glm-logical:stateless"
        assert [m.id for m in agent.model_sources] == [
            "openai-example",
            "vllm-example",
        ]

        from cage.experiment.engine.trial_runner import _trial_model_for_agent

        assert _trial_model_for_agent(agent, 0).id == "openai-example"
        assert _trial_model_for_agent(agent, 1).id == "vllm-example"
        assert _trial_model_for_agent(agent, 2).id == "openai-example"

    def test_agent_model_sources_require_explicit_id(self, tmp_path: Path):
        project_yml = _materialized_strongreject_project(tmp_path)
        source = yaml.safe_load(project_yml.read_text(encoding="utf-8"))
        source.pop("subjects", None)
        source["agents"] = [
            {
                "id": "cc",
                "kind": "claude_code",
                "models": [{"sources": ["openai-example", "vllm-example"]}],
            }
        ]
        project_yml.write_text(yaml.safe_dump(source), encoding="utf-8")
        with pytest.raises(ValueError, match="explicit `id`"):
            resolve(project_yml)

    def test_agent_model_sources_reject_mixed_protocol(self, tmp_path: Path):
        project_yml = _materialized_strongreject_project(tmp_path)
        source = yaml.safe_load(project_yml.read_text(encoding="utf-8"))
        source.pop("subjects", None)
        source["agents"] = [
            {
                "id": "cc",
                "kind": "claude_code",
                # openai-example (openai) + deepseek-v4-pro (anthropic) → reject
                "models": [
                    {"id": "mix", "sources": ["openai-example", "deepseek-v4-pro"]},
                ],
            }
        ]
        project_yml.write_text(yaml.safe_dump(source), encoding="utf-8")
        with pytest.raises(ValueError, match="protocol"):
            resolve(project_yml)

    def test_agent_properties(self, tmp_path: Path):
        config = resolve(_materialized_strongreject_project(tmp_path))
        baseline = config.agents[0]
        assert baseline.id == "claude_code_baseline"
        assert baseline.home == "/home/agent/workspace"
        assert "--verbose" in baseline.session_args

        self_improving = config.agents[1]
        assert self_improving.id == "claude_code_self_improving"
        assert self_improving.skill == "self-improving-agent"
        assert self_improving.shared_paths == ["/home/agent/.claude"]
        assert self_improving.stateful is True

    def test_benchmark_loaded_with_real_data(self, tmp_path: Path):
        config = resolve(_materialized_strongreject_project(tmp_path))
        samples = list(config.benchmark.iter_samples_limited(5))
        assert len(samples) == 5
        assert all("forbidden_prompt" in s for s in samples)

    def test_load_project_proxy_upstream_http_proxy(self, tmp_path: Path):
        project_yml = _materialized_strongreject_project(tmp_path)
        source = yaml.safe_load(project_yml.read_text())
        source["proxy"]["upstream_http_proxy"] = "http://host.docker.internal:7890"

        project_yml.write_text(yaml.safe_dump(source), encoding="utf-8")

        config = resolve(project_yml)

        assert config.proxy.upstream_http_proxy == "http://host.docker.internal:7890"

    def test_benchmark_dir_follows_module_when_project_lives_in_config_subdir(
        self, tmp_path: Path
    ):
        project_yml = _materialized_strongreject_project(tmp_path)
        benchmark_dir = project_yml.parent
        config_dir = benchmark_dir / "configs" / "smoke"
        config_dir.mkdir(parents=True)

        source = yaml.safe_load(project_yml.read_text())
        source["models_file"] = "../../../../config/models.yml"
        source["eval"]["benchmark"]["module"] = "../../benchmark.py"
        nested_project = config_dir / "project.yml"
        nested_project.write_text(yaml.safe_dump(source), encoding="utf-8")

        config = resolve(nested_project)

        assert config.project_file == nested_project.resolve()
        assert config.benchmark_dir == benchmark_dir.resolve()


class TestPluginVolumes:
    def test_resolve_plugin_volumes_from_repo_root(self):
        from cage.experiment.engine.trial_runner import _resolve_plugin_volumes
        # The plugins/ dir is at the cage repo root
        repo_root = Path(__file__).parent.parent
        volumes = _resolve_plugin_volumes(["openviking-memory"], repo_root)
        assert len(volumes) == 1
        key = list(volumes.keys())[0]
        assert key.endswith("openviking-memory-marketplace")
        assert volumes[key] == "/opt/cage-plugins/openviking-memory-marketplace:ro"

    def test_resolve_plugin_volumes_not_found(self):
        from cage.experiment.engine.trial_runner import _resolve_plugin_volumes
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _resolve_plugin_volumes(["nonexistent"], Path("/tmp"))


class TestHooks:
    def test_hook_context_exposes_run_artifacts_dir(self):
        ctx = HookContext(
            experiment_config={},
            samples=[],
            trials_completed=[],
            trials_pending=[],
            run_artifacts_dir=Path("/tmp/demo-run"),
        )

        assert ctx.run_artifacts_dir == Path("/tmp/demo-run")

    def test_default_trial_sequence(self):
        samples = [{"id": f"t{i}", "content": f"c{i}"} for i in range(3)]
        ctx = HookContext(
            experiment_config={},
            samples=samples,
            trials_completed=[],
            trials_pending=[],
        )
        trials = default_trial_sequence(ctx)
        assert len(trials) == 3
        assert all(t.type == TrialType.TASK for t in trials)
        assert trials[0].sample_id == "t0"
        assert trials[2].sample_id == "t2"

    def test_trial_sample_access(self):
        sample = {"id": "sr_001", "content": "test content", "category": "test"}
        trial = Trial(id="trial_0000", index=0, type=TrialType.TASK, sample=sample)
        assert trial.sample_id == "sr_001"
        assert trial.content == "test content"
        assert trial.sample["category"] == "test"


class TestLiveCheckConfig:
    def test_defaults(self):
        cfg = LiveCheckConfig()
        assert cfg.enabled is False
        assert cfg.max_calls == 3
        assert cfg.stop_on_success is True
        assert cfg.reactive.enabled is True
        assert cfg.reactive.check_on_submit is True
        assert cfg.reactive.check_on_9091_call is True
        assert cfg.polling.enabled is False
        assert cfg.polling.interval_seconds == 5.0
        assert cfg.polling.stop_on_success is True

    def test_custom_values(self):
        cfg = LiveCheckConfig(enabled=True, max_calls=5)
        assert cfg.enabled is True
        assert cfg.max_calls == 5

    def test_in_execution_config(self):
        from cage.config.experiment import ExecutionConfig
        ec = ExecutionConfig()
        assert ec.max_target_setups == 1
        assert ec.live_check.enabled is False
        assert ec.live_check.max_calls == 3

    def test_custom_live_check_in_execution_config(self):
        from cage.config.experiment import ExecutionConfig
        ec = ExecutionConfig(live_check=LiveCheckConfig(enabled=True, max_calls=10))
        assert ec.live_check.enabled is True
        assert ec.live_check.max_calls == 10

    def test_load_experiment_live_check_from_yaml(self, tmp_path):
        """Live check config is parsed from YAML runtime.live_check section."""
        import cage.agents.claude_code  # noqa: F401
        models_yaml = tmp_path / "models.yaml"
        models_yaml.write_text(
            "models:\n  test-model:\n    provider: vllm\n    model: test\n"
        )
        bench_py = tmp_path / "benchmark.py"
        bench_py.write_text(
            "from cage.benchmarks import Benchmark\n"
            "from cage.scoring import Scorer\n"
            "from cage.scoring import Score\n"
            "class NoopScorer(Scorer):\n"
            "    name='noop'\n"
            "    def score(self, ctx): return {}\n"
            "class Bench(Benchmark):\n"
            "    name='test_bench'\n"
            "    def iter_samples(self): return iter([])\n"
            "    def prepare_trial(self, c, s, w): pass\n"
            "    def build_prompt(self, s): return ''\n"
            "    def scorer(self): return NoopScorer()\n"
            "    def score(self, o, s, ctx): return {'x': Score(value=0.0)}\n"
        )
        project_yml = tmp_path / "project.yml"
        project_yml.write_text(
            "project:\n  name: test\n"
            "subjects:\n  - test-model\n"
            "eval:\n  benchmark:\n    module: ./benchmark.py\n"
            "agents:\n  - id: a1\n    kind: claude_code\n"
            "target:\n  startup_timeout: 1800\n  compose_up_timeout: 3600\n"
            "runtime:\n  max_target_setups: 2\n  live_check:\n    enabled: true\n    max_calls: 5\n"
            "    stop_on_success: false\n"
            "    reactive:\n      enabled: false\n      check_on_submit: false\n"
            "      check_on_9091_call: false\n"
            "    polling:\n      enabled: true\n      interval_seconds: 2\n"
            "      stop_on_success: false\n"
        )
        config = resolve(project_yml)
        assert config.target.startup_timeout == 1800.0
        assert config.target.compose_up_timeout == 3600.0
        assert config.execution.max_target_setups == 2
        assert config.execution.live_check.enabled is True
        assert config.execution.live_check.max_calls == 5
        assert config.execution.live_check.stop_on_success is False
        assert config.execution.live_check.reactive.enabled is False
        assert config.execution.live_check.reactive.check_on_submit is False
        assert config.execution.live_check.reactive.check_on_9091_call is False
        assert config.execution.live_check.polling.enabled is True
        assert config.execution.live_check.polling.interval_seconds == 2.0
        assert config.execution.live_check.polling.stop_on_success is False

    def test_load_experiment_ignores_removed_max_tool_calls_setting(self, tmp_path):
        """The legacy runtime.max_tool_calls setting is no longer part of execution config."""
        import cage.agents.claude_code  # noqa: F401

        models_yaml = tmp_path / "models.yaml"
        models_yaml.write_text(
            "models:\n  test-model:\n    provider: vllm\n    model: test\n"
        )
        bench_py = tmp_path / "benchmark.py"
        bench_py.write_text(
            "from cage.benchmarks import Benchmark\n"
            "from cage.scoring import Scorer\n"
            "from cage.scoring import Score\n"
            "class NoopScorer(Scorer):\n"
            "    name='noop'\n"
            "    def score(self, ctx): return {}\n"
            "class Bench(Benchmark):\n"
            "    name='test_bench'\n"
            "    def iter_samples(self): return iter([])\n"
            "    def prepare_trial(self, c, s, w): pass\n"
            "    def build_prompt(self, s): return ''\n"
            "    def scorer(self): return NoopScorer()\n"
            "    def score(self, o, s, ctx): return {'x': Score(value=0.0)}\n"
        )
        project_yml = tmp_path / "project.yml"
        project_yml.write_text(
            "project:\n  name: test\n"
            "subjects:\n  - test-model\n"
            "eval:\n  benchmark:\n    module: ./benchmark.py\n"
            "agents:\n  - id: a1\n    kind: claude_code\n"
            "runtime:\n  max_tool_calls: 7\n"
        )

        config = resolve(project_yml)

        assert not hasattr(config.execution, "max_tool_calls")
