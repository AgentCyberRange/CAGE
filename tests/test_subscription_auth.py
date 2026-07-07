"""Subscription-auth path for the Claude Code agent.

Two layers:

* Pure unit tests against a fake ``Container`` — verify that
  :meth:`ClaudeCodeAgent.setup_container` writes the right files, with
  the right shape and permissions, when ``auth_source`` is set.

* Real-docker smoke (skipped automatically when the
  ``cage/claude-code:pentestenv`` image is missing or
  ``CAGE_SUBSCRIPTION_SMOKE=1`` is not set) — spins up an actual
  container, seeds it through the production code path, and verifies
  the files land where the CLI expects them.

The smoke test does *not* require network: it only verifies that the
seeding step produced the expected on-disk artifacts. The OAuth round-
trip to api.anthropic.com is out of scope here and was validated
separately by hand.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from cage.agents.claude_code import ClaudeCodeAgent
from cage.models import ModelConfig


def _model(auth_source: str = "", **overrides: object) -> ModelConfig:
    """Convenience: build a subscription-style ModelConfig for tests."""
    base = dict(
        id="claude-sub",
        provider="anthropic",
        model="claude-opus-4-7[xhigh]",
        auth_source=auth_source,
    )
    base.update(overrides)  # type: ignore[arg-type]
    return ModelConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Fake container — records every operation so we can assert against it.
# --------------------------------------------------------------------- #

class _Recorder:
    def __init__(self) -> None:
        self.exec_calls: list[str] = []
        self.writes: dict[str, str] = {}
        self.volumes: dict[str, str] = {}

    def exec(self, cmd: str, timeout: float | None = None) -> object:
        self.exec_calls.append(cmd)

        class _R:
            exit_code = 0
            stdout = ""
            stderr = ""
        return _R()

    def write_file(self, path: str, content: str) -> None:
        self.writes[path] = content


def _make_host_dir(tmpdir: Path, *, with_full_json: bool = True) -> Path:
    """Build a fake host ``~/.claude/`` tree."""
    claude_dir = tmpdir / ".claude"
    claude_dir.mkdir()
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-FAKE",
            "refreshToken": "sk-ant-ort01-FAKE",
            "expiresAt": 9999999999999,
            "scopes": ["user:inference"],
            "subscriptionType": "team",
            "rateLimitTier": "default_claude_max_5",
        },
        "organizationUuid": "11111111-2222-3333-4444-555555555555",
    }))
    if with_full_json:
        (tmpdir / ".claude.json").write_text(json.dumps({
            "oauthAccount": {
                "accountUuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "emailAddress": "test@example.com",
                "organizationUuid": "11111111-2222-3333-4444-555555555555",
                "subscriptionType": "team",
            },
            "hasCompletedOnboarding": True,
            "hasAvailableSubscription": True,
            "numStartups": 42,
            "firstStartTime": "2024-01-01T00:00:00Z",
            "installMethod": "native",
            # Personal data that must NOT leak into the container:
            "projects": {"/home/user/secret": {"history": ["secret prompt"]}},
            "mcpServers": {"private": {"command": "internal-binary"}},
            "cachedGrowthBookFeatures": {"some": "cache"},
        }))
    return tmpdir


# --------------------------------------------------------------------- #
# Unit tests (no docker).
# --------------------------------------------------------------------- #

class TestSetupContainerClassic(unittest.TestCase):
    def test_classic_mode_writes_only_onboarding_stub(self) -> None:
        """``auth_source`` empty → legacy single-stub behaviour preserved."""
        agent = ClaudeCodeAgent()
        rec = _Recorder()
        agent.setup_container(  # type: ignore[arg-type]
            rec, home_dir="/home/agent", model=_model(),
        )
        self.assertEqual(list(rec.writes.keys()), ["/home/agent/.claude.json"])
        marker = json.loads(rec.writes["/home/agent/.claude.json"])
        self.assertEqual(marker, {"hasCompletedOnboarding": True})

    def test_classic_mode_with_no_model_kwarg(self) -> None:
        """``model=None`` (debug paths) still produces a usable stub."""
        rec = _Recorder()
        ClaudeCodeAgent().setup_container(  # type: ignore[arg-type]
            rec, home_dir="/home/agent",
        )
        self.assertEqual(json.loads(rec.writes["/home/agent/.claude.json"]),
                         {"hasCompletedOnboarding": True})


class TestSetupContainerSubscription(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cage-sub-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_subscription_copies_credentials_and_slim_json(self) -> None:
        host = _make_host_dir(self.tmp)
        agent = ClaudeCodeAgent()
        rec = _Recorder()
        agent.setup_container(  # type: ignore[arg-type]
            rec, home_dir="/home/agent",
            model=_model(auth_source=str(host / ".claude")),
        )

        # Both files written.
        self.assertIn("/home/agent/.claude/.credentials.json", rec.writes)
        self.assertIn("/home/agent/.claude.json", rec.writes)

        # Credentials copied verbatim.
        cred = json.loads(rec.writes["/home/agent/.claude/.credentials.json"])
        self.assertEqual(cred["claudeAiOauth"]["accessToken"], "sk-ant-oat01-FAKE")
        self.assertEqual(cred["claudeAiOauth"]["subscriptionType"], "team")
        self.assertEqual(cred["organizationUuid"],
                         "11111111-2222-3333-4444-555555555555")

        # Slim .claude.json contains only the whitelisted keys.
        slim = json.loads(rec.writes["/home/agent/.claude.json"])
        self.assertEqual(set(slim.keys()), {
            "oauthAccount", "hasCompletedOnboarding",
            "hasAvailableSubscription", "numStartups",
            "firstStartTime", "installMethod",
        })

        # Personal data must NOT leak.
        self.assertNotIn("projects", slim)
        self.assertNotIn("mcpServers", slim)
        self.assertNotIn("cachedGrowthBookFeatures", slim)

        # oauthAccount preserved.
        self.assertEqual(slim["oauthAccount"]["emailAddress"], "test@example.com")

        # mkdir + chown/chmod commands executed.
        joined = "\n".join(rec.exec_calls)
        self.assertIn("mkdir -p /home/agent/.claude", joined)
        self.assertIn("chmod 600 /home/agent/.claude/.credentials.json", joined)
        self.assertIn("chown -R agent:agent /home/agent/.claude", joined)

    def test_subscription_container_resources_mount_only_credentials_file(self) -> None:
        host = _make_host_dir(self.tmp)
        cred_path = host / ".claude" / ".credentials.json"
        cred_path.chmod(0o600)

        resources = ClaudeCodeAgent().container_resources(
            home_dir="/home/agent",
            model=_model(auth_source=str(host / ".claude")),
        )

        self.assertEqual(resources.volumes, {
            str(cred_path): "/home/agent/.claude/.credentials.json",
        })
        self.assertNotIn(str(host / ".claude"), resources.volumes)
        self.assertEqual(resources.group_add, [str(cred_path.stat().st_gid)])
        mode = stat.S_IMODE(cred_path.stat().st_mode)
        self.assertTrue(mode & stat.S_IRGRP)
        self.assertTrue(mode & stat.S_IWGRP)

    def test_subscription_setup_skips_credentials_copy_when_mounted(self) -> None:
        host = _make_host_dir(self.tmp)
        cred_path = host / ".claude" / ".credentials.json"
        rec = _Recorder()
        rec.volumes[str(cred_path)] = "/home/agent/.claude/.credentials.json"

        ClaudeCodeAgent().setup_container(  # type: ignore[arg-type]
            rec, home_dir="/home/agent",
            model=_model(auth_source=str(host / ".claude")),
        )

        self.assertNotIn("/home/agent/.claude/.credentials.json", rec.writes)
        self.assertIn("/home/agent/.claude.json", rec.writes)
        joined = "\n".join(rec.exec_calls)
        self.assertIn("mkdir -p /home/agent/.claude", joined)
        self.assertNotIn("chown -R agent:agent /home/agent/.claude", joined)
        self.assertIn("chown agent:agent /home/agent/.claude /home/agent/.claude.json", joined)

    def test_subscription_accepts_parent_dir(self) -> None:
        """``auth_source`` may point at the parent of ``.claude`` too."""
        host = _make_host_dir(self.tmp)
        rec = _Recorder()
        ClaudeCodeAgent().setup_container(  # type: ignore[arg-type]
            rec, home_dir="/home/agent",
            model=_model(auth_source=str(host)),
        )
        self.assertIn("/home/agent/.claude/.credentials.json", rec.writes)
        cred = json.loads(rec.writes["/home/agent/.claude/.credentials.json"])
        self.assertEqual(cred["claudeAiOauth"]["accessToken"], "sk-ant-oat01-FAKE")

    def test_subscription_accepts_custom_config_dir_with_local_marker(self) -> None:
        """``auth_source`` may point at a CLAUDE_CONFIG_DIR-style directory."""
        auth_dir = self.tmp / ".claude_p"
        auth_dir.mkdir()
        (auth_dir / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {"accessToken": "sk-ant-oat01-CUSTOM"},
        }))
        (auth_dir / ".claude.json").write_text(json.dumps({
            "oauthAccount": {
                "accountUuid": "custom-account",
                "emailAddress": "custom@example.com",
                "organizationUuid": "custom-org",
            },
            "hasCompletedOnboarding": True,
            "projects": {"/secret": {"history": ["do not copy"]}},
        }))

        rec = _Recorder()
        ClaudeCodeAgent().setup_container(  # type: ignore[arg-type]
            rec, home_dir="/home/agent",
            model=_model(auth_source=str(auth_dir)),
        )

        slim = json.loads(rec.writes["/home/agent/.claude.json"])
        self.assertEqual(slim["oauthAccount"]["emailAddress"], "custom@example.com")
        self.assertNotIn("projects", slim)

    def test_subscription_missing_claude_json_falls_back_to_stub(self) -> None:
        # .credentials.json present, .claude.json absent.
        claude_dir = self.tmp / ".claude"
        claude_dir.mkdir()
        (claude_dir / ".credentials.json").write_text("{\"claudeAiOauth\":{}}")
        rec = _Recorder()
        ClaudeCodeAgent().setup_container(  # type: ignore[arg-type]
            rec, home_dir="/home/agent",
            model=_model(auth_source=str(claude_dir)),
        )
        slim = json.loads(rec.writes["/home/agent/.claude.json"])
        self.assertEqual(slim, {"hasCompletedOnboarding": True})


class TestEnvVarsSubscription(unittest.TestCase):
    def test_subscription_skips_anthropic_api_key(self) -> None:
        # api_key set on the model entry would normally be injected;
        # ``auth_source`` taking precedence is what proves the
        # subscription branch wins.
        model = _model(auth_source="~/.claude", api_key="sk-should-NOT-leak")
        env = ClaudeCodeAgent().env_vars(
            proxy_url="http://127.0.0.1:8877/v1", model=model,
        )
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        # Proxy URL still threaded — proxy stays in the loop.
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8877/v1")

    def test_classic_anthropic_still_injects_api_key(self) -> None:
        model = ModelConfig(
            id="claude-stock", provider="anthropic",
            model="claude-opus-4-7", api_key="sk-ant-real",
        )
        env = ClaudeCodeAgent().env_vars(
            proxy_url="http://127.0.0.1:8877/v1", model=model,
        )
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-ant-real")

    def test_agent_specific_model_name_is_used_for_claude_code_only(self) -> None:
        model = ModelConfig(
            id="deepseek-v4-pro",
            provider="anthropic",
            model="deepseek-v4-pro",
            api_key="sk-test",
            agent_model_names={"claude_code": "deepseek-v4-pro[1m]"},
        )

        cmd = ClaudeCodeAgent().build_launch_command("hello", model=model)
        env = ClaudeCodeAgent().env_vars(
            proxy_url="http://127.0.0.1:8877/v1", model=model,
        )

        self.assertIn("--model deepseek-v4-pro[1m]", cmd)
        self.assertEqual(
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"],
            "deepseek-v4-pro[1m]",
        )
        self.assertEqual(
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"],
            "deepseek-v4-pro[1m]",
        )
        self.assertEqual(
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"],
            "deepseek-v4-pro[1m]",
        )
        self.assertEqual(model.model, "deepseek-v4-pro")


class TestValidateAuth(unittest.TestCase):
    """Fail-fast validation triggered by ``_build_agent_instance``."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cage-validate-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_passes_when_credentials_exist(self) -> None:
        _make_host_dir(self.tmp)
        ClaudeCodeAgent().validate_auth(
            _model(auth_source=str(self.tmp / ".claude")),
        )

    def test_passes_when_auth_source_empty(self) -> None:
        ClaudeCodeAgent().validate_auth(_model(auth_source=""))

    def test_raises_with_actionable_message(self) -> None:
        missing = self.tmp / "no-such-dir"
        with self.assertRaises(ValueError) as ctx:
            ClaudeCodeAgent().validate_auth(
                _model(auth_source=str(missing)),
            )
        msg = str(ctx.exception)
        self.assertIn("auth_source", msg)
        self.assertIn(".credentials.json", msg)
        self.assertIn("Host:", msg)
        self.assertIn("User:", msg)
        self.assertIn("Tried:", msg)
        self.assertIn("claude /login", msg)
        # Both candidate paths shown so user can pick the right shape.
        self.assertIn(f"{missing}/.credentials.json", msg)
        self.assertIn(f"{missing}/.claude/.credentials.json", msg)

    def test_tilde_is_expanded(self) -> None:
        """``~`` resolves on the orchestrator host, not literally."""
        # Use a fake HOME so the test doesn't depend on the dev's real ~/.
        fake_home = self.tmp / "home"
        (fake_home / ".claude").mkdir(parents=True)
        (fake_home / ".claude" / ".credentials.json").write_text("{}")
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(fake_home)
        try:
            ClaudeCodeAgent().validate_auth(_model(auth_source="~/.claude"))
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home


class TestHostRunServices(unittest.TestCase):
    """The OAuth refresher is declared (and only declared) in subscription mode.

    These assert on the *spec* the agent returns; the orchestrator owns
    actually spawning it, so nothing here starts a process or hits the
    network — and crucially no real OAuth refresh is ever triggered.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cage-hostsvc-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        _make_host_dir(self.tmp)
        self.auth_source = str(self.tmp / ".claude")
        self.cred_path = str(self.tmp / ".claude" / ".credentials.json")

    def test_subscription_declares_refresher(self) -> None:
        from cage.agents.claude_code import _OAUTH_REFRESH_SCRIPT

        svcs = ClaudeCodeAgent().host_run_services(
            _model(auth_source=self.auth_source),
            http_proxy="http://192.0.2.10:7890",
        )
        self.assertEqual(len(svcs), 1)
        svc = svcs[0]
        self.assertEqual(svc.name, "claude-oauth-refresh")
        # One refresher per credentials file (singleton across agents).
        self.assertEqual(svc.dedup_key, self.cred_path)
        # Daemon mode, pointed at the resolved credentials file.
        self.assertIn("--daemon", svc.argv)
        self.assertIn("--creds", svc.argv)
        self.assertEqual(svc.argv[svc.argv.index("--creds") + 1], self.cred_path)
        self.assertEqual(svc.argv[1], str(_OAUTH_REFRESH_SCRIPT))
        # Run's upstream proxy forwarded so it can reach the OAuth endpoint.
        self.assertEqual(svc.env.get("HTTPS_PROXY"), "http://192.0.2.10:7890")
        self.assertEqual(svc.env.get("https_proxy"), "http://192.0.2.10:7890")

    def test_no_proxy_means_no_proxy_env(self) -> None:
        svc = ClaudeCodeAgent().host_run_services(
            _model(auth_source=self.auth_source),
        )[0]
        self.assertEqual(svc.env, {})

    def test_api_key_mode_declares_nothing(self) -> None:
        self.assertEqual(ClaudeCodeAgent().host_run_services(_model()), [])

    def test_custom_base_url_declares_nothing(self) -> None:
        # auth_source present but a non-Anthropic upstream is not the OAuth
        # endpoint, so the refresher would be pointless (and wrong).
        svcs = ClaudeCodeAgent().host_run_services(
            _model(auth_source=self.auth_source, base_url="http://vllm:8000/v1"),
        )
        self.assertEqual(svcs, [])

    def test_default_agent_type_declares_nothing(self) -> None:
        # The ABC default is a no-op, so agents that don't override it (and
        # don't need host-side maintenance) are unaffected by this feature.
        from cage.agents.base import AgentType

        class _Bare(AgentType):
            def install_command(self, version: str = "latest") -> str:
                return ""

            def build_launch_command(
                self, prompt, *, model, max_rounds=-1, proxy_url="",
            ):
                return ""

            def parse_output(self, result):
                return ""

        self.assertEqual(_Bare().host_run_services(_model(auth_source="x")), [])


# --------------------------------------------------------------------- #
# Real-docker smoke (opt-in).
# --------------------------------------------------------------------- #

def _has_targetenv_image() -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", "cage/claude-code:pentestenv"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


@unittest.skipUnless(
    os.environ.get("CAGE_SUBSCRIPTION_SMOKE") == "1" and _has_targetenv_image(),
    "set CAGE_SUBSCRIPTION_SMOKE=1 and ensure cage/claude-code:pentestenv exists",
)
class TestRealDockerSmoke(unittest.TestCase):
    """End-to-end against a real container.

    Verifies the production code path drops the files where the Claude
    Code CLI looks for them. Does NOT call the network — credential
    validity is out of scope for an automated test.
    """

    def setUp(self) -> None:
        self.name = f"cage-sub-smoke-{os.getpid()}"
        # Build a fake host home so we don't read the developer's real
        # OAuth tokens into the smoke test.
        self.host = Path(tempfile.mkdtemp(prefix="cage-sub-smoke-host-"))
        _make_host_dir(self.host)
        self.addCleanup(shutil.rmtree, self.host, ignore_errors=True)
        subprocess.run(["docker", "rm", "-f", self.name],
                       capture_output=True, check=False)
        subprocess.run([
            "docker", "run", "-d", "--name", self.name,
            "cage/claude-code:pentestenv", "sleep", "300",
        ], check=True, capture_output=True)
        self.addCleanup(
            lambda: subprocess.run(
                ["docker", "rm", "-f", self.name],
                capture_output=True, check=False,
            )
        )

    def _exec(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "exec", self.name, *cmd],
            capture_output=True, text=True, check=False,
        )

    def test_real_seed_drops_files_with_right_perms(self) -> None:
        from cage.sandbox.containers import Container

        c = Container(name=self.name, image="cage/claude-code:pentestenv")
        c._started = True  # bypass start(); we already ran `docker run`.
        ClaudeCodeAgent().setup_container(
            c, home_dir="/home/agent",
            model=_model(auth_source=str(self.host / ".claude")),
        )

        cred = self._exec(["cat", "/home/agent/.claude/.credentials.json"])
        self.assertEqual(cred.returncode, 0)
        self.assertIn("sk-ant-oat01-FAKE", cred.stdout)

        marker = self._exec(["cat", "/home/agent/.claude.json"])
        self.assertEqual(marker.returncode, 0)
        slim = json.loads(marker.stdout)
        self.assertIn("oauthAccount", slim)
        self.assertNotIn("projects", slim)

        # Permissions & ownership.
        perm = self._exec(["stat", "-c", "%a:%U:%G",
                           "/home/agent/.claude/.credentials.json"])
        self.assertEqual(perm.stdout.strip(), "600:agent:agent")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
