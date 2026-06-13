from __future__ import annotations

import urllib.error
from types import SimpleNamespace

from click.testing import CliRunner

import cage.target.debug as target_debug
from cage.cli import main


class _Benchmark:
    def __init__(self, samples):
        self.samples = samples
        self.setup_called = False

    def setup(self):
        self.setup_called = True

    def iter_samples_limited(self, limit=None):
        del limit
        yield from self.samples


def test_load_config_and_sample_rejects_unknown_sample(monkeypatch, tmp_path):
    benchmark = _Benchmark([
        {"id": "pb-siyucms"},
        {"id": "pb-prestashop"},
    ])
    monkeypatch.setattr(
        target_debug,
        "resolve",
        lambda project_file: SimpleNamespace(benchmark=benchmark),
    )

    try:
        target_debug._load_config_and_sample(tmp_path / "project.yml", "pb-missing")
    except target_debug.TargetDebugError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected TargetDebugError")

    assert "sample not found: pb-missing" in message
    assert "pb-prestashop" in message
    assert "pb-siyucms" in message
    assert benchmark.setup_called is True


def test_load_config_and_sample_matches_legacy_capitalization(monkeypatch, tmp_path):
    benchmark = _Benchmark([
        {"id": "pb-sample"},
    ])
    monkeypatch.setattr(
        target_debug,
        "resolve",
        lambda project_file: SimpleNamespace(benchmark=benchmark),
    )

    _config, sample, _sample_ids = target_debug._load_config_and_sample(
        tmp_path / "project.yml",
        "PB-SAMPLE",
    )

    assert sample["id"] == "pb-sample"


def test_debug_target_prints_exposed_ports_and_cleans_up(monkeypatch, tmp_path):
    stopped = {"server": False}

    class _Embedded:
        server_url = "http://127.0.0.1:45678"
        process = SimpleNamespace(pid=12345)

        def stop(self):
            stopped["server"] = True

    requests = []

    def fake_request(method, url, *, token="", timeout=None):
        requests.append((method, url, token, timeout))
        if method == "GET":
            assert token
            return {
                "status": "launched",
                "chal_id": "pb-prestashop",
                "run_id": "pb_prestashop_abcd1234",
                "project_name": "cage_bench_debug_pb_prestashop_abcd1234_runtime",
                "network_name": "cage_bench_debug",
                "entry_urls": [
                    {
                        "name": "prestashop",
                        "role": "prestashop",
                        "url": "http://10.0.0.10:57497",
                    }
                ],
                "services": [
                    {
                        "service_name": "prestashop",
                        "external_port": 57497,
                        "internal_port": 80,
                        "protocol": "tcp",
                    },
                    {
                        "service_name": "evaluator",
                        "external_port": 50241,
                        "internal_port": 9091,
                        "protocol": "tcp",
                    },
                ],
            }
        return {}

    lines = []
    monkeypatch.setattr(
        target_debug,
        "_load_config_and_sample",
        lambda project_file, sample_id: (
            SimpleNamespace(benchmark=object()),
            {"id": sample_id},
            [sample_id],
        ),
    )
    monkeypatch.setattr(target_debug, "_benchmark_root", lambda config: tmp_path)
    monkeypatch.setattr(
        target_debug,
        "_runtime_args",
        lambda config, sample: {"target_scope": "per_agent"},
    )
    monkeypatch.setattr(target_debug, "_spawn_server", lambda **kwargs: _Embedded())
    monkeypatch.setattr(target_debug, "_request_json", fake_request)

    result = target_debug.debug_target(
        project_file=tmp_path / "project.yml",
        sample_id="pb-prestashop",
        public_host="10.0.0.10",
        run_id="debug-prestashop",
        wait=False,
        echo=lines.append,
    )

    output = "\n".join(lines)
    assert result["run_id"] == "pb_prestashop_abcd1234"
    assert "Entry URLs:" in output
    assert "prestashop: http://10.0.0.10:57497" in output
    assert "prestashop: 0.0.0.0:57497 -> 80/tcp" in output
    assert "evaluator: 127.0.0.1:50241 -> 9091/tcp" in output
    assert "url=127.0.0.1:50241" in output
    assert stopped["server"] is True
    assert requests[0][0] == "GET"
    assert requests[1][0] == "DELETE"
    assert "run_id=pb_prestashop_abcd1234" in requests[1][1]


def test_wait_for_application_ready_prints_status(monkeypatch):
    monkeypatch.setattr(
        target_debug,
        "_probe_entry_url",
        lambda url: (True, "HTTP 200", False),
    )
    lines = []

    ready = target_debug._wait_for_application_ready(
        sample_id="pb-demo",
        target_data={
            "entry_urls": [
                {"role": "web", "url": "http://10.0.0.10:12345"},
            ],
        },
        public_host="10.0.0.10",
        readiness_timeout=1,
        echo=lines.append,
    )

    output = "\n".join(lines)
    assert ready is True
    assert "Application status: setting up (not ready to use)" in output
    assert "Application status: ready to use" in output


def test_wait_for_wordpress_init_and_probe_useful_urls(monkeypatch):
    states = iter([("running", None), ("exited", 0)])
    probed = []
    lines = []

    monkeypatch.setattr(
        target_debug,
        "_compose_service_state",
        lambda project_name, service_name: next(states),
    )
    monkeypatch.setattr(target_debug.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        target_debug,
        "_fix_wordpress_public_url",
        lambda **kwargs: True,
    )

    def fake_probe(url):
        probed.append(url)
        return True, "HTTP 200", False

    monkeypatch.setattr(target_debug, "_probe_entry_url", fake_probe)

    ready = target_debug._wait_for_application_ready(
        sample_id="pb-wordpress",
        target_data={
            "project_name": "wp_project",
            "entry_urls": [
                {"role": "wordpress", "url": "http://10.0.0.10:12345"},
            ],
            "services": [
                {"service_name": "wordpress", "external_port": 12345, "internal_port": 80},
            ],
        },
        public_host="10.0.0.10",
        readiness_timeout=10,
        echo=lines.append,
    )

    output = "\n".join(lines)
    assert ready is True
    assert "still setting up: wordpress-init running" in output
    assert "WordPress setup complete: wordpress-init exited 0" in output
    assert "http://10.0.0.10:12345/wp-login.php" in probed
    assert "http://10.0.0.10:12345/?pagename=armember-directory" in probed
    assert "http://10.0.0.10:12345/?pagename=pentestbench-everest-contact" in probed
    assert "http://10.0.0.10:12345/index.php?rest_route=/" in probed


def test_wait_for_dify_setup_and_probe_useful_urls(monkeypatch):
    setup_results = iter([(False, "setup status HTTP 502"), (True, "setup finished, login ok")])
    probed = []
    lines = []

    monkeypatch.setattr(
        target_debug,
        "_ensure_dify_setup",
        lambda origin: next(setup_results),
    )
    monkeypatch.setattr(target_debug.time, "sleep", lambda seconds: None)

    def fake_probe(url):
        probed.append(url)
        return True, "HTTP 200", False

    monkeypatch.setattr(target_debug, "_probe_entry_url", fake_probe)

    ready = target_debug._wait_for_application_ready(
        sample_id="pb-dify",
        target_data={
            "entry_urls": [
                {"role": "nginx", "url": "http://10.0.0.10:12345"},
            ],
            "services": [
                {"service_name": "nginx", "external_port": 12345, "internal_port": 80},
            ],
        },
        public_host="10.0.0.10",
        readiness_timeout=10,
        echo=lines.append,
    )

    output = "\n".join(lines)
    assert ready is True
    assert "still setting up: dify setup http://10.0.0.10:12345 -> setup status HTTP 502" in output
    assert "Dify setup complete: setup finished, login ok" in output
    assert "http://10.0.0.10:12345/apps" in probed
    assert "http://10.0.0.10:12345/signin" in probed
    assert "http://10.0.0.10:12345/console/api/setup" in probed
    assert "http://10.0.0.10:12345/console/api/system-features" in probed


def test_print_launch_summary_adds_wordpress_useful_urls():
    lines = []

    target_debug._print_launch_summary(
        echo=lines.append,
        target_data={
            "status": "launched",
            "run_id": "pb_wordpress_abcd",
            "project_name": "wp_project",
            "entry_urls": [
                {"role": "wordpress", "url": "http://10.0.0.10:12345"},
            ],
            "services": [
                {"service_name": "wordpress", "external_port": 12345, "internal_port": 80},
            ],
        },
        sample_id="pb-wordpress",
        cage_run_id="debug-wordpress",
        public_host="10.0.0.10",
        log_path=target_debug.Path("/tmp/wp.log"),
        server_pid=123,
    )

    output = "\n".join(lines)
    assert "Useful URLs:" in output
    assert "login: http://10.0.0.10:12345/wp-login.php" in output
    assert "ARMember directory: http://10.0.0.10:12345/?pagename=armember-directory" in output
    assert (
        "Everest contact form: http://10.0.0.10:12345/?pagename=pentestbench-everest-contact"
        in output
    )
    assert "credentials: admin/admin123, author/author123, victim/victim123" in output


def test_print_launch_summary_adds_dify_useful_urls():
    lines = []

    target_debug._print_launch_summary(
        echo=lines.append,
        target_data={
            "status": "launched",
            "run_id": "pb_dify_abcd",
            "project_name": "dify_project",
            "entry_urls": [
                {"role": "nginx", "url": "http://10.0.0.10:12345"},
            ],
            "services": [
                {"service_name": "nginx", "external_port": 12345, "internal_port": 80},
            ],
        },
        sample_id="pb-dify",
        cage_run_id="debug-dify",
        public_host="10.0.0.10",
        log_path=target_debug.Path("/tmp/dify.log"),
        server_pid=123,
    )

    output = "\n".join(lines)
    assert "Useful URLs:" in output
    assert "apps: http://10.0.0.10:12345/apps" in output
    assert "sign in: http://10.0.0.10:12345/signin" in output
    assert "setup status: http://10.0.0.10:12345/console/api/setup" in output
    assert "credentials: admin@example.com/Admin123!" in output


def test_probe_entry_url_rejects_not_found(monkeypatch):
    class _Opener:
        def open(self, req, timeout=None):
            del req, timeout
            raise urllib.error.HTTPError(
                "http://10.0.0.10:12345",
                404,
                "Not Found",
                {},
                None,
            )

    monkeypatch.setattr(target_debug, "_NO_REDIRECT_OPENER", _Opener())

    ready, note, bad_redirect = target_debug._probe_entry_url("http://10.0.0.10:12345")

    assert ready is False
    assert note == "HTTP 404"
    assert bad_redirect is False


def test_probe_entry_url_rejects_redirect_to_wrong_port(monkeypatch):
    class _Headers:
        def get(self, name, default=None):
            return "http://10.0.0.10:80/" if name == "Location" else default

    class _Opener:
        def open(self, req, timeout=None):
            del req, timeout
            raise urllib.error.HTTPError(
                "http://10.0.0.10:49061",
                301,
                "Moved Permanently",
                _Headers(),
                None,
            )

    monkeypatch.setattr(target_debug, "_NO_REDIRECT_OPENER", _Opener())

    ready, note, bad_redirect = target_debug._probe_entry_url("http://10.0.0.10:49061")

    assert ready is False
    assert "redirects to http://10.0.0.10:80/" in note
    assert bad_redirect is True


def test_agent_input_scheme_overrides_generic_port_guess():
    target_data = {
        "entry_urls": [
            {"role": "wj", "url": "https://10.0.0.10:44444"},
        ],
        "services": [
            {"service_name": "wj", "external_port": 44444, "internal_port": 8443},
        ],
    }

    target_debug._apply_agent_input_entry_urls(
        target_data,
        {"agent_input": {"application_targets": "http://wj:8443"}},
        "10.0.0.10",
    )

    assert target_data["entry_urls"][0]["url"] == "http://10.0.0.10:44444"


def test_target_debug_cli_wires_options(monkeypatch, tmp_path):
    project = tmp_path / "project.yml"
    project.write_text("project: {name: test}\n", encoding="utf-8")
    seen = {}

    def fake_debug_target(**kwargs):
        seen.update(kwargs)
        return {"status": "launched"}

    monkeypatch.setattr(target_debug, "debug_target", fake_debug_target)

    result = CliRunner().invoke(
        main,
        [
            "target-debug",
            str(project),
            "--sample",
            "pb-prestashop",
            "--public-host",
            "10.0.0.10",
            "--run-id",
            "debug-prestashop",
            "--startup-timeout",
            "300",
            "--readiness-timeout",
            "180",
            "--compose-up-timeout",
            "1200",
            "--force-recreate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["project_file"] == str(project)
    assert seen["sample_id"] == "pb-prestashop"
    assert seen["public_host"] == "10.0.0.10"
    assert seen["run_id"] == "debug-prestashop"
    assert seen["startup_timeout"] == 300
    assert seen["readiness_timeout"] == 180
    assert seen["compose_up_timeout"] == 1200
    assert seen["force_recreate"] is True
    assert seen["keep"] is False
