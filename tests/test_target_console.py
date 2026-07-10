"""Target-range management surface: ``GET /instances`` projection + console.

These exercise the console-facing endpoints without Docker: ``TestClient`` is
built WITHOUT the context-manager form, so FastAPI's lifespan (network setup,
startup GC, health monitor) never fires, and ``probe`` defaults off so no
per-instance health check touches the daemon. The instance registry is an
in-memory dict we seed directly.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from cage.target.server import challenge_server as cs
from cage.target.server import server_state as ss
from cage.target.server.schemas import ServiceInfo


def _seed(**over):
    # The registry stores ServiceInfo objects (see launch.py: final_services =
    # [ServiceInfo(**item) ...]), so seed the same type _build_entry_urls reads.
    base = {
        "chal_id": "SIYUCMS", "run_id": "run-01", "project_name": "p",
        "network_name": "cage_bench_default",
        "network_subnet": "172.30.0.0/24", "network_gateway": "172.30.0.1",
        "services": [ServiceInfo(service_name="web", alias="web", ip="1.2.3.4",
                                 inner_ip="172.30.0.5", internal_port=80,
                                 external_port=32101, external_host="0.0.0.0",
                                 host="9.9.9.9", port=32101, protocol="tcp")],
        "lifecycle_state": "running", "target_scope": "per_agent",
        "cage_run_id": "", "audience": "external",
        "entry_service_keys": ["web"], "created_at": time.time() - 60,
    }
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def _clean_registry():
    with ss.running_instances_lock:
        ss.running_instances.clear()
    yield
    with ss.running_instances_lock:
        ss.running_instances.clear()


def test_project_instance_external_gets_entry_urls_and_uptime():
    info = _seed()
    summ = cs._project_instance(info, audience="external", challenges={}, probe_health=False)
    assert summ.chal_id == "SIYUCMS"
    assert summ.run_id == "run-01"
    assert summ.service_count == 1
    assert summ.healthy is None  # not probed
    assert summ.uptime_s is not None and summ.uptime_s >= 60
    # serve-only network output: the isolated docker network + internal address
    assert summ.network_name == "cage_bench_default"
    assert summ.network_subnet == "172.30.0.0/24"
    assert summ.network_gateway == "172.30.0.1"
    assert summ.container_addr == ["172.30.0.5:80"]  # internal ip:port on the network
    # external audience → published entry url reconstructed from services
    assert len(summ.entry_urls) == 1
    assert summ.entry_urls[0].url.startswith("http://")


def test_project_instance_internal_hides_entry_urls():
    # Same instance, but an INTERNAL caller must not receive host-published URLs.
    # The docker network output (subnet + container_addr) is NOT audience-gated —
    # it is the internal view, which internal callers legitimately have.
    summ = cs._project_instance(_seed(), audience="internal", challenges={}, probe_health=False)
    assert summ.entry_urls == []
    assert summ.container_addr == ["172.30.0.5:80"]
    assert summ.network_subnet == "172.30.0.0/24"


def test_project_instance_pulls_benchmark_from_challenge_meta():
    challenges = {"SIYUCMS": {"benchmark": "agent_pentest_bench", "category": "real-world-web"}}
    summ = cs._project_instance(_seed(), audience="internal", challenges=challenges, probe_health=False)
    assert summ.benchmark == "agent_pentest_bench"
    assert summ.category == "real-world-web"


def test_instances_endpoint_lists_running_targets():
    ss.set_running_instance("run-01", _seed(run_id="run-01"))
    ss.set_running_instance("run-02", _seed(run_id="run-02", chal_id="range-1",
                                            target_scope="per_challenge", audience="internal"))
    client = TestClient(cs.app)  # no context manager → no lifespan/docker
    r = client.get("/instances")
    assert r.status_code == 200
    body = r.json()
    assert {i["run_id"] for i in body} == {"run-01", "run-02"}
    # sorted by (chal_id, run_id) — plain ASCII, so uppercase 'SIYUCMS' < 'range-1'
    assert [i["chal_id"] for i in body] == ["SIYUCMS", "range-1"]


def test_instances_bearer_rejected_when_no_token_configured(monkeypatch):
    monkeypatch.setattr(ss, "EXTERNAL_TOKEN", None, raising=False)
    client = TestClient(cs.app)
    r = client.get("/instances", headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 401


def test_console_index_served_at_root():
    client = TestClient(cs.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "TARGET RANGE" in r.text


def test_submit_is_one_shot_per_instance(monkeypatch):
    # The verdict locks in on the first submit; a repeat returns it unchanged
    # (already_submitted=True) and does NOT re-score, so an agent cannot resubmit
    # to fish for a pass.
    ss.set_running_instance("pb-x_r1", {"chal_id": "pb-x", "services": []})
    monkeypatch.setattr(cs, "load_all_challenges", lambda: {"pb-x": {"id": "pb-x", "full_path": "/x"}})
    calls = {"n": 0}

    def _fake_score(**kwargs):
        calls["n"] += 1
        return {"benchmark_module": "m", "scores": {"s": {"value": float(calls["n"])}}, "run_dir": "rd"}

    monkeypatch.setattr(cs, "score_submission", _fake_score)
    try:
        client = TestClient(cs.app)  # no context manager → no lifespan/docker
        r1 = client.post("/submit/pb-x_r1")
        r2 = client.post("/submit/pb-x_r1")
    finally:
        ss.pop_running_instance("pb-x_r1")

    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["already_submitted"] is False
    assert r1.json()["scores"]["s"]["value"] == 1.0
    assert r2.json()["already_submitted"] is True
    assert r2.json()["scores"]["s"]["value"] == 1.0   # the locked-in first verdict
    assert calls["n"] == 1                            # scored once; the repeat is cached


def test_prompt_endpoint_uses_instance_bound_level(monkeypatch):
    # The hint tier is bound to the instance at launch; GET /prompt renders at
    # THAT level, winning over the server-wide default (env).
    monkeypatch.setenv("TARGET_SERVER_PROMPT_LEVEL", "l0")  # server default
    ss.set_running_instance(
        "pb-x_r1", {"chal_id": "pb-x", "services": [], "prompt_level": "l2"},
    )
    monkeypatch.setattr(
        cs, "load_all_challenges",
        lambda: {"pb-x": {"id": "pb-x", "full_path": "/x", "task_profile": "pentest_remote"}},
    )
    seen = {}

    def _fake_render(run_id, instance, challenge, *, prompt_level="l0"):
        seen["level"] = prompt_level
        return (f"BRIEFING for {challenge['id']} @ {prompt_level}", "BRIEFING @ {{TARGET}}")

    monkeypatch.setattr(cs, "render_task_prompt", _fake_render)
    try:
        client = TestClient(cs.app)
        r = client.get("/prompt/pb-x_r1")
    finally:
        ss.pop_running_instance("pb-x_r1")

    assert r.status_code == 200
    body = r.json()
    assert body["task_prompt"] == "BRIEFING for pb-x @ l2"
    assert body["task_prompt_template"] == "BRIEFING @ {{TARGET}}"
    assert body["prompt_level"] == "l2"       # instance-bound, not the l0 default
    assert body["task_profile"] == "pentest_remote"
    assert seen["level"] == "l2"


def test_prompt_endpoint_falls_back_to_server_default(monkeypatch):
    # An instance launched without a bound level → the server --prompt-level default.
    monkeypatch.setenv("TARGET_SERVER_PROMPT_LEVEL", "l1")
    ss.set_running_instance("pb-x_r1", {"chal_id": "pb-x", "services": []})
    monkeypatch.setattr(
        cs, "load_all_challenges", lambda: {"pb-x": {"id": "pb-x", "full_path": "/x"}}
    )
    monkeypatch.setattr(
        cs, "render_task_prompt",
        lambda *a, prompt_level="l0", **k: (f"@{prompt_level}", "@tmpl"),
    )
    try:
        r = TestClient(cs.app).get("/prompt/pb-x_r1")
    finally:
        ss.pop_running_instance("pb-x_r1")
    assert r.json()["prompt_level"] == "l1"   # fell back to the server default


def test_prompt_endpoint_404_for_unknown_run():
    client = TestClient(cs.app)
    r = client.get("/prompt/nope")
    assert r.status_code == 404
