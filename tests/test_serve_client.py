"""Unit tests for the zero-dependency serve client (no server needed)."""
from __future__ import annotations

import io
import json
import tarfile

import pytest

from cage.target import serve_client as sc
from cage.target.serve_client import Instance, ServeClient, ServeError


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self) -> bytes:
        return self._body


def _install_fake_urlopen(monkeypatch, responses, captured):
    def _fake(req, timeout=None):
        captured.append(req)
        payload = responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return _FakeResp(body)

    monkeypatch.setattr(sc.urllib.request, "urlopen", _fake)


# --- helpers --- #


def test_pack_final_answer_roots_at_final_answer(tmp_path):
    fa = tmp_path / "reports"
    fa.mkdir()
    (fa / "vuln-001.json").write_text('{"x": 1}')
    (fa / "vuln-002.json").write_text('{"y": 2}')

    blob = sc._pack_final_answer(fa)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = sorted(tar.getnames())
    assert "final_answer/vuln-001.json" in names
    assert "final_answer/vuln-002.json" in names


def test_pack_final_answer_rejects_non_dir(tmp_path):
    with pytest.raises(ValueError):
        sc._pack_final_answer(tmp_path / "nope")


def test_encode_multipart_file_shape():
    ct, body = sc._encode_multipart_file("agent_output", "submission.tar.gz", b"DATA")
    assert ct.startswith("multipart/form-data; boundary=")
    boundary = ct.split("boundary=")[1]
    assert boundary.encode() in body
    assert b'name="agent_output"; filename="submission.tar.gz"' in body
    assert b"DATA" in body


# --- client flow --- #


def test_headers_carry_token_and_client_id():
    c = ServeClient("http://h:8000", token="secret", client_id="team-red")
    h = c._headers()
    assert h["Authorization"] == "Bearer secret"
    assert h["X-Client-Id"] == "team-red"


def test_launch_builds_per_agent_request_and_returns_instance(monkeypatch):
    captured = []
    _install_fake_urlopen(
        monkeypatch,
        [{"run_id": "pb-x_ab12", "container_addr": ["172.31.0.4:80"],
          "entry_urls": [], "network_name": "net", "network_subnet": "172.31.0.0/24"}],
        captured,
    )
    c = ServeClient("http://h:8000", token="t")
    inst = c.launch("pb-x")

    assert isinstance(inst, Instance)
    assert inst.run_id == "pb-x_ab12"
    assert inst.container_addr == ["172.31.0.4:80"]
    req = captured[0]
    assert req.get_method() == "GET"
    assert "/launch/pb-x?" in req.full_url
    assert "target_scope=per_agent" in req.full_url
    assert "network_only=true" in req.full_url
    assert "prompt_level" not in req.full_url   # omitted → server default
    assert req.headers["Authorization"] == "Bearer t"


def test_launch_binds_prompt_level_when_given(monkeypatch):
    captured = []
    _install_fake_urlopen(monkeypatch, [{"run_id": "r", "container_addr": []}], captured)
    c = ServeClient("http://h:8000")
    c.launch("pb-x", prompt_level="l1")
    assert "prompt_level=l1" in captured[0].full_url


def test_submit_packs_dir_and_posts_multipart(monkeypatch, tmp_path):
    fa = tmp_path / "final_answer"
    fa.mkdir()
    (fa / "v1.json").write_text('{"a": 1}')

    captured = []
    _install_fake_urlopen(monkeypatch, [{"status": "scored", "scores": {"s": {"value": 1.0}}}], captured)
    c = ServeClient("http://h:8000")
    out = c.submit("pb-x_ab12", final_answer_dir=fa)

    assert out["scores"]["s"]["value"] == 1.0
    req = captured[0]
    assert req.get_method() == "POST"
    assert "/submit/pb-x_ab12" in req.full_url
    assert req.headers["Content-type"].startswith("multipart/form-data")
    # body is a gzip tar containing final_answer/v1.json
    start = req.data.find(b"\x1f\x8b")  # gzip magic inside the multipart body
    assert start != -1


def test_submit_without_output_sends_empty_body(monkeypatch):
    captured = []
    _install_fake_urlopen(monkeypatch, [{"status": "scored", "scores": {}}], captured)
    c = ServeClient("http://h:8000")
    c.submit("range_1")  # marker-only: no final_answer
    assert captured[0].get_method() == "POST"


def test_prompt_fetches_task_briefing(monkeypatch):
    captured = []
    _install_fake_urlopen(
        monkeypatch,
        [{
            "run_id": "pb-x_r1", "chal_id": "pb-x", "prompt_level": "l0",
            "task_prompt": "Exploit http://t:80 ...",
            "task_prompt_template": "Exploit {{APPLICATION_TARGETS}} ...",
        }],
        captured,
    )
    c = ServeClient("http://h:8000")
    inst = Instance(
        run_id="pb-x_r1", chal_id="pb-x", container_addr=[], entry_urls=[],
        network_name="", network_subnet="", _client=c,
    )
    # task_prompt() → the ready-to-use string; prompt() → the full dict
    assert inst.task_prompt() == "Exploit http://t:80 ..."
    assert captured[0].get_method() == "GET"
    assert captured[0].full_url.endswith("/prompt/pb-x_r1")


def test_prompt_returns_both_forms(monkeypatch):
    captured = []
    _install_fake_urlopen(
        monkeypatch,
        [{"run_id": "r", "task_prompt": "T", "task_prompt_template": "TT"}],
        captured,
    )
    c = ServeClient("http://h:8000")
    resp = c.prompt("r")
    assert resp["task_prompt"] == "T" and resp["task_prompt_template"] == "TT"


def test_session_launches_then_closes(monkeypatch):
    captured = []
    _install_fake_urlopen(
        monkeypatch,
        [{"run_id": "pb-x_r1", "container_addr": []}, {"status": "stopped"}],
        captured,
    )
    c = ServeClient("http://h:8000")
    with c.session("pb-x") as inst:
        assert inst.run_id == "pb-x_r1"
    assert captured[0].get_method() == "GET"        # launch
    assert captured[1].get_method() == "DELETE"     # close
    assert "run_id=pb-x_r1" in captured[1].full_url


def test_http_error_becomes_serve_error(monkeypatch):
    import urllib.error

    err = urllib.error.HTTPError("http://h/submit/x", 404, "no such run", {}, io.BytesIO(b"no run"))
    captured = []
    _install_fake_urlopen(monkeypatch, [err], captured)
    c = ServeClient("http://h:8000")
    with pytest.raises(ServeError) as ei:
        c.submit("x")
    assert ei.value.status == 404


# --- attach (anti-cheat network join; shells out to docker) --- #


class _FakeProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def _install_fake_docker(monkeypatch, proc, captured):
    import subprocess

    def _fake_run(argv, capture_output=False, text=False):
        captured.append(argv)
        return proc

    monkeypatch.setattr(subprocess, "run", _fake_run)


def test_attach_instance_connects_to_its_network(monkeypatch):
    captured = []
    _install_fake_docker(monkeypatch, _FakeProc(returncode=0), captured)
    c = ServeClient("http://h:8000")
    inst = Instance(
        run_id="pb-x_r1", chal_id="pb-x", container_addr=["172.31.1.2:80"],
        entry_urls=[], network_name="cage_net_r1", network_subnet="172.31.1.0/24",
        _client=c,
    )
    inst.attach("my-agent")
    assert captured == [["docker", "network", "connect", "cage_net_r1", "my-agent"]]


def test_attach_by_run_id_needs_explicit_network(monkeypatch):
    captured = []
    _install_fake_docker(monkeypatch, _FakeProc(returncode=0), captured)
    c = ServeClient("http://h:8000")
    # bare run id + no network → cannot resolve a network, so it must raise
    with pytest.raises(ServeError):
        c.attach("pb-x_r1", "my-agent")
    # …but an explicit network works
    c.attach("pb-x_r1", "my-agent", network="cage_net_r1")
    assert captured == [["docker", "network", "connect", "cage_net_r1", "my-agent"]]


def test_attach_raises_on_docker_failure(monkeypatch):
    captured = []
    _install_fake_docker(
        monkeypatch, _FakeProc(returncode=1, stderr="No such container"), captured
    )
    c = ServeClient("http://h:8000")
    inst = Instance(
        run_id="r", chal_id="pb-x", container_addr=[], entry_urls=[],
        network_name="net", network_subnet="", _client=c,
    )
    with pytest.raises(ServeError) as ei:
        inst.attach("missing")
    assert "No such container" in ei.value.detail
