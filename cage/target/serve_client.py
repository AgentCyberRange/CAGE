"""Zero-dependency Python client for a CAGE benchmark serve (the PULL API).

Wraps the ``list → launch → submit → close`` loop so an external agent drives a
served benchmark programmatically without hand-rolling HTTP/multipart. **Standard
library only** (``urllib`` + ``tarfile``; the optional :meth:`ServeClient.attach`
helper additionally shells out to the ``docker`` CLI), so it drops into any
environment with no install — copy this one file, or ``from
cage.target.serve_client import ServeClient`` if cage is on the path.

    from cage.target.serve_client import ServeClient

    client = ServeClient("http://host:8000", token="…", client_id="team-red")

    for ch in client.list_challenges():
        print(ch["id"], ch["category"])

    # launch an isolated instance: per_agent → your own run_id + docker network.
    # M agents can launch the SAME challenge concurrently and get M independent
    # copies (see the "Concurrency & isolation" section of the API docs).
    inst = client.launch("pb-prestashop")
    print(inst.run_id, inst.container_addr)     # attack the target at container_addr

    # ... produce final_answer/<vuln_id>.json reports under some dir ...
    verdict = client.submit(inst.run_id, final_answer_dir="./final_answer")
    print(verdict["scores"])                    # {scorer: {value, explanation, ...}}

    client.close(inst)

    # …or let a context manager launch-and-close for you:
    with client.session("pb-prestashop") as inst:
        print(inst.submit(final_answer_dir="./final_answer")["scores"])

Marker-only post-exploitation ranges are scored from live target state, so omit
``final_answer_dir`` entirely: ``client.submit(inst.run_id)``.

**Run your agent in a container, not on the host (anti-cheat).** Each launch is
its own isolated docker network. The supported deployment is to run your agent
as a container *on the serve host* and join it to that network — either at start
(``docker run --network <inst.network_name> your-agent``) or after the fact
(:meth:`ServeClient.attach`) — then reach the target as a network peer at
``container_addr`` (or the service DNS alias). Reaching a target from the host
itself (localhost, the bridge IP, or a mounted docker socket) is a cheat the
server cannot prevent: the host routes to every instance's bridge, and docker
socket access lets a process ``exec`` into the target to read flags/markers. The
docker network is host-local, so this strong-isolation path requires the agent
to be co-located with the server; a genuinely remote agent must fall back to the
weaker host-published ``entry_urls`` (``network_only=false``).
"""
from __future__ import annotations

import io
import json
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


class ServeError(RuntimeError):
    """A serve API call failed. ``status`` is the HTTP code (0 = transport)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"serve API error {status}: {message}")
        self.status = status
        self.detail = message


@dataclass
class Instance:
    """A launched, isolated challenge instance — your handle for submit/close."""

    run_id: str
    chal_id: str
    container_addr: list[str]         # internal ip:port of the target(s) on the docker net
    entry_urls: list[dict[str, Any]]  # host-published URLs (only when ?network_only=false)
    network_name: str
    network_subnet: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    _client: "ServeClient | None" = field(repr=False, default=None)

    def submit(self, final_answer_dir=None, archive=None, close=False) -> dict[str, Any]:
        """Score this instance (see :meth:`ServeClient.submit`)."""
        assert self._client is not None
        return self._client.submit(
            self.run_id, final_answer_dir=final_answer_dir, archive=archive, close=close
        )

    def prompt(self) -> dict[str, Any]:
        """Full task-briefing response for THIS instance (see :meth:`ServeClient.prompt`)."""
        assert self._client is not None
        return self._client.prompt(self.run_id)

    def task_prompt(self) -> str:
        """The ready-to-use task briefing for THIS instance — hand it to your agent.

        Convenience for ``prompt()["task_prompt"]``: the briefing with this
        instance's live target address(es) filled in.
        """
        return str(self.prompt().get("task_prompt") or "")

    def attach(self, container: str, *, network: str | None = None) -> None:
        """Join a local container to THIS instance's isolated network (anti-cheat).

        See :meth:`ServeClient.attach`.
        """
        assert self._client is not None
        return self._client.attach(self, container, network=network)

    def close(self) -> dict[str, Any]:
        """Tear this instance down."""
        assert self._client is not None
        return self._client.close(self)


class ServeClient:
    """Client for a ``cage benchmark serve`` server.

    ``token`` is the bearer for an externally-exposed server (omit for loopback).
    ``client_id`` identifies your agent — each distinct id gets its own scored
    experiment on the server, so concurrent teams never mix results.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        client_id: str = "",
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.client_id = client_id
        self.timeout = timeout

    # --- HTTP plumbing --- #

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.client_id:
            headers["X-Client-Id"] = self.client_id
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = self.base_url + path
        if params:
            query = {k: v for k, v in params.items() if v is not None}
            if query:
                url += "?" + urllib.parse.urlencode(query)
        req = urllib.request.Request(
            url, data=data, method=method, headers=self._headers(headers)
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1000]
            raise ServeError(exc.code, detail) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServeError(0, str(exc)) from None
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    # --- endpoints --- #

    def list_challenges(
        self, benchmark: str | None = None, category: str | None = None
    ) -> list[dict[str, Any]]:
        """``GET /challenges`` — public-safe challenge metadata."""
        return self._request(
            "GET", "/challenges", params={"benchmark": benchmark, "category": category}
        )

    def instances(self, probe: bool = False) -> list[dict[str, Any]]:
        """``GET /instances`` — every instance currently running on the server."""
        return self._request(
            "GET", "/instances", params={"probe": "true" if probe else None}
        )

    def launch(
        self,
        chal_id: str,
        *,
        target_scope: str = "per_agent",
        network_only: bool = True,
        force_recreate: bool = False,
        prompt_level: str | None = None,
    ) -> Instance:
        """``GET /launch/{chal_id}`` — start an isolated instance.

        ``per_agent`` (the default) mints a unique ``run_id`` + docker network +
        fresh containers per call, so concurrent launches never collide.
        ``prompt_level`` (``l0``/``l1``/``l2``) binds the task-briefing hint tier
        to THIS instance (read back via :meth:`prompt`); omit it to use the
        server's ``--prompt-level`` default.
        """
        resp = self._request(
            "GET",
            f"/launch/{urllib.parse.quote(chal_id, safe='')}",
            params={
                "target_scope": target_scope,
                "network_only": "true" if network_only else "false",
                "force_recreate": "true" if force_recreate else None,
                "prompt_level": prompt_level,
            },
        )
        resp = resp or {}
        return Instance(
            run_id=str(resp.get("run_id") or ""),
            chal_id=chal_id,
            container_addr=list(resp.get("container_addr") or []),
            entry_urls=list(resp.get("entry_urls") or []),
            network_name=str(resp.get("network_name") or ""),
            network_subnet=str(resp.get("network_subnet") or ""),
            raw=resp,
            _client=self,
        )

    def prompt(self, run_id: str) -> dict[str, Any]:
        """``GET /prompt/{run_id}`` — the agent-facing task briefing for an instance.

        The same task framing a CAGE-managed agent gets: what to exploit, the live
        target address(es), and the ``final_answer`` output contract, rendered by
        the benchmark's own prompt template at the operator-selected hint level.
        Fetch it right after :meth:`launch` so your agent starts with full context.

        Returns the full response ``{run_id, chal_id, task_profile, prompt_level,
        task_prompt, task_prompt_template}``. ``task_prompt`` is the ready-to-use
        briefing (live target filled in) — hand it straight to your agent;
        ``task_prompt_template`` is the same briefing with placeholder targets.
        Use :meth:`Instance.task_prompt` for just the ready string.
        """
        return self._request("GET", f"/prompt/{urllib.parse.quote(run_id, safe='')}") or {}

    def submit(
        self,
        run_id: str,
        *,
        final_answer_dir: str | Path | None = None,
        archive: str | Path | bytes | None = None,
        close: bool = False,
    ) -> dict[str, Any]:
        """``POST /submit/{run_id}`` — score against the still-running instance.

        Provide the agent output as either ``final_answer_dir`` (a dir with
        ``<vuln_id>.json`` reports — packed into the required ``final_answer/``
        archive for you) or a prebuilt ``archive`` (path or bytes). Omit both for
        marker-only post-exploitation ranges (submit is just the "I'm done, score
        my markers" call). ``close=True`` scores then tears the instance down.

        **One submission per instance.** The verdict locks in on the first call;
        a repeat for the same ``run_id`` returns it unchanged with
        ``already_submitted=True`` (you cannot resubmit to fish for a pass —
        launch a fresh instance for another attempt). Note the returned
        ``scores`` may report ``"no judge model configured"`` for vulns needing
        the ``LLM_judge`` signal (e.g. web_exploit_bench) unless the server was
        started with a judge model.
        """
        if archive is not None:
            payload = (
                Path(archive).read_bytes()
                if isinstance(archive, (str, Path))
                else bytes(archive)
            )
        elif final_answer_dir is not None:
            payload = _pack_final_answer(final_answer_dir)
        else:
            payload = b""  # marker-only ranges
        content_type, body = _encode_multipart_file(
            "agent_output", "submission.tar.gz", payload
        )
        return self._request(
            "POST",
            f"/submit/{urllib.parse.quote(run_id, safe='')}",
            params={"close": "true" if close else None},
            data=body,
            headers={"Content-Type": content_type},
        )

    def attach(
        self,
        instance_or_run: "Instance | str",
        container: str,
        *,
        network: str | None = None,
    ) -> None:
        """Join a local container to an instance's isolated docker network.

        **This is the anti-cheat deployment.** Run your agent as a container *on
        the serve host* and attach it to the instance network, so it reaches the
        target as a network peer at ``container_addr`` (or the service DNS alias)
        and can see nothing else — not other instances, not the host. Equivalent
        to launching the agent with ``docker run --network <inst.network_name>``;
        use this to attach an already-running container after :meth:`launch`.

        Reaching a target from the *host* instead (localhost, the bridge IP, or a
        mounted docker socket) is a cheat the server cannot prevent — the host
        routes to every bridge, and the docker socket lets a process ``exec`` into
        the target to read flags/markers. Keep the agent containerized with no
        docker socket.

        Requires the ``docker`` CLI and that this client runs on the **same host**
        as the server (docker networks are host-local). Pass an :class:`Instance`
        (its ``network_name`` is used) or a bare run id plus an explicit
        ``network``. Raises :class:`ServeError` if the connect fails.
        """
        if isinstance(instance_or_run, Instance):
            net = network or instance_or_run.network_name
        else:
            net = network or ""
        if not net:
            raise ServeError(
                0, "no network to attach to — pass an Instance or network=<name>"
            )
        if not container:
            raise ServeError(0, "no container to attach")
        import subprocess

        try:
            proc = subprocess.run(  # noqa: S603,S607 — fixed argv, no shell
                ["docker", "network", "connect", net, container],
                capture_output=True,
                text=True,
            )
        except (OSError, ValueError) as exc:
            raise ServeError(0, f"could not run docker network connect: {exc}") from None
        if proc.returncode != 0:
            raise ServeError(0, f"docker network connect failed: {proc.stderr.strip()}")
        return None

    def close(self, instance_or_run: "Instance | str", run_id: str | None = None) -> dict[str, Any]:
        """``DELETE /launch/{chal_id}?run_id=`` — stop one instance.

        Pass an :class:`Instance`, or ``chal_id`` + ``run_id``.
        """
        if isinstance(instance_or_run, Instance):
            chal_id, rid = instance_or_run.chal_id, instance_or_run.run_id
        else:
            chal_id, rid = instance_or_run, (run_id or "")
        return self._request(
            "DELETE",
            f"/launch/{urllib.parse.quote(chal_id, safe='')}",
            params={"run_id": rid},
        )

    @contextmanager
    def session(self, chal_id: str, **launch_kwargs: Any) -> Iterator[Instance]:
        """Launch an instance and guarantee it is closed on exit."""
        inst = self.launch(chal_id, **launch_kwargs)
        try:
            yield inst
        finally:
            try:
                self.close(inst)
            except ServeError:
                pass


# --- helpers (module-level so they're unit-testable without a server) --- #


def _pack_final_answer(final_answer_dir: str | Path) -> bytes:
    """Pack a report dir into a ``tar.gz`` whose root is ``final_answer/``.

    The server unpacks the archive and reads ``final_answer/*.json``, so the dir
    is added under that name regardless of what it is called on disk.
    """
    src = Path(final_answer_dir)
    if not src.is_dir():
        raise ValueError(f"final_answer_dir is not a directory: {src}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(src), arcname="final_answer")
    return buf.getvalue()


def _encode_multipart_file(field: str, filename: str, content: bytes) -> tuple[str, bytes]:
    """Encode one file field as ``multipart/form-data``. Returns (content_type, body)."""
    boundary = "----cageserve" + uuid.uuid4().hex
    bb = boundary.encode()
    body = b"".join([
        b"--", bb, b"\r\n",
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode(),
        b"\r\n",
        b"Content-Type: application/gzip\r\n\r\n",
        content, b"\r\n",
        b"--", bb, b"--\r\n",
    ])
    return f"multipart/form-data; boundary={boundary}", body
