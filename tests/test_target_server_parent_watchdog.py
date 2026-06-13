"""The embedded target_server self-exits when its parent cage process dies.

Embedded servers are spawned with ``start_new_session=True`` so a terminal
Ctrl+C can't kill them before cage's controlled teardown. The flip side is that
an *ungraceful* cage death (SIGKILL / OOM) would leave them orphaned — a
never-self-exiting service holding a port — which is how ``cage targets-check``
used to accumulate piles of leftover processes. ``serve.py``'s parent watchdog
closes that gap; this test proves it on the real process topology, without
needing uvicorn or docker.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_embedded_server_self_exits_when_parent_dies(tmp_path: Path) -> None:
    ready_marker = tmp_path / "armed"
    # The watched "server": runs only serve.py's watchdog against its parent,
    # then idles like a long-lived server would. Capture the parent pid on the
    # very first line — before the (multi-second) cage import — exactly as the
    # spawner passes ``--parent-pid``, so the watchdog still fires if the parent
    # dies during startup.
    server_script = tmp_path / "server.py"
    server_script.write_text(
        textwrap.dedent(
            f"""
            import os, sys, threading, time
            parent_pid = os.getppid()
            sys.path.insert(0, {str(_REPO_ROOT)!r})
            from cage.target.serve import _exit_when_orphaned
            threading.Thread(
                target=_exit_when_orphaned,
                args=(parent_pid,),
                kwargs={{"interval": 0.2, "grace": 2.0}},
                daemon=True,
            ).start()
            open({str(ready_marker)!r}, "w").close()
            while True:
                time.sleep(0.1)
            """
        ),
        encoding="utf-8",
    )

    # The stand-in "cage": spawns the server in its own session (exactly like
    # spawn_embedded_target_server), prints its pid, then blocks.
    cage_script = tmp_path / "cage.py"
    cage_script.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys, time
            p = subprocess.Popen(
                [sys.executable, {str(server_script)!r}],
                start_new_session=True,
            )
            print(p.pid, flush=True)
            time.sleep(120)
            """
        ),
        encoding="utf-8",
    )

    cage = subprocess.Popen(
        [sys.executable, str(cage_script)],
        stdout=subprocess.PIPE,
        text=True,
    )
    server_pid = -1
    try:
        line = cage.stdout.readline().strip() if cage.stdout else ""
        server_pid = int(line)
        assert _pid_alive(server_pid), "server did not come up"

        # Wait until the watchdog is armed (covers the slow cage import) so the
        # kill below is deterministic rather than racing startup.
        armed_deadline = time.monotonic() + 15.0
        while time.monotonic() < armed_deadline and not ready_marker.exists():
            time.sleep(0.05)
        assert ready_marker.exists(), "watchdog never armed"

        # Ungraceful parent death: no atexit / finally / signal handler runs.
        cage.kill()
        cage.wait(timeout=5)

        # The orphaned server must reap itself within a couple of watchdog ticks.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _pid_alive(server_pid):
            time.sleep(0.1)
        assert not _pid_alive(server_pid), "embedded server survived parent death"
    finally:
        if cage.poll() is None:
            cage.kill()
        if server_pid > 0 and _pid_alive(server_pid):
            try:
                os.kill(server_pid, 9)
            except OSError:
                pass
