"""Persistent shell — long-lived bash session via host PTY for interactive flows.

The motivating use case is multi-step interactive commands such as:

    run("ssh student@10.10.10.5")     -> ssh asks for a password
    run("hunter2")                     -> password is fed into the same ssh
    run("cat /root/flag")              -> command runs in the remote shell
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import pty
import re
import shlex
import subprocess
import time
from typing import Any, Optional


class PersistentShell:
    """A long-lived bash session inside a container, fronted by a host PTY."""

    PROMPT_TOKEN = "__VBSHELL_RDY_5fb1a7__"
    _PROMPT_BYTES = PROMPT_TOKEN.encode("utf-8")

    INTERACTIVE_PROMPTS = (
        b"password:",
        b"password for ",
        b"[sudo] password",
        b"(yes/no)",
        b"(yes/no/[fingerprint])",
        b"continue connecting (yes/no",
        b"--more--",
        b"[y/n]:",
        b"[y/n]?",
        b"[y/n] ",
        b"[y/N]:",
        b"[y/N] ",
        b"[Y/n]:",
        b"[Y/n] ",
    )

    GENERIC_PROMPT_RE = re.compile(
        rb"[A-Za-z0-9_.\-]+@[A-Za-z0-9_.\-]+(?::[^\r\n]{0,200})?[#$]\s*$"
    )

    def __init__(
        self,
        *,
        container_id: str,
        executable: str = "docker",
        cwd: str = "",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.container_id = container_id
        self.executable = executable
        self.cwd = cwd
        self.logger = logger or logging.getLogger("PersistentShell")
        self._proc: Optional[subprocess.Popen] = None
        self._master_fd: Optional[int] = None
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    def start(self, *, timeout: float = 5.0) -> None:
        master_fd, slave_fd = pty.openpty()
        try:
            import termios
            attrs = termios.tcgetattr(slave_fd)
            attrs[3] &= ~(termios.ECHO | termios.ECHOE | termios.ECHOK | termios.ECHONL)
            attrs[1] &= ~termios.OPOST
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        cmd = [
            self.executable, "exec", "-i", "-t",
            "--env", f"PS1={self.PROMPT_TOKEN}",
            "--env", "PS2=",
            "--env", "PROMPT_COMMAND=",
            "--env", "TERM=dumb",
            "--env", "HISTFILE=/dev/null",
            self.container_id, "bash", "--noprofile", "--norc", "-i",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            os.close(slave_fd)
        self._master_fd = master_fd

        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        self._read_until(self._PROMPT_BYTES, timeout=timeout)

        init_parts = [
            "stty -echo -onlcr 2>/dev/null || true",
            "bind 'set enable-bracketed-paste off' 2>/dev/null || true",
        ]
        if self.cwd:
            init_parts.insert(0, f"cd {shlex.quote(self.cwd)} 2>/dev/null || true")
        self._send("; ".join(init_parts) + "\n")
        self._read_until(self._PROMPT_BYTES, timeout=timeout)

    def is_alive(self) -> bool:
        if self._closed or self._proc is None:
            return False
        return self._proc.poll() is None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._master_fd is not None and self._proc and self._proc.poll() is None:
                try:
                    self._send(b"\x03")
                    self._send(b"exit\n")
                except OSError:
                    pass
                time.sleep(0.05)
        except Exception:
            pass
        try:
            if self._master_fd is not None:
                os.close(self._master_fd)
        except OSError:
            pass
        self._master_fd = None
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def __enter__(self) -> "PersistentShell":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- low-level I/O ------------------------------------------------------

    def _send(self, data: bytes | str) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        if self._master_fd is None:
            return
        try:
            os.write(self._master_fd, data)
        except OSError as exc:
            self.logger.debug("PersistentShell write failed: %s", exc)

    def _read_until(self, needle: bytes, *, timeout: float) -> bytearray:
        import select

        buf = bytearray()
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            r, _, _ = select.select([self._master_fd], [], [], min(0.1, remaining))
            if not r:
                continue
            try:
                chunk = os.read(self._master_fd, 4096)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    continue
                break
            if not chunk:
                break
            buf.extend(chunk)
            if needle in buf:
                return buf
        return buf

    # -- public command interface ------------------------------------------

    def run(
        self,
        command: str,
        *,
        timeout: float = 60.0,
        idle_settle: float = 0.4,
    ) -> dict:
        """Send command and read until shell prompt or interactive prompt is seen."""
        import select as _select

        if not self.is_alive():
            return {
                "output": "[shell closed]",
                "returncode": -1,
                "timed_out": False,
                "waiting_for_input": False,
            }

        cmd_bytes = command.rstrip("\n").encode("utf-8") + b"\n"
        self._send(cmd_bytes)

        deadline = time.time() + timeout
        last_data_time = time.time()
        buf = bytearray()
        prompt_seen = False
        generic_prompt_seen = False
        waiting_for_input = False

        while time.time() < deadline:
            r, _, _ = _select.select([self._master_fd], [], [], 0.1)
            if r:
                try:
                    chunk = os.read(self._master_fd, 4096)
                except OSError as exc:
                    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        continue
                    break
                if chunk:
                    buf.extend(chunk)
                    last_data_time = time.time()

            tail = bytes(buf[-512:])

            if self._PROMPT_BYTES in tail:
                prompt_seen = True
                break

            tail_lower = tail.lower()
            if any(p in tail_lower for p in self.INTERACTIVE_PROMPTS):
                if time.time() - last_data_time > idle_settle:
                    waiting_for_input = True
                    break

            if (
                self.GENERIC_PROMPT_RE.search(tail)
                and time.time() - last_data_time > idle_settle
            ):
                generic_prompt_seen = True
                break

        timed_out = not (prompt_seen or generic_prompt_seen or waiting_for_input)

        if timed_out:
            try:
                self._send(b"\x03")
            except Exception:
                pass
            recovery = self._read_until(self._PROMPT_BYTES, timeout=2.0)
            buf.extend(recovery)
            if self._PROMPT_BYTES in recovery:
                prompt_seen = True

        text = buf.decode("utf-8", errors="replace")
        if prompt_seen:
            idx = text.rfind(self.PROMPT_TOKEN)
            if idx >= 0:
                text = text[:idx]

        returncode = -1
        if prompt_seen and not waiting_for_input:
            returncode = self._fetch_returncode()

        return {
            "output": text,
            "returncode": returncode,
            "timed_out": timed_out,
            "waiting_for_input": waiting_for_input,
        }

    def _fetch_returncode(self) -> int:
        if self._master_fd is None:
            return -1
        marker = "__VBSHELL_RC__"
        try:
            self._send(f"echo {marker}$?\n")
        except Exception:
            return -1
        import select as _select

        deadline = time.time() + 3.0
        buf = bytearray()
        while time.time() < deadline:
            r, _, _ = _select.select([self._master_fd], [], [], 0.1)
            if not r:
                continue
            try:
                chunk = os.read(self._master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)
            if self._PROMPT_BYTES in buf:
                break
        text = buf.decode("utf-8", errors="replace")
        m = re.search(rf"{re.escape(marker)}(-?\d+)", text)
        if not m:
            return -1
        try:
            return int(m.group(1))
        except ValueError:
            return -1
