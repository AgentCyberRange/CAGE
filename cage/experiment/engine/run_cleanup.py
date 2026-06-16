"""Run-scoped resource registry + signal/atexit teardown.

One :class:`RunCleanup` per ``run_experiment`` call owns every resource that must
be released when the run ends — cleanly, on an exception, or on Ctrl+C/SIGTERM:

* the live :class:`ChallengeClient` instances (each closes its per-instance
  ``DELETE /launch`` targets);
* the embedded target_server subprocess;
* host-side per-run services (e.g. an OAuth token refresher);
* the cage-owned agent containers (force-removed to unblock worker threads
  parked in ``docker exec``).

These used to be a handful of module globals plus free functions wired to
``signal``/``atexit``. Collapsing them into one object means the signal handlers
capture *this* instance (not a mutable global), and the conductor threads the
same reference through trial execution. :meth:`teardown_all` is idempotent and
safe to call from the happy path, an exception ``finally``, or a signal handler.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cage.experiment.engine.scheduler import RunScheduler
from cage.target.client import ChallengeClient
from cage.target.provisioning import EmbeddedTargetServer

if TYPE_CHECKING:
    from cage.experiment.engine.run_context import ExperimentRun

logger = logging.getLogger(__name__)


@dataclass
class _ManagedHostService:
    """A host-side background process (e.g. OAuth refresher) owned by the run."""

    name: str
    process: subprocess.Popen

    def stop(self, timeout: float = 5.0) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "host service %s did not exit on SIGTERM; sending SIGKILL",
                    self.name,
                )
                self.process.kill()
                self.process.wait(timeout=2)
        except Exception as exc:
            logger.warning("host service %s stop failed: %s", self.name, exc)


class RunCleanup:
    """Run-scoped resource registry + idempotent teardown for one run.

    The conductor builds one per run, registers resources as they come up, and
    calls :meth:`teardown_all` on exit. Signal/atexit handlers installed by
    :meth:`install_signal_handlers` capture this instance directly.
    """

    def __init__(self, run_id: str, scheduler: RunScheduler) -> None:
        self.run_id = run_id
        self.scheduler = scheduler
        # Live ChallengeClient instances created during the active run. SIGTERM /
        # SIGINT / atexit handlers walk this set to call ``close()`` on each,
        # which iterates the per-instance runtime cache and issues one
        # ``DELETE /launch/<chal_id>?run_id=<rid>`` per target. Anything missed
        # gets label-swept by ``local_cleanup.sweep_run`` afterwards.
        self._clients: "weakref.WeakSet[ChallengeClient]" = weakref.WeakSet()
        self._clients_lock = threading.Lock()
        self._teardown_lock = threading.Lock()
        self._teardown_fired = False
        self._embedded_hub: EmbeddedTargetServer | None = None
        self._host_services: list[_ManagedHostService] = []
        self._host_services_lock = threading.Lock()
        # SIGINT delivery count for the current run. First Ctrl+C does fast
        # unblock + graceful unwind; second forces ``os._exit`` for users who
        # don't want to wait.
        self._sigint_count = 0
        # Latched the instant the forced-quit path begins. A re-entry into the
        # force branch (the user mashing Ctrl+C because "force-quitting now" did
        # not return the prompt instantly) then short-circuits straight to
        # ``os._exit`` — no second banner, no second results table, no repeated
        # teardown. Without this the SIGINT handler is re-entrant during its own
        # slow work and a flurry of presses prints the banner dozens of times
        # and renders the table twice before the process finally exits.
        self._force_in_progress = False
        # Optional zero-arg callback that prints a final status line. The forced
        # exit paths (second Ctrl+C, SIGTERM) ``os._exit`` straight from the
        # signal handler, bypassing the normal end-of-run results table — this
        # hook guarantees the user always sees the final situation first. The
        # conductor binds it to the progress reporter's signal-safe banner.
        self.final_summary_hook: Any = None
        # Optional zero-arg callback that explains the graceful-stop contract on
        # the FIRST Ctrl+C. The handler's ``logger`` message is swallowed by
        # ``quiet_console_logging`` mid-run, so without this the only visible
        # effect is the bar flipping to "stopping" — the user never learns that
        # in-flight trials keep running and a second Ctrl+C force-quits. The
        # conductor binds it to the reporter's signal-safe notice.
        self.first_interrupt_hook: Any = None
        # Optional zero-arg callback that finalizes any trial whose record is
        # still "running" as interrupted. A forced teardown (second Ctrl+C /
        # SIGTERM) kills in-flight trials before they self-finalize, leaving the
        # on-disk record stuck at "running"; this writes the real terminal
        # status so the inspector never shows a phantom "Running" after a dead
        # run. The conductor binds it to the canonical record sweep.
        self.finalize_running_trials_hook: Any = None

    @classmethod
    def inactive(cls) -> "RunCleanup":
        """Return an empty cleanup (no run): teardown is a no-op."""
        return cls("", RunScheduler.inactive())

    # -- registration ---------------------------------------------------------

    def register_client(self, client: ChallengeClient | None) -> None:
        if client is None:
            return
        with self._clients_lock:
            self._clients.add(client)

    @property
    def embedded_hub(self) -> EmbeddedTargetServer | None:
        return self._embedded_hub

    @embedded_hub.setter
    def embedded_hub(self, hub: EmbeddedTargetServer | None) -> None:
        self._embedded_hub = hub

    # -- host services --------------------------------------------------------

    def start_host_services(self, run: ExperimentRun, cage_runs: Path) -> None:
        """Start the host-side services each agent declares for this run.

        Mirrors the embedded target_server lifecycle: spawned here at run start,
        torn down by :meth:`stop_host_services` on the happy path and by
        :meth:`teardown_all` on SIGTERM/SIGINT/atexit. Services are deduplicated
        by ``dedup_key`` so several agents sharing one credential file yield a
        single daemon. A failure to start one service never aborts the run — it
        is logged and the run proceeds.
        """
        http_proxy = getattr(run.proxy, "upstream_http_proxy", "") or ""
        seen: set[str] = set()
        started: list[_ManagedHostService] = []
        for agent in run.agents:
            agent_type = getattr(agent, "agent_type", None)
            model = getattr(agent, "model", None)
            if agent_type is None or model is None:
                continue
            try:
                services = agent_type.host_run_services(model, http_proxy=http_proxy)
            except Exception as exc:
                logger.warning(
                    "host_run_services failed for agent %s: %s", agent.label(), exc,
                )
                continue
            for svc in services or ():
                key = getattr(svc, "dedup_key", "") or svc.name
                if key in seen:
                    continue
                seen.add(key)
                log_path = cage_runs / f"{svc.name}-{self.run_id}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update(getattr(svc, "env", {}) or {})
                log_fh = open(log_path, "ab", buffering=0)
                process = None
                try:
                    process = subprocess.Popen(
                        list(svc.argv),
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        env=env,
                        # New session so SIGTERM/SIGINT delivered to cage don't
                        # kill the daemon before we tear it down deterministically.
                        start_new_session=True,
                    )
                except Exception as exc:
                    logger.warning("failed to start host service %s: %s", svc.name, exc)
                finally:
                    log_fh.close()
                if process is None:
                    continue
                logger.debug(
                    "host service started: %s (pid=%s, log=%s)",
                    svc.name, process.pid, log_path,
                )
                started.append(_ManagedHostService(name=svc.name, process=process))
        if started:
            with self._host_services_lock:
                self._host_services.extend(started)

    def stop_host_services(self) -> None:
        """Terminate every host service started for this run. Idempotent."""
        with self._host_services_lock:
            services = list(self._host_services)
            self._host_services.clear()
        for svc in services:
            svc.stop()
            logger.debug("host service stopped: %s", svc.name)

    # -- container teardown ---------------------------------------------------

    def force_remove_agent_containers(self) -> int:
        """Force-rm cage-owned agent containers tagged ``cage.run_id=<run_id>``.

        Returns the number of container ids passed to ``docker rm -f`` (best
        effort — does not parse rm output). Idempotent. Used by the SIGINT
        handler to unblock worker threads parked in
        ``subprocess.run(["docker", "exec", ...])``: once the target container
        is gone, the host-side ``docker exec`` returns and the worker proceeds
        to its trial cleanup path. Also reused by :meth:`teardown_all` so both
        code paths share the same docker filter logic.
        """
        run_id = self.run_id
        if not run_id:
            return 0
        try:
            ls = subprocess.run(
                [
                    "docker", "ps", "-aq",
                    "--filter", f"label=cage.run_id={run_id}",
                    "--filter", "label=cage.component=agent",
                ],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("force-rm agent: docker ps failed: %s", exc)
            return 0
        ids = [line.strip() for line in (ls.stdout or "").splitlines() if line.strip()]
        if not ids:
            return 0
        try:
            subprocess.run(
                ["docker", "rm", "-f", *ids],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("force-rm agent: docker rm failed: %s", exc)
        return len(ids)

    def teardown_all(self) -> None:
        """Best-effort release of every resource created during this run.

        Idempotent — wired to SIGTERM/SIGINT/atexit, and may also be called from
        the conductor's own ``finally:`` paths. Safe to invoke multiple times.

        Force-removes any leftover cage-owned agent container labelled with this
        ``cage.run_id`` so the target_server networks can finish removing
        themselves (an agent attached to a target network blocks
        ``docker network rm``).
        """
        with self._teardown_lock:
            if self._teardown_fired:
                return
            self._teardown_fired = True

        self.scheduler.request_stop()

        # Force-remove **only** the cage-owned agent containers (those tagged
        # ``cage.component=agent``) before asking the server to clean up. An
        # attached agent endpoint blocks ``docker network rm`` on the server
        # side and the per-instance DELETE /launch requests would each stall
        # past their read timeout. Target stacks share the ``cage.run_id``
        # label but go through the server's graceful path so subnet
        # allocations and instance bookkeeping stay consistent — we
        # deliberately do **not** force-kill them here.
        self.force_remove_agent_containers()

        # The agent containers are gone, so any trial still recorded as
        # "running" was killed mid-flight and will never self-finalize. Write
        # its terminal interrupted status now (before os._exit) so the on-disk
        # record matches reality.
        self._finalize_running_trials()

        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.close()
            except Exception as exc:
                logger.warning("teardown: ChallengeClient.close failed: %s", exc)

        # Stop the embedded target_server subprocess (if any) last — it relies
        # on being alive while ChallengeClient.close() iterates per-instance
        # DELETEs.
        hub = self._embedded_hub
        if hub is not None:
            try:
                hub.stop()
            except Exception as exc:
                logger.warning("teardown: embedded target_server stop failed: %s", exc)
            self._embedded_hub = None

        # Stop host-side per-run services (e.g. the OAuth refresher) too.
        # Idempotent with the happy-path ``stop_host_services`` call.
        self.stop_host_services()

        # Belt-and-suspenders: even after per-instance DELETE /launch teardown,
        # sweep any docker resource still tagged with our run id. Catches cases
        # where a DELETE never landed (server crashed, network error, kill -9,
        # bug inside ``_cleanup_instance_impl``). Without this the
        # ``cage_bench_run_<ts>_*`` containers/networks would survive ``cage
        # run`` exit and eventually exhaust host resources (AIO, ports,
        # networks). Equivalent to targeted ``cage gc --run-id <id> --apply`` —
        # automated.
        run_id = self.run_id
        if run_id:
            try:
                from cage.target.local_cleanup import sweep_run
                res = sweep_run(run_id)
                if res.removed_anything:
                    logger.warning(
                        "teardown: local sweep removed %d container(s) + %d "
                        "network(s) for run_id=%s (per-instance DELETE missed these)",
                        res.containers_removed, res.networks_removed, run_id,
                    )
                for err in res.errors or ():
                    logger.warning("teardown: local sweep error: %s", err)
            except Exception as exc:
                logger.warning("teardown: local sweep failed: %s", exc)

    # -- signal handling ------------------------------------------------------

    def _emit_final_summary(self) -> None:
        """Best-effort final status line before a forced ``os._exit``."""
        hook = self.final_summary_hook
        if hook is None:
            return
        try:
            hook()
        except Exception:  # noqa: BLE001
            pass

    def _emit_first_interrupt_notice(self) -> None:
        """Best-effort graceful-stop explainer on the first Ctrl+C."""
        hook = self.first_interrupt_hook
        if hook is None:
            return
        try:
            hook()
        except Exception:  # noqa: BLE001
            pass

    def _finalize_running_trials(self) -> None:
        """Best-effort: persist interrupted status for force-killed trials."""
        hook = self.finalize_running_trials_hook
        if hook is None:
            return
        try:
            hook()
        except Exception:  # noqa: BLE001
            pass

    def _spawn_background_sweep(self) -> None:
        """Detached, fire-and-forget removal of this run's docker resources.

        The forced-quit handler must hand the prompt back within the blink
        between two Ctrl+C presses, but ``docker rm -f`` of every in-flight
        agent container takes seconds (it was the dominant cost of the old
        synchronous teardown). Launch that work in its own session so it
        outlives the immediate ``os._exit`` and the foreground returns at once.

        Scoped to ``cage.run_id`` exactly like :func:`sweep_run` (containers +
        networks + volumes). Best-effort — its errors are irrelevant; a later
        ``cage gc`` or the next run's sweep reclaims any straggler. Spawned with
        a single non-blocking ``Popen`` so the foreground cost is a few
        milliseconds.
        """
        run_id = self.run_id
        if not run_id:
            return
        flt = f'label=cage.run_id={run_id}'
        sweep = (
            f'ids=$(docker ps -aq --filter "{flt}"); '
            f'[ -n "$ids" ] && docker rm -f -v $ids; '
            f'nets=$(docker network ls -q --filter "{flt}"); '
            f'[ -n "$nets" ] && docker network rm $nets; '
            f'vols=$(docker volume ls -q --filter "{flt}"); '
            f'[ -n "$vols" ] && docker volume rm -f $vols'
        )
        try:
            subprocess.Popen(
                ["bash", "-c", sweep],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                # Survive the parent's ``os._exit`` and detach from the terminal
                # process group so the next Ctrl+C can't reach it.
                start_new_session=True,
            )
        except Exception:  # noqa: BLE001
            pass

    def _stop_host_services_nonblocking(self) -> None:
        """SIGTERM host-side per-run daemons without waiting (force-quit path).

        :meth:`stop_host_services` waits up to 5s per service; the forced-quit
        path can't block on that. Send SIGTERM and move on. The embedded
        target_server is not touched here — it self-terminates once it notices
        cage has reparented it (see ``target/serve.py``).
        """
        with self._host_services_lock:
            services = list(self._host_services)
            self._host_services.clear()
        for svc in services:
            try:
                svc.process.terminate()
            except Exception:  # noqa: BLE001
                pass

    def install_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT/atexit handlers that drive :meth:`teardown_all`.

        First SIGINT is graceful: it requests stop on the run scheduler and
        flags the progress bar as stopping, but does NOT kill containers or
        raise. Queued trials bail (via ``_TrialCancelled`` in the conductor,
        before any container launch) while the in-flight trial(s) finish, so the
        run drains and returns on its own — partial dashboard preserved, target
        stacks torn down on the normal exit path.

        Second SIGINT force-quits and must return the prompt *instantly* — fast
        enough to land inside the gap between two keypresses. So it does only the
        cheap, user-facing work in the foreground (print the force banner + the
        partial results table once, finalize trial records) and hands the slow
        ``docker rm -f`` of every in-flight container to a detached background
        sweep (:meth:`_spawn_background_sweep`) before ``os._exit``. It also
        latches ``_force_in_progress`` and resets SIGINT to ``SIG_DFL`` so any
        further press (mashing) hard-exits at once instead of re-entering this
        handler — which is what used to spam the banner and double-render the
        table. SIGTERM does the full synchronous :meth:`teardown_all` then
        ``os._exit``s (no interactive user to keep waiting).
        """
        atexit.register(self.teardown_all)

        def _handle_terminate(signum: int, frame: Any) -> None:
            # A second terminate signal mid-teardown exits at once.
            if self._force_in_progress:
                os._exit(128 + signum)
            self._force_in_progress = True
            logger.warning("Received signal %s — tearing down active runs", signum)
            self._emit_final_summary()
            try:
                self.teardown_all()
            finally:
                os._exit(128 + signum)

        def _handle_interrupt(signum: int, frame: Any) -> None:
            self._sigint_count += 1
            if self._sigint_count >= 2:
                # Re-entrant press while the forced quit is already underway
                # (mashing): exit NOW — no second banner, table, or teardown.
                if self._force_in_progress:
                    os._exit(128 + signum)
                self._force_in_progress = True
                # Discard any further Ctrl+C for the brief remainder of this
                # handler. The force path is fast and must run to completion so
                # the banner + results table print exactly once before exit.
                # Without this, a rapid follow-up press either re-enters and
                # spams the banner (the original bug) or hard-kills before the
                # table ever renders. SIGTERM / ``kill`` still work if the
                # process genuinely wedges.
                try:
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                except (ValueError, OSError):
                    pass
                # Launch the slow part OFF the critical path FIRST — a detached
                # sweep that does the multi-second ``docker rm -f`` of every
                # in-flight container — so docker cleanup runs while we print and
                # the foreground never blocks on it.
                self._spawn_background_sweep()
                logger.warning(
                    "Second SIGINT — forcing exit without graceful unwind"
                )
                # Cheap, user-facing finish: force banner + partial results table
                # (printed once), then stamp interrupted status on the records the
                # sweep is killing so the inspector shows no phantom "running".
                self._emit_final_summary()
                self._finalize_running_trials()
                self._stop_host_services_nonblocking()
                os._exit(128 + signum)
            # FIRST Ctrl+C — graceful: let the in-flight trial(s) finish, cancel
            # the queued ones. Flag the live progress bar so it stops yanking
            # toward a misleading 100% as interrupted trials resolve, then print
            # the graceful-stop contract so the user knows what just happened and
            # how to force-quit (the logger line below is swallowed mid-run).
            try:
                from cage.contracts.logging import mark_run_stopping
                mark_run_stopping()
            except Exception:  # noqa: BLE001
                pass
            self._emit_first_interrupt_notice()
            # We deliberately do NOT force-kill containers and do NOT raise: the
            # stop request makes queued trials bail (via ``_TrialCancelled`` in
            # the conductor, checked before any container launch) while the
            # currently-running trial(s) complete, so the run drains and returns
            # on its own. The agent's ``docker exec`` runs in its own session
            # (start_new_session) so a terminal Ctrl+C doesn't kill it directly.
            # Press Ctrl+C again to kill everything now (the second-SIGINT path).
            self.scheduler.request_stop()
            logger.warning(
                "SIGINT — stop requested; in-flight trials finish, queued trials "
                "cancelled. Ctrl+C again to force-kill."
            )

        try:
            signal.signal(signal.SIGTERM, _handle_terminate)
        except (ValueError, OSError):
            # Non-main thread or platform without signal support — atexit still fires.
            pass
        try:
            signal.signal(signal.SIGINT, _handle_interrupt)
        except (ValueError, OSError):
            pass
