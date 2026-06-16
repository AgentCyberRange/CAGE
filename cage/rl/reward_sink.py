"""Reward sink — report one trial's reward to an external RL trainer.

Mechanism only. Two facts the trainer needs from Cage, and where they come from:

  * **which trial an LLM call belongs to** — every call this model drives carries
    an ``X-Trial-Id`` header injected by the trial runner; its value is
    :func:`rl_trial_id`. (That injection lives in the trial runner, not here, so
    it rides Cage's existing proxy ``extra_headers`` channel with no proxy edit.)
  * **each trial's reward** — :func:`report_trial_rewards` reads the number from
    the benchmark (``Benchmark.reward``) and POSTs it here.

The join key is computed identically at both sites (header and report), so the
trainer can match them byte-for-byte. The POST is best-effort: it runs on a
daemon thread with bounded retries and never raises, so a flaky or absent sink
can never block or fail a trial. Active only when the model declares
``rl_reward_sink`` — otherwise :func:`report_trial_rewards` is a no-op and an
ordinary eval is completely unaffected.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any

logger = logging.getLogger("cage.rl.reward_sink")

# Bounded total drain after firing a batch of reward POSTs. The trials are
# already finished when we report, so waiting here blocks nothing live; we just
# give the daemon threads a chance to deliver before the run tears down (daemon
# threads are killed at interpreter exit). Generous enough for retries, capped so
# a dead sink never stalls the run.
_DRAIN_DEADLINE_SECONDS = 20.0


def rl_trial_id(run_id: str, trial_id: str) -> str:
    """The trial's globally-unique join key, shared verbatim by the LLM-call
    ``X-Trial-Id`` header and the reward report.

    Deterministic on purpose: both call sites compute it from the same two ids,
    so they always agree without threading a generated value around. ``run_id``
    is unique per run (auto ids already embed a uuid); ``trial_id`` is unique per
    trial within a run (``<task>`` or ``<task>/pass_<n>``); the pair is globally
    unique.
    """

    return f"{run_id}/{trial_id}"


def post_reward(
    sink_url: str,
    payload: dict[str, Any],
    *,
    retries: int = 3,
    timeout: float = 5.0,
) -> threading.Thread:
    """Fire-and-forget POST of one reward payload to ``sink_url``.

    Returns the daemon thread doing the work (callers may join it with a timeout
    to drain before exit; production can ignore it). Never raises — transport
    errors are retried with linear backoff and ultimately logged, never surfaced
    to the trial.
    """

    def _send() -> None:
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(
                    sink_url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    resp.read()
                return
            except Exception as exc:  # noqa: BLE001 — best-effort, must not bubble
                logger.warning(
                    "rl reward POST failed (attempt %d/%d) for %s: %s",
                    attempt,
                    retries,
                    payload.get("trial_id"),
                    exc,
                )
                time.sleep(min(2.0, 0.5 * attempt))
        logger.error(
            "rl reward POST gave up after %d attempts for %s",
            retries,
            payload.get("trial_id"),
        )

    thread = threading.Thread(target=_send, name="rl-reward-post", daemon=True)
    thread.start()
    return thread


def _reward_payload(run_id: str, benchmark: Any, result: Any) -> dict[str, Any]:
    """Build the reward JSON for one finished trial.

    The scalar comes from ``benchmark.reward(result)`` (Layer 2 owns the number);
    we only clamp it to ``[0, 1]`` and assemble the envelope. A benchmark whose
    ``reward`` raises is treated as reward 0 — a reporting fault must not lose the
    trial.
    """

    try:
        reward = float(benchmark.reward(result))
    except Exception:  # noqa: BLE001
        logger.exception("benchmark.reward failed for %s", getattr(result, "trial_id", "?"))
        reward = 0.0
    reward = max(0.0, min(1.0, reward))
    scores = {
        name: float(score.value)
        for name, score in getattr(result, "scores", {}).items()
    }
    return {
        "trial_id": rl_trial_id(run_id, str(getattr(result, "trial_id", ""))),
        "reward": reward,
        "success": reward >= 1.0,
        "sample_id": str(getattr(result, "sample_id", "")),
        "metrics": {
            "exit_code": getattr(result, "exit_code", None),
            "error": getattr(result, "error", None),
            "scores": scores,
        },
    }


def report_trial_rewards(run: Any, agent: Any, results: list[Any]) -> None:
    """Report every finished trial's reward for one agent — exactly once each.

    Called once per agent right after the agent's trials are scored, the single
    point where every trial (success *and* failure/timeout) has its final score
    and the run/model are in scope. No-op unless ``agent.model.rl_reward_sink`` is
    set, so a non-RL run is untouched.

    Fires one best-effort POST per result, then drains the batch with a bounded
    deadline. The trials are already done, so this blocks no live work.
    """

    sink_url = str(getattr(getattr(agent, "model", None), "rl_reward_sink", "") or "")
    if not sink_url:
        return

    run_id = str(getattr(run, "run_id", "") or "")
    benchmark = getattr(run, "benchmark", None)
    if benchmark is None:
        return

    threads: list[threading.Thread] = []
    for result in results:
        payload = _reward_payload(run_id, benchmark, result)
        logger.info(
            "rl_reward_report trial=%s reward=%s sink=%s",
            payload["trial_id"],
            payload["reward"],
            sink_url,
        )
        threads.append(post_reward(sink_url, payload))

    deadline = time.monotonic() + _DRAIN_DEADLINE_SECONDS
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)
