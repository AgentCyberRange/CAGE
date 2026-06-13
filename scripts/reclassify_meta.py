#!/usr/bin/env python3
"""One-off utility: re-classify ``termination_reason`` on existing trial meta.

Why this exists
---------------
Older runs were written with the legacy classifier that lumped every
``exit_code != 0`` outcome into ``agent_exit_nonzero``. The new classifier
distinguishes between four groups by inspecting ``proxy.jsonl``, the
configured round budget, and post-trial container state:

  * **Infra failures** — ``model_quota_exhausted`` / ``model_rate_limited`` /
    ``model_bad_gateway`` / ``model_auth_error`` / ``model_context_overflow`` /
    ``model_timeout`` / ``model_error`` / ``oom_killed`` (HTTP status + error
    type from proxy entries; exit 137 from process metadata).
  * **Budget hit** — ``max_rounds_reached`` (successful non-compact
    proxy rounds ≥ max_rounds ⇒ proxy 429-rejected the next call).
    Checked BEFORE the infra scan so a stray transient 5xx mid-run
    can't outvote a deterministic budget hit.
  * **Ctrl+C kill** — ``user_interrupted``. The SIGINT handler force-removes
    in-flight agent containers (``docker rm -f``); on some Docker versions
    that bug makes ``docker exec`` return 0, so the legacy classifier
    misread these as ``completed``. We detect them retroactively from
    on-disk forensics — see :func:`_was_interrupted` below.
  * **Truly unclassified** — ``agent_exit_nonzero`` (nothing structural
    fit; trial detail page is the source of truth).

Resume's default retry policy only retries the infra reasons + the
interrupt reason, so without re-classification ``cage run --resume``
would do nothing on these old runs.

This script walks one run directory, re-runs ``classify_trial_termination``
on every candidate trial using the on-disk artifacts (output, proxy.jsonl,
exit code, ``sample.max_rounds``, state_post snapshot, ``.cage.runlog``
SIGINT records), and rewrites the four termination-related fields in each
``meta.json``:

    status, termination_reason, termination_detail, termination_source

Other fields are preserved verbatim. The previous values are recorded under
the same keys with a ``legacy_`` prefix, so the original classification
remains auditable.

``max_rounds`` is looked up in this order:
    1. ``meta.max_rounds``                          (written by current orchestrator)
    2. ``task_output.json``'s ``sample.max_rounds`` (benchmark default)
    3. ``0`` — disables the budget-hit branch of the classifier

Interrupt evidence is checked FIRST, even for trials whose meta currently
says ``completed`` or ``target_unavailable`` — that's the whole point: the
docker-exec-returns-0 bug is a silent misclassification, so we have to be
able to punch through those. The interrupt pass uses four structural
signals (any one fires):

  1. ``meta.snapshot_failed == True`` — orchestrator wrote this when
     ``StateSnapshot.has_failures`` because the container disappeared
     mid-trial.
  2. ``meta.error`` contains the literal ``"target setup gate cancelled"``
     — that string is hard-coded in :func:`cage.orchestrator._target_setup_gate`
     and is only emitted when ``_RUN_STOP_EVENT`` is set, i.e. SIGINT
     arrived while the trial was waiting for the target-launch
     semaphore. Old orchestrator versions mis-routed that exception to
     ``target_unavailable``; we punch through here.
  3. ``.cage.runlog`` carries a ``"SIGINT — killed N agent container(s)"``
     entry whose timestamp brackets the trial's ``ended_at_ms``.
  4. Forensic fallback for runs that pre-date (1)+(2)+(3): ``state_post`` is
     empty (snapshot failed silently in the old code), ``exit_code == 0``
     (the suspicious docker-exec-returns-0 case), the trial was *not*
     within ~5% of the configured timeout (rules out
     ``execution_timeout``), the proxy log's budgeted successful rounds
     are < max_rounds (rules out ``max_rounds_reached``), and the last
     LLM response was
     mid-task — i.e. contained function_calls but no terminal assistant
     message.

After the interrupt pass, trials whose previous reason was set by a
non-classifier code path (``target_unavailable``,
``cancelled_before_start``, ``trial_error``, ``execution_timeout``,
``tool_limit``, ``live_success``, ``user_interrupted``) are left alone —
those came from the
orchestrator directly, not from raw exit-code classification.
``max_rounds_reached`` is deliberately rechecked because older proxy
versions counted failed upstream calls against the round budget.
``completed`` is *also* in the skip set for the regular classifier path
(re-deriving a ``completed`` would only ever turn it into a model error
based on a stray transient, which is wrong), but the interrupt pass
runs *before* that skip check, so genuine SIGINT-killed trials still
get fixed.

Usage:
    python scripts/reclassify_meta.py <run_dir>            # write changes
    python scripts/reclassify_meta.py <run_dir> --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cage.core.exceptions import classify_trial_termination


# Reasons that orchestrator code paths write directly (not via the failure
# classifier). Trust them — re-classifying would lose information.
_SKIP_REASONS = {
    "completed",
    "target_unavailable",
    "cancelled_before_start",
    "trial_error",
    "execution_timeout",
    "user_interrupted",
    "tool_limit",
    "live_success",
}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _trial_output_text(trial_dir: Path) -> str:
    """Return the agent's stdout (best-effort) for the classifier's fallback path."""
    obj = _read_json(trial_dir / "task_output.json") or {}
    return str(obj.get("output") or "")


def _trial_max_rounds(meta: dict, trial_dir: Path) -> int:
    """Authoritative ``max_rounds`` for this trial.

    Lookup order:
      1. ``meta.max_rounds``                    — written by current orchestrator
      2. ``task_output.json``'s ``sample.max_rounds`` — benchmark-supplied default,
         present in every trial regardless of orchestrator version
      3. ``0`` (no budget) — disables the max-rounds branch of the classifier
    """
    raw = meta.get("max_rounds")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    task_output = _read_json(trial_dir / "task_output.json") or {}
    sample = task_output.get("sample") if isinstance(task_output, dict) else None
    if isinstance(sample, dict):
        try:
            return int(sample.get("max_rounds") or 0)
        except (TypeError, ValueError):
            pass
    return 0


# ---------------------------------------------------------------------------
# Interrupt detection (forensic — no text scanning)
# ---------------------------------------------------------------------------

# Trial's ``ended_at_ms`` may be recorded BEFORE the SIGINT handler logs its
# "killed N agent containers" line — the kill (``docker rm -f``) happens
# first, the trial worker thread unblocks and writes meta, then the signal
# handler logs. Empirically the gap is sub-second but we widen to ±10 s to
# absorb pre-kill jitter (e.g. snapshot retries) and clock drift.
_SIGINT_MATCH_WINDOW_MS = 10_000


def _runlog_sigint_times_ms(run_dir: Path) -> list[int]:
    """Return millisecond timestamps of every SIGINT-kill log entry.

    Reads ``<run_dir>/.cage.runlog`` (the orchestrator's structured log)
    line-by-line, parses each JSON record, and picks records whose
    ``message`` starts with ``"SIGINT — killed"`` — the exact format
    emitted by ``_handle_interrupt`` in :mod:`cage.orchestrator`.
    """
    log_path = run_dir / ".cage.runlog"
    if not log_path.is_file():
        return []
    out: list[int] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                if "SIGINT" not in raw:  # cheap pre-filter on the raw bytes
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg = str(rec.get("message") or "")
                if not msg.startswith("SIGINT — killed"):
                    continue
                ts_str = rec.get("timestamp")
                if not isinstance(ts_str, str):
                    continue
                ts_ms = _iso_to_ms(ts_str)
                if ts_ms is not None:
                    out.append(ts_ms)
    except OSError:
        return []
    return out


def _iso_to_ms(ts: str) -> int | None:
    """Parse an ISO-8601 timestamp (with optional timezone) to ms since epoch."""
    from datetime import datetime
    try:
        # ``fromisoformat`` accepts ``+00:00`` and ``.microsecond`` suffix.
        return int(datetime.fromisoformat(ts).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def _state_post_empty(trial_dir: Path) -> bool:
    """True if ``state_post/`` exists but contains no snapshot output at all.

    We do NOT filter dotfile entries here: legitimate state_paths such as
    codex's ``.codex`` and claude-code's ``.claude`` are themselves
    dotfile-named, and filtering them would mark every successful codex
    snapshot as "empty". The check is literal — any directory entry at
    all means the snapshot wrote something.
    """
    state_post = trial_dir / "state_post"
    if not state_post.is_dir():
        return False
    try:
        return not any(state_post.iterdir())
    except OSError:
        return False


def _last_proxy_record(proxy_jsonl: Path) -> dict | None:
    """Return the last JSON record from ``proxy.jsonl`` or ``None``."""
    if not proxy_jsonl.is_file():
        return None
    last: dict | None = None
    try:
        with proxy_jsonl.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw.startswith("{"):
                    continue
                try:
                    last = json.loads(raw)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return None
    return last


def _last_call_was_mid_task(proxy_jsonl: Path) -> bool:
    """Did the last LLM response leave the agent mid-task?

    Mid-task means the model emitted ``function_call`` items but did NOT
    emit a terminal assistant message (``message.content[*].type ==
    "output_text"``). A clean self-stop would have a final assistant
    message at the end of the conversation — the agent's "I'm done" — so
    if all we see in the last record are tool calls (and maybe
    reasoning), the trial was almost certainly cut short.
    """
    rec = _last_proxy_record(proxy_jsonl)
    if not isinstance(rec, dict):
        return False
    upstream = rec.get("upstream_response") or {}
    if not isinstance(upstream, dict):
        return False
    output = upstream.get("output") or []
    if not isinstance(output, list):
        return False
    has_function_call = False
    has_final_message = False
    for item in output:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "function_call":
            has_function_call = True
        elif t == "message":
            content = item.get("content") or []
            for c in content if isinstance(content, list) else ():
                if isinstance(c, dict) and c.get("type") == "output_text":
                    text = str(c.get("text") or "").strip()
                    if text:
                        has_final_message = True
                        break
    return has_function_call and not has_final_message


def _proxy_budgeted_round_count(proxy_jsonl: Path) -> int:
    """Count successful non-compact agent rounds in ``proxy.jsonl``."""
    if not proxy_jsonl.is_file():
        return 0
    n = 0
    try:
        with proxy_jsonl.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line.startswith("{"):
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("status") or "").lower() != "success":
                    continue
                openai_request = entry.get("openai_request")
                if (
                    isinstance(openai_request, dict)
                    and openai_request.get("_proxy_compact_rewritten")
                ):
                    continue
                n += 1
    except OSError:
        return 0
    return n


def _project_timeout_seconds(run_dir: Path) -> int:
    """Read ``runtime.timeout`` (seconds) from ``<run_dir>/project.yml``.

    Returns 0 when missing or the yml is unreadable — caller treats 0 as
    "no timeout knowledge", which disables the "rule out timeout" guard
    on the forensic fallback.
    """
    yml_path = run_dir / "project.yml"
    if not yml_path.is_file():
        return 0
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return 0
    try:
        data = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):  # type: ignore[attr-defined]
        return 0
    runtime = data.get("runtime") if isinstance(data, dict) else None
    if not isinstance(runtime, dict):
        return 0
    try:
        return int(runtime.get("timeout") or 0)
    except (TypeError, ValueError):
        return 0


def _was_interrupted(
    *,
    trial_dir: Path,
    meta: dict,
    run_dir: Path,
    sigint_times_ms: list[int],
    project_timeout_s: int,
) -> tuple[bool, str]:
    """Forensic test for "this trial was killed by Ctrl+C".

    Returns ``(matched, evidence)`` so the caller can stash a human-readable
    note in ``termination_detail``. ``matched=False`` returns
    ``evidence=""``.

    Three structural signals, any one is enough:

      1. ``meta.snapshot_failed`` — orchestrator marker (new code path)
      2. SIGINT log line in ``.cage.runlog`` within ±10 s of ``ended_at_ms``
      3. forensic fallback for old runs without (1)+(2): empty state_post
         + clean exit_code + not-near-timeout + not-max-rounds-hit + the
         last LLM call was mid-task

    Per user constraint, the fallback explicitly excludes timeout and
    max-rounds hits so we don't mis-label legitimate budget-bound stops.
    """
    if meta.get("snapshot_failed") is True:
        return True, "meta.snapshot_failed=True (orchestrator marker)"

    # Gate-cancelled exceptions are SIGINT signatures: the literal string
    # ``"target setup gate cancelled"`` only appears in cage's own
    # ``_target_setup_gate`` raise path, which fires when the run stop
    # event is set. Old orchestrator versions mis-classified these as
    # ``target_unavailable``; punch through here.
    for key in ("error", "termination_detail"):
        value = meta.get(key)
        if isinstance(value, str) and "target setup gate cancelled" in value:
            return True, f"meta.{key} carries 'target setup gate cancelled' (SIGINT during gate wait)"

    timing = meta.get("timing") or {}
    ended_ms = timing.get("ended_at_ms")
    if isinstance(ended_ms, int) and sigint_times_ms:
        for sig_ts in sigint_times_ms:
            if -_SIGINT_MATCH_WINDOW_MS <= (sig_ts - ended_ms) <= _SIGINT_MATCH_WINDOW_MS:
                return True, (
                    f".cage.runlog SIGINT entry at {sig_ts}ms within "
                    f"±{_SIGINT_MATCH_WINDOW_MS // 1000}s of ended_at_ms={ended_ms}ms"
                )

    # --- Forensic fallback ---
    # All conditions must hold; any failure means we DON'T claim interrupt.
    if not _state_post_empty(trial_dir):
        return False, ""

    exit_code = meta.get("exit_code")
    try:
        exit_code = int(exit_code) if exit_code is not None else None
    except (TypeError, ValueError):
        exit_code = None
    if exit_code != 0:
        return False, ""

    # Not timed out (user constraint): trial's wall-clock should be safely
    # under the configured timeout. We allow some slack (95%) for cleanup
    # overhead in legitimately-timed-out trials.
    duration_ms = timing.get("duration_ms")
    if (
        project_timeout_s > 0
        and isinstance(duration_ms, int)
        and duration_ms >= int(project_timeout_s * 1000 * 0.95)
    ):
        return False, ""

    # Not max-rounds-reached (user constraint): budgeted successful proxy
    # rounds must be strictly below budget (or budget unset).
    proxy_jsonl = trial_dir / "proxy" / "proxy.jsonl"
    max_rounds = _trial_max_rounds(meta, trial_dir)
    proxy_n = _proxy_budgeted_round_count(proxy_jsonl)
    if max_rounds > 0 and proxy_n >= max_rounds:
        return False, ""

    # Mid-task: the last LLM call had tool calls but no terminal message.
    # This is the structural distinguisher between "agent cleanly self-
    # stopped" (would have a final message) and "agent was cut short".
    if not _last_call_was_mid_task(proxy_jsonl):
        return False, ""

    return True, (
        f"forensic fallback: state_post empty, exit_code=0, "
        f"proxy_rounds={proxy_n}/{max_rounds or '∞'}, "
        f"last LLM call mid-task"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Path to a Cage run directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not (run_dir / "trials").is_dir():
        print(f"error: {run_dir} doesn't look like a run dir (no trials/)", file=sys.stderr)
        return 2

    changes: list[tuple[str, str, str, str]] = []  # (trial_id, old, new, evidence)
    skipped: Counter[str] = Counter()
    rewritten: list[Path] = []

    # Pre-compute run-level forensic context once (cheap; reused per trial).
    sigint_times_ms = _runlog_sigint_times_ms(run_dir)
    project_timeout_s = _project_timeout_seconds(run_dir)

    for meta_path in sorted(run_dir.glob("trials/**/meta.json")):
        trial_dir = meta_path.parent
        trial_id = str(trial_dir.relative_to(run_dir / "trials"))
        meta = _read_json(meta_path)
        if meta is None:
            skipped["unreadable"] += 1
            continue

        old_reason = str(meta.get("termination_reason") or "").strip().lower()

        # PASS 1 — interrupt detection. Runs BEFORE the skip check so we can
        # punch through ``completed``: the docker-exec-returns-0 bug under
        # SIGINT means those trials carry a spurious ``completed`` label.
        # An already-``user_interrupted`` meta is a no-op (we'd reclassify
        # to the same value); skip it explicitly to keep the report clean.
        if old_reason != "user_interrupted":
            interrupted, evidence = _was_interrupted(
                trial_dir=trial_dir,
                meta=meta,
                run_dir=run_dir,
                sigint_times_ms=sigint_times_ms,
                project_timeout_s=project_timeout_s,
            )
            if interrupted:
                info = classify_trial_termination(
                    exit_code=int(meta.get("exit_code") or 0),
                    timed_out=False,
                    terminated_by_limit=False,
                    interrupted=True,
                )
                new_meta = info.to_metadata()
                new_reason = new_meta["termination_reason"]
                changes.append(
                    (trial_id, old_reason or "(blank)", new_reason, evidence),
                )
                if not args.dry_run:
                    for field in (
                        "status", "termination_reason",
                        "termination_detail", "termination_source",
                    ):
                        if field in meta:
                            meta[f"legacy_{field}"] = meta[field]
                    meta.update(new_meta)
                    meta["interrupt_evidence"] = evidence
                    meta_path.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    rewritten.append(meta_path)
                continue

        # PASS 2 — regular structural reclassification (HTTP / proxy / OOM /
        # max_rounds / agent_exit_nonzero). Trust orchestrator-written
        # reasons that don't come from the failure classifier. Recheck
        # max_rounds_reached because older proxy code counted failed upstream
        # attempts against the round budget.
        if old_reason in _SKIP_REASONS:
            skipped[f"skip:{old_reason or 'blank'}"] += 1
            continue

        # Only failed trials are candidates. The classifier won't know about
        # orchestrator-side facts (timed_out / terminated_by_limit / error) so
        # we only feed it the raw signals the agent left behind.
        exit_code = meta.get("exit_code")
        if not isinstance(exit_code, int):
            try:
                exit_code = int(exit_code)
            except (TypeError, ValueError):
                exit_code = -1

        proxy_jsonl = trial_dir / "proxy" / "proxy.jsonl"
        info = classify_trial_termination(
            exit_code=exit_code,
            timed_out=False,
            terminated_by_limit=False,
            output=_trial_output_text(trial_dir),
            proxy_jsonl_path=proxy_jsonl if proxy_jsonl.exists() else None,
            max_rounds=_trial_max_rounds(meta, trial_dir),
        )
        new_meta = info.to_metadata()

        new_reason = new_meta["termination_reason"]
        if new_reason == old_reason:
            skipped["unchanged"] += 1
            continue

        changes.append((trial_id, old_reason or "(blank)", new_reason, ""))
        if args.dry_run:
            continue

        # Preserve previous fields under legacy_* keys for auditability.
        for field in ("status", "termination_reason", "termination_detail", "termination_source"):
            if field in meta:
                meta[f"legacy_{field}"] = meta[field]
        meta.update(new_meta)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rewritten.append(meta_path)

    # Report
    print(f"=== Reclassify summary for {run_dir} ===")
    print(f"Total trial metas scanned: "
          f"{len(changes) + sum(skipped.values())}")
    for key, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  skipped/{key}: {n}")

    if changes:
        # Group by new reason
        by_new = Counter(c[2] for c in changes)
        print()
        print(f"Reclassified: {len(changes)}")
        for r, n in by_new.most_common():
            print(f"  → {r}: {n}")
        print()
        print("Per-trial changes (old → new) — evidence shown for interrupts:")
        for tid, old, new, evidence in changes:
            arrow = "→"
            print(f"  {tid:46s} {old:24s} {arrow} {new}")
            if evidence:
                print(f"      ↳ {evidence}")

    if args.dry_run:
        print()
        print("(dry-run — no files written. Re-run without --dry-run to apply.)")
    elif rewritten:
        print()
        print(f"Wrote {len(rewritten)} meta.json file(s).")
        print("Now run:  cage run <project.yml> --run-id <run_id> --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
