"""Trial-id shape — parse and format the ``…/pass_N`` convention in one place.

A canonical trial id is ``<subject>/<task>/pass_<n>``; a runtime trial id is
the shorter ``<task>/pass_<n>`` (or just ``<task>`` for a single pass). Several
call sites historically inlined the same ``"/pass_"`` ``rsplit`` to recover the
task id and pass index, each repeating the same edge-case handling. This module
owns that string convention so the parsing and the formatting cannot drift.

String-only by design: callers that need to honour the structured
``sample["pass_index"]`` override resolve it through
:func:`cage.contracts.sample_keys.sample_pass_index` and then delegate the
raw-id parsing here.
"""

from __future__ import annotations

_PASS_SEP = "/pass_"


def parse_trial_id(trial_id: str) -> tuple[str, int]:
    """Split ``trial_id`` into ``(task_id, pass_index)``.

    ``"task/pass_3"`` → ``("task", 3)``. With no ``/pass_`` suffix the whole id
    is the task and the pass defaults to ``1``. A non-integer suffix keeps the
    parsed task id but falls back to pass ``1``. The pass index is clamped to
    ``>= 1``.
    """

    raw = str(trial_id)
    if _PASS_SEP in raw:
        task_id, pass_text = raw.rsplit(_PASS_SEP, 1)
        try:
            return task_id, max(1, int(pass_text))
        except ValueError:
            return task_id, 1
    return raw, 1


def format_trial_id(subject_id: str, task_id: str, pass_index: int = 1) -> str:
    """Build the canonical ``<subject>/<task>/pass_<n>`` trial id."""

    return f"{subject_id}/{task_id}/pass_{pass_index}"


def runtime_trial_subpath(task_id: str, pass_index: int, passk: int) -> str:
    """Runtime trial subpath under ``trials/`` for one ``(task, pass)``.

    This mirrors the runtime id that the ``pre_run`` expansion produces (see
    ``cage.experiment.engine.hooks.expand_trials_for_passk``): a single-pass run
    omits the ``pass_<n>`` segment entirely (``<task>``), while pass@k stamps it
    (``<task>/pass_<n>``). It is the physical location where the trial runner
    writes its artifacts, so the canonical record ref derives from it to keep one
    on-disk trial tree instead of a subject-prefixed parallel tree.
    """

    if passk <= 1:
        return str(task_id)
    return f"{task_id}/pass_{pass_index}"
