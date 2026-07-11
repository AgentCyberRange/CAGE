"""`cage score --max-concurrent` fans out scorer.score() without changing output.

The concurrency cap mirrors `cage run --max-concurrent`: it parallelizes only the
expensive scorer.score() call (an LLM_judge signal is one model call per trial),
while every artifact/manifest write stays serialized. So N>1 must produce
byte-identical score files to N=1 — it just runs the judges in parallel.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from click.testing import CliRunner

from cage.cli.commands.score import score

# A scorer that blocks in score() and records the peak number of concurrent
# in-flight calls to PEAK_FILE. The peak is a deterministic parallelism proof:
# serial scoring can never exceed 1, real concurrency must exceed it.
SLOW_SCORER_SRC = '''
from __future__ import annotations

import os
import threading
import time

from cage.scoring import Scorer, ScoringContext
from cage.contracts.scoring import Score

_lock = threading.Lock()
_state = {"cur": 0, "peak": 0}


class SlowScorer(Scorer):
    name = "slow"

    def score(self, ctx: ScoringContext) -> dict[str, Score]:
        with _lock:
            _state["cur"] += 1
            _state["peak"] = max(_state["peak"], _state["cur"])
        try:
            time.sleep(float(os.environ.get("SLEEP_S", "0.25")))
        finally:
            with _lock:
                _state["cur"] -= 1
        peak_file = os.environ.get("PEAK_FILE")
        if peak_file:
            # peak is monotonic, so the last writer records the true peak.
            with open(peak_file, "w") as fh:
                fh.write(str(_state["peak"]))
        value = 1.0 if "win" in (ctx.output or "") else 0.0
        return {
            "slow": Score(
                value=value,
                answer=ctx.output,
                explanation=f"scored {ctx.trial_id}",
            )
        }
'''

N_TRIALS = 6


def _make_run(root: Path) -> Path:
    run = root / "run-x"
    for i in range(N_TRIALS):
        td = run / "trials" / f"trial-{i:02d}"
        td.mkdir(parents=True)
        (td / "task_output.json").write_text(
            json.dumps(
                {
                    "output": "win" if i % 2 == 0 else "lose",
                    "sample": {"id": f"s{i}"},
                    "trial_index": i,
                    "exit_code": 0,
                    "trial_id": f"trial-{i:02d}",
                }
            ),
            encoding="utf-8",
        )
    return run


def _score_files(run: Path) -> dict[str, str]:
    return {
        p.relative_to(run).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(run.glob("trials/*/scores/*.json"))
    }


def test_max_concurrent_parallelizes_and_matches_serial(tmp_path):
    scorer_py = tmp_path / "slow_scorer.py"
    scorer_py.write_text(SLOW_SCORER_SRC, encoding="utf-8")

    # --- serial (default, no flag) --------------------------------------- #
    serial_run = _make_run(tmp_path / "serial")
    serial_peak = tmp_path / "serial_peak.txt"
    t0 = time.monotonic()
    r_serial = CliRunner().invoke(
        score,
        [str(serial_run), "--scorer", str(scorer_py)],
        env={"SLEEP_S": "0.25", "PEAK_FILE": str(serial_peak)},
    )
    serial_dt = time.monotonic() - t0
    assert r_serial.exit_code == 0, r_serial.output
    assert serial_peak.read_text().strip() == "1"  # never overlaps

    # --- concurrent (--max-concurrent 6) --------------------------------- #
    conc_run = _make_run(tmp_path / "conc")
    conc_peak = tmp_path / "conc_peak.txt"
    t0 = time.monotonic()
    r_conc = CliRunner().invoke(
        score,
        [str(conc_run), "--scorer", str(scorer_py), "--max-concurrent", "6"],
        env={"SLEEP_S": "0.25", "PEAK_FILE": str(conc_peak)},
    )
    conc_dt = time.monotonic() - t0
    assert r_conc.exit_code == 0, r_conc.output
    assert "6 concurrent workers" in r_conc.output

    # Deterministic parallelism proof: score() calls genuinely overlapped.
    assert int(conc_peak.read_text().strip()) >= 2
    # And it is materially faster than fully serialized sleeps.
    assert conc_dt < serial_dt * 0.6, f"serial={serial_dt:.2f}s conc={conc_dt:.2f}s"

    # Byte-identical score artifacts regardless of concurrency.
    serial_files = _score_files(serial_run)
    conc_files = _score_files(conc_run)
    assert len(serial_files) == N_TRIALS
    assert serial_files == conc_files


def test_unset_flag_scores_serially(tmp_path):
    """No flag == the pre-existing serial behavior (peak concurrency of 1)."""
    scorer_py = tmp_path / "slow_scorer.py"
    scorer_py.write_text(SLOW_SCORER_SRC, encoding="utf-8")

    run = _make_run(tmp_path / "default")
    peak = tmp_path / "peak.txt"
    result = CliRunner().invoke(
        score,
        [str(run), "--scorer", str(scorer_py)],
        env={"SLEEP_S": "0.05", "PEAK_FILE": str(peak)},
    )
    assert result.exit_code == 0, result.output
    assert "concurrent workers" not in result.output
    assert peak.read_text().strip() == "1"
