"""Tests for cage.contracts.logging — structured logging system."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from cage.contracts.logging import (
    LoggingConfig,
    ProgressReporter,
    add_file_handlers,
    bind_run_context,
    bind_trial_context,
    clear_all_context,
    clear_trial_context,
    remove_file_handlers,
    setup_logging,
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging state between tests."""
    # Clear structlog config cache
    import cage.contracts.logging as log_mod
    log_mod._setup_done = False
    # Clear all context vars
    clear_all_context()
    # Remove all handlers from root logger
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    yield
    clear_all_context()


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for log files."""
    d = tmp_path / "logs"
    d.mkdir()
    return d


# ------------------------------------------------------------------ #
# TestLoggingSetup
# ------------------------------------------------------------------ #

class TestLoggingSetup:
    def test_setup_creates_console_handler(self):
        setup_logging(LoggingConfig(console_level="INFO"))
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_setup_respects_console_level(self):
        setup_logging(LoggingConfig(console_level="WARNING"))
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert any(h.level == logging.WARNING for h in stream_handlers)

    def test_setup_idempotent(self):
        setup_logging(LoggingConfig(console_level="INFO"))
        setup_logging(LoggingConfig(console_level="DEBUG"))
        # Should not add duplicate handlers
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) == 1

    def test_add_file_handlers_creates_file(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1
        # The log file may not exist yet until first write — that's OK
        remove_file_handlers()

    def test_add_file_handlers_writes_to_file(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        logger = logging.getLogger("test_write")
        logger.info("hello file")

        remove_file_handlers()

        content = log_path.read_text().strip()
        assert content  # file has content
        lines = content.split("\n")
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["message"] == "hello file"

    def test_add_debug_file_handler(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        debug_path = log_dir / "cage-debug.log"
        add_file_handlers(log_path, debug_path)

        logger = logging.getLogger("test_debug")
        logger.debug("debug message")

        remove_file_handlers()

        # Debug log should contain the DEBUG message
        debug_content = debug_path.read_text().strip()
        assert debug_content
        debug_obj = json.loads(debug_content.split("\n")[0])
        assert debug_obj["message"] == "debug message"
        assert debug_obj["level"] == "debug"

    def test_remove_file_handlers(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        root = logging.getLogger()
        file_handlers_before = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "_cage_log", False)
        ]
        assert len(file_handlers_before) >= 1

        remove_file_handlers()

        file_handlers_after = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "_cage_log", False)
        ]
        assert len(file_handlers_after) == 0


# ------------------------------------------------------------------ #
# TestContextBinding
# ------------------------------------------------------------------ #

class TestContextBinding:
    def test_bind_run_context(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        bind_run_context(run_id="run-123", agent_label="claude:glm:stateless")
        logging.getLogger("test").info("with context")

        remove_file_handlers()

        obj = _read_first_jsonl(log_path)
        assert obj["run_id"] == "run-123"
        assert obj["agent_label"] == "claude:glm:stateless"

    def test_bind_trial_context(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        bind_run_context(run_id="run-123")
        bind_trial_context(trial_id="trial_001", trial_index=0, sample_id="abc")
        logging.getLogger("test").info("during trial")

        remove_file_handlers()

        obj = _read_first_jsonl(log_path)
        assert obj["trial_id"] == "trial_001"
        assert obj["trial_index"] == 0
        assert obj["sample_id"] == "abc"
        assert obj["run_id"] == "run-123"

    def test_clear_trial_context(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        bind_run_context(run_id="run-123")
        bind_trial_context(trial_id="trial_001", trial_index=0, sample_id="abc")
        logging.getLogger("test").info("during trial")

        clear_trial_context()
        logging.getLogger("test").info("after trial clear")

        remove_file_handlers()

        lines = log_path.read_text().strip().split("\n")
        obj1 = json.loads(lines[0])
        obj2 = json.loads(lines[1])

        assert "trial_id" in obj1
        assert "trial_id" not in obj2
        assert obj2["run_id"] == "run-123"  # run context preserved

    def test_clear_all_context(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        bind_run_context(run_id="run-123")
        logging.getLogger("test").info("with context")
        clear_all_context()
        logging.getLogger("test").info("without context")

        remove_file_handlers()

        lines = log_path.read_text().strip().split("\n")
        obj1 = json.loads(lines[0])
        obj2 = json.loads(lines[1])

        assert "run_id" in obj1
        assert "run_id" not in obj2


# ------------------------------------------------------------------ #
# TestJSONLOutput
# ------------------------------------------------------------------ #

class TestJSONLOutput:
    def test_each_line_is_valid_json(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        logger = logging.getLogger("test_jsonl")
        logger.info("line one")
        logger.warning("line two")
        logger.error("line three")

        remove_file_handlers()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "timestamp" in obj
            assert "level" in obj
            assert "message" in obj

    def test_jsonl_contains_timestamp(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        logging.getLogger("test").info("ts test")

        remove_file_handlers()

        obj = _read_first_jsonl(log_path)
        assert "timestamp" in obj
        # ISO-8601 format
        assert "T" in obj["timestamp"]

    def test_jsonl_contains_level(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        logging.getLogger("test").warning("level test")

        remove_file_handlers()

        obj = _read_first_jsonl(log_path)
        assert obj["level"] == "warning"

    def test_jsonl_contains_context(self, log_dir: Path):
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        bind_run_context(run_id="r1")
        bind_trial_context(trial_id="t1")
        logging.getLogger("test").info("context test")

        remove_file_handlers()

        obj = _read_first_jsonl(log_path)
        assert obj["run_id"] == "r1"
        assert obj["trial_id"] == "t1"


# ------------------------------------------------------------------ #
# TestBackwardCompatibility
# ------------------------------------------------------------------ #

class TestBackwardCompatibility:
    def test_stdlib_logger_works(self, log_dir: Path):
        """Existing logging.getLogger(__name__) calls produce output."""
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        # Simulate existing module pattern
        logger = logging.getLogger("cage.experiment.engine.conductor")
        logger.info("Starting experiment: test (run_id=run-1)")

        remove_file_handlers()

        obj = _read_first_jsonl(log_path)
        assert "Starting experiment" in obj["message"]

    def test_stdlib_logger_gains_context(self, log_dir: Path):
        """Stdlib logger calls pick up structlog context bindings."""
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        bind_run_context(run_id="run-42")
        stdlib_logger = logging.getLogger("cage.sandbox.containers")
        stdlib_logger.info("Container started: cage-test (image=test)")

        remove_file_handlers()

        obj = _read_first_jsonl(log_path)
        assert obj["run_id"] == "run-42"
        assert "Container started" in obj["message"]

    def test_stdlib_exception_logging(self, log_dir: Path):
        """logger.exception() works through the structlog bridge."""
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        try:
            raise ValueError("test error")
        except ValueError:
            logging.getLogger("cage.test").exception("Something failed")

        remove_file_handlers()

        obj = _read_first_jsonl(log_path)
        assert obj["level"] == "error"
        assert "Something failed" in obj["message"]


# ------------------------------------------------------------------ #
# TestProgressReporter
# ------------------------------------------------------------------ #

class TestProgressReporter:
    def test_trial_completed(self, capsys):
        reporter = ProgressReporter(total_trials=5)
        reporter.trial_started("trial_001", "sample_abc123456789")
        reporter.trial_completed("trial_001", 1500, 0)

        captured = capsys.readouterr()
        assert "[1/5]" in captured.err
        assert "trial_001" in captured.err
        assert "OK" in captured.err
        assert "1.5s" in captured.err

    def test_trial_failed(self, capsys):
        reporter = ProgressReporter(total_trials=3)
        reporter.trial_started("trial_001", "sample_x")
        reporter.trial_failed("trial_001", "Container timeout")

        captured = capsys.readouterr()
        assert "FAILED" in captured.err
        assert "Container timeout" in captured.err

    def test_counts_track(self):
        reporter = ProgressReporter(total_trials=4)
        reporter.trial_started("t1", "s1")
        reporter.trial_completed("t1", 1000, 0)
        reporter.trial_started("t2", "s2")
        reporter.trial_failed("t2", "error")
        reporter.trial_started("t3", "s3")
        reporter.trial_completed("t3", 2000, 0)

        assert reporter.completed == 2
        assert reporter.failed == 1
        assert reporter.summary() == "3/4 done (1 failed)"

    def test_nonzero_exit_code_shown(self, capsys):
        reporter = ProgressReporter(total_trials=1)
        reporter.trial_started("t1", "s1")
        reporter.trial_completed("t1", 3000, 1)

        captured = capsys.readouterr()
        assert "exit=1" in captured.err


# ------------------------------------------------------------------ #
# TestThreadSafety
# ------------------------------------------------------------------ #

class TestThreadSafety:
    def test_concurrent_log_writes(self, log_dir: Path):
        """Multiple threads writing to the same JSONL file produce valid output."""
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        n_threads = 5
        n_messages = 20
        barrier = threading.Barrier(n_threads)

        def writer(thread_id: int) -> None:
            barrier.wait()
            for i in range(n_messages):
                logging.getLogger(f"thread-{thread_id}").info(
                    "msg-%d from thread-%d", i, thread_id
                )

        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        remove_file_handlers()

        lines = log_path.read_text().strip().split("\n")
        # Each line must be valid JSON
        parsed = []
        for line in lines:
            obj = json.loads(line)
            parsed.append(obj)

        assert len(parsed) == n_threads * n_messages

    def test_context_isolation_in_threads(self, log_dir: Path):
        """Context bound in one thread does not leak to another."""
        setup_logging(LoggingConfig(console_level="INFO"))
        log_path = log_dir / "cage.log"
        add_file_handlers(log_path)

        # Write from thread A with its own context
        def thread_a() -> None:
            bind_run_context(run_id="run-A")
            logging.getLogger("thread-a").info("from A")

        def thread_b() -> None:
            bind_run_context(run_id="run-B")
            logging.getLogger("thread-b").info("from B")

        t_a = threading.Thread(target=thread_a)
        t_a.start()
        t_a.join()

        t_b = threading.Thread(target=thread_b)
        t_b.start()
        t_b.join()

        remove_file_handlers()

        # Read all lines and find messages from each thread
        lines = log_path.read_text().strip().split("\n")
        msg_a = None
        msg_b = None
        for line in lines:
            obj = json.loads(line)
            if "from A" in obj.get("message", ""):
                msg_a = obj
            elif "from B" in obj.get("message", ""):
                msg_b = obj

        assert msg_a is not None
        assert msg_b is not None
        # Each message should have its own run_id (context is per-thread)
        assert msg_a.get("run_id") == "run-A"
        assert msg_b.get("run_id") == "run-B"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _read_first_jsonl(path: Path) -> dict[str, Any]:
    """Read the first line of a JSONL file and parse it."""
    content = path.read_text().strip()
    lines = content.split("\n")
    return json.loads(lines[0])