import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class TargetLockRegistry:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}

    def get_lock(self, runtime_key: str) -> threading.RLock:
        with self._state_lock:
            lock = self._locks.get(runtime_key)
            if lock is None:
                lock = threading.RLock()
                self._locks[runtime_key] = lock
            return lock


class TargetRecoveryCoordinator:
    def __init__(self, recent_recovery_window_s: float = 5.0) -> None:
        self.recent_recovery_window_s = recent_recovery_window_s
        self._state_lock = threading.Lock()
        self._conditions: dict[str, threading.Condition] = {}
        self._inflight: set[str] = set()
        self._recent_recoveries: dict[str, float] = {}

    def run_serialized_recovery(
        self,
        runtime_key: str,
        is_healthy: Callable[[], bool],
        recover_action: Callable[[], T],
    ) -> T | str:
        condition = self._get_condition(runtime_key)
        waited_for_inflight = False
        with condition:
            while runtime_key in self._inflight:
                waited_for_inflight = True
                condition.wait()

            last_recovery = self._recent_recoveries.get(runtime_key)
            if waited_for_inflight and last_recovery is not None:
                age_s = time.monotonic() - last_recovery
                if age_s <= self.recent_recovery_window_s and is_healthy():
                    return "reused_recent"

            self._inflight.add(runtime_key)

        recovered = False
        try:
            result = recover_action()
            recovered = True
            return result
        finally:
            with condition:
                self._inflight.discard(runtime_key)
                if recovered:
                    self._recent_recoveries[runtime_key] = time.monotonic()
                condition.notify_all()

    def _get_condition(self, runtime_key: str) -> threading.Condition:
        with self._state_lock:
            condition = self._conditions.get(runtime_key)
            if condition is None:
                condition = threading.Condition()
                self._conditions[runtime_key] = condition
            return condition


ChallengeLockRegistry = TargetLockRegistry
ChallengeRecoveryCoordinator = TargetRecoveryCoordinator
