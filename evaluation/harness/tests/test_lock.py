"""Test harness lock serializes critical sections across threads."""

import threading
import time
from pathlib import Path

import pytest

from evaluation.harness import lock as lock_mod


def test_acquire_serializes_concurrent_holders(tmp_path: Path) -> None:
    """No two threads ever hold the lock simultaneously."""
    lock_path = tmp_path / ".lock"
    inside_count = [0]
    max_inside = [0]
    state_guard = threading.Lock()

    def worker() -> None:
        with lock_mod.acquire(lock_path):
            with state_guard:
                inside_count[0] += 1
                if inside_count[0] > max_inside[0]:
                    max_inside[0] = inside_count[0]
            time.sleep(0.05)
            with state_guard:
                inside_count[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_inside[0] == 1, (
        f"lock allowed {max_inside[0]} concurrent holders"
    )


def test_acquire_creates_lock_file_if_missing(tmp_path: Path) -> None:
    lock_path = tmp_path / "nested" / "dir" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_mod.acquire(lock_path):
        assert lock_path.exists()


def test_acquire_releases_on_exception(tmp_path: Path) -> None:
    lock_path = tmp_path / ".lock"
    with pytest.raises(RuntimeError):
        with lock_mod.acquire(lock_path):
            raise RuntimeError("boom")
    # Lock should be re-acquirable
    with lock_mod.acquire(lock_path):
        pass
