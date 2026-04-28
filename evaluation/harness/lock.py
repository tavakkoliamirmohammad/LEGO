"""File-based exclusive lock for serializing build+measure cycles.

Uses fcntl.flock(2). The lock is per-open-file-description, so each
acquire() opens a fresh fd and any concurrent holder (thread or process)
will block until release.

Default lock path: evaluation/.lock (one above this file).
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_LOCK_PATH = Path(__file__).resolve().parent.parent / ".lock"


@contextmanager
def acquire(lock_path: Path | None = None) -> Iterator[None]:
    """Acquire an exclusive flock on lock_path; release on context exit.

    Creates lock_path if missing. Releases the lock on normal exit and
    on exception.
    """
    path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = open(path, "a")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    finally:
        if fd is not None:
            fd.close()
