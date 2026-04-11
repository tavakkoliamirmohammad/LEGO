"""Strip debug symbols and deduplicate versioned .so files for wheel builds.

Usage: python dedup_libs.py <directory>

For each shared library group in <directory>:
  1. Keeps the real file (the one extensions link against via DT_NEEDED)
  2. Removes symlinks (pip expands these into full copies in wheels)
  3. Removes pip artifacts (libFoo.so.. double-dot files)
  4. Strips debug symbols (Linux: strip --strip-unneeded)

This prevents pip/setuptools from expanding symlinks into duplicate copies
that can triple the wheel size.
"""
import platform
import subprocess
import sys
from pathlib import Path


def dedup_and_strip(lib_dir: Path) -> None:
    if not lib_dir.is_dir():
        print(f"[dedup_libs] WARNING: {lib_dir} is not a directory, skipping")
        return

    # Collect all .so/.dylib files grouped by base name
    # e.g. libFoo.so, libFoo.so.23.0git, libFoo.so.. -> base "libFoo"
    groups: dict[str, list[Path]] = {}
    for f in sorted(lib_dir.iterdir()):
        name = f.name
        if ".so" in name:
            base = name[: name.index(".so")]
        elif ".dylib" in name:
            base = name[: name.index(".dylib")]
        else:
            continue
        groups.setdefault(base, []).append(f)

    for base, files in groups.items():
        if len(files) <= 1:
            # Single file, just strip it
            f = files[0]
            if not f.is_symlink():
                _strip(f)
            continue

        # Multiple files: keep the real file that extensions reference
        # (the largest non-symlink), remove all others
        real_files = [f for f in files if not f.is_symlink() and f.is_file()]
        if not real_files:
            continue

        keep = max(real_files, key=lambda f: f.stat().st_size)

        for f in files:
            if f == keep:
                continue
            f.unlink()
            print(f"[dedup_libs] Removed {f.name}")

        _strip(keep)


def _strip(path: Path) -> None:
    if platform.system() == "Linux":
        r = subprocess.run(
            ["strip", "--strip-unneeded", str(path)],
            check=False, capture_output=True,
        )
    elif platform.system() == "Darwin":
        r = subprocess.run(
            ["strip", "-x", str(path)],
            check=False, capture_output=True,
        )
    else:
        return
    if r.returncode == 0:
        print(f"[dedup_libs] Stripped {path.name}")
    else:
        print(f"[dedup_libs] WARNING: strip failed for {path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(1)
    dedup_and_strip(Path(sys.argv[1]))
