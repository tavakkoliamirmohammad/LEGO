"""Thin wrapper around the naga binary for SPIR-V → WGSL / MSL / GLSL."""

import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class NagaError(Exception):
    """Raised when naga conversion fails."""
    pass


_NAGA_BIN: Optional[str] = None


def _find_naga_binary() -> str:
    """Locate the naga binary."""
    # 1. Bundled location (inside the wheel)
    pkg_dir = Path(__file__).parent.parent
    bin_name = "naga.exe" if platform.system() == "Windows" else "naga"

    bundled = pkg_dir / "bin" / bin_name
    if bundled.exists():
        return str(bundled)

    # 2. System PATH
    from shutil import which
    system_naga = which("naga")
    if system_naga:
        return system_naga

    raise FileNotFoundError(
        "naga binary not found. Install with: cargo install naga-cli"
    )


def _naga() -> str:
    global _NAGA_BIN
    if _NAGA_BIN is None:
        _NAGA_BIN = _find_naga_binary()
    return _NAGA_BIN


def convert(
    input_path: str,
    output_path: str,
    extra_args: Optional[list] = None,
) -> str:
    """
    Convert shader using naga. Format is inferred from file extensions.

    Supported: .spv → .wgsl | .metal | .comp | .frag | .vert
    """
    cmd = [_naga(), input_path, output_path]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise NagaError(
            f"naga conversion failed: {input_path} → {output_path}\n"
            f"stderr: {result.stderr}"
        )
    return output_path


def spv_to_wgsl(spv_path: str, output_path: Optional[str] = None) -> str:
    """Convert SPIR-V to WGSL (WebGPU)."""
    if output_path is None:
        output_path = str(Path(spv_path).with_suffix(".wgsl"))
    return convert(spv_path, output_path)


def spv_to_metal(spv_path: str, output_path: Optional[str] = None) -> str:
    """Convert SPIR-V to Metal Shading Language."""
    if output_path is None:
        output_path = str(Path(spv_path).with_suffix(".metal"))
    return convert(spv_path, output_path)


def spv_to_glsl(
    spv_path: str,
    output_path: Optional[str] = None,
    entry_point: Optional[str] = None,
) -> str:
    """Convert SPIR-V compute shader to GLSL (WebGL / OpenGL ES 3.1+).

    Uses .comp extension (GLSL compute shader) since naga infers
    shader stage from extension.
    """
    if output_path is None:
        output_path = str(Path(spv_path).with_suffix(".comp"))
    if output_path.endswith(".glsl"):
        output_path = output_path[:-5] + ".comp"

    args = ["--shader-stage", "compute"]
    if entry_point:
        args.extend(["--entry-point", entry_point])
    else:
        # Auto-detect: get the entry point name from WGSL conversion
        try:
            wgsl_tmp = output_path + ".tmp.wgsl"
            convert(spv_path, wgsl_tmp)
            with open(wgsl_tmp) as f:
                for line in f:
                    if line.strip().startswith("fn ") and "@compute" not in line:
                        continue
                    if "@compute" in line:
                        # Next fn line has the entry point name
                        for line2 in f:
                            if line2.strip().startswith("fn "):
                                ep = line2.strip().split("(")[0].replace("fn ", "")
                                args.extend(["--entry-point", ep])
                                break
                        break
            os.unlink(wgsl_tmp)
        except Exception:
            pass  # proceed without --entry-point, naga may still find it

    return convert(spv_path, output_path, extra_args=args)


def spv_bytes_to_file(spv_words: list, output_path: str) -> str:
    """Write a list of uint32 SPIR-V words to a binary .spv file."""
    import struct
    with open(output_path, "wb") as f:
        for w in spv_words:
            f.write(struct.pack("<I", w & 0xFFFFFFFF))
    return output_path


def validate(shader_path: str) -> bool:
    """Validate a shader file (any naga-supported format)."""
    result = subprocess.run(
        [_naga(), shader_path], capture_output=True, text=True
    )
    return result.returncode == 0
