"""Detect the GPU the benchmarks run on: PTX target, machine model, and provenance metadata.

Nothing in the suite should assume a specific card. `detect_target()` reads the compute
capability (torch, then nvidia-smi), maps it to an `sm_XX` chip and a PTX ISA version the
bundled LLVM NVPTX backend supports, and reports SM count and shared-memory limits so
sweeps size themselves; `metadata()` records everything a result file needs to be
reproduced (device, driver, toolkit and library versions, git commit, timestamp).
"""
import datetime
import os
import platform
import subprocess
from dataclasses import dataclass, asdict

_MAX_LLVM_SM = 90   # highest SM the bundled LLVM NVPTX backend can target; newer GPUs run PTX for this


@dataclass(frozen=True)
class Target:
    name: str
    cc: str            # e.g. "8.6"
    chip: str          # e.g. "sm_86" (capped at the LLVM backend's maximum)
    features: str      # e.g. "+ptx78"
    sm_count: int
    smem_per_block_optin: int   # bytes of dynamic shared memory a block may opt into
    tag: str           # filesystem-friendly device tag, e.g. "nvidia_rtx_a6000_sm86"


def _cc_from_nvidia_smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=10)
    cc, name = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",", 1)]
    return cc, name


def detect_target() -> Target:
    sm_count, optin = 0, 48 * 1024
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        cc, name, sm_count = f"{p.major}.{p.minor}", p.name, p.multi_processor_count
        try:
            import cupy
            optin = int(cupy.cuda.runtime.deviceGetAttribute(cupy.cuda.runtime.cudaDevAttrMaxSharedMemoryPerBlockOptin, 0))
        except Exception:
            optin = 0
    except Exception:
        cc, name = _cc_from_nvidia_smi()
    major, minor = (int(x) for x in cc.split("."))
    sm = min(major * 10 + minor, _MAX_LLVM_SM)
    features = "+ptx80" if sm >= 90 else "+ptx78" if sm >= 80 else "+ptx70"
    if not optin:   # per-architecture opt-in maxima when CuPy is unavailable
        optin = {7: 96, 8: 100 if sm >= 86 else 164, 9: 228}.get(major, 48) * 1024
    tag = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_") + f"_sm{major}{minor}"
    return Target(name=name, cc=cc, chip=f"sm_{sm}", features=features, sm_count=sm_count,
                  smem_per_block_optin=optin, tag=tag)


def metadata(target: Target = None) -> dict:
    t = target or detect_target()
    info = {"device": asdict(t), "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(), "python": platform.python_version()}
    for mod in ("torch", "triton", "cupy", "numpy", "sympy", "z3"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", None) or (m.get_version_string() if mod == "z3" else "?")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda_runtime"] = torch.version.cuda
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        info["driver"] = r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        info["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    return info


def metadata_header(target: Target = None) -> str:
    """Markdown block to prepend to every result file."""
    info = metadata(target)
    dev = info.pop("device")
    lines = ["<!-- provenance -->", f"Device: {dev['name']} (cc {dev['cc']}, {dev['sm_count']} SMs, PTX target {dev['chip']} {dev['features']}, "
             f"opt-in smem {dev['smem_per_block_optin'] // 1024} KB)"]
    lines.append(" | ".join(f"{k}: {v}" for k, v in info.items()))
    return "\n".join(lines) + "\n"
