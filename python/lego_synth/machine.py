"""Machine models: the hardware parameters that layout properties depend on."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MachineModel:
    name: str
    banks: int          # number of shared-memory banks
    bank_bytes: int     # bytes per bank word
    warp: int           # lanes per warp / wavefront
    sector_bytes: int   # global-memory sector granularity
    line_bytes: int     # cache line / max transaction size
    max_vec_bytes: int  # widest per-lane vector access


def nvidia(name: str = "nvidia") -> MachineModel:
    """Shared-memory and warp geometry shared by every NVIDIA architecture from Volta on:
    32 banks x 4 B, 32-lane warps, 32 B sectors, 128 B lines, 16 B vector accesses."""
    return MachineModel(name=name, banks=32, bank_bytes=4, warp=32,
                        sector_bytes=32, line_bytes=128, max_vec_bytes=16)


def ampere() -> MachineModel:
    return nvidia("ampere")


def hopper() -> MachineModel:
    return nvidia("hopper")


def for_device() -> MachineModel:
    """Machine model of the GPU in this machine (detected), for the properties the solver certifies."""
    try:
        from .kernels.target import detect_target
        t = detect_target()
        return nvidia(f"nvidia_cc{t.cc}")
    except Exception:
        return nvidia()
