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


def ampere() -> MachineModel:
    return MachineModel(name="ampere", banks=32, bank_bytes=4, warp=32,
                        sector_bytes=32, line_bytes=128, max_vec_bytes=16)
