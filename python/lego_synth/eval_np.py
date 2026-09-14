"""Exhaustive NumPy evaluator for hardware properties of concrete layouts."""
import numpy as np

from .machine import MachineModel


def bank_conflicts(byte_addrs, elem_bytes: int, machine: MachineModel) -> int:
    """Extra shared-memory wavefronts for one warp instruction.

    `byte_addrs[t]` is the byte address accessed by lane `t`, each lane reading
    `elem_bytes` contiguous bytes. Lanes are processed in groups that fit one
    full-width transaction (banks * bank_bytes); inside a group, lanes hitting
    the same bank word are served together (broadcast) and distinct words in
    the same bank serialize. Returns total wavefronts minus the ideal count.
    """
    addrs = np.asarray(byte_addrs, dtype=np.int64)
    width = machine.banks * machine.bank_bytes
    group = min(machine.warp, max(1, width // elem_bytes))
    words_per_lane = max(1, elem_bytes // machine.bank_bytes)
    extra = 0
    for start in range(0, len(addrs), group):
        lane_addrs = addrs[start:start + group]
        words = (lane_addrs[:, None] // machine.bank_bytes) + np.arange(words_per_lane)[None, :]
        words = np.unique(words.ravel())
        banks = words % machine.banks
        wavefronts = int(np.bincount(banks, minlength=machine.banks).max()) if len(words) else 0
        extra += max(wavefronts - 1, 0)
    return extra


def conflicts(layout, pattern, elem_bytes: int, machine: MachineModel):
    """Total extra wavefronts over every instruction of `pattern` on `layout`.

    `elem_bytes` scales element indices to byte addresses; the pattern decides how
    many bytes each lane touches. A lane access that is not naturally aligned to
    its width is illegal for vector and ldmatrix loads, reported as `math.inf`.
    """
    import math
    width = pattern.access_width(elem_bytes)
    total = 0
    for k in range(pattern.n_instr):
        i, j = pattern.coords(k)
        addrs = layout.flat(i, j) * elem_bytes
        if width > 1 and np.any(addrs % width):
            return math.inf
        total += bank_conflicts(addrs, width, machine)
    return total
