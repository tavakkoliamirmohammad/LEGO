"""Tiered search over templates.

Tiers model the expressiveness of existing systems so the same solver can be
used for a space ablation:
  f2         power-of-two shapes, XOR swizzles only (Triton linear layouts)
  affine_xor strides/padding plus XOR swizzle atoms on power-of-two rows (CuTe, Hexcute)
  full       everything LEGO can express: padding, in-bounds XOR on any width, rotations
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import time

from .machine import MachineModel
from .solve import solve_conflict_free
from .templates import XorTemplate, PaddedTemplate, RotationTemplate

TIERS = ("f2", "affine_xor", "full")


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _is_pow2(n: int) -> bool:
    return n & (n - 1) == 0


@dataclass(frozen=True)
class Problem:
    rows: int
    cols: int
    elem_bytes: int
    patterns: Sequence
    machine: MachineModel


@dataclass(frozen=True)
class Solution:
    layout: object
    tier: str
    template: str
    footprint: int
    solve_time: float


def templates_for(prob: Problem, tier: str) -> List[object]:
    if tier == "f2":
        return [XorTemplate(prob.rows, _next_pow2(prob.cols))]
    if tier == "affine_xor":
        ts = [PaddedTemplate(prob.rows, prob.cols, max_pad=8)]
        if _is_pow2(prob.cols):
            ts.append(XorTemplate(prob.rows, prob.cols))
        return ts
    if tier == "full":
        return [XorTemplate(prob.rows, prob.cols),
                RotationTemplate(prob.rows, prob.cols),
                PaddedTemplate(prob.rows, prob.cols, max_pad=8)]
    raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")


def synthesize(prob: Problem, tier: str) -> Optional[Solution]:
    """Smallest-footprint conflict-free layout among the tier's templates, or None."""
    best = None
    for t in templates_for(prob, tier):
        t0 = time.time()
        minimize = "pad" if isinstance(t, PaddedTemplate) else None
        layout = solve_conflict_free(t, prob.patterns, prob.elem_bytes, prob.machine, minimize=minimize)
        dt = time.time() - t0
        if layout is None:
            continue
        sol = Solution(layout, tier, type(t).__name__, layout.footprint, dt)
        if best is None or sol.footprint < best.footprint:
            best = sol
    return best
