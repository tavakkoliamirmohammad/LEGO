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
    unit: int = 1   # vector granularity in elements that layouts must keep contiguous
    kind: Optional[str] = None   # named problem (lets a tier regenerate patterns for a padded tile)


@dataclass(frozen=True)
class Solution:
    layout: object
    tier: str
    template: str
    footprint: int
    solve_time: float
    problem: Problem = None   # the problem actually solved (a tier may pad the tile)


def templates_for(prob: Problem, tier: str) -> List[object]:
    u = prob.unit
    if tier == "f2":
        return [XorTemplate(prob.rows, _next_pow2(prob.cols), unit=u)]
    if tier == "affine_xor":
        ts = [PaddedTemplate(prob.rows, prob.cols, max_pad=8 * u)]
        if _is_pow2(prob.cols):
            ts.append(XorTemplate(prob.rows, prob.cols, unit=u))
        return ts
    if tier == "full":
        return [XorTemplate(prob.rows, prob.cols, unit=u),
                RotationTemplate(prob.rows, prob.cols, unit=u),
                PaddedTemplate(prob.rows, prob.cols, max_pad=8 * u)]
    raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")


def synthesize(prob: Problem, tier: str) -> Optional[Solution]:
    """Smallest-footprint conflict-free layout among the tier's templates, or None.

    The f2 tier models a system whose block shapes are powers of two: the tile is
    padded to the next power of two *logically* (extra columns masked), so its
    access patterns are regenerated over the padded width when the problem is named.
    """
    if tier == "f2" and prob.kind is not None and not _is_pow2(prob.cols):
        from .problems import problem_for
        prob = problem_for(prob.kind, prob.rows, _next_pow2(prob.cols), prob.elem_bytes, prob.machine)
    best = None
    for t in templates_for(prob, tier):
        t0 = time.time()
        minimize = "pad" if isinstance(t, PaddedTemplate) else None
        layout = solve_conflict_free(t, prob.patterns, prob.elem_bytes, prob.machine, minimize=minimize)
        dt = time.time() - t0
        if layout is None:
            continue
        sol = Solution(layout, tier, type(t).__name__, layout.footprint, dt, problem=prob)
        if best is None or sol.footprint < best.footprint:
            best = sol
    return best
