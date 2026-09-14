"""Fill template holes with Z3 so that every pattern instruction is bank-conflict-free."""
from typing import Optional, Sequence
import z3

from .machine import MachineModel
from .templates import BV, _bv


def _bank(word: z3.BitVecRef, machine: MachineModel) -> z3.BitVecRef:
    if machine.banks & (machine.banks - 1) == 0:
        return word & _bv(machine.banks - 1)
    return z3.URem(word, _bv(machine.banks))


def _word(flat: z3.BitVecRef, elem_bytes: int, machine: MachineModel) -> z3.BitVecRef:
    byte = flat * _bv(elem_bytes)
    if machine.bank_bytes & (machine.bank_bytes - 1) == 0:
        return z3.LShR(byte, machine.bank_bytes.bit_length() - 1)
    return z3.UDiv(byte, _bv(machine.bank_bytes))


def conflict_free_constraints(template, patterns: Sequence, elem_bytes: int,
                              machine: MachineModel, h) -> list:
    """Pairwise: two lanes in one wavefront group either touch the same bank word or different banks."""
    width = machine.banks * machine.bank_bytes
    group = min(machine.warp, max(1, width // elem_bytes))
    words_per_lane = max(1, elem_bytes // machine.bank_bytes)
    cs = []
    for pat in patterns:
        for k in range(pat.n_instr):
            ii, jj = pat.coords(k)
            coords = list(zip(ii.tolist(), jj.tolist()))
            for start in range(0, len(coords), group):
                grp = coords[start:start + group]
                words = {c: _word(template.flat_z3(c[0], c[1], h), elem_bytes, machine) for c in set(grp)}
                uniq = list(words)
                for a in range(len(uniq)):
                    for b in range(a + 1, len(uniq)):
                        for u in range(words_per_lane):
                            for v in range(words_per_lane):
                                wa = words[uniq[a]] + _bv(u)
                                wb = words[uniq[b]] + _bv(v)
                                cs.append(z3.Or(wa == wb, _bank(wa, machine) != _bank(wb, machine)))
    return cs


def solve_conflict_free(template, patterns: Sequence, elem_bytes: int, machine: MachineModel,
                        timeout_ms: int = 60_000, minimize: Optional[str] = None):
    """Return an instantiated layout with zero conflicts on all patterns, or None.

    If `minimize` names a hole, the smallest feasible value of that hole is returned.
    """
    h = {name: z3.BitVec(name, BV) for name in template.holes()}
    s = z3.Optimize() if minimize else z3.Solver()
    s.set("timeout", timeout_ms)
    if minimize:
        s.minimize(h[minimize])
    for name, (lo, hi) in template.holes().items():
        s.add(z3.UGE(h[name], _bv(lo)), z3.ULE(h[name], _bv(hi)))
    s.add(*template.side_constraints(h))
    s.add(*conflict_free_constraints(template, patterns, elem_bytes, machine, h))
    if s.check() != z3.sat:
        return None
    m = s.model()
    return template.instantiate({name: m.eval(h[name], model_completion=True).as_long() for name in h})
