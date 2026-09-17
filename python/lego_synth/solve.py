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
    """Constraints making every wavefront group of every instruction conflict-free.

    Assumes the template is injective on the tile (all templates are), so lanes
    with distinct coordinates touch distinct element indices. For accesses at
    least one bank word wide that means distinct words, and conflict-freedom is
    exactly "all touched words fall in distinct banks": one `Distinct` per group.
    For sub-word accesses two lanes may legitimately share a word, so fall back
    to pairwise "same word or different bank".
    """
    cs = []
    for pat in patterns:
        access = pat.access_width(elem_bytes)
        width = machine.banks * machine.bank_bytes
        group = min(machine.warp, max(1, width // access))
        words_per_lane = max(1, access // machine.bank_bytes)
        subword = access < machine.bank_bytes
        for k in range(pat.n_instr):
            ii, jj = pat.coords(k)
            coords = list(zip(ii.tolist(), jj.tolist()))
            for start in range(0, len(coords), group):
                uniq = list(dict.fromkeys(coords[start:start + group]))
                flats = [template.flat_z3(i, j, h) for (i, j) in uniq]
                if access > machine.bank_bytes:
                    for f in flats:   # natural alignment of each lane's vector access
                        cs.append((f * _bv(elem_bytes)) & _bv(access - 1) == _bv(0))
                words = [_word(f, elem_bytes, machine) for f in flats]
                if not subword:
                    banks = [_bank(w + _bv(u), machine) for w in words for u in range(words_per_lane)]
                    if len(banks) > 1:
                        cs.append(z3.Distinct(*banks))
                    continue
                for a in range(len(words)):
                    for b in range(a + 1, len(words)):
                        wa, wb = words[a], words[b]
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
