"""Space ablation: one solver, three expressiveness tiers, a sweep of tile widths."""
from typing import List, Sequence

from .eval_np import conflicts
from .machine import ampere
from .problems import problem_for
from .search import synthesize, TIERS


def ablation_rows(widths: Sequence[int], rows: int, elem_bytes: int, machine=ampere(),
                  tiers: Sequence[str] = TIERS, kind: str = "transpose") -> List[dict]:
    out = []
    for w in widths:
        prob = problem_for(kind, rows, w, elem_bytes, machine)
        pats = prob.patterns
        for tier in tiers:
            sol = synthesize(prob, tier)
            row = {"width": w, "tier": tier, "template": None, "footprint": None,
                   "overhead": None, "conflicts": None, "solve_time": None}
            if sol is not None:
                # Verify against the patterns the tier actually solved (f2 pads the tile).
                verified = sum(conflicts(sol.layout, p, elem_bytes, machine) for p in sol.problem.patterns)
                row.update(template=sol.template, footprint=sol.footprint,
                           overhead=sol.footprint / (rows * w), conflicts=verified,
                           solve_time=sol.solve_time, layout=sol.layout)
            out.append(row)
    return out


def to_markdown(rows: List[dict]) -> str:
    tiers = list(dict.fromkeys(r["tier"] for r in rows))
    widths = list(dict.fromkeys(r["width"] for r in rows))
    by = {(r["width"], r["tier"]): r for r in rows}
    head = "| width | " + " | ".join(f"{t} footprint (overhead, template)" for t in tiers) + " |"
    sep = "|---|" + "|".join("---" for _ in tiers) + "|"
    lines = [head, sep]
    for w in widths:
        cells = []
        for t in tiers:
            r = by[(w, t)]
            if r["footprint"] is None:
                cells.append("none")
            else:
                cells.append(f"{r['footprint']} ({r['overhead']:.2f}x, {r['template']})")
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
