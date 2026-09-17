"""Synthesize one certified layout for a named problem and print it as JSON.

Usage: python -m lego_synth.scripts.synth --kind staging --rows 64 --width 80 --elem 2 --tier full
"""
import argparse
import dataclasses
import json
import numpy as np

from lego_synth import machine as machines
from lego_synth.eval_np import conflicts
from lego_synth.harness.cuda_emit import index_expr
from lego_synth.problems import problem_for
from lego_synth.search import synthesize


def _bijective(layout) -> bool:
    i = np.repeat(np.arange(layout.rows), layout.cols)
    j = np.tile(np.arange(layout.cols), layout.rows)
    f = layout.flat(i, j)
    return bool(np.unique(f).size == f.size and f.min() >= 0 and f.max() < layout.footprint)


def synth_json(kind: str, rows: int, width: int, elem: int, tier: str, machine: str = "ampere") -> str:
    m = getattr(machines, machine)()
    prob = problem_for(kind, rows, width, elem, m)
    sol = synthesize(prob, tier)
    out = {"problem": {"kind": kind, "rows": rows, "cols": width, "elem_bytes": elem, "machine": machine,
                       "patterns": [p.name for p in prob.patterns]},
           "tier": tier, "layout": None, "footprint": None, "verified": None,
           "cuda_index_expr": None, "solve_time_s": None}
    if sol is not None:
        L = sol.layout
        extra = sum(conflicts(L, p, elem, m) for p in sol.problem.patterns)
        out.update(layout={"type": type(L).__name__, **dataclasses.asdict(L)},
                   footprint=L.footprint,
                   verified={"extra_wavefronts": extra, "bijective": _bijective(L),
                             "solved_cols": sol.problem.cols},
                   cuda_index_expr=index_expr(L), solve_time_s=sol.solve_time)
    return json.dumps(out, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="staging"); ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--width", type=int, default=80); ap.add_argument("--elem", type=int, default=2)
    ap.add_argument("--tier", default="full"); ap.add_argument("--machine", default="ampere")
    a = ap.parse_args()
    print(synth_json(a.kind, a.rows, a.width, a.elem, a.tier, a.machine))


if __name__ == "__main__":
    main()
