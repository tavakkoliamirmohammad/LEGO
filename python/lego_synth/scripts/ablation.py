"""Space ablation CLI: same solver, three tiers, a sweep of tile widths.

Usage:
  python -m lego_synth.scripts.ablation --kind transpose --elem 4 --rows 32 --widths 24,32,40,48,64,80,96,112,128
  python -m lego_synth.scripts.ablation --kind staging   --elem 2 --rows 64 --widths 32,48,64,80,96,112,128
"""
import argparse
import time

from lego_synth.ablation import ablation_rows, to_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="transpose", choices=["transpose", "staging"])
    ap.add_argument("--elem", type=int, default=4)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--widths", default="24,32,40,48,64,80,96,112,128")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]
    t0 = time.time()
    rows = ablation_rows(widths=widths, rows=args.rows, elem_bytes=args.elem, kind=args.kind)
    md = [f"### {args.kind}, elem={args.elem}B, rows={args.rows}: conflict-free footprint by tier", "",
          to_markdown(rows), "",
          "solutions: " + "; ".join(f"{r['width']}/{r['tier']}={r.get('layout')}" for r in rows if r.get("layout") is not None), "",
          "solve times (s): " + ", ".join(f"{r['width']}/{r['tier']}={r['solve_time']:.1f}" for r in rows if r['solve_time']),
          f"total {time.time() - t0:.0f}s"]
    text = "\n".join(md)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
