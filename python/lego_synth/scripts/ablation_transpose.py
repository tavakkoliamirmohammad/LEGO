"""Reproduce the transpose-staging space ablation for a given element size.

Usage: python -m lego_synth.scripts.ablation_transpose --elem 4 --rows 32 --out results/ablation_f32.md
"""
import argparse
import time

from lego_synth.ablation import ablation_rows, to_markdown

WIDTHS = [24, 32, 40, 48, 64, 80, 96, 112, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elem", type=int, default=4)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    rows = ablation_rows(widths=WIDTHS, rows=args.rows, elem_bytes=args.elem)
    md = [f"### transpose staging, elem={args.elem}B, rows={args.rows}: conflict-free footprint by tier", "",
          to_markdown(rows), "",
          "solve times (s): " + ", ".join(f"{r['width']}/{r['tier']}={r['solve_time']:.1f}" for r in rows if r['solve_time']),
          f"total {time.time() - t0:.0f}s"]
    text = "\n".join(md)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
