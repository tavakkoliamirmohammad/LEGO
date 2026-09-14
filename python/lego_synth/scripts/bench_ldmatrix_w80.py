"""Hardware check of the width-80 staging result: ldmatrix.x4 on a 64x80 b16 tile.

Usage: python -m lego_synth.scripts.bench_ldmatrix_w80 --out lego_synth/results/ldmatrix_w80.md
"""
import argparse
import torch

from lego_synth.layouts import RowMajor, XorSwizzle
from lego_synth.harness.ldmatrix_bench import LdmatrixBench, predicted_wavefronts_per_ldmatrix


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); args = ap.parse_args()
    layouts = {"naive_160B": RowMajor(64, 80), "pad8_176B": RowMajor(64, 80, pad=8),
               "synth_exact": XorSwizzle(64, 80, shift=2, mask=1, unit=8),
               "pow2_pad128_sw128": XorSwizzle(64, 128, shift=0, mask=7, unit=8)}
    b = LdmatrixBench(layouts, rows=64, cols=80).build()
    lines = [f"### ldmatrix.x4 loop on {torch.cuda.get_device_name()} (64x80 b16 tile, 1024 reps, 336 blocks x 4 warps)", "",
             "| layout | footprint (elems) | fragments correct | predicted wavefronts / ldmatrix.x4 | measured ms |", "|---|---|---|---|---|"]
    for name, L in layouts.items():
        ok = b.check(name)
        ms = b.time(name, reps=1024, blocks=84 * 4)
        lines.append(f"| {name} | {L.footprint} | {ok} | {predicted_wavefronts_per_ldmatrix(L)} | {ms:.3f} |")
    text = "\n".join(lines); print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
