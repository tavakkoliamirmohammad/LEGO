"""Calibrate the wavefront model on hardware: smem-bound loop time vs predicted wavefronts.

Usage: python -m lego_synth.scripts.calibrate_smem --out lego_synth/results/smem_calibration.md
"""
import argparse
import os
import torch
from lego_synth.kernels.target import detect_target, metadata_header

from lego_synth.layouts import RowMajor, XorSwizzle
from lego_synth.harness.smem_bench import SmemBench, predicted_wavefronts_per_instr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); args = ap.parse_args()
    # XorSwizzle(shift=s, mask=m): a column read groups 2^s lanes per bank -> 2^s wavefronts.
    layouts = {"w1": XorSwizzle(32, 32, 0, 31), "w2": XorSwizzle(32, 32, 1, 15), "w4": XorSwizzle(32, 32, 2, 7),
               "w8": XorSwizzle(32, 32, 3, 3), "w16": XorSwizzle(32, 32, 4, 1), "w32": RowMajor(32, 32),
               "pad1": RowMajor(32, 32, pad=1)}
    b = SmemBench(layouts).build()
    lines = [f"### smem-bound column-read loop on {torch.cuda.get_device_name()} (32x32 f32 tile, 2048 reps, 672 blocks)", "",
             "| layout | predicted wavefronts/instr | measured ms | ratio to 1 wavefront |", "|---|---|---|---|"]
    base = None
    for name, L in layouts.items():
        ms = b.time(name, reps=2048)
        base = base or ms
        lines.append(f"| {name} | {predicted_wavefronts_per_instr(L)} | {ms:.3f} | {ms / base:.2f} |")
    target = detect_target()
    text = metadata_header(target) + "\n".join(lines); print(text)
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "results", target.tag, "smem_calibration.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write(text + "\n")


if __name__ == "__main__":
    main()
