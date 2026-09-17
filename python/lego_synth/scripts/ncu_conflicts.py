"""Shared-memory bank conflicts measured with Nsight Compute vs the synthesizer's model, for the
32x32 f32 transpose harness (row-vector cp stores, column-vector loads).  Needs `ncu` on PATH and
readable counters; writes results/<device-tag>/ncu_conflicts.md."""
import os
import sys
from lego_synth.layouts import RowMajor, XorSwizzle, Rotation
from lego_synth.harness.transpose_cuda import TransposeHarness, predicted_conflicts
from lego_synth.kernels.target import detect_target, metadata_header

LAYOUTS = {"naive": RowMajor(32, 32), "pad1": RowMajor(32, 32, pad=1),
           "xor": XorSwizzle(32, 32, shift=0, mask=31), "rot1": Rotation(32, 32, r=1)}


def main(n=4096):
    target = detect_target()
    h = TransposeHarness(LAYOUTS, tile=32, block_rows=8)
    h.build()
    measured = h.ncu_shared_conflicts(n=n)
    lines = [metadata_header(target), f"### Shared-memory bank conflicts, {n}x{n} f32 transpose (32x32 tiles): Nsight Compute vs model", "",
             "| layout | predicted load conflicts | measured load conflicts | predicted store conflicts | measured store conflicts |", "|---|---|---|---|---|"]
    for name, L in LAYOUTS.items():
        pl, ps = predicted_conflicts(L, n=n, tile=32)
        ml, ms = measured[name]
        lines.append(f"| {name}: {L} | {pl:,} | {ml:,} | {ps:,} | {ms:,} |")
        print(lines[-1], flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", target.tag, "ncu_conflicts.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4096)
