# Triton 3.6.0 behavior used as the F2 baseline (measured 2026-09-14, RTX A6000, sm_86)

Probe scripts: `triton_width_probe.py`, `triton_dot_swizzle.py` (session scratchpad; trivial to recreate:
a `tl.load` of a BM x BN half-precision tile followed by a transposed store, and a `tl.dot` tile).

1. Non-power-of-two tile widths are refused at compile time:
   - BN=80: `arange's range must be a power of 2`
   - BN=96: `arange's range must be a power of 2`
   Consequence: a Triton kernel with head dimension 80 or 96 allocates 128-wide tiles and masks,
   which is what the `f2` tier models (block shape padded to the next power of two).

2. Shared-memory layout Triton chooses for `tl.dot` operands (b16), read via ldmatrix (PTX confirmed):
   - BK=64  (128 B rows): `#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>`
   - BK=128 (256 B rows): `#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>`
   Both are "chunk ^= row % 8" on 16-byte chunks, i.e. `XorSwizzle(shift=0, mask=7, unit=8)`,
   the same solution `lego_synth` produces at widths 64 and 128 in every tier.
