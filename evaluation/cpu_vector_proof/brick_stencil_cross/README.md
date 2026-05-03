# Cross-brick stencil proof-point benchmark

Demonstrates `@cpu_kernel` + `--lego-to-x86-vector` on a stencil-style kernel.

## v1 status — within-brick pattern only

**This benchmark currently exercises a within-brick pattern (1D 3-point
stencil) and does NOT perform cross-brick neighbor reads.**

Full cross-brick stencil correctness (e.g. 3D 7-point stencil with halos)
requires the brick-aware second-block base computation, tracked as roadmap
entry **R12**. When R12 lands, this benchmark will be updated to enable the
cross-brick reads and the full 3D kernel.

### Why cross-brick reads are not in v1

Phase C (Task 13) built the IR-shape scaffolding for cross-brick shuffle
emission. The second-block base address is currently computed from the
boundary lane index (correct for the FileCheck IR shape test). For real
cross-brick stencil correctness, the base must use the actual *brick stride*
so that neighbor brick's data is found at the right offset. This gap flips
candidates 11, 12, 13, 14, and 29 from WIN to LOSS in the current evaluation.
See the full analysis in `evaluation/roadmap.md` under R12.

## Kernel (v1)

1D 3-point stencil:

```
B[i+1] = A[i] + A[i+1] + A[i+2]
```

for `i` in `[0, N-2)`, tiled at TILE=16. This exercises the inner-axis
vectorization path without cross-brick complexity.

## Run

```bash
source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_vector_proof/brick_stencil_cross
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python python measure.py
```

## Measured result (v1, Intel Xeon Gold 6330)

JIT compilation **fails** at the `lego-to-x86-vector` pipeline stage on this
kernel:

```
error: 'memref.store' op failed to verify that type of 'value' matches element type of 'memref'
  "memref.store"(%259, %arg1, %90) : (vector<16xf32>, memref<1024xf32>, index) -> ()
```

The pipeline vectorizes the load (`A[i] + A[i+1] + A[i+2]`) to `vector<16xf32>`
but the store target `B[i+1]` remains a scalar `memref<1024xf32>`. The
vector-to-memref store lowering is missing for stencil patterns where the output
index differs from the tile-local variable (here `i+1` instead of `i`). This is
a v1 infrastructure gap; `measure.py` falls back to scalar-only measurement with
`"speedup": "N/A"`.

The within-brick benchmark (`brick_within_cell/`) does compile and run because
its store index matches the tile IV exactly (`B[i]`).

Fix path: extend the `lego-to-x86-vector` pipeline's vector-store lowering to
handle affine-offset stores (`B[i+k]` for compile-time constant `k`). This is
a narrower infrastructure item than full R12.

## Expected outcome (after pipeline fix)

`speedup >= 2.0x` over the numpy scalar baseline on AVX-512 hardware for the
1D 3-point stencil. The cross-brick speedup numbers (expected 1.9x-4.9x per
published BrickLib results) will be measurable after R12 lands.

## Roadmap entries

| Entry | Description | Status |
|-------|-------------|--------|
| R1    | CPU vector pipeline shipped | CLOSED in v1 |
| R12   | Cross-brick shuffle / brick-stride second-block base | open — imminent |

### What R12 requires

1. Compute the second-block base address from the actual brick stride
   (number of floats per brick × brick index), not from the boundary lane.
2. Emit cross-block vector shuffles for the neighbor-lane boundary loads.
3. Update this benchmark: replace `stencil_3pt` with a 3D 7-point stencil
   that reads from `A[i-NX*NY]`, `A[i-NX]`, `A[i-1]`, `A[i]`, `A[i+1]`,
   `A[i+NX]`, `A[i+NX*NY]` using the brick-aware base.

### Affected evaluation candidates (flip LOSS → WIN when R12 lands)

- 11 (`bricklib-3d7pt-brick`) — currently LOSS 0.93× AMD / LOSS 0.86× Intel
- 12 (`bricklib-3d13pt-brick`) — currently PARITY AMD / marginal WIN 1.06× Intel
- 13 (`polybench-heat3d-brick`) — currently MIXED AMD / LOSS 0.80× Intel
- 14 (`polybench-jacobi2d-brick`) — currently LOSS 0.67× AMD / LOSS 0.19× Intel
- 29 (`bricklib-stencil-nonpow2-brick`) — currently LOSS 0.75× AMD / LOSS 0.61× Intel
