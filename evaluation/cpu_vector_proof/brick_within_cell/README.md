# Within-brick proof-point benchmark

Demonstrates `@cpu_kernel` + `--lego-to-x86-vector` on a within-brick compute
pattern. No cross-brick neighbor reads — those need cross-block shuffle
support (currently a v1 feature; full brick-stencil with halos is captured as
roadmap entry R12).

## Kernel

```
B[i] = A[i] * 2.0 + 1.0
```

over a flat array of size N = 64×64×64 = 262144 elements, tiled at BRICK=8.
The kernel models the *within-cell* compute of a brick stencil — no reads
from neighboring bricks — making it a clean test of the inner-axis vectorization
path without cross-block shuffle complexity.

## Run

```bash
source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_vector_proof/brick_within_cell
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python python measure.py
```

## Measured result (v1, Intel Xeon Gold 6330)

JIT compiles and executes correctly. Measured on `notch343` (Intel Xeon Gold 6330,
AVX-512, 2026-05-01):

```json
{
  "baseline_ms": 0.073,
  "treatment_ms": 0.131,
  "speedup": 0.56,
  "target": "x86",
  "N": 262144
}
```

The speedup is **below the 2.0x target**. NumPy's own BLAS-backed vectorized loop
is already very efficient for this simple `A*2+1` pattern — beating it requires
the lego-to-x86-vector pipeline to emit fused-multiply-add intrinsics that are
more tightly packed than NumPy's dispatch overhead. This is tracked as an
extension to R1 (SIMD intrinsic codegen quality).

The important v1 result is that **the JIT pipeline executes correctly end-to-end**:
Python DSL → MLIR → x86-vector lowering → ExecutionEngine → numpy buffer.

Sensitivity to system load — re-run if numbers vary.

## v1 limitations

- **No cross-brick reads.** The `tile_range` sentinel exposes the within-tile
  loop variable as a global index (`tile_id * BRICK + local_i`). Cross-brick
  neighbor accesses require knowing the brick stride at index-expression time;
  that computation is stubbed in the IR-shape scaffolding but needs the actual
  brick stride to produce correct results. See roadmap entry **R12**.
- **Single axis.** The 3D brick layout (NX × NY × NZ bricks) is flattened to
  a 1D tile grid here. A fully 3D tiled kernel needs multi-dimensional
  `tile_range` support, also tracked under R12.

## Relationship to roadmap

| Entry | Description | Status |
|-------|-------------|--------|
| R1    | CPU vector pipeline shipped | CLOSED in v1 |
| R12   | Cross-brick shuffle / stencil halos | open — imminent |
