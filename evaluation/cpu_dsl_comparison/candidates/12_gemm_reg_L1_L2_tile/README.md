# 12_gemm_reg_L1_L2_tile

**CASTLE candidate:** 4 — GEMM with register + L1 + L2 tiling (unit-stride inner loop)
**Layout class:** Reg+L1+L2 tile
**Prior verdicts:** AMD WIN, Intel WIN

## Kernel

GEMM with register + L1 + L2 tiling (unit-stride inner loop)

## Expected behavior

Expected WIN/PARITY vs scalar-JIT.
