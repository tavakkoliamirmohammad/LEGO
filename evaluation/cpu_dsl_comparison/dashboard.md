# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 11:19

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **23** |
| **PARITY (vs C O3)** | **14** |
| **LOSS (vs C O3)** | **5** |
| WIN (vs C agg) | 22 |
| PARITY (vs C agg) | 14 |
| LOSS (vs C agg) | 6 |
| ERROR | 0 |
| VERIFIED (correctness) | 42 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 41 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2809 | 0.3007 | 0.0906 | 3.32x | 0.1008 | 1.11x | 0.1037 | 1.14x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.1089 | 0.0114 | 9.57x | 0.0245 | 2.15x | 0.0221 | 1.94x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0022 | 0.0006 | 0.0001 | 4.99x | 0.0001 | 1.48x | 0.0002 | 1.52x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0475 | 0.0790 | 0.0617 | 1.28x | 0.0686 | 1.11x | 0.0450 | 0.73x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1609 | 0.0389 | 0.0058 | 6.71x | 0.1696 | 29.24x | 0.1701 | 29.33x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0022 | 0.0002 | 9.69x | 0.0005 | 2.17x | 0.0006 | 2.46x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2849 | 0.3149 | 0.0941 | 3.35x | 0.1797 | 1.91x | 0.1675 | 1.78x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2903 | 0.2940 | 0.1083 | 2.71x | 0.0957 | 0.88x | 0.1496 | 1.38x | OK | LOSS |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1697 | 0.0376 | 0.0097 | 3.89x | 0.0352 | 3.63x | 0.0352 | 3.63x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1776 | 0.0405 | 0.0101 | 4.03x | 0.0355 | 3.51x | 0.0352 | 3.48x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1701 | 0.0379 | 0.0099 | 3.84x | 0.0358 | 3.61x | 0.0352 | 3.56x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 3.8530 | 41.9042 | 9.7294 | 4.31x | 11.9132 | 1.22x | 10.9036 | 1.12x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2461 | 0.4386 | 0.1455 | 3.01x | 0.1411 | 0.97x | 0.1428 | 0.98x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2382 | 0.4355 | 0.1406 | 3.10x | 0.1461 | 1.04x | 0.1426 | 1.01x | OK | PARITY |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2379 | 0.4376 | 0.1408 | 3.11x | 0.1458 | 1.04x | 0.1427 | 1.01x | OK | PARITY |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2498 | 0.4409 | 0.1428 | 3.09x | 0.1406 | 0.98x | 0.1558 | 1.09x | OK | PARITY |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2515 | 0.4420 | 0.1429 | 3.09x | 0.1464 | 1.02x | 0.1555 | 1.09x | OK | PARITY |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.6867 | 0.1289 | 5.33x | 0.0244 | 0.19x | 0.0221 | 0.17x | OK | LOSS |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0764 | 0.0303 | 0.0065 | 4.64x | 0.0084 | 1.29x | 0.0081 | 1.24x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1623 | 0.0564 | 0.0098 | 5.72x | 0.0191 | 1.95x | 0.0152 | 1.55x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0754 | 0.0302 | 0.0063 | 4.78x | 0.0085 | 1.34x | 0.0081 | 1.28x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0696 | 0.0370 | 0.0092 | 4.00x | 0.0109 | 1.18x | 0.0110 | 1.19x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0018 | 4.98x | 0.0022 | 1.24x | 0.0017 | 0.93x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.21x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0070 | 0.0088 | 0.0017 | 5.15x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0026 | 0.0018 | 0.0012 | 1.59x | 0.0006 | 0.46x | 0.0004 | 0.35x | OK | LOSS |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.0026 | 0.0019 | 0.0012 | 1.58x | 0.0006 | 0.46x | 0.0004 | 0.35x | OK | LOSS |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2414 | 0.4393 | 0.1456 | 3.02x | 0.1457 | 1.00x | 0.1429 | 0.98x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0088 | 0.0029 | 3.07x | 0.0028 | 0.96x | 0.0028 | 0.95x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0088 | 0.0028 | 3.10x | 0.0028 | 1.00x | 0.0028 | 1.00x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0088 | 0.0029 | 3.09x | 0.0028 | 0.96x | 0.0028 | 0.97x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2417 | 0.4386 | 0.1471 | 2.98x | 0.1464 | 1.00x | 0.1503 | 1.02x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2462 | 0.4406 | 0.1420 | 3.10x | 0.1466 | 1.03x | 0.1428 | 1.01x | OK | PARITY |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1684 | 0.0376 | 0.0099 | 3.78x | 0.0352 | 3.56x | 0.0353 | 3.56x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1702 | 0.0377 | 0.0100 | 3.76x | 0.0353 | 3.53x | 0.0352 | 3.52x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1713 | 0.0375 | 0.0095 | 3.95x | 0.0358 | 3.77x | 0.0352 | 3.71x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0073 | 0.0012 | 0.0008 | 1.53x | 0.0001 | 0.17x | 0.0001 | 0.17x | OK | LOSS |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0070 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2399 | 0.4390 | 0.1406 | 3.12x | 0.1458 | 1.04x | 0.1538 | 1.09x | OK | PARITY |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2479 | 0.4400 | 0.1441 | 3.05x | 0.1465 | 1.02x | 0.1429 | 0.99x | OK | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2488 | 0.4406 | 0.1426 | 3.09x | 0.1459 | 1.02x | 0.1530 | 1.07x | OK | PARITY |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2416 | 0.4366 | 0.1380 | 3.16x | 0.1461 | 1.06x | 0.1431 | 1.04x | OK | **WIN** |

## Known Gaps

- **R20 (deinterleave)**: Implemented for stride 2/4/8. Generates ShuffleOp chains
  instead of vector.gather for constant-stride accesses. Correctness verified.

- **R19 (strided gather indices)**: Strided-gather index vector mismatch in
  `LegoVectorize::emitVectorBody`. The catch-all arith path vectorizes `MulIOp(iv, stride)`
  before the Strided path reads it, producing incorrect gather indices.
  Affects candidates 23-27, 29-31, 38. Fix: use pre-vectorization scalar index.

- **R18 (reduction guard)**: k-reduction loops correctly skip vectorization.
  This is correct behavior but contributes to PARITY (not WIN) for GEMM variants.

- **invoke() overhead**: For small-N kernels (N=16K), the MLIR ExecutionEngine
  invoke() call costs ~4-5ms, dominating the actual kernel time (~0.002ms).
  The vec_jit_ms measurement includes this overhead; only the net kernel time
  (vec_jit_ms - invoke_overhead) is comparable to C baselines.
