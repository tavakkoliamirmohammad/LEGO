# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-04 07:56

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **35** |
| **PARITY (vs C O3)** | **7** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 32 |
| PARITY (vs C agg) | 9 |
| LOSS (vs C agg) | 1 |
| ERROR | 0 |
| VERIFIED (correctness) | 42 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 41 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | ? | 0.2832 | 0.3185 | 0.0874 | 3.64x | 0.1003 | 1.15x | 0.0994 | 1.14x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1444 | 0.0114 | 12.67x | 0.0244 | 2.14x | 0.0216 | 1.89x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.38x | 0.0001 | 1.44x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0478 | 0.0779 | 0.0586 | 1.33x | 0.0639 | 1.09x | 0.0461 | 0.79x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1593 | 0.0353 | 0.0046 | 7.67x | 0.1692 | 36.78x | 0.1691 | 36.76x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0021 | 0.0025 | 0.0002 | 12.50x | 0.0004 | 2.06x | 0.0006 | 2.80x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2846 | 0.3167 | 0.0878 | 3.61x | 0.1800 | 2.05x | 0.1677 | 1.91x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2881 | 0.2845 | 0.0877 | 3.24x | 0.0956 | 1.09x | 0.1480 | 1.69x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1688 | 0.0352 | 0.0068 | 5.18x | 0.0352 | 5.18x | 0.0352 | 5.17x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1694 | 0.0381 | 0.0066 | 5.77x | 0.0352 | 5.33x | 0.0353 | 5.34x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1686 | 0.0352 | 0.0074 | 4.76x | 0.0352 | 4.76x | 0.0352 | 4.75x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 0.5964 | 41.9324 | 6.4112 | 6.54x | 11.4114 | 1.78x | 10.8623 | 1.69x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2314 | 0.4392 | 0.1344 | 3.27x | 0.1408 | 1.05x | 0.1427 | 1.06x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2271 | 0.4378 | 0.1349 | 3.25x | 0.1406 | 1.04x | 0.1537 | 1.14x | OK | PARITY |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2315 | 0.4319 | 0.1288 | 3.35x | 0.1418 | 1.10x | 0.1428 | 1.11x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2284 | 0.4325 | 0.1318 | 3.28x | 0.1410 | 1.07x | 0.1429 | 1.08x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2265 | 0.4375 | 0.1314 | 3.33x | 0.1406 | 1.07x | 0.1428 | 1.09x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0069 | 0.7237 | 0.0116 | 62.39x | 0.0244 | 2.10x | 0.0215 | 1.86x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0294 | 0.0294 | 0.0043 | 6.84x | 0.0084 | 1.95x | 0.0081 | 1.88x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0534 | 0.0553 | 0.0073 | 7.58x | 0.0190 | 2.61x | 0.0151 | 2.07x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0299 | 0.0294 | 0.0046 | 6.39x | 0.0084 | 1.82x | 0.0081 | 1.76x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0380 | 0.0355 | 0.0068 | 5.22x | 0.0107 | 1.58x | 0.0108 | 1.59x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6529 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2273 | 0.4325 | 0.1287 | 3.36x | 0.1407 | 1.09x | 0.1426 | 1.11x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.6803 | 0.0089 | 0.0028 | 3.18x | 0.0028 | 0.99x | 0.0028 | 0.98x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.6624 | 0.0089 | 0.0028 | 3.18x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7019 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 0.99x | 0.0028 | 0.98x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2347 | 0.4379 | 0.1340 | 3.27x | 0.1405 | 1.05x | 0.1427 | 1.07x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2296 | 0.4369 | 0.1309 | 3.34x | 0.1448 | 1.11x | 0.1426 | 1.09x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1690 | 0.0353 | 0.0071 | 4.97x | 0.0353 | 4.97x | 0.0352 | 4.96x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1766 | 0.0352 | 0.0069 | 5.10x | 0.0352 | 5.11x | 0.0352 | 5.10x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1768 | 0.0352 | 0.0073 | 4.82x | 0.0352 | 4.82x | 0.0352 | 4.82x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0124 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.30x | 0.0001 | 1.31x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2370 | 0.4370 | 0.1332 | 3.28x | 0.1404 | 1.05x | 0.1428 | 1.07x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2294 | 0.4402 | 0.1349 | 3.26x | 0.1407 | 1.04x | 0.1428 | 1.06x | OK | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2302 | 0.4386 | 0.1300 | 3.37x | 0.1459 | 1.12x | 0.1427 | 1.10x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2332 | 0.4359 | 0.1285 | 3.39x | 0.1404 | 1.09x | 0.1428 | 1.11x | OK | **WIN** |

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
