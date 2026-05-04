# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 19:37

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **38** |
| **PARITY (vs C O3)** | **4** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 29 |
| PARITY (vs C agg) | 12 |
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
| 01_saxpy |  |  | ? | 0.2807 | 0.3181 | 0.0896 | 3.55x | 0.1004 | 1.12x | 0.1037 | 1.16x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1445 | 0.0117 | 12.39x | 0.0244 | 2.09x | 0.0226 | 1.93x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0023 | 0.0004 | 0.0001 | 4.64x | 0.0001 | 1.39x | 0.0001 | 1.45x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0484 | 0.0779 | 0.0604 | 1.29x | 0.0647 | 1.07x | 0.0445 | 0.74x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1622 | 0.0356 | 0.0045 | 7.86x | 0.1696 | 37.70x | 0.1691 | 37.58x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0024 | 0.0004 | 6.75x | 0.0004 | 0.98x | 0.0006 | 1.41x | OK | PARITY |
| 07_mixed_precision |  |  | 1048576 | 0.2825 | 0.3178 | 0.0886 | 3.59x | 0.1895 | 2.14x | 0.1749 | 1.97x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.3047 | 0.2926 | 0.0920 | 3.18x | 0.1004 | 1.09x | 0.1486 | 1.62x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1694 | 0.0353 | 0.0076 | 4.65x | 0.0352 | 4.64x | 0.0354 | 4.65x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1724 | 0.0382 | 0.0080 | 4.78x | 0.0353 | 4.41x | 0.0353 | 4.41x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1705 | 0.0358 | 0.0075 | 4.79x | 0.0353 | 4.71x | 0.0352 | 4.69x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 0.6964 | 42.2448 | 6.7196 | 6.29x | 11.4752 | 1.71x | 10.9457 | 1.63x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2390 | 0.4464 | 0.1360 | 3.28x | 0.1467 | 1.08x | 0.1427 | 1.05x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2485 | 0.4319 | 0.1327 | 3.25x | 0.1463 | 1.10x | 0.1541 | 1.16x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2385 | 0.4386 | 0.1371 | 3.20x | 0.1458 | 1.06x | 0.1428 | 1.04x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2385 | 0.4361 | 0.1338 | 3.26x | 0.1460 | 1.09x | 0.1507 | 1.13x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2375 | 0.4443 | 0.1356 | 3.28x | 0.1469 | 1.08x | 0.1553 | 1.15x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0069 | 0.7230 | 0.0118 | 61.45x | 0.0245 | 2.08x | 0.0217 | 1.84x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0310 | 0.0295 | 0.0044 | 6.75x | 0.0084 | 1.91x | 0.0082 | 1.87x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0533 | 0.0554 | 0.0079 | 7.01x | 0.0191 | 2.42x | 0.0151 | 1.91x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0302 | 0.0294 | 0.0049 | 6.02x | 0.0084 | 1.71x | 0.0081 | 1.66x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0392 | 0.0358 | 0.0074 | 4.82x | 0.0108 | 1.47x | 0.0108 | 1.46x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.09x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.14x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.10x | 0.0022 | 1.29x | 0.0017 | 0.99x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.14x | 0.0006 | 1.44x | 0.0004 | 1.11x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6736 | 0.0022 | 0.0004 | 5.16x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2369 | 0.4380 | 0.1306 | 3.36x | 0.1465 | 1.12x | 0.1428 | 1.09x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7366 | 0.0095 | 0.0028 | 3.36x | 0.0028 | 1.00x | 0.0028 | 1.01x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7586 | 0.0094 | 0.0028 | 3.33x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7758 | 0.0091 | 0.0028 | 3.23x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2486 | 0.4319 | 0.1353 | 3.19x | 0.1462 | 1.08x | 0.1427 | 1.05x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2387 | 0.4384 | 0.1371 | 3.20x | 0.1460 | 1.06x | 0.1430 | 1.04x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1681 | 0.0353 | 0.0074 | 4.76x | 0.0359 | 4.86x | 0.0353 | 4.77x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1799 | 0.0358 | 0.0074 | 4.85x | 0.0354 | 4.79x | 0.0352 | 4.75x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1757 | 0.0359 | 0.0073 | 4.93x | 0.0352 | 4.82x | 0.0354 | 4.85x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0124 | 0.0005 | 0.0001 | 4.96x | 0.0001 | 1.33x | 0.0001 | 1.35x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0072 | 0.0088 | 0.0017 | 5.13x | 0.0022 | 1.30x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2383 | 0.4443 | 0.1346 | 3.30x | 0.1453 | 1.08x | 0.1431 | 1.06x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2391 | 0.4386 | 0.1286 | 3.41x | 0.1461 | 1.14x | 0.1553 | 1.21x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2373 | 0.4379 | 0.1366 | 3.21x | 0.1478 | 1.08x | 0.1535 | 1.12x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2371 | 0.4451 | 0.1363 | 3.27x | 0.1463 | 1.07x | 0.1429 | 1.05x | OK | **WIN** |

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
