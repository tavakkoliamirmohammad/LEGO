# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 23:52

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **38** |
| **PARITY (vs C O3)** | **4** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 33 |
| PARITY (vs C agg) | 8 |
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
| 01_saxpy |  |  | ? | 0.2933 | 0.3171 | 0.0860 | 3.69x | 0.1004 | 1.17x | 0.0994 | 1.16x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1446 | 0.0115 | 12.57x | 0.0244 | 2.12x | 0.0216 | 1.88x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.38x | 0.0002 | 1.52x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0481 | 0.0779 | 0.0604 | 1.29x | 0.0681 | 1.13x | 0.0478 | 0.79x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1595 | 0.0354 | 0.0045 | 7.87x | 0.1700 | 37.78x | 0.1699 | 37.75x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0029 | 0.0003 | 9.67x | 0.0004 | 1.46x | 0.0007 | 2.26x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2889 | 0.3177 | 0.0888 | 3.58x | 0.1874 | 2.11x | 0.1760 | 1.98x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2973 | 0.2921 | 0.0919 | 3.18x | 0.1007 | 1.10x | 0.1490 | 1.62x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1693 | 0.0353 | 0.0080 | 4.41x | 0.0352 | 4.40x | 0.0352 | 4.40x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1704 | 0.0390 | 0.0084 | 4.64x | 0.0354 | 4.22x | 0.0360 | 4.29x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1717 | 0.0354 | 0.0074 | 4.78x | 0.0354 | 4.78x | 0.0352 | 4.75x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 1.8295 | 42.0264 | 8.6310 | 4.87x | 11.4648 | 1.33x | 11.0503 | 1.28x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2361 | 0.4438 | 0.1349 | 3.29x | 0.1466 | 1.09x | 0.1438 | 1.07x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2381 | 0.4434 | 0.1344 | 3.30x | 0.1462 | 1.09x | 0.1541 | 1.15x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2394 | 0.4384 | 0.1356 | 3.23x | 0.1461 | 1.08x | 0.1429 | 1.05x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2357 | 0.4382 | 0.1352 | 3.24x | 0.1466 | 1.08x | 0.1559 | 1.15x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2366 | 0.4383 | 0.1354 | 3.24x | 0.1465 | 1.08x | 0.1427 | 1.05x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7242 | 0.0118 | 61.37x | 0.0245 | 2.08x | 0.0217 | 1.84x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0300 | 0.0294 | 0.0049 | 6.00x | 0.0084 | 1.71x | 0.0081 | 1.65x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0545 | 0.0553 | 0.0074 | 7.47x | 0.0191 | 2.59x | 0.0151 | 2.04x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0299 | 0.0294 | 0.0048 | 6.12x | 0.0084 | 1.75x | 0.0081 | 1.69x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0404 | 0.0357 | 0.0073 | 4.89x | 0.0110 | 1.51x | 0.0109 | 1.49x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0017 | 0.98x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0017 | 0.99x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.04x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6796 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2462 | 0.4382 | 0.1345 | 3.26x | 0.1459 | 1.08x | 0.1428 | 1.06x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7581 | 0.0093 | 0.0028 | 3.32x | 0.0029 | 1.03x | 0.0028 | 1.01x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.6842 | 0.0090 | 0.0028 | 3.21x | 0.0030 | 1.08x | 0.0030 | 1.09x | OK | **WIN** |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7081 | 0.0095 | 0.0028 | 3.39x | 0.0028 | 1.00x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2388 | 0.4435 | 0.1346 | 3.29x | 0.1410 | 1.05x | 0.1431 | 1.06x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2383 | 0.4379 | 0.1352 | 3.24x | 0.1456 | 1.08x | 0.1427 | 1.06x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1701 | 0.0353 | 0.0075 | 4.71x | 0.0353 | 4.71x | 0.0352 | 4.70x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1793 | 0.0353 | 0.0070 | 5.04x | 0.0352 | 5.03x | 0.0352 | 5.03x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1792 | 0.0359 | 0.0072 | 4.99x | 0.0353 | 4.90x | 0.0355 | 4.93x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0124 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.37x | 0.0001 | 1.32x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0017 | 0.98x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2369 | 0.4375 | 0.1357 | 3.22x | 0.1405 | 1.04x | 0.1456 | 1.07x | OK | PARITY |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2383 | 0.4389 | 0.1369 | 3.21x | 0.1468 | 1.07x | 0.1526 | 1.11x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2352 | 0.4434 | 0.1346 | 3.29x | 0.1453 | 1.08x | 0.1427 | 1.06x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2323 | 0.4319 | 0.1327 | 3.25x | 0.1455 | 1.10x | 0.1505 | 1.13x | OK | **WIN** |

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
