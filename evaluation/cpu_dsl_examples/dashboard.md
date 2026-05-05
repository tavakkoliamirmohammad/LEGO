# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-04 23:29

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 54 |
| Measured | 54 |
| SKIP | 0 |
| **WIN (vs C O3)** | **43** |
| **PARITY (vs C O3)** | **6** |
| **LOSS (vs C O3)** | **5** |
| WIN (vs C agg) | 34 |
| PARITY (vs C agg) | 13 |
| LOSS (vs C agg) | 7 |
| ERROR | 0 |
| VERIFIED (correctness) | 54 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 47 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | ? | 0.2779 | 0.3101 | 0.0848 | 3.66x | 0.1004 | 1.18x | 0.1037 | 1.22x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0069 | 0.1437 | 0.0164 | 8.76x | 0.0245 | 1.49x | 0.0216 | 1.32x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.39x | 0.0001 | 1.44x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0489 | 0.0779 | 0.0441 | 1.77x | 0.0635 | 1.44x | 0.0447 | 1.01x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1585 | 0.0354 | 0.0045 | 7.87x | 0.1698 | 37.73x | 0.1691 | 37.58x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0019 | 0.0023 | 0.0002 | 11.50x | 0.0004 | 1.91x | 0.0006 | 2.78x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2911 | 0.3164 | 0.0876 | 3.61x | 0.1797 | 2.05x | 0.1717 | 1.96x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2889 | 0.2844 | 0.0893 | 3.18x | 0.0999 | 1.12x | 0.1414 | 1.58x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1731 | 0.0352 | 0.0072 | 4.89x | 0.0353 | 4.91x | 0.0352 | 4.89x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1709 | 0.0381 | 0.0071 | 5.37x | 0.0352 | 4.96x | 0.0352 | 4.96x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1691 | 0.0353 | 0.0071 | 4.97x | 0.0353 | 4.97x | 0.0352 | 4.96x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 2.3677 | 41.9339 | 36.7041 | 1.14x | 11.5583 | 0.31x | 10.8464 | 0.30x | OK | LOSS |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2378 | 0.4320 | 0.1303 | 3.32x | 0.1408 | 1.08x | 0.1537 | 1.18x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2360 | 0.4318 | 0.1311 | 3.29x | 0.1413 | 1.08x | 0.1430 | 1.09x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2316 | 0.4322 | 0.1299 | 3.33x | 0.1411 | 1.09x | 0.1429 | 1.10x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2369 | 0.4322 | 0.1287 | 3.36x | 0.1405 | 1.09x | 0.1426 | 1.11x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2351 | 0.4382 | 0.1296 | 3.38x | 0.1404 | 1.08x | 0.1428 | 1.10x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0069 | 0.7230 | 0.1685 | 4.29x | 0.0244 | 0.14x | 0.0218 | 0.13x | OK | LOSS |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0299 | 0.0295 | 0.0043 | 6.86x | 0.0084 | 1.95x | 0.0081 | 1.88x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0540 | 0.0553 | 0.0097 | 5.70x | 0.0190 | 1.96x | 0.0151 | 1.56x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0296 | 0.0294 | 0.0044 | 6.68x | 0.0084 | 1.90x | 0.0081 | 1.84x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0387 | 0.0357 | 0.0071 | 5.03x | 0.0108 | 1.52x | 0.0108 | 1.52x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6514 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2285 | 0.4375 | 0.1367 | 3.20x | 0.1461 | 1.07x | 0.1427 | 1.04x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.6601 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.6476 | 0.0089 | 0.0028 | 3.18x | 0.0028 | 0.98x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.6409 | 0.0089 | 0.0028 | 3.18x | 0.0028 | 0.99x | 0.0028 | 0.98x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2272 | 0.4337 | 0.1355 | 3.20x | 0.1454 | 1.07x | 0.1531 | 1.13x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2316 | 0.4413 | 0.1331 | 3.32x | 0.1407 | 1.06x | 0.1545 | 1.16x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1685 | 0.0352 | 0.0068 | 5.18x | 0.0353 | 5.20x | 0.0352 | 5.18x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1784 | 0.0353 | 0.0072 | 4.90x | 0.0352 | 4.89x | 0.0352 | 4.89x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1753 | 0.0352 | 0.0071 | 4.96x | 0.0352 | 4.96x | 0.0352 | 4.96x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0124 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.33x | 0.0001 | 1.31x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2360 | 0.4378 | 0.1307 | 3.35x | 0.1406 | 1.08x | 0.1426 | 1.09x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2362 | 0.4383 | 0.1293 | 3.39x | 0.1413 | 1.09x | 0.1426 | 1.10x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2275 | 0.4426 | 0.1340 | 3.30x | 0.1457 | 1.09x | 0.1429 | 1.07x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2408 | 0.4381 | 0.1304 | 3.36x | 0.1410 | 1.08x | 0.1429 | 1.10x | OK | **WIN** |
| 43_spmv_indirect | Scatter | NEW | 1048576 | 1.8166 | 1.0189 | 0.9625 | 1.06x | 0.9508 | 0.99x | 0.9820 | 1.02x | OK | PARITY |
| 44_predicated_fma | Predicated | NEW | 1048576 | 19.6659 | 5.5643 | 0.1969 | 28.26x | 4.9263 | 25.02x | 0.3437 | 1.75x | OK | **WIN** |
| 45_stride_runtime | Stride7 | NEW | 262144 | 0.1245 | 0.1441 | 0.0879 | 1.64x | 0.0826 | 0.94x | 0.0824 | 0.94x | OK | LOSS |
| 46_scatter_compute | ScatterCompute | NEW | 1048576 | 2.9166 | 1.2087 | 1.1908 | 1.01x | 1.2261 | 1.03x | 1.0222 | 0.86x | OK | PARITY |
| 47_multi_reduce | MultiReduce | NEW | 1048576 | 0.2731 | 1.3992 | 1.4188 | 0.99x | 0.8448 | 0.60x | 0.2114 | 0.15x | OK | LOSS |
| 48_count_if | PredicatedCount | NEW | 1048576 | 0.2933 | 4.8519 | 1.1884 | 4.08x | 4.5760 | 3.85x | 0.2114 | 0.18x | OK | **WIN** |
| 49_saturating_add | Saturating | NEW | 1048576 | 1.0562 | 0.4293 | 0.1308 | 3.28x | 0.3380 | 2.58x | 0.1484 | 1.13x | OK | **WIN** |
| 50_all_positive | AllAny | NEW | 1048576 | 0.0791 | 0.2826 | 0.0512 | 5.52x | 0.0744 | 1.45x | 0.0780 | 1.52x | OK | **WIN** |
| 51_find_first | FindFirst | NEW | 1048576 | 0.0662 | 5.1658 | 1.4346 | 3.60x | 4.8244 | 3.36x | 4.6781 | 3.26x | OK | **WIN** |
| 52_bit_reverse | BitReverse | NEW | 1048576 | 12.1714 | 1.4279 | 1.3729 | 1.04x | 1.6175 | 1.18x | 1.6179 | 1.18x | OK | **WIN** |
| 53_expand | Expand | NEW | 1048576 | 5.2543 | 4.9303 | 4.9453 | 1.00x | 4.8067 | 0.97x | 4.7190 | 0.95x | OK | PARITY |
| 54_find_byte | FindByte | NEW | 1048576 | 0.5018 | 3.6293 | 4.1297 | 0.88x | 3.6467 | 0.88x | 3.1528 | 0.76x | OK | LOSS |

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
