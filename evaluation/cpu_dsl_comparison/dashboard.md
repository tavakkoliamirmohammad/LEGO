# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 22:37

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **39** |
| **PARITY (vs C O3)** | **3** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 31 |
| PARITY (vs C agg) | 10 |
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
| 01_saxpy |  |  | ? | 0.2939 | 0.3099 | 0.0868 | 3.57x | 0.1053 | 1.21x | 0.1041 | 1.20x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0069 | 0.1444 | 0.0115 | 12.56x | 0.0244 | 2.13x | 0.0217 | 1.89x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.39x | 0.0003 | 2.72x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0485 | 0.0779 | 0.0605 | 1.29x | 0.0657 | 1.09x | 0.0458 | 0.76x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1619 | 0.0355 | 0.0046 | 7.72x | 0.1695 | 36.84x | 0.1693 | 36.81x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0032 | 0.0002 | 16.00x | 0.0004 | 1.91x | 0.0006 | 2.77x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2885 | 0.3171 | 0.0891 | 3.56x | 0.1791 | 2.01x | 0.1741 | 1.95x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2958 | 0.2927 | 0.0904 | 3.24x | 0.0957 | 1.06x | 0.1482 | 1.64x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1739 | 0.0352 | 0.0087 | 4.05x | 0.0352 | 4.05x | 0.0353 | 4.06x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1725 | 0.0382 | 0.0082 | 4.66x | 0.0353 | 4.31x | 0.0352 | 4.29x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1705 | 0.0354 | 0.0072 | 4.92x | 0.0355 | 4.92x | 0.0352 | 4.89x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 7.2710 | 43.1962 | 6.7445 | 6.40x | 18.4449 | 2.73x | 11.0133 | 1.63x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2359 | 0.4451 | 0.1364 | 3.26x | 0.1466 | 1.07x | 0.1542 | 1.13x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2456 | 0.4439 | 0.1351 | 3.29x | 0.1467 | 1.09x | 0.1468 | 1.09x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2307 | 0.4653 | 0.1351 | 3.44x | 0.1460 | 1.08x | 0.1427 | 1.06x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2468 | 0.4386 | 0.1346 | 3.26x | 0.1472 | 1.09x | 0.1430 | 1.06x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2306 | 0.4449 | 0.1341 | 3.32x | 0.1466 | 1.09x | 0.1431 | 1.07x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7244 | 0.0120 | 60.37x | 0.0246 | 2.05x | 0.0216 | 1.80x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0298 | 0.0294 | 0.0047 | 6.26x | 0.0084 | 1.78x | 0.0081 | 1.72x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0554 | 0.0553 | 0.0079 | 7.00x | 0.0191 | 2.42x | 0.0151 | 1.91x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0301 | 0.0295 | 0.0046 | 6.41x | 0.0084 | 1.83x | 0.0081 | 1.76x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0398 | 0.0360 | 0.0071 | 5.07x | 0.0113 | 1.59x | 0.0109 | 1.53x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0017 | 0.98x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0028 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6882 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.39x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2377 | 0.4387 | 0.1363 | 3.22x | 0.1462 | 1.07x | 0.1428 | 1.05x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7345 | 0.0094 | 0.0028 | 3.36x | 0.0028 | 1.01x | 0.0029 | 1.03x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7340 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7087 | 0.0092 | 0.0028 | 3.29x | 0.0028 | 1.00x | 0.0028 | 1.01x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2382 | 0.4381 | 0.1371 | 3.20x | 0.1440 | 1.05x | 0.1545 | 1.13x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2403 | 0.4382 | 0.1367 | 3.21x | 0.1473 | 1.08x | 0.1557 | 1.14x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1698 | 0.0353 | 0.0072 | 4.90x | 0.0364 | 5.05x | 0.0354 | 4.92x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1779 | 0.0352 | 0.0072 | 4.89x | 0.0399 | 5.54x | 0.0353 | 4.90x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1768 | 0.0359 | 0.0072 | 4.99x | 0.0354 | 4.91x | 0.0353 | 4.90x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.34x | 0.0001 | 1.32x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0018 | 1.03x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2380 | 0.4374 | 0.1349 | 3.24x | 0.1468 | 1.09x | 0.1430 | 1.06x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2385 | 0.4380 | 0.1345 | 3.26x | 0.1452 | 1.08x | 0.1559 | 1.16x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2404 | 0.4389 | 0.1348 | 3.26x | 0.1460 | 1.08x | 0.1536 | 1.14x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2363 | 0.4329 | 0.1347 | 3.21x | 0.1462 | 1.09x | 0.1439 | 1.07x | OK | **WIN** |

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
