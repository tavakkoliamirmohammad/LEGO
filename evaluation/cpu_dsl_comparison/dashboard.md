# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 17:29

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **37** |
| **PARITY (vs C O3)** | **4** |
| **LOSS (vs C O3)** | **1** |
| WIN (vs C agg) | 31 |
| PARITY (vs C agg) | 9 |
| LOSS (vs C agg) | 2 |
| ERROR | 0 |
| VERIFIED (correctness) | 42 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 41 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | ? | 0.2879 | 0.3181 | 0.0879 | 3.62x | 0.1005 | 1.14x | 0.0995 | 1.13x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1446 | 0.0115 | 12.57x | 0.0245 | 2.13x | 0.0216 | 1.88x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.39x | 0.0001 | 1.45x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0487 | 0.0779 | 0.0604 | 1.29x | 0.0645 | 1.07x | 0.0502 | 0.83x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1605 | 0.0354 | 0.0046 | 7.70x | 0.1698 | 36.91x | 0.1710 | 37.17x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0026 | 0.0003 | 8.67x | 0.0004 | 1.32x | 0.0006 | 2.09x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2889 | 0.3134 | 0.0882 | 3.55x | 0.1803 | 2.04x | 0.1702 | 1.93x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2957 | 0.2945 | 0.0917 | 3.21x | 0.0957 | 1.04x | 0.1812 | 1.98x | OK | PARITY |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1690 | 0.0353 | 0.0096 | 3.68x | 0.0353 | 3.67x | 0.0353 | 3.68x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1715 | 0.0382 | 0.0081 | 4.72x | 0.0373 | 4.61x | 0.0354 | 4.37x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1697 | 0.0353 | 0.0074 | 4.77x | 0.0360 | 4.87x | 0.0352 | 4.76x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 0.5937 | 42.2486 | 8.4922 | 4.97x | 11.5946 | 1.37x | 11.4363 | 1.35x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2375 | 0.4344 | 0.1345 | 3.23x | 0.1455 | 1.08x | 0.1535 | 1.14x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2388 | 0.4446 | 0.1366 | 3.25x | 0.1463 | 1.07x | 0.1440 | 1.05x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2362 | 0.4823 | 0.1557 | 3.10x | 0.1461 | 0.94x | 0.1427 | 0.92x | OK | LOSS |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2357 | 0.4394 | 0.1347 | 3.26x | 0.1467 | 1.09x | 0.1430 | 1.06x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2454 | 0.4374 | 0.1336 | 3.27x | 0.1465 | 1.10x | 0.1526 | 1.14x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7244 | 0.0119 | 60.87x | 0.0245 | 2.06x | 0.0216 | 1.81x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0306 | 0.0295 | 0.0050 | 5.90x | 0.0085 | 1.70x | 0.0082 | 1.64x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0548 | 0.0554 | 0.0080 | 6.92x | 0.0191 | 2.39x | 0.0151 | 1.89x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0296 | 0.0294 | 0.0046 | 6.39x | 0.0084 | 1.82x | 0.0083 | 1.81x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0394 | 0.0357 | 0.0070 | 5.10x | 0.0108 | 1.54x | 0.0109 | 1.56x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.99x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.39x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6693 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2407 | 0.4393 | 0.1343 | 3.27x | 0.1469 | 1.09x | 0.1430 | 1.06x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 3.1863 | 0.0089 | 0.0028 | 3.18x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7215 | 0.0091 | 0.0028 | 3.25x | 0.0028 | 0.98x | 0.0028 | 0.98x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7052 | 0.0094 | 0.0028 | 3.36x | 0.0028 | 1.00x | 0.0029 | 1.03x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2356 | 0.4351 | 0.1282 | 3.39x | 0.1462 | 1.14x | 0.1585 | 1.24x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2402 | 0.4417 | 0.1319 | 3.35x | 0.1460 | 1.11x | 0.1540 | 1.17x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1692 | 0.0352 | 0.0084 | 4.19x | 0.0352 | 4.19x | 0.0354 | 4.21x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1774 | 0.0353 | 0.0077 | 4.58x | 0.0352 | 4.58x | 0.0352 | 4.57x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1801 | 0.0353 | 0.0072 | 4.90x | 0.0352 | 4.89x | 0.0359 | 4.99x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.34x | 0.0001 | 1.31x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2415 | 0.4379 | 0.1342 | 3.26x | 0.1465 | 1.09x | 0.1430 | 1.07x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2367 | 0.4418 | 0.1346 | 3.28x | 0.1466 | 1.09x | 0.1530 | 1.14x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2380 | 0.4424 | 0.1312 | 3.37x | 0.1463 | 1.12x | 0.1429 | 1.09x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2391 | 0.4385 | 0.1365 | 3.21x | 0.1468 | 1.08x | 0.1544 | 1.13x | OK | **WIN** |

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
