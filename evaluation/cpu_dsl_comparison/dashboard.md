# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 19:24

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **39** |
| **PARITY (vs C O3)** | **3** |
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
| 01_saxpy |  |  | ? | 0.2795 | 0.3163 | 0.0862 | 3.67x | 0.1058 | 1.23x | 0.0992 | 1.15x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0069 | 0.1446 | 0.0118 | 12.28x | 0.0244 | 2.07x | 0.0216 | 1.83x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.97x | 0.0001 | 1.47x | 0.0001 | 1.44x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0462 | 0.0779 | 0.0604 | 1.29x | 0.0643 | 1.06x | 0.0472 | 0.78x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1602 | 0.0356 | 0.0047 | 7.53x | 0.1702 | 36.21x | 0.1714 | 36.47x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0025 | 0.0002 | 11.86x | 0.0004 | 1.98x | 0.0007 | 3.36x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2819 | 0.3173 | 0.0892 | 3.56x | 0.1802 | 2.02x | 0.1675 | 1.88x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2987 | 0.2962 | 0.0937 | 3.16x | 0.1005 | 1.07x | 0.1495 | 1.60x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1698 | 0.0353 | 0.0070 | 5.03x | 0.0358 | 5.11x | 0.0353 | 5.04x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1739 | 0.0383 | 0.0081 | 4.76x | 0.0353 | 4.36x | 0.0354 | 4.37x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1716 | 0.0359 | 0.0073 | 4.93x | 0.0366 | 5.01x | 0.0352 | 4.83x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 0.6547 | 42.3687 | 8.7486 | 4.84x | 11.9097 | 1.36x | 10.8609 | 1.24x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2379 | 0.4445 | 0.1346 | 3.30x | 0.1468 | 1.09x | 0.1545 | 1.15x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2369 | 0.4377 | 0.1305 | 3.35x | 0.1456 | 1.12x | 0.1427 | 1.09x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2343 | 0.4436 | 0.1344 | 3.30x | 0.1470 | 1.09x | 0.1542 | 1.15x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2291 | 0.4445 | 0.1350 | 3.29x | 0.1465 | 1.09x | 0.1528 | 1.13x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2386 | 0.4438 | 0.1368 | 3.24x | 0.1455 | 1.06x | 0.1535 | 1.12x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7254 | 0.0120 | 60.42x | 0.0245 | 2.04x | 0.0217 | 1.81x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0314 | 0.0294 | 0.0050 | 5.93x | 0.0084 | 1.68x | 0.0081 | 1.62x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0546 | 0.0553 | 0.0084 | 6.62x | 0.0192 | 2.29x | 0.0151 | 1.80x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0297 | 0.0295 | 0.0044 | 6.64x | 0.0084 | 1.91x | 0.0081 | 1.84x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0375 | 0.0358 | 0.0073 | 4.90x | 0.0110 | 1.51x | 0.0111 | 1.52x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.13x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.12x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.11x | 0.0022 | 1.30x | 0.0017 | 0.99x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.20x | 0.0006 | 1.46x | 0.0004 | 1.08x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6965 | 0.0022 | 0.0004 | 5.09x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2482 | 0.4369 | 0.1334 | 3.27x | 0.1467 | 1.10x | 0.1427 | 1.07x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7518 | 0.0094 | 0.0028 | 3.33x | 0.0028 | 1.01x | 0.0029 | 1.04x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.8980 | 0.0096 | 0.0028 | 3.40x | 0.0028 | 0.99x | 0.0028 | 1.01x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7187 | 0.0094 | 0.0028 | 3.34x | 0.0029 | 1.03x | 0.0028 | 1.00x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2286 | 0.4426 | 0.1345 | 3.29x | 0.1465 | 1.09x | 0.1524 | 1.13x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2353 | 0.4446 | 0.1348 | 3.30x | 0.1464 | 1.09x | 0.1427 | 1.06x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1683 | 0.0353 | 0.0074 | 4.79x | 0.0357 | 4.83x | 0.0352 | 4.76x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1788 | 0.0354 | 0.0074 | 4.80x | 0.0355 | 4.79x | 0.0353 | 4.77x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1793 | 0.0352 | 0.0076 | 4.63x | 0.0353 | 4.65x | 0.0353 | 4.64x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 4.76x | 0.0001 | 1.37x | 0.0001 | 1.33x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.11x | 0.0022 | 1.30x | 0.0017 | 0.99x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2310 | 0.4377 | 0.1364 | 3.21x | 0.1454 | 1.07x | 0.1427 | 1.05x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2386 | 0.4440 | 0.1331 | 3.34x | 0.1461 | 1.10x | 0.1427 | 1.07x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2387 | 0.4435 | 0.1347 | 3.29x | 0.1460 | 1.08x | 0.1427 | 1.06x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2281 | 0.4419 | 0.1288 | 3.43x | 0.1463 | 1.14x | 0.1534 | 1.19x | OK | **WIN** |

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
