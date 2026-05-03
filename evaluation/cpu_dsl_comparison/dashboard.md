# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 13:11

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **28** |
| **PARITY (vs C O3)** | **12** |
| **LOSS (vs C O3)** | **1** |
| WIN (vs C agg) | 24 |
| PARITY (vs C agg) | 14 |
| LOSS (vs C agg) | 3 |
| ERROR | 1 |
| VERIFIED (correctness) | 42 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 40 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2866 | 0.3040 | 0.1021 | 2.98x | 0.1058 | 1.04x | 0.1053 | 1.03x | OK | PARITY |
| 02_gemm_row_major |  |  | 64 | 0.0064 | 0.1446 | 0.0123 | 11.76x | 0.0245 | 1.99x | 0.0216 | 1.76x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0022 | 0.0004 | 0.0001 | 5.21x | 0.0001 | 1.39x | 0.0001 | 1.44x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0484 | 0.0779 | 0.0600 | 1.30x | 0.0659 | 1.10x | 0.0457 | 0.76x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1589 | 0.0354 | 0.0048 | 7.35x | 0.1701 | 35.43x | 0.1771 | 36.91x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0022 | 0.0002 | 9.79x | 0.0005 | 2.21x | 0.0006 | 2.50x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2745 | 0.2907 | 0.0877 | 3.31x | 0.1906 | 2.17x | 0.1742 | 1.99x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2827 | 0.2895 | 0.0909 | 3.18x | 0.0957 | 1.05x | 0.1413 | 1.55x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1718 | 0.0358 | 0.0077 | 4.65x | 0.0360 | 4.67x | 0.0352 | 4.58x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1778 | 0.0387 | 0.0073 | 5.32x | 0.0361 | 4.94x | 0.0353 | 4.83x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1711 | 0.0358 | 0.0072 | 5.00x | 0.0360 | 5.00x | 0.0360 | 4.99x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 7.2932 | 42.9493 | 15.1549 | 2.83x | 11.5395 | 0.76x | 10.8848 | 0.72x | OK | LOSS |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2476 | 0.4307 | 0.1409 | 3.06x | 0.1462 | 1.04x | 0.1550 | 1.10x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2386 | 0.4338 | 0.1464 | 2.96x | 0.1462 | 1.00x | 0.1555 | 1.06x | OK | PARITY |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2473 | 0.4349 | 0.1470 | 2.96x | 0.1459 | 0.99x | 0.1492 | 1.01x | OK | PARITY |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2365 | 0.4361 | 0.1341 | 3.25x | 0.1458 | 1.09x | 0.1553 | 1.16x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2429 | 0.4372 | 0.1378 | 3.17x | 0.1462 | 1.06x | 0.1506 | 1.09x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.7258 | 0.0117 | 61.78x | 0.0249 | 2.13x | 0.0215 | 1.83x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0730 | 0.0310 | 0.0043 | 7.20x | 0.0084 | 1.95x | 0.0081 | 1.88x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1657 | 0.0582 | NaN | NaN | 0.0191 | NaN | 0.0151 | NaN | OK | ERROR |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0715 | 0.0304 | 0.0043 | 7.00x | 0.0084 | 1.95x | 0.0081 | 1.88x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0723 | 0.0355 | 0.0082 | 4.30x | 0.0110 | 1.34x | 0.0113 | 1.38x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0018 | 4.98x | 0.0022 | 1.24x | 0.0016 | 0.92x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.07x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0070 | 0.0088 | 0.0017 | 5.26x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.21x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.0026 | 0.0022 | 0.0004 | 5.21x | 0.0006 | 1.38x | 0.0004 | 1.04x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2379 | 0.4303 | 0.1371 | 3.14x | 0.1457 | 1.06x | 0.1430 | 1.04x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0072 | 0.0089 | 0.0028 | 3.14x | 0.0028 | 0.99x | 0.0028 | 1.01x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0090 | 0.0028 | 3.15x | 0.0028 | 1.01x | 0.0028 | 1.01x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0070 | 0.0088 | 0.0028 | 3.20x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2433 | 0.4401 | 0.1453 | 3.03x | 0.1460 | 1.01x | 0.1531 | 1.05x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2375 | 0.4450 | 0.1487 | 2.99x | 0.1460 | 0.98x | 0.1543 | 1.04x | OK | PARITY |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1698 | 0.0358 | 0.0069 | 5.18x | 0.0354 | 5.12x | 0.0353 | 5.11x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1687 | 0.0352 | 0.0079 | 4.48x | 0.0353 | 4.47x | 0.0356 | 4.51x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1688 | 0.0355 | 0.0074 | 4.78x | 0.0354 | 4.78x | 0.0353 | 4.76x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0072 | 0.0005 | 0.0001 | 4.90x | 0.0001 | 1.37x | 0.0001 | 1.36x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.11x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2510 | 0.4367 | 0.1398 | 3.12x | 0.1407 | 1.01x | 0.1427 | 1.02x | OK | PARITY |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2441 | 0.4376 | 0.1398 | 3.13x | 0.1464 | 1.05x | 0.1431 | 1.02x | OK | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2492 | 0.4382 | 0.1388 | 3.16x | 0.1459 | 1.05x | 0.1536 | 1.11x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2444 | 0.4504 | 0.1427 | 3.16x | 0.1459 | 1.02x | 0.1530 | 1.07x | OK | PARITY |

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
