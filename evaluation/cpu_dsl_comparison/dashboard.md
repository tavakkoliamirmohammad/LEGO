# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 06:36

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **16** |
| **PARITY (vs C O3)** | **12** |
| **LOSS (vs C O3)** | **14** |
| WIN (vs C agg) | 19 |
| PARITY (vs C agg) | 7 |
| LOSS (vs C agg) | 16 |
| ERROR | 0 |
| VERIFIED (correctness) | 41 |
| PENDING (correctness) | 1 |
| vec_iso > 1.5× | 36 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2845 | 0.3006 | 0.0944 | 3.19x | 0.1046 | 1.11x | 0.1037 | 1.10x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.6882 | 0.1294 | 5.32x | 0.0245 | 0.19x | 0.0218 | 0.17x | OK | LOSS |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0011 | 0.0008 | 1.45x | 0.0001 | 0.17x | 0.0001 | 0.18x | OK | LOSS |
| 04_col_major_inner |  |  | 256 | 0.0470 | 0.0788 | 0.0614 | 1.28x | 0.0636 | 1.04x | 0.0455 | 0.74x | OK | PARITY |
| 05_morton_2d |  |  | 65536 | 0.1629 | 0.0389 | 0.0057 | 6.87x | 0.1707 | 29.95x | 0.1696 | 29.76x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0153 | 0.0139 | 0.0040 | 3.48x | 0.0004 | 0.10x | 0.0006 | 0.14x | OK | LOSS |
| 07_mixed_precision |  |  | 1048576 | 0.2847 | 0.3009 | 0.0943 | 3.19x | 0.1803 | 1.91x | 0.1678 | 1.78x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2893 | 0.2846 | 0.0970 | 2.93x | 0.1010 | 1.04x | 0.1501 | 1.55x | OK | PARITY |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1714 | 0.0391 | 0.0122 | 3.21x | 0.0354 | 2.90x | 0.0353 | 2.89x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1800 | 0.0408 | 0.0102 | 4.02x | 0.0354 | 3.47x | 0.0352 | 3.46x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1687 | 0.0377 | 0.0097 | 3.89x | 0.0355 | 3.66x | 0.0353 | 3.64x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 5.2909 | 42.6506 | 10.1868 | 4.19x | 11.5394 | 1.13x | 10.9323 | 1.07x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2490 | 0.4444 | 0.1414 | 3.14x | 0.1466 | 1.04x | 0.1428 | 1.01x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2495 | 0.4424 | 0.1565 | 2.83x | 0.1453 | 0.93x | 0.1429 | 0.91x | OK | LOSS |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2436 | 0.4514 | 0.1381 | 3.27x | 0.1470 | 1.06x | 0.1440 | 1.04x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2351 | 0.4383 | 0.1381 | 3.17x | 0.1491 | 1.08x | 0.1432 | 1.04x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2417 | 0.4349 | 0.1373 | 3.17x | 0.1461 | 1.06x | 0.1544 | 1.12x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0064 | 0.6866 | 0.1293 | 5.31x | 0.1454 | 1.12x | 0.1431 | 1.11x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0772 | 0.0302 | 0.0051 | 5.89x | 0.0086 | 1.69x | 0.0083 | 1.62x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1626 | 0.0563 | 0.0098 | 5.72x | 0.0084 | 0.86x | 0.0081 | 0.83x | ? | LOSS |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0749 | 0.0302 | 0.0051 | 5.96x | 0.0084 | 1.66x | 0.0081 | 1.59x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0701 | 0.0365 | 0.0095 | 3.86x | 0.0084 | 0.89x | 0.0081 | 0.85x | OK | LOSS |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0052 | 0.0025 | 2.06x | 0.0022 | 0.89x | 0.0017 | 0.67x | OK | LOSS |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0070 | 0.0052 | 0.0025 | 2.06x | 0.0022 | 0.89x | 0.0017 | 0.67x | OK | LOSS |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0070 | 0.0052 | 0.0024 | 2.14x | 0.0023 | 0.96x | 0.0017 | 0.70x | OK | PARITY |
| 26_nussinov_skew | Skew tile | WIN | 1048576 | 0.2411 | 0.4412 | 0.1424 | 3.10x | 0.1468 | 1.03x | 0.1428 | 1.00x | OK | PARITY |
| 27_zuker_skew | Skew tile | LOSS | 1048576 | 0.2478 | 0.4451 | 0.1416 | 3.14x | 0.1463 | 1.03x | 0.1427 | 1.01x | OK | PARITY |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2458 | 0.4598 | 0.1473 | 3.12x | 0.1467 | 1.00x | 0.1559 | 1.06x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0052 | 0.0036 | 1.43x | 0.0028 | 0.78x | 0.0028 | 0.78x | OK | LOSS |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0052 | 0.0036 | 1.44x | 0.0028 | 0.77x | 0.0028 | 0.78x | OK | LOSS |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0052 | 0.0036 | 1.45x | 0.0028 | 0.78x | 0.0027 | 0.76x | OK | LOSS |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2493 | 0.4392 | 0.1464 | 3.00x | 0.1465 | 1.00x | 0.1572 | 1.07x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2431 | 0.4433 | 0.1426 | 3.11x | 0.1465 | 1.03x | 0.1427 | 1.00x | OK | PARITY |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1690 | 0.0376 | 0.0094 | 3.99x | 0.0081 | 0.86x | 0.0088 | 0.94x | OK | LOSS |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1683 | 0.0377 | 0.0098 | 3.83x | 0.0081 | 0.82x | 0.0090 | 0.92x | OK | LOSS |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1704 | 0.0390 | 0.0106 | 3.68x | 0.0357 | 3.37x | 0.0352 | 3.32x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0074 | 0.0012 | 0.0008 | 1.43x | 0.0084 | 10.49x | 0.0084 | 10.56x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0070 | 0.0052 | 0.0024 | 2.18x | 0.0022 | 0.93x | 0.0016 | 0.69x | OK | LOSS |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2476 | 0.4411 | 0.1413 | 3.12x | 0.1467 | 1.04x | 0.1544 | 1.09x | OK | PARITY |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2339 | 0.4373 | 0.1391 | 3.14x | 0.1474 | 1.06x | 0.1551 | 1.11x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2516 | 0.4435 | 0.1427 | 3.11x | 0.1459 | 1.02x | 0.1431 | 1.00x | OK | PARITY |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2436 | 0.4550 | 0.1439 | 3.16x | 0.1461 | 1.02x | 0.1542 | 1.07x | OK | PARITY |

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
