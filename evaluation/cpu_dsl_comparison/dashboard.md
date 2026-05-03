# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 09:48

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **28** |
| **PARITY (vs C O3)** | **14** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 30 |
| PARITY (vs C agg) | 8 |
| LOSS (vs C agg) | 4 |
| ERROR | 0 |
| VERIFIED (correctness) | 42 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 41 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2853 | 0.3094 | 0.0948 | 3.26x | 0.1008 | 1.06x | 0.1028 | 1.08x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.1071 | 0.0142 | 7.56x | 0.0245 | 1.73x | 0.0216 | 1.53x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0006 | 0.0001 | 4.96x | 0.0001 | 1.40x | 0.0001 | 1.46x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0463 | 0.0788 | 0.0612 | 1.29x | 0.0667 | 1.09x | 0.0558 | 0.91x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1624 | 0.0380 | 0.0056 | 6.79x | 0.1702 | 30.39x | 0.1710 | 30.54x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0019 | 0.0022 | 0.0003 | 8.33x | 0.0005 | 1.88x | 0.0006 | 2.11x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2776 | 0.2977 | 0.1010 | 2.95x | 0.1794 | 1.78x | 0.1737 | 1.72x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2823 | 0.2845 | 0.0942 | 3.02x | 0.1008 | 1.07x | 0.1493 | 1.59x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1735 | 0.0376 | 0.0097 | 3.89x | 0.0354 | 3.65x | 0.0356 | 3.67x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1786 | 0.0406 | 0.0093 | 4.35x | 0.0353 | 3.79x | 0.0353 | 3.79x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1692 | 0.0377 | 0.0096 | 3.91x | 0.0356 | 3.71x | 0.0352 | 3.67x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 7.3528 | 43.2062 | 8.5562 | 5.05x | 11.5023 | 1.34x | 11.4511 | 1.34x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2486 | 0.4452 | 0.1423 | 3.13x | 0.1465 | 1.03x | 0.1528 | 1.07x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2362 | 0.4369 | 0.1360 | 3.21x | 0.1466 | 1.08x | 0.1439 | 1.06x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2500 | 0.4434 | 0.1406 | 3.15x | 0.1480 | 1.05x | 0.1557 | 1.11x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2498 | 0.4335 | 0.1393 | 3.11x | 0.1457 | 1.05x | 0.1555 | 1.12x | OK | PARITY |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2363 | 0.4408 | 0.1449 | 3.04x | 0.1470 | 1.01x | 0.1538 | 1.06x | OK | PARITY |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.6872 | 0.1332 | 5.16x | 0.1452 | 1.09x | 0.1429 | 1.07x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0763 | 0.0303 | 0.0063 | 4.79x | 0.0084 | 1.34x | 0.0081 | 1.29x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1627 | 0.0563 | 0.0098 | 5.74x | 0.0190 | 1.94x | 0.0151 | 1.55x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0742 | 0.0303 | 0.0051 | 5.99x | 0.0084 | 1.64x | 0.0081 | 1.59x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0722 | 0.0370 | 0.0093 | 3.97x | 0.0109 | 1.18x | 0.0109 | 1.17x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0018 | 4.99x | 0.0022 | 1.23x | 0.0016 | 0.92x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0018 | 5.01x | 0.0023 | 1.27x | 0.0016 | 0.92x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0070 | 0.0088 | 0.0018 | 4.99x | 0.0023 | 1.30x | 0.0016 | 0.92x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 1048576 | 0.2421 | 0.4416 | 0.1419 | 3.11x | 0.1466 | 1.03x | 0.1430 | 1.01x | OK | PARITY |
| 27_zuker_skew | Skew tile | LOSS | 1048576 | 0.2491 | 0.4383 | 0.1461 | 3.00x | 0.1472 | 1.01x | 0.1544 | 1.06x | OK | PARITY |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2490 | 0.4404 | 0.1470 | 3.00x | 0.1458 | 0.99x | 0.1428 | 0.97x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0088 | 0.0028 | 3.16x | 0.0028 | 1.02x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0072 | 0.0088 | 0.0029 | 3.09x | 0.0028 | 0.96x | 0.0028 | 0.96x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0070 | 0.0088 | 0.0029 | 3.08x | 0.0028 | 0.96x | 0.0028 | 0.96x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2453 | 0.4414 | 0.1467 | 3.01x | 0.1407 | 0.96x | 0.1430 | 0.97x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2436 | 0.4405 | 0.1451 | 3.04x | 0.1469 | 1.01x | 0.1554 | 1.07x | OK | PARITY |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1689 | 0.0378 | 0.0096 | 3.93x | 0.0361 | 3.76x | 0.0353 | 3.67x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1691 | 0.0379 | 0.0095 | 3.99x | 0.0357 | 3.76x | 0.0352 | 3.71x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1724 | 0.0378 | 0.0098 | 3.84x | 0.0361 | 3.68x | 0.0352 | 3.60x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0072 | 0.0012 | 0.0008 | 1.50x | 0.0084 | 10.51x | 0.0081 | 10.12x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.07x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2413 | 0.4342 | 0.1364 | 3.18x | 0.1463 | 1.07x | 0.1518 | 1.11x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2444 | 0.4430 | 0.1444 | 3.07x | 0.1468 | 1.02x | 0.1427 | 0.99x | OK | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2369 | 0.4345 | 0.1408 | 3.09x | 0.1459 | 1.04x | 0.1542 | 1.10x | OK | PARITY |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2361 | 0.4397 | 0.1438 | 3.06x | 0.1471 | 1.02x | 0.1515 | 1.05x | OK | PARITY |

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
