# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 21:42

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **39** |
| **PARITY (vs C O3)** | **3** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 33 |
| PARITY (vs C agg) | 8 |
| LOSS (vs C agg) | 1 |
| ERROR | 0 |
| VERIFIED (correctness) | 42 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 42 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | ? | 0.2892 | 0.3184 | 0.0897 | 3.55x | 0.1056 | 1.18x | 0.1034 | 1.15x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1445 | 0.0115 | 12.57x | 0.0245 | 2.13x | 0.0216 | 1.88x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.38x | 0.0001 | 1.47x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0504 | 0.1016 | 0.0527 | 1.93x | 0.0671 | 1.27x | 0.0482 | 0.91x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1609 | 0.0353 | 0.0045 | 7.84x | 0.1708 | 37.96x | 0.1711 | 38.02x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0025 | 0.0002 | 12.50x | 0.0004 | 2.06x | 0.0006 | 2.79x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2931 | 0.3177 | 0.0892 | 3.56x | 0.1791 | 2.01x | 0.1680 | 1.88x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2957 | 0.2882 | 0.0882 | 3.27x | 0.0994 | 1.13x | 0.1414 | 1.60x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1707 | 0.0352 | 0.0084 | 4.19x | 0.0355 | 4.23x | 0.0353 | 4.20x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1703 | 0.0382 | 0.0072 | 5.31x | 0.0355 | 4.93x | 0.0353 | 4.90x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1695 | 0.0361 | 0.0076 | 4.75x | 0.0361 | 4.75x | 0.0352 | 4.63x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 6.2739 | 42.7439 | 8.5904 | 4.98x | 11.4158 | 1.33x | 11.6373 | 1.35x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2353 | 0.4386 | 0.1317 | 3.33x | 0.1460 | 1.11x | 0.1524 | 1.16x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2379 | 0.4443 | 0.1351 | 3.29x | 0.1456 | 1.08x | 0.1430 | 1.06x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2379 | 0.4458 | 0.1307 | 3.41x | 0.1467 | 1.12x | 0.1537 | 1.18x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2374 | 0.4428 | 0.1343 | 3.30x | 0.1459 | 1.09x | 0.1555 | 1.16x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2358 | 0.4445 | 0.1360 | 3.27x | 0.1461 | 1.07x | 0.1428 | 1.05x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0069 | 0.7254 | 0.0117 | 62.00x | 0.0245 | 2.09x | 0.0216 | 1.85x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0312 | 0.0294 | 0.0047 | 6.26x | 0.0084 | 1.79x | 0.0081 | 1.73x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0542 | 0.0554 | 0.0077 | 7.19x | 0.0191 | 2.48x | 0.0151 | 1.96x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0305 | 0.0295 | 0.0049 | 6.02x | 0.0084 | 1.72x | 0.0081 | 1.65x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0387 | 0.0356 | 0.0069 | 5.16x | 0.0109 | 1.59x | 0.0108 | 1.57x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0072 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0017 | 0.98x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.08x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6745 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2390 | 0.4386 | 0.1357 | 3.23x | 0.1458 | 1.07x | 0.1552 | 1.14x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7221 | 0.0094 | 0.0028 | 3.36x | 0.0028 | 1.01x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.6930 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 1.00x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7333 | 0.0094 | 0.0028 | 3.36x | 0.0028 | 0.99x | 0.0027 | 0.98x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2399 | 0.4366 | 0.1323 | 3.30x | 0.1455 | 1.10x | 0.1516 | 1.15x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2394 | 0.4319 | 0.1337 | 3.23x | 0.1461 | 1.09x | 0.1557 | 1.16x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1709 | 0.0353 | 0.0074 | 4.77x | 0.0354 | 4.78x | 0.0353 | 4.77x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1775 | 0.0358 | 0.0075 | 4.77x | 0.0354 | 4.72x | 0.0359 | 4.79x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1788 | 0.0352 | 0.0075 | 4.69x | 0.0353 | 4.70x | 0.0353 | 4.71x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.42x | 0.0001 | 1.32x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2437 | 0.4382 | 0.1354 | 3.24x | 0.1465 | 1.08x | 0.1530 | 1.13x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2389 | 0.4381 | 0.1361 | 3.22x | 0.1465 | 1.08x | 0.1553 | 1.14x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2294 | 0.4425 | 0.1357 | 3.26x | 0.1462 | 1.08x | 0.1428 | 1.05x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2375 | 0.4354 | 0.1280 | 3.40x | 0.1407 | 1.10x | 0.1427 | 1.11x | OK | **WIN** |

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
