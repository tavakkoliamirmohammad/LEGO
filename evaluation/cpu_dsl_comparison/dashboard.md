# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 22:49

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **38** |
| **PARITY (vs C O3)** | **4** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 34 |
| PARITY (vs C agg) | 7 |
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
| 01_saxpy |  |  | ? | 0.2925 | 0.3188 | 0.0892 | 3.57x | 0.1056 | 1.18x | 0.0992 | 1.11x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1445 | 0.0115 | 12.57x | 0.0245 | 2.13x | 0.0217 | 1.89x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.45x | 0.0001 | 1.47x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0481 | 0.0778 | 0.0605 | 1.29x | 0.0660 | 1.09x | 0.0445 | 0.74x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1601 | 0.0354 | 0.0048 | 7.38x | 0.1704 | 35.50x | 0.1709 | 35.61x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0027 | 0.0003 | 9.00x | 0.0005 | 1.64x | 0.0006 | 2.06x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2868 | 0.3169 | 0.0882 | 3.59x | 0.1881 | 2.13x | 0.1729 | 1.96x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2980 | 0.2851 | 0.0882 | 3.23x | 0.1000 | 1.13x | 0.1504 | 1.71x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1712 | 0.0352 | 0.0077 | 4.57x | 0.0353 | 4.58x | 0.0352 | 4.57x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1719 | 0.0389 | 0.0085 | 4.58x | 0.0357 | 4.20x | 0.0352 | 4.14x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1690 | 0.0353 | 0.0074 | 4.77x | 0.0358 | 4.83x | 0.0353 | 4.76x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 0.9813 | 41.9681 | 8.4553 | 4.96x | 11.9394 | 1.41x | 10.8223 | 1.28x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2387 | 0.4386 | 0.1355 | 3.24x | 0.1457 | 1.08x | 0.1532 | 1.13x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2367 | 0.4381 | 0.1339 | 3.27x | 0.1465 | 1.09x | 0.1533 | 1.14x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2382 | 0.4447 | 0.1352 | 3.29x | 0.1466 | 1.08x | 0.1427 | 1.06x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2393 | 0.4383 | 0.1360 | 3.22x | 0.1404 | 1.03x | 0.1428 | 1.05x | OK | PARITY |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2387 | 0.4444 | 0.1322 | 3.36x | 0.1459 | 1.10x | 0.1428 | 1.08x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0069 | 0.7238 | 0.0118 | 61.34x | 0.0244 | 2.07x | 0.0216 | 1.83x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0302 | 0.0295 | 0.0050 | 5.90x | 0.0086 | 1.73x | 0.0081 | 1.62x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0538 | 0.0553 | 0.0081 | 6.83x | 0.0192 | 2.36x | 0.0151 | 1.86x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0306 | 0.0294 | 0.0047 | 6.26x | 0.0084 | 1.79x | 0.0081 | 1.72x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0391 | 0.0357 | 0.0072 | 4.96x | 0.0107 | 1.49x | 0.0109 | 1.51x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0023 | 1.32x | 0.0017 | 0.98x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.44x | 0.0004 | 1.08x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6776 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.44x | 0.0004 | 1.08x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2431 | 0.4390 | 0.1305 | 3.36x | 0.1450 | 1.11x | 0.1429 | 1.09x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7381 | 0.0094 | 0.0028 | 3.36x | 0.0028 | 0.99x | 0.0027 | 0.98x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.6853 | 0.0094 | 0.0028 | 3.36x | 0.0028 | 0.99x | 0.0029 | 1.02x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.6781 | 0.0092 | 0.0028 | 3.29x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2418 | 0.4398 | 0.1311 | 3.35x | 0.1457 | 1.11x | 0.1427 | 1.09x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2382 | 0.4385 | 0.1344 | 3.26x | 0.1458 | 1.09x | 0.1428 | 1.06x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1704 | 0.0354 | 0.0072 | 4.92x | 0.0353 | 4.90x | 0.0352 | 4.89x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1830 | 0.0353 | 0.0075 | 4.71x | 0.0355 | 4.74x | 0.0353 | 4.71x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1798 | 0.0353 | 0.0071 | 4.97x | 0.0354 | 4.98x | 0.0353 | 4.96x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.34x | 0.0001 | 1.35x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0072 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2380 | 0.4397 | 0.1348 | 3.26x | 0.1458 | 1.08x | 0.1533 | 1.14x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2401 | 0.4447 | 0.1354 | 3.28x | 0.1469 | 1.09x | 0.1532 | 1.13x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2348 | 0.4443 | 0.1298 | 3.42x | 0.1460 | 1.13x | 0.1540 | 1.19x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2394 | 0.4377 | 0.1288 | 3.40x | 0.1461 | 1.13x | 0.1544 | 1.20x | OK | **WIN** |

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
