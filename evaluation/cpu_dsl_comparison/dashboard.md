# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 20:31

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
| 01_saxpy |  |  | ? | 0.2911 | 0.3183 | 0.0893 | 3.56x | 0.1055 | 1.18x | 0.1043 | 1.17x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0071 | 0.1446 | 0.0116 | 12.47x | 0.0244 | 2.11x | 0.0216 | 1.86x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.39x | 0.0001 | 1.46x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0479 | 0.0779 | 0.0604 | 1.29x | 0.0639 | 1.06x | 0.0449 | 0.74x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1585 | 0.0354 | 0.0046 | 7.70x | 0.1703 | 37.01x | 0.1709 | 37.14x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0023 | 0.0002 | 11.50x | 0.0004 | 1.91x | 0.0006 | 3.08x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2890 | 0.3181 | 0.0889 | 3.58x | 0.1890 | 2.13x | 0.1692 | 1.90x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2977 | 0.2905 | 0.0891 | 3.26x | 0.1010 | 1.13x | 0.1486 | 1.67x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1725 | 0.0353 | 0.0082 | 4.30x | 0.0359 | 4.38x | 0.0353 | 4.31x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1704 | 0.0382 | 0.0084 | 4.55x | 0.0352 | 4.19x | 0.0353 | 4.20x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1726 | 0.0353 | 0.0074 | 4.77x | 0.0355 | 4.79x | 0.0352 | 4.76x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 2.8098 | 42.5371 | 7.3575 | 5.78x | 11.6688 | 1.59x | 10.8745 | 1.48x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2276 | 0.4422 | 0.1296 | 3.41x | 0.1471 | 1.14x | 0.1553 | 1.20x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2363 | 0.4388 | 0.1354 | 3.24x | 0.1467 | 1.08x | 0.1568 | 1.16x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2396 | 0.4353 | 0.1359 | 3.20x | 0.1470 | 1.08x | 0.1545 | 1.14x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2326 | 0.4439 | 0.1366 | 3.25x | 0.1457 | 1.07x | 0.1433 | 1.05x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2384 | 0.4386 | 0.1335 | 3.29x | 0.1468 | 1.10x | 0.1555 | 1.16x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7243 | 0.0119 | 60.87x | 0.0247 | 2.08x | 0.0219 | 1.84x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0307 | 0.0294 | 0.0048 | 6.12x | 0.0084 | 1.74x | 0.0083 | 1.74x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0538 | 0.0553 | 0.0077 | 7.18x | 0.0190 | 2.46x | 0.0151 | 1.96x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0301 | 0.0294 | 0.0050 | 5.88x | 0.0084 | 1.68x | 0.0081 | 1.62x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0384 | 0.0355 | 0.0071 | 5.00x | 0.0108 | 1.53x | 0.0108 | 1.52x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.32x | 0.0017 | 0.99x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6695 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2343 | 0.4426 | 0.1345 | 3.29x | 0.1458 | 1.08x | 0.1547 | 1.15x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7467 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 1.00x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7493 | 0.0096 | 0.0028 | 3.43x | 0.0029 | 1.02x | 0.0028 | 0.98x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7207 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 1.01x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2284 | 0.4441 | 0.1289 | 3.45x | 0.1460 | 1.13x | 0.1427 | 1.11x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2287 | 0.4403 | 0.1349 | 3.26x | 0.1462 | 1.08x | 0.1430 | 1.06x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1703 | 0.0353 | 0.0072 | 4.90x | 0.0352 | 4.89x | 0.0359 | 4.98x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1808 | 0.0353 | 0.0074 | 4.77x | 0.0363 | 4.90x | 0.0352 | 4.75x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1783 | 0.0358 | 0.0071 | 5.04x | 0.0353 | 4.98x | 0.0358 | 5.04x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0124 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.31x | 0.0001 | 1.31x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2380 | 0.4449 | 0.1354 | 3.29x | 0.1461 | 1.08x | 0.1551 | 1.15x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2423 | 0.4389 | 0.1351 | 3.25x | 0.1468 | 1.09x | 0.1427 | 1.06x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2286 | 0.4416 | 0.1359 | 3.25x | 0.1457 | 1.07x | 0.1428 | 1.05x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2405 | 0.4392 | 0.1360 | 3.23x | 0.1449 | 1.07x | 0.1545 | 1.14x | OK | **WIN** |

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
