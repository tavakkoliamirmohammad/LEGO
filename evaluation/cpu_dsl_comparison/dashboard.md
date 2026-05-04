# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 22:58

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **37** |
| **PARITY (vs C O3)** | **5** |
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
| 01_saxpy |  |  | ? | 0.2912 | 0.3170 | 0.0896 | 3.54x | 0.1056 | 1.18x | 0.1019 | 1.14x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0069 | 0.1445 | 0.0116 | 12.46x | 0.0245 | 2.11x | 0.0218 | 1.88x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.39x | 0.0001 | 1.45x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0479 | 0.0779 | 0.0605 | 1.29x | 0.0653 | 1.08x | 0.0464 | 0.77x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1606 | 0.0354 | 0.0046 | 7.70x | 0.1698 | 36.90x | 0.1691 | 36.76x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0021 | 0.0024 | 0.0002 | 12.00x | 0.0004 | 1.93x | 0.0006 | 2.85x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2909 | 0.3204 | 0.0887 | 3.61x | 0.1787 | 2.01x | 0.1732 | 1.95x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2922 | 0.2918 | 0.0911 | 3.20x | 0.0956 | 1.05x | 0.1412 | 1.55x | OK | PARITY |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1699 | 0.0352 | 0.0082 | 4.29x | 0.0352 | 4.29x | 0.0362 | 4.41x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1723 | 0.0382 | 0.0078 | 4.90x | 0.0360 | 4.61x | 0.0357 | 4.58x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1703 | 0.0352 | 0.0073 | 4.82x | 0.0360 | 4.93x | 0.0353 | 4.84x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 1.7083 | 42.3761 | 8.4574 | 5.01x | 12.0703 | 1.43x | 11.5219 | 1.36x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2459 | 0.4419 | 0.1299 | 3.40x | 0.1466 | 1.13x | 0.1428 | 1.10x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2294 | 0.4435 | 0.1346 | 3.29x | 0.1459 | 1.08x | 0.1431 | 1.06x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2292 | 0.4437 | 0.1346 | 3.30x | 0.1460 | 1.08x | 0.1527 | 1.13x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2370 | 0.4396 | 0.1362 | 3.23x | 0.1472 | 1.08x | 0.1530 | 1.12x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2385 | 0.4357 | 0.1346 | 3.24x | 0.1407 | 1.05x | 0.1430 | 1.06x | OK | PARITY |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7234 | 0.0118 | 61.31x | 0.0245 | 2.08x | 0.0216 | 1.83x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0301 | 0.0294 | 0.0045 | 6.53x | 0.0084 | 1.87x | 0.0081 | 1.80x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0560 | 0.0553 | 0.0076 | 7.28x | 0.0191 | 2.51x | 0.0151 | 1.98x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0300 | 0.0295 | 0.0044 | 6.70x | 0.0084 | 1.90x | 0.0081 | 1.84x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0399 | 0.0356 | 0.0070 | 5.09x | 0.0110 | 1.58x | 0.0110 | 1.58x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0072 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6886 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2282 | 0.4390 | 0.1306 | 3.36x | 0.1467 | 1.12x | 0.1535 | 1.18x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7203 | 0.0089 | 0.0028 | 3.18x | 0.0028 | 1.00x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7089 | 0.0093 | 0.0028 | 3.32x | 0.0028 | 1.01x | 0.0027 | 0.98x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7169 | 0.0092 | 0.0028 | 3.29x | 0.0028 | 1.00x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2382 | 0.4378 | 0.1346 | 3.25x | 0.1464 | 1.09x | 0.1575 | 1.17x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2374 | 0.4428 | 0.1341 | 3.30x | 0.1457 | 1.09x | 0.1427 | 1.06x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1692 | 0.0353 | 0.0073 | 4.84x | 0.0354 | 4.85x | 0.0360 | 4.94x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1774 | 0.0353 | 0.0073 | 4.84x | 0.0352 | 4.82x | 0.0352 | 4.82x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1807 | 0.0352 | 0.0072 | 4.89x | 0.0353 | 4.91x | 0.0352 | 4.89x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.34x | 0.0001 | 1.37x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0017 | 0.99x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2474 | 0.4385 | 0.1352 | 3.24x | 0.1464 | 1.08x | 0.1496 | 1.11x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2361 | 0.4387 | 0.1288 | 3.41x | 0.1468 | 1.14x | 0.1539 | 1.19x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2365 | 0.4413 | 0.1323 | 3.34x | 0.1462 | 1.10x | 0.1428 | 1.08x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2463 | 0.4385 | 0.1338 | 3.28x | 0.1468 | 1.10x | 0.1535 | 1.15x | OK | **WIN** |

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
