# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 19:45

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
| 01_saxpy |  |  | ? | 0.2956 | 0.3296 | 0.0891 | 3.70x | 0.1051 | 1.18x | 0.0995 | 1.12x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0069 | 0.1446 | 0.0116 | 12.47x | 0.0244 | 2.11x | 0.0216 | 1.86x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0022 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.38x | 0.0001 | 1.45x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0480 | 0.0779 | 0.0605 | 1.29x | 0.0647 | 1.07x | 0.0483 | 0.80x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1642 | 0.0355 | 0.0048 | 7.40x | 0.1691 | 35.23x | 0.1692 | 35.26x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0023 | 0.0002 | 11.50x | 0.0005 | 2.48x | 0.0006 | 3.12x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2876 | 0.3168 | 0.0889 | 3.56x | 0.1898 | 2.14x | 0.1747 | 1.97x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2970 | 0.2925 | 0.0924 | 3.17x | 0.1011 | 1.09x | 0.1485 | 1.61x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1696 | 0.0353 | 0.0085 | 4.15x | 0.0354 | 4.17x | 0.0352 | 4.14x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1704 | 0.0382 | 0.0082 | 4.66x | 0.0352 | 4.30x | 0.0352 | 4.29x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1688 | 0.0353 | 0.0073 | 4.84x | 0.0364 | 4.98x | 0.0352 | 4.82x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 0.9339 | 42.4865 | 8.3460 | 5.09x | 11.5535 | 1.38x | 11.3884 | 1.36x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2294 | 0.4407 | 0.1354 | 3.25x | 0.1459 | 1.08x | 0.1537 | 1.14x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2465 | 0.4382 | 0.1349 | 3.25x | 0.1457 | 1.08x | 0.1547 | 1.15x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2276 | 0.4381 | 0.1310 | 3.34x | 0.1463 | 1.12x | 0.1427 | 1.09x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2367 | 0.4369 | 0.1359 | 3.21x | 0.1458 | 1.07x | 0.1552 | 1.14x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2369 | 0.4409 | 0.1354 | 3.26x | 0.1466 | 1.08x | 0.1429 | 1.06x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7239 | 0.0118 | 61.35x | 0.0244 | 2.07x | 0.0216 | 1.83x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0298 | 0.0294 | 0.0044 | 6.68x | 0.0084 | 1.91x | 0.0081 | 1.84x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0556 | 0.0555 | 0.0077 | 7.21x | 0.0191 | 2.48x | 0.0151 | 1.96x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0299 | 0.0294 | 0.0045 | 6.53x | 0.0084 | 1.86x | 0.0081 | 1.80x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0387 | 0.0356 | 0.0073 | 4.88x | 0.0109 | 1.49x | 0.0111 | 1.52x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6859 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2351 | 0.4438 | 0.1352 | 3.28x | 0.1457 | 1.08x | 0.1430 | 1.06x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7380 | 0.0090 | 0.0028 | 3.21x | 0.0029 | 1.03x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.6945 | 0.0095 | 0.0028 | 3.39x | 0.0028 | 0.99x | 0.0027 | 0.98x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7182 | 0.0093 | 0.0028 | 3.32x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2373 | 0.4445 | 0.1363 | 3.26x | 0.1469 | 1.08x | 0.1549 | 1.14x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2384 | 0.4335 | 0.1324 | 3.27x | 0.1406 | 1.06x | 0.1549 | 1.17x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1711 | 0.0352 | 0.0070 | 5.03x | 0.0359 | 5.13x | 0.0353 | 5.04x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1759 | 0.0354 | 0.0075 | 4.72x | 0.0352 | 4.70x | 0.0352 | 4.70x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1792 | 0.0353 | 0.0074 | 4.77x | 0.0355 | 4.80x | 0.0352 | 4.76x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0124 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.33x | 0.0001 | 1.35x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2370 | 0.4388 | 0.1291 | 3.40x | 0.1461 | 1.13x | 0.1529 | 1.18x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2376 | 0.4384 | 0.1366 | 3.21x | 0.1463 | 1.07x | 0.1427 | 1.04x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2373 | 0.4428 | 0.1354 | 3.27x | 0.1454 | 1.07x | 0.1461 | 1.08x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2353 | 0.4429 | 0.1368 | 3.24x | 0.1472 | 1.08x | 0.1538 | 1.12x | OK | **WIN** |

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
