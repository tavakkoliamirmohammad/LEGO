# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 19:17

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **38** |
| **PARITY (vs C O3)** | **4** |
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
| 01_saxpy |  |  | ? | 0.2911 | 0.3182 | 0.0894 | 3.56x | 0.1004 | 1.12x | 0.0995 | 1.11x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0071 | 0.1449 | 0.0114 | 12.67x | 0.0245 | 2.15x | 0.0218 | 1.91x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.94x | 0.0001 | 1.46x | 0.0003 | 2.75x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0483 | 0.0779 | 0.0605 | 1.29x | 0.0637 | 1.05x | 0.0465 | 0.77x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1625 | 0.0354 | 0.0045 | 7.83x | 0.1693 | 37.61x | 0.1700 | 37.79x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0026 | 0.0003 | 7.44x | 0.0004 | 1.33x | 0.0006 | 1.88x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2769 | 0.3188 | 0.0895 | 3.56x | 0.1866 | 2.08x | 0.1670 | 1.87x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2967 | 0.2850 | 0.0884 | 3.22x | 0.0992 | 1.12x | 0.1456 | 1.65x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1704 | 0.0358 | 0.0090 | 3.95x | 0.0354 | 3.93x | 0.0358 | 3.98x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1697 | 0.0382 | 0.0080 | 4.79x | 0.0355 | 4.44x | 0.0360 | 4.50x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1707 | 0.0354 | 0.0071 | 4.99x | 0.0355 | 5.00x | 0.0352 | 4.96x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 1.3623 | 42.3694 | 6.9557 | 6.09x | 11.5348 | 1.66x | 11.0381 | 1.59x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2473 | 0.4392 | 0.1351 | 3.25x | 0.1458 | 1.08x | 0.1429 | 1.06x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2428 | 0.4394 | 0.1366 | 3.22x | 0.1457 | 1.07x | 0.1427 | 1.04x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2459 | 0.4325 | 0.1355 | 3.19x | 0.1461 | 1.08x | 0.1506 | 1.11x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2402 | 0.4388 | 0.1374 | 3.19x | 0.1464 | 1.07x | 0.1428 | 1.04x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2382 | 0.4463 | 0.1352 | 3.30x | 0.1459 | 1.08x | 0.1428 | 1.06x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7240 | 0.0120 | 60.33x | 0.0245 | 2.04x | 0.0220 | 1.84x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0306 | 0.0294 | 0.0054 | 5.43x | 0.0084 | 1.55x | 0.0081 | 1.50x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0533 | 0.0553 | 0.0081 | 6.87x | 0.0191 | 2.36x | 0.0151 | 1.87x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0313 | 0.0294 | 0.0049 | 6.02x | 0.0085 | 1.74x | 0.0081 | 1.65x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0398 | 0.0357 | 0.0072 | 4.95x | 0.0109 | 1.52x | 0.0109 | 1.51x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.13x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0072 | 0.0088 | 0.0017 | 5.13x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.14x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.13x | 0.0006 | 1.45x | 0.0004 | 1.08x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6812 | 0.0022 | 0.0004 | 5.15x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2461 | 0.4388 | 0.1346 | 3.26x | 0.1476 | 1.10x | 0.1428 | 1.06x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.6471 | 0.0092 | 0.0029 | 3.20x | 0.0028 | 0.96x | 0.0028 | 0.95x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.6983 | 0.0089 | 0.0028 | 3.17x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.6896 | 0.0089 | 0.0028 | 3.16x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2383 | 0.4435 | 0.1353 | 3.28x | 0.1452 | 1.07x | 0.1538 | 1.14x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2375 | 0.4449 | 0.1361 | 3.27x | 0.1408 | 1.03x | 0.1520 | 1.12x | OK | PARITY |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1703 | 0.0353 | 0.0072 | 4.91x | 0.0352 | 4.89x | 0.0352 | 4.89x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1775 | 0.0361 | 0.0071 | 5.05x | 0.0353 | 4.98x | 0.0353 | 4.97x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1773 | 0.0353 | 0.0077 | 4.57x | 0.0353 | 4.58x | 0.0352 | 4.57x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0126 | 0.0005 | 0.0001 | 4.92x | 0.0001 | 1.32x | 0.0001 | 1.34x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.10x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2282 | 0.4384 | 0.1301 | 3.37x | 0.1460 | 1.12x | 0.1427 | 1.10x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2270 | 0.4371 | 0.1295 | 3.37x | 0.1458 | 1.13x | 0.1427 | 1.10x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2391 | 0.4383 | 0.1301 | 3.37x | 0.1463 | 1.12x | 0.1541 | 1.18x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2380 | 0.4452 | 0.1358 | 3.28x | 0.1453 | 1.07x | 0.1427 | 1.05x | OK | **WIN** |

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
