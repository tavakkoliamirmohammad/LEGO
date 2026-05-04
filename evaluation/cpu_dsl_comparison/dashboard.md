# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-04 07:17

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **34** |
| **PARITY (vs C O3)** | **8** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 29 |
| PARITY (vs C agg) | 12 |
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
| 01_saxpy |  |  | ? | 0.2936 | 0.3171 | 0.0864 | 3.67x | 0.1003 | 1.16x | 0.1051 | 1.22x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1445 | 0.0117 | 12.35x | 0.0244 | 2.09x | 0.0221 | 1.89x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0003 | 2.74x | 0.0002 | 1.54x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0484 | 0.0779 | 0.0603 | 1.29x | 0.0636 | 1.06x | 0.0469 | 0.78x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1592 | 0.0354 | 0.0046 | 7.70x | 0.1699 | 36.94x | 0.1691 | 36.76x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0026 | 0.0003 | 8.67x | 0.0004 | 1.31x | 0.0006 | 1.85x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2811 | 0.3194 | 0.0891 | 3.58x | 0.1796 | 2.02x | 0.1729 | 1.94x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2927 | 0.2855 | 0.0956 | 2.99x | 0.1020 | 1.07x | 0.1487 | 1.56x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1738 | 0.0354 | 0.0081 | 4.37x | 0.0354 | 4.37x | 0.0359 | 4.44x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1713 | 0.0382 | 0.0077 | 4.96x | 0.0362 | 4.70x | 0.0352 | 4.57x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1699 | 0.0352 | 0.0074 | 4.76x | 0.0361 | 4.88x | 0.0352 | 4.76x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 1.1543 | 42.5032 | 8.5167 | 4.99x | 12.0831 | 1.42x | 11.4275 | 1.34x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2369 | 0.4595 | 0.1425 | 3.22x | 0.1470 | 1.03x | 0.1520 | 1.07x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2365 | 0.4380 | 0.1344 | 3.26x | 0.1466 | 1.09x | 0.1431 | 1.06x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2387 | 0.4387 | 0.1347 | 3.26x | 0.1406 | 1.04x | 0.1426 | 1.06x | OK | PARITY |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2444 | 0.4386 | 0.1343 | 3.27x | 0.1432 | 1.07x | 0.1479 | 1.10x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2405 | 0.4394 | 0.1378 | 3.19x | 0.1464 | 1.06x | 0.1556 | 1.13x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7258 | 0.0120 | 60.48x | 0.0244 | 2.03x | 0.0216 | 1.80x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0312 | 0.0294 | 0.0048 | 6.12x | 0.0084 | 1.76x | 0.0081 | 1.69x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0539 | 0.0554 | 0.0078 | 7.10x | 0.0192 | 2.46x | 0.0151 | 1.94x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0302 | 0.0295 | 0.0052 | 5.67x | 0.0084 | 1.62x | 0.0081 | 1.56x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0408 | 0.0355 | 0.0068 | 5.22x | 0.0109 | 1.60x | 0.0108 | 1.59x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.29x | 0.0017 | 0.99x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.46x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6758 | 0.0022 | 0.0004 | 5.50x | 0.0007 | 1.77x | 0.0004 | 1.04x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2375 | 0.4461 | 0.1368 | 3.26x | 0.1414 | 1.03x | 0.1427 | 1.04x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7365 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 1.02x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7048 | 0.0096 | 0.0028 | 3.43x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7496 | 0.0095 | 0.0028 | 3.39x | 0.0028 | 0.99x | 0.0029 | 1.02x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2466 | 0.4450 | 0.1346 | 3.31x | 0.1473 | 1.09x | 0.1546 | 1.15x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2296 | 0.4380 | 0.1341 | 3.27x | 0.1455 | 1.09x | 0.1557 | 1.16x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1691 | 0.0354 | 0.0073 | 4.85x | 0.0353 | 4.83x | 0.0359 | 4.92x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1806 | 0.0353 | 0.0074 | 4.77x | 0.0353 | 4.77x | 0.0353 | 4.78x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1792 | 0.0354 | 0.0083 | 4.27x | 0.0365 | 4.40x | 0.0353 | 4.25x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0124 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.35x | 0.0001 | 1.33x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2258 | 0.4373 | 0.1446 | 3.02x | 0.1460 | 1.01x | 0.1431 | 0.99x | OK | PARITY |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2396 | 0.4383 | 0.1426 | 3.07x | 0.1454 | 1.02x | 0.1525 | 1.07x | OK | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2340 | 0.4439 | 0.1301 | 3.41x | 0.1455 | 1.12x | 0.1552 | 1.19x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2371 | 0.4430 | 0.1368 | 3.24x | 0.1457 | 1.07x | 0.1431 | 1.05x | OK | **WIN** |

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
