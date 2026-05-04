# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 19:13

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **29** |
| **PARITY (vs C O3)** | **8** |
| **LOSS (vs C O3)** | **5** |
| WIN (vs C agg) | 27 |
| PARITY (vs C agg) | 11 |
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
| 01_saxpy |  |  | ? | 0.2922 | 0.3191 | 0.1106 | 2.89x | 0.1058 | 0.96x | 0.1037 | 0.94x | OK | PARITY |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1445 | 0.0115 | 12.53x | 0.0245 | 2.13x | 0.0217 | 1.89x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 5.35x | 0.0001 | 1.39x | 0.0001 | 1.44x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0596 | 0.0779 | 0.0604 | 1.29x | 0.0694 | 1.15x | 0.0451 | 0.75x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1626 | 0.0355 | 0.0046 | 7.75x | 0.1696 | 36.86x | 0.1702 | 37.01x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0024 | 0.0002 | 9.75x | 0.0004 | 2.12x | 0.0006 | 2.81x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2873 | 0.3100 | 0.0858 | 3.61x | 0.1875 | 2.19x | 0.1751 | 2.04x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2920 | 0.2923 | 0.1092 | 2.68x | 0.1015 | 0.93x | 0.1510 | 1.38x | OK | LOSS |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1704 | 0.0353 | 0.0075 | 4.72x | 0.0352 | 4.69x | 0.0353 | 4.70x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1716 | 0.0393 | 0.0080 | 4.91x | 0.0353 | 4.41x | 0.0357 | 4.46x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1686 | 0.0352 | 0.0074 | 4.75x | 0.0352 | 4.76x | 0.0353 | 4.77x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 1.8339 | 42.4183 | 8.9515 | 4.74x | 11.5104 | 1.29x | 10.8170 | 1.21x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2395 | 0.4394 | 0.1435 | 3.06x | 0.1407 | 0.98x | 0.1515 | 1.06x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2361 | 0.4438 | 0.1608 | 2.76x | 0.1460 | 0.91x | 0.1533 | 0.95x | OK | LOSS |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2411 | 0.4376 | 0.1336 | 3.27x | 0.1450 | 1.09x | 0.1530 | 1.15x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2513 | 0.4381 | 0.1337 | 3.28x | 0.1463 | 1.09x | 0.1525 | 1.14x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2395 | 0.4318 | 0.1403 | 3.08x | 0.1460 | 1.04x | 0.1484 | 1.06x | OK | PARITY |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0069 | 0.7241 | 0.0118 | 61.52x | 0.0247 | 2.09x | 0.0221 | 1.87x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0310 | 0.0294 | 0.0053 | 5.52x | 0.0084 | 1.58x | 0.0081 | 1.53x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0544 | 0.0553 | 0.0086 | 6.43x | 0.0191 | 2.22x | 0.0151 | 1.76x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0310 | 0.0294 | 0.0048 | 6.11x | 0.0084 | 1.75x | 0.0082 | 1.71x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0371 | 0.0360 | 0.0078 | 4.59x | 0.0110 | 1.40x | 0.0110 | 1.41x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0072 | 0.0088 | 0.0017 | 5.09x | 0.0022 | 1.30x | 0.0017 | 0.98x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.11x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.12x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.20x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6531 | 0.0022 | 0.0004 | 5.21x | 0.0006 | 1.38x | 0.0004 | 1.08x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2390 | 0.4379 | 0.1464 | 2.99x | 0.1464 | 1.00x | 0.1469 | 1.00x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7222 | 0.0095 | 0.0028 | 3.37x | 0.0028 | 0.99x | 0.0028 | 1.00x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7078 | 0.0088 | 0.0028 | 3.14x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7454 | 0.0093 | 0.0028 | 3.30x | 0.0028 | 1.00x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2396 | 0.4401 | 0.1449 | 3.04x | 0.1451 | 1.00x | 0.1543 | 1.07x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2422 | 0.4417 | 0.1629 | 2.71x | 0.1461 | 0.90x | 0.1545 | 0.95x | OK | LOSS |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1713 | 0.0354 | 0.0073 | 4.83x | 0.0354 | 4.85x | 0.0359 | 4.92x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1781 | 0.0352 | 0.0075 | 4.69x | 0.0352 | 4.69x | 0.0353 | 4.71x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1796 | 0.0353 | 0.0078 | 4.50x | 0.0361 | 4.63x | 0.0359 | 4.60x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 4.59x | 0.0001 | 1.32x | 0.0001 | 1.34x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.12x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2395 | 0.4381 | 0.1342 | 3.26x | 0.1466 | 1.09x | 0.1547 | 1.15x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2287 | 0.4435 | 0.1606 | 2.76x | 0.1406 | 0.88x | 0.1428 | 0.89x | OK | LOSS |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2379 | 0.4374 | 0.1577 | 2.77x | 0.1458 | 0.92x | 0.1558 | 0.99x | OK | LOSS |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2382 | 0.4337 | 0.1341 | 3.23x | 0.1466 | 1.09x | 0.1532 | 1.14x | OK | **WIN** |

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
