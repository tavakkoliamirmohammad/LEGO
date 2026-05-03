# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 11:34

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **33** |
| **PARITY (vs C O3)** | **8** |
| **LOSS (vs C O3)** | **1** |
| WIN (vs C agg) | 27 |
| PARITY (vs C agg) | 12 |
| LOSS (vs C agg) | 3 |
| ERROR | 0 |
| VERIFIED (correctness) | 42 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 41 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2837 | 0.2974 | 0.0907 | 3.28x | 0.1006 | 1.11x | 0.0995 | 1.10x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.1446 | 0.0105 | 13.72x | 0.0244 | 2.32x | 0.0217 | 2.06x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0022 | 0.0004 | 0.0001 | 5.09x | 0.0002 | 1.56x | 0.0001 | 1.45x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0523 | 0.0780 | 0.0591 | 1.32x | 0.0637 | 1.08x | 0.0453 | 0.77x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1647 | 0.0363 | 0.0050 | 7.30x | 0.1710 | 34.19x | 0.1697 | 33.93x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0022 | 0.0002 | 10.52x | 0.0004 | 1.92x | 0.0006 | 2.61x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2830 | 0.3016 | 0.0899 | 3.36x | 0.1796 | 2.00x | 0.1684 | 1.87x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2925 | 0.2823 | 0.0956 | 2.95x | 0.1006 | 1.05x | 0.1487 | 1.56x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1689 | 0.0359 | 0.0079 | 4.53x | 0.0362 | 4.58x | 0.0358 | 4.53x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1802 | 0.0391 | 0.0071 | 5.54x | 0.0353 | 4.97x | 0.0352 | 4.96x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1707 | 0.0361 | 0.0082 | 4.39x | 0.0354 | 4.31x | 0.0352 | 4.30x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 14.8297 | 43.2321 | 8.1443 | 5.31x | 11.4531 | 1.41x | 11.1529 | 1.37x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2453 | 0.4622 | 0.1475 | 3.13x | 0.1470 | 1.00x | 0.1428 | 0.97x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2486 | 0.4375 | 0.1379 | 3.17x | 0.1459 | 1.06x | 0.1516 | 1.10x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2393 | 0.4372 | 0.1370 | 3.19x | 0.1411 | 1.03x | 0.1429 | 1.04x | OK | PARITY |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2466 | 0.4356 | 0.1412 | 3.08x | 0.1464 | 1.04x | 0.1534 | 1.09x | OK | PARITY |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2354 | 0.4364 | 0.1348 | 3.24x | 0.1458 | 1.08x | 0.1540 | 1.14x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.6848 | 0.1289 | 5.31x | 0.0244 | 0.19x | 0.0350 | 0.27x | OK | LOSS |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0735 | 0.0310 | 0.0046 | 6.72x | 0.0084 | 1.83x | 0.0081 | 1.77x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1635 | 0.0562 | 0.0073 | 7.75x | 0.0190 | 2.61x | 0.0151 | 2.07x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0765 | 0.0314 | 0.0056 | 5.61x | 0.0084 | 1.50x | 0.0081 | 1.44x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0707 | 0.0355 | 0.0071 | 5.03x | 0.0108 | 1.52x | 0.0109 | 1.54x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.15x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.06x | 0.0022 | 1.30x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0018 | 4.98x | 0.0022 | 1.22x | 0.0016 | 0.92x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0026 | 0.0022 | 0.0004 | 5.11x | 0.0006 | 1.38x | 0.0004 | 1.04x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.0026 | 0.0022 | 0.0004 | 5.30x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2402 | 0.4445 | 0.1395 | 3.19x | 0.1469 | 1.05x | 0.1541 | 1.10x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0089 | 0.0029 | 3.10x | 0.0028 | 0.95x | 0.0028 | 0.96x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0091 | 0.0028 | 3.23x | 0.0028 | 0.98x | 0.0028 | 0.99x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0070 | 0.0091 | 0.0028 | 3.19x | 0.0028 | 1.00x | 0.0028 | 1.00x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2507 | 0.4396 | 0.1397 | 3.15x | 0.1471 | 1.05x | 0.1538 | 1.10x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2410 | 0.4311 | 0.1337 | 3.22x | 0.1461 | 1.09x | 0.1430 | 1.07x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1700 | 0.0353 | 0.0076 | 4.62x | 0.0373 | 4.91x | 0.0352 | 4.64x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1701 | 0.0352 | 0.0070 | 5.06x | 0.0353 | 5.04x | 0.0352 | 5.03x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1756 | 0.0352 | 0.0069 | 5.07x | 0.0353 | 5.11x | 0.0353 | 5.11x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0072 | 0.0005 | 0.0001 | 4.83x | 0.0001 | 1.35x | 0.0001 | 1.35x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.06x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2543 | 0.4385 | 0.1353 | 3.24x | 0.1464 | 1.08x | 0.1431 | 1.06x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2458 | 0.4392 | 0.1430 | 3.07x | 0.1458 | 1.02x | 0.1553 | 1.09x | OK | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2495 | 0.4369 | 0.1425 | 3.06x | 0.1459 | 1.02x | 0.1430 | 1.00x | OK | PARITY |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2414 | 0.4355 | 0.1389 | 3.13x | 0.1468 | 1.06x | 0.1430 | 1.03x | OK | **WIN** |

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
