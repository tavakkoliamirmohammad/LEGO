# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 06:01

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **7** |
| **PARITY (vs C O3)** | **1** |
| **LOSS (vs C O3)** | **34** |
| WIN (vs C agg) | 8 |
| PARITY (vs C agg) | 8 |
| LOSS (vs C agg) | 26 |
| ERROR | 0 |
| VERIFIED (correctness) | 41 |
| PENDING (correctness) | 1 |
| vec_iso > 1.5× | 32 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2822 | 0.3279 | 0.1285 | 2.55x | 0.1055 | 0.82x | 0.1048 | 0.82x | OK | LOSS |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.7047 | 0.1442 | 4.89x | 0.0244 | 0.17x | 0.0222 | 0.15x | OK | LOSS |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0111 | 0.0105 | 1.06x | 0.0001 | 0.01x | 0.0001 | 0.01x | OK | LOSS |
| 04_col_major_inner |  |  | 256 | 0.0491 | 0.0911 | 0.0735 | 1.24x | 0.0634 | 0.86x | 0.0556 | 0.76x | OK | LOSS |
| 05_morton_2d |  |  | 65536 | 0.1592 | 0.0485 | 0.0167 | 2.91x | 0.1690 | 10.12x | 0.1700 | 10.18x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0152 | 0.0238 | 0.0137 | 1.73x | 0.0004 | 0.03x | 0.0006 | 0.04x | OK | LOSS |
| 07_mixed_precision |  |  | 1048576 | 0.2782 | 0.3128 | 0.1184 | 2.64x | 0.1817 | 1.53x | 0.1739 | 1.47x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2838 | 0.3079 | 0.1148 | 2.68x | 0.1009 | 0.88x | 0.1414 | 1.23x | OK | LOSS |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1794 | 0.0560 | 0.0239 | 2.34x | 0.0352 | 1.47x | 0.0352 | 1.47x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1765 | 0.0554 | 0.0242 | 2.29x | 0.0358 | 1.48x | 0.0352 | 1.46x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1692 | 0.0531 | 0.0237 | 2.24x | 0.0353 | 1.49x | 0.0353 | 1.49x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 27.0571 | 42.7703 | 8.8440 | 4.84x | 12.0706 | 1.36x | 10.9979 | 1.24x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2351 | 0.4512 | 0.1565 | 2.88x | 0.1452 | 0.93x | 0.1470 | 0.94x | OK | LOSS |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2512 | 0.4690 | 0.1647 | 2.85x | 0.1459 | 0.89x | 0.1448 | 0.88x | OK | LOSS |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2447 | 0.4565 | 0.1551 | 2.94x | 0.1460 | 0.94x | 0.1541 | 0.99x | OK | LOSS |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2357 | 0.4556 | 0.1578 | 2.89x | 0.1407 | 0.89x | 0.1427 | 0.90x | OK | LOSS |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2475 | 0.4570 | 0.1631 | 2.80x | 0.1460 | 0.90x | 0.1522 | 0.93x | OK | LOSS |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.7005 | 0.1421 | 4.93x | 0.1466 | 1.03x | 0.1428 | 1.00x | OK | PARITY |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0748 | 0.0410 | 0.0151 | 2.72x | 0.0088 | 0.59x | 0.0081 | 0.54x | OK | LOSS |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1732 | 0.0674 | 0.0184 | 3.67x | 0.0084 | 0.46x | 0.0081 | 0.44x | ? | LOSS |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0741 | 0.0412 | 0.0149 | 2.76x | 0.0084 | 0.56x | 0.0085 | 0.57x | OK | LOSS |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0695 | 0.0480 | 0.0201 | 2.38x | 0.0084 | 0.42x | 0.0081 | 0.40x | OK | LOSS |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0072 | 0.0151 | 0.0124 | 1.21x | 0.0022 | 0.18x | 0.0017 | 0.13x | OK | LOSS |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0155 | 0.0126 | 1.23x | 0.0022 | 0.17x | 0.0017 | 0.13x | OK | LOSS |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0070 | 0.0149 | 0.0122 | 1.22x | 0.0022 | 0.18x | 0.0017 | 0.14x | OK | LOSS |
| 26_nussinov_skew | Skew tile | WIN | 1048576 | 0.2414 | 0.4534 | 0.1542 | 2.94x | 0.1462 | 0.95x | 0.1541 | 1.00x | OK | LOSS |
| 27_zuker_skew | Skew tile | LOSS | 1048576 | 0.2549 | 0.4546 | 0.1587 | 2.86x | 0.1464 | 0.92x | 0.1428 | 0.90x | OK | LOSS |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2529 | 0.4570 | 0.1601 | 2.85x | 0.1481 | 0.93x | 0.1578 | 0.99x | OK | LOSS |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0154 | 0.0137 | 1.12x | 0.0028 | 0.21x | 0.0028 | 0.20x | OK | LOSS |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0154 | 0.0140 | 1.10x | 0.0028 | 0.20x | 0.0028 | 0.20x | OK | LOSS |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0072 | 0.0158 | 0.0143 | 1.10x | 0.0028 | 0.19x | 0.0029 | 0.20x | OK | LOSS |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2491 | 0.4588 | 0.1630 | 2.81x | 0.1471 | 0.90x | 0.1428 | 0.88x | OK | LOSS |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2396 | 0.4574 | 0.1576 | 2.90x | 0.1469 | 0.93x | 0.1532 | 0.97x | OK | LOSS |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1685 | 0.0526 | 0.0244 | 2.15x | 0.0087 | 0.36x | 0.0088 | 0.36x | OK | LOSS |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1708 | 0.0528 | 0.0241 | 2.19x | 0.0084 | 0.35x | 0.0088 | 0.37x | OK | LOSS |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1719 | 0.0531 | 0.0241 | 2.20x | 0.0357 | 1.48x | 0.0353 | 1.46x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0073 | 0.0111 | 0.0108 | 1.03x | 0.0084 | 0.78x | 0.0081 | 0.75x | OK | LOSS |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0151 | 0.0123 | 1.23x | 0.0022 | 0.18x | 0.0017 | 0.14x | OK | LOSS |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2397 | 0.4590 | 0.1610 | 2.85x | 0.1468 | 0.91x | 0.1552 | 0.96x | OK | LOSS |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2396 | 0.4577 | 0.1612 | 2.84x | 0.1464 | 0.91x | 0.1537 | 0.95x | OK | LOSS |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2351 | 0.4492 | 0.1572 | 2.86x | 0.1457 | 0.93x | 0.1524 | 0.97x | OK | LOSS |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2513 | 0.4576 | 0.1591 | 2.88x | 0.1437 | 0.90x | 0.1472 | 0.93x | OK | LOSS |

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
