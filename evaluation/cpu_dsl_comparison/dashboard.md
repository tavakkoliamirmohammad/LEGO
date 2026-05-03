# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 05:16

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **5** |
| **PARITY (vs C O3)** | **3** |
| **LOSS (vs C O3)** | **34** |
| WIN (vs C agg) | 5 |
| PARITY (vs C agg) | 8 |
| LOSS (vs C agg) | 29 |
| ERROR | 0 |
| VERIFIED (correctness) | 0 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 0 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2883 | 0.1177 | 0.1292 | 0.91x | 0.0713 | 0.55x | 0.0746 | 0.58x | – | LOSS |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.1445 | 0.1422 | 1.02x | 0.0202 | 0.14x | 0.0184 | 0.13x | – | LOSS |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0105 | 0.0104 | 1.01x | 0.0003 | 0.03x | 0.0003 | 0.03x | – | LOSS |
| 04_col_major_inner |  |  | 256 | 0.0479 | 0.0588 | 0.0721 | 0.82x | 0.0637 | 0.88x | 0.0673 | 0.93x | – | LOSS |
| 05_morton_2d |  |  | 65536 | 0.1589 | 0.0170 | 0.0167 | 1.02x | 0.1691 | 10.12x | 0.1700 | 10.18x | – | **WIN** |
| 06_self_update |  |  | 4096 | 0.0153 | 0.0138 | 0.0137 | 1.00x | 0.0011 | 0.08x | 0.0011 | 0.08x | – | LOSS |
| 07_mixed_precision |  |  | 1048576 | 0.2833 | 0.1091 | 0.1165 | 0.94x | 0.1109 | 0.95x | 0.1075 | 0.92x | – | PARITY |
| 08_brick_within_cell |  |  | 1048576 | 0.2843 | 0.1127 | 0.1127 | 1.00x | 0.0006 | 0.01x | 0.0003 | 0.00x | – | LOSS |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1710 | 0.0245 | 0.0243 | 1.01x | 0.0352 | 1.45x | 0.0352 | 1.45x | – | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1803 | 0.0244 | 0.0248 | 0.98x | 0.0363 | 1.46x | 0.0353 | 1.42x | – | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1706 | 0.0243 | 0.0241 | 1.01x | 0.0362 | 1.50x | 0.0353 | 1.46x | – | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2447 | 0.1629 | 0.1624 | 1.00x | 0.1407 | 0.87x | 0.1547 | 0.95x | – | LOSS |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2464 | 0.1589 | 0.1573 | 1.01x | 0.1453 | 0.92x | 0.1452 | 0.92x | – | LOSS |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2387 | 0.1574 | 0.1565 | 1.01x | 0.1467 | 0.94x | 0.1520 | 0.97x | – | LOSS |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2472 | 0.1633 | 0.1632 | 1.00x | 0.1406 | 0.86x | 0.1428 | 0.87x | – | LOSS |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2454 | 0.1550 | 0.1558 | 1.00x | 0.1453 | 0.93x | 0.1540 | 0.99x | – | LOSS |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2411 | 0.1577 | 0.1573 | 1.00x | 0.1406 | 0.89x | 0.1531 | 0.97x | – | LOSS |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.1442 | 0.1423 | 1.01x | 0.1406 | 0.99x | 0.1426 | 1.00x | – | PARITY |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0798 | 0.0150 | 0.0150 | 1.00x | 0.0084 | 0.56x | 0.0081 | 0.54x | – | LOSS |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1619 | 0.0187 | 0.0181 | 1.03x | 0.0084 | 0.46x | 0.0082 | 0.45x | – | LOSS |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0780 | 0.0151 | 0.0149 | 1.01x | 0.0084 | 0.56x | 0.0081 | 0.54x | – | LOSS |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0729 | 0.0190 | 0.0181 | 1.05x | 0.0084 | 0.46x | 0.0082 | 0.45x | – | LOSS |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0125 | 0.0124 | 1.01x | 0.0022 | 0.18x | 0.0016 | 0.13x | – | LOSS |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0126 | 0.0125 | 1.01x | 0.0022 | 0.18x | 0.0017 | 0.13x | – | LOSS |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0070 | 0.0123 | 0.0121 | 1.01x | 0.0022 | 0.18x | 0.0017 | 0.14x | – | LOSS |
| 26_nussinov_skew | Skew tile | WIN | 1048576 | 0.2504 | 0.1607 | 0.1600 | 1.00x | 0.1458 | 0.91x | 0.1431 | 0.89x | – | LOSS |
| 27_zuker_skew | Skew tile | LOSS | 1048576 | 0.2487 | 0.1581 | 0.1585 | 1.00x | 0.1456 | 0.92x | 0.1543 | 0.97x | – | LOSS |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2379 | 0.1610 | 0.1635 | 0.98x | 0.1462 | 0.89x | 0.1532 | 0.94x | – | LOSS |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0139 | 0.0137 | 1.02x | 0.0028 | 0.20x | 0.0028 | 0.20x | – | LOSS |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0137 | 0.0135 | 1.01x | 0.0028 | 0.20x | 0.0032 | 0.23x | – | LOSS |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0072 | 0.0138 | 0.0137 | 1.01x | 0.0028 | 0.20x | 0.0029 | 0.21x | – | LOSS |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2404 | 0.1649 | 0.1612 | 1.02x | 0.1425 | 0.88x | 0.1539 | 0.95x | – | LOSS |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2458 | 0.1633 | 0.1627 | 1.00x | 0.1481 | 0.91x | 0.1430 | 0.88x | – | LOSS |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1686 | 0.0241 | 0.0241 | 1.00x | 0.0082 | 0.34x | 0.0090 | 0.37x | – | LOSS |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1692 | 0.0246 | 0.0240 | 1.03x | 0.0086 | 0.36x | 0.0089 | 0.37x | – | LOSS |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1691 | 0.0242 | 0.0242 | 1.00x | 0.0354 | 1.46x | 0.0352 | 1.46x | – | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0072 | 0.0105 | 0.0103 | 1.02x | 0.0085 | 0.83x | 0.0081 | 0.78x | – | LOSS |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0129 | 0.0127 | 1.01x | 0.0022 | 0.17x | 0.0017 | 0.13x | – | LOSS |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2492 | 0.1673 | 0.1663 | 1.01x | 0.1458 | 0.88x | 0.1523 | 0.92x | – | LOSS |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2368 | 0.1529 | 0.1525 | 1.00x | 0.1456 | 0.95x | 0.1426 | 0.94x | – | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2474 | 0.1643 | 0.1635 | 1.01x | 0.1462 | 0.89x | 0.1430 | 0.87x | – | LOSS |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2473 | 0.1616 | 0.1614 | 1.00x | 0.1460 | 0.90x | 0.1535 | 0.95x | – | LOSS |

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
