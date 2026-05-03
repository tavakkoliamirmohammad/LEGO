# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 04:26

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| WIN | 2 |
| PARITY | 32 |
| LOSS | 8 |
| ERROR | 0 |
| VERIFIED (correctness) | 32 |
| PENDING (correctness) | 10 |
| vec_iso > 1.5× | 0 |

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | vs_numpy | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2787 | 0.1206 | 0.1143 | 1.05x | 2.44x | ✓ | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.1438 | 0.1420 | 1.01x | 0.05x | ✓ | **PARITY** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0115 | 0.0112 | 1.02x | 0.19x | ✓ | **PARITY** |
| 04_col_major_inner |  |  | 256 | 0.0479 | 0.0592 | 0.0727 | 0.81x | 0.66x | ✓ | **LOSS** |
| 05_morton_2d |  |  | 65536 | 0.1592 | 0.0166 | 0.0164 | 1.01x | 9.70x | ✓ | **PARITY** |
| 06_self_update |  |  | 4096 | 0.0156 | 0.0145 | 0.0140 | 1.04x | 1.11x | ✓ | **PARITY** |
| 07_mixed_precision |  |  | 1048576 | 0.2882 | 0.1195 | 0.1101 | 1.08x | 2.62x | ✓ | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2916 | 0.1126 | 0.1082 | 1.04x | 2.69x | ✓ | **PARITY** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1746 | 0.0253 | 0.0246 | 1.03x | 7.09x | ✓ | **PARITY** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1770 | 0.0246 | 0.0244 | 1.01x | 7.25x | ✓ | **PARITY** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1681 | 0.0246 | 0.0243 | 1.01x | 6.92x | ✓ | **PARITY** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2373 | 0.1580 | 0.1540 | 1.03x | 1.54x | ✓ | **PARITY** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2420 | 0.1584 | 0.1537 | 1.03x | 1.57x | ✓ | **PARITY** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2449 | 0.1612 | 0.1607 | 1.00x | 1.52x | ✓ | **PARITY** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2469 | 0.1609 | 0.1561 | 1.03x | 1.58x | ✓ | **PARITY** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2386 | 0.1525 | 0.1516 | 1.01x | 1.57x | ✓ | **PARITY** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2387 | 0.1638 | 0.1583 | 1.03x | 1.51x | ✓ | **PARITY** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.1485 | 0.1420 | 1.05x | 0.05x | ✓ | **PARITY** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0779 | 0.0153 | 0.0151 | 1.02x | 5.17x | ✓ | **PARITY** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1587 | 0.0209 | 0.0199 | 1.05x | 7.97x | ? | **PARITY** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0729 | 0.0149 | 0.0150 | 1.00x | 4.86x | ✓ | **PARITY** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0706 | 0.0193 | 0.0186 | 1.04x | 3.80x | ✓ | **PARITY** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0070 | 0.0125 | 0.0175 | 0.71x | 0.40x | ? | **LOSS** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0128 | 0.0176 | 0.73x | 0.40x | ? | **LOSS** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0123 | 0.0172 | 0.72x | 0.41x | ? | **LOSS** |
| 26_nussinov_skew | Skew tile | WIN | 1048576 | 0.2507 | 0.1604 | 0.1557 | 1.03x | 1.61x | ? | **PARITY** |
| 27_zuker_skew | Skew tile | LOSS | 1048576 | 0.2506 | 0.1607 | 0.1606 | 1.00x | 1.56x | ? | **PARITY** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2536 | 0.1621 | 0.1573 | 1.03x | 1.61x | ✓ | **PARITY** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0147 | 0.0179 | 0.82x | 0.39x | ? | **LOSS** |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0142 | 0.0188 | 0.76x | 0.38x | ? | **LOSS** |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0140 | 0.0179 | 0.78x | 0.40x | ? | **LOSS** |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2355 | 0.1561 | 0.1558 | 1.00x | 1.51x | ✓ | **PARITY** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2414 | 0.1654 | 0.1641 | 1.01x | 1.47x | ✓ | **PARITY** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1729 | 0.0251 | 0.0250 | 1.01x | 6.92x | ✓ | **PARITY** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1717 | 0.0257 | 0.0256 | 1.00x | 6.72x | ✓ | **PARITY** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1728 | 0.0244 | 0.0244 | 1.00x | 7.08x | ✓ | **PARITY** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0075 | 0.0107 | 0.0108 | 0.99x | 0.70x | ✓ | **PARITY** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0126 | 0.0173 | 0.73x | 0.41x | ? | **LOSS** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2464 | 0.1601 | 0.1599 | 1.00x | 1.54x | ✓ | **PARITY** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2495 | 0.1635 | 0.1636 | 1.00x | 1.53x | ✓ | **PARITY** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2388 | 0.1595 | 0.1586 | 1.01x | 1.51x | ✓ | **PARITY** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2444 | 0.1640 | 0.1636 | 1.00x | 1.49x | ✓ | **PARITY** |

## Known Gaps

- **R19**: Strided-gather index vector mismatch in `LegoVectorize::emitVectorBody`.
  The catch-all arith path vectorizes `MulIOp(iv, stride)` before the Strided path
  reads it, producing incorrect gather indices. Affects candidates 23–27, 29–31, 38.
  Fix: use pre-vectorization scalar index in Strided path.

- **R18 (reduction guard)**: k-reduction loops correctly skip vectorization.
  This is correct behavior but contributes to PARITY (not WIN) for GEMM variants.
