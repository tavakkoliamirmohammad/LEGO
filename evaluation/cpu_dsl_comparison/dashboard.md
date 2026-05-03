# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 04:40

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| WIN | 3 |
| PARITY | 30 |
| LOSS | 9 |
| ERROR | 0 |
| VERIFIED (correctness) | 41 |
| PENDING (correctness) | 1 |
| vec_iso > 1.5× | 0 |

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | vs_numpy | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2743 | 0.1277 | 0.1164 | 1.10x | 2.36x | ✓ | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.1439 | 0.1420 | 1.01x | 0.05x | ✓ | **PARITY** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0105 | 0.0104 | 1.01x | 0.20x | ✓ | **PARITY** |
| 04_col_major_inner |  |  | 256 | 0.0479 | 0.0593 | 0.0725 | 0.82x | 0.66x | ✓ | **LOSS** |
| 05_morton_2d |  |  | 65536 | 0.1621 | 0.0176 | 0.0164 | 1.07x | 9.87x | ✓ | **WIN** |
| 06_self_update |  |  | 4096 | 0.0152 | 0.0140 | 0.0139 | 1.01x | 1.09x | ✓ | **PARITY** |
| 07_mixed_precision |  |  | 1048576 | 0.2844 | 0.1094 | 0.1126 | 0.97x | 2.53x | ✓ | **PARITY** |
| 08_brick_within_cell |  |  | 1048576 | 0.2818 | 0.1048 | 0.1091 | 0.96x | 2.58x | ✓ | **PARITY** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1723 | 0.0252 | 0.0256 | 0.99x | 6.74x | ✓ | **PARITY** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1807 | 0.0252 | 0.0248 | 1.01x | 7.28x | ✓ | **PARITY** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1696 | 0.0245 | 0.0243 | 1.01x | 6.99x | ✓ | **PARITY** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2400 | 0.1615 | 0.1585 | 1.02x | 1.51x | ✓ | **PARITY** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2496 | 0.1684 | 0.1674 | 1.01x | 1.49x | ✓ | **PARITY** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2516 | 0.1638 | 0.1644 | 1.00x | 1.53x | ✓ | **PARITY** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2368 | 0.1629 | 0.1592 | 1.02x | 1.49x | ✓ | **PARITY** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2412 | 0.1601 | 0.1537 | 1.04x | 1.57x | ✓ | **PARITY** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2487 | 0.1631 | 0.1585 | 1.03x | 1.57x | ✓ | **PARITY** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.1447 | 0.1431 | 1.01x | 0.05x | ✓ | **PARITY** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0772 | 0.0163 | 0.0177 | 0.92x | 4.36x | ✓ | **LOSS** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1572 | 0.0189 | 0.0181 | 1.04x | 8.69x | ? | **PARITY** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0767 | 0.0164 | 0.0165 | 0.99x | 4.64x | ✓ | **PARITY** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0698 | 0.0200 | 0.0191 | 1.05x | 3.65x | ✓ | **PARITY** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0126 | 0.0181 | 0.70x | 0.39x | ✓ | **LOSS** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0125 | 0.0184 | 0.68x | 0.39x | ✓ | **LOSS** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0125 | 0.0181 | 0.69x | 0.39x | ✓ | **LOSS** |
| 26_nussinov_skew | Skew tile | WIN | 1048576 | 0.2499 | 0.1601 | 0.1598 | 1.00x | 1.56x | ✓ | **PARITY** |
| 27_zuker_skew | Skew tile | LOSS | 1048576 | 0.2404 | 0.1640 | 0.1639 | 1.00x | 1.47x | ✓ | **PARITY** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2469 | 0.1634 | 0.1634 | 1.00x | 1.51x | ✓ | **PARITY** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0142 | 0.0189 | 0.75x | 0.37x | ✓ | **LOSS** |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0072 | 0.0143 | 0.0188 | 0.76x | 0.38x | ✓ | **LOSS** |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0140 | 0.0188 | 0.74x | 0.38x | ✓ | **LOSS** |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2420 | 0.1634 | 0.1595 | 1.02x | 1.52x | ✓ | **PARITY** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2399 | 0.1531 | 0.1532 | 1.00x | 1.57x | ✓ | **PARITY** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1689 | 0.0249 | 0.0244 | 1.02x | 6.92x | ✓ | **PARITY** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1687 | 0.0244 | 0.0240 | 1.01x | 7.02x | ✓ | **PARITY** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1736 | 0.0256 | 0.0243 | 1.05x | 7.14x | ✓ | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0077 | 0.0107 | 0.0106 | 1.01x | 0.73x | ✓ | **PARITY** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0122 | 0.0180 | 0.68x | 0.40x | ✓ | **LOSS** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2361 | 0.1634 | 0.1599 | 1.02x | 1.48x | ✓ | **PARITY** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2512 | 0.1587 | 0.1584 | 1.00x | 1.59x | ✓ | **PARITY** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2386 | 0.1625 | 0.1585 | 1.03x | 1.51x | ✓ | **PARITY** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2539 | 0.1603 | 0.1560 | 1.03x | 1.63x | ✓ | **PARITY** |

## Known Gaps

- **R19**: Strided-gather index vector mismatch in `LegoVectorize::emitVectorBody`.
  The catch-all arith path vectorizes `MulIOp(iv, stride)` before the Strided path
  reads it, producing incorrect gather indices. Affects candidates 23–27, 29–31, 38.
  Fix: use pre-vectorization scalar index in Strided path.

- **R18 (reduction guard)**: k-reduction loops correctly skip vectorization.
  This is correct behavior but contributes to PARITY (not WIN) for GEMM variants.
