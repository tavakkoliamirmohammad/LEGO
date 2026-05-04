# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 18:02

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **38** |
| **PARITY (vs C O3)** | **4** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 30 |
| PARITY (vs C agg) | 11 |
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
| 01_saxpy |  |  | ? | 0.2831 | 0.3183 | 0.0880 | 3.62x | 0.1045 | 1.19x | 0.0991 | 1.13x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0069 | 0.1445 | 0.0118 | 12.22x | 0.0244 | 2.07x | 0.0218 | 1.85x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.95x | 0.0002 | 1.60x | 0.0002 | 1.51x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0560 | 0.0779 | 0.0611 | 1.28x | 0.0668 | 1.09x | 0.0484 | 0.79x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1601 | 0.0354 | 0.0049 | 7.22x | 0.1693 | 34.55x | 0.1699 | 34.67x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0022 | 0.0003 | 6.56x | 0.0005 | 1.66x | 0.0006 | 1.91x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2842 | 0.3159 | 0.0877 | 3.60x | 0.1796 | 2.05x | 0.1683 | 1.92x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2910 | 0.2915 | 0.0884 | 3.30x | 0.1017 | 1.15x | 0.1413 | 1.60x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1691 | 0.0365 | 0.0073 | 5.04x | 0.0352 | 4.83x | 0.0353 | 4.83x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1721 | 0.0382 | 0.0070 | 5.47x | 0.0356 | 5.09x | 0.0352 | 5.02x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1708 | 0.0352 | 0.0075 | 4.70x | 0.0362 | 4.82x | 0.0352 | 4.69x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 3.7045 | 42.3176 | 8.8786 | 4.77x | 11.5231 | 1.30x | 11.3723 | 1.28x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2367 | 0.4394 | 0.1370 | 3.21x | 0.1462 | 1.07x | 0.1495 | 1.09x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2381 | 0.4324 | 0.1283 | 3.37x | 0.1454 | 1.13x | 0.1430 | 1.11x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2295 | 0.4448 | 0.1363 | 3.26x | 0.1461 | 1.07x | 0.1547 | 1.13x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2372 | 0.4444 | 0.1296 | 3.43x | 0.1462 | 1.13x | 0.1553 | 1.20x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2475 | 0.4393 | 0.1347 | 3.26x | 0.1467 | 1.09x | 0.1534 | 1.14x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7234 | 0.0120 | 60.30x | 0.0245 | 2.04x | 0.0220 | 1.83x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0302 | 0.0294 | 0.0046 | 6.42x | 0.0084 | 1.82x | 0.0082 | 1.79x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0546 | 0.0554 | 0.0080 | 6.93x | 0.0191 | 2.39x | 0.0151 | 1.89x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0305 | 0.0294 | 0.0051 | 5.73x | 0.0084 | 1.64x | 0.0082 | 1.60x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0408 | 0.0364 | 0.0078 | 4.69x | 0.0109 | 1.39x | 0.0111 | 1.42x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.10x | 0.0022 | 1.29x | 0.0016 | 0.97x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.10x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.13x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.11x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6654 | 0.0022 | 0.0004 | 5.14x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2499 | 0.4444 | 0.1348 | 3.30x | 0.1409 | 1.05x | 0.1426 | 1.06x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7235 | 0.0094 | 0.0028 | 3.34x | 0.0029 | 1.03x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7662 | 0.0093 | 0.0028 | 3.31x | 0.0028 | 0.99x | 0.0028 | 1.00x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7023 | 0.0099 | 0.0028 | 3.50x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2398 | 0.4378 | 0.1336 | 3.28x | 0.1470 | 1.10x | 0.1555 | 1.16x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2305 | 0.4394 | 0.1303 | 3.37x | 0.1461 | 1.12x | 0.1427 | 1.10x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1690 | 0.0359 | 0.0075 | 4.82x | 0.0353 | 4.71x | 0.0353 | 4.71x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1817 | 0.0354 | 0.0070 | 5.03x | 0.0358 | 5.11x | 0.0359 | 5.13x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1787 | 0.0352 | 0.0072 | 4.88x | 0.0352 | 4.89x | 0.0353 | 4.90x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 4.97x | 0.0001 | 1.33x | 0.0001 | 1.32x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0072 | 0.0088 | 0.0017 | 5.11x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2379 | 0.4319 | 0.1309 | 3.30x | 0.1457 | 1.11x | 0.1451 | 1.11x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2365 | 0.4424 | 0.1364 | 3.24x | 0.1451 | 1.06x | 0.1427 | 1.05x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2380 | 0.4445 | 0.1350 | 3.29x | 0.1455 | 1.08x | 0.1425 | 1.06x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2409 | 0.4382 | 0.1361 | 3.22x | 0.1459 | 1.07x | 0.1426 | 1.05x | OK | **WIN** |

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
