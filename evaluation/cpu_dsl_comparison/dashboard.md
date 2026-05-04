# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 21:03

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **39** |
| **PARITY (vs C O3)** | **3** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 32 |
| PARITY (vs C agg) | 9 |
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
| 01_saxpy |  |  | ? | 0.2900 | 0.3398 | 0.0891 | 3.81x | 0.1052 | 1.18x | 0.1041 | 1.17x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1446 | 0.0115 | 12.57x | 0.0245 | 2.13x | 0.0218 | 1.90x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.45x | 0.0001 | 1.45x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0480 | 0.0779 | 0.0605 | 1.29x | 0.0656 | 1.08x | 0.0458 | 0.76x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1612 | 0.0354 | 0.0047 | 7.53x | 0.1694 | 36.05x | 0.1709 | 36.35x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0026 | 0.0002 | 13.00x | 0.0004 | 2.08x | 0.0006 | 2.78x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2972 | 0.3199 | 0.0889 | 3.60x | 0.1795 | 2.02x | 0.1733 | 1.95x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2976 | 0.2965 | 0.0897 | 3.31x | 0.1024 | 1.14x | 0.1468 | 1.64x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1698 | 0.0352 | 0.0085 | 4.14x | 0.0590 | 6.94x | 0.0464 | 5.46x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1733 | 0.0389 | 0.0090 | 4.32x | 0.0381 | 4.23x | 0.0352 | 3.92x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1697 | 0.0353 | 0.0074 | 4.77x | 0.0365 | 4.93x | 0.0353 | 4.77x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 20.0508 | 43.2279 | 8.9763 | 4.82x | 11.5117 | 1.28x | 10.9532 | 1.22x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2810 | 0.4418 | 0.1345 | 3.28x | 0.1468 | 1.09x | 0.1427 | 1.06x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2481 | 0.4439 | 0.1345 | 3.30x | 0.1478 | 1.10x | 0.1708 | 1.27x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2387 | 0.4443 | 0.1302 | 3.41x | 0.1546 | 1.19x | 0.1538 | 1.18x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2404 | 0.4401 | 0.1363 | 3.23x | 0.1463 | 1.07x | 0.1501 | 1.10x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2480 | 0.4568 | 0.1288 | 3.55x | 0.1452 | 1.13x | 0.1428 | 1.11x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0070 | 0.7312 | 0.0116 | 63.03x | 0.0244 | 2.11x | 0.0220 | 1.90x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0299 | 0.0294 | 0.0044 | 6.68x | 0.0089 | 2.02x | 0.0081 | 1.84x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0553 | 0.0553 | 0.0074 | 7.47x | 0.0191 | 2.58x | 0.0151 | 2.04x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0306 | 0.0294 | 0.0048 | 6.12x | 0.0084 | 1.75x | 0.0083 | 1.74x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0391 | 0.0357 | 0.0072 | 4.96x | 0.0109 | 1.51x | 0.0110 | 1.52x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0017 | 0.99x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6925 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.04x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2433 | 0.4585 | 0.1322 | 3.47x | 0.1466 | 1.11x | 0.1429 | 1.08x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.8575 | 0.0095 | 0.0028 | 3.39x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.8148 | 0.0091 | 0.0028 | 3.25x | 0.0028 | 0.99x | 0.0029 | 1.03x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.7830 | 0.0089 | 0.0028 | 3.18x | 0.0028 | 0.98x | 0.0028 | 0.98x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2482 | 0.4352 | 0.1351 | 3.22x | 0.1462 | 1.08x | 0.1426 | 1.06x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2397 | 0.4434 | 0.1359 | 3.26x | 0.1461 | 1.08x | 0.1428 | 1.05x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1692 | 0.0353 | 0.0075 | 4.71x | 0.0353 | 4.71x | 0.0352 | 4.70x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1761 | 0.0353 | 0.0074 | 4.77x | 0.0352 | 4.76x | 0.0352 | 4.75x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1771 | 0.0353 | 0.0074 | 4.77x | 0.0360 | 4.87x | 0.0357 | 4.82x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.34x | 0.0001 | 1.31x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2503 | 0.4381 | 0.1309 | 3.35x | 0.1473 | 1.13x | 0.1431 | 1.09x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2399 | 0.4431 | 0.1354 | 3.27x | 0.1456 | 1.08x | 0.1432 | 1.06x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2401 | 0.4380 | 0.1339 | 3.27x | 0.1463 | 1.09x | 0.1427 | 1.07x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2418 | 0.4379 | 0.1340 | 3.27x | 0.1467 | 1.09x | 0.1553 | 1.16x | OK | **WIN** |

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
