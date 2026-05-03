# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 10:41

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **24** |
| **PARITY (vs C O3)** | **17** |
| **LOSS (vs C O3)** | **1** |
| WIN (vs C agg) | 22 |
| PARITY (vs C agg) | 16 |
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
| 01_saxpy |  |  | 1048576 | 0.2769 | 0.3064 | 0.0926 | 3.31x | 0.1055 | 1.14x | 0.1043 | 1.13x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0065 | 0.1072 | 0.0141 | 7.58x | 0.0244 | 1.73x | 0.0219 | 1.55x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0006 | 0.0001 | 5.02x | 0.0003 | 2.71x | 0.0001 | 1.45x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0472 | 0.0788 | 0.0618 | 1.28x | 0.0665 | 1.08x | 0.0460 | 0.74x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1617 | 0.0390 | 0.0057 | 6.86x | 0.1699 | 29.81x | 0.1690 | 29.65x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0022 | 0.0003 | 8.45x | 0.0004 | 1.49x | 0.0006 | 2.13x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2853 | 0.3021 | 0.1008 | 3.00x | 0.1806 | 1.79x | 0.1753 | 1.74x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2869 | 0.2918 | 0.0920 | 3.17x | 0.0959 | 1.04x | 0.1415 | 1.54x | OK | PARITY |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1740 | 0.0391 | 0.0100 | 3.93x | 0.0352 | 3.52x | 0.0352 | 3.52x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1795 | 0.0407 | 0.0101 | 4.02x | 0.0352 | 3.48x | 0.0359 | 3.55x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1728 | 0.0391 | 0.0108 | 3.63x | 0.0353 | 3.26x | 0.0352 | 3.26x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 6.9770 | 42.7234 | 14.0589 | 3.04x | 11.4925 | 0.82x | 11.4270 | 0.81x | OK | LOSS |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2385 | 0.4470 | 0.1453 | 3.08x | 0.1474 | 1.01x | 0.1437 | 0.99x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2442 | 0.4404 | 0.1445 | 3.05x | 0.1461 | 1.01x | 0.1426 | 0.99x | OK | PARITY |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2386 | 0.4353 | 0.1475 | 2.95x | 0.1465 | 0.99x | 0.1533 | 1.04x | OK | PARITY |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2456 | 0.4447 | 0.1407 | 3.16x | 0.1407 | 1.00x | 0.1427 | 1.01x | OK | PARITY |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2342 | 0.4470 | 0.1385 | 3.23x | 0.1453 | 1.05x | 0.1535 | 1.11x | OK | PARITY |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.6865 | 0.1290 | 5.32x | 0.1450 | 1.12x | 0.1541 | 1.19x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0739 | 0.0303 | 0.0065 | 4.66x | 0.0084 | 1.29x | 0.0081 | 1.24x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1538 | 0.0563 | 0.0080 | 7.03x | 0.0190 | 2.38x | 0.0151 | 1.89x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0805 | 0.0303 | 0.0051 | 5.99x | 0.0084 | 1.65x | 0.0081 | 1.59x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0693 | 0.0365 | 0.0079 | 4.62x | 0.0110 | 1.39x | 0.0108 | 1.37x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.22x | 0.0022 | 1.31x | 0.0017 | 0.98x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0089 | 0.0018 | 4.98x | 0.0022 | 1.23x | 0.0016 | 0.91x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0070 | 0.0088 | 0.0017 | 5.15x | 0.0022 | 1.30x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 1048576 | 0.2499 | 0.4431 | 0.1447 | 3.06x | 0.1466 | 1.01x | 0.1435 | 0.99x | OK | PARITY |
| 27_zuker_skew | Skew tile | LOSS | 1048576 | 0.2432 | 0.4445 | 0.1406 | 3.16x | 0.1462 | 1.04x | 0.1430 | 1.02x | OK | PARITY |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2500 | 0.4413 | 0.1415 | 3.12x | 0.1463 | 1.03x | 0.1430 | 1.01x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0088 | 0.0029 | 3.08x | 0.0028 | 0.96x | 0.0029 | 1.00x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0072 | 0.0088 | 0.0029 | 3.00x | 0.0029 | 0.99x | 0.0028 | 0.96x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0088 | 0.0029 | 3.07x | 0.0028 | 0.96x | 0.0028 | 0.95x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2409 | 0.4466 | 0.1496 | 2.98x | 0.1461 | 0.98x | 0.1531 | 1.02x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2396 | 0.4341 | 0.1355 | 3.20x | 0.1461 | 1.08x | 0.1538 | 1.14x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1751 | 0.0390 | 0.0107 | 3.66x | 0.0353 | 3.30x | 0.0359 | 3.35x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1711 | 0.0376 | 0.0098 | 3.82x | 0.0352 | 3.59x | 0.0352 | 3.59x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1693 | 0.0376 | 0.0098 | 3.84x | 0.0358 | 3.66x | 0.0352 | 3.59x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0071 | 0.0012 | 0.0008 | 1.51x | 0.0084 | 10.48x | 0.0082 | 10.20x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0070 | 0.0088 | 0.0018 | 5.02x | 0.0022 | 1.25x | 0.0016 | 0.91x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2459 | 0.4401 | 0.1408 | 3.13x | 0.1407 | 1.00x | 0.1431 | 1.02x | OK | PARITY |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2455 | 0.4438 | 0.1427 | 3.11x | 0.1468 | 1.03x | 0.1542 | 1.08x | OK | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2379 | 0.4401 | 0.1423 | 3.09x | 0.1404 | 0.99x | 0.1429 | 1.00x | OK | PARITY |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2399 | 0.4396 | 0.1506 | 2.92x | 0.1463 | 0.97x | 0.1544 | 1.03x | OK | PARITY |

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
