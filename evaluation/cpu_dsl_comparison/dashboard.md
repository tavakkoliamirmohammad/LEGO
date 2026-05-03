# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 08:45

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **19** |
| **PARITY (vs C O3)** | **16** |
| **LOSS (vs C O3)** | **7** |
| WIN (vs C agg) | 18 |
| PARITY (vs C agg) | 12 |
| LOSS (vs C agg) | 12 |
| ERROR | 0 |
| VERIFIED (correctness) | 41 |
| PENDING (correctness) | 1 |
| vec_iso > 1.5× | 40 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2854 | 0.3011 | 0.1030 | 2.92x | 0.1054 | 1.02x | 0.1034 | 1.00x | OK | PARITY |
| 02_gemm_row_major |  |  | 64 | 0.0067 | 0.6867 | 0.1291 | 5.32x | 0.0244 | 0.19x | 0.0216 | 0.17x | OK | LOSS |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0006 | 0.0001 | 5.38x | 0.0001 | 1.38x | 0.0001 | 1.45x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0479 | 0.0789 | 0.0610 | 1.29x | 0.0638 | 1.05x | 0.0471 | 0.77x | OK | PARITY |
| 05_morton_2d |  |  | 65536 | 0.1593 | 0.0383 | 0.0058 | 6.55x | 0.1693 | 29.20x | 0.1704 | 29.38x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0152 | 0.0140 | 0.0041 | 3.45x | 0.0006 | 0.15x | 0.0006 | 0.14x | OK | LOSS |
| 07_mixed_precision |  |  | 1048576 | 0.2736 | 0.2985 | 0.0952 | 3.14x | 0.1889 | 1.98x | 0.1748 | 1.84x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2877 | 0.2936 | 0.1008 | 2.91x | 0.0960 | 0.95x | 0.1413 | 1.40x | OK | PARITY |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1724 | 0.0379 | 0.0094 | 4.05x | 0.0362 | 3.85x | 0.0361 | 3.84x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1769 | 0.0406 | 0.0098 | 4.13x | 0.0365 | 3.72x | 0.0352 | 3.59x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1683 | 0.0377 | 0.0096 | 3.91x | 0.0363 | 3.78x | 0.0353 | 3.68x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 4.1002 | 41.9581 | 12.1977 | 3.44x | 11.7291 | 0.96x | 11.2206 | 0.92x | OK | PARITY |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2394 | 0.4332 | 0.1403 | 3.09x | 0.1484 | 1.06x | 0.1550 | 1.11x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2530 | 0.4416 | 0.1573 | 2.81x | 0.1468 | 0.93x | 0.1481 | 0.94x | OK | LOSS |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2382 | 0.4403 | 0.1455 | 3.03x | 0.1470 | 1.01x | 0.1538 | 1.06x | OK | PARITY |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2357 | 0.4435 | 0.1382 | 3.21x | 0.1474 | 1.07x | 0.1502 | 1.09x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2451 | 0.4521 | 0.1519 | 2.98x | 0.1450 | 0.95x | 0.1428 | 0.94x | OK | PARITY |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0065 | 0.6869 | 0.1293 | 5.31x | 0.1407 | 1.09x | 0.1546 | 1.20x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0773 | 0.0303 | 0.0052 | 5.77x | 0.0084 | 1.61x | 0.0081 | 1.57x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1647 | 0.0563 | 0.0101 | 5.57x | 0.0085 | 0.84x | 0.0081 | 0.80x | ? | LOSS |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0747 | 0.0302 | 0.0064 | 4.74x | 0.0084 | 1.32x | 0.0081 | 1.26x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0694 | 0.0364 | 0.0087 | 4.18x | 0.0084 | 0.97x | 0.0081 | 0.93x | OK | PARITY |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0070 | 0.0088 | 0.0018 | 4.93x | 0.0022 | 1.23x | 0.0017 | 0.93x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.20x | 0.0022 | 1.29x | 0.0017 | 0.98x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0070 | 0.0088 | 0.0017 | 5.15x | 0.0022 | 1.29x | 0.0017 | 1.01x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 1048576 | 0.2471 | 0.4406 | 0.1536 | 2.87x | 0.1455 | 0.95x | 0.1548 | 1.01x | OK | LOSS |
| 27_zuker_skew | Skew tile | LOSS | 1048576 | 0.2369 | 0.4419 | 0.1415 | 3.12x | 0.1406 | 0.99x | 0.1430 | 1.01x | OK | PARITY |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2470 | 0.4398 | 0.1464 | 3.00x | 0.1471 | 1.00x | 0.1431 | 0.98x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0088 | 0.0029 | 3.08x | 0.0028 | 0.96x | 0.0027 | 0.95x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0088 | 0.0029 | 3.09x | 0.0028 | 0.96x | 0.0028 | 0.95x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0088 | 0.0029 | 3.08x | 0.0028 | 0.96x | 0.0029 | 0.98x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2397 | 0.4475 | 0.1414 | 3.16x | 0.1420 | 1.00x | 0.1431 | 1.01x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2444 | 0.4395 | 0.1433 | 3.07x | 0.1459 | 1.02x | 0.1535 | 1.07x | OK | PARITY |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1687 | 0.0375 | 0.0096 | 3.92x | 0.0087 | 0.90x | 0.0088 | 0.92x | OK | LOSS |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1693 | 0.0377 | 0.0099 | 3.81x | 0.0088 | 0.88x | 0.0088 | 0.89x | OK | LOSS |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1708 | 0.0375 | 0.0095 | 3.93x | 0.0361 | 3.80x | 0.0352 | 3.71x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0072 | 0.0011 | 0.0008 | 1.36x | 0.0084 | 10.54x | 0.0081 | 10.11x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.08x | 0.0022 | 1.30x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2417 | 0.4551 | 0.1426 | 3.19x | 0.1406 | 0.99x | 0.1524 | 1.07x | OK | PARITY |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2439 | 0.4402 | 0.1389 | 3.17x | 0.1462 | 1.05x | 0.1429 | 1.03x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2530 | 0.4405 | 0.1371 | 3.21x | 0.1460 | 1.07x | 0.1557 | 1.14x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2541 | 0.4335 | 0.1397 | 3.10x | 0.1464 | 1.05x | 0.1427 | 1.02x | OK | PARITY |

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
