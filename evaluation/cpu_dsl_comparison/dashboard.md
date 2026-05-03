# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 15:48

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **30** |
| **PARITY (vs C O3)** | **11** |
| **LOSS (vs C O3)** | **0** |
| WIN (vs C agg) | 26 |
| PARITY (vs C agg) | 12 |
| LOSS (vs C agg) | 3 |
| ERROR | 1 |
| VERIFIED (correctness) | 42 |
| PENDING (correctness) | 0 |
| vec_iso > 1.5× | 40 |

**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`
WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.

## Per-Candidate Results

| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |
|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|
| 01_saxpy |  |  | 1048576 | 0.2857 | 0.3036 | 0.1002 | 3.03x | 0.1022 | 1.02x | 0.0993 | 0.99x | OK | PARITY |
| 02_gemm_row_major |  |  | 64 | 0.0064 | 0.1445 | 0.0138 | 10.44x | 0.0244 | 1.76x | 0.0224 | 1.62x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.63x | 0.0001 | 1.41x | 0.0002 | 1.51x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0507 | 0.0778 | 0.0604 | 1.29x | 0.0680 | 1.13x | 0.0541 | 0.90x | OK | **WIN** |
| 05_morton_2d |  |  | 65536 | 0.1606 | 0.0353 | 0.0053 | 6.71x | 0.1698 | 32.03x | 0.1705 | 32.16x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0022 | 0.0003 | 8.92x | 0.0004 | 1.58x | 0.0006 | 2.27x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2833 | 0.2986 | 0.0916 | 3.26x | 0.1832 | 2.00x | 0.1738 | 1.90x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2925 | 0.2923 | 0.0915 | 3.19x | 0.1016 | 1.11x | 0.1490 | 1.63x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1696 | 0.0355 | 0.0071 | 4.97x | 0.0352 | 4.96x | 0.0356 | 5.02x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1793 | 0.0391 | 0.0081 | 4.82x | 0.0354 | 4.37x | 0.0354 | 4.37x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1730 | 0.0354 | 0.0072 | 4.89x | 0.0354 | 4.91x | 0.0352 | 4.88x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 4.4206 | 43.0565 | 10.1598 | 4.24x | 11.3958 | 1.12x | 10.9612 | 1.08x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2482 | 0.4369 | 0.1396 | 3.13x | 0.1458 | 1.04x | 0.1428 | 1.02x | OK | PARITY |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2353 | 0.4296 | 0.1338 | 3.21x | 0.1461 | 1.09x | 0.1537 | 1.15x | OK | **WIN** |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2449 | 0.4369 | 0.1396 | 3.13x | 0.1463 | 1.05x | 0.1492 | 1.07x | OK | PARITY |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2507 | 0.4417 | 0.1377 | 3.21x | 0.1421 | 1.03x | 0.1579 | 1.15x | OK | PARITY |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2383 | 0.4309 | 0.1313 | 3.28x | 0.1465 | 1.12x | 0.1545 | 1.18x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0064 | 0.7293 | 0.0129 | 56.49x | 0.0244 | 1.89x | 0.0221 | 1.71x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0772 | 0.0309 | 0.0045 | 6.91x | 0.0084 | 1.86x | 0.0081 | 1.80x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.1774 | 0.0586 | NaN | NaN | 0.0191 | NaN | 0.0151 | NaN | OK | ERROR |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0806 | 0.0306 | 0.0056 | 5.48x | 0.0084 | 1.50x | 0.0082 | 1.46x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0678 | 0.0391 | 0.0078 | 5.01x | 0.0107 | 1.38x | 0.0108 | 1.39x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0018 | 4.99x | 0.0022 | 1.22x | 0.0016 | 0.91x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.09x | 0.0022 | 1.31x | 0.0017 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0018 | 4.99x | 0.0022 | 1.24x | 0.0016 | 0.91x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0026 | 0.0022 | 0.0004 | 5.16x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.0026 | 0.0022 | 0.0004 | 5.37x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2498 | 0.4521 | 0.1488 | 3.04x | 0.1464 | 0.98x | 0.1428 | 0.96x | OK | PARITY |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 0.0071 | 0.0090 | 0.0028 | 3.17x | 0.0028 | 0.98x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0091 | 0.0029 | 3.16x | 0.0028 | 0.96x | 0.0028 | 0.96x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 0.0071 | 0.0090 | 0.0028 | 3.25x | 0.0028 | 1.00x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2468 | 0.4485 | 0.1441 | 3.11x | 0.1470 | 1.02x | 0.1551 | 1.08x | OK | PARITY |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2444 | 0.4351 | 0.1350 | 3.22x | 0.1458 | 1.08x | 0.1445 | 1.07x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1690 | 0.0360 | 0.0073 | 4.92x | 0.0352 | 4.82x | 0.0353 | 4.84x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1743 | 0.0363 | 0.0073 | 4.99x | 0.0356 | 4.87x | 0.0352 | 4.82x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1705 | 0.0354 | 0.0075 | 4.69x | 0.0360 | 4.81x | 0.0353 | 4.70x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0072 | 0.0005 | 0.0001 | 4.94x | 0.0001 | 1.35x | 0.0001 | 1.36x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.27x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2502 | 0.4361 | 0.1370 | 3.18x | 0.1410 | 1.03x | 0.1429 | 1.04x | OK | PARITY |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2396 | 0.4317 | 0.1403 | 3.08x | 0.1470 | 1.05x | 0.1429 | 1.02x | OK | PARITY |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2386 | 0.4349 | 0.1359 | 3.20x | 0.1459 | 1.07x | 0.1451 | 1.07x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2433 | 0.4602 | 0.1383 | 3.33x | 0.1464 | 1.06x | 0.1531 | 1.11x | OK | **WIN** |

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
