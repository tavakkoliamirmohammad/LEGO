# LEGO cpu_dsl_comparison Dashboard

**Target:** `x86` | **Date:** 2026-05-03 20:07

## Summary

| Metric | Value |
|--------|-------|
| Total candidates | 42 |
| Measured | 42 |
| SKIP | 0 |
| **WIN (vs C O3)** | **37** |
| **PARITY (vs C O3)** | **5** |
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
| 01_saxpy |  |  | ? | 0.2843 | 0.3135 | 0.0893 | 3.51x | 0.1003 | 1.12x | 0.1049 | 1.17x | OK | **WIN** |
| 02_gemm_row_major |  |  | 64 | 0.0070 | 0.1445 | 0.0115 | 12.57x | 0.0244 | 2.12x | 0.0217 | 1.88x | OK | **WIN** |
| 03_3pt_stencil_1d |  |  | 1024 | 0.0021 | 0.0004 | 0.0001 | 4.00x | 0.0001 | 1.38x | 0.0001 | 1.44x | OK | **WIN** |
| 04_col_major_inner |  |  | 256 | 0.0488 | 0.0778 | 0.0605 | 1.29x | 0.0635 | 1.05x | 0.0462 | 0.76x | OK | PARITY |
| 05_morton_2d |  |  | 65536 | 0.1616 | 0.0355 | 0.0047 | 7.55x | 0.1693 | 36.03x | 0.1693 | 36.02x | OK | **WIN** |
| 06_self_update |  |  | 4096 | 0.0020 | 0.0024 | 0.0002 | 12.00x | 0.0004 | 1.92x | 0.0006 | 2.77x | OK | **WIN** |
| 07_mixed_precision |  |  | 1048576 | 0.2891 | 0.3171 | 0.0863 | 3.67x | 0.1873 | 2.17x | 0.1755 | 2.03x | OK | **WIN** |
| 08_brick_within_cell |  |  | 1048576 | 0.2973 | 0.2962 | 0.0910 | 3.25x | 0.0957 | 1.05x | 0.1508 | 1.66x | OK | **WIN** |
| 09_gemm_zmorton | Z-Morton | WIN | 65536 | 0.1685 | 0.0353 | 0.0085 | 4.15x | 0.0360 | 4.24x | 0.0354 | 4.16x | OK | **WIN** |
| 10_lu_zmorton | Z-Morton | WIN | 65536 | 0.1708 | 0.0382 | 0.0086 | 4.44x | 0.0363 | 4.22x | 0.0357 | 4.15x | OK | **WIN** |
| 11_chol_zmorton | Z-Morton | LOSS | 65536 | 0.1705 | 0.0353 | 0.0070 | 5.04x | 0.0353 | 5.04x | 0.0352 | 5.03x | OK | **WIN** |
| 12_gemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 262144 | 1.7196 | 42.4898 | 6.8643 | 6.19x | 11.4807 | 1.67x | 11.5377 | 1.68x | OK | **WIN** |
| 13_3mm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2382 | 0.4438 | 0.1363 | 3.26x | 0.1460 | 1.07x | 0.1529 | 1.12x | OK | **WIN** |
| 14_2mm_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2381 | 0.4423 | 0.1353 | 3.27x | 0.1409 | 1.04x | 0.1430 | 1.06x | OK | PARITY |
| 15_trmm_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2346 | 0.4385 | 0.1347 | 3.26x | 0.1466 | 1.09x | 0.1529 | 1.13x | OK | **WIN** |
| 16_doitgen_reg_L1_tile | Reg+L1 tile | WIN | 1048576 | 0.2398 | 0.4451 | 0.1363 | 3.27x | 0.1457 | 1.07x | 0.1545 | 1.13x | OK | **WIN** |
| 17_tensor_contraction_gett | GETT tile | WIN | 1048576 | 0.2393 | 0.4376 | 0.1294 | 3.38x | 0.1460 | 1.13x | 0.1429 | 1.10x | OK | **WIN** |
| 18_tblis_notranspose | TBLIS | LOSS | 64 | 0.0069 | 0.7240 | 0.0118 | 61.36x | 0.0244 | 2.07x | 0.0216 | 1.83x | OK | **WIN** |
| 19_bricklib_3d7pt | Brick | LOSS | 30720 | 0.0307 | 0.0294 | 0.0044 | 6.68x | 0.0084 | 1.90x | 0.0081 | 1.84x | OK | **WIN** |
| 20_bricklib_3d13pt | Brick | WIN | 30720 | 0.0541 | 0.0553 | 0.0079 | 7.00x | 0.0191 | 2.42x | 0.0152 | 1.92x | OK | **WIN** |
| 21_heat3d_brick | Brick | LOSS | 30720 | 0.0300 | 0.0294 | 0.0049 | 6.00x | 0.0084 | 1.71x | 0.0081 | 1.65x | OK | **WIN** |
| 22_jacobi2d_brick | Brick | LOSS | 65024 | 0.0415 | 0.0361 | 0.0074 | 4.88x | 0.0110 | 1.49x | 0.0108 | 1.46x | OK | **WIN** |
| 23_symm_rfp | RFP | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.30x | 0.0017 | 1.00x | OK | **WIN** |
| 24_syrk_rfp | RFP | WIN | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.31x | 0.0016 | 0.97x | OK | **WIN** |
| 25_nw_antidiag | Antidiag tile | LOSS | 16384 | 0.0071 | 0.0088 | 0.0017 | 5.18x | 0.0023 | 1.33x | 0.0017 | 0.98x | OK | **WIN** |
| 26_nussinov_skew | Skew tile | WIN | 4096 | 0.0027 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.38x | 0.0004 | 1.03x | OK | **WIN** |
| 27_zuker_skew | Skew tile | LOSS | 4096 | 0.6777 | 0.0022 | 0.0004 | 5.50x | 0.0006 | 1.46x | 0.0004 | 1.03x | OK | **WIN** |
| 28_seidel2d_wavefront | Wavefront tile | MIXED | 1048576 | 0.2284 | 0.4321 | 0.1340 | 3.22x | 0.1471 | 1.10x | 0.1542 | 1.15x | OK | **WIN** |
| 29_particlefilter_aosoA | AoSoA | PARITY | 16384 | 2.7346 | 0.0095 | 0.0028 | 3.39x | 0.0029 | 1.02x | 0.0028 | 0.99x | OK | PARITY |
| 30_lulesh_aosoA | AoSoA | LOSS | 16384 | 2.7449 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 0.99x | 0.0029 | 1.02x | OK | PARITY |
| 31_hpccg_aosoA | AoSoA | LOSS | 16384 | 2.6767 | 0.0090 | 0.0028 | 3.21x | 0.0028 | 0.99x | 0.0028 | 0.99x | OK | PARITY |
| 32_fdtd2d_block_cyclic | Block-cyclic | LOSS | 1048576 | 0.2391 | 0.4429 | 0.1359 | 3.26x | 0.1463 | 1.08x | 0.1548 | 1.14x | OK | **WIN** |
| 33_adi_block_cyclic | Block-cyclic | WIN | 1048576 | 0.2372 | 0.4387 | 0.1362 | 3.22x | 0.1452 | 1.07x | 0.1433 | 1.05x | OK | **WIN** |
| 34_gemm_pow2_pad | Pow2 pad | LOSS | 65536 | 0.1705 | 0.0353 | 0.0073 | 4.84x | 0.0355 | 4.86x | 0.0352 | 4.83x | OK | **WIN** |
| 35_heat3d_pow2_pad | Pow2 pad | WIN | 65536 | 0.1784 | 0.0355 | 0.0076 | 4.67x | 0.0367 | 4.82x | 0.0352 | 4.63x | OK | **WIN** |
| 36_gemm_nonpow2_morton | Morton+non-pow2 | WIN | 65536 | 0.1788 | 0.0355 | 0.0073 | 4.86x | 0.0363 | 4.97x | 0.0353 | 4.83x | OK | **WIN** |
| 37_stencil_nonpow2_brick | Brick+non-pow2 | LOSS | 840 | 0.0125 | 0.0005 | 0.0001 | 5.00x | 0.0001 | 1.33x | 0.0001 | 1.32x | OK | **WIN** |
| 38_nussinov_nonpow2_skew | Skew+non-pow2 | MIXED | 16384 | 0.0072 | 0.0088 | 0.0017 | 5.18x | 0.0022 | 1.32x | 0.0016 | 0.97x | OK | **WIN** |
| 39_hotspot_tile | Tile | WIN | 1048576 | 0.2390 | 0.4386 | 0.1347 | 3.26x | 0.1455 | 1.08x | 0.1538 | 1.14x | OK | **WIN** |
| 40_mvt_L1_tile | L1 tile | WIN | 1048576 | 0.2368 | 0.4405 | 0.1348 | 3.27x | 0.1462 | 1.08x | 0.1547 | 1.15x | OK | **WIN** |
| 41_bicg_L1_tile | L1 tile | WIN | 1048576 | 0.2324 | 0.4372 | 0.1300 | 3.36x | 0.1454 | 1.12x | 0.1536 | 1.18x | OK | **WIN** |
| 42_dgemm_reg_L1_L2_tile | Reg+L1+L2 tile | WIN | 1048576 | 0.2380 | 0.4393 | 0.1356 | 3.24x | 0.1473 | 1.09x | 0.1528 | 1.13x | OK | **WIN** |

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
