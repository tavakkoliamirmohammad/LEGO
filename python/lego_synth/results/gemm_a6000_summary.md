# CASTLE GEMM benchmark (f16 in, f32 accumulate), NVIDIA RTX A6000 (sm_86), 2026-09-14

Same-method timing: median over 3 batches of 30 back-to-back launches. CASTLE kernels: cp.async + ldmatrix + mma.sync
through the LEGO/CASTLE MLIR pipeline; shared tiles use solver-synthesized conflict-free layouts per tile config;
best of {128x64,128x128,128x256,256x128} x warps x stages {2,3} x grouped block ordering {off, 8} (v5). Triton: tutorial matmul with autotune over 10 configs. cuBLAS via torch.matmul.

| M x N x K | cuBLAS ms (TF) | Triton autotuned ms (TF) | CASTLE ms (TF) | CASTLE config (sSTAGES gGROUP) | Triton/CASTLE | cuBLAS/CASTLE |
|---|---|---|---|---|---|---|
| 2048x2048x2048 | 0.171 (100.6) | 0.177 (97.2) | 0.200 (86.0) | 128x128x32w4x2s3g0 | 0.89 | 0.86 |
| 4096x4096x4096 | 1.289 (106.6) | 1.498 (91.7) | 1.446 (95.0) | 128x128x32w4x2s3g8 | 1.04 | 0.89 |
| 4096x6144x4096 | 2.164 (95.3) | 2.179 (94.6) | 2.195 (93.9) | 128x128x32w4x2s3g8 | 0.99 | 0.99 |
| 4096x14336x4096 | 4.187 (114.9) | 5.056 (95.1) | 5.049 (95.3) | 128x128x32w4x2s3g8 | 1.00 | 0.83 |
| 4096x4096x14336 | 4.752 (101.2) | 5.030 (95.6) | 5.116 (94.0) | 128x256x32w2x4s2g0 | 0.98 | 0.93 |
| 4096x11008x4096 | 3.298 (112.0) | 3.870 (95.5) | 3.892 (94.9) | 128x128x32w4x2s3g8 | 0.99 | 0.85 |
| 4096x4096x11008 | 3.633 (101.7) | 3.863 (95.6) | 3.962 (93.2) | 128x256x32w2x4s2g0 | 0.97 | 0.92 |
| 4096x18944x3584 | 4.863 (114.4) | 6.061 (91.8) | 6.060 (91.8) | 128x256x32w2x4s2g8 | 1.00 | 0.80 |
| 4096x3584x18944 | 5.395 (103.1) | 5.742 (96.9) | 5.768 (96.4) | 256x128x32w4x2s2g0 | 1.00 | 0.94 |

Layout ablation (same kernel, 4096^3, 128x128x32 w4x2 s3): naive row-major 2.815 ms (48.8 TF) | padded +8 1.524 ms (90.2 TF, +12% smem) | synthesized swizzle 1.492 ms (92.1 TF). See gemm_layout_ablation_a6000.md.
