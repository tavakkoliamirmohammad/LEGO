# CASTLE GEMM benchmark (f16 in, f32 accumulate), NVIDIA RTX A6000 (sm_86), 2026-09-14

Same-method timing: median over 3 batches of 30 back-to-back launches. CASTLE kernels: cp.async + ldmatrix + mma.sync
through the LEGO/CASTLE MLIR pipeline; shared tiles use solver-synthesized conflict-free layouts per tile config;
best of {64x64,128x64,128x128,128x256,256x128} x warps x stages {2,4}. Triton: tutorial matmul with autotune over 10 configs. cuBLAS via torch.matmul.

| M x N x K | cuBLAS ms (TF) | Triton autotuned ms (TF) | CASTLE ms (TF) | CASTLE config | Triton/CASTLE | cuBLAS/CASTLE |
|---|---|---|---|---|---|---|
| 1024x1024x1024 | 0.024 (89.5) | 0.042 (50.7) | 0.036 (59.6) | 128x128x32w4x2s2 | 1.17 | 0.67 |
| 2048x2048x2048 | 0.172 (99.8) | 0.182 (94.5) | 0.205 (83.8) | 128x64x32w2x2s2 | 0.89 | 0.84 |
| 4096x4096x4096 | 1.290 (106.6) | 1.509 (91.1) | 1.498 (91.8) | 128x128x32w4x2s2 | 1.01 | 0.86 |
| 4096x6144x4096 | 2.145 (96.1) | 2.200 (93.7) | 2.291 (90.0) | 256x128x32w4x2s2 | 0.96 | 0.94 |
| 4096x14336x4096 | 4.159 (115.6) | 5.051 (95.2) | 5.285 (91.0) | 256x128x32w4x2s2 | 0.96 | 0.79 |
| 4096x4096x14336 | 4.716 (102.0) | 5.035 (95.5) | 5.116 (94.0) | 128x256x32w2x4s2 | 0.98 | 0.92 |
| 4096x11008x4096 | 3.327 (111.0) | 3.864 (95.6) | 4.058 (91.0) | 256x128x32w4x2s2 | 0.95 | 0.82 |
| 4096x4096x11008 | 3.641 (101.5) | 3.851 (95.9) | 3.963 (93.2) | 128x256x32w2x4s2 | 0.97 | 0.92 |
| 4096x18944x3584 | 4.905 (113.4) | 6.073 (91.6) | 6.166 (90.2) | 256x128x32w4x2s2 | 0.98 | 0.80 |
| 4096x3584x18944 | 5.419 (102.6) | 5.773 (96.3) | 5.772 (96.4) | 128x256x32w2x4s2 | 1.00 | 0.94 |

Layout ablation (same kernel, 4096^3, 128x128x32 w4x2 s3): naive row-major 2.815 ms (48.8 TF) | padded +8 1.524 ms (90.2 TF, +12% smem) | synthesized swizzle 1.492 ms (92.1 TF). See gemm_layout_ablation_a6000.md.
