# CASTLE attention forward (non-causal, f16 in, f32 out), NVIDIA RTX A6000, 2026-09-14

BH = 32 (batch x heads). Same-method timing (median of 3 batches of 20 back-to-back launches). CASTLE: FlashAttention-2
structure through the LEGO/CASTLE MLIR pipeline, K/V shared tiles use solver-synthesized layouts for the exact head
dimension (no padding); best of BLOCK_M x BLOCK_N x stages in {64x32s2, 64x64s2, 128x32s2, 128x64s2, 64x64s3}.
Baselines: PyTorch SDPA backends (flash = FlashAttention-2, which pads head dims 80->96 and 112->128; cuDNN; memory-efficient).

| N | D | flash ms | cuDNN ms | efficient ms | CASTLE ms (TF) | CASTLE config | flash/CASTLE | cuDNN/CASTLE |
|---|---|---|---|---|---|---|---|---|
| 1024 | 64 | 0.109 | 0.108 | 0.149 | 0.103 (83.7) | 64x32s2 | 1.06 | 1.05 |
| 1024 | 80 | 0.160 | 0.199 | 0.293 | 0.134 (80.0) | 64x32s2 | 1.19 | 1.48 |
| 1024 | 96 | 0.147 | 0.198 | 0.298 | 0.159 (81.2) | 64x32s2 | 0.93 | 1.25 |
| 1024 | 112 | 0.213 | 0.203 | 0.321 | 0.184 (81.7) | 64x32s2 | 1.16 | 1.10 |
| 1024 | 128 | 0.204 | 0.204 | 0.322 | 0.206 (83.3) | 64x32s2 | 0.99 | 0.99 |
| 2048 | 64 | 0.346 | 0.388 | 0.564 | 0.349 (98.4) | 64x32s2 | 0.99 | 1.11 |
| 2048 | 80 | 0.537 | 0.649 | 1.091 | 0.466 (92.1) | 64x32s2 | 1.15 | 1.39 |
| 2048 | 96 | 0.537 | 0.650 | 1.129 | 0.568 (90.7) | 64x32s2 | 0.95 | 1.14 |
| 2048 | 112 | 0.737 | 0.647 | 1.249 | 0.668 (90.1) | 64x32s2 | 1.10 | 0.97 |
| 2048 | 128 | 0.743 | 0.664 | 1.248 | 0.764 (89.9) | 64x32s2 | 0.97 | 0.87 |
| 4096 | 64 | 1.404 | 1.423 | 2.259 | 1.541 (89.2) | 128x32s2 | 0.91 | 0.92 |
| 4096 | 80 | 1.971 | 2.576 | 4.141 | 1.982 (86.7) | 64x32s2 | 0.99 | 1.30 |
| 4096 | 96 | 1.950 | 2.625 | 4.261 | 2.163 (95.3) | 128x64s2 | 0.90 | 1.21 |
| 4096 | 112 | 2.757 | 2.692 | 4.574 | 2.708 (88.8) | 64x32s2 | 1.02 | 0.99 |
| 4096 | 128 | 2.706 | 2.738 | 4.679 | 3.038 (90.5) | 128x64s2 | 0.89 | 0.90 |
