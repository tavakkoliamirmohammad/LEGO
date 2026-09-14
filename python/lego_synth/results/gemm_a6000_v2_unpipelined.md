# GEMM v2 (scf.for, no pipeline) vs cuBLAS vs Triton — RTX A6000, f16 in / f32 acc, back-to-back batch timing (2026-09-14)

```
  n=1024 cfg=64x64x32w2x2: 0.039 ms 55.4 TFLOP/s correct=True
  n=1024 cfg=128x128x32w2x2: 0.054 ms 40.0 TFLOP/s correct=True
  n=1024 cfg=128x64x32w2x2: 0.039 ms 54.8 TFLOP/s correct=True
  n=1024 cfg=128x128x32w4x2: 0.047 ms 45.8 TFLOP/s correct=True
n=1024: cuBLAS 0.024 ms (89.7 TF) | Triton fixed 0.030 ms (71.7 TF) | Triton autotuned 0.044 ms (48.8 TF) | CASTLE best ('64x64x32w2x2', 0.0387935996055603)
  n=2048 cfg=64x64x32w2x2: 0.218 ms 78.6 TFLOP/s correct=True
  n=2048 cfg=128x128x32w2x2: 0.245 ms 70.2 TFLOP/s correct=True
  n=2048 cfg=128x64x32w2x2: 0.186 ms 92.2 TFLOP/s correct=True
  n=2048 cfg=128x128x32w4x2: 0.243 ms 70.7 TFLOP/s correct=True
n=2048: cuBLAS 0.179 ms (96.0 TF) | Triton fixed 0.221 ms (77.6 TF) | Triton autotuned 0.178 ms (96.6 TF) | CASTLE best ('128x64x32w2x2', 0.18636053403218586)
  n=4096 cfg=64x64x32w2x2: 2.321 ms 59.2 TFLOP/s correct=True
  n=4096 cfg=128x128x32w2x2: 1.748 ms 78.6 TFLOP/s correct=True
  n=4096 cfg=128x64x32w2x2: 1.861 ms 73.9 TFLOP/s correct=True
  n=4096 cfg=128x128x32w4x2: 1.767 ms 77.8 TFLOP/s correct=True
n=4096: cuBLAS 1.290 ms (106.5 TF) | Triton fixed 1.536 ms (89.5 TF) | Triton autotuned 1.515 ms (90.7 TF) | CASTLE best ('128x128x32w2x2', 1.7480864206949869)
```

Shared tiles use solver-synthesized conflict-free layouts (staging problem, full tier) per tile config.
