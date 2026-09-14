  1024x1024x1024 cfg=64x64x32w2x2s2: 0.038 ms 57.1 TFLOP/s correct=True
  1024x1024x1024 cfg=64x64x32w2x2s4: 0.039 ms 55.3 TFLOP/s correct=True
  1024x1024x1024 cfg=128x128x32w2x2s2: 0.037 ms 58.7 TFLOP/s correct=True
  1024x1024x1024 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  1024x1024x1024 cfg=128x64x32w2x2s2: 0.037 ms 58.8 TFLOP/s correct=True
  1024x1024x1024 cfg=128x64x32w2x2s4: 0.037 ms 57.4 TFLOP/s correct=True
  1024x1024x1024 cfg=128x128x32w4x2s2: 0.036 ms 59.6 TFLOP/s correct=True
  1024x1024x1024 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  1024x1024x1024 cfg=128x256x32w2x4s2: 0.051 ms 41.8 TFLOP/s correct=True
  1024x1024x1024 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  1024x1024x1024 cfg=256x128x32w4x2s2: 0.053 ms 40.5 TFLOP/s correct=True
  1024x1024x1024 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
1024x1024x1024: cuBLAS 0.024 ms (89.5 TF) | Triton fixed 0.033 ms (65.0 TF) | Triton autotuned 0.042 ms (50.7 TF) | CASTLE best ('128x128x32w4x2s2', 0.036008532842000326)
  2048x2048x2048 cfg=64x64x32w2x2s2: 0.230 ms 74.8 TFLOP/s correct=True
  2048x2048x2048 cfg=64x64x32w2x2s4: 0.249 ms 69.1 TFLOP/s correct=True
  2048x2048x2048 cfg=128x128x32w2x2s2: 0.213 ms 80.6 TFLOP/s correct=True
  2048x2048x2048 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  2048x2048x2048 cfg=128x64x32w2x2s2: 0.205 ms 83.8 TFLOP/s correct=True
  2048x2048x2048 cfg=128x64x32w2x2s4: 0.216 ms 79.6 TFLOP/s correct=True
  2048x2048x2048 cfg=128x128x32w4x2s2: 0.205 ms 83.7 TFLOP/s correct=True
  2048x2048x2048 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  2048x2048x2048 cfg=128x256x32w2x4s2: 0.215 ms 80.0 TFLOP/s correct=True
  2048x2048x2048 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  2048x2048x2048 cfg=256x128x32w4x2s2: 0.219 ms 78.5 TFLOP/s correct=True
  2048x2048x2048 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
2048x2048x2048: cuBLAS 0.172 ms (99.8 TF) | Triton fixed 0.224 ms (76.6 TF) | Triton autotuned 0.182 ms (94.5 TF) | CASTLE best ('128x64x32w2x2s2', 0.20493440628051757)
  4096x4096x4096 cfg=64x64x32w2x2s2: 2.933 ms 46.9 TFLOP/s correct=True
  4096x4096x4096 cfg=64x64x32w2x2s4: 3.018 ms 45.5 TFLOP/s correct=True
  4096x4096x4096 cfg=128x128x32w2x2s2: 1.537 ms 89.4 TFLOP/s correct=True
  4096x4096x4096 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  4096x4096x4096 cfg=128x64x32w2x2s2: 2.118 ms 64.9 TFLOP/s correct=True
  4096x4096x4096 cfg=128x64x32w2x2s4: 1.976 ms 69.6 TFLOP/s correct=True
  4096x4096x4096 cfg=128x128x32w4x2s2: 1.498 ms 91.8 TFLOP/s correct=True
  4096x4096x4096 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  4096x4096x4096 cfg=128x256x32w2x4s2: 1.514 ms 90.8 TFLOP/s correct=True
  4096x4096x4096 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  4096x4096x4096 cfg=256x128x32w4x2s2: 1.537 ms 89.4 TFLOP/s correct=True
  4096x4096x4096 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
4096x4096x4096: cuBLAS 1.290 ms (106.6 TF) | Triton fixed 1.537 ms (89.4 TF) | Triton autotuned 1.509 ms (91.1 TF) | CASTLE best ('128x128x32w4x2s2', 1.4978645324707032)
  4096x6144x4096 cfg=64x64x32w2x2s2: 5.525 ms 37.3 TFLOP/s correct=True
  4096x6144x4096 cfg=64x64x32w2x2s4: 5.750 ms 35.9 TFLOP/s correct=True
  4096x6144x4096 cfg=128x128x32w2x2s2: 2.547 ms 80.9 TFLOP/s correct=True
  4096x6144x4096 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  4096x6144x4096 cfg=128x64x32w2x2s2: 3.272 ms 63.0 TFLOP/s correct=True
  4096x6144x4096 cfg=128x64x32w2x2s4: 3.247 ms 63.5 TFLOP/s correct=True
  4096x6144x4096 cfg=128x128x32w4x2s2: 2.328 ms 88.6 TFLOP/s correct=True
  4096x6144x4096 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  4096x6144x4096 cfg=128x256x32w2x4s2: 2.302 ms 89.6 TFLOP/s correct=True
  4096x6144x4096 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  4096x6144x4096 cfg=256x128x32w4x2s2: 2.291 ms 90.0 TFLOP/s correct=True
  4096x6144x4096 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
4096x6144x4096: cuBLAS 2.145 ms (96.1 TF) | Triton fixed 2.695 ms (76.5 TF) | Triton autotuned 2.200 ms (93.7 TF) | CASTLE best ('256x128x32w4x2s2', 2.2913663228352865)
  4096x14336x4096 cfg=64x64x32w2x2s2: 13.586 ms 35.4 TFLOP/s correct=True
  4096x14336x4096 cfg=64x64x32w2x2s4: 13.466 ms 35.7 TFLOP/s correct=True
  4096x14336x4096 cfg=128x128x32w2x2s2: 6.775 ms 71.0 TFLOP/s correct=True
  4096x14336x4096 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  4096x14336x4096 cfg=128x64x32w2x2s2: 7.423 ms 64.8 TFLOP/s correct=True
  4096x14336x4096 cfg=128x64x32w2x2s4: 7.625 ms 63.1 TFLOP/s correct=True
  4096x14336x4096 cfg=128x128x32w4x2s2: 6.792 ms 70.8 TFLOP/s correct=True
  4096x14336x4096 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  4096x14336x4096 cfg=128x256x32w2x4s2: 5.477 ms 87.8 TFLOP/s correct=True
  4096x14336x4096 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  4096x14336x4096 cfg=256x128x32w4x2s2: 5.285 ms 91.0 TFLOP/s correct=True
  4096x14336x4096 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
4096x14336x4096: cuBLAS 4.159 ms (115.6 TF) | Triton fixed 8.188 ms (58.7 TF) | Triton autotuned 5.051 ms (95.2 TF) | CASTLE best ('256x128x32w4x2s2', 5.285017395019532)
  4096x4096x14336 cfg=64x64x32w2x2s2: 14.816 ms 32.5 TFLOP/s correct=True
  4096x4096x14336 cfg=64x64x32w2x2s4: 13.295 ms 36.2 TFLOP/s correct=True
  4096x4096x14336 cfg=128x128x32w2x2s2: 6.135 ms 78.4 TFLOP/s correct=True
  4096x4096x14336 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  4096x4096x14336 cfg=128x64x32w2x2s2: 9.936 ms 48.4 TFLOP/s correct=True
  4096x4096x14336 cfg=128x64x32w2x2s4: 7.578 ms 63.5 TFLOP/s correct=True
  4096x4096x14336 cfg=128x128x32w4x2s2: 5.993 ms 80.3 TFLOP/s correct=True
  4096x4096x14336 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  4096x4096x14336 cfg=128x256x32w2x4s2: 5.116 ms 94.0 TFLOP/s correct=True
  4096x4096x14336 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  4096x4096x14336 cfg=256x128x32w4x2s2: 5.167 ms 93.1 TFLOP/s correct=True
  4096x4096x14336 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
4096x4096x14336: cuBLAS 4.716 ms (102.0 TF) | Triton fixed 5.526 ms (87.1 TF) | Triton autotuned 5.035 ms (95.5 TF) | CASTLE best ('128x256x32w2x4s2', 5.116412862141927)
  4096x11008x4096 cfg=64x64x32w2x2s2: 10.408 ms 35.5 TFLOP/s correct=True
  4096x11008x4096 cfg=64x64x32w2x2s4: 10.395 ms 35.5 TFLOP/s correct=True
  4096x11008x4096 cfg=128x128x32w2x2s2: 5.192 ms 71.1 TFLOP/s correct=True
  4096x11008x4096 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  4096x11008x4096 cfg=128x64x32w2x2s2: 5.752 ms 64.2 TFLOP/s correct=True
  4096x11008x4096 cfg=128x64x32w2x2s4: 5.860 ms 63.0 TFLOP/s correct=True
  4096x11008x4096 cfg=128x128x32w4x2s2: 5.170 ms 71.4 TFLOP/s correct=True
  4096x11008x4096 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  4096x11008x4096 cfg=128x256x32w2x4s2: 4.118 ms 89.7 TFLOP/s correct=True
  4096x11008x4096 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  4096x11008x4096 cfg=256x128x32w4x2s2: 4.058 ms 91.0 TFLOP/s correct=True
  4096x11008x4096 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
4096x11008x4096: cuBLAS 3.327 ms (111.0 TF) | Triton fixed 5.914 ms (62.5 TF) | Triton autotuned 3.864 ms (95.6 TF) | CASTLE best ('256x128x32w4x2s2', 4.058419291178385)
  4096x4096x11008 cfg=64x64x32w2x2s2: 11.040 ms 33.5 TFLOP/s correct=True
  4096x4096x11008 cfg=64x64x32w2x2s4: 8.948 ms 41.3 TFLOP/s correct=True
  4096x4096x11008 cfg=128x128x32w2x2s2: 4.285 ms 86.2 TFLOP/s correct=True
  4096x4096x11008 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  4096x4096x11008 cfg=128x64x32w2x2s2: 7.162 ms 51.6 TFLOP/s correct=True
  4096x4096x11008 cfg=128x64x32w2x2s4: 5.519 ms 66.9 TFLOP/s correct=True
  4096x4096x11008 cfg=128x128x32w4x2s2: 4.084 ms 90.4 TFLOP/s correct=True
  4096x4096x11008 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  4096x4096x11008 cfg=128x256x32w2x4s2: 3.963 ms 93.2 TFLOP/s correct=True
  4096x4096x11008 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  4096x4096x11008 cfg=256x128x32w4x2s2: 4.010 ms 92.1 TFLOP/s correct=True
  4096x4096x11008 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
4096x4096x11008: cuBLAS 3.641 ms (101.5 TF) | Triton fixed 4.078 ms (90.6 TF) | Triton autotuned 3.851 ms (95.9 TF) | CASTLE best ('128x256x32w2x4s2', 3.9630612691243488)
  4096x18944x3584 cfg=64x64x32w2x2s2: 15.861 ms 35.1 TFLOP/s correct=True
  4096x18944x3584 cfg=64x64x32w2x2s4: 15.750 ms 35.3 TFLOP/s correct=True
  4096x18944x3584 cfg=128x128x32w2x2s2: 7.934 ms 70.1 TFLOP/s correct=True
  4096x18944x3584 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  4096x18944x3584 cfg=128x64x32w2x2s2: 8.627 ms 64.5 TFLOP/s correct=True
  4096x18944x3584 cfg=128x64x32w2x2s4: 9.115 ms 61.0 TFLOP/s correct=True
  4096x18944x3584 cfg=128x128x32w4x2s2: 8.042 ms 69.2 TFLOP/s correct=True
  4096x18944x3584 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  4096x18944x3584 cfg=128x256x32w2x4s2: 7.097 ms 78.4 TFLOP/s correct=True
  4096x18944x3584 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  4096x18944x3584 cfg=256x128x32w4x2s2: 6.166 ms 90.2 TFLOP/s correct=True
  4096x18944x3584 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
4096x18944x3584: cuBLAS 4.905 ms (113.4 TF) | Triton fixed 9.433 ms (59.0 TF) | Triton autotuned 6.073 ms (91.6 TF) | CASTLE best ('256x128x32w4x2s2', 6.165835571289063)
  4096x3584x18944 cfg=64x64x32w2x2s2: 18.232 ms 30.5 TFLOP/s correct=True
  4096x3584x18944 cfg=64x64x32w2x2s4: 16.721 ms 33.3 TFLOP/s correct=True
  4096x3584x18944 cfg=128x128x32w2x2s2: 6.693 ms 83.1 TFLOP/s correct=True
  4096x3584x18944 cfg=128x128x32w2x2s4: skipped (static shared 64 KB > 48 KB)
  4096x3584x18944 cfg=128x64x32w2x2s2: 12.416 ms 44.8 TFLOP/s correct=True
  4096x3584x18944 cfg=128x64x32w2x2s4: 8.763 ms 63.5 TFLOP/s correct=True
  4096x3584x18944 cfg=128x128x32w4x2s2: 5.979 ms 93.0 TFLOP/s correct=True
  4096x3584x18944 cfg=128x128x32w4x2s4: skipped (static shared 64 KB > 48 KB)
  4096x3584x18944 cfg=128x256x32w2x4s2: 5.772 ms 96.4 TFLOP/s correct=True
  4096x3584x18944 cfg=128x256x32w2x4s4: skipped (static shared 96 KB > 48 KB)
  4096x3584x18944 cfg=256x128x32w4x2s2: 5.777 ms 96.3 TFLOP/s correct=True
  4096x3584x18944 cfg=256x128x32w4x2s4: skipped (static shared 96 KB > 48 KB)
4096x3584x18944: cuBLAS 5.419 ms (102.6 TF) | Triton fixed 8.555 ms (65.0 TF) | Triton autotuned 5.773 ms (96.3 TF) | CASTLE best ('128x256x32w2x4s2', 5.772335815429687)
