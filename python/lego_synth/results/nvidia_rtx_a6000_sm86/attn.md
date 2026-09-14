<!-- provenance -->
Device: NVIDIA RTX A6000 (cc 8.6, 84 SMs, PTX target sm_86 +ptx78, opt-in smem 99 KB)
timestamp: 2026-09-14T15:02:00 | host: notch368 | python: 3.12.12 | torch: 2.11.0+cu128 | triton: 3.6.0 | cupy: 14.2.0 | numpy: 2.5.2 | sympy: 1.14.0 | z3: 4.16.0 | cuda_runtime: 12.8 | driver: 580.173.02 | git_commit: e514b1a | git_dirty: True

seed=0 bh=32 configs=64x32s2,64x64s2,128x32s2,128x64s2,64x64s3 causal=False

  N=1024 D=64 cfg=64x32s2: 0.103 ms 83.8 TFLOP/s correct=True
  N=1024 D=64 cfg=64x64s2: 0.102 ms 83.8 TFLOP/s correct=True
  N=1024 D=64 cfg=128x32s2: 0.123 ms 69.6 TFLOP/s correct=True
  N=1024 D=64 cfg=128x64s2: 0.128 ms 67.3 TFLOP/s correct=True
  N=1024 D=64 cfg=64x64s3: FAILED 
N=1024 D=64 BH=32: SDPA flash 0.109 ms | cuDNN 0.108 ms | efficient 0.149 ms | CASTLE best ('64x64s2', 0.10249119997024536)
  N=1024 D=80 cfg=64x32s2: 0.134 ms 80.1 TFLOP/s correct=True
  N=1024 D=80 cfg=64x64s2: 0.161 ms 66.7 TFLOP/s correct=True
  N=1024 D=80 cfg=128x32s2: 0.160 ms 67.2 TFLOP/s correct=True
  N=1024 D=80 cfg=128x64s2: 0.161 ms 66.9 TFLOP/s correct=True
  N=1024 D=80 cfg=64x64s3: FAILED 
N=1024 D=80 BH=32: SDPA flash 0.155 ms | cuDNN 0.199 ms | efficient 0.294 ms | CASTLE best ('64x32s2', 0.1341215968132019)
  N=1024 D=96 cfg=64x32s2: 0.156 ms 82.5 TFLOP/s correct=True
  N=1024 D=96 cfg=64x64s2: 0.188 ms 68.5 TFLOP/s correct=True
  N=1024 D=96 cfg=128x32s2: 0.219 ms 58.8 TFLOP/s correct=True
  N=1024 D=96 cfg=128x64s2: 0.186 ms 69.1 TFLOP/s correct=True
  N=1024 D=96 cfg=64x64s3: FAILED 
N=1024 D=96 BH=32: SDPA flash 0.148 ms | cuDNN 0.199 ms | efficient 0.296 ms | CASTLE best ('64x32s2', 0.15627360343933105)
  N=1024 D=112 cfg=64x32s2: 0.189 ms 79.6 TFLOP/s correct=True
  N=1024 D=112 cfg=64x64s2: 0.227 ms 66.3 TFLOP/s correct=True
  N=1024 D=112 cfg=128x32s2: 0.224 ms 67.2 TFLOP/s correct=True
  N=1024 D=112 cfg=128x64s2: 0.218 ms 69.0 TFLOP/s correct=True
  N=1024 D=112 cfg=64x64s3: FAILED 
N=1024 D=112 BH=32: SDPA flash 0.212 ms | cuDNN 0.203 ms | efficient 0.323 ms | CASTLE best ('64x32s2', 0.18873759508132934)
  N=1024 D=128 cfg=64x32s2: 0.205 ms 83.8 TFLOP/s correct=True
  N=1024 D=128 cfg=64x64s2: 0.241 ms 71.4 TFLOP/s correct=True
  N=1024 D=128 cfg=128x32s2: 0.250 ms 68.7 TFLOP/s correct=True
  N=1024 D=128 cfg=128x64s2: 0.240 ms 71.6 TFLOP/s correct=True
  N=1024 D=128 cfg=64x64s3: skipped (shared memory above the device opt-in limit)
N=1024 D=128 BH=32: SDPA flash 0.205 ms | cuDNN 0.204 ms | efficient 0.315 ms | CASTLE best ('64x32s2', 0.20500800609588624)
  N=2048 D=64 cfg=64x32s2: 0.351 ms 98.0 TFLOP/s correct=True
  N=2048 D=64 cfg=64x64s2: 0.365 ms 94.2 TFLOP/s correct=True
  N=2048 D=64 cfg=128x32s2: 0.403 ms 85.2 TFLOP/s correct=True
  N=2048 D=64 cfg=128x64s2: 0.409 ms 84.1 TFLOP/s correct=True
  N=2048 D=64 cfg=64x64s3: FAILED 
N=2048 D=64 BH=32: SDPA flash 0.363 ms | cuDNN 0.388 ms | efficient 0.554 ms | CASTLE best ('64x32s2', 0.3506495952606201)
  N=2048 D=80 cfg=64x32s2: 0.460 ms 93.3 TFLOP/s correct=True
  N=2048 D=80 cfg=64x64s2: 0.581 ms 73.9 TFLOP/s correct=True
  N=2048 D=80 cfg=128x32s2: 0.540 ms 79.5 TFLOP/s correct=True
  N=2048 D=80 cfg=128x64s2: 0.499 ms 86.1 TFLOP/s correct=True
  N=2048 D=80 cfg=64x64s3: FAILED 
N=2048 D=80 BH=32: SDPA flash 0.499 ms | cuDNN 0.652 ms | efficient 1.110 ms | CASTLE best ('64x32s2', 0.4603568077087402)
  N=2048 D=96 cfg=64x32s2: 0.536 ms 96.2 TFLOP/s correct=True
  N=2048 D=96 cfg=64x64s2: 0.655 ms 78.7 TFLOP/s correct=True
  N=2048 D=96 cfg=128x32s2: 0.713 ms 72.3 TFLOP/s correct=True
  N=2048 D=96 cfg=128x64s2: 0.574 ms 89.8 TFLOP/s correct=True
  N=2048 D=96 cfg=64x64s3: FAILED 
N=2048 D=96 BH=32: SDPA flash 0.479 ms | cuDNN 0.651 ms | efficient 1.145 ms | CASTLE best ('64x32s2', 0.5357215881347657)
  N=2048 D=112 cfg=64x32s2: 0.663 ms 90.7 TFLOP/s correct=True
  N=2048 D=112 cfg=64x64s2: 0.834 ms 72.1 TFLOP/s correct=True
  N=2048 D=112 cfg=128x32s2: 0.743 ms 80.9 TFLOP/s correct=True
  N=2048 D=112 cfg=128x64s2: 0.686 ms 87.7 TFLOP/s correct=True
  N=2048 D=112 cfg=64x64s3: FAILED 
N=2048 D=112 BH=32: SDPA flash 0.755 ms | cuDNN 0.658 ms | efficient 1.226 ms | CASTLE best ('64x32s2', 0.6630752086639404)
  N=2048 D=128 cfg=64x32s2: 0.761 ms 90.3 TFLOP/s correct=True
  N=2048 D=128 cfg=64x64s2: 0.895 ms 76.8 TFLOP/s correct=True
  N=2048 D=128 cfg=128x32s2: 0.821 ms 83.7 TFLOP/s correct=True
  N=2048 D=128 cfg=128x64s2: 0.771 ms 89.1 TFLOP/s correct=True
  N=2048 D=128 cfg=64x64s3: skipped (shared memory above the device opt-in limit)
N=2048 D=128 BH=32: SDPA flash 0.737 ms | cuDNN 0.659 ms | efficient 1.248 ms | CASTLE best ('64x32s2', 0.7613647937774658)
  N=4096 D=64 cfg=64x32s2: 1.519 ms 90.5 TFLOP/s correct=True
  N=4096 D=64 cfg=64x64s2: 1.542 ms 89.1 TFLOP/s correct=True
  N=4096 D=64 cfg=128x32s2: 1.568 ms 87.7 TFLOP/s correct=True
  N=4096 D=64 cfg=128x64s2: 1.544 ms 89.0 TFLOP/s correct=True
  N=4096 D=64 cfg=64x64s3: FAILED 
N=4096 D=64 BH=32: SDPA flash 1.409 ms | cuDNN 1.412 ms | efficient 2.268 ms | CASTLE best ('64x32s2', 1.5193424224853516)
  N=4096 D=80 cfg=64x32s2: 1.971 ms 87.2 TFLOP/s correct=True
  N=4096 D=80 cfg=64x64s2: 2.232 ms 77.0 TFLOP/s correct=True
  N=4096 D=80 cfg=128x32s2: 2.048 ms 83.9 TFLOP/s correct=True
  N=4096 D=80 cfg=128x64s2: 1.927 ms 89.2 TFLOP/s correct=True
  N=4096 D=80 cfg=64x64s3: FAILED 
N=4096 D=80 BH=32: SDPA flash 1.973 ms | cuDNN 2.580 ms | efficient 4.150 ms | CASTLE best ('128x64s2', 1.9268016815185547)
  N=4096 D=96 cfg=64x32s2: 2.264 ms 91.1 TFLOP/s correct=True
  N=4096 D=96 cfg=64x64s2: 2.586 ms 79.7 TFLOP/s correct=True
  N=4096 D=96 cfg=128x32s2: 2.600 ms 79.3 TFLOP/s correct=True
  N=4096 D=96 cfg=128x64s2: 2.214 ms 93.1 TFLOP/s correct=True
  N=4096 D=96 cfg=64x64s3: FAILED 
N=4096 D=96 BH=32: SDPA flash 1.951 ms | cuDNN 2.606 ms | efficient 4.255 ms | CASTLE best ('128x64s2', 2.2137760162353515)
  N=4096 D=112 cfg=64x32s2: 2.777 ms 86.6 TFLOP/s correct=True
  N=4096 D=112 cfg=64x64s2: 3.183 ms 75.6 TFLOP/s correct=True
  N=4096 D=112 cfg=128x32s2: 2.694 ms 89.3 TFLOP/s correct=True
  N=4096 D=112 cfg=128x64s2: 2.579 ms 93.3 TFLOP/s correct=True
  N=4096 D=112 cfg=64x64s3: FAILED 
N=4096 D=112 BH=32: SDPA flash 2.760 ms | cuDNN 2.697 ms | efficient 4.614 ms | CASTLE best ('128x64s2', 2.578688049316406)
  N=4096 D=128 cfg=64x32s2: 3.021 ms 91.0 TFLOP/s correct=True
  N=4096 D=128 cfg=64x64s2: 3.385 ms 81.2 TFLOP/s correct=True
  N=4096 D=128 cfg=128x32s2: 3.015 ms 91.2 TFLOP/s correct=True
  N=4096 D=128 cfg=128x64s2: 2.853 ms 96.3 TFLOP/s correct=True
  N=4096 D=128 cfg=64x64s3: skipped (shared memory above the device opt-in limit)
N=4096 D=128 BH=32: SDPA flash 2.706 ms | cuDNN 2.709 ms | efficient 4.679 ms | CASTLE best ('128x64s2', 2.853059196472168)
