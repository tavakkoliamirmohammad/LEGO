<!-- provenance -->
Device: NVIDIA RTX A6000 (cc 8.6, 84 SMs, PTX target sm_86 +ptx78, opt-in smem 99 KB)
timestamp: 2026-09-14T15:08:55 | host: notch368 | python: 3.12.12 | torch: 2.11.0+cu128 | triton: 3.6.0 | cupy: 14.2.0 | numpy: 2.5.2 | sympy: 1.14.0 | z3: 4.16.0 | cuda_runtime: 12.8 | driver: 580.173.02 | git_commit: de7591c | git_dirty: False

seed=0 bh=32 configs=64x32s2,64x64s2,128x32s2,128x64s2,64x64s3 causal=True

  N=1024 D=64 cfg=64x32s2: 0.079 ms 54.6 TFLOP/s correct=True
  N=1024 D=64 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=1024 D=64 cfg=128x32s2: 0.082 ms 52.5 TFLOP/s correct=True
  N=1024 D=64 cfg=128x64s2: 0.089 ms 48.1 TFLOP/s correct=True
  N=1024 D=64 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=1024 D=64 BH=32 causal: SDPA flash 0.087 ms | cuDNN 0.094 ms | efficient 0.102 ms | CASTLE best ('64x32s2', 0.07869759798049927)
  N=1024 D=80 cfg=64x32s2: 0.093 ms 58.0 TFLOP/s correct=True
  N=1024 D=80 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=1024 D=80 cfg=128x32s2: 0.116 ms 46.2 TFLOP/s correct=True
  N=1024 D=80 cfg=128x64s2: 0.103 ms 52.3 TFLOP/s correct=True
  N=1024 D=80 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=1024 D=80 BH=32 causal: SDPA flash 0.103 ms | cuDNN 0.124 ms | efficient 0.175 ms | CASTLE best ('64x32s2', 0.09263520240783692)
  N=1024 D=96 cfg=64x32s2: 0.103 ms 62.8 TFLOP/s correct=True
  N=1024 D=96 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=1024 D=96 cfg=128x32s2: 0.126 ms 50.9 TFLOP/s correct=True
  N=1024 D=96 cfg=128x64s2: 0.133 ms 48.6 TFLOP/s correct=True
  N=1024 D=96 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=1024 D=96 BH=32 causal: SDPA flash 0.096 ms | cuDNN 0.126 ms | efficient 0.177 ms | CASTLE best ('64x32s2', 0.10258239507675171)
  N=1024 D=112 cfg=64x32s2: 0.125 ms 60.0 TFLOP/s correct=True
  N=1024 D=112 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=1024 D=112 cfg=128x32s2: 0.138 ms 54.3 TFLOP/s correct=True
  N=1024 D=112 cfg=128x64s2: 0.140 ms 53.9 TFLOP/s correct=True
  N=1024 D=112 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=1024 D=112 BH=32 causal: SDPA flash 0.132 ms | cuDNN 0.127 ms | efficient 0.188 ms | CASTLE best ('64x32s2', 0.12521439790725708)
  N=1024 D=128 cfg=64x32s2: 0.132 ms 64.9 TFLOP/s correct=True
  N=1024 D=128 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=1024 D=128 cfg=128x32s2: 0.159 ms 54.2 TFLOP/s correct=True
  N=1024 D=128 cfg=128x64s2: 0.161 ms 53.4 TFLOP/s correct=True
  N=1024 D=128 cfg=64x64s3: skipped (shared memory above the device opt-in limit)
N=1024 D=128 BH=32 causal: SDPA flash 0.131 ms | cuDNN 0.128 ms | efficient 0.188 ms | CASTLE best ('64x32s2', 0.1324399948120117)
  N=2048 D=64 cfg=64x32s2: 0.225 ms 76.2 TFLOP/s correct=True
  N=2048 D=64 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=2048 D=64 cfg=128x32s2: 0.240 ms 71.5 TFLOP/s correct=True
  N=2048 D=64 cfg=128x64s2: 0.256 ms 67.1 TFLOP/s correct=True
  N=2048 D=64 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=2048 D=64 BH=32 causal: SDPA flash 0.228 ms | cuDNN 0.258 ms | efficient 0.315 ms | CASTLE best ('64x32s2', 0.22539200782775878)
  N=2048 D=80 cfg=64x32s2: 0.286 ms 75.1 TFLOP/s correct=True
  N=2048 D=80 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=2048 D=80 cfg=128x32s2: 0.340 ms 63.2 TFLOP/s correct=True
  N=2048 D=80 cfg=128x64s2: 0.298 ms 72.1 TFLOP/s correct=True
  N=2048 D=80 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=2048 D=80 BH=32 causal: SDPA flash 0.316 ms | cuDNN 0.376 ms | efficient 0.604 ms | CASTLE best ('64x32s2', 0.28594560623168946)
  N=2048 D=96 cfg=64x32s2: 0.315 ms 81.7 TFLOP/s correct=True
  N=2048 D=96 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=2048 D=96 cfg=128x32s2: 0.365 ms 70.6 TFLOP/s correct=True
  N=2048 D=96 cfg=128x64s2: 0.389 ms 66.3 TFLOP/s correct=True
  N=2048 D=96 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=2048 D=96 BH=32 causal: SDPA flash 0.295 ms | cuDNN 0.377 ms | efficient 0.624 ms | CASTLE best ('64x32s2', 0.31542561054229734)
  N=2048 D=112 cfg=64x32s2: 0.391 ms 77.0 TFLOP/s correct=True
  N=2048 D=112 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=2048 D=112 cfg=128x32s2: 0.404 ms 74.4 TFLOP/s correct=True
  N=2048 D=112 cfg=128x64s2: 0.404 ms 74.5 TFLOP/s correct=True
  N=2048 D=112 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=2048 D=112 BH=32 causal: SDPA flash 0.404 ms | cuDNN 0.382 ms | efficient 0.676 ms | CASTLE best ('64x32s2', 0.39055519104003905)
  N=2048 D=128 cfg=64x32s2: 0.411 ms 83.6 TFLOP/s correct=True
  N=2048 D=128 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=2048 D=128 cfg=128x32s2: 0.463 ms 74.1 TFLOP/s correct=True
  N=2048 D=128 cfg=128x64s2: 0.472 ms 72.7 TFLOP/s correct=True
  N=2048 D=128 cfg=64x64s3: skipped (shared memory above the device opt-in limit)
N=2048 D=128 BH=32 causal: SDPA flash 0.375 ms | cuDNN 0.384 ms | efficient 0.692 ms | CASTLE best ('64x32s2', 0.41078720092773435)
  N=4096 D=64 cfg=64x32s2: 0.865 ms 79.5 TFLOP/s correct=True
  N=4096 D=64 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=4096 D=64 cfg=128x32s2: 0.874 ms 78.6 TFLOP/s correct=True
  N=4096 D=64 cfg=128x64s2: 0.956 ms 71.9 TFLOP/s correct=True
  N=4096 D=64 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=4096 D=64 BH=32 causal: SDPA flash 0.798 ms | cuDNN 0.833 ms | efficient 1.226 ms | CASTLE best ('64x32s2', 0.8648847579956055)
  N=4096 D=80 cfg=64x32s2: 1.161 ms 74.0 TFLOP/s correct=True
  N=4096 D=80 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=4096 D=80 cfg=128x32s2: 1.265 ms 67.9 TFLOP/s correct=True
  N=4096 D=80 cfg=128x64s2: 1.117 ms 76.9 TFLOP/s correct=True
  N=4096 D=80 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=4096 D=80 BH=32 causal: SDPA flash 1.209 ms | cuDNN 1.386 ms | efficient 2.189 ms | CASTLE best ('128x64s2', 1.1172112464904784)
  N=4096 D=96 cfg=64x32s2: 1.269 ms 81.2 TFLOP/s correct=True
  N=4096 D=96 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=4096 D=96 cfg=128x32s2: 1.367 ms 75.4 TFLOP/s correct=True
  N=4096 D=96 cfg=128x64s2: 1.412 ms 73.0 TFLOP/s correct=True
  N=4096 D=96 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=4096 D=96 BH=32 causal: SDPA flash 1.187 ms | cuDNN 1.408 ms | efficient 2.245 ms | CASTLE best ('64x32s2', 1.268951988220215)
  N=4096 D=112 cfg=64x32s2: 1.562 ms 77.0 TFLOP/s correct=True
  N=4096 D=112 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=4096 D=112 cfg=128x32s2: 1.542 ms 78.0 TFLOP/s correct=True
  N=4096 D=112 cfg=128x64s2: 1.520 ms 79.1 TFLOP/s correct=True
  N=4096 D=112 cfg=64x64s3: FAILED per-query-block key count must be a multiple of the pipeline depth
N=4096 D=112 BH=32 causal: SDPA flash 1.657 ms | cuDNN 1.486 ms | efficient 2.435 ms | CASTLE best ('128x64s2', 1.5201248168945312)
  N=4096 D=128 cfg=64x32s2: 1.673 ms 82.1 TFLOP/s correct=True
  N=4096 D=128 cfg=64x64s2: FAILED per-query-block key count must be a multiple of the pipeline depth
  N=4096 D=128 cfg=128x32s2: 1.714 ms 80.2 TFLOP/s correct=True
  N=4096 D=128 cfg=128x64s2: 1.713 ms 80.2 TFLOP/s correct=True
  N=4096 D=128 cfg=64x64s3: skipped (shared memory above the device opt-in limit)
N=4096 D=128 BH=32 causal: SDPA flash 1.586 ms | cuDNN 1.486 ms | efficient 2.486 ms | CASTLE best ('64x32s2', 1.6732671737670899)
