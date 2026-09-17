### GEMM layout ablation, 4096x4096x4096, cfg 128x128x32w4x2s3, RTX A6000

| shared layouts (A, B) | predicted ldmatrix conflicts per tile (A+B) | footprint (A+B, KB/stage) | ms | TFLOP/s | correct |
|---|---|---|---|---|---|
| naive: RowMajor(rows=128, cols=32, pad=0), RowMajor(rows=32, cols=128, pad=0) | 640 | 16 | 2.815 | 48.8 | True |
| pad8: RowMajor(rows=128, cols=32, pad=8), RowMajor(rows=32, cols=128, pad=8) | 64 | 18 | 1.524 | 90.2 | True |
| synthesized: XorSwizzle(rows=128, cols=32, shift=1, mask=3, unit=8), XorSwizzle(rows=32, cols=128, shift=0, mask=7, unit=8) | 0 | 16 | 1.492 | 92.1 | True |
