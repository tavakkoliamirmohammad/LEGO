### transpose_dense, elem=4B, rows=32: conflict-free footprint by tier

| width | f2 footprint (overhead, template) | affine_xor footprint (overhead, template) | full footprint (overhead, template) |
|---|---|---|---|
| 24 | 1024 (1.33x, XorTemplate) | none | 768 (1.00x, XorTemplate) |
| 32 | 1024 (1.00x, XorTemplate) | 1024 (1.00x, XorTemplate) | 1024 (1.00x, XorTemplate) |
| 40 | 2048 (1.60x, XorTemplate) | none | 1280 (1.00x, XorTemplate) |
| 48 | 2048 (1.33x, XorTemplate) | none | 1536 (1.00x, XorTemplate) |
| 64 | 2048 (1.00x, XorTemplate) | 2048 (1.00x, XorTemplate) | 2048 (1.00x, XorTemplate) |
| 80 | 4096 (1.60x, XorTemplate) | none | 2560 (1.00x, XorTemplate) |
| 96 | 4096 (1.33x, XorTemplate) | 3104 (1.01x, PaddedTemplate) | 3072 (1.00x, XorTemplate) |
| 112 | 4096 (1.14x, XorTemplate) | none | 3584 (1.00x, XorTemplate) |
| 128 | 4096 (1.00x, XorTemplate) | 4096 (1.00x, XorTemplate) | 4096 (1.00x, XorTemplate) |

solutions: 24/f2=XorSwizzle(rows=32, cols=32, shift=0, mask=31, unit=1); 24/full=XorSwizzle(rows=32, cols=24, shift=2, mask=15, unit=1); 32/f2=XorSwizzle(rows=32, cols=32, shift=0, mask=31, unit=1); 32/affine_xor=XorSwizzle(rows=32, cols=32, shift=0, mask=31, unit=1); 32/full=XorSwizzle(rows=32, cols=32, shift=0, mask=31, unit=1); 40/f2=XorSwizzle(rows=32, cols=64, shift=0, mask=31, unit=1); 40/full=XorSwizzle(rows=32, cols=40, shift=2, mask=7, unit=1); 48/f2=XorSwizzle(rows=32, cols=64, shift=0, mask=31, unit=1); 48/full=XorSwizzle(rows=32, cols=48, shift=1, mask=15, unit=1); 64/f2=XorSwizzle(rows=32, cols=64, shift=0, mask=31, unit=1); 64/affine_xor=XorSwizzle(rows=32, cols=64, shift=0, mask=31, unit=1); 64/full=XorSwizzle(rows=32, cols=64, shift=0, mask=31, unit=1); 80/f2=XorSwizzle(rows=32, cols=128, shift=0, mask=31, unit=1); 80/full=XorSwizzle(rows=32, cols=80, shift=1, mask=15, unit=1); 96/f2=XorSwizzle(rows=32, cols=128, shift=0, mask=31, unit=1); 96/affine_xor=RowMajor(rows=32, cols=96, pad=1); 96/full=XorSwizzle(rows=32, cols=96, shift=0, mask=31, unit=1); 112/f2=XorSwizzle(rows=32, cols=128, shift=0, mask=31, unit=1); 112/full=XorSwizzle(rows=32, cols=112, shift=1, mask=15, unit=1); 128/f2=XorSwizzle(rows=32, cols=128, shift=0, mask=31, unit=1); 128/affine_xor=XorSwizzle(rows=32, cols=128, shift=0, mask=31, unit=1); 128/full=XorSwizzle(rows=32, cols=128, shift=0, mask=31, unit=1)

solve times (s): 24/f2=0.7, 24/full=0.6, 32/f2=1.1, 32/affine_xor=0.8, 32/full=0.8, 40/f2=1.5, 40/full=1.1, 48/f2=2.0, 48/full=1.4, 64/f2=1.8, 64/affine_xor=1.6, 64/full=1.6, 80/f2=3.3, 80/full=2.2, 96/f2=3.9, 96/affine_xor=1.7, 96/full=2.7, 112/f2=3.6, 112/full=3.3, 128/f2=4.3, 128/affine_xor=3.7, 128/full=3.7
total 160s
