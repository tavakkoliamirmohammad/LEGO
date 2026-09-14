### staging, elem=2B, rows=64: conflict-free footprint by tier

| width | f2 footprint (overhead, template) | affine_xor footprint (overhead, template) | full footprint (overhead, template) |
|---|---|---|---|
| 32 | 2048 (1.00x, XorTemplate) | 2048 (1.00x, XorTemplate) | 2048 (1.00x, XorTemplate) |
| 48 | 4096 (1.33x, XorTemplate) | none | 3072 (1.00x, XorTemplate) |
| 64 | 4096 (1.00x, XorTemplate) | 4096 (1.00x, XorTemplate) | 4096 (1.00x, XorTemplate) |
| 80 | 8192 (1.60x, XorTemplate) | none | 5120 (1.00x, XorTemplate) |
| 96 | 8192 (1.33x, XorTemplate) | none | 6144 (1.00x, XorTemplate) |
| 112 | 8192 (1.14x, XorTemplate) | none | 7168 (1.00x, XorTemplate) |
| 128 | 8192 (1.00x, XorTemplate) | 8192 (1.00x, XorTemplate) | 8192 (1.00x, XorTemplate) |

solutions: 32/f2=XorSwizzle(rows=64, cols=32, shift=1, mask=3, unit=8); 32/affine_xor=XorSwizzle(rows=64, cols=32, shift=1, mask=3, unit=8); 32/full=XorSwizzle(rows=64, cols=32, shift=1, mask=3, unit=8); 48/f2=XorSwizzle(rows=64, cols=64, shift=0, mask=7, unit=8); 48/full=XorSwizzle(rows=64, cols=48, shift=2, mask=1, unit=8); 64/f2=XorSwizzle(rows=64, cols=64, shift=0, mask=7, unit=8); 64/affine_xor=XorSwizzle(rows=64, cols=64, shift=0, mask=7, unit=8); 64/full=XorSwizzle(rows=64, cols=64, shift=0, mask=7, unit=8); 80/f2=XorSwizzle(rows=64, cols=128, shift=0, mask=7, unit=8); 80/full=XorSwizzle(rows=64, cols=80, shift=2, mask=1, unit=8); 96/f2=XorSwizzle(rows=64, cols=128, shift=0, mask=7, unit=8); 96/full=XorSwizzle(rows=64, cols=96, shift=1, mask=3, unit=8); 112/f2=XorSwizzle(rows=64, cols=128, shift=0, mask=7, unit=8); 112/full=XorSwizzle(rows=64, cols=112, shift=2, mask=1, unit=8); 128/f2=XorSwizzle(rows=64, cols=128, shift=0, mask=7, unit=8); 128/affine_xor=XorSwizzle(rows=64, cols=128, shift=0, mask=7, unit=8); 128/full=XorSwizzle(rows=64, cols=128, shift=0, mask=7, unit=8)

solve times (s): 32/f2=0.4, 32/affine_xor=0.4, 32/full=0.4, 48/f2=0.9, 48/full=0.7, 64/f2=1.0, 64/affine_xor=0.9, 64/full=0.9, 80/f2=1.8, 80/full=1.2, 96/f2=2.0, 96/full=1.5, 112/f2=1.9, 112/full=1.8, 128/f2=2.0, 128/affine_xor=1.8, 128/full=1.8
total 59s
