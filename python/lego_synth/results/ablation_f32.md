### transpose staging, elem=4B, rows=32: conflict-free footprint by tier

| width | f2 footprint (overhead, template) | affine_xor footprint (overhead, template) | full footprint (overhead, template) |
|---|---|---|---|
| 24 | 1024 (1.33x, XorTemplate) | 800 (1.04x, PaddedTemplate) | 768 (1.00x, XorTemplate) |
| 32 | 1024 (1.00x, XorTemplate) | 1024 (1.00x, XorTemplate) | 1024 (1.00x, XorTemplate) |
| 40 | 2048 (1.60x, XorTemplate) | 1312 (1.02x, PaddedTemplate) | 1280 (1.00x, XorTemplate) |
| 48 | 2048 (1.33x, XorTemplate) | 1568 (1.02x, PaddedTemplate) | 1536 (1.00x, XorTemplate) |
| 64 | 2048 (1.00x, XorTemplate) | 2048 (1.00x, XorTemplate) | 2048 (1.00x, XorTemplate) |
| 80 | 4096 (1.60x, XorTemplate) | 2592 (1.01x, PaddedTemplate) | 2560 (1.00x, XorTemplate) |
| 96 | 4096 (1.33x, XorTemplate) | 3104 (1.01x, PaddedTemplate) | 3072 (1.00x, XorTemplate) |
| 112 | 4096 (1.14x, XorTemplate) | 3616 (1.01x, PaddedTemplate) | 3584 (1.00x, XorTemplate) |
| 128 | 4096 (1.00x, XorTemplate) | 4096 (1.00x, XorTemplate) | 4096 (1.00x, XorTemplate) |

solve times (s): 24/f2=0.5, 24/affine_xor=0.7, 24/full=0.5, 32/f2=1.0, 32/affine_xor=0.7, 32/full=0.7, 40/f2=0.9, 40/affine_xor=0.8, 40/full=0.9, 48/f2=1.5, 48/affine_xor=1.2, 48/full=1.2, 64/f2=1.8, 64/affine_xor=1.5, 64/full=1.5, 80/f2=2.0, 80/affine_xor=1.7, 80/full=2.0, 96/f2=3.2, 96/affine_xor=1.9, 96/full=2.6, 112/f2=3.0, 112/affine_xor=2.3, 112/full=3.0, 128/f2=4.0, 128/affine_xor=3.3, 128/full=3.3
total 156s
