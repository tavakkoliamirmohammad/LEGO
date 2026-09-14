### smem-bound column-read loop on NVIDIA RTX A6000 (32x32 f32 tile, 2048 reps, 672 blocks)

| layout | predicted wavefronts/instr | measured ms | ratio to 1 wavefront |
|---|---|---|---|
| w1 | 1 | 0.880 | 1.00 |
| w2 | 2 | 0.825 | 0.94 |
| w4 | 4 | 1.216 | 1.38 |
| w8 | 8 | 2.356 | 2.68 |
| w16 | 16 | 4.690 | 5.33 |
| w32 | 32 | 8.739 | 9.93 |
| pad1 | 1 | 0.801 | 0.91 |
