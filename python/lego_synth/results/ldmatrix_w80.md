### ldmatrix.x4 loop on NVIDIA RTX A6000 (64x80 b16 tile, 1024 reps, 336 blocks x 4 warps)

| layout | footprint (elems) | fragments correct | predicted wavefronts / ldmatrix.x4 | measured ms |
|---|---|---|---|---|
| naive_160B | 5120 | True | 8 | 1.479 |
| pad8_176B | 5632 | True | 4 | 0.763 |
| synth_exact | 5120 | True | 4 | 0.761 |
| pow2_pad128_sw128 | 8192 | True | 4 | 0.804 |
