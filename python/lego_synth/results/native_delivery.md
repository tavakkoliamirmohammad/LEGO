# Native delivery of a synthesized layout through the CASTLE pipeline (2026-09-14, A6000, sm_86)

Kernel: 256x256 f32 transpose in the native `@gpu_kernel` model, 32x32 tile in `Shared(layout)`,
block (32, 8). Layout exported from `lego_synth` as a LEGO `GenP` (XOR expressed arithmetically)
via `lego_synth.export.to_lego_tile`, compiled with `lego-to-nvvm`, executed with `mlir-runner`.

| shared layout | compiles (PTX) | runs correctly | arith ops after `lego-lower` (whole kernel) |
|---|---|---|---|
| RowMajor(32, 32) | yes | yes | addi 10, shli 10 (20) |
| XorSwizzle(32, 32, shift=0, mask=31) | yes | yes | addi 28, shli 18 (46), no `xori` |

Finding: the pipeline strength-reduces the bit-decomposed XOR to adds and shifts but never
recovers a native XOR, so a 5-bit swizzle costs ~26 extra index operations instead of 1.
An XOR-aware lowering (recognize the bit-decomposition pattern, or add a bitwise primitive
to the algebra) is a concrete pipeline improvement to evaluate (index-arithmetic cost term).

Lit regression suite on this build: 63/63 passed (146 s).
