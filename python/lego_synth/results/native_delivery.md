# Native delivery of a synthesized layout through the CASTLE pipeline (2026-09-14, A6000, sm_86)

Kernel: 256x256 f32 transpose in the native `@gpu_kernel` model, 32x32 tile in `Shared(layout)`,
block (32, 8). Layout exported from `lego_synth` as a LEGO `GenP` (XOR expressed arithmetically)
via `lego_synth.export.to_lego_tile`, compiled with `lego-to-nvvm`, executed with `mlir-runner`.

| shared layout | compiles (PTX) | runs correctly | arith ops after `lego-lower` (whole kernel) |
|---|---|---|---|
| RowMajor(32, 32) | yes | yes | addi 10, shli 10 (20) |
| XorSwizzle(32, 32, shift=0, mask=31), XOR by bit decomposition | yes | yes | addi 28, shli 18 (46), no `xori` |
| XorSwizzle(32, 32, shift=0, mask=31), `BitXor`/`BitAnd` primitives | yes | yes | addi 10, shli 10, andi 2, xori 2 (24) |

Finding and fix: the pipeline strength-reduced the bit-decomposed XOR to adds and shifts but
never recovered a native XOR (46 ops). Adding `BitXor` / `BitAnd` to the algebra (lowered to
`arith.xori` / `arith.andi`, mapped back in the reverse direction, printed as `^` / `&`,
`ieor`/`iand` in Fortran, `xor` in Julia) brings the swizzled kernel to 24 ops, i.e. 4 more
than the unswizzled tile: two XORs and two ANDs for the two tile accesses. Project tests: 418 pass.

Lit regression suite on this build: 63/63 passed (146 s).
