# CASTLE CPU Evaluation Survey

Scout output for CASTLE/TACO paper Section 7.5. One entry per
(kernel × layout-trick) tuple. Sorted by layout class, then descending
predicted speedup within each class.

---

## Layout Class 1: Cache-Oblivious Recursive Layouts (Z-Morton, Hilbert)

---

### 01-polybench-gemm-zmorton

Z-Morton (Z-order curve, "Morton order") stores a 2-D matrix so that
any 2^k × 2^k sub-block is contiguous in memory, giving optimal
spatial locality for any cache size without tuning parameters. Walker
(2018) applied recursive Morton ordering to dense matrix multiply and
reported consistent speedup over row-major code across power-of-two
sizes on multicore platforms. The CASTLE non-power-of-two capability
adds direct paper value: Morton layouts are typically only studied at
powers of two, and CASTLE can express the same layout at, e.g., 1000 ×
1000 without padding or rounding.

```yaml
id: 01-polybench-gemm-zmorton
suite: PolyBench/C 4.2.1
kernel: gemm (general matrix-matrix multiply)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/blas/gemm/gemm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/blas/gemm/gemm.c utilities/polybench.c -o gemm"
  threading: single-threaded
layout_trick: Recursive Z-Morton (Z-order curve) storage for both A and B input matrices; elements at (i,j) stored at interleaved bit-position index(i,j) = scatter(i) | scatter(j) where scatter interleaves zero bits between each bit of the index
layout_trick_citation: walker2018morton
why_compiler_cant: |
  Z-Morton storage requires a non-affine index mapping: the address of
  element (i,j) is given by bit-interleaving i and j, which is not an
  affine function of the loop indices. GCC and Clang's polyhedral
  frameworks (Graphite / Polly) require affine loop bounds and array
  subscripts; they reject non-affine subscripts entirely (-floop-nest-optimize
  / -mllvm -polly). Even with -O3 -march=native the compiler sees
  only the explicit linearisation formula and cannot recover the recursive
  block structure or eliminate the extra index-computation overhead.
lego_expressibility: |
  # Interleave bits of row index r and col index c to get Morton index.
  # In LEGO, express as a two-level TileBy that recursively halves both
  # dimensions — equivalent to one level of the Morton recursion:
  #
  #   A_morton = OrderBy(A, TileBy(rows, T) @ TileBy(cols, T))
  #              for T = 2^k chosen so the leaf tile fits in L1.
  #
  # Full Morton recursion = repeated application of TileBy at each
  # level, doubling T at each step until the full matrix size.
  # LEGO's multi-level TileBy directly models this:
  #
  #   import lego
  #   T0, T1, T2 = 4, 16, 64   # register, L1, L2 tile sizes
  #   A_lego = lego.OrderBy(A,
  #              lego.TileBy(lego.Row, T0) @
  #              lego.TileBy(lego.Col, T0) @
  #              lego.TileBy(lego.Row, T1) @
  #              lego.TileBy(lego.Col, T1))
  # GroupBy not needed; pure OrderBy + TileBy.
predicted_win:
  value: "1.3x – 2.0x"
  source: walker2018morton
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: true
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 1000, 1024,
  1500, 2048.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Morton index computation overhead may dominate at small N
  - Interleaved bit addressing requires explicit index remapping not present in baseline
  - Non-power-of-two Morton requires padding to next power-of-two or fractional-cascade variant
```

---

### 02-polybench-lu-zmorton

LU factorisation of a dense matrix exhibits the same data-reuse pattern
as GEMM but with a triangular dependency that prevents straightforward
tiling. Perdacher et al. (2020) demonstrated that Morton-order storage
improves data locality for LU factorisation over row-major storage and
reported speedups on the block-factorisation path, which reduces to
matrix multiplication. CASTLE can target non-power-of-two matrix
dimensions where Morton-indexed LU has not previously been benchmarked.

```yaml
id: 02-polybench-lu-zmorton
suite: PolyBench/C 4.2.1
kernel: lu (LU factorisation without pivoting)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/solvers/lu/lu.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/solvers/lu/lu.c utilities/polybench.c -o lu"
  threading: single-threaded
layout_trick: Z-Morton storage of the N×N matrix; both read and write accesses use bit-interleaved (row,col) addressing
layout_trick_citation: perdacher2020mortonlu
why_compiler_cant: |
  The polyhedral model (used by Graphite in GCC and Polly in LLVM)
  cannot model non-affine subscripts like Morton index-interleaving.
  Even -O3 -march=native only performs loop interchange and basic
  vectorisation; the compiler cannot introduce a globally non-affine
  memory mapping. The loop structure of PolyBench LU (three nested
  loops with triangular bounds) is outside the class of Morton-
  optimisable code unless the compiler can perform non-affine
  memory remapping, which neither GCC -fgraphite-identity nor
  Polly -polly supports.
lego_expressibility: |
  # Two-level TileBy models two levels of Morton recursion:
  #   T_reg, T_L1 = 4, 32
  #   A_lego = lego.OrderBy(A,
  #              lego.TileBy(lego.Row, T_reg) @
  #              lego.TileBy(lego.Col, T_reg) @
  #              lego.TileBy(lego.Row, T_L1) @
  #              lego.TileBy(lego.Col, T_L1))
  # No GroupBy required.
predicted_win:
  value: "1.2x – 1.8x"
  source: perdacher2020mortonlu
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: true
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 512, 768, 1000, 1024.
estimated_builder_effort: "2-3 days"
risk_flags:
  - LU triangular update loop modifies the matrix in-place; Morton layout requires index translation on every read and write
  - Pivoting absent in PolyBench lu; numerical stability not a concern but the loop structure is simpler than LAPACK DGETRF
```

---

### 03-polybench-chol-zmorton

Cholesky factorisation of a symmetric positive-definite matrix reduces to
a sequence of DGEMM calls on sub-blocks. Walker (2018) measured Morton
ordering for Cholesky and found consistent speedup relative to row-major
storage across multiple multicore platforms. The PolyBench cholesky
kernel uses a non-BLAS reference implementation, making it an ideal
testbed to isolate the layout effect without library interference.

```yaml
id: 03-polybench-chol-zmorton
suite: PolyBench/C 4.2.1
kernel: cholesky (Cholesky factorisation)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/solvers/cholesky/cholesky.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/solvers/cholesky/cholesky.c utilities/polybench.c -o cholesky"
  threading: single-threaded
layout_trick: Z-Morton storage of the N×N symmetric matrix; upper triangle is packed using Morton indexing
layout_trick_citation: walker2018morton
why_compiler_cant: |
  GCC -floop-nest-optimize (Graphite) and Clang -polly require affine
  subscripts. The Morton mapping index(i,j) = bit_interleave(i,j) is
  not an affine function of (i,j); neither pass can discover or apply
  it. The triangular loop bounds of Cholesky additionally prevent
  standard tiling from applying even at the loop level; Polly rejects
  non-rectangular iteration domains unless -polly-allow-nonaffine is
  set, at which point it falls back to identity transformation.
lego_expressibility: |
  # Symmetric matrix: only lower-triangle is written.
  # TileBy at two levels approximates Morton recursion:
  #   T_reg, T_L1 = 4, 32
  #   A_lego = lego.OrderBy(A,
  #              lego.TileBy(lego.Row, T_reg) @
  #              lego.TileBy(lego.Col, T_reg) @
  #              lego.TileBy(lego.Row, T_L1) @
  #              lego.TileBy(lego.Col, T_L1))
  # No GroupBy required.
predicted_win:
  value: "1.2x – 1.5x"
  source: walker2018morton
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: true
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 512, 768, 1000.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Cholesky loop has a square-root per column; compute is not fully memory-bound so the layout win may be smaller than for pure GEMM
```

---

## Layout Class 2: Multi-Level Cache-Conscious Tiling

---

### 04-polybench-gemm-reg-L1-L2-tile

The Goto/van de Geijn "anatomy of GEMM" identifies a three-level packing
hierarchy (register micro-tile, L1 panel, L2 panel) that is required for
near-peak floating-point performance. The PolyBench gemm baseline at -O3
receives auto-vectorisation and basic tiling from GCC's -floop-nest-optimize
but not the full three-level packed packing structure described in
[goto2008anatomy]. CASTLE can express the three-level tile as a composition
of TileBy primitives, making the LEGO expression concise and verifiable.

```yaml
id: 04-polybench-gemm-reg-L1-L2-tile
suite: PolyBench/C 4.2.1
kernel: gemm (general matrix-matrix multiply)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/blas/gemm/gemm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/blas/gemm/gemm.c utilities/polybench.c -o gemm"
  threading: single-threaded
layout_trick: Three-level register × L1 × L2 tile packing; B matrix packed into L1 panels of size kc × nr, A matrix packed into L2 panels of size mr × kc, following the Goto/BLIS micro-kernel model
layout_trick_citation: goto2008anatomy
why_compiler_cant: |
  GCC's polyhedral tiling (-floop-nest-optimize / Graphite) applies a
  single-level rectangular tile and does not generate the explicit packing
  buffers (Ã, B̃) required by the three-level Goto hierarchy. Auto-
  vectorisation (-ftree-vectorize) produces SIMD loads for the inner
  loop but without aligned packed panels the cache-line utilisation
  remains suboptimal. The specific pass that would be needed—multi-level
  loop tiling with explicit scratchpad allocation—is not implemented in
  GCC -O3 or Clang -O3 for arbitrary loop nests; it is only available in
  hand-tuned BLAS libraries or frameworks like BLIS.
lego_expressibility: |
  # Register tile mr x nr, L1 tile mc x kc, L2 tile nc x kc:
  #   mr, nr = 6, 8     # AVX2 register micro-tile
  #   mc, kc = 72, 256  # L1 panel
  #   nc     = 3072     # L2 panel
  #   B_packed = lego.TileBy(B, kc) @ lego.TileBy(B, nr)
  #   A_packed = lego.TileBy(A, mc) @ lego.TileBy(A, mr)
  #   # Expressed as:
  #   B_lego = lego.OrderBy(B, lego.TileBy(lego.Row, kc) @
  #                            lego.TileBy(lego.Col, nr))
  #   A_lego = lego.OrderBy(A, lego.TileBy(lego.Row, mc) @
  #                            lego.TileBy(lego.Col, kc) @
  #                            lego.TileBy(lego.Row, mr))
  # No GroupBy required.
predicted_win:
  value: "2.0x – 4.0x"
  source: goto2008anatomy
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 800, 1000, 1024, 2048.
estimated_builder_effort: "3-4 days"
risk_flags:
  - Panel sizes (mc, kc, nc) are microarchitecture-dependent; wrong sizes can harm performance
  - LEGO expression targets layout only; the micro-kernel itself still uses scalar C unless CASTLE emits SIMD
```

---

### 05-polybench-3mm-reg-L1-L2-tile

The 3mm kernel chains three matrix multiplications (D = A×B, E = C×D,
F = E×G). Each GEMM is individually optimisable by the Goto packing
hierarchy; additionally the intermediate matrices D and E can be kept in
the same packed layout, avoiding a re-pack between multiplications. This
makes the 3mm layout exploration two-dimensional: per-GEMM tiling plus
cross-GEMM layout persistence.

```yaml
id: 05-polybench-3mm-reg-L1-L2-tile
suite: PolyBench/C 4.2.1
kernel: 3mm (three matrix multiplications chained)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/kernels/3mm/3mm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/kernels/3mm/3mm.c utilities/polybench.c -o 3mm"
  threading: single-threaded
layout_trick: Three-level register × L1 × L2 tile for each GEMM stage; intermediate result kept in L1-panel layout to avoid re-packing between the first and second GEMM
layout_trick_citation: salvado2023pack
why_compiler_cant: |
  GCC Graphite and Clang Polly each tile individual loop nests in
  isolation; they do not fuse the three-nested-loop sequences comprising
  each GEMM or persist a packed intermediate across kernel boundaries.
  The GPAT paper (CGO 2023, cited as salvado2023pack) demonstrates
  that even the MLIR Affine dialect needs an explicit packing
  transformation pass to discover this opportunity; -O3 with
  -floop-nest-optimize does not cross loop-nest boundaries to keep
  intermediates in a packed layout.
lego_expressibility: |
  #   mr, nr = 6, 8; mc, kc = 72, 256; nc = 3072
  #   E_shared_layout = lego.OrderBy(E,
  #       lego.TileBy(lego.Row, mc) @
  #       lego.TileBy(lego.Col, kc))
  #   # First GEMM writes E in packed layout;
  #   # Second GEMM reads E without repacking.
  # No GroupBy required.
predicted_win:
  value: "1.5x – 2.5x"
  source: salvado2023pack
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 800, 1000, 1024.
estimated_builder_effort: "3-4 days"
risk_flags:
  - Three independent tile-size parameters must be jointly tuned
  - Intermediate layout must be consistent between GEMM stages; layout mismatch is a correctness risk
```

---

### 06-polybench-2mm-reg-L1-tile

The 2mm kernel (D = alpha*A*B + beta*C, then E = alpha*D*F + beta*G) is
a two-stage fused matrix multiply. Like 3mm, the intermediate D can be
left in the packed panel layout. Salvado et al. (CGO 2023) show that
the GPAT generalised packing transformation achieves measurable speedup on
2mm within PolyBench by identifying and applying packing on both GEMM
loop nests.

```yaml
id: 06-polybench-2mm-reg-L1-tile
suite: PolyBench/C 4.2.1
kernel: 2mm (two matrix multiplications fused)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/kernels/2mm/2mm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/kernels/2mm/2mm.c utilities/polybench.c -o 2mm"
  threading: single-threaded
layout_trick: Two-level L1 × register packing of the A and B panels; intermediate result D kept in L1-panel layout
layout_trick_citation: salvado2023pack
why_compiler_cant: |
  Same reasoning as 05-polybench-3mm-reg-L1-L2-tile: GCC Graphite
  tiles each loop nest independently, producing a single-level tile
  (selected by heuristic); it does not introduce the explicit A-panel
  and B-panel packing buffers. The -floop-nest-optimize pass is
  bounded to a single loop nest and cannot persist a packed intermediate
  across the two GEMM loop nests in 2mm.c. Clang Polly -polly similarly
  tiles within a single SCoP.
lego_expressibility: |
  #   mr, nr = 6, 8; mc, kc = 72, 256
  #   A_packed = lego.OrderBy(A, lego.TileBy(lego.Row, mc) @
  #                              lego.TileBy(lego.Col, kc) @
  #                              lego.TileBy(lego.Row, mr))
  #   B_packed = lego.OrderBy(B, lego.TileBy(lego.Row, kc) @
  #                              lego.TileBy(lego.Col, nr))
  # No GroupBy required.
predicted_win:
  value: "1.4x – 2.0x"
  source: salvado2023pack
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 800, 1000, 1024.
estimated_builder_effort: "2-3 days"
risk_flags:
  - BLIS-style panel sizes differ across microarchitectures; tuning required
```

---

### 07-polybench-trmm-L1-L2-tile

The trmm (triangular matrix-matrix multiply) kernel uses only the lower
or upper triangle of one operand. The RFP (Rectangular Full Packed) layout
stores both triangles as two adjacent rectangular blocks, enabling full
BLAS-3 performance without the wasted storage of full format. Gustavson et
al. (TOMS 2010) report speedup of up to 43× over conventional packed-format
routines on multiple platforms when using RFP.

```yaml
id: 07-polybench-trmm-L1-L2-tile
suite: PolyBench/C 4.2.1
kernel: trmm (triangular matrix-matrix multiply)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/blas/trmm/trmm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/blas/trmm/trmm.c utilities/polybench.c -o trmm"
  threading: single-threaded
layout_trick: L1 × register two-level tile for the triangular operand; inner loop processes full-width register tiles to maintain SIMD alignment
layout_trick_citation: goto2008anatomy
why_compiler_cant: |
  The triangular loop bound (j <= i) makes the iteration space
  non-rectangular. GCC Graphite requires rectangular tiles for its
  rectangular tiling pass (-floop-nest-optimize); the triangular bound
  causes Graphite to fall back to no transformation. Polly with
  -polly-allow-nonaffine similarly cannot tile the non-rectangular
  domain. The compiler therefore emits scalar code with a conditional
  branch inside the inner loop (or loop splitting that breaks
  vectorisation).
lego_expressibility: |
  #   mr, nr = 6, 8; mc, kc = 72, 256
  #   # A is triangular; tile only the rectangular sub-blocks:
  #   A_tiled = lego.OrderBy(A, lego.TileBy(lego.Row, mc) @
  #                             lego.TileBy(lego.Col, kc))
  #   # No GroupBy required; triangular masking happens at compute level.
predicted_win:
  value: "1.5x – 3.0x"
  source: gustavson2010rfpcholesky
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 512, 1000, 1024.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Triangular loop bounds require careful tile boundary handling to avoid out-of-bound accesses
  - Predicted win is extrapolated from LAPACK RFP results; direct CPU numbers for PolyBench trmm are not published
```

---

### 08-polybench-doitgen-reg-L1-tile

The doitgen kernel computes a multi-dimensional reduction over a 4D
array. The innermost access pattern has poor spatial locality in the
baseline row-major storage; tiling the two inner dimensions to fit in
L1 is the standard optimisation. Salvado et al. (CGO 2023) show GPAT
produces measurable speedup on doitgen within PolyBench by applying
packing to the innermost loop's reduction operand.

```yaml
id: 08-polybench-doitgen-reg-L1-tile
suite: PolyBench/C 4.2.1
kernel: doitgen (multi-dimensional reduction)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/kernels/doitgen/doitgen.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/kernels/doitgen/doitgen.c utilities/polybench.c -o doitgen"
  threading: single-threaded
layout_trick: L1-level tile of the two innermost dimensions of the 4D reduction array; inner tile sized to fit NR×NR in L1
layout_trick_citation: salvado2023pack
why_compiler_cant: |
  Doitgen has three nested loops where the innermost reduction
  accumulates over the NR dimension. GCC Graphite at -O3 applies
  interchange and a single-level tile, but the heuristic tile size
  does not account for the outer two loops' working set. The packing
  transformation that copies a NR×NR block into a contiguous scratchpad
  is not generated by -floop-nest-optimize; the pass does not model
  explicit scratchpad allocation. Polly with -polly-use-runtime-aliasing
  similarly cannot insert packing buffers automatically.
lego_expressibility: |
  #   T = 32  # tile size to fit in L1
  #   C4_tiled = lego.OrderBy(C4, lego.TileBy(lego.Row, T) @
  #                               lego.TileBy(lego.Col, T))
  # No GroupBy required.
predicted_win:
  value: "1.2x – 1.8x"
  source: salvado2023pack
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; NR = NQ = NP = 140, 256.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Four-dimensional array indexing is verbose in LEGO; mapping must be consistent across all four indices
```

---

### 09-tccg-tensor-contraction-GETT-tile

The TCCG tensor-contraction benchmark suite collects contractions from
computational chemistry (CCSD, CCSD(T)) and quantum chemistry codes.
Springer and Bientinesi (TOMS 2018) show that the GETT algorithm,
which packs sub-tensors into multi-level cache buffers following the
Goto GEMM hierarchy, outperforms naive contraction by up to 12.4× on
bandwidth-bound contractions. CASTLE can express the multi-level
packing layout for arbitrary tensor modes using TileBy compositions.

```yaml
id: 09-tccg-tensor-contraction-GETT-tile
suite: TCCG tensor-contraction benchmark (HPAC, 2016)
kernel: generic tensor contraction (T[a,b,c,d] += A[a,e,f] * B[b,c,d,e,f])
upstream_url: https://github.com/HPAC/tccg
license: MIT
language: cpp
baseline:
  source_files: [tccg/tccg.py, benchmark/benchmark.py]
  build: "python tccg/tccg.py --alpha 1 --beta 0 --compiler gcc --floatType double <contraction_string>"
  threading: single-threaded
layout_trick: Multi-level packing of tensor sub-blocks into L1 and L2 cache buffers, following the Goto/BLIS micro-kernel model applied to tensors; contraction indices identified as m, n, k analogues
layout_trick_citation: springer2018gett
why_compiler_cant: |
  Tensor contractions are expressed as multi-dimensional loop nests
  with non-unit strides in the innermost dimension when the contraction
  index is not the fastest-varying mode. GCC and Clang -O3 auto-
  vectorise only when the innermost loop has stride-1 access; for
  contractions where the reduction index runs along a non-contiguous
  mode, the compiler emits gather/scatter instructions or scalar loops.
  The explicit packing of sub-tensors into contiguous buffers—required
  for the GETT approach—is not inserted by -floop-nest-optimize or
  Polly: both passes lack the sub-tensor packing primitive.
lego_expressibility: |
  #   For a rank-3 × rank-3 contraction with modes a,b vs c,d:
  #   T_reg = 8  # register tile
  #   T_L1  = 64 # L1 tile
  #   A_packed = lego.OrderBy(A,
  #                lego.TileBy(lego.Row, T_L1) @
  #                lego.TileBy(lego.Col, T_reg))
  #   B_packed = lego.OrderBy(B,
  #                lego.TileBy(lego.Row, T_L1) @
  #                lego.TileBy(lego.Col, T_reg))
  # General mode mapping handled by OrderBy permutation.
predicted_win:
  value: "2.0x – 12.4x"
  source: springer2018gett
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled.
estimated_builder_effort: "4-5 days"
risk_flags:
  - TCCG generates C++ code; the baseline may already apply some tiling via the code generator
  - Contraction-specific mode mapping from tensor modes to LEGO Row/Col is non-trivial for rank > 3
  - License of TCCG benchmark scripts not explicitly stated in repo; verify MIT before building
```

---

### 10-tblis-tensor-contraction-notranspose

Matthews (2018) introduces TBLIS, which performs tensor contractions
without explicit transposition by fusing the index reordering with the
BLIS packing phase. On bandwidth-bound contractions the approach matches
near-peak GEMM performance. CASTLE can express the same fusion at the
layout level: the OrderBy primitive reorders tensor modes before the
TileBy packs them into cache buffers, avoiding the separate transpose step.

```yaml
id: 10-tblis-tensor-contraction-notranspose
suite: TBLIS benchmark (Matthews 2018, arXiv:1607.00291)
kernel: mode-k tensor-matrix product (TTM)
upstream_url: https://arxiv.org/abs/1607.00291
license: BSD-3-Clause
language: cpp
baseline:
  source_files: []
  build: "cmake -DCMAKE_BUILD_TYPE=Release . && make -j16"
  threading: single-threaded
layout_trick: Mode permutation fused with L1 panel packing; tensor mode k is mapped to the innermost (fastest-varying) layout axis before packing, eliminating a separate explicit transposition step
layout_trick_citation: matthews2018tblis
why_compiler_cant: |
  A tensor-matrix product along mode k accesses the tensor with stride
  equal to the product of all mode sizes smaller than k. For k > 0
  this is a non-unit stride inner loop. GCC -O3 -march=native generates
  gather instructions for AVX2 targets but cannot reorder the tensor
  layout to make mode k contiguous: that requires a non-loop-level
  transformation (a data layout change) which -floop-nest-optimize and
  Polly do not perform. TBLIS avoids the explicit transposition by
  altering the packing phase, a technique not available to any standard
  compiler pass.
lego_expressibility: |
  #   Permute tensor T so that contraction mode k becomes last (Row):
  #   T_perm = lego.OrderBy(T, lego.Col @ lego.Row)  # for rank-2
  #   # For rank-3 with contraction along mode 1:
  #   T_perm = lego.OrderBy(T, (0, 2, 1))  # mode permutation
  #   T_packed = lego.OrderBy(T_perm,
  #                lego.TileBy(lego.Row, mc) @
  #                lego.TileBy(lego.Col, kc))
  # No GroupBy required.
predicted_win:
  value: "1.5x – 4.0x"
  source: matthews2018tblis
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled.
estimated_builder_effort: "3-4 days"
risk_flags:
  - TBLIS source availability and build system complexity; verify BSD-3-Clause before building
  - Mode permutation in LEGO requires careful index-to-dimension mapping
```

---

## Layout Class 3: Recursive Bricking for Stencils

---

### 11-bricklib-3d7pt-brick

The brick data structure partitions a 3D stencil grid into fixed-size
bricks (e.g., 8×8×8) stored contiguously regardless of the surrounding
grid dimensions. Zhao et al. (SC 2019) show 1.9×–4.9× speedup over
auto-vectorised row-major code for 7-point and 13-point stencils on
Skylake CPUs, because all elements of a brick fit simultaneously in L1
and can be accessed with unit stride after the layout transform. The
BrickLib suite is MIT-licensed.

```yaml
id: 11-bricklib-3d7pt-brick
suite: BrickLib stencil suite (r0.1)
kernel: 3D 7-point Laplacian stencil
upstream_url: https://github.com/CtopCsUtahEdu/bricklib/archive/refs/tags/r0.1.tar.gz
license: MIT
language: cpp
baseline:
  source_files: [examples/stencils/stencil3d7pt.cpp]
  build: "cmake -DCMAKE_BUILD_TYPE=Release . && make -j16 stencil3d7pt_naive"
  threading: single-threaded
layout_trick: Brick layout with 8×8×8 bricks stored contiguously; grid elements reindexed so that (i,j,k) → brick_id * 512 + local_i * 64 + local_j * 8 + local_k
layout_trick_citation: zhao2019bricks
why_compiler_cant: |
  GCC and Clang auto-vectorise only loops with stride-1 innermost
  access. In a 3D 7-point stencil the accesses in the j and k
  directions have stride N and N^2 respectively. -floop-nest-optimize
  (Graphite) can interchange loops to bring k innermost, giving unit
  stride in k, but the j-direction stencil neighbour access (stride N)
  remains a non-unit stride scatter that prevents full vectorisation.
  The brick layout makes all six neighbours stride-1 within the brick,
  which the compiler cannot discover by loop interchange alone: it
  requires a global memory renumbering that is outside the scope of any
  affine polyhedral pass.
lego_expressibility: |
  #   Brick size B = 8 in each dimension:
  #   grid = lego.OrderBy(grid3d,
  #            lego.TileBy(lego.Row, 8) @
  #            lego.TileBy(lego.Col, 8) @
  #            lego.TileBy(lego.Depth, 8))
  #   # lego.Depth is the third spatial dimension (GenP axis).
  # No GroupBy required.
predicted_win:
  value: "1.9x – 4.9x"
  source: zhao2019bricks
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; grid = 128^3, 256^3.
estimated_builder_effort: "3-4 days"
risk_flags:
  - Boundary handling at domain edges requires padding or conditional code; impacts non-power-of-two grids
  - BrickLib build system (CMake) requires LLVM headers for code-gen path; pure C++ path must be isolated
```

---

### 12-bricklib-3d13pt-brick

The 13-point 3D stencil extends the 7-point stencil with diagonal
neighbours, increasing arithmetic intensity. Zhao et al. (SC 2019)
report similar or higher speedup for the 13-point kernel because the
larger stencil footprint benefits more strongly from brick locality:
all 13 neighbours of a brick-interior point reside within the same brick
or one adjacent brick, so the cache working set halves compared to
row-major storage.

```yaml
id: 12-bricklib-3d13pt-brick
suite: BrickLib stencil suite (r0.1)
kernel: 3D 13-point stencil (face + edge neighbours)
upstream_url: https://github.com/CtopCsUtahEdu/bricklib/archive/refs/tags/r0.1.tar.gz
license: MIT
language: cpp
baseline:
  source_files: [examples/stencils/stencil3d13pt.cpp]
  build: "cmake -DCMAKE_BUILD_TYPE=Release . && make -j16 stencil3d13pt_naive"
  threading: single-threaded
layout_trick: Brick layout with 8×8×8 bricks, same renumbering as 11-bricklib-3d7pt-brick; the 13-point stencil additionally accesses edge-adjacent bricks
layout_trick_citation: zhao2019bricks
why_compiler_cant: |
  Same argument as 11-bricklib-3d7pt-brick: the 13-point stencil adds
  edge-diagonal neighbours at strides N±1, N^2±N, N^2±1, none of
  which are unit stride. GCC and Clang cannot convert global row-major
  storage to per-brick contiguous storage via any affine pass;
  -floop-nest-optimize is limited to loop-level transformations with
  affine subscripts.
lego_expressibility: |
  #   Same expression as 11-bricklib-3d7pt-brick:
  #   grid = lego.OrderBy(grid3d,
  #            lego.TileBy(lego.Row, 8) @
  #            lego.TileBy(lego.Col, 8) @
  #            lego.TileBy(lego.Depth, 8))
predicted_win:
  value: "2.0x – 4.9x"
  source: zhao2019bricks
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; grid = 128^3, 256^3.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Edge bricks require ghost-cell handling; boundary correctness must be verified
```

---

### 13-polybench-heat3d-brick

The PolyBench heat-3d kernel runs Jacobi iteration over a 3D grid using
a 7-point stencil. Unlike the BrickLib suite it uses a plain C reference
without explicit vectorisation hints, making it a clean baseline. Applying
the brick layout to heat-3d isolates the layout effect from the BrickLib
code-generation infrastructure.

```yaml
id: 13-polybench-heat3d-brick
suite: PolyBench/C 4.2.1
kernel: heat-3d (3D heat equation, Jacobi time-stepping)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [stencils/heat-3d/heat-3d.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities stencils/heat-3d/heat-3d.c utilities/polybench.c -o heat-3d"
  threading: single-threaded
layout_trick: 8×8×8 brick layout applied to the grid array; each brick of 512 doubles (4 KB) fits in L1 data cache
layout_trick_citation: zhao2019bricks
why_compiler_cant: |
  GCC -floop-nest-optimize applies loop tiling with a heuristic tile
  size (typically 32–64) and loop interchange. It cannot introduce a
  globally non-affine renumbering of the array elements into bricks
  because (a) brick addresses are non-affine functions of (i,j,k) and
  (b) the pass does not allocate or populate a new data structure. The
  result is tiled row-major code that still suffers from stride-N and
  stride-N^2 accesses in the j and k directions respectively.
lego_expressibility: |
  #   B = 8
  #   A_brick = lego.OrderBy(A, lego.TileBy(lego.Row, B) @
  #                             lego.TileBy(lego.Col, B) @
  #                             lego.TileBy(lego.Depth, B))
predicted_win:
  value: "1.5x – 3.0x"
  source: zhao2019bricks
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 100, 128, 200.
estimated_builder_effort: "2-3 days"
risk_flags:
  - PolyBench heat-3d iterates TSTEPS times; wall-clock measurement must cover enough iterations
  - Predicted win is extrapolated from BrickLib results; PolyBench baseline uses C not C++
```

---

### 14-polybench-jacobi2d-brick

The Jacobi 2D stencil kernel accesses 4 neighbours per grid point with
strides 1 and N. A 2D brick layout (e.g., 32×32 tiles) makes all four
neighbours stride-1 within the tile. This is a simpler case than 3D
bricks, making it a good entry point for CASTLE layout exploration and
a low-risk candidate.

```yaml
id: 14-polybench-jacobi2d-brick
suite: PolyBench/C 4.2.1
kernel: jacobi-2d (2D Jacobi iteration, 5-point stencil)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [stencils/jacobi-2d/jacobi-2d.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities stencils/jacobi-2d/jacobi-2d.c utilities/polybench.c -o jacobi-2d"
  threading: single-threaded
layout_trick: 32×32 tile (brick) layout for the 2D grid; all five stencil points of an interior point reside within the same or adjacent tiles
layout_trick_citation: zhao2019bricks
why_compiler_cant: |
  GCC Graphite (-floop-nest-optimize) generates a single-level
  rectangular tile for Jacobi-2d. The heuristic tile size (32 or 64)
  does not account for the double-buffer requirement (two grids
  alternating). More critically, GCC does not renumber the array into
  a 2D tiled layout; it only reorders loop iterations. The stride-N
  access in the j-direction remains a non-unit stride load even after
  GCC's tile: the tile iterates over (i_tile, j_tile, i, j) but
  element A[i+1][j] still has stride N in the flat row-major array.
lego_expressibility: |
  #   T = 32
  #   A_tiled = lego.OrderBy(A, lego.TileBy(lego.Row, T) @
  #                             lego.TileBy(lego.Col, T))
  #   B_tiled = lego.OrderBy(B, lego.TileBy(lego.Row, T) @
  #                             lego.TileBy(lego.Col, T))
predicted_win:
  value: "1.3x – 2.0x"
  source: zhao2019bricks
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 400, 500, 512.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Two-grid Jacobi requires both A and B in brick layout; layout must be consistent
  - 2D stencil benefit is smaller than 3D; speedup may approach PARITY threshold
```

---

## Layout Class 4: Triangular / Symmetric Packing (RFP-Style)

---

### 15-polybench-symm-rfp

The symm kernel computes C = alpha*A*B + beta*C where A is symmetric.
The baseline PolyBench code stores A in full N×N format, wasting half
the storage. The RFP layout (Gustavson et al., TOMS 2010) stores both
triangles as two N/2 × N rectangular panels packed back-to-back, fitting
in half the cache space while enabling full BLAS-3 access patterns.
Gustavson reports up to 43× speedup over conventional packed-format
routines in the serial path.

```yaml
id: 15-polybench-symm-rfp
suite: PolyBench/C 4.2.1
kernel: symm (symmetric matrix-matrix multiply)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/blas/symm/symm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/blas/symm/symm.c utilities/polybench.c -o symm"
  threading: single-threaded
layout_trick: Rectangular Full Packed (RFP) storage of the symmetric matrix A; upper and lower triangles stored as two N/2 × N panels, eliminating wasted cache lines
layout_trick_citation: gustavson2010rfpcholesky
why_compiler_cant: |
  The PolyBench symm baseline stores A as a full N×N array and accesses
  only the lower triangle with a conditional branch (if j <= i). GCC
  -O3 cannot compress A into RFP format: doing so would require
  introducing a new data structure with a non-affine address mapping
  (the RFP index formula depends on whether i < N/2 or i >= N/2).
  -floop-nest-optimize does not restructure global data layout; it only
  reorders iterations. The result is that half the cache lines loaded
  for A are wasted on unused upper-triangle elements.
lego_expressibility: |
  #   RFP layout: store lower triangle as A[:N//2, :] and
  #   upper triangle as A[N//2:, :] in transposed form.
  #   In LEGO, model as two half-height panels:
  #   A_lower = lego.OrderBy(A_lower_triangle,
  #               lego.TileBy(lego.Row, N//2))
  #   A_upper = lego.OrderBy(A_upper_triangle,
  #               lego.TileBy(lego.Col, N//2))
  #   # Concatenated in memory. GroupBy not required.
predicted_win:
  value: "1.5x – 5.0x"
  source: gustavson2010rfpcholesky
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 512, 1000, 1024.
estimated_builder_effort: "3-4 days"
risk_flags:
  - RFP index formula differs for odd vs even N; requires careful boundary handling
  - Gustavson speedup numbers are relative to packed-format LAPACK, not relative to full-format; baseline comparison must be clarified
```

---

### 16-polybench-syrk-rfp

The syrk kernel computes C = alpha*A*A^T + beta*C where C is symmetric
positive semi-definite. Only the lower (or upper) triangle of C is
updated. RFP storage of C halves the cache footprint of the output
matrix and enables column-contiguous access patterns that the compiler
cannot discover without global data-structure remapping.

```yaml
id: 16-polybench-syrk-rfp
suite: PolyBench/C 4.2.1
kernel: syrk (symmetric rank-k update)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/blas/syrk/syrk.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/blas/syrk/syrk.c utilities/polybench.c -o syrk"
  threading: single-threaded
layout_trick: RFP storage of output matrix C; only N(N+1)/2 doubles stored instead of N^2, halving cache pressure on C reads/writes
layout_trick_citation: gustavson2010rfpcholesky
why_compiler_cant: |
  The syrk loop iterates with j <= i (lower triangle update). GCC
  -O3's -ftree-loop-distribute-patterns recognises the triangular
  iteration but does not reorganise the output array storage into RFP.
  -floop-nest-optimize tiles the loop nest but keeps C in full N×N
  format, so every cache line loaded for C carries at most N/2 useful
  elements (the upper triangle cells are loaded but never written).
  Introducing RFP requires a global address mapping that no polyhedral
  or tree-level compiler pass implements.
lego_expressibility: |
  #   Same RFP expression as 15-polybench-symm-rfp, applied to C:
  #   C_rfp = lego.OrderBy(C_lower,
  #              lego.TileBy(lego.Row, N//2))
predicted_win:
  value: "1.3x – 3.0x"
  source: gustavson2010rfpcholesky
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 512, 1000, 1024.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Same RFP boundary risk as 15-polybench-symm-rfp
  - Speedup predicted from RFP paper; PolyBench-specific measurement not published
```

---

## Layout Class 5: Skewed / Shifted Layouts (Dynamic-Programming Wavefronts)

---

### 17-rodinia-nw-antidiag-tile

The Rodinia Needleman-Wunsch benchmark fills a dynamic-programming matrix
where each cell depends on its left, upper, and diagonal-upper-left
neighbours. The natural anti-diagonal wavefront has poor spatial locality
in row-major storage: each anti-diagonal accesses elements from two
adjacent rows. Pałkowski et al. (ICAISC 2018) apply loop skewing and
tiling to the NW loop nest, demonstrating super-linear speedup through
increased cache locality on a multi-core platform.

```yaml
id: 17-rodinia-nw-antidiag-tile
suite: Rodinia 3.1
kernel: nw (Needleman-Wunsch global sequence alignment, CPU OpenMP)
upstream_url: https://github.com/HPC-FAIR/rodinia_3.1/archive/refs/heads/main.tar.gz
license: BSD-3-Clause
language: cpp
baseline:
  source_files: [openmp/nw/needle.cpp, openmp/nw/needle_omp.cpp]
  build: "g++ -O3 -march=native -fopenmp openmp/nw/needle.cpp openmp/nw/needle_omp.cpp -o nw"
  threading: single-threaded
layout_trick: Skewed anti-diagonal tile storage; DP matrix stored in diagonal-stripe order so that each anti-diagonal wave of cells is contiguous in memory, enabling stride-1 access during the wavefront sweep
layout_trick_citation: palkowski2018nw
why_compiler_cant: |
  The Needleman-Wunsch loop has the recurrence dependency
  score[i][j] = max(score[i-1][j-1], score[i-1][j], score[i][j-1])
  which is not a uniform dependence: the anti-diagonal direction mixes
  row and column indices. GCC Graphite (-floop-nest-optimize) recognises
  the skewing opportunity (skewing by loop index i+j is within the
  polyhedral model) but does not combine skewing with a global array
  renumbering to store anti-diagonals contiguously. The resulting tiled
  code still accesses the DP matrix in row-major order, incurring
  stride-N accesses for the upward neighbour.
lego_expressibility: |
  #   Anti-diagonal storage: element (i,j) maps to stripe index i+j
  #   and position within stripe i.
  #   In LEGO, express as skewed TileBy:
  #   score_skewed = lego.OrderBy(score,
  #                    lego.TileBy(lego.Row + lego.Col, T) @
  #                    lego.TileBy(lego.Row, T))
  #   # Row + Col is the anti-diagonal index; Row within that tile.
  #   # T chosen so each anti-diagonal tile fits in L1.
predicted_win:
  value: "1.5x – 3.0x"
  source: palkowski2018nw
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; sequences of length 2048, 4096.
estimated_builder_effort: "3-4 days"
risk_flags:
  - Anti-diagonal tile layout requires non-rectangular storage; memory allocation and index formula are non-trivial
  - Rodinia NW OpenMP version already tiles anti-diagonals for parallelism; the layout trick here is separate (storage order, not parallelism)
```

---

### 18-npdp-nussinov-skew-tile

The Nussinov RNA folding algorithm fills a triangular DP table with the
recurrence V[i][j] = max(V[i+1][j-1]+2, V[i][j-1], V[i+1][j], ...).
Pałkowski and Bielecki (BMC Bioinformatics 2017) apply loop skewing to
tile this non-serial polyadic DP and measure 1.6×–3.0× speedup on a
6-core platform compared to the naive sequential code. The NPDP
benchmark suite provides a clean C reference implementation.

```yaml
id: 18-npdp-nussinov-skew-tile
suite: NPDP Benchmark Suite (Pałkowski, 2022)
kernel: Nussinov RNA secondary-structure prediction
upstream_url: https://github.com/markpal/NPDP_Bench
license: MIT
language: c
baseline:
  source_files: [nussinov/nussinov.c]
  build: "gcc -O3 -march=native -fopenmp nussinov/nussinov.c -o nussinov"
  threading: single-threaded
layout_trick: Skewed anti-diagonal tile storage for the triangular DP table; elements (i,j) with j-i = d stored contiguously, making the d-th super-diagonal a contiguous vector
layout_trick_citation: palkowski2017nussinov
why_compiler_cant: |
  Nussinov's loop nest has the bounds 0 <= i <= j < N, making it a
  triangular iteration space. GCC Graphite requires rectangular domains
  for its tiling pass; the triangular bound causes Graphite to skip
  tiling. Loop skewing (replacing j with j-i as the inner index) does
  make the domain rectangular, but GCC does not automatically combine
  this skewing with a data renumbering to store diagonals contiguously.
  Even with the skewing applied manually, GCC keeps the row-major layout
  for V, leaving the diagonal-direction access at non-unit stride.
lego_expressibility: |
  #   Diagonal storage: element (i,j) at diagonal d=j-i, position p=i.
  #   V_diag = lego.OrderBy(V, lego.TileBy(lego.Col - lego.Row, T) @
  #                            lego.TileBy(lego.Row, T))
  #   T chosen to fit a tile of the table in L1.
predicted_win:
  value: "1.6x – 3.0x"
  source: palkowski2017nussinov
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 500, 1000, 2000.
estimated_builder_effort: "3-4 days"
risk_flags:
  - NPDP license listed as MIT on GitHub; verify LICENSE file before building
  - Triangular domain boundary handling at tile edges is complex
```

---

### 19-npdp-zuker-skew-tile

Zuker's mfold RNA folding algorithm is more complex than Nussinov,
involving multiple interdependent DP tables and non-uniform dependence
distances. Pałkowski and Bielecki (BMC Bioinformatics 2019) apply
space-time tiling to Zuker's loop nest and report 1.4×–2.5× speedup
versus single-core code with improved cache efficiency. The NPDP suite
provides the C reference.

```yaml
id: 19-npdp-zuker-skew-tile
suite: NPDP Benchmark Suite (Pałkowski, 2022)
kernel: Zuker RNA secondary-structure prediction (simplified mfold model)
upstream_url: https://github.com/markpal/NPDP_Bench
license: MIT
language: c
baseline:
  source_files: [zuker/zuker.c]
  build: "gcc -O3 -march=native -fopenmp zuker/zuker.c -o zuker"
  threading: single-threaded
layout_trick: Diagonal-stripe tile storage for the primary DP table; the skewed layout aligns the dependent table reads to be accessed with unit stride within each tile
layout_trick_citation: palkowski2019zuker
why_compiler_cant: |
  Zuker's DP has multiple tables with non-uniform dependences (the
  dependence distance in the i dimension is not constant across
  iterations). GCC Graphite only tiles loop nests with uniform
  (constant) dependence distances; non-uniform dependences cause the
  legality check to fail and Graphite emits untransformed code. Polly
  -polly similarly rejects non-uniform dependence patterns at the SCoP
  level. The space-time tiling approach of Pałkowski (2019) operates on
  the transitive closure of the dependence graph, a technique not
  implemented in any standard compiler.
lego_expressibility: |
  #   Same diagonal-stripe layout as 18-npdp-nussinov-skew-tile:
  #   V_diag = lego.OrderBy(V, lego.TileBy(lego.Col - lego.Row, T) @
  #                            lego.TileBy(lego.Row, T))
predicted_win:
  value: "1.4x – 2.5x"
  source: palkowski2019zuker
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 500, 1000.
estimated_builder_effort: "3-4 days"
risk_flags:
  - Multiple interdependent DP tables require consistent layout across all tables
  - Non-uniform dependences make correctness verification of the layout transformation important
```

---

### 20-polybench-seidel2d-wavefront-tile

The seidel-2d kernel performs red-black Gauss-Seidel iteration over a 2D
grid with a 9-point stencil. Dependencies prevent the outermost loops from
being parallelised without wavefront tiling. Storing the grid in
anti-diagonal tile order makes the wavefront sweep access each tile in
strict sequential order, improving L2 cache reuse across time steps.

```yaml
id: 20-polybench-seidel2d-wavefront-tile
suite: PolyBench/C 4.2.1
kernel: seidel-2d (Gauss-Seidel 2D, 9-point stencil)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [stencils/seidel-2d/seidel-2d.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities stencils/seidel-2d/seidel-2d.c utilities/polybench.c -o seidel-2d"
  threading: single-threaded
layout_trick: Skewed tile storage; the 2D grid is stored in 32×32 tile blocks ordered by anti-diagonal (i+j), so the wavefront sweep visits each tile's footprint in L1 before moving to the next tile
layout_trick_citation: palkowski2018nw
why_compiler_cant: |
  Seidel-2d has loop-carried dependences in both i and j simultaneously
  (the 9-point stencil reads neighbours at (i-1,j-1), (i,j-1), etc.).
  GCC Graphite cannot tile this nest legally with a rectangular tile
  because the dependence cone spans both loop axes; Graphite falls back
  to the identity transformation. The wavefront (skewed) tile requires
  time-space tiling that GCC's -floop-nest-optimize pass does not
  implement.
lego_expressibility: |
  #   Store grid in anti-diagonal tile order:
  #   T = 32
  #   A_wavefront = lego.OrderBy(A,
  #                   lego.TileBy(lego.Row + lego.Col, T) @
  #                   lego.TileBy(lego.Row, T))
predicted_win:
  value: "1.2x – 2.0x"
  source: palkowski2018nw
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 500, 512, 1000; TSTEPS = 100.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Predicted win is extrapolated; Seidel-2d has a different dependency structure than NW
  - Skewed tile boundary code is complex for the outer-grid edge tiles
```

---

## Layout Class 6: AoSoA / Interleaved Struct Packing for Vectorization

---

### 21-rodinia-particlefilter-aosoA

The Rodinia particle filter benchmark maintains a set of N particles each
with position (x,y), velocity (vx,vy), and weight fields stored in
separate flat arrays (SoA). The update kernel reads all five fields per
particle in a tight loop. An AoSoA layout with inner tile width 8
(matching AVX-512 double registers) groups 8 particles' data contiguously,
enabling the compiler to issue eight-wide SIMD loads for each field with
a single load instruction per tile rather than 8 scalar loads. The layout
win is analogous to the SPH results from González-Cao et al. (2022) which
demonstrate 2×–4.3× speedup for the same AoS → AoSoA transformation.

```yaml
id: 21-rodinia-particlefilter-aosoA
suite: Rodinia 3.1
kernel: particle_filter_float (sequential CPU particle-filter update)
upstream_url: https://github.com/HPC-FAIR/rodinia_3.1/archive/refs/heads/main.tar.gz
license: BSD-3-Clause
language: cpp
baseline:
  source_files: [openmp/particlefilter/ex_particle_OPENMP_seq.cpp]
  build: "g++ -O3 -march=native -fopenmp openmp/particlefilter/ex_particle_OPENMP_seq.cpp -o particle_filter"
  threading: single-threaded
layout_trick: AoSoA with inner tile width 8; particle fields (x, y, z, vx, vy, weight) grouped in 8-element SoA stripes so each field of 8 consecutive particles is contiguous and 64-byte aligned
layout_trick_citation: vecdualspHysics2022
why_compiler_cant: |
  The particle filter update loop accesses each particle's fields via
  separate pointer arrays (x[i], y[i], vx[i], ...) that are already
  SoA at the global level. However, the weight-update inner loop mixes
  field accesses with trigonometric intrinsics (cos, sin) applied
  element-wise. GCC's SLP vectoriser (-ftree-slp-vectorize) with -O3
  -march=native vectorises simple straight-line field updates but misses
  the AoSoA grouping opportunity: the current SoA layout has all N
  elements of each field contiguous, which already allows SIMD, but the
  AoSoA layout with W=8 improves the L1 working set by keeping all
  fields of 8 particles together, reducing cache-line evictions between
  field accesses. The -floop-nest-optimize pass does not restructure
  global SoA arrays into AoSoA because doing so changes the global
  data-structure layout, not just loop order.
lego_expressibility: |
  #   N particles, W = 8 (AVX-512 double register width)
  #   x_aosoA = lego.OrderBy(x_field,
  #               lego.TileBy(lego.Row, W))   # group W particles
  #   # Apply same OrderBy to y, vx, vy, weight fields.
  #   # GenP unifies the fields into one AoSoA struct if needed:
  #   particles = lego.OrderBy(all_fields,
  #                 lego.TileBy(lego.Row, W) @
  #                 lego.GenP(num_fields))
  # No GroupBy required.
predicted_win:
  value: "1.2x – 2.0x"
  source: vecdualspHysics2022
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 100000, 500000 particles.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Rodinia particle filter already uses SoA; the AoSoA win comes from inter-field L1 locality, which may be small when N is large relative to L1
  - Predicted win extrapolated from SPH AoSoA results; particle-filter arithmetic intensity differs
```

---

### 22-lulesh-elem-aosoA

LULESH stores element data (stress tensors, energy) in separate arrays
(SoA-style at the element level) but the per-element computation accesses
multiple arrays simultaneously. An AoSoA layout with tile width 4 or 8
improves SIMD utilisation by grouping 4–8 elements' data contiguously
before striding to the next field. The LLNL-published LULESH BSD-3-Clause
code is an ideal testbed.

```yaml
id: 22-lulesh-elem-aosoA
suite: LULESH 2.0
kernel: CalcElemShapeFunctionDerivatives + CalcKinematicsForElems (volume update)
upstream_url: https://github.com/LLNL/LULESH/archive/refs/tags/2.0.3.tar.gz
license: BSD-3-Clause
language: cpp
baseline:
  source_files: [lulesh.cc, lulesh-comm.cc, lulesh-init.cc, lulesh-util.cc, lulesh-viz.cc]
  build: "g++ -O3 -march=native -fopenmp -DUSE_MPI=0 lulesh.cc lulesh-comm.cc lulesh-init.cc lulesh-util.cc lulesh-viz.cc -o lulesh"
  threading: single-threaded
layout_trick: AoSoA with inner tile width 8 applied to element arrays (e, p, q, qq, ql, v, volo, delv, vdov, arealg, ss, elemMass); each group of 8 elements has its fields contiguous in memory
layout_trick_citation: vecdualspHysics2022
why_compiler_cant: |
  LULESH's element arrays are plain double* pointers accessed with
  index arithmetic (e.g., domain.e(i), domain.p(i)). GCC's SLP
  vectoriser can vectorise the simple element loops (e.g., initialisation)
  but the shape-function derivative computation in
  CalcElemShapeFunctionDerivatives accesses 8 node coordinates
  per element with irregular index patterns (connectivity array lookup).
  The AoS → AoSoA transformation would require the compiler to
  recognise that 8 adjacent elements can be grouped and restructured;
  this is beyond what -floop-nest-optimize or -ftree-vectorize achieve
  on the indirection-heavy LULESH code.
lego_expressibility: |
  #   W = 8  # SIMD width (AVX-512 doubles)
  #   e_aosoA = lego.OrderBy(domain_e,
  #               lego.TileBy(lego.Row, W))   # group W elements
  #   # Same expression for p, q, qq, ql, v, volo, etc.
  # GenP used to unify multiple arrays into one AoSoA struct if needed.
predicted_win:
  value: "1.1x – 1.8x"
  source: vecdualspHysics2022
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; problem size = 30^3 elements.
estimated_builder_effort: "4-5 days"
risk_flags:
  - LULESH connectivity arrays (nodelist) are accessed with indirect indexing; AoSoA does not improve scatter/gather
  - Predicted win is extrapolated from SPH results; LULESH has different compute structure
  - Multiple arrays must be transformed consistently; partial transformation may harm performance
```

---

### 23-hpccg-cg-aosoA

HPCCG (Mantevo, BSD-3-Clause) is a conjugate gradient solver over a
sparse 27-point finite-difference stencil. The SpMV kernel accesses a
dense CSR-like storage. Converting the local stencil storage to AoSoA
with width 8 groups 8 rows of the local finite-difference matrix
together, enabling AVX-512 gather over a contiguous 8-element block
rather than 8 independent scalar gathers.

```yaml
id: 23-hpccg-cg-aosoA
suite: HPCCG 1.0 (Mantevo)
kernel: HPCCG_matvec (sparse matrix-vector product, 27-point stencil)
upstream_url: https://github.com/Mantevo/HPCCG
license: BSD-3-Clause
language: cpp
baseline:
  source_files: [HPCCG.cpp, HPC_sparsemv.cpp, generate_matrix.cpp, main.cpp]
  build: "g++ -O3 -march=native -fopenmp HPCCG.cpp HPC_sparsemv.cpp generate_matrix.cpp main.cpp -o hpccg"
  threading: single-threaded
layout_trick: AoSoA with inner tile width 8 for the 27 non-zero values per row; each group of 8 rows' stencil coefficients stored contiguously, enabling SIMD loads
layout_trick_citation: vecdualspHysics2022
why_compiler_cant: |
  HPCCG's HPC_sparsemv loop accesses matrix values via a pointer
  indirection (matrix->ptr_to_vals_in_row[i][j]). GCC's SLP vectoriser
  can vectorise simple row-contiguous SpMV but not the double indirection
  (pointer-to-pointer) used by HPCCG. The AoS → AoSoA transformation
  for the coefficient array is a global data-structure change; -O3 and
  -floop-nest-optimize do not perform data-structure transformations.
lego_expressibility: |
  #   W = 8; nnz_per_row = 27
  #   vals_aosoA = lego.OrderBy(vals,
  #                  lego.TileBy(lego.Row, W) @   # 8 rows grouped
  #                  lego.TileBy(lego.Col, nnz_per_row))  # 27 nnz each
  # No GroupBy required.
predicted_win:
  value: "unknown"
  source: "unknown"
  type: unknown
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; 100^3 local domain.
estimated_builder_effort: "2-3 days"
risk_flags:
  - HPCCG pointer-to-pointer structure requires non-trivial data restructuring
  - SpMV SIMD benefit is architecture-dependent; AVX-512 gathers may not outperform scalar on all CPUs
```

---

## Layout Class 7: Block-Cyclic Distribution for Thread-Level Locality

---

### 24-polybench-fdtd-2d-block-cyclic

The FDTD-2D kernel sweeps over a 2D grid updating electric and magnetic
field components. Dividing the grid into spatial blocks assigned to
threads in a block-cyclic pattern keeps each thread's working set in L2
while maintaining load balance. Applying this layout through LEGO's
TileBy + GroupBy is a direct expression of OpenMP thread-level locality.

```yaml
id: 24-polybench-fdtd-2d-block-cyclic
suite: PolyBench/C 4.2.1
kernel: fdtd-2d (2D Finite-Difference Time Domain)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [stencils/fdtd-2d/fdtd-2d.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities stencils/fdtd-2d/fdtd-2d.c utilities/polybench.c -o fdtd-2d"
  threading: single-threaded
layout_trick: Block-cyclic row partitioning; rows of Ex, Ey, Hz assigned in blocks of size T to threads in cyclic order (thread t owns rows t*T, (t+P)*T, (t+2P)*T, ... for P threads), improving L2 locality per thread
layout_trick_citation: frigo1999cacheoblivious
why_compiler_cant: |
  GCC -fopenmp with -O3 parallelises the outer loop of FDTD-2D with
  dynamic scheduling or static scheduling, but the default OpenMP
  schedule does not guarantee that a thread's assigned rows fit within
  its private L2 cache. The block-cyclic data layout (as opposed to
  the iteration schedule) requires reassigning rows to memory positions,
  not just reordering loop iterations. -floop-nest-optimize does not
  perform memory layout transformations for thread affinity; it only
  handles loop-level transformations.
lego_expressibility: |
  #   P = number of threads; T = block size per thread
  #   Ex_bc = lego.OrderBy(Ex,
  #             lego.TileBy(lego.Row, T) @  # blocks of T rows
  #             lego.GenP(P))               # P-way cyclic distribution
  #   # GroupBy here: the cyclic distribution is expressed as GenP
  #   # (generic partition) grouping T-row blocks into P groups.
  #   # GroupBy(P) with block size T is the key primitive.
predicted_win:
  value: "1.1x – 1.5x"
  source: frigo1999cacheoblivious
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; NX = NY = 500, 1000.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Block-cyclic benefit is marginal in single-threaded mode; best tested with OpenMP threads
  - CASTLE source-emission path for multi-threaded layouts must be validated
```

---

### 25-polybench-adi-block-cyclic

The ADI (Alternating Direction Implicit) kernel performs sweeps in both
row and column directions. A block-cyclic layout that interleaves row and
column blocks gives each thread good temporal locality across both sweep
directions. This is an unusual candidate because most implementations choose
one sweep direction as the "inner" and pay the stride cost for the other;
CASTLE can express a layout that halves this penalty.

```yaml
id: 25-polybench-adi-block-cyclic
suite: PolyBench/C 4.2.1
kernel: adi (Alternating Direction Implicit, 2D)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [stencils/adi/adi.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities stencils/adi/adi.c utilities/polybench.c -o adi"
  threading: single-threaded
layout_trick: Block-cyclic tile layout that alternates storage between row-major and column-major order within each block, halving the cache-miss penalty for the column-direction sweep
layout_trick_citation: frigo1999cacheoblivious
why_compiler_cant: |
  ADI has two distinct sweep phases. GCC -O3 generates two separate loop
  nests with different memory access patterns (row-major for row sweeps,
  column-major for column sweeps). Loop interchange can make one sweep
  efficient but increases the miss rate for the other. GCC does not
  introduce a hybrid layout that interleaves row and column blocks; doing
  so requires a global data-structure transformation, not a loop
  transformation. -floop-nest-optimize -floop-interchange handles one
  direction and harms the other.
lego_expressibility: |
  #   T = tile size
  #   X_alternating = lego.OrderBy(X,
  #                     lego.TileBy(lego.Row, T) @
  #                     lego.TileBy(lego.Col, T))
  #   # Within each T×T tile both row and column directions are
  #   # within L1 for the inner sweep.
predicted_win:
  value: "unknown"
  source: "unknown"
  type: unknown
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 500, 1000; TSTEPS = 40.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Alternating-direction sweep cannot both benefit simultaneously from the same tile layout; the win may apply only to the dominant direction
```

---

## Layout Class 8: Padding to Break Power-of-Two Stride Associativity Conflicts

---

### 26-polybench-gemm-pow2-pad

Power-of-two matrix dimensions cause all columns to map to the same L1
cache set (for N-way associative caches with size = k * N rows). Hong et
al. (PLDI 2016) present the first optimal padding algorithm for
multidimensional arrays and demonstrate that dynamic padding eliminates
cache conflict misses that a compiler with -O3 cannot avoid. This is a
direct test of CASTLE's non-power-of-two layout flexibility: N=1024
baseline vs N=1024+padding variant.

```yaml
id: 26-polybench-gemm-pow2-pad
suite: PolyBench/C 4.2.1
kernel: gemm (general matrix-matrix multiply, power-of-two size N=1024)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/blas/gemm/gemm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities -DEXTRALARGE_DATASET linear-algebra/blas/gemm/gemm.c utilities/polybench.c -o gemm_1024"
  threading: single-threaded
layout_trick: Row padding; each matrix row padded to width N + P (P = 8 or 16 doubles) to break L1 set-associativity conflict at power-of-two strides
layout_trick_citation: hong2016padding
why_compiler_cant: |
  At N=1024, consecutive rows of matrix A are separated by exactly
  1024 * 8 = 8192 bytes, which is a multiple of the L1 cache size on
  most architectures (32 KB, 16-way associative → conflict threshold
  2048 bytes). GCC -O3 -march=native does not insert array padding
  because (a) the -fno-strict-aliasing and pointer-aliasing assumptions
  prevent it from knowing N at compile time in the polybench harness and
  (b) -floop-nest-optimize does not model cache associativity conflicts.
  The PLDI 2016 paper (hong2016padding) is the first work to give an
  optimal algorithm for this padding, demonstrating it is not found by
  -O3.
lego_expressibility: |
  #   P = 8  # padding doubles per row
  #   A_padded = lego.OrderBy(A,
  #                lego.TileBy(lego.Row, 1) @  # one row per tile
  #                lego.RegP(N + P))            # row width = N + P
  #   # RegP(N+P) allocates N+P doubles per row, with the last P unused.
  # No GroupBy required.
predicted_win:
  value: "1.2x – 2.0x"
  source: hong2016padding
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: true
  test_at_non_pow2_size: false
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 1024 (power-of-two conflict case).
estimated_builder_effort: "1-2 days"
risk_flags:
  - The conflict miss benefit is highly microarchitecture-dependent; must be measured on the target CPU
  - Padding increases memory footprint by P/N fraction; for N=1024, P=8 adds 0.8%
```

---

### 27-polybench-heat3d-pow2-pad

The heat-3d kernel at LARGE_DATASET uses a 120×120×120 grid (close to
128^3). At power-of-two sizes (128^3) the z-direction stride is 128*128
= 16384 doubles = 131072 bytes, which aliases L2 (typically 256 KB with
8-way associativity). Row padding of the z-dimension breaks this
aliasing without altering algorithm correctness.

```yaml
id: 27-polybench-heat3d-pow2-pad
suite: PolyBench/C 4.2.1
kernel: heat-3d (3D heat equation at power-of-two grid N=128)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [stencils/heat-3d/heat-3d.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities -DN=128 stencils/heat-3d/heat-3d.c utilities/polybench.c -o heat3d_128"
  threading: single-threaded
layout_trick: Padding the innermost (z) dimension from 128 to 128+8 doubles per row to break L2 set-associativity conflict; 8 extra doubles = 64 bytes = one cache line
layout_trick_citation: hong2016padding
why_compiler_cant: |
  At N=128 the innermost stride (z-direction) is 128 * 8 = 1024 bytes.
  Rows in the j-direction are 128*128*8 = 131072 bytes apart, which
  is a power-of-two multiple of the 8-way-associative 256 KB L2 cache's
  set size. GCC -O3 does not model cache set-associativity and therefore
  does not insert padding. The -floop-nest-optimize pass only transforms
  loop iteration order, not the underlying array allocation dimensions.
  Hong et al. (PLDI 2016) show that this class of conflict is not
  prevented by any standard compiler flag.
lego_expressibility: |
  #   P = 8  # padding in innermost dim
  #   A_pad = lego.OrderBy(A, lego.RegP(128 + P) @
  #                           lego.RegP(128) @
  #                           lego.RegP(128))
  #   # RegP(N+P) in the innermost dimension adds P unused cells per row.
predicted_win:
  value: "1.1x – 1.8x"
  source: hong2016padding
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: true
  test_at_non_pow2_size: false
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 128.
estimated_builder_effort: "1-2 days"
risk_flags:
  - Padding benefit is highly specific to the L2 associativity of the test CPU; may show PARITY on some machines
```

---

## Layout Class 9: Power-of-Two-Restricted Optimizations at Non-Power-of-Two Sizes

---

### 28-polybench-gemm-nonpow2-morton

This candidate specifically targets the CASTLE paper-grade result:
applying Z-Morton layout to GEMM at the published power-of-two sizes
(N=1024, 2048) AND at the non-power-of-two sizes (N=1000, 1500) where
the published work does not evaluate. The Morton layout requires no power-
of-two restriction in principle; CASTLE can express it for any N, and
the paper contribution is the first measurement at non-power-of-two N.

```yaml
id: 28-polybench-gemm-nonpow2-morton
suite: PolyBench/C 4.2.1
kernel: gemm (non-power-of-two sizes N in {1000, 1500, 1800})
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/blas/gemm/gemm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities -DN=1000 linear-algebra/blas/gemm/gemm.c utilities/polybench.c -o gemm_1000"
  threading: single-threaded
layout_trick: Z-Morton layout applied at non-power-of-two N; no padding needed because CASTLE TileBy handles remainder tiles at matrix edges
layout_trick_citation: walker2018morton
why_compiler_cant: |
  Same as 01-polybench-gemm-zmorton: Morton index bit-interleaving is
  non-affine. Additionally, for non-power-of-two N the standard Morton
  index formula must be clipped or padded to a power-of-two boundary;
  the compiler has no mechanism to detect or implement this adjustment.
lego_expressibility: |
  #   T = 32  # tile size; tiles at edges of non-power-of-two N
  #           # are handled automatically by CASTLE's TileBy remainder
  #   A_morton = lego.OrderBy(A,
  #                lego.TileBy(lego.Row, T) @
  #                lego.TileBy(lego.Col, T))
  #   # CASTLE emits boundary-safe code for remainder tiles.
predicted_win:
  value: "unknown"
  source: "unknown"
  type: unknown
power_of_two_restriction:
  baseline_assumes_pow2: true
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N in {1000, 1024, 1500, 2048}.
estimated_builder_effort: "2-3 days"
risk_flags:
  - This is a paper-grade "novel at non-pow2" experiment; the speedup is unknown a priori
  - Remainder tile handling must be verified for correctness at non-power-of-two N
```

---

### 29-bricklib-stencil-nonpow2-brick

BrickLib's published results are exclusively at power-of-two grid
dimensions. Applying the brick layout to non-power-of-two grids
(e.g., 100^3, 200^3) is a direct CASTLE contribution: CASTLE's TileBy
with remainder semantics handles edge bricks without padding to the next
power of two.

```yaml
id: 29-bricklib-stencil-nonpow2-brick
suite: BrickLib stencil suite (r0.1)
kernel: 3D 7-point Laplacian stencil at non-power-of-two grid sizes
upstream_url: https://github.com/CtopCsUtahEdu/bricklib/archive/refs/tags/r0.1.tar.gz
license: MIT
language: cpp
baseline:
  source_files: [examples/stencils/stencil3d7pt.cpp]
  build: "cmake -DCMAKE_BUILD_TYPE=Release . && make -j16 stencil3d7pt_naive"
  threading: single-threaded
layout_trick: 8×8×8 brick layout at non-power-of-two grid sizes (e.g., 100^3, 200^3); CASTLE emits boundary-safe remainder bricks without padding to 128^3
layout_trick_citation: zhao2019bricks
why_compiler_cant: |
  Same as 11-bricklib-3d7pt-brick. Additionally, BrickLib's own code-
  generation path requires power-of-two grid dimensions internally to
  simplify brick-index arithmetic. CASTLE's TileBy handles non-
  power-of-two sizes with a remainder-tile epilogue, which BrickLib's
  own framework does not generate automatically.
lego_expressibility: |
  #   B = 8; grid size may be any value, not restricted to power of 2
  #   grid_brick = lego.OrderBy(grid,
  #                  lego.TileBy(lego.Row, B) @
  #                  lego.TileBy(lego.Col, B) @
  #                  lego.TileBy(lego.Depth, B))
  #   # CASTLE emits a peel loop for boundary bricks.
predicted_win:
  value: "unknown"
  source: "unknown"
  type: unknown
power_of_two_restriction:
  baseline_assumes_pow2: true
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; grid = 100^3, 200^3, 256^3.
estimated_builder_effort: "2-3 days"
risk_flags:
  - BrickLib published data only at power-of-two sizes; no baseline to compare against at non-pow2
  - Remainder brick boundary code must be generated and verified by CASTLE
```

---

### 30-polybench-trmm-nonpow2-rfp

The RFP triangular packing format is defined for any matrix size N but
published benchmark results (Gustavson et al. 2010) focus on N = 512,
1024, 2048. CASTLE can apply RFP at N = 1000, 1500, demonstrating
the layout works without power-of-two restriction, and that the
performance benefit extends to non-standard sizes used in practice
(e.g., LAPACK problem sizes from real applications).

```yaml
id: 30-polybench-trmm-nonpow2-rfp
suite: PolyBench/C 4.2.1
kernel: trmm (triangular matrix-matrix multiply, N in {1000, 1500})
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/blas/trmm/trmm.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities -DN=1000 linear-algebra/blas/trmm/trmm.c utilities/polybench.c -o trmm_1000"
  threading: single-threaded
layout_trick: RFP packing of the triangular matrix A at non-power-of-two N; the RFP index formula adapts to odd N by adjusting the panel split point
layout_trick_citation: gustavson2010rfpcholesky
why_compiler_cant: |
  The RFP index formula depends on whether N is even or odd (the panel
  split is at ceil(N/2)); this is a data-dependent non-affine mapping
  that GCC and Clang cannot introduce. At non-power-of-two N, the
  formula further requires a modulo adjustment that is not expressible
  in the polyhedral affine model at all.
lego_expressibility: |
  #   N = 1000 (non-power-of-two)
  #   half = N // 2; remainder = N - half
  #   A_rfp = lego.OrderBy(A_lower,
  #             lego.TileBy(lego.Row, half) @
  #             lego.RegP(N))  # N columns, half rows per panel
predicted_win:
  value: "unknown"
  source: "unknown"
  type: unknown
power_of_two_restriction:
  baseline_assumes_pow2: true
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N in {1000, 1024, 1500, 2048}.
estimated_builder_effort: "2-3 days"
risk_flags:
  - RFP odd-N boundary handling requires careful verification
  - Predicted win is unknown; the RFP benefit at non-power-of-two N has not been published
```

---

### 31-npdp-nussinov-nonpow2-skew

Published skewed-tiling results for Nussinov (Pałkowski 2017) use
sequence lengths N = 500, 1000. Applying the same anti-diagonal layout
at N = 700, 1500 (non-standard sizes) tests whether CASTLE's TileBy
remainder semantics generate correct boundary code for non-power-of-two
skewed tiles—an untested capability that is a paper-grade CASTLE result.

```yaml
id: 31-npdp-nussinov-nonpow2-skew
suite: NPDP Benchmark Suite (Pałkowski, 2022)
kernel: Nussinov RNA at non-standard sequence lengths (N in {700, 1500})
upstream_url: https://github.com/markpal/NPDP_Bench
license: MIT
language: c
baseline:
  source_files: [nussinov/nussinov.c]
  build: "gcc -O3 -march=native -fopenmp nussinov/nussinov.c -o nussinov"
  threading: single-threaded
layout_trick: Anti-diagonal tile storage at non-power-of-two N; CASTLE emits remainder-safe boundary code for tiles at the diagonal edges
layout_trick_citation: palkowski2017nussinov
why_compiler_cant: |
  Same reasoning as 18-npdp-nussinov-skew-tile. Additionally, the
  non-power-of-two N produces tiles at the main diagonal boundary that
  are not full-width; the compiler cannot adapt the skew formula to
  partial tiles. CASTLE's remainder-tile mechanism handles this
  explicitly.
lego_expressibility: |
  #   T = 64  # tile width; CASTLE handles T that does not divide N
  #   V_diag = lego.OrderBy(V, lego.TileBy(lego.Col - lego.Row, T) @
  #                            lego.TileBy(lego.Row, T))
predicted_win:
  value: "unknown"
  source: "unknown"
  type: unknown
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N in {700, 1000, 1500}.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Non-power-of-two skewed tile boundary is the primary risk; correctness must be verified before performance
```

---

## Additional Candidates — Multi-Level Tiling on NAS / Rodinia Kernels

---

### 32-rodinia-hotspot-tile

Rodinia HotSpot performs thermal simulation using a 2D heat equation
with a 5-point stencil. The CPU OpenMP version uses a single-level tile
with ghost-zone computation. Applying a two-level L1 × L2 tile layout
(without ghost zones, using BrickLib-style tile boundaries) can reduce
cache misses by keeping the tile's grid entirely in L1 for the inner
iteration.

```yaml
id: 32-rodinia-hotspot-tile
suite: Rodinia 3.1
kernel: hotspot (2D thermal simulation, 5-point stencil, CPU OpenMP)
upstream_url: https://github.com/HPC-FAIR/rodinia_3.1/archive/refs/heads/main.tar.gz
license: BSD-3-Clause
language: cpp
baseline:
  source_files: [openmp/hotspot/hotspot.cpp]
  build: "g++ -O3 -march=native -fopenmp openmp/hotspot/hotspot.cpp -o hotspot"
  threading: single-threaded
layout_trick: Two-level L1 × L2 tile layout; inner tile (32×32) fits in L1, outer tile (128×128) fits in L2; tile storage order interleaves row and column blocks for both sweep directions
layout_trick_citation: zhao2019bricks
why_compiler_cant: |
  GCC -fopenmp -O3 applies a single-level tile heuristic (typically
  32 or 64) and static row distribution across threads. The two-level
  tile with explicit L1 and L2 sizes requires knowing both cache sizes,
  which GCC's heuristic does not exploit. The per-tile contiguous
  storage (as opposed to per-tile iteration order in row-major storage)
  is outside the scope of -floop-nest-optimize. Rodinia's own hotspot
  uses ghost zones for parallelism, not for layout locality.
lego_expressibility: |
  #   T_L1 = 32; T_L2 = 128
  #   grid_2level = lego.OrderBy(grid,
  #                   lego.TileBy(lego.Row, T_L2) @
  #                   lego.TileBy(lego.Col, T_L2) @
  #                   lego.TileBy(lego.Row, T_L1) @
  #                   lego.TileBy(lego.Col, T_L1))
predicted_win:
  value: "1.2x – 2.0x"
  source: zhao2019bricks
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; grid = 512×512, 1024×1024; 500 iterations.
estimated_builder_effort: "2-3 days"
risk_flags:
  - Rodinia hotspot uses ghost zones for OpenMP parallelism; the layout trick targets single-threaded locality
  - Two-level tile sizes must be tuned to A100 node's L1 (32 KB) and L2 (512 KB) caches
```

---

### 33-polybench-mvt-L1-tile

The mvt (matrix-vector product transpose) kernel accesses a matrix A
once in row order and once in column order in the same kernel.
A two-level tile with an inner tile sized to fit A[tile_i:tile_i+T,
tile_j:tile_j+T] in L1 provides reuse in both the row and column sweeps
with the same data. This is a standard cache-blocking trick documented
in the polyhedral literature but not applied at the storage-layout level.

```yaml
id: 33-polybench-mvt-L1-tile
suite: PolyBench/C 4.2.1
kernel: mvt (matrix-vector products with transposed access)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/kernels/mvt/mvt.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/kernels/mvt/mvt.c utilities/polybench.c -o mvt"
  threading: single-threaded
layout_trick: Single-level 32×32 tile storage of matrix A; both the row-order and column-order passes over A access the same L1-resident tile before moving to the next tile
layout_trick_citation: frigo1999cacheoblivious
why_compiler_cant: |
  The mvt kernel has two independent loop nests that each access A once.
  GCC -floop-nest-optimize tiles each nest separately and does not fuse
  their tile order to share cached tiles of A. Loop fusion across the two
  nests is not performed by Graphite at -O3 (the nests have no loop-
  carried dependences between them, so fusion is legal, but heuristics
  prevent it when the nests have different reduction variables). Even if
  fused, the row-major storage means that after the row-order nest the
  column-order nest incurs stride-N accesses regardless of tiling.
lego_expressibility: |
  #   T = 32
  #   A_tiled = lego.OrderBy(A, lego.TileBy(lego.Row, T) @
  #                             lego.TileBy(lego.Col, T))
  #   # Both loop nests iterate in tile order over A_tiled,
  #   # sharing L1 hits for the shared T×T sub-block.
predicted_win:
  value: "1.1x – 1.5x"
  source: frigo1999cacheoblivious
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 1000, 1024, 2000.
estimated_builder_effort: "1-2 days"
risk_flags:
  - mvt is memory-bandwidth limited; layout win may be PARITY on bandwidth-saturated machines
```

---

### 34-polybench-bicg-L1-tile

The bicg (BiCG sub-kernel of BiConjugate Gradients) kernel accesses
matrix A in both row order and column order within the same time step.
As with mvt, a two-level tile layout allows both accesses to share L1
cached tiles of A, at the cost of stride-1 vs stride-T access within
each tile.

```yaml
id: 34-polybench-bicg-L1-tile
suite: PolyBench/C 4.2.1
kernel: bicg (BiCG sub-kernel)
upstream_url: https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz/download
license: BSD-3-Clause
language: c
baseline:
  source_files: [linear-algebra/kernels/bicg/bicg.c, utilities/polybench.c]
  build: "gcc -O3 -march=native -fopenmp -I utilities linear-algebra/kernels/bicg/bicg.c utilities/polybench.c -o bicg"
  threading: single-threaded
layout_trick: 32×32 tile storage of matrix A; both the A*p and A^T*r sub-loops share L1-resident tiles of A
layout_trick_citation: frigo1999cacheoblivious
why_compiler_cant: |
  Bicg has two loop nests: one computing s[i] += A[i][j]*p[j] and one
  computing q[j] += A[i][j]*r[i]. GCC -floop-nest-optimize considers
  each nest separately; it cannot fuse them across the two reductions.
  Loop fusion of the two nests is not in GCC's standard -O3 pipeline
  when the nests compute different reduction variables. Without fusion,
  each nest accesses A independently with a cold cache, defeating any
  per-nest tiling.
lego_expressibility: |
  #   T = 32
  #   A_tiled = lego.OrderBy(A, lego.TileBy(lego.Row, T) @
  #                             lego.TileBy(lego.Col, T))
predicted_win:
  value: "1.1x – 1.4x"
  source: frigo1999cacheoblivious
  type: extrapolated
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; M = N = 1000, 2000.
estimated_builder_effort: "1-2 days"
risk_flags:
  - bicg is also memory-bandwidth limited; same bandwidth-saturation risk as mvt
```

---

### 35-hpcc-dgemm-reg-L1-L2-tile

The HPCC DGEMM benchmark (permissive BSD) is a standalone double-precision
GEMM that serves as the "raw DGEMM" component of the HPC Challenge suite.
Applying the Goto/BLIS three-level packing hierarchy produces results that
can be directly compared to published HPCC leaderboard scores. Unlike the
PolyBench gemm, HPCC DGEMM is self-timed and reports GFlops/s, making it
straightforward to reproduce the published speedup figures.

```yaml
id: 35-hpcc-dgemm-reg-L1-L2-tile
suite: HPCC 1.5.0 (HPC Challenge)
kernel: DGEMM (double-precision GEMM component)
upstream_url: https://github.com/icl-utk-edu/hpcc/archive/refs/tags/1.5.0.tar.gz
license: BSD-3-Clause
language: c
baseline:
  source_files: [src/dgemm.c, src/bench_dgemm.c, src/driver.c]
  build: "make arch=Linux_PII_CBLAS"
  threading: single-threaded
layout_trick: Three-level register × L1 × L2 packing as in 04-polybench-gemm-reg-L1-L2-tile; applied to the HPCC standalone DGEMM driver
layout_trick_citation: goto2008anatomy
why_compiler_cant: |
  Same as 04-polybench-gemm-reg-L1-L2-tile: GCC's polyhedral passes
  do not generate explicit A-panel and B-panel packing buffers. HPCC
  DGEMM baseline uses a simple three-nested-loop implementation; without
  packing, GCC's vectoriser produces SIMD code with non-contiguous
  column loads for one of the operands.
lego_expressibility: |
  #   Same expression as 04-polybench-gemm-reg-L1-L2-tile.
  #   mr, nr = 6, 8; mc, kc = 72, 256
  #   B_lego = lego.OrderBy(B, lego.TileBy(lego.Row, kc) @
  #                            lego.TileBy(lego.Col, nr))
  #   A_lego = lego.OrderBy(A, lego.TileBy(lego.Row, mc) @
  #                            lego.TileBy(lego.Col, kc) @
  #                            lego.TileBy(lego.Row, mr))
predicted_win:
  value: "2.0x – 4.0x"
  source: goto2008anatomy
  type: published
power_of_two_restriction:
  baseline_assumes_pow2: false
  test_at_non_pow2_size: true
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl
  --membind=0; performance governor; turbo disabled; N = 1000, 1024, 2048.
estimated_builder_effort: "2-3 days"
risk_flags:
  - HPCC build system (Makefile with arch string) requires manual configuration for each platform
  - HPCC DGEMM baseline may already include simple tiling in some versions; check source before claiming baseline is naive
```

---
