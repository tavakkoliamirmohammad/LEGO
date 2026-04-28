# CASTLE CPU Evaluation Survey Summary

Scout output — drop-list, class breakdown, and survivor count.

---

## Layout Classes Represented (with candidate counts)

| Class | Layout Trick | Candidates |
|-------|-------------|-----------|
| 1 | Cache-Oblivious Recursive Layouts (Z-Morton, Hilbert) | 3 |
| 2 | Multi-Level Cache-Conscious Tiling (register × L1 × L2) | 7 |
| 3 | Recursive Bricking for Stencils (BrickLib-style) | 4 |
| 4 | Triangular / Symmetric Packing (RFP-style) | 2 |
| 5 | Skewed / Shifted Layouts (DP wavefronts) | 4 |
| 6 | AoSoA / Interleaved Struct Packing | 3 |
| 7 | Block-Cyclic Distribution for Thread Locality | 2 |
| 8 | Padding to Break Power-of-Two Stride Conflicts | 2 |
| 9 | Power-of-Two-Restricted Optimizations at Non-Pow2 Sizes | 4 |

**Total survivors: 31**

(Candidates 23, 25, 28-31 carry `predicted_win.type: unknown`; they
remain in the survey as paper-grade "novel measurement" candidates.)

---

## Layout Classes With No Survivors (and Why)

### Hilbert-Curve Layouts
Hilbert-curve storage gives slightly better cache behaviour than Z-Morton
for 3D stencils but requires a non-trivial recursive bit-manipulation
formula for address computation. Every published performance result for
Hilbert ordering is relative to Z-Morton (not row-major) and shows only
marginal improvement (< 5%) over Morton. No published paper reports a
Hilbert vs row-major speedup with a DOI-verifiable number. Dropped:
`predicted_win.type` would be `unknown` with no citation anchor.

### Block-Sparse / BCSR Matrix Formats
BCSR (Block Compressed Sparse Row) is a well-studied layout for SpMV
on matrices with block structure, but none of the eligible permissively-
licensed benchmark suites (PolyBench, Rodinia, HPCCG, BrickLib) contain
sparse matrices with the block structure needed to benefit from BCSR.
HPCCG is a structured 27-point stencil stored as a dense row; the SpMV
access pattern is regular enough that AoSoA is a better fit. Dropped.

### Space-Filling Curve Layouts for Unstructured Meshes (LULESH connectivity)
LULESH's connectivity arrays (nodelist, etc.) encode an unstructured mesh
where space-filling curve reordering of nodes improves locality. The
reordering requires a graph partitioning / renumbering step that is outside
CASTLE's layout algebra: LEGO's TileBy and OrderBy primitives operate on
rectangular index spaces, not on graph adjacency structures. Dropped:
`lego_expressibility` would require primitives outside the allowed set.

### Data-Dependent / Adaptive Layouts
Cache-oblivious B-trees and van Emde Boas tree layouts require pointer-
based data structures that are not representable as dense arrays. CASTLE
targets dense array-based kernels. Dropped.

### Fortran Array-of-Structures-of-Arrays in NAS BT / LU
NPB BT and LU are written in Fortran and access multi-component arrays
(5-component flow variables). AoSoA would benefit vectorisation. However,
NPB is released under the NASA Open Source Agreement (NOSA), which is
OSI-approved but is not in the accepted SPDX list (MIT, BSD-2-Clause,
BSD-3-Clause, Apache-2.0, ISC, public-domain, CC0). Dropped: license
not in the accepted list.

### MiniFE Sparse Finite-Element (Mantevo)
MiniFE performs unstructured finite-element assembly and solve. The
license is LGPL-3 (copyleft), which is not in the accepted list. Dropped.

---

## Kernels Considered and Dropped (with reasons)

| Kernel / Suite | Drop Reason |
|---------------|-------------|
| NPB BT (block tri-diagonal) | NOSA license not in accepted list |
| NPB LU (lower-upper Gauss-Seidel) | NOSA license not in accepted list |
| NPB MG (multi-grid) | NOSA license not in accepted list |
| NPB CG (conjugate gradient) | NOSA license not in accepted list |
| MiniFE (Mantevo finite-element) | LGPL-3 copyleft license |
| MiniGhost (Mantevo) | LGPL-3 copyleft license |
| LULESH connectivity reordering | Requires graph-partitioning primitive outside LEGO algebra |
| Hilbert-curve stencil | No DOI-verifiable published speedup over row-major baseline |
| Rodinia backprop (CPU) | Neural-network weights are row-major and the bottleneck is activation reuse, not layout; compiler vectorises inner loop with -O3; no layout-level published speedup |
| Rodinia k-nearest neighbour | k-NN search requires irregular memory access; no layout trick from the eligible classes applies; would require new MLIR ops for gather |
| Rodinia srad (CPU) | SRAD stencil uses non-uniform coefficients; the layout trick requires coefficient array AoSoA; compiler already vectorises the uniform part with -O3; published speedup not found with DOI |
| TBLIS (tensor contraction library) | License verified BSD-3-Clause (see 10-tblis-tensor-contraction-notranspose); retained |
| TCCG benchmark | License cannot be positively verified as permissive from public sources; marked as MIT tentatively in candidate 09; builder must reverify |
| DualSPHysics full suite | LGPL-2.1 copyleft; replaced by equivalent Rodinia particlefilter candidate |
| Polybench gramschmidt | Gram-Schmidt orthogonalisation accesses vectors sequentially with no spatial-locality deficit in the baseline; layout win is < 5% in the one published measurement found; too close to PARITY |
| Polybench gesummv | Matrix-vector multiply with two matrices; both accessed row-major with good spatial locality at -O3; no layout class from the eligible list provides a measurable win |
| Polybench correlation / covariance | These kernels are dominated by the outer GEMM-like step which is already covered by 04-polybench-gemm-reg-L1-L2-tile; adding them would duplicate the result without new layout insight |
| HPCC RandomAccess | Purely latency-bound (random 64-bit updates); no spatial layout trick reduces latency for random access; dropped |
| HPCC PTRANS | Matrix transpose; the in-place recursive Morton transpose is already covered by walker2018morton; a separate row would duplicate 01-polybench-gemm-zmorton |

---

## Total Count of Survivors

**31 candidates** across **9 layout classes**.

- Candidates with `predicted_win.type: published`: 17
- Candidates with `predicted_win.type: extrapolated`: 7
- Candidates with `predicted_win.type: unknown` (paper-grade novel experiments): 7

---

## Non-Obvious Choices and Notes for Orchestrator

1. **TCCG license (candidate 09):** The HPAC/tccg GitHub repository does
   not display an explicit LICENSE file in the main repository listing.
   The builder assigned to candidate 09 must check the repository root
   for a LICENSE or COPYING file before proceeding. If no permissive
   license is found, candidate 09 must be dropped.

2. **DualSPHysics (LGPL-2.1):** Was initially considered for AoSoA layout
   but dropped because LGPL-2.1 is copyleft and not in the accepted SPDX
   list. Replaced by candidate 21-rodinia-particlefilter-aosoA (BSD-3-Clause)
   which applies the same AoSoA layout trick from the same citation
   (vecdualspHysics2022) to a permissively-licensed kernel.

3. **`predicted_win.type: unknown` candidates (28-31):** These are
   deliberately included because they represent the core CASTLE paper
   contribution: layout-level transforms at non-power-of-two sizes where
   the existing published work only covers power-of-two sizes. A positive
   result here is a primary paper result, not a reproduction.

4. **NASA NOSA license:** NPB benchmarks use NOSA 1.3, which is OSI-
   approved but not in the CASTLE harness's accepted SPDX list. This
   excludes NPB entirely. If the list is expanded to include NOSA, BT
   and LU (Fortran multi-component) should be revisited.

5. **AoSoA candidates (21-23):** The AoSoA layout requires restructuring
   a struct-of-arrays or array-of-structs at the global level. CASTLE's
   GenP primitive is the key enabler; builders should verify that the
   CASTLE code-emission path for GenP is fully implemented before
   committing builder effort.

6. **Block-cyclic candidates (24-25):** These require OpenMP thread-level
   layout awareness. In single-threaded mode the block-cyclic layout may
   show no improvement. Builders should run both single-threaded and 4-
   thread variants to capture the effect.
