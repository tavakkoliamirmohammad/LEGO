# LEGO CPU Vector Pipeline & Sibling CPU DSL — Design

**Date:** 2026-05-01
**Author:** Amir Mohammad Tavakkoli
**Status:** Draft (under review)
**Related roadmap entry:** R1 (re-framed)

## 1. Problem statement

The CASTLE CPU evaluation surfaced a coherent class of LOSSes — every brick-layout candidate (11, 12, 13, 14, 29) lost on AMD AVX2 and reproduced on Intel AVX-512. Root cause analysis traced this to LEGO's source-emission CPU path (`python/lego/frontends/cxx_gen` → C++ → GCC), in which vectorization is delegated to the host C compiler. GCC handles row-major auto-vectorization well but cannot pattern-match register-level layout folds (the `brick()` macro's lane fold, BrickLib's published mechanism for 1.9×–4.9× wins).

The roadmap's original R1 framed this as "add SIMD intrinsic emission to source generation." After review, this is the wrong layer: source emission is fundamentally a Python-AST → C++ rewriter that cannot reason about layout-aware vectorization without becoming a compiler in its own right.

This spec defines an alternate path: a peer of the existing GPU DSLs that performs end-to-end MLIR codegen on CPU, with vectorization done as an MLIR analysis pass. The source-emission path remains unchanged for users who want readable C++ output.

## 2. Goals

- A new MLIR target pipeline (`--lego-to-x86-vector`) that lowers the Lego dialect through MLIR's vector dialect to LLVM IR, producing AVX-512 (and AVX2) intrinsics directly.
- A new Python frontend `@cpu_tile_jit` that mirrors the structure and code-organization of `cutile_jit` but routes lowered IR through the new pipeline.
- A new MLIR pass `lego-vectorize` that derives vector length from a symbolic stride analysis on lowered arith expressions. **The pass is layout-agnostic** — it does not pattern-match on layout op types and does not consult a layout interface.
- Mixed-vector-width support within a single loop body (e.g., f32 input lanes + f64 accumulator lanes co-existing).
- The new pipeline produces correct output for all five layout cases enumerated in §6 (row, column, Morton, brick within-brick, anti-diagonal wavefront), with vectorization applied wherever the analysis can prove unit-stride contiguity.

## 3. Non-goals

- Modifying `cxx_gen` or the source-emission path. It remains exactly as it is.
- Cross-brick shuffle synthesis (the BrickLib `vec_kp = shuffle(brick_n, brick_n+1)` pattern). This is required to flip stencil candidates 11/12/13/14/29 from LOSS to WIN and is captured as roadmap entry **R12 (future work)**.
- Strided / gathered vector loads. When an access's symbolic stride is non-unit-and-non-constant, the pass falls back to scalar.
- SMT-based dependence analysis on the hot path. The default dep analyzer uses memref base distinctness (sufficient for Jacobi-style and embarrassingly parallel kernels). SMT can be added later as an opt-in for self-update kernels.
- ARM Neon / SVE / GPU-side equivalents. They follow the same shape but are out of scope for this spec.
- AOT compilation of evaluation candidates through this path. The proof point uses JIT for in-process measurement; AOT plumbing is captured separately as **R13 (future work)**.

## 4. Architecture overview

```
                 cxx_gen path (UNCHANGED)
   user Python ─┬─► AST rewrite ─► C++ source ─► GCC ─► binary    [source-emission, frozen]
                │
                │   cuTile path (UNCHANGED)
                ├─► CutileAdapter ─► Lego dialect ─► LegoToNVVMPipeline ─► PTX  [GPU DSL]
                │
                │   NEW: CPU DSL path
                └─► CPUTileAdapter ─► Lego dialect ─► LegoX86VectorPipeline ─► LLVM IR ─► JIT
                                                                  │
                                                                  └─ uses lego-vectorize (NEW)
```

The new pipeline phase reuses the entire shared front-end (`buildLegoLowerPipeline`) and the entire shared LLVM tail (`convert-vector-to-llvm`, `convert-arith-to-llvm`, `convert-memref-to-llvm`, etc.). Only one new pass and one new pipeline file are introduced.

## 5. Detailed design

### 5.1 Frontend: `cpu_tile_jit`

**File:** `python/lego/frontends/cpu_tile_jit.py` (new).

Mirrors `python/lego/frontends/cutile_jit.py` structurally:
- `CPUTileAdapter(DSLAdapter)` — overrides `unwrap()` and `find_runtime_vars()` for CPU primitives.
- `cpu_tile_jit` decorator — Python-AST rewrite (via `lego.rewriter`) producing Lego dialect MLIR.
- Compiles by invoking `LegoX86VectorPipeline` via the parsed pass-pipeline string `"builtin.module(lego-to-x86-vector{cpu=zen3|skx|...})"`, then materializes via `mlir::ExecutionEngine`.

**User-facing primitives.** Same tile/range/iter primitives as cuTile where they make sense on CPU. CPU-only additions (`prefetch_distance`) deferred. GPU-only primitives (`shared_memory`, warp shuffle, `mma_sync`) are not exposed.

**No `vector_inner` annotation.** Vectorization is fully driven by the pipeline's analysis pass; the user expresses the kernel and the layout, nothing else.

### 5.2 Pipeline: `LegoX86VectorPipeline`

**File:** `lib/Lego/LegoX86VectorPipeline.cpp` (new). Header declarations in `include/Lego/Passes.h`.

Mirrors `LegoNVVMPipeline.cpp`'s three-phase template:

1. **Front-end (shared).** `buildLegoLowerPipeline(pm)` — same as the existing CPU JIT. Produces normalized Lego dialect, then lowers to `arith + memref + scf + func` via `--lego-to-arith`.

2. **CPU vector phase (new).**
   - `arith` canonicalization + `IntegerRangeAnalysis` to clean up address expressions. Optionally include layout-aware patterns (Row, TileBy) as canonicalization helpers. These are *optimizations*, not required for correctness.
   - `lego-vectorize` (new pass — see §5.3). Output: `vector + arith + memref + scf` (no Lego dialect ops remain).
   - `convert-vector-to-llvm` lowers vector ops to LLVM dialect IR (`<8 x double>` types, standard LLVM intrinsics). The target-specific lane width (512 bits for AVX-512, 256 for AVX2) is realized later by LLVM's backend (`llc`) via the LLVM target triple and target features (`-target-cpu=skx`, `-target-cpu=znver3`, etc.) attached to the module. This pipeline does **not** route through the X86Vector dialect for v1 — the `<N x double>` LLVM types plus correct target features are sufficient for AVX-512 codegen of FMA/load/store patterns. X86Vector is only required for x86-specific intrinsics (e.g., AVX2 horizontal ops) that don't have a clean MLIR vector representation; out of scope for v1.

3. **LLVM tail (shared).** `buildLegoToLLVMPipeline` — `SCFToControlFlow`, `ArithToLLVM`, `MemRefToLLVM`, `FuncToLLVM`, `ReconcileUnrealizedCasts`. Then `mlir-translate -mlir-to-llvmir` and `llc` for AOT (deferred), or `mlir::ExecutionEngine` for JIT.

**Pipeline options:** `LegoToX86VectorPipelineOptions` carries `cpu={zen3|skx|skl|...}` to select the LLVM target triple and feature flags. Mirrors the format-selection options on `LegoToNVVMPipelineOptions`.

**Registration.** Added to `Passes.cpp:registerLegoPipelines()` via `PassPipelineRegistration<LegoToX86VectorPipelineOptions>("lego-to-x86-vector", ..., buildLegoToX86VectorPipeline)`. No changes to `tools/lego-opt/lego-opt.cpp` (driver discovers pipelines through `legoRegisterPasses()`).

### 5.3 Pass: `lego-vectorize`

**File:** `lib/Lego/Conversion/LegoVectorize.cpp` (new). Pass declaration in `include/Lego/Passes.h`.

Operates on `func.func` containing `scf + arith + memref` (post `lego-to-arith`). Layout types are gone — all that remains is concrete integer arithmetic over loop induction variables. The pass is **layout-agnostic**: it never inspects which layout op produced an address.

#### 5.3.1 Algorithm

For each `scf.for` loop in the function:

```
1. Per-access stride analysis.
   For each memref.load / memref.store inside the loop body:
     Let addr = the address arith DAG.
     Compute S(k) = simplify( clone(addr)[iv := iv+k] - addr )
       via IRMapping + applyPatternsAndFoldGreedily with arith
       canonicalization patterns and integer range analysis.

     Classify:
       unit         : S(k) reduces to (k * elem_size)
       strided c    : S(k) reduces to (k * c) where c != elem_size
       broadcast    : S(k) reduces to 0
       non_affine   : simplified S(k) still references iv

2. Per-access maximum vector length.
   For each access op:
     R_T = target_register_width / sizeof(element_type_of_access)
     Ld  = loop_carried_dep_distance (∞ if base distinctness proves no dep)
     T   = trip count (or unbounded sentinel if dynamic)

     If access is unit:        Ln_access = min(R_T, T, Ld)
     Elif access is broadcast: Ln_access = max(others) [does not constrain]
     Else:                     Ln_access = 1 [this loop is not vectorizable]

   If any access has Ln_access == 1: skip this loop.

3. Strip-mine factor.
   L_strip = lcm(Ln_access for all accesses in body)

4. Loop selection.
   Score each candidate loop by (L_strip × number_of_arith_ops_in_body).
   This is a deliberately simple v1 heuristic; the intent is "vectorize the
   loop where the most work happens at the largest width." Future versions
   may incorporate cache-residency or cost models.
   Select the loop with the maximum score (ties broken by innermost-first).

5. Vectorization rewrite.
   Strip-mine the selected loop:
     scf.for %ti = lb to ub step L_strip { vector body }
     scf.for %ti = ub_aligned to ub step 1 { scalar tail }   // residual

   Rewrite the vector body:
     For each memref op at width Ln_access:
       Emit (L_strip / Ln_access) vector ops, each at width Ln_access.
       For loads:  vector.transfer_read at sequential offsets
       For stores: vector.transfer_write at sequential offsets

     For each scalar arith op:
       Determine result width from operand widths.
       Vectorize at that width, emitting (L_strip / op_width) ops.
       Insert vector.shape_cast / vector.extract_strided_slice /
         vector.insert_strided_slice / vector.broadcast at width transitions.

   For broadcast accesses: emit a single vector.broadcast outside the strip-mined loop.
```

#### 5.3.2 Mixed vector widths

`L_strip` is the synchronization point between accesses of different natural widths (e.g., `f32` loads at L=16 and `f64` loads at L=8 on AVX-512). Width transitions inside the body — type conversions, reductions, broadcasts — are realized via:

- `vector.shape_cast` — reshape between vector types of the same total bitwidth.
- `vector.extract_strided_slice` / `vector.insert_strided_slice` — extract/insert sub-vectors when widths divide evenly.
- `vector.broadcast` — scalar to vector or narrow vector to wide vector.
- `vector.reduction` — vector to scalar.

All are upstream MLIR ops and lower through `convert-vector-to-llvm` to LLVM IR `extractelement` / `insertelement` / `shufflevector`, which LLVM's backend folds into target-specific moves.

#### 5.3.3 Dependence analysis (default, no SMT)

For two accesses A and B in the same loop:
- If they reference different memref SSA values (different `memref.alloc`/`memref.subview` results, different function arguments) and there is no aliasing through `memref.cast` between them: no dependence. **This is the common case in LEGO** — separate `!lego.view`s lower to separate memref values.
- If they reference the same memref SSA value (or one is a `memref.subview` / `memref.cast` of the other) and one is a write: a potential loop-carried dep exists. v1 conservatively sets `Ld = 1` (refuse to vectorize this loop) unless the address expressions are *provably* disjoint via `IntegerRangeAnalysis` (e.g., disjoint constant offset ranges).

This handles Jacobi stencils (read A, write B), GEMM (read A, B; write C), and SAXPY/elementwise kernels — i.e., the bulk of the eval suite. Self-update kernels (Gauss-Seidel, in-place reductions) fall back to scalar in v1. SMT-driven dep analysis using `LegoExternalSMTVerifier` is captured as **R14 (future work)**.

#### 5.3.4 Layout information as opportunistic hints

Per the design directive, the vectorizer never *requires* layout info. But layout info can speed up the analysis and improve simplification quality:

- **Loop-search priors.** A `lego.tile_by` op carries the inner-tile size as an attribute. The search can try the loop matching that tile dim first (likely winner). This is search-pruning, not correctness.
- **Layout-aware canonicalization patterns.** Lifted into the standard `arith` canonicalization phase, contributed via a new patterns file. They simplify Row, TileBy, OrderBy address expressions. The vectorizer doesn't depend on them being present.
- **Aliasing metadata.** When `lego-to-arith` knows two `!lego.view`s reference disjoint memrefs, that information can be threaded as a `disjoint` attribute on the resulting memref ops, accelerating the dep check. v1 derives this from view distinctness; future versions can be smarter.

### 5.4 Test structure

#### 5.4.1 MLIR FileCheck tests

`test/Lego/lego_to_x86vector.mlir` (new) — covers each of the five layout cases enumerated in §6, asserting the expected vector ops and L value via FileCheck patterns.

`test/Lego/lego_vectorize.mlir` (new) — unit tests for the `lego-vectorize` pass alone, using `--lego-vectorize` directly on hand-written `arith + memref + scf` IR. Test cases:
- Unit-stride row access → vector at register width.
- Strided access → unchanged (scalar).
- Non-affine address (Morton-style bit ops) → unchanged.
- Mixed f32/f64 access → mixed-width vectorization with shape casts.
- Loop-carried dep (same memref base, write-then-read) → unchanged (scalar).
- Disjoint memref bases → vectorized.

#### 5.4.2 Python integration tests

`python/tests/test_cpu_tile_jit.py` (new):
- SAXPY via `@cpu_tile_jit`, compared against numpy reference.
- Row-major GEMM, brick within-brick local kernel, anti-diagonal wavefront kernel.
- Each test asserts (a) numerical correctness against a scalar reference, (b) presence of `vector.transfer_read/write` ops in the lowered IR (via `lego-opt -emit-after lego-vectorize`).

#### 5.4.3 Performance proof point

A small standalone benchmark (~150 LoC) reproducing the structure of evaluation candidate 11 (3D 7-point stencil) but **within-brick only** (no halo, no cross-brick reads), to avoid the R12 future-work boundary.

- Baseline: scalar version through the existing `LegoToLLVMPipeline` (no `lego-vectorize`).
- Treatment: same kernel through `LegoToX86VectorPipeline` (with `lego-vectorize`).
- Measurement: median of 30 runs, warmup 5, `OMP_NUM_THREADS=1`, fixed CPU governor.
- Expected outcome: WIN ≥ 2× on AVX-512 (8-lane FMA on the inner brick z axis).

The benchmark lives at `evaluation/cpu_vector_proof/brick_within_cell/` to keep the eval directory format consistent. It is **not** an evaluation candidate in the formal CASTLE-paper sense (no upstream baseline) — it exists to validate the new pipeline.

### 5.5 File summary

```
NEW:
  lib/Lego/LegoX86VectorPipeline.cpp                       — pipeline file
  lib/Lego/Conversion/LegoVectorize.cpp                    — the analysis pass
  python/lego/frontends/cpu_tile_jit.py                    — frontend decorator
  test/Lego/lego_to_x86vector.mlir                         — pipeline FileCheck test
  test/Lego/lego_vectorize.mlir                            — pass-only FileCheck test
  python/tests/test_cpu_tile_jit.py                        — Python integration tests
  evaluation/cpu_vector_proof/brick_within_cell/       — proof-point benchmark

MODIFIED:
  include/Lego/Passes.h                                    — add LegoToX86VectorPipelineOptions + decl
  lib/Lego/Passes.cpp                                      — register --lego-to-x86-vector and --lego-vectorize
  lib/Lego/CMakeLists.txt                                  — add new source files + link components
  evaluation/roadmap.md                                    — replace R1, add R12/R13/R14 future work

UNCHANGED:
  python/lego/frontends/cxx_gen.py                         — source-emission path frozen
  python/lego/frontends/cutile_jit.py                      — GPU DSL unchanged
  lib/Lego/LegoToArith.cpp                                 — input to lego-vectorize, unchanged
  lib/Lego/LegoNVVMPipeline.cpp et al.                     — GPU pipelines unchanged
  lib/Lego/CAPI/Dialects.cpp                               — already calls registerLegoPipelines(), picks up new pipeline automatically
  tools/lego-opt/lego-opt.cpp                              — driver auto-discovers new pipeline through registerLegoPipelines()
```

## 6. Layout-case validation

The design has been walked through the five layout cases that motivate the design:

| Case | Address shape | S(k) class | L derived | Result |
|---|---|---|---|---|
| Row-major | `(i*N + j)*8` | unit (along inner j) | min(R, N) | full vec on j |
| Col-major | `(j*M + i)*8` | unit (along inner i) | min(R, M) | algorithm same as row, picks i |
| Morton (GenP) | `interleave(i, j)` (bit ops) | non_affine | 1 | scalar; correct, naive Morton inner not contig |
| Brick 8×8×8 within-brick | `... + ((Ix%8)*64 + (Iy%8)*8 + (Iz%8))*8` | unit (along inner Iz) | 8 | full vec at L=8, perfect AVX-512 fit |
| Anti-diagonal wavefront | `diag_start[i+j] + j` | unit (along within-diag c) | min(R, T(d), Ld) | full vec on long diags; partial on short / lookback-bounded |

Cross-brick stencil reads (the `A[i, j, k+1]` pattern crossing brick boundaries at lane 7) produce a piecewise S(k) that the canonicalizer cannot reduce to closed form — classified non_affine, falls back to scalar. This is correct for v1 and captured as R12 future work.

## 7. Risks and open questions

### 7.1 Canonicalization quality

The accuracy of the analysis hinges on how well `arith` canonicalization simplifies the substituted-and-subtracted expression `addr(iv+k) - addr(iv)`. For Row/TileBy/Brick this is straightforward. For unusual user-defined `gen_p` ops with non-trivial apply-region bodies, simplification may stall and the access classifies as `non_affine` even when human inspection would say it's vectorizable. **Mitigation:** add layout-aware canonicalization patterns over time; document which `gen_p` shapes the analysis recognizes.

### 7.2 Mixed-precision shape transitions

The `vector.extract_strided_slice` / `vector.insert_strided_slice` infrastructure is upstream-stable but rarely exercised in LEGO. The shape-cast patterns may need debugging when first integrated. **Mitigation:** the FileCheck test for the mixed f32/f64 case is part of v1 acceptance.

### 7.3 Trip count when not statically known

For dynamically-sized loops, `T` is unknown at compile time. The analysis falls back to `Ln = min(R_T, Ld)` and emits a residual scalar tail. LLVM optimizes the tail well; this is not a correctness risk but may be a performance risk if the residual fraction is non-trivial relative to the main loop. **Mitigation:** the proof-point benchmark uses static sizes; future autotune work (separate roadmap item) will explore residual minimization.

### 7.4 Coexistence with existing JIT path

The existing `--lego-to-llvm` pipeline produces scalar code and is the default for `LayoutCompiler.compile()`. Switching the default to `--lego-to-x86-vector` is **out of scope for this spec** — `cpu_tile_jit` is opt-in. After validation, a follow-up may switch the JIT default to the vectorizing pipeline; that is a separate change.

## 8. Acceptance criteria

The design is delivered when:

1. `lego-vectorize` pass exists, has FileCheck coverage of all five §6 cases plus mixed-precision and dep-distance cases.
2. `LegoX86VectorPipeline` exists, registered as `--lego-to-x86-vector`, and produces LLVM IR with AVX-512 vector intrinsics for the Row, brick within-brick, and anti-diagonal wavefront test cases.
3. `cpu_tile_jit` decorator exists, mirrors `cutile_jit` organization, compiles a SAXPY kernel and produces numerically-correct output that matches a scalar reference.
4. The proof-point benchmark (3D 7-point within-brick stencil) runs under `cpu_tile_jit` and demonstrates ≥2× speedup over the scalar `LegoToLLVMPipeline` baseline on a single AVX-512 thread.
5. `check-lego-all` passes (existing tests unbroken; new tests added).
6. `evaluation/roadmap.md` updated: R1 replaced with this design's framing, R12/R13/R14 added.

## 9. Future work captured

- **R12** — Cross-brick neighbor shuffle support. Add a pattern in `lego-vectorize` that recognizes piecewise-linear S(k) crossing a layout block boundary and synthesizes the equivalent `vector.shuffle` over two adjacent block reads. Unblocks brick stencils 11/12/13/14/29.
- **R13** — AOT object-file path for evaluation candidate migration. Plumb `mlir-translate` + `llc` to produce relocatable objects; migrate eval candidates from `cxx_gen` to `cpu_tile_jit` for performance candidates.
- **R14** — SMT-driven dep analysis as opt-in for self-update kernels. Reuse the existing `LegoExternalSMTVerifier` infrastructure, gated behind a pipeline option.
- **R15** — ARM Neon / SVE pipeline. Same `LegoX86VectorPipeline` shape, different target dialects (`arm_neon`, `arm_sve`).
- **R16** — Strided / gathered vector support. Extend `lego-vectorize` to emit `vector.gather` / `vector.scatter` for `affine_strided` and `non_affine` accesses where target latency allows.

## 10. Out of scope (explicit)

- Source-emission (`cxx_gen`) modifications. Frozen.
- GPU DSL changes. `cutile_jit` and the GPU pipelines are untouched.
- LayoutVectorizationInterface or any layout-op interface for vectorization. The design explicitly rejects this in favor of layout-agnostic arith analysis.
- `vector_inner` user annotation on `TileByOp`. The design explicitly drops this in favor of analysis-derived vector length.
- Trial-and-check L search. Replaced by symbolic stride solve.
- Single-L-per-loop simplification. Replaced by per-access L with `lcm` strip-mining.
