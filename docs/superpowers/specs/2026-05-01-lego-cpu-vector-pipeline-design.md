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
- A peer pipeline (`--lego-to-arm-neon`) for ARM NEON, mirroring the x86 pipeline's structure. Demonstrates that the lowering is target-portable. (SVE is captured as a small follow-up — its scalable-vector machinery warrants its own mini-design.)
- A new Python frontend `@cpu_jit` that JIT-compiles user kernels through the Lego dialect into the new pipeline. Reuses the existing shared `DSLAdapter` / `lego.rewriter` infrastructure; introduces no new AST-parsing machinery.
- A new MLIR pass `lego-vectorize` that derives vector length from a symbolic stride analysis on lowered arith expressions, with a speculative-unroll fallback for piecewise patterns. **The pass is layout-agnostic** — it does not pattern-match on layout op types and does not consult a layout interface.
- **Cross-block shuffle synthesis** (the BrickLib pattern). When an access spans a layout block boundary (e.g., `A[i,j,k+1]` reading lane 7 from the next brick), the analysis detects the piecewise structure and emits two adjacent block reads + `vector.shuffle`. This is the mechanism that flips the brick-stencil LOSSes (cands 11/12/13/14/29) to WIN.
- **Strided and gathered vector access.** When symbolic stride is non-unit but constant, emit `vector.transfer_read` with a permutation map (strided load). When access is non-affine and not piecewise (Morton-style bit ops), emit `vector.gather` with a runtime-computed index vector. Gather is roughly 2× faster than scalar even on slow targets, so emitting it is a net win whenever the alternative is a scalar fallback.
- Mixed-vector-width support within a single loop body (e.g., f32 input lanes + f64 accumulator lanes co-existing).
- The new pipeline produces correct output for all six layout cases enumerated in §6 (row, column, Morton, brick within-brick, brick cross-block stencil, anti-diagonal wavefront) on both x86 (AVX-512/AVX2) and ARM NEON.

## 3. Non-goals

- Modifying `cxx_gen` or the source-emission path. It remains exactly as it is.
- Behavioral changes to `cutile_jit`. The GPU DSL keeps producing the same output for the same inputs. (cuTile is *lightly refactored* to extract decorator-chain helpers into `_adapter.py` so the new CPU adapter can reuse them — see §5.1. This refactor is byte-for-byte behavior-preserving.)
- ARM SVE. The scalable-vector path requires a different analysis discipline (vector length is a runtime quantity) and is captured as **R15 (follow-up)**. NEON is in scope.
- GPU-side equivalents. The same architectural pattern would extend to GPU lane folds; out of scope here, captured as future work.
- SMT-based dependence analysis on the hot path. The default dep analyzer uses memref base distinctness (sufficient for Jacobi-style and embarrassingly parallel kernels). SMT can be added later as an opt-in for self-update kernels (**R14**).
- AOT compilation of evaluation candidates through this path. The proof point uses JIT for in-process measurement; AOT plumbing is captured separately as **R13 (future work)**.

## 4. Architecture overview

```
                 cxx_gen path (UNCHANGED)
   user Python ─┬─► AST rewrite ─► C++ source ─► GCC ─► binary    [source-emission, frozen]
                │
                │   cuTile path (behavior unchanged; helpers extracted into _adapter.py)
                ├─► CutileAdapter ─► Lego dialect ─► LegoToNVVMPipeline ─► PTX  [GPU DSL]
                │
                │   NEW: CPU DSL path
                └─► CPUJITAdapter ─► Lego dialect ─► lego-vectorize ─┬─► LegoX86VectorPipeline ─► LLVM IR ─► JIT
                                                                     │
                                                                     └─► LegoArmNeonPipeline   ─► LLVM IR ─► JIT/AOT

  All three paths share: lego.rewriter (Python AST → LEGO source rewrite),
                         _adapter.py (DSLAdapter base + decorator-chain helpers),
                         the Lego MLIR dialect.

  The two CPU pipelines share lego-vectorize (target-aware via pipeline option);
  only the bottom phase differs: x86 uses LLVM with -target-cpu=skx/znver3,
  ARM uses LLVM with -target=aarch64 -mattr=+neon.
```

The new pipeline phase reuses the entire shared front-end (`buildLegoLowerPipeline`) and the entire shared LLVM tail (`convert-vector-to-llvm`, `convert-arith-to-llvm`, `convert-memref-to-llvm`, etc.). Only one new pass and one new pipeline file are introduced.

## 5. Detailed design

### 5.1 Frontend: `cpu_jit` (Python decorator) + shared-helper extraction

The CPU JIT entry point reuses the existing shared infrastructure in `python/lego/frontends/`:

- `DSLAdapter` (abstract base) and `write_and_exec_temp_file` already live in `_adapter.py`.
- `lego.rewriter.rewrite()` is the shared Python-AST rewrite engine that drives any `DSLAdapter`.

The GPU DSL (`cutile_jit.py`) sits on top of that infrastructure. The new CPU JIT does the same — there is no need to duplicate Python AST parsing or rewriter machinery.

**Step A — Extract decorator-chain helpers into `_adapter.py`.**

`CutileAdapter.unwrap` currently inlines four decorator-chain strategies. Strategies 2–4 (`.fn` chain, `py_func` attribute, `__wrapped__` attribute) are generic Python-decorator handling, not cuTile-specific. They get hoisted as helper functions in `_adapter.py`:

```python
def try_fn_chain_unwrap(fn) -> tuple[callable, list]: ...
def try_py_func_unwrap(fn)  -> tuple[callable, list]: ...
def try_wrapped_unwrap(fn)  -> tuple[callable, list]: ...
def walk_to_source_fn(fn)   -> callable: ...
```

`CutileAdapter.unwrap` is refactored to call these helpers. Strategy 1 (`_pyfunc`, the `cuda.tile.kernel` attribute) stays cuTile-specific. **Behavior is byte-for-byte preserved**; FileCheck/Python tests that exercise cuTile keep passing.

**Step B — New file `python/lego/frontends/cpu_jit.py`.**

Defines `CPUJITAdapter(DSLAdapter)`:

- `unwrap`: uses only the common helpers (no `_pyfunc` strategy — the CPU path doesn't have a cuTile-style outer decorator to peel).
- `find_runtime_vars`: detects tensor-typed parameters (CPU convention TBD in implementation; default: any parameter not flagged as a constant integer is treated as runtime).
- `get_code_printer`: returns the standard `LEGOPythonCodePrinter`. No CPU-specific syntax sugar in v1.
- `compile_and_wrap`: takes the rewritten source, builds the Lego-dialect MLIR module via the existing `LayoutCompiler` machinery, runs the new pipeline `--lego-to-x86-vector`, and JITs via `mlir::ExecutionEngine`. Returns a Python-callable that invokes the JIT-compiled function with the user's arguments.

The `cpu_jit` decorator is a one-liner mirroring `cutile_jit`:

```python
def cpu_jit(fn=None, **kwargs):
    def decorator(fn):
        return rewrite(fn, CPUJITAdapter(), **kwargs)
    return decorator(fn) if fn is not None else decorator
```

**Step C — Wire `LayoutCompiler` to accept a pipeline name.**

The current `LayoutCompiler.compile()` (`python/lego/backend/compiler.py:339`) hardcodes `lego-to-llvm` in its pass-manager parse. Add an optional `pipeline_name=` parameter (default `"lego-to-llvm"` to preserve existing behavior). `CPUJITAdapter.compile_and_wrap` passes `pipeline_name="lego-to-x86-vector"`. Existing callers of `compile()` are unchanged.

**No `vector_inner` annotation.** Vectorization is fully driven by the pipeline's analysis pass; the user expresses the kernel and the layout, nothing else.

**User-facing primitives are deliberately minimal in v1.** The CPU DSL exposes the existing Lego layout primitives (`Row`, `TileBy`, `OrderBy`, `GroupBy`, etc.) and the standard Python control flow that the rewriter already handles. CPU-specific extensions (e.g., `prefetch_distance` hints) are deferred. The CPU DSL gains nothing from cuTile's GPU primitives (`shared_memory`, warp shuffle, `mma_sync`) and does not expose them.

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
1. Per-access stride analysis (two-tier: solve, then speculative-unroll fallback).

   For each memref.load / memref.store inside the loop body:
     Let addr = the address arith DAG.

     Tier A — Symbolic solve.
       Compute S(k) = simplify( clone(addr)[iv := iv+k] - addr )
         via IRMapping + applyPatternsAndFoldGreedily with arith
         canonicalization patterns and integer range analysis.
       If S(k) reduces to (k * elem_size):       classify as unit.
       Elif S(k) reduces to (k * c), c constant: classify as strided(c).
       Elif S(k) reduces to 0:                   classify as broadcast.
       Else:                                     proceed to Tier B.

     Tier B — Speculative unroll.
       For trial L ∈ {R_T, R_T/2, R_T/4, ...} down to 2:
         Compute concrete addresses addr(iv+0), addr(iv+1), ..., addr(iv+L-1)
           by IRMapping clone + canonicalization for each concrete k.
         Inspect the L computed addresses:
           - If all differ by elem_size from the previous: classify as unit
             (rare — the symbolic solver usually catches this).
           - If they partition into exactly two contiguous runs of length p and L-p
             with a single jump between them: classify as cross_block(boundary=p, L).
           - If they form a constant-stride pattern: classify as strided(c).
           - Otherwise: classify as non_affine (gather candidate).
         Stop at the first L that yields a non-non_affine classification.

2. Per-access maximum vector length.
   For each access op:
     R_T = target_register_width / sizeof(element_type_of_access)
       (R_T_f64 = 8 on AVX-512, 4 on AVX2, 2 on NEON; R_T_f32 = 16 / 8 / 4)
     Ld  = loop_carried_dep_distance (∞ if base distinctness proves no dep)
     T   = trip count (or unbounded sentinel if dynamic)

     If unit:                        Ln_access = min(R_T, T, Ld)
     Elif cross_block(boundary, Lmax): Ln_access = min(R_T, T, Ld, Lmax)
     Elif strided(c):                Ln_access = min(R_T, T, Ld) [via gather]
     Elif broadcast:                 Ln_access = max(others)
     Elif non_affine:                Ln_access = min(R_T, T, Ld) [via gather]

   Loops with no vectorizable access (every access is non_affine + Ld < 2): skip.

3. Strip-mine factor.
   L_strip = lcm(Ln_access for all accesses in body)

4. Loop selection.
   Score each candidate loop by (L_strip × number_of_arith_ops_in_body) ÷
                                  (cost_factor for non-unit accesses).
   cost_factor = 1.0 for unit / cross_block, ~5 for strided, ~10 for non_affine
   (gather is ≈10× a unit load on x86 AVX-512). The penalty discourages
   vectorizing loops where every access is a gather and the win is marginal.
   Select the loop with the maximum score (ties broken by innermost-first).

5. Vectorization rewrite.
   Strip-mine the selected loop:
     scf.for %ti = lb to ub step L_strip { vector body }
     scf.for %ti = ub_aligned to ub step 1 { scalar tail }   // residual

   Rewrite the vector body — emission strategy per classification:

     unit              → vector.transfer_read / vector.transfer_write
                         (single contiguous load/store).

     cross_block(p)    → two vector.transfer_reads at adjacent block bases,
                         followed by vector.shuffle [p, p+1, ..., p+L-1] to
                         construct the lane group spanning the boundary.

     strided(c)        → vector.transfer_read with permutation_map
                         (d0) -> (d0 * c / elem_size); LLVM lowers to a
                         strided gather (vgatherdpd on AVX-512, scalarized
                         load chain on NEON).

     broadcast         → single vector.broadcast outside the strip-mined loop.

     non_affine        → build a runtime index vector via vector.from_elements
                         applied to the concrete addr(iv+0..L-1) expressions,
                         then vector.gather using that index vector.

   For each scalar arith op:
     Determine result width from operand widths.
     Vectorize at that width, emitting (L_strip / op_width) ops.
     Insert vector.shape_cast / vector.extract_strided_slice /
       vector.insert_strided_slice / vector.broadcast at width transitions.
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

`test/Lego/lego_to_x86vector.mlir` (new) — covers each of the six layout cases enumerated in §6, asserting the expected vector ops and L value via FileCheck patterns.

`test/Lego/lego_to_arm_neon.mlir` (new) — same six cases, asserting NEON-appropriate vector widths (R_f64 = 2, R_f32 = 4) and ARM target features.

`test/Lego/lego_vectorize.mlir` (new) — unit tests for the `lego-vectorize` pass alone, using `--lego-vectorize{target=avx512|avx2|neon}` directly on hand-written `arith + memref + scf` IR. Test cases:
- Unit-stride row access → vector.transfer_read at register width.
- Strided access (col-major) → vector.transfer_read with permutation_map (strided load).
- Non-affine address (Morton-style bit ops) → vector.gather with computed index.
- Cross-block stencil (A[i,j,k+1] crossing brick boundary) → two transfer_reads + vector.shuffle.
- Mixed f32/f64 access → mixed-width vectorization with shape casts.
- Loop-carried dep (same memref base, write-then-read) → unchanged (scalar).
- Disjoint memref bases → vectorized.
- Target-cpu switch → asserts L=8 for avx512, L=4 for avx2, L=2 for neon (all f64).

#### 5.4.2 Python integration tests

`python/tests/test_cpu_jit.py` (new):
- SAXPY via `@cpu_jit`, compared against numpy reference.
- Row-major GEMM, brick within-brick local kernel, anti-diagonal wavefront kernel.
- Each test asserts (a) numerical correctness against a scalar reference, (b) presence of `vector.transfer_read/write` ops in the lowered IR (via `lego-opt -emit-after lego-vectorize`).

#### 5.4.3 Performance proof point

**Two proof-point benchmarks.**

(a) **Brick within-cell kernel** (~150 LoC). 3D within-brick compute (no cross-brick reads). Baseline: scalar `LegoToLLVMPipeline`. Treatment: `LegoToX86VectorPipeline`. Expected: WIN ≥ 2× on AVX-512 (8-lane FMA on inner brick z axis), ≥ 1.5× on NEON (2-lane f64).

(b) **Brick 7-point stencil with cross-brick reads** (~200 LoC) — re-creates the structure of evaluation candidate 11. Baseline: scalar `LegoToLLVMPipeline`. Treatment: `LegoToX86VectorPipeline` with `lego-vectorize` emitting cross_block shuffles for the ±X/Y/Z neighbors that span brick boundaries. Expected: WIN ≥ 2× on AVX-512, replicating BrickLib's published gain. **This is the headline result that flips eval candidate 11/12/13/14/29 from LOSS to WIN under the new path.**

Measurement protocol (both benchmarks): median of 30 runs, warmup 5, `OMP_NUM_THREADS=1`, fixed CPU governor.

The benchmarks live at `evaluation/cpu_vector_proof/{brick_within_cell,brick_stencil_cross}/` to keep the eval directory format consistent. They are **not** evaluation candidates in the formal CASTLE-paper sense (no upstream baseline) — they exist to validate the new pipeline.

### 5.5 File summary

```
NEW:
  lib/Lego/LegoX86VectorPipeline.cpp                       — x86 pipeline file
  lib/Lego/LegoArmNeonPipeline.cpp                         — ARM NEON pipeline file
  lib/Lego/Conversion/LegoVectorize.cpp                    — the analysis pass (target-aware)
  python/lego/frontends/cpu_jit.py                         — CPU JIT decorator + adapter
  test/Lego/lego_to_x86vector.mlir                         — x86 pipeline FileCheck test
  test/Lego/lego_to_arm_neon.mlir                          — ARM NEON pipeline FileCheck test
  test/Lego/lego_vectorize.mlir                            — pass-only FileCheck test
  python/tests/test_cpu_jit.py                             — Python integration tests
  evaluation/cpu_vector_proof/brick_within_cell/           — within-brick proof point
  evaluation/cpu_vector_proof/brick_stencil_cross/         — cross-brick stencil proof point (flips cand 11)

MODIFIED:
  include/Lego/Passes.h                                    — add LegoTo{X86Vector,ArmNeon}PipelineOptions + decls
  lib/Lego/Passes.cpp                                      — register --lego-to-x86-vector, --lego-to-arm-neon, and --lego-vectorize
  lib/Lego/CMakeLists.txt                                  — add new source files + link components
  python/lego/frontends/_adapter.py                        — add try_fn_chain_unwrap / try_py_func_unwrap / try_wrapped_unwrap / walk_to_source_fn helpers
  python/lego/frontends/cutile_jit.py                      — refactor CutileAdapter.unwrap to use new helpers (zero behavioral change; existing tests must pass unmodified)
  python/lego/backend/compiler.py                          — add optional pipeline_name= parameter to LayoutCompiler.compile() (default preserves existing behavior)
  evaluation/roadmap.md                                    — replace R1, add R12/R13/R14 future work

UNCHANGED:
  python/lego/frontends/cxx_gen.py                         — source-emission path frozen
  lib/Lego/LegoToArith.cpp                                 — input to lego-vectorize, unchanged
  lib/Lego/LegoNVVMPipeline.cpp et al.                     — GPU pipelines unchanged
  lib/Lego/CAPI/Dialects.cpp                               — already calls registerLegoPipelines(), picks up new pipeline automatically
  tools/lego-opt/lego-opt.cpp                              — driver auto-discovers new pipeline through registerLegoPipelines()
```

## 6. Layout-case validation

The design handles six layout cases, on both x86 (AVX-512/AVX2) and ARM NEON. The classification is target-independent; only L_target differs.

| Case | Address shape | Tier-A class | Tier-B (if used) | Emission | L on AVX-512/NEON |
|---|---|---|---|---|---|
| Row-major | `(i*N + j)*8` | unit | — | transfer_read | 8 / 2 |
| Col-major (inner=i) | `(j*M + i)*8` | unit | — | transfer_read | 8 / 2 |
| Col-major (inner=j) | `(j*M + i)*8` | strided(M) | — | transfer_read+permutation_map (gather) | 8 / 2 (slower than unit) |
| Morton 2D (GenP, inner-j) | `interleave(i,j)` | non_affine | non_affine | vector.gather | 8 / 2 (slowest) |
| Brick 8×8×8 within-brick | `... + ((Ix%8)*64 + (Iy%8)*8 + (Iz%8))*8` | unit | — | transfer_read | 8 / 2 |
| **Brick 7pt cross-brick** | `... + ((kk+1)/8)*4096 + ((kk+1)%8)*8` | non_affine | cross_block(p=7,L=8) | 2× transfer_read + shuffle | 8 / 2 |
| Anti-diagonal wavefront | `diag_start[i+j] + j` | unit | — | transfer_read | min(R, T(d), Ld) |

The **cross-brick row** is what was previously deferred. With the speculative-unroll fallback, the analyzer recognizes the piecewise structure (lanes 0..6 in brick n, lane 7 in brick n+1) and emits two adjacent block reads + `vector.shuffle`. **This is the mechanism that flips eval candidates 11/12/13/14/29 from LOSS to WIN.**

Morton on inner-j is now classified `non_affine` and emits a `vector.gather` rather than falling back to scalar. The gather is ~10× a unit load on x86, but ~2× faster than the scalar alternative — net win whenever the kernel has any non-trivial compute.

## 7. Risks and open questions

### 7.1 Mixed-precision shape transitions

The `vector.extract_strided_slice` / `vector.insert_strided_slice` infrastructure is upstream-stable but rarely exercised in LEGO. The shape-cast patterns may need debugging when first integrated. **Mitigation:** the FileCheck test for the mixed f32/f64 case is part of v1 acceptance.

### 7.2 Trip count when not statically known

For dynamically-sized loops, `T` is unknown at compile time. The analysis falls back to `Ln = min(R_T, Ld)` and emits a residual scalar tail. LLVM optimizes the tail well; this is not a correctness risk but may be a performance risk if the residual fraction is non-trivial relative to the main loop. **Mitigation:** the proof-point benchmark uses static sizes; future autotune work (separate roadmap item) will explore residual minimization.

**Note on canonicalization quality.** Not a risk: by the time `lego-vectorize` runs, `buildLegoLowerPipeline` has already executed two rounds of `LegoToArith` plus fixed-point arith canonicalization, integer-range analysis, CSE, and strength reduction. The synthesized `addr(iv+k) - addr(iv)` expression operates on already-canonical arith, so simplification quality is a property of the upstream pipeline (well-tested) rather than something this spec needs to guarantee.

## 8. Acceptance criteria

The design is delivered when:

1. `lego-vectorize` pass exists, target-aware (`avx512`/`avx2`/`neon` option), with FileCheck coverage of all six §6 cases including cross-brick shuffle and gather emission, plus mixed-precision and dep-distance cases.
2. `LegoX86VectorPipeline` exists, registered as `--lego-to-x86-vector`, and produces LLVM IR with AVX-512 (and AVX2 with feature flag) vector intrinsics for all six §6 cases.
3. `LegoArmNeonPipeline` exists, registered as `--lego-to-arm-neon`, and produces LLVM IR for the same six cases targeting `aarch64 -mattr=+neon`. Validation can run via cross-compilation + qemu-aarch64 if no native ARM hardware is available.
4. `cpu_jit` decorator exists, sits on the shared `DSLAdapter` base, compiles a SAXPY kernel and produces numerically-correct output that matches a scalar reference. cuTile tests pass unmodified after the helper extraction.
5. **Proof point (a) — within-brick:** ≥ 2× over scalar baseline on AVX-512, ≥ 1.5× on NEON.
6. **Proof point (b) — cross-brick stencil:** ≥ 2× over scalar baseline on AVX-512. This is the headline result that demonstrates the brick-class LOSSes are addressable in the new path.
7. `check-lego-all` passes (existing tests unbroken; new tests added).
8. `evaluation/roadmap.md` updated: R1 replaced with this design's framing, R13/R14/R15 retained as remaining future work.

## 9. Future work captured

- **R13** — AOT object-file path for evaluation candidate migration. Plumb `mlir-translate` + `llc` to produce relocatable objects; migrate eval candidates from `cxx_gen` to `cpu_jit` for performance candidates.
- **R14** — SMT-driven dep analysis as opt-in for self-update kernels (Gauss-Seidel, in-place reductions). Reuse the existing `LegoExternalSMTVerifier` infrastructure, gated behind a pipeline option.
- **R15** — ARM SVE pipeline. Scalable-vector machinery (vector length is a runtime quantity) requires its own analysis discipline — different from the fixed-width NEON path delivered here. Same `Lego*Pipeline` skeleton, different target dialect (`arm_sve`) and a scalable-aware `lego-vectorize` mode.
- **R17** — GPU lane-fold equivalent. Apply the same architectural pattern (CPU-vectorize-style speculative unroll → `vector.shuffle` over adjacent thread-block reads) to GPU pipelines for warp-level intrinsics.

## 10. Out of scope (explicit)

- Source-emission (`cxx_gen`) modifications. Frozen.
- Behavioral changes to the GPU DSL or GPU pipelines. (`cutile_jit` is lightly refactored to extract decorator-chain helpers; behavior is byte-for-byte preserved.)
- LayoutVectorizationInterface or any layout-op interface for vectorization. The design explicitly rejects this in favor of layout-agnostic arith analysis.
- `vector_inner` user annotation on `TileByOp`. The design explicitly drops this in favor of analysis-derived vector length.
- Single-L-per-loop simplification. Replaced by per-access L with `lcm` strip-mining.
- ARM SVE — the scalable-vector path. Captured as R15.
- AOT object-file emission. Captured as R13.
- SMT-based dep analysis on the hot path. Captured as R14.
