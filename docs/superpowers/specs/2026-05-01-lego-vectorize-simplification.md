# LEGO Vectorize Simplification Proposal

**Date:** 2026-05-01
**Author:** Claude (architecture review)
**Status:** Proposal — pending user review
**Context:** Post-v1 review. Branch `feat/cpu-vector-pipeline` at commit d250120. Test
status: 79/79 lit + 14/14 cpu_dsl pass. Performance status: 16 WIN + 12 PARITY out of
42 candidates vs gcc -O3.

---

## Goal

Simplify the LEGO CPU vectorization codebase to improve maintainability, generalizability,
and onboarding without losing performance or breaking tests. Each finding is an incremental
refactoring that preserves the test pass-rate and benchmark numbers.

---

## Scope

**In scope:**
- `lib/Lego/LegoVectorize.cpp` (2306 lines)
- `lib/Lego/LegoVectorizeUtils.h` (78 lines)
- `lib/Lego/LegoX86VectorPipeline.cpp` (78 lines)
- `lib/Lego/LegoArmNeonPipeline.cpp` (67 lines)
- `python/lego/backend/cpu_dsl.py` (774 lines)
- `python/lego/backend/cpu_builder.py` (707 lines)

**Out of scope:** changing the algorithm (Tier-A/Tier-B), changing the public API
(`@cpu_kernel` decorator surface), changing the test files.

---

## Findings

### Finding 1: Split `LegoVectorize.cpp` into Analysis + Rewrite TUs

**Current state:** `lib/Lego/LegoVectorize.cpp` is a single 2306-line translation unit
containing three conceptually distinct concerns:

1. **Analysis** (lines 34–578): `AffineVal` struct, `evalAffine`, `isLoopInvariant`,
   `solveAccessTierA`, `concreteEvaluate`, `solveAccessTierB`. These are pure IR analysis
   functions that do not mutate the IR.
2. **Decision + scheduling** (lines 580–1018): `getRegisterLanesForType`, `lcm_i64`,
   `LoopAnalysis`, `collectCandidateLoops`, `memrefBasesDisjoint`, `computeMinDepDistance`,
   `computeStripMineFactor`. Consumes classification results; decides whether to vectorize
   and with what factor.
3. **Rewrite** (lines 1020–2307): `StripMineResult`, `stripMineForOp`, `cloneAddrDAG`,
   `emitVectorBody` (~1000 lines), `emitTailBody`, `LegoVectorizePass::runOnOperation`.
   Mutates the IR.

The mix means that a contributor modifying the stride solver reads past all 1000 lines of
emission logic before reaching the relevant code, and vice versa. A bug in the SAXPY
emission path forces reading past the Tier-B solver.

The split proposed in the spec design doc (A.split) is the right decomposition:

- **`LegoVectorizeAnalysis.cpp`**: Everything from the current file in the
  `mlir::lego` namespace plus the anonymous-namespace `concreteEvaluate` helper.
  Exports: `solveAccessTierA`, `solveAccessTierB` (already in `LegoVectorizeUtils.h`);
  add `getRegisterLanesForType` and `computeStripMineFactor`'s helper predicates
  (the `allUnit`, `hasAnyUnit`, reduction-guard logic) to `LegoVectorizeUtils.h`.

- **`LegoVectorizeRewrite.cpp`**: `stripMineForOp`, `cloneAddrDAG`, `emitVectorBody`,
  `emitTailBody`, `LegoVectorizePass`, `collectCandidateLoops`, `computeMinDepDistance`,
  `computeStripMineFactor`. The pass owns the rewrite; it calls the analysis API.

The `CMakeLists.txt` for `lib/Lego` gains one source file. The tablegen-generated pass
base class (`GEN_PASS_DEF_LEGOVECTORIZEPASS`) stays with the rewrite TU.

**Why this preserves performance:** No behavior change. The split is a file-boundary
reorganization only; all function bodies are identical. The compiler will produce the same
object code.

**Effort:** M (3–5 days — mechanical split + CMake update + verify all-tests pass)

**Dependencies:** None. Land first; all other C++ findings depend on a clean TU structure.

---

### Finding 2: Extract `_BaseCompiler` shared between `cpu_dsl.py` and `gpu_dsl.py`

**Current state:** `cpu_dsl.py` and `gpu_dsl.py` each define a class `_Compiler` (~290
lines in cpu_dsl, ~330 lines in gpu_dsl) with identical or near-identical implementations
of:

- `_stmt`, `_assign`, `_aug_assign` — 100% identical
- `_while` — 100% identical
- `_compare` — 100% identical
- `_binop_ct`, `_binop_rt` — 100% identical (gpu_dsl has deferred `from lego.mlir.ir import
  IntegerType` inside the method bodies rather than at module level; cpu_dsl imports at the
  top)
- `_unary` — 100% identical
- `_idx`, `_modified_names`, `_collect_assigns`, `_eval_ct` — 100% identical
- `_indices` — 100% identical

The diverging parts are:
- `_for`: gpu_dsl only supports `range()`; cpu_dsl additionally supports the
  `tile_range` sentinel rewrite.
- `_attribute`: gpu_dsl dispatches `block_id`, `thread_id`, `block_dim`; cpu_dsl raises
  `RuntimeError` for those names.
- `_call`: gpu_dsl dispatches ~25 GPU-specific functions (barrier, lane_id, shuffle_*,
  mma_*); cpu_dsl dispatches ~5 shared functions (exp, sqrt, rsqrt, apply, apply_inverse,
  set_layout) and raises `RuntimeError` for the GPU-only set.
- `_method_call`: gpu_dsl implements TensorCore method dispatch; cpu_dsl raises
  `NotImplementedError`.
- `__init__`: gpu_dsl does not handle `scalar_params`; cpu_dsl does.

This duplication means any bug fix (e.g., the R16 bitwise-op i32 fix, the `_collect_assigns`
fix for nested loops) must be applied twice. Both files were evolved in parallel during
development; they are already slightly out of sync in their use of deferred vs. top-level
`from lego.mlir.ir import IntegerType`.

**Proposed simplification:** Create `python/lego/backend/_dsl_base.py` with:

```python
class _BaseCompiler:
    """Shared AST walker for both @cpu_kernel and @gpu_kernel DSLs."""
    def __init__(self, ctx, func_def, buf_params, outer):
        self.ctx = ctx
        self.func_def = func_def
        self.outer = outer
        self.env = {}
        self.buf_map = {name: i for i, (name, _) in enumerate(buf_params)}

    def run(self):
        for stmt in self.func_def.body:
            self._stmt(stmt)

    # --- All identical methods: _stmt, _assign, _aug_assign, _for (range-only),
    #     _while, _expr, _name, _binop, _binop_ct, _binop_rt, _unary, _compare,
    #     _idx, _indices, _modified_names, _collect_assigns, _eval_ct ---

    # Subclass hooks — raise NotImplementedError to force overrides:
    def _attribute(self, node): raise NotImplementedError
    def _call(self, node): raise NotImplementedError
    def _method_call(self, node): raise NotImplementedError
    def _load(self, node): raise NotImplementedError
    def _store(self, node, val, tag): raise NotImplementedError
```

Then:
- `_CPUCompiler(_BaseCompiler)` overrides `_for` (adds `tile_range` rewrite),
  `_attribute` (raises GPU error), `_call` (shared funcs + GPU guard), `_method_call`
  (raises), `_load`/`_store` (with name-check error messages), `__init__` (adds
  `scalar_params` population).
- `_GPUCompiler(_BaseCompiler)` overrides `_attribute` (dispatches GPU dims), `_call`
  (full GPU dispatch), `_method_call` (TensorCore), `_load`/`_store` (no name-check),
  `__init__` (existing body, no scalar_params).

**Why this preserves performance:** No MLIR emission changes. Both `_CPUCompiler` and
`_GPUCompiler` produce identical IR to the current `_Compiler` classes. The refactor is
purely Python class hierarchy — zero effect on the compiled MLIR.

**Effort:** M (3–5 days — extract base, update both DSLs, run all Python tests)

**Dependencies:** None. Can land in parallel with Finding 1.

---

### Finding 3: Replace `emitVectorBody`'s if-else chain with per-kind emit helpers

**Current state:** `emitVectorBody` (lines 1098–2097, ~1000 lines) is a single function
with a sequential scan over `origLoop.getBody()` operations. For `memref.load` it
dispatches over five `AccessKind` variants via an if-else chain:

```
Unit (lines 1250–1270) — ~20 lines
Broadcast (lines 1271–1288) — ~18 lines
CrossBlock (lines 1289–1355, if Ln == L_strip) — ~67 lines
Strided:
  deinterleave path (lines 1397–1576) — ~180 lines per stride case
  gather fallback (lines 1578–1648) — ~70 lines
NonAffine gather (lines 1649–1699) — ~50 lines
scalar fallback (lines 1700–1704)
```

The shared state (`subVectorMap`, `mapping`, `makeOffset`, `getLnForAccess`, `builder`,
`loc`) must be captured as closed-over state by all paths. Each path also independently
constructs the mask + passThru constants for gather ops — the construction is copy-pasted
between the Strided-gather fallback (lines 1632–1643) and NonAffine-gather (lines 1684–1696)
with zero differences.

**Proposed simplification:** Extract per-kind load helpers as private static functions
with a shared `EmitContext` parameter bag:

```cpp
struct EmitContext {
  OpBuilder &builder;
  Location loc;
  IRMapping &mapping;
  DenseMap<Value, SmallVector<Value>> &subVectorMap;
  int64_t L_strip;
  llvm::StringRef target;
  scf::ForOp origLoop;
  scf::ForOp vecLoop;
};

static void emitUnitLoad(memref::LoadOp load, const AccessClassification &cls,
                         EmitContext &ctx);
static void emitBroadcastLoad(memref::LoadOp load, const AccessClassification &cls,
                              EmitContext &ctx);
static void emitCrossBlockLoad(memref::LoadOp load, const AccessClassification &cls,
                               EmitContext &ctx);
static void emitStridedLoad(memref::LoadOp load, const AccessClassification &cls,
                            EmitContext &ctx);
static void emitGatherLoad(memref::LoadOp load, const AccessClassification &cls,
                           EmitContext &ctx);
```

The dispatch in `emitVectorBody` becomes:

```cpp
switch (cls.kind) {
  case lego::AccessKind::Unit:       emitUnitLoad(load, cls, ectx); break;
  case lego::AccessKind::Broadcast:  emitBroadcastLoad(load, cls, ectx); break;
  case lego::AccessKind::CrossBlock: emitCrossBlockLoad(load, cls, ectx); break;
  case lego::AccessKind::Strided:    emitStridedLoad(load, cls, ectx); break;
  case lego::AccessKind::NonAffine:  emitGatherLoad(load, cls, ectx); break;
}
```

The two copy-pasted mask+passThru constructions collapse into one `buildGatherMaskAndPassThru`
helper called from both `emitStridedLoad` (gather fallback) and `emitGatherLoad`.

Similarly the pre-pass scalar broadcast loop (lines 1182–1234) and the catch-all arith
vectorization (lines 1794–2096) can each become a static helper.

**Why this preserves performance:** The change is purely structural — same emission logic,
same MLIR ops produced, same SSA graph. LLVM sees identical IR.

**Effort:** M (3–5 days — mechanical extraction + all FileCheck tests must pass
unchanged)

**Dependencies:** Finding 1 (clean TU split makes the extraction easier to review).

---

### Finding 4: Hoist cost model magic numbers into a `CostModel` struct

**Current state:** The two hardware-calibrated penalty constants (`5.0` for strided gather,
`10.0` for non-affine gather) appear inline in `computeStripMineFactor` (lines 927–929)
with a detailed comment block (lines 900–935). The ILP unroll factor `kILPFactor = 4`
appears at line 985 with a separate comment block. The register-lane formula
(`getRegisterLanesForType`, lines 592–599) is a free function. Together these form the
implicit cost model, but there is no single place to look up all tuning parameters.

**Proposed simplification:** Extract to a `CostModel` struct in `LegoVectorizeUtils.h`:

```cpp
struct CostModel {
  // Penalty applied to pure gather-only loops.
  // SOURCE: Intel Optimization Reference Manual §2.5.5, Table 2-9.
  static constexpr double kStridedGatherPenalty  = 5.0;
  // SOURCE: Polychroniou et al., SIGMOD'15; Pandey et al., SC'19.
  static constexpr double kNonAffineGatherPenalty = 10.0;
  // ILP unroll multiplier for pure unit-stride loops.
  // SOURCE: Agner Fog §12.7; matches Clang's UnrollFactor for AVX-512.
  static constexpr int64_t kILPFactor = 4;
  // Max register lanes per element on each target.
  static int64_t registerLanes(llvm::StringRef target, int64_t elementBytes);
};
```

`computeStripMineFactor` replaces `5.0` with `CostModel::kStridedGatherPenalty`,
`10.0` with `CostModel::kNonAffineGatherPenalty`, `4` with `CostModel::kILPFactor`.
`getRegisterLanesForType` becomes `CostModel::registerLanes`.

This makes the cost model immediately visible to anyone reading the header, centralizes
all tuning parameters, and makes it straightforward to add a future pass option
(`--lego-vectorize-cost-model=conservative`) that adjusts the constants without grep.

**Why this preserves performance:** All constant values are identical. The struct is a
pure organizational change.

**Effort:** S (1–2 days)

**Dependencies:** Finding 1 (declare the struct in `LegoVectorizeUtils.h` after the TU
split).

---

### Finding 5: Add skip-reason diagnostics (LLVM_DEBUG remarks)

**Current state:** When the vectorizer skips a loop — due to `bodyOK = false`, `L_strip
<= 1`, or the reduction guard — the user gets no feedback. From the call site (e.g., a
Python `@cpu_kernel` invocation), the only observable effect is that the kernel runs
scalar. There is no way to distinguish "the vectorizer processed this loop and chose
L_strip=1 because the cost model rejected it" from "the vectorizer rejected the loop body
because it contains an unsupported op". Debugging the `lego_vectorize_*.mlir` tests
requires adding `LLVM_DEBUG(llvm::dbgs() << ...)` manually and recompiling.

There are 14 distinct skip points in `runOnOperation` + `computeStripMineFactor`:
- `computeStripMineFactor` returning 1 (6 paths: reduction guard, Ln<=1,
  cost-model rejection, no constraining accesses, unknown cls.kind, static trip unknown
  for ILP)
- `bodyOK = false` (4 paths: G2 index_cast, G3 unknown dialect, G4 scf.if with else,
  G4 non-store then-body)
- `L_strip <= 1` after `computeStripMineFactor`

**Proposed simplification:** Add `LLVM_DEBUG` (not a hard warning) at each skip point:

```cpp
// In runOnOperation, after L_strip <= 1 guard:
LLVM_DEBUG(llvm::dbgs() << "[lego-vectorize] skip " << func.getName()
           << " loop at " << a.forOp.getLoc()
           << " — L_strip=" << a.L_strip
           << " (reason: " << skipReason << ")\n");
```

Where `skipReason` is a `llvm::StringRef` set at each early-return site. The strings are
compile-time constants (`"reduction-loop"`, `"cost-model"`, `"body-unsupported-op"`,
`"body-index-cast"`, `"body-scf-if-else"`, etc.).

These become visible with `--mlir-print-ir-before-all` or by running
`lego-opt --debug-only=lego-vectorize`. The strings are stable and can be grepped in
test logs.

This finding also proposes adding one `mlir::emitRemark` (not LLVM_DEBUG) at the
`bodyOK = false` path for the `G2 index_cast` guard, since this is the most
counterintuitive rejection (the user wrote legal-looking code; the op is in the arith
dialect; it fails only for IV-dependent index_cast). A remark surfaces this in the
MLIR diagnostic stream without requiring `--debug` compilation.

**Why this preserves performance:** Diagnostic emission is gated behind `LLVM_DEBUG`
which is compiled out in release builds. The one `emitRemark` for G2 emits a diagnostic
only when `bodyOK` becomes false, i.e., the vectorizer was going to do nothing anyway.

**Effort:** S (1–2 days)

**Dependencies:** None. Can land standalone.

---

### Finding 6: Remove the duplicated fast-math comment block

**Current state:** Lines 1997–2017 and lines 2021–2046 of `LegoVectorize.cpp` contain
two nearly-identical comment blocks describing the `fastmath<contract>` injection
rationale. The first block (lines 1997–2017) ends with `// (c) no fastmath attr is
already present on the original op.` The second block (lines 2021–2046) is the final,
more accurate version ending with `// check whether the existing flags already include
the contract bit.` The first block is a verbatim copy of an earlier iteration that was
superseded when the implementation switched from `!op.hasAttr("fastmath")` to checking
the existing flags.

**Proposed simplification:** Delete lines 1997–2020 (the outdated first block). The
surviving second block (lines 2021–2046) is the accurate description of the
implementation.

**Why this preserves performance:** Comment-only change.

**Effort:** S (<1 day — single edit)

**Dependencies:** None. Land immediately; it is a mechanical one-line diff review.

---

### Finding 7: Package `subVectorMap` + `mapping` as a `VectorFrame` helper struct

**Current state:** `emitVectorBody` (after Finding 3's extraction, now the dispatch
function) passes both `IRMapping &mapping` and
`DenseMap<Value, SmallVector<Value>> &subVectorMap` through every per-kind emit helper.
These two data structures represent a single conceptual abstraction: "for an original
scalar Value, what vector Values cover its L_strip-wide vectorized form?" They have an
invariant: `mapping[v]` is always `subVectorMap[v][0]` (the first sub-vector), enforced
at every emission site by the pattern `mapping.map(v, subs[0]); subVectorMap[v] = subs;`.
This invariant is currently maintained by convention across 17 subVectorMap update sites
and 33 IRMapping sites — any future emit helper that forgets one of the two updates will
produce IR that silently misses a mapping.

**Proposed simplification:** Introduce a lightweight `VectorFrame` struct:

```cpp
struct VectorFrame {
  IRMapping mapping;
  DenseMap<Value, SmallVector<Value>> subVecs;

  /// Register a vectorized result.
  /// Always maps mapping[orig] = subs[0] and subVecs[orig] = subs.
  void map(Value orig, SmallVector<Value> subs) {
    assert(!subs.empty());
    mapping.map(orig, subs[0]);
    subVecs[orig] = std::move(subs);
  }

  /// Look up sub-vectors for orig; falls back to {mapping.lookupOrDefault(orig)}.
  SmallVector<Value> getSubsFor(Value orig) const {
    auto it = subVecs.find(orig);
    if (it != subVecs.end()) return it->second;
    return {mapping.lookupOrDefault(orig)};
  }
};
```

The `EmitContext` proposed in Finding 3 carries a `VectorFrame &frame` instead of
separate `mapping` and `subVectorMap` references. All 17 `subVectorMap[v] = ...` +
`mapping.map(v, ...)` pairs collapse to `frame.map(v, ...)`.

This enforces the invariant structurally rather than by convention, and halves the
surface area for future emit helpers.

**Why this preserves performance:** No behavioral change. The frame struct is a
zero-overhead wrapper at compile time (its `map` method inlines to the same two
assignments). LLVM sees identical IR.

**Effort:** S (1–2 days after Findings 3 and 1 land)

**Dependencies:** Finding 3 (needs `EmitContext`); Finding 1 (for a clean rewrite TU).

---

### Finding 8: Consolidate overlapping FileCheck tests

**Current state:** The 11 `lego_vectorize_*.mlir` test files (1551 lines total) have
some cross-file overlap:

- `lego_vectorize.mlir:@cross_brick_stencil` (lines 126–145) and
  `lego_vectorize_cross_block.mlir:@cross_block_boundary7` (lines 13–42) both test
  a boundary-7 CrossBlock pattern on `memref<?xf64>`. They use different address
  expression formulations but test the same classification + emission path. The
  `lego_vectorize.mlir` version also appears in the "general" test file without a
  clear rationale for its location there rather than in the dedicated cross_block file.

- `lego_vectorize.mlir:@col_major_strided` (lines 87–103) and
  `lego_vectorize_strided.mlir` both cover constant-stride gather. The main file's
  version is the "smoke test" — it suffices to keep one in each file only if the
  main-file version is retained as a quick sanity check.

- `lego_vectorize.mlir:@morton_gather` (lines 149–167) and
  `lego_vectorize_gather.mlir:@morton_style` (lines 14–38) both test Morton-style
  non-affine gather on an 8-element loop with `andi` + `ori` + `shli`. Different
  loop sizes and constants, but the same classification path.

**Proposed simplification:** Apply the following moves:

1. Move `@cross_brick_stencil` from `lego_vectorize.mlir` into
   `lego_vectorize_cross_block.mlir` (deduplicate; the dedicated file is the right home).
2. Keep `@col_major_strided` in `lego_vectorize.mlir` as a smoke test but add a comment
   `// Full strided coverage in lego_vectorize_strided.mlir.`
3. Move `@morton_gather` from `lego_vectorize.mlir` into `lego_vectorize_gather.mlir`,
   keeping only a minimal gather smoke test in `lego_vectorize.mlir`.

The reorganization makes `lego_vectorize.mlir` the "one of each" smoke-test file (Unit,
Broadcast, Strided-smoke, CrossBlock-smoke, NonAffine-smoke, self-update-skip) and the
`lego_vectorize_<kind>.mlir` files the exhaustive per-kind suites.

**Why this preserves performance:** No code changes; test-file reorganization only.

**Effort:** S (1 day — move tests, verify FileCheck still passes)

**Dependencies:** None. Can land standalone, but naturally follows any code refactor that
changes the test count (to avoid a confusing "79 tests → 79 tests, different layout").

---

### Finding 9: Normalize `gpu_dsl.py`'s deferred `IntegerType` imports to top-level

**Current state:** `gpu_dsl.py` has 9 occurrences of `from lego.mlir.ir import
IntegerType` or `from lego.mlir.ir import IndexType` scattered inside method bodies
(e.g., lines 108, 425, 461, 467, 479, 504, 508, 585, 779). These were added piecemeal
during development to avoid a circular-import issue that no longer exists (the MLIR
Python bindings are imported at module load in `cpu_dsl.py` without issue). Each deferred
import adds ~1 µs of overhead per first call to the importing method (Python `import`
machinery re-checks `sys.modules` and the `importlib` lock). In the DSL hot path this is
negligible, but it makes the import surface of the file invisible without reading every
method body.

**Proposed simplification:** When `_BaseCompiler` is extracted (Finding 2), the base
class module `_dsl_base.py` imports both `IntegerType` and `IndexType` at module level
(as `cpu_dsl.py` already does). The deferred imports in `gpu_dsl.py` disappear when
`_GPUCompiler` inherits from `_BaseCompiler` and the shared methods move to the base.

If Finding 2 is deferred: apply the normalization directly to `gpu_dsl.py` — add
`IntegerType, IndexType` to the top-level `from lego.mlir.ir import ...` line and delete
the 9 deferred imports.

**Why this preserves performance:** Minor improvement. The deferred imports are
`sys.modules`-cached on first call, so subsequent calls have no overhead. The change
improves readability and correctness-by-inspection of the import surface.

**Effort:** S (<1 day — trivially mechanical)

**Dependencies:** Finding 2 (if done as part of the base class extraction); otherwise
standalone.

---

### Finding 10: Rename `evalAffine`/`concreteEvaluate` to self-documenting names

**Current state:** The three analysis functions in `LegoVectorize.cpp` have names that
describe *what they do* mechanically but not *why* or *at what level*:

- `evalAffine` — evaluates whether a Value is affine in `iv`, returns `AffineVal`.
  More precisely it is a "symbolic stride solver" returning a linear model
  `coeff * iv + constant + Σ invariant_i`.
- `concreteEvaluate` — evaluates a Value at a concrete `iv=k`. This is the "speculative
  probe" function used only by Tier-B.
- `cloneAddrDAG` — clones the def-use DAG of an address expression with lane-specific
  IV substitution. The name is reasonable but "DAG" is an unusual term in this MLIR
  context (the standard term is "def-use chain" or "address computation").

In `LegoVectorizeUtils.h` the exported solvers are `solveAccessTierA` and
`solveAccessTierB` — self-documenting. The internal helpers should match that register.

**Proposed renames:**

| Old name | New name | Rationale |
|---|---|---|
| `evalAffine` | `evalLinearInIV` | Precise: computes the linear coefficient of `iv`; "affine" is overloaded in MLIR (affine dialect ≠ this) |
| `concreteEvaluate` | `evalConcreteIV` | Symmetric with `evalLinearInIV`; "concrete" = substituting a fixed iv value |
| `cloneAddrDAG` | `cloneAddrChain` | MLIR convention uses "chain" for def-use sequences; avoids graph-theory jargon |

`AffineVal` → `LinearIVExpr` if we are renaming consistently; this is optional and higher
effort (it touches every usage). The struct rename is **optional**; the function renames
are the high-value part.

**Why this preserves performance:** Rename only; identical object code.

**Effort:** S (1 day — mechanical rename + grep for any external callers in test code)

**Dependencies:** Finding 1 (cleaner to do the rename in the analysis TU after the split).

---

## Sequencing

The recommended landing order minimizes merge conflicts and provides early validation:

```
1. Finding 6  — delete duplicated fast-math comment (trivial, lands in minutes)
2. Finding 9  — normalize gpu_dsl.py imports (trivial, lands standalone)
3. Finding 5  — add LLVM_DEBUG skip-reason diagnostics (standalone, enables easier
               debugging during subsequent refactors)
4. Finding 4  — CostModel struct in LegoVectorizeUtils.h (small, standalone)
5. Finding 1  — TU split (M effort; all subsequent C++ findings depend on this)
6. Finding 10 — rename evalAffine / concreteEvaluate / cloneAddrDAG (after TU split)
7. Finding 3  — extract per-kind emit helpers + EmitContext (after TU split)
8. Finding 7  — VectorFrame struct (after Finding 3 lands EmitContext)
9. Finding 2  — _BaseCompiler Python extraction (can overlap with 5–8; Python side only)
10. Finding 8 — consolidate FileCheck tests (land last, after all code is stable)
```

Findings 2 and 5 can proceed in parallel with findings 1 and 4 since they touch disjoint
files (Python vs C++).

---

## Out of scope (noted explicitly)

- Do not change the algorithmic approach (Tier-A/B is correct; see spec §5.2).
- Do not add new features (the `@cpu_kernel` surface is final for v1; R12, R13, R14,
  R15 are tracked separately in `evaluation/roadmap.md`).
- Do not add new SIMD shuffle patterns or cost-model tuning — that is a performance
  task, not a simplification task.
- Do not propose changes to `LegoX86VectorPipeline.cpp` or `LegoArmNeonPipeline.cpp`
  (they are already minimal, ~70 lines each, and require no simplification).

---

## Estimate

| # | Finding | Effort |
|---|---------|--------|
| 1 | Split `LegoVectorize.cpp` into Analysis + Rewrite TUs | M (3–5 days) |
| 2 | Extract `_BaseCompiler` for `cpu_dsl.py` / `gpu_dsl.py` | M (3–5 days) |
| 3 | Per-kind emit helpers + `EmitContext` | M (3–5 days) |
| 4 | `CostModel` struct with named constants | S (1–2 days) |
| 5 | Skip-reason LLVM_DEBUG diagnostics | S (1–2 days) |
| 6 | Remove duplicated fast-math comment | S (<1 day) |
| 7 | `VectorFrame` struct (subVectorMap + IRMapping) | S (1–2 days) |
| 8 | Consolidate overlapping FileCheck tests | S (1 day) |
| 9 | Normalize `gpu_dsl.py` deferred imports | S (<1 day) |
| 10 | Rename `evalAffine` / `concreteEvaluate` / `cloneAddrDAG` | S (1 day) |

Total: approximately 3–4 weeks to land all 10 findings sequentially. Findings 6, 9, 5
can be done in parallel with any other work; together they take ≤ 2 days.

---

## Risk

| # | What could break | Mitigation |
|---|-----------------|-----------|
| 1 | CMake build errors from the new TU; ODR violations if `concreteEvaluate`'s anonymous namespace is not correctly scoped | Run `check-lego-all` after each file move; keep anonymous namespace in the Analysis TU |
| 2 | Behavioral divergence between CPU and GPU DSL if a shared method is subtly wrong in the base class | Run all 14 `test_cpu_dsl.py` + GPU DSL tests before merge |
| 3 | A per-kind helper misses a `subVectorMap` or `mapping` update | All 79 lit FileCheck tests exercise at least one emission path each; the per-kind tests (`lego_vectorize_broadcast.mlir` etc.) cover each helper in isolation |
| 4 | No risk — constant values identical | Run `check-lego-all` as sanity check |
| 5 | A `LLVM_DEBUG` statement at a hot-path site (not the skip site) could appear in the opt=0 build on CI | Gate all diagnostics behind `LLVM_DEBUG` (not `llvm::errs()`); the one `emitRemark` is at a bodyOK=false path where vectorization is already skipped |
| 6 | None | Trivial comment deletion |
| 7 | A future `VectorFrame::map` call with an empty `subs` list hits the assert and crashes in debug builds | The assert is a valid invariant check; any crash reveals a real bug |
| 8 | A test is accidentally deleted rather than moved | `git diff --stat` review before merge; FileCheck count must remain 79 |
| 9 | A circular import is re-introduced if `_BaseCompiler` module imports something that imports `cpu_dsl` | Run `python -c "import lego.backend.cpu_dsl; import lego.backend.gpu_dsl"` as a smoke test |
| 10 | An external caller (eval script, documentation) uses the old function name | `grep -r evalAffine concreteEvaluate cloneAddrDAG` in the repo; only internal callers exist |

---

## Validation plan

After all findings land, the full validation gate:

1. `cmake --build build -j16 --target check-lego-all` — all 79 lit FileCheck tests pass.
2. `python -m pytest python/tests/test_cpu_dsl.py -v` — all 14 cpu_dsl Python tests pass.
3. `python evaluation/cpu_dsl_comparison/run_all.py` — same 16 WIN + 12 PARITY result
   (dashboard unchanged vs pre-refactor baseline at commit d250120).
4. `check-lego-all` green (includes all GPU DSL tests — Finding 2 must not regress the
   GPU path).

The benchmark validation (step 3) is the critical gate: if any finding introduces a
behavior change in the emission pipeline, it will manifest as a performance regression
on one or more of the 16 WIN candidates. The correct outcome is identical speedup
numbers within measurement noise (±2%).

---

## Conclusion

The highest-leverage finding is **Finding 1** (TU split), because it unblocks Findings
3, 7, and 10 and makes the largest immediate improvement to onboardability. A new
contributor working on the stride solver today has to read ~1700 lines of emission code
to get to the analysis. After the split, `LegoVectorizeAnalysis.cpp` is a ~600-line
self-contained module focused entirely on stride analysis.

The second-highest-leverage finding is **Finding 2** (`_BaseCompiler`), because it
eliminates the only persistent duplication that requires applying bug fixes twice.
R16 (index-typed compare), R19 (strided-gather scalar-index), and the bitwise-op i32
fix were each applied to both `_Compiler` classes. The next bug fix should not have to
be.

Both Finding 1 and Finding 2 are M-effort (3–5 days each) but carry low risk: neither
changes emission logic or the public API, and the test suite fully covers both.
