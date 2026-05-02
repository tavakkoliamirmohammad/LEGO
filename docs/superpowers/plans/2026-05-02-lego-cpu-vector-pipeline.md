# LEGO CPU Vector Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end MLIR vectorization path for LEGO that lowers the Lego dialect through MLIR's vector dialect to LLVM IR with target-specific intrinsics (AVX-512/AVX2/NEON), a new `@cpu_jit` Python frontend, and proof-point benchmarks demonstrating ≥2× speedup on brick stencils — flipping the eval candidates 11/12/13/14/29 from LOSS to WIN under the new path.

**Architecture:** A new `lego-vectorize` MLIR pass operates on the lowered `arith + memref + scf` IR (post `lego-to-arith`). It is layout-agnostic — it derives vector length per-access from a symbolic stride solve (Tier A) with a speculative-unroll fallback (Tier B) for piecewise-linear access patterns like cross-brick stencils. Two new pipeline files (`LegoX86VectorPipeline.cpp`, `LegoArmNeonPipeline.cpp`) compose the pass with the existing front-end and LLVM tail. A new Python decorator `@cpu_jit` reuses the `DSLAdapter` base shared with `cutile_jit`.

**Tech Stack:** MLIR (vector / arith / memref / scf / func dialects), LLVM (IR + llc backends for x86 and aarch64), Python (DSLAdapter machinery, `lego.rewriter`, `LayoutCompiler`), CMake, lit/FileCheck for MLIR tests, pytest for Python tests, qemu-aarch64 for ARM cross-compile validation.

**Spec reference:** `docs/superpowers/specs/2026-05-01-lego-cpu-vector-pipeline-design.md`

---

## File structure

```
NEW:
  lib/Lego/Conversion/LegoVectorize.cpp                 — the analysis pass implementation
  lib/Lego/Conversion/LegoVectorize.h                   — internal utility headers (StrideAnalyzer, etc.)
  lib/Lego/LegoX86VectorPipeline.cpp                    — x86 pipeline (lego-to-x86-vector)
  lib/Lego/LegoArmNeonPipeline.cpp                      — ARM NEON pipeline (lego-to-arm-neon)
  python/lego/frontends/cpu_jit.py                      — CPU JIT adapter + @cpu_jit decorator
  test/Lego/lego_vectorize.mlir                         — pass-only FileCheck
  test/Lego/lego_to_x86vector.mlir                      — x86 pipeline FileCheck
  test/Lego/lego_to_arm_neon.mlir                       — ARM NEON pipeline FileCheck
  python/tests/test_cpu_jit.py                          — Python integration tests
  python/tests/test_adapter_helpers.py                  — refactor regression tests
  evaluation/cpu_vector_proof/brick_within_cell/        — within-brick proof-point benchmark
  evaluation/cpu_vector_proof/brick_stencil_cross/      — cross-brick stencil proof-point

MODIFIED:
  include/Lego/Passes.h                                 — pipeline option structs + decls
  lib/Lego/Passes.cpp                                   — register pipelines and pass
  lib/Lego/CMakeLists.txt                               — wire new sources, link libs
  python/lego/frontends/_adapter.py                     — add decorator-chain helpers
  python/lego/frontends/cutile_jit.py                   — refactor to use helpers (zero behavior change)
  python/lego/backend/compiler.py                       — add pipeline_name= parameter
  evaluation/roadmap.md                                 — replace R1, retain R13/R14/R15/R17
```

---

## Phase A — Foundation: helper extraction, pipeline plumbing, pass scaffold

### Task 1: Extract decorator-chain helpers in `_adapter.py`

**Files:**
- Modify: `python/lego/frontends/_adapter.py`
- Test: `python/tests/test_adapter_helpers.py` (new)

- [ ] **Step 1: Write failing test for the helpers**

```python
# python/tests/test_adapter_helpers.py
"""Regression test for decorator-chain helpers extracted from CutileAdapter."""

import functools
from lego.frontends._adapter import (
    try_fn_chain_unwrap,
    try_py_func_unwrap,
    try_wrapped_unwrap,
    walk_to_source_fn,
)


def test_fn_chain_unwrap_walks_dot_fn_chain():
    def inner(): pass
    class Wrapper:
        def __init__(self, fn): self.fn = fn
    outer = Wrapper(Wrapper(inner))
    fn, wrappers = try_fn_chain_unwrap(outer)
    assert fn is inner
    assert wrappers == [outer, outer.fn]


def test_fn_chain_unwrap_returns_input_when_no_chain():
    def plain(): pass
    fn, wrappers = try_fn_chain_unwrap(plain)
    assert fn is plain
    assert wrappers == []


def test_py_func_unwrap_extracts_one_level():
    def real(): pass
    class Numba:
        def __init__(self, fn): self.py_func = fn
    n = Numba(real)
    fn, wrappers = try_py_func_unwrap(n)
    assert fn is real
    assert wrappers == [n]


def test_py_func_unwrap_returns_input_when_no_attr():
    def plain(): pass
    fn, wrappers = try_py_func_unwrap(plain)
    assert fn is plain
    assert wrappers == []


def test_wrapped_unwrap_handles_functools():
    def inner(): pass
    @functools.wraps(inner)
    def outer(): pass
    fn, wrappers = try_wrapped_unwrap(outer)
    assert fn is inner
    assert wrappers == [outer]


def test_walk_to_source_fn_follows_src_fn():
    def base(): pass
    class Layer:
        def __init__(self, fn): self.src_fn = fn
    chained = Layer(Layer(base))
    assert walk_to_source_fn(chained) is base


def test_walk_to_source_fn_no_attr():
    def plain(): pass
    assert walk_to_source_fn(plain) is plain
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /scratch/general/vast/u1419116/LEGO/.venv/bin/activate
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python pytest python/tests/test_adapter_helpers.py -v
```
Expected: FAIL with `ImportError: cannot import name 'try_fn_chain_unwrap'`.

- [ ] **Step 3: Add the helpers to `_adapter.py`**

Append to `python/lego/frontends/_adapter.py` (after the existing `write_and_exec_temp_file` function):

```python
# ---------------------------------------------------------------------------
# Decorator-chain unwrap helpers (shared by all DSL adapters)
# ---------------------------------------------------------------------------

def try_fn_chain_unwrap(fn):
    """Walk .fn chain (Triton-style). Returns (innermost, wrappers_outer_to_inner)."""
    wrappers = []
    while hasattr(fn, 'fn'):
        wrappers.append(fn)
        fn = fn.fn
    return fn, wrappers


def try_py_func_unwrap(fn):
    """Numba-style py_func attribute. Returns (innermost, [wrapper] or [])."""
    if hasattr(fn, 'py_func'):
        return fn.py_func, [fn]
    return fn, []


def try_wrapped_unwrap(fn):
    """functools-style __wrapped__ attribute. Returns (innermost, [wrapper] or [])."""
    if hasattr(fn, '__wrapped__'):
        return fn.__wrapped__, [fn]
    return fn, []


def walk_to_source_fn(fn):
    """Follow .src_fn chain to the bottom. Returns the innermost fn."""
    while hasattr(fn, 'src_fn'):
        fn = fn.src_fn
    return fn
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python pytest python/tests/test_adapter_helpers.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add python/lego/frontends/_adapter.py python/tests/test_adapter_helpers.py
git commit -m "frontends: add decorator-chain unwrap helpers to _adapter.py"
```

---

### Task 2: Refactor `CutileAdapter.unwrap` to use the helpers

**Files:**
- Modify: `python/lego/frontends/cutile_jit.py:40-69`

- [ ] **Step 1: Verify existing cuTile tests pass before refactor**

```bash
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python pytest python/tests/ -k cutile -v
```
Record the test count + pass status. This is the regression baseline.

- [ ] **Step 2: Refactor `unwrap` method**

Replace the body of `CutileAdapter.unwrap` in `python/lego/frontends/cutile_jit.py` (lines 40-69) with:

```python
    def unwrap(self, fn):
        from lego.frontends._adapter import (
            try_fn_chain_unwrap,
            try_py_func_unwrap,
            try_wrapped_unwrap,
            walk_to_source_fn,
        )

        original_fn = fn
        wrappers = []

        # Strategy 1: _pyfunc (cuda.tile.kernel-specific)
        if hasattr(original_fn, '_pyfunc'):
            wrappers.append(original_fn)
            original_fn = original_fn._pyfunc

        # Strategies 2-4: generic decorator-chain helpers
        if not wrappers:
            original_fn, wrappers = try_fn_chain_unwrap(original_fn)
        if not wrappers:
            original_fn, wrappers = try_py_func_unwrap(original_fn)
        if not wrappers:
            original_fn, wrappers = try_wrapped_unwrap(original_fn)

        return walk_to_source_fn(original_fn), original_fn, wrappers
```

- [ ] **Step 3: Run cuTile tests to verify byte-for-byte behavior preserved**

```bash
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python pytest python/tests/ -k cutile -v
```
Expected: identical pass count to Step 1.

- [ ] **Step 4: Run full check-lego-all to confirm no other regression**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target check-lego-all -j16
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/lego/frontends/cutile_jit.py
git commit -m "frontends(cutile): use shared decorator-chain helpers from _adapter.py"
```

---

### Task 3: Add `pipeline_name=` parameter to `LayoutCompiler.compile()`

**Files:**
- Modify: `python/lego/backend/compiler.py:339-371` (around `LayoutCompiler.compile`)
- Test: `python/tests/test_compiler_pipeline_param.py` (new)

- [ ] **Step 1: Read the existing compile() method**

```bash
sed -n '330,380p' /scratch/general/vast/u1419116/LEGO/python/lego/backend/compiler.py
```
Note the exact pass-pipeline string used (likely `"builtin.module(lego-to-llvm)"`).

- [ ] **Step 2: Write failing test**

Create `python/tests/test_compiler_pipeline_param.py`:

```python
"""LayoutCompiler.compile() must accept a pipeline_name= override."""

import pytest
from lego.backend.compiler import LayoutCompiler


def test_compile_accepts_pipeline_name_kwarg():
    """The compile() method must accept pipeline_name without TypeError."""
    # Build a trivial module from any minimal LEGO IR we already have.
    # If there's a test fixture, use it; otherwise this test will exercise
    # only the parameter plumbing.
    compiler = LayoutCompiler()
    # We expect no TypeError on the kwarg itself.
    sig = compiler.compile.__code__.co_varnames
    assert 'pipeline_name' in sig, \
        f"compile() must accept pipeline_name kwarg; got {sig}"


def test_compile_default_pipeline_unchanged():
    """Calling compile() without pipeline_name uses 'lego-to-llvm'."""
    import inspect
    sig = inspect.signature(LayoutCompiler.compile)
    assert sig.parameters['pipeline_name'].default == 'lego-to-llvm'
```

- [ ] **Step 3: Run test to verify failure**

```bash
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python pytest python/tests/test_compiler_pipeline_param.py -v
```
Expected: FAIL — `pipeline_name` not in compile() signature.

- [ ] **Step 4: Modify `compile()` in `python/lego/backend/compiler.py`**

Find the `compile(self, ...)` method around line 339. Add `pipeline_name='lego-to-llvm'` to the signature, and replace the hardcoded pipeline string. The exact change:

```python
# Before:
def compile(self, module, opt_level=2):
    pm = PassManager.parse("builtin.module(lego-to-llvm)")
    pm.run(module.operation)
    self._engine = ExecutionEngine(module, opt_level=opt_level)

# After:
def compile(self, module, opt_level=2, pipeline_name='lego-to-llvm'):
    pm = PassManager.parse(f"builtin.module({pipeline_name})")
    pm.run(module.operation)
    self._engine = ExecutionEngine(module, opt_level=opt_level)
```

(The actual function may have additional logic; preserve it. Only the pipeline-string source changes.)

- [ ] **Step 5: Run test to verify pass**

```bash
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python pytest python/tests/test_compiler_pipeline_param.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Run check-lego-all to confirm no regression**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target check-lego-all -j16
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add python/lego/backend/compiler.py python/tests/test_compiler_pipeline_param.py
git commit -m "backend(compiler): parameterize pipeline name in LayoutCompiler.compile()"
```

---

### Task 4: Add `lego-vectorize` pass scaffolding (no-op pass that registers cleanly)

**Files:**
- Create: `lib/Lego/Conversion/LegoVectorize.cpp`
- Modify: `include/Lego/Passes.h`
- Modify: `lib/Lego/Passes.cpp`
- Modify: `lib/Lego/CMakeLists.txt`
- Test: `test/Lego/lego_vectorize.mlir` (new)

- [ ] **Step 1: Write a failing FileCheck test for the no-op pass**

Create `test/Lego/lego_vectorize.mlir`:

```mlir
// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// CHECK-LABEL: func.func @noop_passthrough
// CHECK: arith.addi
// CHECK: return
func.func @noop_passthrough(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
```

- [ ] **Step 2: Run test to verify it fails (pass not registered)**

```bash
cd /scratch/general/vast/u1419116/LEGO/build
cmake --build . --target lego-opt -j16
./bin/lego-opt /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir --lego-vectorize
```
Expected: ERROR — unknown pass `lego-vectorize`.

- [ ] **Step 3: Add pass declaration to `include/Lego/Passes.h`**

Append (in the `mlir::lego` namespace, alongside other pass decls around line 80-150):

```cpp
namespace mlir::lego {
struct LegoVectorizePassOptions {
  std::string target = "avx512";  // "avx512" | "avx2" | "neon"
};

std::unique_ptr<Pass> createLegoVectorizePass(
    const LegoVectorizePassOptions &options = {});
}  // namespace mlir::lego
```

- [ ] **Step 4: Create `lib/Lego/Conversion/LegoVectorize.cpp` with a no-op pass implementation**

```cpp
//===- LegoVectorize.cpp - Layout-agnostic vectorization pass -------------===//
//
// Lowers loops over Lego-derived arith address expressions to MLIR vector
// dialect ops by symbolic stride analysis. Layout-agnostic: operates on
// post-LegoToArith IR (arith + memref + scf).
//
//===----------------------------------------------------------------------===//

#include "Lego/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {
class LegoVectorizePass
    : public PassWrapper<LegoVectorizePass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegoVectorizePass)

  LegoVectorizePass() = default;
  LegoVectorizePass(const LegoVectorizePass &other)
      : PassWrapper(other), target_(other.target_) {}
  explicit LegoVectorizePass(const lego::LegoVectorizePassOptions &options)
      : target_(options.target) {}

  StringRef getArgument() const final { return "lego-vectorize"; }
  StringRef getDescription() const final {
    return "Layout-agnostic vectorization for LEGO via symbolic stride solve";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() final {
    // No-op for now. Tier-A/B analysis lands in subsequent tasks.
  }

 private:
  Option<std::string> target_{*this, "target",
                              llvm::cl::desc("avx512|avx2|neon"),
                              llvm::cl::init("avx512")};
};
}  // namespace

namespace mlir::lego {
std::unique_ptr<Pass> createLegoVectorizePass(
    const LegoVectorizePassOptions &options) {
  return std::make_unique<LegoVectorizePass>(options);
}
}  // namespace mlir::lego
```

- [ ] **Step 5: Register the pass in `lib/Lego/Passes.cpp`**

Find `registerLegoPipelines()` (around line 209). Just before it (or in its sibling registration function), add a `PassRegistration`:

```cpp
// In the appropriate place — typically inside legoRegisterPasses() or its
// callee — register the pass:
PassRegistration<LegoVectorizePass>();
```

If `Passes.cpp` doesn't include `LegoVectorize.cpp`'s impl directly, ensure the `createLegoVectorizePass` declaration in `Passes.h` is reachable, and consider adding a registration in `registerLegoPipelines()`:

```cpp
mlir::lego::registerLegoVectorizePass();  // see next bullet
```

In `lib/Lego/Conversion/LegoVectorize.cpp`, append after the `namespace mlir::lego` block:

```cpp
namespace mlir::lego {
void registerLegoVectorizePass() {
  PassRegistration<::LegoVectorizePass>();
}
}
```

And declare it in `include/Lego/Passes.h`:

```cpp
namespace mlir::lego {
void registerLegoVectorizePass();
}
```

- [ ] **Step 6: Add the new source to `lib/Lego/CMakeLists.txt`**

Find the `add_mlir_dialect_library` call for the Lego library. Add `Conversion/LegoVectorize.cpp` to its sources list. Ensure the link components include `MLIRVectorDialect` and `MLIRArithDialect`.

```cmake
# Find the SOURCES list and add:
Conversion/LegoVectorize.cpp

# Ensure linked:
LINK_LIBS PUBLIC
  MLIRVectorDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
```

- [ ] **Step 7: Build and re-run the FileCheck test**

```bash
cd /scratch/general/vast/u1419116/LEGO/build
cmake --build . --target lego-opt check-lego-all -j16
./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS (no-op pass leaves IR unchanged).

- [ ] **Step 8: Commit**

```bash
git add include/Lego/Passes.h lib/Lego/Conversion/LegoVectorize.cpp lib/Lego/Passes.cpp lib/Lego/CMakeLists.txt test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: scaffold no-op pass + registration + smoke test"
```

---

## Phase B — Tier A symbolic stride solve

### Task 5: StrideAnalyzer utility — symbolic substitution `iv → iv+k`

**Files:**
- Create: `lib/Lego/Conversion/LegoVectorize.h`
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`
- Test: `test/Lego/lego_vectorize.mlir` (extend with stride-detection test)

- [ ] **Step 1: Write the FileCheck test for stride solve on a Row-major access**

Append to `test/Lego/lego_vectorize.mlir`:

```mlir
// -----

// A trivially unit-stride access: addr = base + iv*8 (f64 row-major)
// CHECK-LABEL: func.func @row_major_unit_stride
// CHECK: vector.transfer_read
// CHECK-NOT: memref.load
func.func @row_major_unit_stride(%A: memref<1024xf64>, %B: memref<1024xf64>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    %v = memref.load %A[%i] : memref<1024xf64>
    memref.store %v, %B[%i] : memref<1024xf64>
  }
  return
}
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /scratch/general/vast/u1419116/LEGO/build
./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: FAIL — pass is still no-op; IR has `memref.load`, not `vector.transfer_read`.

- [ ] **Step 3: Create internal header `LegoVectorize.h` with StrideAnalyzer**

```cpp
//===- LegoVectorize.h - Internal utilities for lego-vectorize ------------===//

#ifndef LEGO_CONVERSION_LEGOVECTORIZE_H
#define LEGO_CONVERSION_LEGOVECTORIZE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir::lego {

enum class AccessKind {
  Unit,        // S(k) = k * elem_size
  Strided,     // S(k) = k * c, c constant != elem_size
  Broadcast,   // S(k) = 0
  CrossBlock,  // piecewise unit-stride with single boundary (Tier B only)
  NonAffine,   // simplification stalls; iv survives in S(k)
};

struct AccessClassification {
  AccessKind kind;
  int64_t stride = 0;        // for Strided
  int64_t boundary = -1;     // for CrossBlock: lane index of the discontinuity
  int64_t elementBytes = 0;  // sizeof(element) in bytes
};

// Symbolic stride solver. Given a load/store op and the candidate IV,
// returns a classification of S(k) = simplify(addr(iv+k) - addr(iv)).
// Tier A only — Tier B (speculative unroll) lives in solveAccessTierB.
AccessClassification solveAccessTierA(Operation *memrefOp, Value iv,
                                      int64_t elementBytes);

}  // namespace mlir::lego

#endif  // LEGO_CONVERSION_LEGOVECTORIZE_H
```

- [ ] **Step 4: Implement `solveAccessTierA` in `LegoVectorize.cpp`**

Add the implementation. Key API references:
- `mlir::IRMapping` (in `mlir/IR/IRMapping.h`) — for cloning the address expression with `iv → iv+k` substitution.
- `mlir::OpBuilder` — for constructing `arith.addi` and `arith.subi` to form S(k).
- `mlir::applyPatternsAndFoldGreedily` (in `mlir/Transforms/GreedyPatternRewriteDriver.h`) — for canonicalization.
- `arith::populateArithCanonicalizationPatterns` — patterns to apply.

```cpp
#include "LegoVectorize.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

namespace mlir::lego {

static Value getMemRefIndex(Operation *op) {
  // memref.load and memref.store both have indices() — return the single
  // (or last) index for a 1D access. For multi-dim, future work flattens
  // via memref's strides; v1 expects 1D-flattened addresses post lego-to-arith.
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return load.getIndices().front();
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getIndices().front();
  return nullptr;
}

static Operation *cloneAddressDAG(Value addr, Value iv, Value newIv,
                                  IRMapping &mapping, OpBuilder &builder) {
  // Walk the def-use chain backwards from `addr`, cloning each defining op
  // with iv->newIv substitution. Returns the cloned root op.
  // Implementation note: use a worklist or recursive walker that respects
  // SSA def-before-use. For v1, only handle pure arith ops + block args.
  mapping.map(iv, newIv);
  Operation *defOp = addr.getDefiningOp();
  if (!defOp) return nullptr;  // block arg, not iv-dependent
  // Recursive clone
  for (Value operand : defOp->getOperands()) {
    if (operand == iv) continue;
    if (Operation *opDef = operand.getDefiningOp()) {
      cloneAddressDAG(operand, iv, mapping.lookupOrDefault(iv), mapping, builder);
    }
  }
  return builder.clone(*defOp, mapping);
}

AccessClassification solveAccessTierA(Operation *memrefOp, Value iv,
                                      int64_t elementBytes) {
  Value addr = getMemRefIndex(memrefOp);
  if (!addr) return {AccessKind::NonAffine, 0, -1, elementBytes};

  OpBuilder builder(memrefOp);
  builder.setInsertionPoint(memrefOp);

  // Construct k as a fresh symbolic value (we'll fold-check the result).
  Type indexTy = builder.getIndexType();
  Value k = builder.create<arith::ConstantIndexOp>(memrefOp->getLoc(), 1);
  // newIv = iv + k
  Value newIv = builder.create<arith::AddIOp>(memrefOp->getLoc(), iv, k);

  // Clone addr DAG with substitution
  IRMapping mapping;
  Operation *clonedRoot = cloneAddressDAG(addr, iv, newIv, mapping, builder);
  if (!clonedRoot) {
    // addr doesn't depend on iv — broadcast
    return {AccessKind::Broadcast, 0, -1, elementBytes};
  }

  Value clonedAddr = clonedRoot->getResult(0);
  Value diff = builder.create<arith::SubIOp>(memrefOp->getLoc(),
                                             clonedAddr, addr);

  // Canonicalize the diff expression
  RewritePatternSet patterns(memrefOp->getContext());
  arith::populateArithCanonicalizationPatterns(patterns);
  // Run the patterns on the operation containing `diff`
  (void)applyPatternsAndFoldGreedily(diff.getDefiningOp(), std::move(patterns));

  // Inspect the simplified diff. Three positive cases:
  //   constant 0       -> Broadcast
  //   constant c       -> Strided(c) if c != elementBytes, else Unit
  //   muli %k_const c  -> the stride constant (since k=1, this is c)
  AccessClassification result{AccessKind::NonAffine, 0, -1, elementBytes};

  if (auto cst = diff.getDefiningOp<arith::ConstantIndexOp>()) {
    int64_t c = cst.value();
    if (c == 0) {
      result.kind = AccessKind::Broadcast;
    } else if (c == elementBytes) {
      result.kind = AccessKind::Unit;
    } else {
      result.kind = AccessKind::Strided;
      result.stride = c;
    }
    // Cleanup the temporary diff/k/newIv ops
    diff.getDefiningOp()->erase();
    k.getDefiningOp()->erase();
    newIv.getDefiningOp()->erase();
    return result;
  }

  // Fallback: diff still depends on iv -> non_affine.
  // Cleanup
  diff.getDefiningOp()->erase();
  k.getDefiningOp()->erase();
  newIv.getDefiningOp()->erase();
  return result;
}

}  // namespace mlir::lego
```

(Note: this is the v1 sketch. The clone-and-canonicalize approach has subtleties around inserting and removing ops cleanly. In implementation, prefer doing the analysis in a temporary cloned region, or use `Symbol`-style abstract evaluation that doesn't mutate the IR. A safer alternative is to express S(k) in a Presburger constraint via `mlir::affine::FlatAffineValueConstraints` if the addr is affine; for non-affine cases, the cloning approach is necessary.)

- [ ] **Step 5: Wire `runOnOperation` to use `solveAccessTierA` and emit `vector.transfer_read/write` for unit accesses**

Replace the no-op `runOnOperation` body with:

```cpp
void LegoVectorizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  func.walk([&](scf::ForOp forOp) {
    Value iv = forOp.getInductionVar();
    int64_t elementBytes = 8;  // f64 default; refine in later tasks

    // Collect all memref.load/store inside this loop
    SmallVector<Operation *> accesses;
    forOp.getBody()->walk([&](Operation *op) {
      if (isa<memref::LoadOp, memref::StoreOp>(op))
        accesses.push_back(op);
    });
    if (accesses.empty()) return;

    // Classify
    bool allUnit = true;
    for (Operation *op : accesses) {
      auto cls = solveAccessTierA(op, iv, elementBytes);
      if (cls.kind != AccessKind::Unit && cls.kind != AccessKind::Broadcast) {
        allUnit = false;
        break;
      }
    }
    if (!allUnit) return;

    // Strip-mining + emission lands in subsequent tasks (6-9).
    // For Task 5, we only verify the analyzer fires correctly — rely on
    // the FileCheck test to confirm classification was at least attempted.
    // The actual rewrite to vector.transfer_read happens in Task 8.
  });
}
```

- [ ] **Step 6: Build and run test**

```bash
cd /scratch/general/vast/u1419116/LEGO/build
cmake --build . --target lego-opt check-lego-all -j16
```
Expected: build succeeds. The `lego_vectorize.mlir` test still fails on the new case (we haven't emitted vector ops yet) — that's intentional; the test will pass after Task 8.

For now, **comment out** the `// CHECK: vector.transfer_read` line so the test passes the no-op behavior:

```mlir
// CHECK-LABEL: func.func @row_major_unit_stride
// (vectorization comes online in Task 8)
// CHECK: scf.for
// CHECK: return
```

Run lit:
```bash
./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add include/Lego/Passes.h lib/Lego/Conversion/LegoVectorize.h lib/Lego/Conversion/LegoVectorize.cpp test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: Tier-A stride solve scaffolding (classifier only)"
```

---

### Task 6: Loop-selection scaffolding — collect candidate loops and accesses

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`

- [ ] **Step 1: Refactor `runOnOperation` into a `LoopAnalysis` struct**

Define inside the anonymous namespace in `LegoVectorize.cpp`:

```cpp
struct LoopAnalysis {
  scf::ForOp forOp;
  SmallVector<Operation *> accesses;
  SmallVector<lego::AccessClassification> classes;
  int64_t L_strip = 0;
  int score = 0;
};

static SmallVector<LoopAnalysis> collectCandidateLoops(
    func::FuncOp func, int64_t elementBytes) {
  SmallVector<LoopAnalysis> result;
  func.walk([&](scf::ForOp forOp) {
    LoopAnalysis a;
    a.forOp = forOp;
    forOp.getBody()->walk([&](Operation *op) {
      if (isa<memref::LoadOp, memref::StoreOp>(op))
        a.accesses.push_back(op);
    });
    if (!a.accesses.empty()) result.push_back(a);
  });
  return result;
}
```

Then `runOnOperation` becomes:

```cpp
void LegoVectorizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  int64_t elementBytes = 8;
  auto loops = collectCandidateLoops(func, elementBytes);

  for (auto &a : loops) {
    Value iv = a.forOp.getInductionVar();
    for (Operation *op : a.accesses)
      a.classes.push_back(lego::solveAccessTierA(op, iv, elementBytes));
  }

  // Lemma: a loop is vectorizable iff every access is Unit or Broadcast
  // (Tier B: also CrossBlock — added in Phase C).
  // Strip-mining + rewrite: Tasks 7-9.
}
```

- [ ] **Step 2: Build**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16
```
Expected: success.

- [ ] **Step 3: Re-run lego_vectorize.mlir test**

```bash
./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS (still no-op rewrite).

- [ ] **Step 4: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp
git commit -m "lego-vectorize: extract LoopAnalysis + collectCandidateLoops"
```

---

### Task 7: Per-access vector length and L_strip computation

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`

- [ ] **Step 1: Add per-target register width helper**

In the anonymous namespace:

```cpp
static int64_t getRegisterLanesForType(StringRef target, int64_t elementBytes) {
  // Returns lanes per register for the given element size.
  if (target == "avx512") return 64 / elementBytes;
  if (target == "avx2")   return 32 / elementBytes;
  if (target == "neon")   return 16 / elementBytes;
  return 16 / elementBytes;
}
```

- [ ] **Step 2: Add the per-access Ln + L_strip step**

```cpp
static int64_t gcd(int64_t a, int64_t b) { while (b) { a %= b; std::swap(a, b); } return a; }
static int64_t lcm(int64_t a, int64_t b) { return a / gcd(a, b) * b; }

static int64_t computeStripMineFactor(LoopAnalysis &a, StringRef target,
                                      int64_t elementBytes) {
  int64_t R_T = getRegisterLanesForType(target, elementBytes);
  // Conservatively use trip count if statically known; else use R_T as cap.
  int64_t T = std::numeric_limits<int64_t>::max();
  // For v1, attempt to extract lower/upper/step constants:
  if (auto lb = a.forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>())
    if (auto ub = a.forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>())
      if (auto st = a.forOp.getStep().getDefiningOp<arith::ConstantIndexOp>())
        T = (ub.value() - lb.value()) / st.value();

  int64_t L_strip = 1;
  for (auto &cls : a.classes) {
    int64_t Ln = 1;
    if (cls.kind == lego::AccessKind::Unit) Ln = std::min(R_T, T);
    else if (cls.kind == lego::AccessKind::Broadcast) Ln = R_T;  // doesn't constrain
    else { Ln = 1; break; }  // strided/non_affine in Tier-A: skip in v1 minimum

    L_strip = (L_strip == 1) ? Ln : lcm(L_strip, Ln);
  }
  return L_strip;
}
```

- [ ] **Step 3: Wire it into `runOnOperation`**

```cpp
void LegoVectorizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  int64_t elementBytes = 8;
  auto loops = collectCandidateLoops(func, elementBytes);

  for (auto &a : loops) {
    Value iv = a.forOp.getInductionVar();
    for (Operation *op : a.accesses)
      a.classes.push_back(lego::solveAccessTierA(op, iv, elementBytes));
    a.L_strip = computeStripMineFactor(a, target_.getValue(), elementBytes);
  }

  // Strip-mine + emit happens in Task 8.
}
```

- [ ] **Step 4: Build + lit**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16 && ./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS (still no rewrite emitted).

- [ ] **Step 5: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp
git commit -m "lego-vectorize: per-access Ln + lcm strip-mine factor"
```

---

### Task 8: Emit `vector.transfer_read` / `vector.transfer_write` for unit accesses

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`
- Modify: `test/Lego/lego_vectorize.mlir`

- [ ] **Step 1: Add a helper that strip-mines a `scf::ForOp` by L**

```cpp
// Splits forOp into:
//   scf.for %ti = lb to alignedUb step (L*step) { vector body }
//   scf.for %ti = alignedUb to ub step step    { residual scalar tail }
// Returns the new vector for-op (the original op is replaced).
static scf::ForOp stripMineLoop(scf::ForOp forOp, int64_t L, OpBuilder &builder) {
  Location loc = forOp.getLoc();
  builder.setInsertionPoint(forOp);
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();

  // newStep = step * L
  Value Lval = builder.create<arith::ConstantIndexOp>(loc, L);
  Value newStep = builder.create<arith::MulIOp>(loc, step, Lval);

  // alignedUb = ub - ((ub - lb) % newStep)  — for simplicity, use: floor((ub - lb) / newStep) * newStep + lb
  Value extent = builder.create<arith::SubIOp>(loc, ub, lb);
  Value q = builder.create<arith::DivUIOp>(loc, extent, newStep);
  Value alignedSpan = builder.create<arith::MulIOp>(loc, q, newStep);
  Value alignedUb = builder.create<arith::AddIOp>(loc, lb, alignedSpan);

  auto vecLoop = builder.create<scf::ForOp>(loc, lb, alignedUb, newStep);
  // Move body to the new vector loop: caller will do this via IRMapping.
  // For v1, return vecLoop and let the rewrite step move ops in.
  // The residual loop:
  auto resLoop = builder.create<scf::ForOp>(loc, alignedUb, ub, step);

  return vecLoop;
}
```

(In implementation: the body migration is the tricky part. Use `IRMapping` to clone the original body into both the vector loop and the residual loop, with the vector body's loads/stores rewritten as `vector.transfer_read/write`.)

- [ ] **Step 2: Add the rewrite step that emits vector ops**

```cpp
static void emitVectorBody(scf::ForOp vecLoop, scf::ForOp origLoop,
                           int64_t L_strip, int64_t elementBytes,
                           ArrayRef<Operation *> accesses,
                           ArrayRef<lego::AccessClassification> classes,
                           OpBuilder &builder) {
  // For each access at L_access:
  //   replace memref.load with vector.transfer_read of vector<L_access x type>
  //   replace memref.store with vector.transfer_write
  // For arith ops: vectorize by replacing scalar types with vector<L x ...>
  //
  // Implementation: clone origLoop's body into vecLoop with mapping:
  //   iv -> vecLoop.getInductionVar() (the new ti)
  //   for each load %v = memref.load %m[%iv]:
  //     emit %v_vec = vector.transfer_read %m[%ti] : vector<L_access x f64>
  //     replace uses of %v in the cloned body with %v_vec.
  //   ... etc.
  //
  // For v1 minimum, support only Unit accesses with single-loop iv.
  Location loc = origLoop.getLoc();
  Value newIv = vecLoop.getInductionVar();
  Type elemTy = builder.getF64Type();  // refine in later tasks
  VectorType vecTy = VectorType::get({L_strip}, elemTy);

  builder.setInsertionPointToStart(vecLoop.getBody());

  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), newIv);
  for (Operation &op : origLoop.getBody()->getOperations()) {
    if (auto load = dyn_cast<memref::LoadOp>(&op)) {
      Value vec = builder.create<vector::TransferReadOp>(
          loc, vecTy, load.getMemRef(),
          ValueRange{mapping.lookupOrDefault(load.getIndices().front())},
          /*permutationMap=*/AffineMap{},
          /*padding=*/builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elemTy)),
          /*mask=*/Value{},
          /*inBounds=*/builder.getBoolArrayAttr({true}));
      mapping.map(load.getResult(), vec);
    } else if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      builder.create<vector::TransferWriteOp>(
          loc, mapping.lookupOrDefault(store.getValue()),
          store.getMemRef(),
          ValueRange{mapping.lookupOrDefault(store.getIndices().front())},
          /*permutationMap=*/AffineMap{},
          /*mask=*/Value{},
          /*inBounds=*/builder.getBoolArrayAttr({true}));
    } else if (isa<scf::YieldOp>(op)) {
      // skip — vecLoop has its own yield
    } else {
      // arith op — clone with mapping; result type promotes from scalar to vector
      Operation *cloned = builder.clone(op, mapping);
      // Promote result types to vector
      for (OpResult res : cloned->getResults()) {
        if (!res.getType().isa<VectorType>()) {
          // Replace scalar type with vector type of same element type
          res.setType(VectorType::get({L_strip}, res.getType()));
        }
      }
      mapping.map(op.getResult(0), cloned->getResult(0));
    }
  }
}
```

(The above is a v1 sketch — type promotion is more nuanced for ops that take constants; in implementation, walk operands and replace scalar constants with `vector.broadcast`s. The arith dialect's type-promote utilities (`mlir::vector::populateVectorRewritePatterns`) help.)

- [ ] **Step 3: Wire the rewrite into `runOnOperation`**

```cpp
void LegoVectorizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  int64_t elementBytes = 8;
  auto loops = collectCandidateLoops(func, elementBytes);

  for (auto &a : loops) {
    Value iv = a.forOp.getInductionVar();
    for (Operation *op : a.accesses)
      a.classes.push_back(lego::solveAccessTierA(op, iv, elementBytes));
    a.L_strip = computeStripMineFactor(a, target_.getValue(), elementBytes);

    if (a.L_strip <= 1) continue;

    OpBuilder builder(a.forOp);
    scf::ForOp vecLoop = stripMineLoop(a.forOp, a.L_strip, builder);
    emitVectorBody(vecLoop, a.forOp, a.L_strip, elementBytes,
                   a.accesses, a.classes, builder);
    a.forOp.erase();
  }
}
```

- [ ] **Step 4: Re-enable the FileCheck assertion in `lego_vectorize.mlir`**

```mlir
// CHECK-LABEL: func.func @row_major_unit_stride
// CHECK: vector.transfer_read {{.*}} : memref<1024xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<1024xf64>
// CHECK: return
```

- [ ] **Step 5: Build and run**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16 && ./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS — `vector.transfer_read` and `vector.transfer_write` appear.

- [ ] **Step 6: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: emit vector.transfer_read/write for unit-stride loops"
```

---

### Task 9: Vectorize scalar arith ops with type promotion

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`
- Modify: `test/Lego/lego_vectorize.mlir`

- [ ] **Step 1: Add SAXPY test (multiply + add inside the loop)**

Append to `test/Lego/lego_vectorize.mlir`:

```mlir
// -----

// CHECK-LABEL: func.func @saxpy
// CHECK: %[[A:.+]] = vector.broadcast {{.*}} : f64 to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.mulf %[[A]], {{.*}} : vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>
func.func @saxpy(%a: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi = memref.load %X[%i] : memref<?xf64>
    %yi = memref.load %Y[%i] : memref<?xf64>
    %p  = arith.mulf %a, %xi : f64
    %s  = arith.addf %p, %yi : f64
    memref.store %s, %Y[%i] : memref<?xf64>
  }
  return
}
```

- [ ] **Step 2: Run test to see what currently fails**

```bash
./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: FAIL — `%a` is loop-invariant scalar; current code doesn't broadcast it. Also `arith.mulf` / `arith.addf` need vector-typed operands.

- [ ] **Step 3: Add a broadcast-loop-invariants pre-step**

In `emitVectorBody`, before cloning the body, walk the loop body and identify SSA values defined *outside* the loop that are used inside as scalar but flow into vectorized arith. Wrap each in a `vector.broadcast` to vector<L_strip x T>.

```cpp
// Inside emitVectorBody, before the body-clone walk:
DenseMap<Value, Value> broadcastMap;
origLoop.getBody()->walk([&](Operation *op) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    bool definedOutside = !defOp || !origLoop->isAncestor(defOp);
    if (definedOutside && operand.getType().isF64() &&
        !broadcastMap.contains(operand)) {
      Value bc = builder.create<vector::BroadcastOp>(
          origLoop.getLoc(), VectorType::get({L_strip}, operand.getType()),
          operand);
      broadcastMap[operand] = bc;
    }
  }
});
// Then in the clone walk, when an operand is in broadcastMap, use the broadcast.
```

When cloning arith ops in the body-clone walk, before cloning operate on each operand and substitute it with the broadcast version if applicable. This requires extending `mapping`:

```cpp
for (auto &kv : broadcastMap) mapping.map(kv.first, kv.second);
```

- [ ] **Step 4: Update arith-op cloning to set vector result types**

Replace the `// arith op` branch in the body-clone loop with:

```cpp
} else {
  // Arith op: clone with mapping; promote result types to vector<L_strip x T>.
  Operation *cloned = builder.clone(op, mapping);
  for (OpResult res : cloned->getResults()) {
    Type t = res.getType();
    if (!isa<VectorType>(t)) {
      res.setType(VectorType::get({L_strip}, t));
    }
  }
  for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults())) {
    mapping.map(oldRes, newRes);
  }
}
```

- [ ] **Step 5: Build + lit**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16 && ./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS — both `row_major_unit_stride` and `saxpy` cases.

- [ ] **Step 6: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: vectorize arith ops + broadcast loop-invariant scalars"
```

---

### Task 10: Mixed-precision (f32 input, f64 accumulator)

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`
- Modify: `test/Lego/lego_vectorize.mlir`

- [ ] **Step 1: Add mixed-precision test**

Append to `test/Lego/lego_vectorize.mlir`:

```mlir
// -----

// CHECK-LABEL: func.func @mixed_precision
// CHECK: vector.transfer_read {{.*}} : memref<?xf32>, vector<16xf32>
// CHECK: vector.extract_strided_slice {{.*}} : vector<16xf32> to vector<8xf32>
// CHECK: arith.extf {{.*}} : vector<8xf32> to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @mixed_precision(%X: memref<?xf32>, %C: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi32 = memref.load %X[%i] : memref<?xf32>
    %xi64 = arith.extf %xi32 : f32 to f64
    %ci = memref.load %C[%i] : memref<?xf64>
    %s = arith.addf %ci, %xi64 : f64
    memref.store %s, %C[%i] : memref<?xf64>
  }
  return
}
```

- [ ] **Step 2: Update `getRegisterLanesForType` and per-access elementBytes**

Each access carries its own element type. Refactor `solveAccessTierA` and `LoopAnalysis` to track the element bytes per access:

```cpp
// In LegoVectorize.h:
struct AccessClassification {
  AccessKind kind;
  int64_t stride = 0;
  int64_t boundary = -1;
  int64_t elementBytes = 0;  // already there; ensure it's set per access
};

// In collectCandidateLoops:
for (auto &access : a.accesses) {
  Type elemTy;
  if (auto load = dyn_cast<memref::LoadOp>(access))
    elemTy = load.getType();
  else if (auto store = dyn_cast<memref::StoreOp>(access))
    elemTy = store.getValue().getType();
  int64_t bytes = elemTy.getIntOrFloatBitWidth() / 8;
  // pass bytes to solveAccessTierA
}
```

- [ ] **Step 3: Update `computeStripMineFactor` to lcm over per-access Ln**

Each access's Ln is computed against its own `R_T` (depends on its element bytes). The strip-mine factor is the lcm.

```cpp
// In computeStripMineFactor:
int64_t L_strip = 1;
for (size_t i = 0; i < a.classes.size(); ++i) {
  int64_t bytes = a.classes[i].elementBytes;
  int64_t R_T = getRegisterLanesForType(target, bytes);
  int64_t Ln = (a.classes[i].kind == lego::AccessKind::Unit) ? std::min(R_T, T)
              : (a.classes[i].kind == lego::AccessKind::Broadcast) ? R_T : 1;
  if (Ln == 1) return 1;
  L_strip = (L_strip == 1) ? Ln : lcm(L_strip, Ln);
}
```

- [ ] **Step 4: Emit `vector.extract_strided_slice` at width transitions**

In `emitVectorBody`, when an access's natural width is smaller than `L_strip`, emit `(L_strip / Ln_access)` sub-vector ops at width `Ln_access`, using `vector.extract_strided_slice` to slice the wider vector:

```cpp
// Pseudocode in body emission:
for (Operation &op : origLoop.getBody()->getOperations()) {
  if (auto load = dyn_cast<memref::LoadOp>(&op)) {
    int64_t Ln = getLnForAccess(/* lookup */);
    int subOps = L_strip / Ln;
    SmallVector<Value> chunks;
    for (int j = 0; j < subOps; ++j) {
      Value off = builder.create<arith::AddIOp>(
          loc, mapping.lookupOrDefault(load.getIndices().front()),
          builder.create<arith::ConstantIndexOp>(loc, j * Ln));
      Value chunk = builder.create<vector::TransferReadOp>(
          loc, VectorType::get({Ln}, load.getType()),
          load.getMemRef(), ValueRange{off}, /* ... */);
      chunks.push_back(chunk);
    }
    // Stitch chunks into a wider vector via insert_strided_slice if needed,
    // OR keep as multiple narrower vectors and propagate through arith.
    // Simplest: assemble into vector<L_strip x ...>:
    Value stitched = builder.create<arith::ConstantOp>(loc, /*zero vector*/);
    for (int j = 0; j < subOps; ++j) {
      stitched = builder.create<vector::InsertStridedSliceOp>(
          loc, chunks[j], stitched, /*offsets=*/{j * Ln}, /*strides=*/{1});
    }
    mapping.map(load.getResult(), stitched);
  }
  // ... similar for arith.extf where input is vector<N x f32> and result is vector<N x f64>
}
```

- [ ] **Step 5: Build + lit**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16 && ./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS — mixed_precision test now produces the expected sub-vector ops.

- [ ] **Step 6: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: mixed-precision sub-vector emission via extract_strided_slice"
```

---

## Phase C — Tier B speculative unroll + cross-block

### Task 11: Speculative unroll for piecewise detection

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.h`
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`

- [ ] **Step 1: Declare `solveAccessTierB` in `LegoVectorize.h`**

```cpp
// Speculative unroll: compute concrete addr(iv+0..L-1), classify based on the
// actual address sequence. Returns CrossBlock with boundary set if a single
// piecewise jump is detected; otherwise the same kinds as Tier A.
AccessClassification solveAccessTierB(Operation *memrefOp, Value iv,
                                      int64_t elementBytes, int64_t L);
```

- [ ] **Step 2: Implement `solveAccessTierB`**

```cpp
AccessClassification solveAccessTierB(Operation *memrefOp, Value iv,
                                      int64_t elementBytes, int64_t L) {
  Value addr = getMemRefIndex(memrefOp);
  if (!addr) return {AccessKind::Broadcast, 0, -1, elementBytes};

  // For k in [0, L), compute addr(iv+k) as a constant if possible.
  // Approach: clone the addr DAG with iv->iv+k_const, fold, check if the
  // result is a constant. If yes for all k, we have a concrete address sequence.
  OpBuilder builder(memrefOp);
  builder.setInsertionPoint(memrefOp);
  SmallVector<int64_t> addrs;
  bool allConstant = true;
  for (int64_t k = 0; k < L; ++k) {
    Value kVal = builder.create<arith::ConstantIndexOp>(memrefOp->getLoc(), k);
    Value newIv = builder.create<arith::AddIOp>(memrefOp->getLoc(), iv, kVal);
    IRMapping mapping;
    mapping.map(iv, newIv);
    Operation *cloned = cloneAddressDAG(addr, iv, newIv, mapping, builder);
    if (!cloned) { allConstant = false; break; }
    Value clonedAddr = cloned->getResult(0);
    // Run canonicalization
    RewritePatternSet patterns(memrefOp->getContext());
    arith::populateArithCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(cloned, std::move(patterns));
    if (auto cst = clonedAddr.getDefiningOp<arith::ConstantIndexOp>()) {
      addrs.push_back(cst.value());
    } else {
      allConstant = false;
      break;
    }
    // Cleanup intermediate ops:
    cloned->erase();
    newIv.getDefiningOp()->erase();
    kVal.getDefiningOp()->erase();
  }

  AccessClassification result{AccessKind::NonAffine, 0, -1, elementBytes};
  if (!allConstant) return result;

  // Inspect the address sequence:
  //   - all differ by elementBytes from previous: unit
  //   - constant non-unit stride: strided
  //   - two contiguous runs with single boundary: cross_block(boundary)
  //   - else: non_affine
  if (addrs.size() < 2) return result;
  bool isUnit = true, isStrided = true;
  int64_t s0 = addrs[1] - addrs[0];
  for (size_t i = 1; i < addrs.size(); ++i) {
    int64_t s = addrs[i] - addrs[i-1];
    if (s != elementBytes) isUnit = false;
    if (s != s0) isStrided = false;
  }
  if (isUnit) { result.kind = AccessKind::Unit; return result; }
  if (isStrided) { result.kind = AccessKind::Strided; result.stride = s0; return result; }

  // Detect cross_block: exactly one position p where the diff differs.
  int boundaryCount = 0;
  int64_t boundary = -1;
  for (size_t i = 1; i < addrs.size(); ++i) {
    int64_t s = addrs[i] - addrs[i-1];
    if (s != elementBytes) {
      boundaryCount++;
      boundary = i;
    }
  }
  if (boundaryCount == 1) {
    result.kind = AccessKind::CrossBlock;
    result.boundary = boundary;
    return result;
  }
  return result;  // NonAffine
}
```

- [ ] **Step 3: Wire Tier B into the analysis flow**

In `runOnOperation`, after Tier A:

```cpp
for (auto &a : loops) {
  Value iv = a.forOp.getInductionVar();
  int64_t targetLanes = getRegisterLanesForType(target_.getValue(), elementBytes);
  for (Operation *op : a.accesses) {
    auto cls = lego::solveAccessTierA(op, iv, elementBytes);
    if (cls.kind == lego::AccessKind::NonAffine) {
      // Try Tier B
      cls = lego::solveAccessTierB(op, iv, elementBytes, targetLanes);
    }
    a.classes.push_back(cls);
  }
  // ...
}
```

- [ ] **Step 4: Build (no test yet — Task 13 adds the cross-block test)**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16
```
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp lib/Lego/Conversion/LegoVectorize.h
git commit -m "lego-vectorize: Tier-B speculative unroll for piecewise / cross_block detection"
```

---

### Task 12: Cross-block emission via `vector.shuffle`

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`

- [ ] **Step 1: Extend the access-emission switch with CrossBlock branch**

In `emitVectorBody`, when handling a load with `cls.kind == CrossBlock`:

```cpp
} else if (cls.kind == lego::AccessKind::CrossBlock) {
  // Two adjacent block reads + shuffle.
  int64_t boundary = cls.boundary;       // lane index of discontinuity
  // Block n: addresses [iv+0 .. iv+(boundary-1)]
  // Block n+1: addresses [iv+boundary .. iv+(L-1)] but at the next-block base.
  // Synthesis: load lanes [0..L-1] from both blocks, shuffle.
  Value blockN_base = mapping.lookupOrDefault(load.getIndices().front());
  Value blockNp1_base = builder.create<arith::AddIOp>(
      loc, blockN_base, builder.create<arith::ConstantIndexOp>(loc, boundary));
  Value blockN = builder.create<vector::TransferReadOp>(
      loc, VectorType::get({L_strip}, elemTy),
      load.getMemRef(), ValueRange{blockN_base},
      AffineMap{}, /*padding*/zero, Value{}, builder.getBoolArrayAttr({true}));
  Value blockNp1 = builder.create<vector::TransferReadOp>(
      loc, VectorType::get({L_strip}, elemTy),
      load.getMemRef(), ValueRange{blockNp1_base},
      AffineMap{}, /*padding*/zero, Value{}, builder.getBoolArrayAttr({true}));
  // Shuffle: take lanes [0..boundary-1] from blockN, [boundary..L-1] from blockNp1.
  // vector.shuffle takes indices into the concat [blockN, blockNp1]: lanes 0..L-1 = blockN, L..2L-1 = blockNp1.
  SmallVector<int64_t> indices;
  for (int64_t lane = 0; lane < L_strip; ++lane) {
    if (lane < boundary) indices.push_back(lane);                  // from blockN
    else                 indices.push_back(L_strip + (lane - boundary));  // from blockNp1
  }
  Value shuffled = builder.create<vector::ShuffleOp>(
      loc, blockN, blockNp1, builder.getI64ArrayAttr(indices));
  mapping.map(load.getResult(), shuffled);
}
```

- [ ] **Step 2: Build (test in Task 13)**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16
```
Expected: success.

- [ ] **Step 3: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp
git commit -m "lego-vectorize: emit two-block read + vector.shuffle for cross_block accesses"
```

---

### Task 13: FileCheck test for cross-brick stencil

**Files:**
- Modify: `test/Lego/lego_vectorize.mlir`

- [ ] **Step 1: Add cross-brick stencil test**

Append to `test/Lego/lego_vectorize.mlir`:

```mlir
// -----

// Cross-brick read: A[i+1] within an inner loop where i is the inner brick z dim.
// Brick size = 8 (so when i=7, i+1=8 is in the next brick — discontinuity at lane 7).
// CHECK-LABEL: func.func @cross_brick_stencil
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.shuffle {{.*}} [1, 2, 3, 4, 5, 6, 7, 8]
func.func @cross_brick_stencil(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  // Inner brick loop: 8 iterations on z axis, brick size = 8.
  // The +1 read crosses the brick boundary at lane 7.
  scf.for %z = %c0 to %c8 step %c1 {
    %zp1 = arith.addi %z, %c1 : index
    // Address layout: imagine a brick layout where addr(z) = base + z*8 within
    // the brick; addr(8) jumps to the next brick. For test-IR purposes we
    // synthesize the piecewise structure with arith ops.
    %brick_idx = arith.divui %zp1, %c8 : index
    %inner = arith.remui %zp1, %c8 : index
    %inner_off = arith.muli %inner, %c1 : index
    %brick_off = arith.muli %brick_idx, %c8 : index
    %total = arith.addi %inner_off, %brick_off : index
    %v = memref.load %A[%total] : memref<?xf64>
    memref.store %v, %B[%z] : memref<?xf64>
  }
  return
}
```

- [ ] **Step 2: Run test**

```bash
./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS — the test exercises Tier B detection of the cross_block pattern (the divui/remui fold to a piecewise access) and emits shuffle.

If the canonicalizer doesn't fold `divui (z+1) 8` and `remui (z+1) 8` into piecewise constants for k=0..7 in the speculative unroll, the test will need either: (a) explicit constant folding patterns in the analyzer's canonicalization run, or (b) a simpler test IR that directly exposes the piecewise structure. Pick (b) if (a) is too ambitious for v1.

- [ ] **Step 3: Commit**

```bash
git add test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: FileCheck for cross-brick shuffle synthesis"
```

---

## Phase D — Strided + gather

### Task 14: Strided access emission via `vector.transfer_read` + permutation_map

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`
- Modify: `test/Lego/lego_vectorize.mlir`

- [ ] **Step 1: Add strided test**

```mlir
// -----

// Column-major access in inner-row loop: stride is N (matrix row count).
// Vectorizable as a strided gather.
// CHECK-LABEL: func.func @col_major_strided
// CHECK: vector.transfer_read {{.*}} permutation_map
func.func @col_major_strided(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %off = arith.muli %i, %N : index    // stride N
    %v = memref.load %A[%off] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}
```

- [ ] **Step 2: Implement strided emission**

In `emitVectorBody`:

```cpp
} else if (cls.kind == lego::AccessKind::Strided) {
  // Strided gather: emit vector.transfer_read with permutation_map = (d0)->(d0 * c).
  // MLIR's permutation_map handles only unit-stride; for non-unit, lower to
  // explicit gather. For v1, use vector.gather always for non-unit constant stride.
  int64_t stride = cls.stride;
  // Build a vector<L_strip x index> of [0, stride, 2*stride, ..., (L-1)*stride] and use vector.gather.
  SmallVector<Attribute> indexAttrs;
  for (int64_t j = 0; j < L_strip; ++j)
    indexAttrs.push_back(builder.getIndexAttr(j * stride));
  Value indexVec = builder.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(VectorType::get({L_strip}, builder.getIndexType()),
                                   indexAttrs));
  Value baseIv = mapping.lookupOrDefault(load.getIndices().front());
  Value mask = /* all-true vector mask */;
  Value passThru = builder.create<arith::ConstantOp>(loc, /*zero vector*/);
  Value gathered = builder.create<vector::GatherOp>(
      loc, VectorType::get({L_strip}, elemTy),
      load.getMemRef(), ValueRange{baseIv}, indexVec, mask, passThru);
  mapping.map(load.getResult(), gathered);
}
```

- [ ] **Step 3: Build + lit**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16 && ./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: emit vector.gather for strided access"
```

---

### Task 15: Gather emission for non_affine accesses

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`
- Modify: `test/Lego/lego_vectorize.mlir`

- [ ] **Step 1: Add Morton-style non_affine test**

```mlir
// -----

// Non-affine access (bit-interleave). Vectorizer emits vector.gather with
// runtime-computed index vector.
// CHECK-LABEL: func.func @morton_gather
// CHECK: vector.gather
func.func @morton_gather(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c5 = arith.constant 5 : index   // 0x55... mask, simplified
  %i_outer = arith.constant 0 : index
  scf.for %j = %c0 to %c8 step %c1 {
    %ti = arith.andi %i_outer, %c5 : index
    %tj = arith.andi %j, %c5 : index
    %tj_shl = arith.shli %tj, %c1 : index
    %morton = arith.ori %ti, %tj_shl : index
    %v = memref.load %A[%morton] : memref<?xf64>
    memref.store %v, %B[%j] : memref<?xf64>
  }
  return
}
```

- [ ] **Step 2: Implement non_affine gather emission**

In `emitVectorBody`:

```cpp
} else if (cls.kind == lego::AccessKind::NonAffine) {
  // Build runtime index vector: for k in [0, L_strip), evaluate addr(iv+k).
  // Simplest: speculatively unroll the addr DAG and assemble via insertelement.
  Value indexVec = builder.create<arith::ConstantOp>(
      loc, /*zero vector of vector<L_strip x index>*/);
  for (int64_t k = 0; k < L_strip; ++k) {
    // Construct addr(iv+k) by cloning the DAG with iv -> (newIv + k).
    // Use IRMapping; insert in builder.
    Value kk = builder.create<arith::ConstantIndexOp>(loc, k);
    Value ivPlusK = builder.create<arith::AddIOp>(loc, newIv, kk);
    IRMapping localMap;
    localMap.map(origLoop.getInductionVar(), ivPlusK);
    Operation *clonedAddr = cloneAddressDAG(load.getIndices().front(),
                                            origLoop.getInductionVar(),
                                            ivPlusK, localMap, builder);
    Value addrK = clonedAddr->getResult(0);
    indexVec = builder.create<vector::InsertElementOp>(
        loc, addrK, indexVec, builder.create<arith::ConstantIndexOp>(loc, k));
  }
  Value mask = /* all-true vector mask of L_strip lanes */;
  Value passThru = /* zero vector */;
  Value gathered = builder.create<vector::GatherOp>(
      loc, VectorType::get({L_strip}, elemTy),
      load.getMemRef(), /*indices=*/ValueRange{}, indexVec, mask, passThru);
  mapping.map(load.getResult(), gathered);
}
```

- [ ] **Step 3: Update loop scoring with cost factor**

In `computeStripMineFactor` or a new `scoreLoop` helper:

```cpp
static double scoreLoop(const LoopAnalysis &a, int64_t L_strip) {
  double cost = 1.0;
  for (auto &cls : a.classes) {
    if (cls.kind == lego::AccessKind::Strided ||
        cls.kind == lego::AccessKind::NonAffine) cost *= 5.0;
    if (cls.kind == lego::AccessKind::NonAffine) cost *= 2.0; // stack to ~10
  }
  return double(L_strip) / cost;  // higher = better
}
```

- [ ] **Step 4: Build + lit**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16 && ./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS — morton_gather produces vector.gather.

- [ ] **Step 5: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: emit vector.gather for non_affine + cost-factored scoring"
```

---

### Task 16: Loop-carried dependence analysis (memref base distinctness)

**Files:**
- Modify: `lib/Lego/Conversion/LegoVectorize.cpp`
- Modify: `test/Lego/lego_vectorize.mlir`

- [ ] **Step 1: Add same-base read-then-write test**

```mlir
// -----

// In-place self-update: same memref base for read and write -> conservative skip.
// CHECK-LABEL: func.func @self_update
// CHECK-NOT: vector.transfer_read
// CHECK: memref.load
// CHECK: memref.store
func.func @self_update(%A: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    %im1 = arith.subi %i, %c1 : index
    %prev = memref.load %A[%im1] : memref<?xf64>
    %cur  = memref.load %A[%i] : memref<?xf64>
    %s = arith.addf %prev, %cur : f64
    memref.store %s, %A[%i] : memref<?xf64>
  }
  return
}
```

- [ ] **Step 2: Add base-distinctness check**

Add a helper:

```cpp
static bool memrefBasesDisjoint(Operation *op1, Operation *op2) {
  // Return true if op1 and op2 reference distinct memref SSA values
  // (different memref.alloc results, different function arguments) and
  // there is no memref.cast/subview chain linking them.
  Value m1 = isa<memref::LoadOp>(op1) ? cast<memref::LoadOp>(op1).getMemRef()
                                      : cast<memref::StoreOp>(op1).getMemRef();
  Value m2 = isa<memref::LoadOp>(op2) ? cast<memref::LoadOp>(op2).getMemRef()
                                      : cast<memref::StoreOp>(op2).getMemRef();
  // Walk through casts/subviews to root memref.
  auto rootOf = [](Value v) -> Value {
    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<memref::CastOp>(defOp))
        v = defOp->getOperand(0);
      else if (auto sv = dyn_cast<memref::SubViewOp>(defOp))
        v = sv.getSource();
      else break;
    }
    return v;
  };
  return rootOf(m1) != rootOf(m2);
}

static int64_t computeMinDepDistance(const LoopAnalysis &a) {
  // For each (load, store) pair where store comes after load (or vice versa)
  // in iteration order: if bases overlap (not disjoint) and one is a write,
  // conservatively return Ld = 1 (refuse).
  for (size_t i = 0; i < a.accesses.size(); ++i) {
    for (size_t j = i + 1; j < a.accesses.size(); ++j) {
      bool isWrite_i = isa<memref::StoreOp>(a.accesses[i]);
      bool isWrite_j = isa<memref::StoreOp>(a.accesses[j]);
      if (!(isWrite_i || isWrite_j)) continue;
      if (!memrefBasesDisjoint(a.accesses[i], a.accesses[j])) return 1;
    }
  }
  return std::numeric_limits<int64_t>::max();
}
```

- [ ] **Step 3: Use Ld in `computeStripMineFactor`**

```cpp
int64_t Ld = computeMinDepDistance(a);
// In the per-access Ln formula: Ln = std::min({R_T, T, Ld});
```

- [ ] **Step 4: Build + lit**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16 && ./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_vectorize.mlir
```
Expected: PASS — self_update is left scalar.

- [ ] **Step 5: Commit**

```bash
git add lib/Lego/Conversion/LegoVectorize.cpp test/Lego/lego_vectorize.mlir
git commit -m "lego-vectorize: conservative dep analysis via memref base distinctness"
```

---

## Phase E — Pipelines

### Task 17: `LegoX86VectorPipeline.cpp` — pipeline file mirroring `LegoNVVMPipeline.cpp`

**Files:**
- Create: `lib/Lego/LegoX86VectorPipeline.cpp`
- Modify: `include/Lego/Passes.h`
- Modify: `lib/Lego/Passes.cpp`
- Modify: `lib/Lego/CMakeLists.txt`

- [ ] **Step 1: Add pipeline-options struct + pipeline-builder declaration to `Passes.h`**

```cpp
namespace mlir::lego {
struct LegoToX86VectorPipelineOptions
    : public PassPipelineOptions<LegoToX86VectorPipelineOptions> {
  Option<std::string> cpu{*this, "cpu",
                          llvm::cl::desc("zen3|skx|skl|... (LLVM target-cpu)"),
                          llvm::cl::init("skx")};
};

void buildLegoToX86VectorPipeline(OpPassManager &pm,
                                  const LegoToX86VectorPipelineOptions &opts);
}  // namespace mlir::lego
```

- [ ] **Step 2: Implement pipeline in `LegoX86VectorPipeline.cpp`**

```cpp
//===- LegoX86VectorPipeline.cpp - x86 (AVX-512/AVX2) end-to-end MLIR -----===//

#include "Lego/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::lego {

void buildLegoToX86VectorPipeline(OpPassManager &pm,
                                  const LegoToX86VectorPipelineOptions &opts) {
  // Phase 1 — shared front-end (reuses existing LEGO lowering).
  buildLegoLowerPipeline(pm);

  // Phase 2 — vectorization.
  pm.addPass(arith::createArithIntRangeOptsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  LegoVectorizePassOptions vecOpts;
  vecOpts.target = "avx512";  // overridable via opts.cpu in future
  pm.addPass(createLegoVectorizePass(vecOpts));

  // Phase 3 — vector → LLVM, the rest is shared LLVM tail.
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

}  // namespace mlir::lego
```

- [ ] **Step 3: Register pipeline in `Passes.cpp:registerLegoPipelines()`**

Add to the function that registers existing pipelines:

```cpp
PassPipelineRegistration<LegoToX86VectorPipelineOptions>(
    "lego-to-x86-vector",
    "Lower LEGO dialect to LLVM IR with x86 vector intrinsics (AVX-512/AVX2)",
    buildLegoToX86VectorPipeline);
```

- [ ] **Step 4: Add source to `lib/Lego/CMakeLists.txt`**

Append to the SOURCES list: `LegoX86VectorPipeline.cpp`. Ensure link components include all conversion libraries used.

- [ ] **Step 5: Build**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16
```
Expected: success.

- [ ] **Step 6: Smoke test — verify the pipeline registers**

```bash
./bin/lego-opt --help | grep "lego-to-x86-vector"
```
Expected: line containing `lego-to-x86-vector`.

- [ ] **Step 7: Commit**

```bash
git add include/Lego/Passes.h lib/Lego/LegoX86VectorPipeline.cpp lib/Lego/Passes.cpp lib/Lego/CMakeLists.txt
git commit -m "pipeline: add LegoX86VectorPipeline (--lego-to-x86-vector)"
```

---

### Task 18: `LegoX86VectorPipeline` end-to-end FileCheck

**Files:**
- Create: `test/Lego/lego_to_x86vector.mlir`

- [ ] **Step 1: Write FileCheck for full-pipeline run on a Row-major SAXPY**

```mlir
// RUN: lego-opt %s --lego-to-x86-vector | FileCheck %s

// CHECK-LABEL: llvm.func @saxpy
// CHECK: llvm.intr.fmuladd
// CHECK: llvm.return
func.func @saxpy(%a: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi = memref.load %X[%i] : memref<?xf64>
    %yi = memref.load %Y[%i] : memref<?xf64>
    %p  = arith.mulf %a, %xi : f64
    %s  = arith.addf %p, %yi : f64
    memref.store %s, %Y[%i] : memref<?xf64>
  }
  return
}
```

- [ ] **Step 2: Build and run**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16 && ./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_to_x86vector.mlir
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/Lego/lego_to_x86vector.mlir
git commit -m "test: end-to-end FileCheck for lego-to-x86-vector"
```

---

### Task 19: `LegoArmNeonPipeline.cpp`

**Files:**
- Create: `lib/Lego/LegoArmNeonPipeline.cpp`
- Modify: `include/Lego/Passes.h`
- Modify: `lib/Lego/Passes.cpp`
- Modify: `lib/Lego/CMakeLists.txt`

- [ ] **Step 1: Add options struct and decl to `Passes.h`**

```cpp
namespace mlir::lego {
struct LegoToArmNeonPipelineOptions
    : public PassPipelineOptions<LegoToArmNeonPipelineOptions> {
  Option<std::string> cpu{*this, "cpu",
                          llvm::cl::desc("aarch64 cortex-a76|... target-cpu"),
                          llvm::cl::init("cortex-a76")};
};

void buildLegoToArmNeonPipeline(OpPassManager &pm,
                                const LegoToArmNeonPipelineOptions &opts);
}
```

- [ ] **Step 2: Implement `LegoArmNeonPipeline.cpp`** (mirror x86, with `target = "neon"` for vectorize)

```cpp
#include "Lego/Passes.h"
// ... same includes as LegoX86VectorPipeline.cpp ...

namespace mlir::lego {

void buildLegoToArmNeonPipeline(OpPassManager &pm,
                                const LegoToArmNeonPipelineOptions &opts) {
  buildLegoLowerPipeline(pm);

  pm.addPass(arith::createArithIntRangeOptsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  LegoVectorizePassOptions vecOpts;
  vecOpts.target = "neon";
  pm.addPass(createLegoVectorizePass(vecOpts));

  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

}
```

- [ ] **Step 3: Register and add to CMakeLists**

In `Passes.cpp`:

```cpp
PassPipelineRegistration<LegoToArmNeonPipelineOptions>(
    "lego-to-arm-neon",
    "Lower LEGO dialect to LLVM IR with ARM NEON intrinsics",
    buildLegoToArmNeonPipeline);
```

- [ ] **Step 4: Build**

```bash
cd /scratch/general/vast/u1419116/LEGO/build && cmake --build . --target lego-opt -j16
```
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add include/Lego/Passes.h lib/Lego/LegoArmNeonPipeline.cpp lib/Lego/Passes.cpp lib/Lego/CMakeLists.txt
git commit -m "pipeline: add LegoArmNeonPipeline (--lego-to-arm-neon)"
```

---

### Task 20: ARM NEON FileCheck (vector width = 2 for f64)

**Files:**
- Create: `test/Lego/lego_to_arm_neon.mlir`

- [ ] **Step 1: Write FileCheck**

```mlir
// RUN: lego-opt %s --lego-to-arm-neon | FileCheck %s

// CHECK-LABEL: llvm.func @saxpy_neon
// NEON 128-bit registers: 2 lanes f64.
// CHECK: vector<2xf64>
func.func @saxpy_neon(%a: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi = memref.load %X[%i] : memref<?xf64>
    %yi = memref.load %Y[%i] : memref<?xf64>
    %p  = arith.mulf %a, %xi : f64
    %s  = arith.addf %p, %yi : f64
    memref.store %s, %Y[%i] : memref<?xf64>
  }
  return
}
```

- [ ] **Step 2: Run**

```bash
./bin/llvm-lit -v /scratch/general/vast/u1419116/LEGO/test/Lego/lego_to_arm_neon.mlir
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/Lego/lego_to_arm_neon.mlir
git commit -m "test: ARM NEON pipeline FileCheck (L=2 for f64)"
```

---

## Phase F — Frontend `cpu_jit`

### Task 21: `CPUJITAdapter` class skeleton

**Files:**
- Create: `python/lego/frontends/cpu_jit.py`

- [ ] **Step 1: Write the adapter class with stub methods**

```python
"""CPU JIT adapter — peer of CutileAdapter, routing through the new x86/ARM vector pipeline."""

import ast

from lego.python_printer import LEGOPythonCodePrinter
from lego.frontends._adapter import (
    DSLAdapter,
    write_and_exec_temp_file,
    try_fn_chain_unwrap,
    try_py_func_unwrap,
    try_wrapped_unwrap,
    walk_to_source_fn,
)
from lego.rewriter import rewrite


class CPUJITAdapter(DSLAdapter):
    def unwrap(self, fn):
        original_fn = fn
        wrappers = []
        # No CPU-specific Strategy 1 — just use the generic helpers.
        if not wrappers:
            original_fn, wrappers = try_fn_chain_unwrap(original_fn)
        if not wrappers:
            original_fn, wrappers = try_py_func_unwrap(original_fn)
        if not wrappers:
            original_fn, wrappers = try_wrapped_unwrap(original_fn)
        return walk_to_source_fn(original_fn), original_fn, wrappers

    def find_runtime_vars(self, func_def):
        # CPU convention: any param that's not a constant integer is runtime.
        runtime = set()
        for arg in func_def.args.args:
            # All arguments are runtime by default; constants live in the body.
            runtime.add(arg.arg)
        return runtime

    def get_code_printer(self):
        return LEGOPythonCodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        result = write_and_exec_temp_file(
            new_source, tree, original_fn, return_source=return_source)
        if return_source:
            source_text, _ = result
            return source_text
        namespace, transformed_fn = result
        # Compile via LayoutCompiler with the new pipeline.
        from lego.backend.compiler import LayoutCompiler
        compiler = LayoutCompiler()
        # Build the MLIR module — implementation depends on existing LayoutCompiler API.
        # For v1, defer the JIT-compile detail to LayoutCompiler.compile() with pipeline_name.
        return transformed_fn   # Placeholder until JIT integration in Task 22.


def cpu_jit(fn=None, **kwargs):
    """Decorator that compiles a Python kernel through Lego dialect → x86 vector pipeline → JIT."""
    def decorator(fn):
        return rewrite(fn, CPUJITAdapter(), **kwargs)

    if fn is not None:
        return decorator(fn)
    return decorator
```

- [ ] **Step 2: Verify import works**

```bash
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python python -c "from lego.frontends.cpu_jit import cpu_jit, CPUJITAdapter; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add python/lego/frontends/cpu_jit.py
git commit -m "frontends(cpu_jit): scaffold CPUJITAdapter and decorator"
```

---

### Task 22: Wire `compile_and_wrap` to the new pipeline

**Files:**
- Modify: `python/lego/frontends/cpu_jit.py`

- [ ] **Step 1: Update `compile_and_wrap` to call `LayoutCompiler.compile(..., pipeline_name='lego-to-x86-vector')`**

Replace the stub:

```python
def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                     return_source=False):
    result = write_and_exec_temp_file(
        new_source, tree, original_fn, return_source=return_source)
    if return_source:
        source_text, _ = result
        return source_text
    namespace, transformed_fn = result

    # transformed_fn is a Python function whose body contains LEGO IR-builder
    # calls. Invoke it to materialize the MLIR module, then compile via
    # LayoutCompiler with the new pipeline.
    from lego.backend.compiler import LayoutCompiler
    # Determine pipeline based on host arch — default x86 for CHPC AMD/Intel.
    import platform
    arch = platform.machine().lower()
    if arch in ('x86_64', 'amd64'):
        pipeline = 'lego-to-x86-vector'
    elif arch in ('aarch64', 'arm64'):
        pipeline = 'lego-to-arm-neon'
    else:
        pipeline = 'lego-to-llvm'   # safe fallback

    compiler = LayoutCompiler()
    # The exact module-construction API depends on existing LayoutCompiler;
    # follow the cuTile path's mirror invocation. For v1, the simplest contract
    # is: transformed_fn returns or assigns into a known module attribute that
    # LayoutCompiler.compile() can consume.
    module = transformed_fn(*[])  # adjust per LayoutCompiler API
    compiler.compile(module, pipeline_name=pipeline)
    return compiler   # caller invokes via __call__ semantics defined below
```

(The exact module-construction handoff depends on `LayoutCompiler`'s public API; the implementation MUST verify by reading `LayoutCompiler` and following the same pattern that `LayoutCompiler` users already follow — typically it's a `compile_and_jit(callable)` method or similar.)

- [ ] **Step 2: Smoke test — import and create the decorator works**

```bash
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python python -c "
from lego.frontends.cpu_jit import cpu_jit
@cpu_jit
def f(x): return x
print('decorator applied, type:', type(f))
"
```
Expected: prints something (does NOT need to be a callable — verifies the decorator chain doesn't raise).

- [ ] **Step 3: Commit**

```bash
git add python/lego/frontends/cpu_jit.py
git commit -m "frontends(cpu_jit): wire compile_and_wrap to new pipeline"
```

---

### Task 23: Python integration test — SAXPY end-to-end

**Files:**
- Create: `python/tests/test_cpu_jit.py`

- [ ] **Step 1: Write SAXPY test**

```python
"""End-to-end test for @cpu_jit on SAXPY."""

import numpy as np
import pytest

from lego.frontends.cpu_jit import cpu_jit
from lego import Row, TileBy


@cpu_jit
def saxpy(a, X, Y, N):
    L = TileBy(Row(N), tile_dims=[8])
    for i in L:
        Y[i] = a * X[i] + Y[i]


def test_saxpy_correctness():
    N = 1024
    rng = np.random.default_rng(0)
    a = 2.5
    X = rng.standard_normal(N)
    Y = rng.standard_normal(N)
    Y_ref = a * X + Y
    saxpy(a, X, Y, N)
    np.testing.assert_allclose(Y, Y_ref, rtol=1e-12)
```

- [ ] **Step 2: Run the test**

```bash
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python pytest python/tests/test_cpu_jit.py -v
```
Expected: PASS — the JIT-compiled SAXPY produces numerically-identical output.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_cpu_jit.py
git commit -m "test(cpu_jit): SAXPY end-to-end correctness"
```

---

## Phase G — Proof points

### Task 24: Within-brick proof-point benchmark

**Files:**
- Create: `evaluation/cpu_vector_proof/brick_within_cell/kernel.py`
- Create: `evaluation/cpu_vector_proof/brick_within_cell/measure.py`
- Create: `evaluation/cpu_vector_proof/brick_within_cell/README.md`

- [ ] **Step 1: Write the kernel + scalar reference**

`kernel.py`:

```python
"""Within-brick compute kernel — no cross-brick reads."""
import numpy as np

from lego.frontends.cpu_jit import cpu_jit
from lego import Row, TileBy

BRICK = 8


@cpu_jit
def brick_compute(A, B, NX, NY, NZ):
    L = TileBy(Row(NX, NY, NZ), tile_dims=[BRICK, BRICK, BRICK])
    for ix, iy, iz in L:
        # Within-brick computation: no neighbor access across brick boundary.
        v = A[ix, iy, iz]
        B[ix, iy, iz] = v * 2.0 + 1.0


def brick_compute_scalar(A, B):
    B[:] = A * 2.0 + 1.0
```

- [ ] **Step 2: Write the measurement script**

`measure.py`:

```python
"""Measure within-brick proof-point speedup."""
import json
import time
import numpy as np

from kernel import brick_compute, brick_compute_scalar

NX = NY = NZ = 64
WARMUP = 5
TIMED = 30


def measure(fn, A, B):
    for _ in range(WARMUP): fn(A, B)
    times = []
    for _ in range(TIMED):
        t0 = time.perf_counter_ns()
        fn(A, B)
        times.append(time.perf_counter_ns() - t0)
    return np.median(times) / 1e6   # ms


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((NX, NY, NZ))
    B = np.empty_like(A)

    treatment = lambda a, b: brick_compute(a, b, NX, NY, NZ)
    baseline = lambda a, b: brick_compute_scalar(a, b)

    t_treat = measure(treatment, A, B)
    t_base = measure(baseline, A, B)
    speedup = t_base / t_treat

    print(json.dumps({"baseline_ms": t_base, "treatment_ms": t_treat, "speedup": speedup}, indent=2))
    if speedup < 2.0:
        raise SystemExit(f"speedup {speedup:.2f}× < 2× target")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-run the benchmark**

```bash
source /scratch/general/vast/u1419116/LEGO/.venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_vector_proof/brick_within_cell
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python python measure.py
```
Expected: prints JSON with `speedup ≥ 2.0` on AVX-512 hardware.

- [ ] **Step 4: Commit**

```bash
git add evaluation/cpu_vector_proof/brick_within_cell/
git commit -m "eval: within-brick proof-point benchmark"
```

---

### Task 25: Cross-brick stencil proof-point benchmark

**Files:**
- Create: `evaluation/cpu_vector_proof/brick_stencil_cross/kernel.py`
- Create: `evaluation/cpu_vector_proof/brick_stencil_cross/measure.py`
- Create: `evaluation/cpu_vector_proof/brick_stencil_cross/README.md`

- [ ] **Step 1: Write 3D 7-point stencil kernel with cross-brick reads**

`kernel.py`:

```python
"""3D 7-point stencil with explicit cross-brick neighbor reads."""
import numpy as np

from lego.frontends.cpu_jit import cpu_jit
from lego import Row, TileBy

BRICK = 8


@cpu_jit
def stencil_7pt_brick(A, B, NX, NY, NZ):
    L = TileBy(Row(NX, NY, NZ), tile_dims=[BRICK, BRICK, BRICK])
    for ix, iy, iz in L:
        c  = A[ix,   iy,   iz]
        xm = A[ix-1, iy,   iz]
        xp = A[ix+1, iy,   iz]
        ym = A[ix,   iy-1, iz]
        yp = A[ix,   iy+1, iz]
        zm = A[ix,   iy,   iz-1]
        zp = A[ix,   iy,   iz+1]
        B[ix, iy, iz] = (c + xm + xp + ym + yp + zm + zp) / 7.0


def stencil_7pt_scalar(A, B):
    B[1:-1, 1:-1, 1:-1] = (
        A[1:-1, 1:-1, 1:-1] +
        A[0:-2, 1:-1, 1:-1] + A[2:, 1:-1, 1:-1] +
        A[1:-1, 0:-2, 1:-1] + A[1:-1, 2:, 1:-1] +
        A[1:-1, 1:-1, 0:-2] + A[1:-1, 1:-1, 2:]
    ) / 7.0
```

- [ ] **Step 2: Measurement script**

`measure.py`:

```python
"""Measure cross-brick stencil proof-point speedup."""
import json
import time
import numpy as np

from kernel import stencil_7pt_brick, stencil_7pt_scalar

NX = NY = NZ = 64
WARMUP = 5
TIMED = 30


def measure(fn, A, B):
    for _ in range(WARMUP): fn(A, B)
    times = []
    for _ in range(TIMED):
        t0 = time.perf_counter_ns()
        fn(A, B)
        times.append(time.perf_counter_ns() - t0)
    return np.median(times) / 1e6   # ms


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((NX, NY, NZ))
    B = np.empty_like(A)

    treatment = lambda a, b: stencil_7pt_brick(a, b, NX, NY, NZ)
    baseline = lambda a, b: stencil_7pt_scalar(a, b)

    t_treat = measure(treatment, A, B)
    t_base = measure(baseline, A, B)
    speedup = t_base / t_treat

    print(json.dumps({"baseline_ms": t_base, "treatment_ms": t_treat, "speedup": speedup}, indent=2))
    if speedup < 2.0:
        raise SystemExit(f"speedup {speedup:.2f}× < 2× target")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-run + verify ≥2× speedup**

```bash
source /scratch/general/vast/u1419116/LEGO/.venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO/evaluation/cpu_vector_proof/brick_stencil_cross
PYTHONPATH=/scratch/general/vast/u1419116/LEGO/python python measure.py
```
Expected: JSON with `speedup ≥ 2.0`. **This is the headline brick-LOSS-flip result.**

- [ ] **Step 4: Commit**

```bash
git add evaluation/cpu_vector_proof/brick_stencil_cross/
git commit -m "eval: cross-brick stencil proof-point (flips cand 11 LOSS pattern)"
```

---

## Phase H — Roadmap update

### Task 26: Replace R1 in `evaluation/roadmap.md` and retain R13/R14/R15/R17

**Files:**
- Modify: `evaluation/roadmap.md`

- [ ] **Step 1: Read current R1 entry**

```bash
sed -n '13,65p' /scratch/general/vast/u1419116/LEGO/evaluation/roadmap.md
```

- [ ] **Step 2: Replace R1 with the new framing**

Replace the R1 section (lines roughly 13–65) with:

```markdown
## R1 — End-to-end MLIR vectorization pipeline (DONE in v1)

**Status:** delivered by `LegoX86VectorPipeline` + `LegoArmNeonPipeline` + `lego-vectorize` pass and the `@cpu_jit` Python frontend. Spec at `docs/superpowers/specs/2026-05-01-lego-cpu-vector-pipeline-design.md`. See proof points at `evaluation/cpu_vector_proof/{brick_within_cell,brick_stencil_cross}/`.

**Re-test list:** brick stencils 11/12/13/14/29 should be re-measured under the new path once eval-candidate migration (R13) lands.
```

- [ ] **Step 3: Append (or update) R13/R14/R15/R17 entries**

Ensure the roadmap contains:

```markdown
## R13 — AOT object-file path for evaluation candidate migration

**Status:** open. Plumb `mlir-translate` + `llc` so `cpu_jit`-decorated kernels produce relocatable .o files; migrate eval candidates from `cxx_gen` to `cpu_jit`. **Re-test list:** all brick + GEMM-tile candidates.

## R14 — SMT-driven dep analysis (opt-in, for self-update kernels)

**Status:** open. Reuse `LegoExternalSMTVerifier`, gated behind `--lego-vectorize{smt-dep=true}`. Unlocks Gauss-Seidel / in-place reductions.

## R15 — ARM SVE pipeline (scalable vectors)

**Status:** open. NEON delivered in v1. SVE adds scalable-vector machinery (vector length is a runtime quantity); requires `vector.scalable_extract` and a scalable-aware `lego-vectorize` mode.

## R17 — GPU lane-fold equivalent

**Status:** open. Apply the cross-block speculative-unroll pattern to the GPU pipelines for warp-level intrinsics.
```

- [ ] **Step 4: Update the roadmap summary table**

Locate the summary table (around line 391 in the current file) and change R1's row to "closed (v1)". Add R13/R14/R15/R17 rows.

- [ ] **Step 5: Commit**

```bash
git add evaluation/roadmap.md
git commit -m "roadmap: R1 closed (CPU vector pipeline shipped); add R13/R14/R15/R17"
```

---

## Self-review checklist

Run this once after working through all tasks before declaring complete:

- [ ] Every spec acceptance criterion (§8 of the design doc) maps to at least one task above.
- [ ] `check-lego-all` passes after Tasks 4, 8, 17, 19, 23.
- [ ] FileCheck tests cover the six §6 cases: row/col-major (Task 9), Morton/non_affine (Task 15), brick within-cell (Task 8 + Task 24), cross-brick (Tasks 12-13), anti-diagonal wavefront (add to lego_vectorize.mlir if not yet present).
- [ ] cuTile regression: pre-Task-2 baseline matches post-Task-2 counts.
- [ ] Both proof-point benchmarks achieve ≥2× speedup on AVX-512.
- [ ] ARM NEON pipeline registers and produces non-empty IR (Task 19/20); cross-compile + qemu-aarch64 run is recorded in the implementation log (out-of-band of this plan).
- [ ] Roadmap updated; spec referenced.
