# LEGO Feature Roadmap & Architecture Improvement Spec

**Date:** 2026-04-09
**Author:** Amir Mohammad Tavakkoli (with Claude)
**Status:** Draft
**Context:** LEGO is a compiler-agnostic framework for memory layout algebra and code generation, published at CGO 2026. This spec defines the feature roadmap and architectural improvements needed to grow adoption among GPU kernel developers and compiler researchers.

---

## 1. Goals & Constraints

### Target audiences (in priority order)
1. **GPU kernel developers** — People writing Triton/CUDA kernels who want better layouts without understanding the algebra
2. **Compiler researchers** — People building new GPU compilers or extending existing MLIR-based tools

### Key barriers to address
- **Discovery:** Users don't know LEGO exists or what it can do
- **Integration:** Hard to plug into existing Triton/PyTorch/CUDA workflows

### Entry points (all functional today)
- `pip install lego` — Python package
- LEGO Studio — Browser-based IDE with WASM compiler
- MLIR dialect — `lego-opt` tool and C++ API

### Success criteria (6-month horizon)
- At least one external team using LEGO in a real GPU kernel pipeline
- LEGO referenced in Triton/CUTLASS community discussions
- Follow-up publications building on CGO 2026 paper
- Feature parity with CuTe/Triton layout capabilities

### Resource reality
- Solo dedicated developer, full-time

---

## 2. Critical Bugs (Fix Immediately)

These are correctness issues that should be fixed before any new feature work.

### B1: `super().__init__(*args, kwargs)` — missing `**` in 7 printer files

**Files:** `c_printer.py:7`, `cxx_printer.py:9`, `fortran_printer.py:11`, `rust_printer.py:9`, `js_printer.py:9`, `glsl_printer.py:9`, `julia_printer.py:9`

**Impact:** All printer keyword arguments (settings, precision) are silently swallowed as a positional arg. Any non-default SymPy printer configuration is corrupted.

**Fix:** Change `super().__init__(*args, kwargs)` to `super().__init__(*args, **kwargs)` in all 7 files.

### B2: `arith.divui` used for signed index division

**Files:** `python/lego/backend/symbolic.py:119`, `symbolic.py:133`

**Impact:** Unsigned division on negative indices gives wrong results. Affects any layout involving negative offsets.

**Fix:** Change `arith.divui` to `arith.divsi`, or document why unsigned is safe for the layouts currently supported.

### B3: Dead code — `GroupBy.transform` / `inverse_transform`

**File:** `python/lego/core.py:448-461`

**Impact:** These methods reference a non-existent `get_compiler` import. Calling them crashes.

**Fix:** Delete the dead methods or implement them properly by delegating to `LayoutCompiler`.

### B4: Ghost pass — `createLegoVerifyGenpConsistencyPass`

**File:** `include/Lego/Passes.h:19`

**Impact:** Declared but never defined anywhere in `lib/Lego/`. Would cause a linker error if invoked.

**Fix:** Remove the declaration, or implement the pass.

### B5: Missing `bench_utils.py` source in puzzles

**Location:** `python/examples/puzzles/`

**Impact:** Only `.pyc` exists in `__pycache__/`. Puzzles cannot run from a clean checkout. Not portable across Python versions.

**Fix:** Restore `bench_utils.py` as a committed source file.

### B6: Typo in `symbolic/graphene.py`

**File:** `python/examples/symbolic/graphene.py:3`

**Impact:** `postive=True` should be `positive=True`. SymPy silently ignores the misspelled constraint.

**Fix:** One-character fix: `postive` -> `positive`.

### B7: Spurious `torch` import in JAX example

**File:** `python/examples/jax/hello_world.py:5`

**Impact:** Imports `torch` in a JAX-only example. Confusing and creates a spurious dependency.

**Fix:** Remove the `import torch` line.

---

## 3. Architecture Improvements

### Quick wins (1-2 days each)

#### A1: Extract `LEGOStaticLangPrinter` base class

**Problem:** `_print_BroadcastRange`, `_print_floor`, `_print_Pow`, `_print_Mod` are copy-pasted across 7 printer files with only trivial syntax differences.

**Fix:** Create a shared mixin class (e.g., `LEGOStaticLangPrinter`) that holds the common logic. Each language printer overrides only the syntax-specific parts (e.g., `pow` vs `std::pow` vs `.powi()`). Eliminates ~120 duplicated lines.

**Files:** `c_printer.py`, `cxx_printer.py`, `fortran_printer.py`, `rust_printer.py`, `js_printer.py`, `glsl_printer.py`, `julia_printer.py`

#### A2: Extract `_write_and_exec_temp_file` helper

**Problem:** `triton_jit.py:254-299` and `cutile_jit.py:91-128` have nearly identical temp-file management: env-var reads, `makedirs`, `atexit.register` cleanup, `code.replace(co_filename=...)`.

**Fix:** Extract shared logic into `_adapter.py`. Each adapter provides only the DSL-specific re-wrapping.

#### A3: Fix Triton import in shared rewriter

**Problem:** `rewriter.py:128` imports `extract_block_ptr_metadata` from `triton_jit` unconditionally. This is a layering violation — the DSL-agnostic rewriter shouldn't depend on Triton.

**Fix:** Move the block-ptr logic into `TritonAdapter` via a callback or adapter method. The rewriter calls an adapter hook instead of importing directly.

#### A4: Replace wildcard import in `symbolic.py`

**Problem:** `symbolic.py:4` does `from lego.core import *` — anti-pattern in an internal module.

**Fix:** Explicit imports of the 6 layout classes actually used.

### Medium effort (3-5 days each)

#### A5: Rename `OrderBy.OrderBy` and fix mutation

**Problem:** The `OrderBy` class has an instance method named `OrderBy` (same as the class). Additionally, calling `.OrderBy()` mutates `self.chain` in-place via `append`, creating shared mutable state.

**Fix:** Rename to `OrderBy.then()` or `OrderBy.followed_by()`. Return a new `OrderBy` instance instead of mutating `self`.

**File:** `python/lego/core.py:246-267`

#### A6: Extend SMT verifiers beyond GenP

**Problem:** `lego-verify-coalescing` and `lego-verify-bank-conflicts` silently skip non-GenP layouts (return `success()` with no warning in `SMTUtils.cpp:463-465`). Users think verification passed when it was actually skipped.

**Fix:** For non-GenP layouts, lower them to an equivalent GenP representation internally before verification. Alternatively, emit a warning when verification is skipped.

**Files:** `lib/Lego/SMTUtils.cpp`, `lib/Lego/LegoVerifyCoalescing.cpp`, `lib/Lego/LegoVerifyBankConflicts.cpp`

#### A7: Fix `LoadOpLowering` for function-argument views

**Problem:** `LegoToArith.cpp:500-502` hard-requires the view's defining op to be a `CastViewOp` in the same block. Views passed through function arguments cannot be lowered.

**Fix:** Support views from block arguments by looking up the `CastViewOp` through the call chain, or by attaching layout metadata as function argument attributes.

#### A8: Add missing lit tests

**Gaps identified:**
- No lit test for `lego-to-nvvm` (NVVM pipeline)
- No lit test for `lego-to-llvmspirv` (LLVM SPIR-V pipeline)
- No standalone test for `lego-materialize-assume-bounds`
- Coalescing/bank-conflict silent-skip behavior untested
- `view_ops.mlir` has a placeholder inverse that returns `(0, 0)` for all inputs — the broken inverse is unchecked

---

## 4. Feature Roadmap

### Phase 1: Integration-first (Months 1-2)

#### F1: Auto-layout selection

**Status:** Idea — needs further design thinking.

**Concept:** Given an access pattern, automatically pick the optimal layout. `lego.auto(shape, access_pattern="coalesced")`. Uses SMT verifiers to validate candidates statically.

**Depends on:** A6 (SMT verifiers must work beyond GenP)

**Open questions:**
- How to specify access patterns — declarative or inferred from code?
- How to rank candidates when multiple pass SMT verification?
- Should this be eager (pick at import time) or lazy (pick at first kernel launch)?

#### F2: Triton auto-tuning integration

**Status:** Idea — needs further design thinking.

**Concept:** A `@lego.autotune` decorator that generates a layout search space (base type x tile sizes x swizzle bits) and integrates with `@triton.autotune`. Optional SMT pre-filtering to prune invalid candidates.

**Open questions:**
- How to express layout search spaces composably?
- How to bridge LEGO's symbolic parameters with Triton's `tl.constexpr` configs?
- Performance of SMT pre-filtering vs just benchmarking all candidates?

#### F3: PyTorch custom op integration

**Concept:** A `@lego.torch_op` decorator that wraps a LEGO GPU kernel as a `torch.library` custom op with autograd support.

**Depends on:** PyTorch integration redesign (see Section 5)

#### F4: CLI codegen tool

**Concept:** `lego-gen --layout "Tiled(M,N, tile=(64,64))" --target cuda_c` prints optimized index code to stdout. Works with pipes and Makefiles.

**Deliverable:** A standalone CLI entry point that parses a layout spec string, runs the MLIR pipeline, and prints code via the existing printer infrastructure.

#### F5: Fix `torch.compile` backend

**Problem:** `fx_backend.py` is a complete no-op. `_find_inverse_pairs` returns `[]`. `optimize_lego_graph` returns the graph unchanged. Backend registration is import-order-dependent.

**Deliverable:** Either implement real layout fusion in the FX graph, or remove the backend registration to avoid misleading users. See Section 5 for the full PyTorch redesign.

### Phase 2: Showcase-first (Months 2-3)

#### F6: Benchmark suite

**Concept:** Automated benchmarks comparing LEGO-generated code vs hand-written for: vecadd, matmul, softmax, flash attention. Across CUDA and ROCm. Published as CI artifacts with charts.

**Why:** Every competing framework has benchmarks. LEGO doesn't. "Show, don't tell."

#### F7: LEGO Studio improvements

**Concept:** (a) Ship pre-built WASM binary or add `viz/Makefile` with build instructions, (b) URL-shareable state via `location.hash`, (c) More presets (swizzle, Z-curve, warp-tiled), (d) Populate the formula display (`#formula-output` exists but is never written to), (e) Add `RegP` to the GUI builder.

**Note:** The `frontend-design` skill can be used for the Studio redesign in a separate session.

#### F8: Examples overhaul

**Deliverables:**
- Add `python/examples/README.md` index by category and required hardware
- Fill missing puzzles: 20, 30, 31, 32
- Fix puzzle 33 tensor core f16 buffer issue
- Add Flash Attention example
- Add PyTorch custom op example
- Extend codegen backends beyond row-major hello world (add tiled layout examples)
- Add at least one `KernelBuilder` API example beyond vecadd (e.g., matmul)

#### F9: Persistent kernel example

**Concept:** A work-queue kernel that runs as a single launch, processing tiles in a loop. Demonstrates LEGO layout for tile-to-thread mapping. Hot pattern in LLM inference.

### Phase 3: Capability-first (Months 3-5)

#### F10: Warp specialization support

**Concept:** New layout ops or combinators for expressing producer/consumer warp roles. Integration with Triton's warp-spec roadmap.

#### F11: Blackwell/SM100 layouts

**Concept:** Support for TMA 2D descriptors, cluster-level layouts, and the new distributed shared memory model. Complete puzzle 34's cluster API (currently replaced by independent block execution).

#### F12: Intel GPU backend (XeVM)

**Concept:** Phase-2 pipeline only — `LegoXeVMPipeline.cpp`. The existing GPU pipeline architecture is well-factored for this (Phase 1 and Phase 3 are shared; only Phase 2 varies).

#### F13: Sparse layout support

**Concept:** New `lego.sparse` op family for CSR/COO/block-sparse index spaces. Integration with MLIR's `sparse_tensor` dialect. Novel research contribution — no competitor does this well.

#### F14: Multi-GPU / distributed layouts

**Status:** Promising research direction. Correctness of existing prototype needs validation before building APIs.

**What exists:** `python/examples/distributed/comm_derive.py` — a symbolic communication derivation engine that derives collective types and volumes from layout algebra. Tested on 6 patterns. `summa_e2e.py` — full SUMMA derivation.

**Correctness concerns:**
- `_classify_reduction` (AllGather vs AllReduce) uses a heuristic, not a proof
- Block sizes assume exact divisibility (`floor(M/Pr)`) — non-divisible shapes may give wrong ownership
- Stencil classifier doesn't handle wrap-around or boundary conditions
- Only tested on 6 patterns with symbolic values — no numerical stress testing

**Proposed levels (contingent on correctness validation):**
- Level 1: Validate existing prototype against known results (SUMMA, Cannon, etc.)
- Level 2: Promote `TensorDist` + `derive_communication` to `lego/distributed/` as a library
- Level 3: Add `compare_distributions()` — rank alternative distributions by symbolic volume
- Level 4: Codegen for `torch.distributed` / MPI / NCCL
- Level 5: Integration with `LegoLayout` and `LegoTensor`

---

## 5. PyTorch Integration Redesign

**Status:** Needs separate deep-dive session for architecture decisions.

### Current state summary

The PyTorch integration is 5 disconnected layers:

| Layer | File | Status |
|-------|------|--------|
| SymPy-to-PyTorch compiler | `torch_layout.py` | Works but limited (no vmap, no TorchScript) |
| LegoTensor subclass | `torch_tensor.py` | Uses `__torch_function__` (wrong hook for torch.compile) |
| Custom op | `torch_ops.py` | Never called by the transform path |
| torch.compile backend | `fx_backend.py` | Complete no-op |
| User API | `python_mlir.py` | Inconsistent paths for torch vs numpy |

### Critical issues found

1. **`compose()` broken for torch:** `_composed_perm` is set but never checked in the torch transform path. Composed layouts silently apply only the first layout.
2. **`torch.compile(backend="lego")` is a no-op:** Zero optimization. Delegates unchanged to inductor.
3. **`LegoTensor` uses wrong hook:** `__torch_function__` is bypassed by Dynamo. Layout metadata lost inside compiled functions.
4. **`torch.ops.lego.permute` disconnected:** Has autograd + fake tensor impl but nothing calls it.
5. **`BatchedLayout` uses slower path:** Materializes O(numel) perm table while unbatched uses O(1) arithmetic.
6. **Autotune timing broken:** No `torch.cuda.synchronize()`. All GPU benchmarks are wrong.
7. **Backend registration order-dependent:** `import lego; import torch` -> backend never registered.

### Redesign direction (to be finalized in a follow-up session)

Unify into 3 clean layers:
1. **Op layer** — All transforms go through `torch.ops.lego.*` custom ops with autograd, fake tensor impls, and torch.export support
2. **Tensor layer** — `LegoTensor` with `__torch_dispatch__` (survives torch.compile), pickling, contiguous() override
3. **Compiler layer** — Real layout fusion in the FX graph using MLIR codegen

---

## 6. Recommended Execution Order

```
Month 1:  Bugs (B1-B7) + Architecture quick wins (A1-A4) + F4 (CLI) + F5 (torch.compile decision)
Month 2:  Architecture medium (A5-A8) + F6 (benchmarks) + PyTorch redesign session
Month 3:  F7 (Studio, using frontend-design skill) + F8 (examples overhaul) + F3 (PyTorch custom op)
Month 4:  F9 (persistent kernels) + F10 (warp spec) + F14 Level 1 (correctness validation)
Month 5:  F11 (Blackwell) + F12 (Intel) + F14 Level 2-3 (if validated)
```

F1 (auto-layout) and F2 (auto-tuner) are slotted after further design thinking — they require A6 (SMT beyond GenP) as a prerequisite.

F13 (sparse layouts) is deferred as a research exploration — no timeline commitment.

---

## 7. Open Questions

1. **F1/F2 design:** How should auto-layout selection and auto-tuning interact? Should they be separate features or one unified system?
2. **PyTorch architecture:** Which of the 3 proposed layers should be built first? Does Level 1 (ops) unlock everything else?
3. **F14 correctness:** What's the right formalization for "correct communication derivation"? Can SMT verify the derivation engine itself?
4. **F11 Blackwell:** When will Triton's Blackwell support stabilize enough to build on?
5. **Benchmark baselines:** Which hand-written kernels should LEGO be compared against? CuTe, Triton-default, or vendor-tuned (cuBLAS/rocBLAS)?

---

## Competitive Landscape Reference

| Feature | LEGO | CuTe/CUTLASS | Triton | Composable Kernel |
|---------|------|-------------|--------|-------------------|
| Layout algebra | Full | Full (hierarchical) | BlockedEncoding | Morton/Swizzle |
| Multi-backend codegen | 7+ languages, 4 GPU targets | CUDA only | CUDA, ROCm, Intel | ROCm only |
| TMA/block_ptr | Triton frontend | Native | Native (Hopper+) | N/A |
| Warp specialization | Not present | Native | In development | N/A |
| Auto-tuning | Broken (no CUDA sync) | Profile-guided | Built-in autotuner | Built-in |
| Browser IDE | LEGO Studio (WASM) | None | None | None |
| SMT verification | Yes (bijectivity, coalescing, bank conflicts) | No | No | No |
| Distributed derivation | Prototype (examples only) | N/A | N/A | N/A |
| PyTorch integration | Partial (5 disconnected layers) | Via Python DSL | Native | Via HIP |
