# LEGO: Remaining Tasks & Future Directions

**Date:** 2026-04-10 (updated)
**Purpose:** Reference for parallel agent sessions. Each section is independent and can be worked on by a separate agent.

---

## Completed Work

### ~~A5: Rename OrderBy.OrderBy → .then()~~
**Status:** Done — PR #81

### ~~A6: Extend SMT verifiers beyond GenP~~
**Status:** Done — PR #83
**What changed:** Replaced the old standalone `lego-verify-coalescing` and `lego-verify-bank-conflicts` passes with a unified `lego.check` op + `lego-verify` pass. Verification now runs post-lowering on optimized arith inside the `lego-lower` pipeline. Works with all layout types.

### ~~A7: Fix LoadOpLowering for function-argument views~~
**Status:** Done — PR #83
**What changed:** Added `resolveViewArguments()` pre-pass in `LegoToArithPass` that expands function signatures to include the underlying memref, clones layout-defining ops into the callee, and inserts a `CastViewOp` at the function entry.

### ~~F12: Intel GPU backend (XeVM)~~
**Status:** Done — PR #85
**What changed:** Added `lego-to-xevm` pipeline following the three-phase GPU pattern. Phase 2 uses `SetXeVMTargetPass` (sets `#xevm.target` with spirv64 triple) + `GPUToLLVMSPV` conversion. Gated on `LEGO_HAS_XEVM` (derived from SPIRV LLVM target). Registered `"intel"` GPUTarget in Python backend. All 29 puzzles compile to Intel target. Also fixed `check-lego-puzzles` to always register (was gated on `LEGO_ENABLE_RUNNERS`).

### ~~A8: Add missing lit tests~~
**Status:** Done — PR #86
**What changed:** Added lit tests for NVVM pipeline (`lego_to_nvvm.mlir`), LLVM SPIR-V pipeline (`lego_to_llvmspirv.mlir`), and `lego-materialize-assume-bounds` standalone pass (`materialize_assume_bounds.mlir`). Fixed `view_ops.mlir` placeholder inverse test.

---

## Feature Ideas (ready for design)

### F1: Auto-layout selection
**Concept:** `lego.auto(shape, access_pattern="coalesced")` — given an access pattern, automatically pick the optimal layout using SMT verification to rank candidates.
**Depends on:** Nothing (A6 is done — `lego.check` makes this natural: generate candidate layouts, insert checks, run `lego-lower`, see which pass)
**Open questions:** Declarative vs inferred access patterns, eager vs lazy selection
**Effort:** 2-3 weeks design + implementation

### F2: Triton auto-tuning integration
**Concept:** `@lego.autotune` decorator that generates layout search spaces (base type x tile sizes x swizzle bits) and integrates with `@triton.autotune`. Optional SMT pre-filtering.
**Open questions:** How to bridge LEGO symbolic params with Triton constexpr configs
**Effort:** 2-3 weeks

### F4: CLI codegen tool
**Concept:** `lego-gen --layout "TiledPermute(M,N, tile=(64,64))" --target cuda_c` -> prints optimized index code to stdout. Works with pipes and Makefiles.
**Depends on:** Nothing — fully independent
**Effort:** 1 week

### F6: Benchmark suite
**Concept:** Automated benchmarks comparing LEGO-generated code vs hand-written for vecadd, matmul, softmax, flash attention. Across CUDA and ROCm. CI artifacts with charts.
**Depends on:** Nothing — fully independent
**Effort:** 1-2 weeks

### F7: LEGO Studio improvements
**Concept:** (a) Ship pre-built WASM or add viz/Makefile, (b) URL-shareable state, (c) more presets, (d) populate formula display, (e) add RegP to GUI builder
**Depends on:** Nothing
**Effort:** 1-2 weeks

### F8: Examples overhaul
**Concept:** Add examples README, fill missing puzzles (20, 30-32), fix puzzle 33 tensor core issue, add Flash Attention example, extend codegen backends beyond row-major
**Depends on:** Nothing
**Effort:** 1-2 weeks

### F9: Persistent kernel example
**Concept:** Work-queue kernel processing tiles in a loop. Demonstrates LEGO layout for tile-to-thread mapping. Hot pattern in LLM inference.
**Depends on:** Nothing
**Effort:** 3-5 days

---

## Verification System Extensions

### V1: Bijectivity on `lego.check`
**Concept:** Add `lego.check %layout {bijective}` variant that takes a `!lego.layout` operand (instead of `index`). Unifies the user interface — currently bijectivity uses a separate `lego-verify-bijectivity` pass that only works with GenP.
**Depends on:** Nothing
**Effort:** 3-5 days

### V2: Parallel Z3 invocations
**Concept:** For modules with many `lego.check` ops, parallelize Z3 calls using MLIR's ThreadPool. Currently verification is serial — one Z3 subprocess per check op.
**Depends on:** Thread safety audit of SMTBuilder
**Effort:** 1 week

---

## Research Directions (need more design thinking)

### F10: Warp specialization support
**Concept:** Layout ops/combinators for producer/consumer warp roles. Integration with Triton's warp-spec roadmap.
**Status:** Idea only — Triton's warp-spec API is still in development
**Effort:** Unknown — depends on Triton stabilization

### F11: Blackwell/SM100 layouts
**Concept:** TMA 2D descriptors, cluster-level layouts, distributed shared memory. Complete puzzle 34's cluster API.
**Status:** Idea only — needs Blackwell hardware access for testing
**Effort:** 2-3 weeks once hardware available

### F13: Sparse layout support
**Concept:** `lego.sparse` op family for CSR/COO/block-sparse index spaces. Integration with MLIR sparse_tensor dialect.
**Status:** Research direction — novel, no competitor does this well
**Effort:** Unknown — research project

### F14: Multi-GPU / distributed layouts
**Concept:** Promote comm_derive.py into `lego/distributed/` API. Symbolic communication derivation from layout algebra.
**Status:** Prototype exists (comm_derive.py, summa_e2e.py). Correctness needs validation before building APIs.
**Concerns:**
- `_classify_reduction` heuristic needs formalization
- Block sizes assume exact divisibility
- Stencil classifier doesn't handle wrap-around
- Only 6 test patterns, no numerical stress testing
**Levels:**
1. Validate correctness against known algorithms (SUMMA, Cannon)
2. Promote to `lego/distributed/` library
3. Add `compare_distributions()` — rank alternatives by symbolic volume
4. Codegen for torch.distributed / MPI / NCCL
5. Integration with LegoTensor and DTensor
**Effort:** 1-2 months for full path

---

## PyTorch Integration Redesign
**Branch:** `refactor/pytorch-integration` (mega branch, sub-PRs merge here)
**Spec:** `docs/superpowers/specs/2026-04-09-pytorch-integration-design.md`
**Status:** Design complete, implementation not started

| Phase | Description | Dependencies | Est. Effort |
|-------|-------------|-------------|-------------|
| Phase 0 | Delete old PyTorch code (torch_tensor.py, torch_ops.py, torch_layout.py, fx_backend.py, autotune.py, TiledView) | None | 1 day |
| Phase 1 | Layer 1: torch.library ops (lego::mm, lego::bmm), lego.rearrange(), Triton codegen | Phase 0 | 1-2 weeks |
| Phase 2 | Layer 2: LegoTensor with __torch_dispatch__, lego.annotate(), 4-tier op dispatch | Phase 1 | 1-2 weeks |
| Phase 3 | Layer 3: torch.compile backend, inductor extension (Path B + C), layout planner | Phase 2 | 2-3 weeks |

---

## Competitive Features to Track

| Feature | Competitor | LEGO Status |
|---------|-----------|-------------|
| Warp specialization | CuTe (native), Triton (in dev) | Not present |
| Auto-tuning | Triton (built-in), CK (built-in) | Broken (no CUDA sync) |
| Blackwell/SM100 | CuTe (native), Triton (in dev) | Not present |
| Intel GPU | Triton (expanding) | `lego-to-xevm` pipeline (PR #85) |
| Flash Attention | All competitors | No LEGO example |
| Persistent kernels | TensorRT-LLM, vLLM | No LEGO example |
| Sparse layouts | None do well | Research opportunity |
| Communication derivation | None | LEGO unique (prototype) |
| Browser IDE | None | LEGO unique (LEGO Studio) |
| SMT verification | None | LEGO unique (now unified via `lego.check`) |

---

## Notes for Agents

- **Branch strategy:** Feature branches merge into mega branches, mega branches merge into main
- **Never push** spec/plan files to the repository
- **Never use** internal labels (B1, A1, Month X, Task N) in commits, PRs, or code
- **Never add** Co-Authored-By lines in commits
- **Always use** Opus model for implementation subagents
- **Always run** `check-lego-all` via CMake before claiming tests pass
- **PyTorch tests** require MLIR build: `PYTHONPATH=build/python_packages/lego` and run from `build/python_packages/lego`
- **Verification system:** Use `lego.check` ops + `lego-lower` pipeline (not the old standalone passes, which are deleted)
