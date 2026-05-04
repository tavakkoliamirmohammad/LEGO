#ifndef LEGO_PASSES_TD
#define LEGO_PASSES_TD

include "mlir/Pass/PassBase.td"

def LegoNormalizationPass : Pass<"lego-normalization", "mlir::ModuleOp"> {
  let summary = "Normalize LEGO dialect syntactic sugars (like TileBy)";
  let description = [{
    This pass simplifies LEGO dialect operations by rewriting syntactic sugar ops
    (e.g., TileBy) into their primitive components (GroupBy, OrderBy, RegP).
  }];
  let constructor = "mlir::lego::createLegoNormalizationPass()";
  let dependentDialects = ["mlir::lego::LegoDialect"];
}

def LegoToArithPass : Pass<"lego-to-arith", "mlir::ModuleOp"> {
  let summary = "Lower LEGO dialect to Arith/Affine";
  let description = [{
    This pass lowers LEGO dialect ops (apply, apply_inverse) to standard arith and affine operations.
  }];
  let constructor = "mlir::lego::createLegoToArithPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def LegoArithSimplificationPass : Pass<"lego-arith-simplification", "mlir::ModuleOp"> {
  let summary = "Simplify arithmetic operations using LEGO-specific algebraic identities";
  let description = [{
    This pass implements algebraic simplifications that standard Arith canonicalizations misses,
    such as (q * d + r) % d -> r % d.
  }];
  let constructor = "mlir::lego::createLegoArithSimplificationPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def LegoMaterializeAssumeBoundsPass : Pass<"lego-materialize-assume-bounds", "mlir::ModuleOp"> {
  let summary = "Convert lego.assume upper-bound constraints into arith.remui for range analysis";
  let description = [{
    For each `lego.assume(arith.cmpi slt, %x, %d)`, inserts
    `%x_bounded = arith.remui %x, %d` and rewrites downstream uses of %x
    to use %x_bounded.  This establishes the range [0, d-1] so that
    --int-range-optimizations can fold divui/remui to constants.

    With --cleanup, removes remui(block_arg, d) ops that were previously
    inserted, replacing them with the original block argument (since the
    user-declared bounds guarantee x < d, making remui identity).
  }];
  let constructor = "mlir::lego::createLegoMaterializeAssumeBoundsPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::lego::LegoDialect"];
  let options = [
    Option<"cleanup", "cleanup", "bool", /*default=*/"false",
           "Remove remui(block_arg, d) inserted by a previous materialization">
  ];
}

def LegoGenerateBoundsChecksPass : Pass<"lego-generate-bounds-checks", "mlir::ModuleOp"> {
  let summary = "Insert bounds check assertions for lego.apply and lego.apply_inverse ops";
  let description = [{
    Finds `lego.apply` and `lego.apply_inverse` ops and inserts `lego.assert_apply_bounds`
    and `lego.assert_inv_bounds` before them for verification.
  }];
  let constructor = "mlir::lego::createLegoGenerateBoundsChecksPass()";
  let dependentDialects = ["mlir::lego::LegoDialect"];
}

def LegoExternalSMTVerifierPass : Pass<"lego-external-smt-verifier", "mlir::ModuleOp"> {
  let summary = "Verify layout bounds using an external SMT solver (Z3)";
  let description = "Extracts arithmetic slices for lego.assert_* ops and runs external z3.";
  let constructor = "mlir::lego::createLegoExternalSMTVerifierPass()";
}

def LegoVerifyBijectivityPass : Pass<"lego-verify-bijectivity", "mlir::ModuleOp"> {
  let summary = "Verify that layouts are bijective";
  let description = [{
    This pass verifies that for each gen_p layout, the apply and inv regions
    are true inverses: ∀x. inv(apply(x)) = x and ∀y. apply(inv(y)) = y.
    Uses Z3 to check these properties formally.
  }];
  let constructor = "mlir::lego::createLegoVerifyBijectivityPass()";
}

def LegoVerifyPass : Pass<"lego-verify", "mlir::ModuleOp"> {
  let summary = "Verify memory access properties marked by lego.check ops";
  let description = [{
    Finds all lego.check ops in the module, encodes their operand's arith
    def-use chain to SMT, verifies the requested properties (coalescing,
    bank-conflict freedom) with Z3, emits diagnostics, and erases the ops.

    This pass should run after arith optimization (canonicalize + CSE) so
    that Z3 gets the smallest possible formula. It runs as part of the
    lego-lower pipeline, before strength reduction.
  }];
  let constructor = "mlir::lego::createLegoVerifyPass()";
}

def LegoStrengthReductionPass : Pass<"lego-strength-reduction", "mlir::ModuleOp"> {
  let summary = "Convert power-of-2 muli/divui/remui to shift/mask operations";
  let description = [{
    This pass converts:
      muli(x, 2^k)  → shli(x, k)
      divui(x, 2^k) → shrui(x, k)
      remui(x, 2^k) → andi(x, 2^k - 1)

    Runs after the main algebraic simplification to avoid interfering
    with div/rem pattern matchers in the fixed-point loop.
  }];
  let constructor = "mlir::lego::createLegoStrengthReductionPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def LegoVectorizeCompactPass
    : Pass<"lego-vectorize-compact", "mlir::func::FuncOp"> {
  let summary = "Recognize compaction loops and emit vector.compressstore";
  let description = [{
    Recognizes the canonical stream-filter / compaction pattern that
    ``clang -O3`` scalarises:

        k = 0
        for i in 0..N:
            if cond[i] > threshold:
                B[k] = A[i]
                k = k + 1

    Detected as ``scf.for`` with one ``index``-typed iter_arg whose body is
    ``load + cmp + scf.if(p) { store + add 1 + yield } else { yield }``.
    Rewrites to a strip-mined vector loop emitting:

      * ``vector.transfer_read``  (data + condition)
      * ``arith.cmpf`` to build the lane mask
      * ``vector.compressstore`` (the lane-selecting write — what clang
        misses; lowers to AVX-512 ``vpcompressps`` or ARM SVE compact+st1)
      * ``math.ctpop`` of the bitcast mask to advance the iter_arg

    Uses upstream MLIR vector ops only — no architecture-specific
    intrinsics.  Falls back to the original scalar body for the
    ``(N mod L)`` tail.

    Runs *before* ``lego-vectorize`` so the main vectoriser never sees
    the rewritten loop.
  }];
  let constructor = "mlir::lego::createLegoVectorizeCompactPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
    "mlir::math::MathDialect",
  ];
  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"avx512\"",
           "Target backend (controls vector lane width)">,
  ];
}

def LegoVectorizeScatterAddPass
    : Pass<"lego-vectorize-scatter-add", "mlir::func::FuncOp"> {
  let summary = "Recognize scatter-add / histogram loops and emit "
                "vector.gather + vector.scatter with conflict serialization";
  let description = [{
    Recognizes the canonical scatter-add / histogram pattern that
    ``clang -O3`` scalarises:

        for i in 0..N:
            b = bin[i]
            count[b] = count[b] + delta(i)

    where ``delta(i)`` may be a unit-stride load (``A[i]``) or a
    loop-invariant scalar.  The data-dependent index ``b`` defeats
    ``clang``'s auto-vec because it cannot prove conflict freedom.

    Detected as ``scf.for`` (no iter_args, step=1) whose body is
    ``load(bin[i]) + index_cast + load(count[b]) + addf + store(count[b])``.

    Rewrites to a strip-mined vector loop emitting:

      * ``vector.transfer_read``  (bin[i:i+L])
      * ``L-1`` ``vector.shuffle`` + ``arith.cmpi`` + ``vector.reduction`` to
        detect any pair of conflicting lanes (portable conflict detection)
      * ``scf.if`` chooses between:
          - scalar fallback (if any conflicts)
          - ``vector.gather`` + ``arith.addf`` + ``vector.scatter`` (fast path)

    Uses upstream MLIR vector ops only — no architecture-specific
    intrinsics.  Falls back to the original scalar body for the
    ``(N mod L)`` tail.

    Runs *before* ``lego-vectorize`` so the main vectoriser never sees
    the rewritten loop.
  }];
  let constructor = "mlir::lego::createLegoVectorizeScatterAddPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"avx512\"",
           "Target backend (controls vector lane width)">,
  ];
}

def LegoVectorizeArgminPass
    : Pass<"lego-vectorize-argmin", "mlir::func::FuncOp"> {
  let summary = "Recognize argmin / argmax loops and emit a vector "
                "min/max + per-lane index-select reduction";
  let description = [{
    Recognizes the canonical argmin / argmax pattern that
    ``clang -O3`` scalarises (paired iter_args defeat its heuristics):

        m  = +inf
        mi = 0
        for i in 0..N:
            if A[i] < m:
                m  = A[i]
                mi = i

    Detected as ``scf.for`` with two iter_args ``(T, index)``, body
    ``load(A[i]) + cmpf <pred> + scf.if -> (T, index) { yield A[i], i }
    else { yield m, mi }``.

    Rewrites to a strip-mined vector loop with vector accumulators and
    a final reduction:

      * vector accumulator per lane: ``acc_min : vector<L x T>``,
        ``acc_idx : vector<L x index>``
      * inside loop: ``arith.minimumf`` + ``arith.select`` (no scalar
        branches)
      * after loop: ``vector.reduction <minimumf>`` then locate the
        smallest lane index where the reduced min is reached
      * scalar tail loop for ``(N mod L)`` iterations seeded from the
        reduced (val, idx)

    Predicates supported: ``olt`` (argmin) and ``ogt`` (argmax) — both
    map directly to ``arith.minimumf`` / ``arith.maximumf`` and the
    matching ``vector.reduction`` kinds.

    Uses upstream MLIR vector ops only — no architecture-specific
    intrinsics.  Runs *before* ``lego-vectorize`` so the main
    vectoriser never sees the rewritten loop.
  }];
  let constructor = "mlir::lego::createLegoVectorizeArgminPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"avx512\"",
           "Target backend (controls vector lane width)">,
  ];
}

def LegoVectorizeScanPass
    : Pass<"lego-vectorize-scan", "mlir::func::FuncOp"> {
  let summary = "Recognize inclusive prefix-scan loops and emit a "
                "Hillis-Steele in-vector scan with scalar carry";
  let description = [{
    Recognizes the canonical inclusive prefix-scan / cumulative-sum
    pattern that ``clang -O3`` scalarises (loop-carried scalar acc
    defeats vectorisation):

        acc = 0
        for i in 0..N:
            acc = acc + A[i]
            B[i] = acc

    Detected as ``scf.for`` with one float iter_arg, body
    ``load(A[i]) + addf(acc, A[i]) + store(addf, B[i]) + yield(addf)``.

    Rewrites to a strip-mined vector loop:
      * ``vector.transfer_read``  (A[i:i+L])
      * Hillis-Steele in-vector prefix sum: log2(L) stages of
        ``vector.shuffle`` + ``arith.addf``
      * ``arith.addf`` with broadcast of the running scalar carry
      * ``vector.transfer_write`` (B[i:i+L])
      * ``vector.extract`` of the last lane → new scalar carry
      * Tail loop preserves the original scalar body for ``(N mod L)``.

    Combine op restricted to ``arith.addf`` for v1 (``mulf`` /
    ``maximumf`` / ``minimumf`` are trivial extensions).  Uses upstream
    MLIR vector ops only — no architecture-specific intrinsics.
  }];
  let constructor = "mlir::lego::createLegoVectorizeScanPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"avx512\"",
           "Target backend (controls vector lane width)">,
  ];
}

def LegoVectorizeFilteredReducePass
    : Pass<"lego-vectorize-filtered-reduce", "mlir::func::FuncOp"> {
  let summary = "Recognize filtered (predicated) reductions and emit a "
                "vector accumulator + per-chunk masked select";
  let description = [{
    Recognises the canonical filtered-reduction pattern that
    ``clang -O3`` scalarises:

        acc = identity
        for i in 0..N:
            if cond[i] <pred> threshold:
                acc = combine(acc, A[i])

    where ``combine`` is one of ``arith.addf`` / ``mulf`` /
    ``maximumf`` / ``minimumf``.  Detected as ``scf.for`` with one float
    iter_arg and body ``cmpf + scf.if -> T { combine; yield } else { yield acc }``.

    Rewrites to a strip-mined vector loop with a vector accumulator:
    per chunk, compute ``mask = cmpf <pred>, v_cond, threshold`` and
    ``contrib = select(mask, v_a, identity)`` — non-passing lanes get the
    combine's identity element so they don't perturb the accumulator.
    A single ``vector.reduction`` collapses to scalar after the loop;
    the original initAcc is folded back in.

    Uses upstream MLIR vector ops only — no architecture-specific
    intrinsics.
  }];
  let constructor = "mlir::lego::createLegoVectorizeFilteredReducePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
  ];
  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"avx512\"",
           "Target backend (controls vector lane width)">,
  ];
}

def LegoVectorizeRLEPass
    : Pass<"lego-vectorize-rle", "mlir::func::FuncOp"> {
  let summary = "Recognize RLE / edge-detect compaction loops and emit "
                "vector.compressstore with a shuffled-in carry";
  let description = [{
    Recognises the canonical run-length-encoding / edge-detection
    pattern that ``clang -O3`` scalarises:

        k = 0; prev = sentinel
        for i in 0..N: if A[i] != prev: out[k] = A[i]; prev = A[i]; k++

    Detected as ``scf.for`` with two iter_args ``(index, T)`` and
    body ``cmpf one (or une) + scf.if -> (index, T)``.

    Rewrites to a strip-mined vector loop:
      * vector.transfer_read of A[i:i+L]
      * vector.shuffle (prev_vec, v) [L-1, L+0, ..., L+(L-2)] — pulls
        prev_vec[L-1] into lane 0 and v[0..L-2] into lanes 1..L-1
      * arith.cmpf one for the lane mask
      * vector.compressstore out[k], mask, v
      * math.ctpop on the bitcast mask to advance k
      * yield (k_new, v) — v's last lane becomes next chunk's prev
      * Tail loop: scalar body for (N mod L) iterations.
  }];
  let constructor = "mlir::lego::createLegoVectorizeRLEPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
    "mlir::math::MathDialect",
  ];
  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"avx512\"",
           "Target backend (controls vector lane width)">,
  ];
}

def LegoVectorizePass : Pass<"lego-vectorize", "mlir::func::FuncOp"> {
  let summary = "Layout-agnostic vectorization for LEGO via symbolic stride solve";
  let description = [{
    Lowers loops over Lego-derived arith address expressions to MLIR vector
    dialect ops by symbolic stride analysis. Layout-agnostic: operates on
    post-LegoToArith IR (arith + memref + scf).

    The `target` option controls vector register width:
      avx512 — 64-byte ZMM registers (8xf64, 16xf32)
      avx2   — 32-byte YMM registers (4xf64, 8xf32)
      neon   — 16-byte NEON registers (2xf64, 4xf32)
      sve    — 16-byte minimum (vscale=1); LLVM promotes to full SVE width
               on hardware with +sve target features (R15).
    All vectorization decisions use getRegisterLanesForType(target, elemBytes)
    so NO AVX-specific lane widths are hardcoded.
  }];
  let constructor = "mlir::lego::createLegoVectorizePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect"
  ];
  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"avx512\"",
           "Target backend: avx512 | avx2 | neon | sve (controls vector register lane width)">,
    Option<"enableSmtDepAnalysis", "enable-smt-dep-analysis", "bool",
           /*default=*/"false",
           "Opt-in: use Z3 SMT solver to prove loop-carried dep independence "
           "for conservative cases. Default off — SMT invocation is slow. "
           "When enabled, loops where computeMinDepDistance returns 1 due to "
           "differing invariant sets or non-affine addresses are re-analyzed "
           "with Z3. If Z3 proves UNSAT (no dep), the loop is vectorized.">,
  ];
}

def ConvertLegoToLinalgPass : Pass<"convert-lego-to-linalg", "mlir::func::FuncOp"> {
  let summary = "Raise affine scf.for loops to linalg.generic";
  let description = [{
    Detects ``scf.for`` loops whose memref accesses are all affine in the IV
    (per the ``tryBuildAffineExpr`` walker) and rewrites them to
    ``linalg.generic`` so downstream upstream MLIR passes
    (``linalg::vectorize``, ``transform.tile``, dependence analysis) can
    consume them.

    Loops that fail the affine check pass through unchanged — the existing
    custom ``lego-vectorize`` pass handles them. This is the bridge that lets
    LEGO reuse upstream vectorisation for the affine majority of layouts
    while keeping custom code only for genuinely non-affine accesses
    (Z-Morton, irregular brick neighbour reads, AoSoA struct fields).
  }];
  let constructor = "mlir::lego::createConvertLegoToLinalgPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // LEGO_PASSES_TD
