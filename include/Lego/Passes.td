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
