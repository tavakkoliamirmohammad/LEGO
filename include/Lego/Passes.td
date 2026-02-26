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

def LegoVerifyGenpConsistencyPass : Pass<"lego-verify-genp-consistency", "mlir::ModuleOp"> {
  let summary = "Verify the consistency between apply and inv regions in gen_p";
  let description = [{
    This pass analyzes lego.gen_p operations to ensure that the 'apply' and 'inv'
    regions are truly inverses of each other. It performs symbolic composition
    and simplification to detect inconsistencies.
  }];
  let constructor = "mlir::lego::createLegoVerifyGenpConsistencyPass()";
  let dependentDialects = [
    "mlir::lego::LegoDialect",
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect"
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

def LegoVerifyCoalescingPass : Pass<"lego-verify-coalescing", "mlir::ModuleOp"> {
  let summary = "Verify that layouts produce coalesced memory access patterns";
  let description = [{
    This pass verifies that for a warp of 32 consecutive threads, the layout
    transformation produces memory addresses within a 128-byte cache line
    (coalesced access). Uses Z3 to verify the property.
  }];
  let constructor = "mlir::lego::createLegoVerifyCoalescingPass()";
}

def LegoVerifyBankConflictsPass : Pass<"lego-verify-bank-conflicts", "mlir::ModuleOp"> {
  let summary = "Verify freedom from shared memory bank conflicts";
  let description = [{
    This pass verifies that for a warp of threads accessing shared memory,
    no two threads access different addresses in the same bank.
    By default, verifies 32 threads, 32 banks, and 4-byte elements.
    Uses Z3 to verify conflict-freedom.
  }];
  let constructor = "mlir::lego::createLegoVerifyBankConflictsPass()";
  let options = [
    Option<"warpSize", "warp-size", "int", "32",
           "Number of threads in a warp">,
    Option<"numBanks", "num-banks", "int", "32",
           "Number of shared memory banks">,
    Option<"baseThreadVar", "base-thread-var", "std::string", "\"base_thread\"",
           "Name of the base thread variable expected in the layout">,
    Option<"elementSize", "element-size", "int", "4",
           "Size of each element in bytes (used to compute bank indices)">
  ];
}

#endif // LEGO_PASSES_TD
