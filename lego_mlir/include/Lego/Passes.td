#ifndef LEGO_PASSES_TD
#define LEGO_PASSES_TD

include "mlir/Pass/PassBase.td"

def LegoDesugarPass : Pass<"lego-desugar", "mlir::ModuleOp"> {
  let summary = "Desugar LEGO dialect syntactic sugars (like TileBy)";
  let description = [{
    This pass simplifies LEGO dialect operations by rewriting syntactic sugar ops
    (e.g., TileBy) into their primitive components (GroupBy, OrderBy, RegP).
  }];
  let constructor = "mlir::lego::createLegoDesugarPass()";
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
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect"
  ];
}

def LegoVerifyConsistencyPass : Pass<"lego-verify-consistency", "mlir::ModuleOp"> {
  let summary = "Verify the consistency between apply and inv regions in gen_p";
  let description = [{
    This pass analyzes lego.gen_p operations to ensure that the 'apply' and 'inv'
    regions are truly inverses of each other. It performs symbolic composition
    and simplification to detect inconsistencies.
  }];
  let constructor = "mlir::lego::createLegoVerifyConsistencyPass()";
  let dependentDialects = [
    "mlir::lego::LegoDialect",
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::math::MathDialect"
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

#endif // LEGO_PASSES_TD
