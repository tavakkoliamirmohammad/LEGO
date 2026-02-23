#ifndef LEGO_PASSES_H
#define LEGO_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "Lego/LegoOps.h"
#include <memory>

namespace mlir {
namespace lego {

std::unique_ptr<Pass> createLegoToArithPass();
std::unique_ptr<Pass> createLegoNormalizationPass();
std::unique_ptr<Pass> createLegoVerifyGenpConsistencyPass();
std::unique_ptr<Pass> createLegoArithSimplificationPass();
std::unique_ptr<Pass> createLegoGenerateBoundsChecksPass();
std::unique_ptr<Pass> createLegoExternalSMTVerifierPass();

void registerLegoPipelines();

#define GEN_PASS_DECL_LEGOTOARITHPASS
#define GEN_PASS_DECL_LEGONORMALIZATIONPASS
#define GEN_PASS_DECL_LEGOVERIFYGENPCONSISTENCYPASS
#define GEN_PASS_DECL_LEGOARITHSIMPLIFICATIONPASS
#define GEN_PASS_DECL_LEGOGENERATEBOUNDSCHECKSPASS
#define GEN_PASS_DECL_LEGOEXTERNALSMTVERIFIERPASS
#define GEN_PASS_REGISTRATION
#include "Lego/Passes.h.inc"

} // namespace lego
} // namespace mlir

#endif // LEGO_PASSES_H
