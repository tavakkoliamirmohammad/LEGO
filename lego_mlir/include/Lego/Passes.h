#ifndef LEGO_PASSES_H
#define LEGO_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace lego {

std::unique_ptr<Pass> createLegoToArithPass();
std::unique_ptr<Pass> createLegoDesugarPass();

#define GEN_PASS_DECL_LEGOTOARITHPASS
#define GEN_PASS_REGISTRATION
#include "Lego/Passes.h.inc"

} // namespace lego
} // namespace mlir

#endif // LEGO_PASSES_H
