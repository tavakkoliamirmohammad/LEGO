#ifndef LEGO_PASSES_H
#define LEGO_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
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
std::unique_ptr<Pass> createLegoNormalizationPass(bool skipTileBy = false);
std::unique_ptr<Pass> createLegoVerifyGenpConsistencyPass();
std::unique_ptr<Pass> createLegoArithSimplificationPass();
std::unique_ptr<Pass> createLegoMaterializeAssumeBoundsPass(bool cleanup = false);
std::unique_ptr<Pass> createLegoGenerateBoundsChecksPass();
std::unique_ptr<Pass> createLegoExternalSMTVerifierPass();
std::unique_ptr<Pass> createLegoVerifyBijectivityPass();
std::unique_ptr<Pass> createLegoVerifyCoalescingPass();
std::unique_ptr<Pass> createLegoVerifyBankConflictsPass();
std::unique_ptr<Pass> createLegoStrengthReductionPass();

/// Options for the lego-to-spirv pipeline.
struct LegoToSPIRVPipelineOptions
    : public PassPipelineOptions<LegoToSPIRVPipelineOptions> {
  PassOptions::Option<std::string> spirvVersion{
      *this, "spirv-version",
      llvm::cl::desc("SPIR-V version (1.0, 1.1, ..., 1.6)"),
      llvm::cl::init("1.5")};
  PassOptions::Option<std::string> clientAPI{
      *this, "client-api",
      llvm::cl::desc("Client API: vulkan or opencl"),
      llvm::cl::init("vulkan")};
};

/// Options for the lego-to-nvvm pipeline.
struct LegoToNVVMPipelineOptions
    : public PassPipelineOptions<LegoToNVVMPipelineOptions> {
  PassOptions::Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("CUDA compute capability (e.g., sm_50, sm_80, sm_90)"),
      llvm::cl::init("sm_70")};
  PassOptions::Option<std::string> features{
      *this, "features",
      llvm::cl::desc("PTX feature string (e.g., +ptx60, +ptx78)"),
      llvm::cl::init("+ptx60")};
  PassOptions::Option<int> optLevel{
      *this, "opt-level",
      llvm::cl::desc("NVVM optimization level (0-3)"),
      llvm::cl::init(2)};
  PassOptions::Option<std::string> format{
      *this, "format",
      llvm::cl::desc("Output format: fatbin, assembly, or binary"),
      llvm::cl::init("fatbin")};
};

void registerLegoPipelines();
void buildLegoLowerPipeline(OpPassManager &pm);
void buildLegoToLLVMPipeline(OpPassManager &pm);
void buildLegoToSPIRVPipeline(OpPassManager &pm,
                               const LegoToSPIRVPipelineOptions &options);
void buildLegoToNVVMPipeline(OpPassManager &pm,
                              const LegoToNVVMPipelineOptions &options);

#define GEN_PASS_DECL_LEGOTOARITHPASS
#define GEN_PASS_DECL_LEGONORMALIZATIONPASS
#define GEN_PASS_DECL_LEGOVERIFYGENPCONSISTENCYPASS
#define GEN_PASS_DECL_LEGOARITHSIMPLIFICATIONPASS
#define GEN_PASS_DECL_LEGOMATERIALIZEASSUMEBOUNDSPASS
#define GEN_PASS_DECL_LEGOGENERATEBOUNDSCHECKSPASS
#define GEN_PASS_DECL_LEGOEXTERNALSMTVERIFIERPASS
#define GEN_PASS_DECL_LEGOVERIFYBIJECTIVITYPASS
#define GEN_PASS_DECL_LEGOVERIFYCOALESCINGPASS
#define GEN_PASS_DECL_LEGOVERIFYBANKCONFLICTSPASS
#define GEN_PASS_DECL_LEGOSTRENGTHREDUCTIONPASS
#define GEN_PASS_REGISTRATION
#include "Lego/Passes.h.inc"

} // namespace lego
} // namespace mlir

#endif // LEGO_PASSES_H
