#ifndef LEGO_PASSES_H
#define LEGO_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "Lego/LegoOps.h"
#include <memory>

namespace mlir {
namespace lego {

std::unique_ptr<Pass> createLegoToArithPass();
std::unique_ptr<Pass> createLegoNormalizationPass(bool skipTileBy = false);
std::unique_ptr<Pass> createLegoArithSimplificationPass();
std::unique_ptr<Pass> createLegoMaterializeAssumeBoundsPass(bool cleanup = false);
std::unique_ptr<Pass> createLegoGenerateBoundsChecksPass();
std::unique_ptr<Pass> createLegoExternalSMTVerifierPass();
std::unique_ptr<Pass> createLegoVerifyBijectivityPass();
std::unique_ptr<Pass> createLegoVerifyPass();
std::unique_ptr<Pass> createLegoStrengthReductionPass();
std::unique_ptr<Pass> createLegoVectorizePass();
std::unique_ptr<Pass> createLegoVectorizePass(llvm::StringRef target);
std::unique_ptr<Pass> createConvertLegoToLinalgPass();
std::unique_ptr<Pass> createConvertLegoToLinalgPass(bool vectorize);

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

#ifdef LEGO_HAS_NVPTX
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
      llvm::cl::init(3)};
  PassOptions::Option<std::string> format{
      *this, "format",
      llvm::cl::desc("Output format: fatbin, assembly, or binary"),
      llvm::cl::init("fatbin")};
};
#endif

/// Options for the lego-to-llvmspirv pipeline.
/// Produces LLVM dialect with SPIR-V calling conventions.
struct LegoToLLVMSPIRVPipelineOptions
    : public PassPipelineOptions<LegoToLLVMSPIRVPipelineOptions> {
  PassOptions::Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("Target chip (unused, for GPUTarget compatibility)"),
      llvm::cl::init("generic")};
  PassOptions::Option<std::string> format{
      *this, "format",
      llvm::cl::desc("Output format (unused, for GPUTarget compatibility)"),
      llvm::cl::init("assembly")};
};

#ifdef LEGO_HAS_AMDGPU
/// Options for the lego-to-rocdl pipeline.
struct LegoToROCDLPipelineOptions
    : public PassPipelineOptions<LegoToROCDLPipelineOptions> {
  PassOptions::Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("AMD GPU chip (e.g., gfx900, gfx90a, gfx1100)"),
      llvm::cl::init("gfx900")};
  PassOptions::Option<std::string> features{
      *this, "features",
      llvm::cl::desc("AMDGPU feature string"),
      llvm::cl::init("")};
  PassOptions::Option<int> optLevel{
      *this, "opt-level",
      llvm::cl::desc("ROCDL optimization level (0-3)"),
      llvm::cl::init(3)};
  PassOptions::Option<std::string> format{
      *this, "format",
      llvm::cl::desc("Output format: fatbin, assembly, or binary"),
      llvm::cl::init("fatbin")};
};
#endif

#ifdef LEGO_HAS_XEVM
/// Options for the lego-to-xevm pipeline.
struct LegoToXeVMPipelineOptions
    : public PassPipelineOptions<LegoToXeVMPipelineOptions> {
  PassOptions::Option<std::string> chip{
      *this, "chip",
      llvm::cl::desc("Intel GPU chip (e.g., bmg, pvc)"),
      llvm::cl::init("bmg")};
  PassOptions::Option<int> optLevel{
      *this, "opt-level",
      llvm::cl::desc("XeVM optimization level (0-3)"),
      llvm::cl::init(2)};
  PassOptions::Option<std::string> format{
      *this, "format",
      llvm::cl::desc("Output format: fatbin, assembly, or binary"),
      llvm::cl::init("fatbin")};
};
#endif


/// Options for the lego-to-x86-vector pipeline.
struct LegoToX86VectorPipelineOptions
    : public PassPipelineOptions<LegoToX86VectorPipelineOptions> {
  PassOptions::Option<std::string> cpu{
      *this, "cpu",
      llvm::cl::desc("Target CPU: skx|znver3|... (sets LLVM target features)"),
      llvm::cl::init("skx")};

  // reassociate-fp-reductions (default: true)
  //
  // Allows LLVM to reorder floating-point reductions across SIMD lanes, which
  // is semantically equivalent to GCC's -ffast-math reassoc flag.  This is
  // SAFE for kernels whose correctness checks use rtol/atol tolerance (the
  // typical case for numerical benchmarks).  Disable if the kernel requires
  // strict IEEE-754 reproducible reductions (e.g., compensated summation).
  //
  // Set to false via:  cpu_kernel(grid=..., fast_math=False)  (Python side)
  // or via the pipeline option:  lego-to-x86-vector{reassoc-fp=false}
  //
  // Reference: LLVM ConvertVectorToLLVMPass docs; GCC manual -ffast-math §3.10.
  PassOptions::Option<bool> reassocFP{
      *this, "reassoc-fp",
      llvm::cl::desc("Allow FP reduction reordering (fast-math reassoc); "
                     "default true; set false for strict IEEE-754 reproducibility"),
      llvm::cl::init(true)};

  // use-vector-alignment (default: false)
  //
  // When true, emit aligned load/store hints (64-byte for AVX-512).  This can
  // give ~30% additional speed-up when ALL input buffers are 64-byte aligned.
  // NumPy/ctypes allocations are only 16-byte aligned; enabling this for
  // unaligned buffers causes SIGSEGV on the first misaligned transfer.
  //
  // Safe default: false.  Users with aligned buffers (e.g., posix_memalign
  // at 64 bytes, or __attribute__((aligned(64))) C arrays) may set this true.
  PassOptions::Option<bool> useVecAlignment{
      *this, "use-vec-alignment",
      llvm::cl::desc("Emit aligned load/store hints (64-byte for AVX-512); "
                     "default false; only safe when ALL buffers are 64-byte aligned"),
      llvm::cl::init(false)};

  // use-linalg-vectorize (default: false; opt-in during the lego→linalg
  // refactor)
  //
  // When true, run convert-lego-to-linalg + upstream linalg::vectorize
  // BEFORE the custom lego-vectorize pass. The convert pass raises every
  // affine scf.for to linalg.generic; linalg::vectorize then turns those
  // into vector dialect ops directly. Any loop that wasn't affine
  // (Z-Morton et al) stays as scf.for and the custom lego-vectorize
  // handles it as before.
  //
  // Default false until the dashboard A/B confirms the linalg path
  // matches or beats the custom path on every affine candidate. Will
  // become the only path once the custom code is deleted.
  PassOptions::Option<bool> useLinalgVectorize{
      *this, "use-linalg-vectorize",
      llvm::cl::desc("Use convert-lego-to-linalg + upstream linalg::vectorize "
                     "for affine loops; falls through to custom lego-vectorize "
                     "for non-affine loops (Z-Morton et al). Default: true."),
      llvm::cl::init(true)};
};

/// Options for the lego-to-arm-neon pipeline.
struct LegoToArmNeonPipelineOptions
    : public PassPipelineOptions<LegoToArmNeonPipelineOptions> {
  PassOptions::Option<std::string> cpu{
      *this, "cpu",
      llvm::cl::desc("ARM CPU: cortex-a76|... (sets LLVM target features)"),
      llvm::cl::init("cortex-a76")};
};

/// Options for the lego-to-arm-sve pipeline (R15).
/// Emits fixed-width SVE-shaped vectors (vscale=1, 128-bit minimum).
/// On SVE hardware, the LLVM AArch64 backend promotes these to the full
/// runtime SVE vector length (+sve target feature).
/// For runtime execution: mlir-translate --mlir-to-llvmir |
///   llc -mtriple=aarch64-linux-gnu -mattr=+sve
struct LegoToArmSvePipelineOptions
    : public PassPipelineOptions<LegoToArmSvePipelineOptions> {
  PassOptions::Option<std::string> cpu{
      *this, "cpu",
      llvm::cl::desc("ARM CPU with SVE: neoverse-v1|neoverse-v2|... "
                     "(sets LLVM target features)"),
      llvm::cl::init("neoverse-v1")};
};

void registerLegoPipelines();
void buildLegoLowerPipeline(OpPassManager &pm);
void buildLegoToLLVMPipeline(OpPassManager &pm);
void buildLegoToX86VectorPipeline(OpPassManager &pm,
                                  const LegoToX86VectorPipelineOptions &opts);
void buildLegoToArmNeonPipeline(OpPassManager &pm,
                                const LegoToArmNeonPipelineOptions &opts);
void buildLegoToArmSvePipeline(OpPassManager &pm,
                               const LegoToArmSvePipelineOptions &opts);
void buildLegoToSPIRVPipeline(OpPassManager &pm,
                               const LegoToSPIRVPipelineOptions &options);
#ifdef LEGO_HAS_SPIRV
void buildLegoToLLVMSPIRVPipeline(OpPassManager &pm,
                                   const LegoToLLVMSPIRVPipelineOptions &options);
#endif
#ifdef LEGO_HAS_NVPTX
void buildLegoToNVVMPipeline(OpPassManager &pm,
                              const LegoToNVVMPipelineOptions &options);
#endif
#ifdef LEGO_HAS_AMDGPU
void buildLegoToROCDLPipeline(OpPassManager &pm,
                               const LegoToROCDLPipelineOptions &options);
#endif
#ifdef LEGO_HAS_XEVM
void buildLegoToXeVMPipeline(OpPassManager &pm,
                              const LegoToXeVMPipelineOptions &options);
#endif


// =========================================================================
// Shared GPU pipeline building blocks.
//
// Every GPU backend (NVVM, ROCDL, XeVM, …) follows the same
// three-phase pattern:
//
//   1. buildLegoGPUOutlinePipeline    — LEGO lower → canonicalize → outline
//   2. <backend-specific passes>      — set target attr + GPU→dialect
//   3. buildGPUToLLVMAndBinaryPipeline — host LLVM lowering → binary
//
// Adding a new backend only requires writing phase 2 (a SetTargetPass +
// one createConvertGpuOpsTo*() call) and registering the pipeline.
// =========================================================================

/// Phase 1: LEGO lower + canonicalize/CSE + GPU kernel outlining.
void buildLegoGPUOutlinePipeline(OpPassManager &pm);

/// Phase 1.5 helpers: lower gpu.all_reduce / gpu.subgroup_reduce.
/// Shared by all GPU backends — call between outlining and backend conversion.
void addGpuAllReduceLoweringPass(OpPassManager &pm);
void addGpuSubgroupReduceLoweringPass(OpPassManager &pm,
                                      unsigned subgroupSize = 32);

/// Phase 3a: host-side LLVM lowering only (no binary compilation).
void buildGPUHostLLVMPipeline(OpPassManager &pm);

/// Phase 3: host-side LLVM lowering + gpu-module-to-binary compilation.
void buildGPUToLLVMAndBinaryPipeline(OpPassManager &pm, StringRef format);

#define GEN_PASS_DECL_LEGOTOARITHPASS
#define GEN_PASS_DECL_LEGONORMALIZATIONPASS
#define GEN_PASS_DECL_LEGOVERIFYGENPCONSISTENCYPASS
#define GEN_PASS_DECL_LEGOARITHSIMPLIFICATIONPASS
#define GEN_PASS_DECL_LEGOMATERIALIZEASSUMEBOUNDSPASS
#define GEN_PASS_DECL_LEGOGENERATEBOUNDSCHECKSPASS
#define GEN_PASS_DECL_LEGOEXTERNALSMTVERIFIERPASS
#define GEN_PASS_DECL_LEGOVERIFYBIJECTIVITYPASS
#define GEN_PASS_DECL_LEGOVERIFYPASS
#define GEN_PASS_DECL_LEGOSTRENGTHREDUCTIONPASS
#define GEN_PASS_DECL_LEGOVECTORIZEPASS
#define GEN_PASS_DECL_CONVERTLEGOTOLINALGPASS
#define GEN_PASS_REGISTRATION
#include "Lego/Passes.h.inc"

} // namespace lego
} // namespace mlir

#endif // LEGO_PASSES_H
