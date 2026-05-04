//===- LegoToLinalg.cpp - Raise affine scf.for loops to linalg.generic -===//
//
// Detects scf.for loops whose body's memref accesses are all affine in the IV
// (per ``tryBuildAffineExpr``) and rewrites them to ``linalg.generic`` so
// downstream upstream MLIR passes (``linalg::vectorize``, ``transform.tile``,
// dependence analysis) can consume them.
//
// Loops that fail the affine check pass through unchanged — the existing
// custom ``lego-vectorize`` pass handles them. This is the bridge that lets
// us reuse upstream vectorisation for the affine majority while keeping
// custom code only for genuinely-non-affine layouts (Z-Morton, irregular
// brick neighbour reads, AoSoA struct fields).
//
// Current scope (this commit):
//   - Single, perfectly-nested ``scf.for`` (1-D loops).
//   - Step == 1, lower bound 0 OR a loop-invariant value.
//   - Body contains memref.load / memref.store / arith ops only.
//   - All accesses must be 1-D memref reads/writes with affine index.
//
// Out of scope (follow-up):
//   - Multi-dimensional perfectly-nested loops (extend ``iterator_types``).
//   - Reductions (iter_args / loop-carried values → iterator_types =
//     ["reduction"]).
//   - Loops mixing affine and non-affine accesses (would need a hybrid
//     linalg.generic with explicit memref.load fallthrough — punt for now).
//
//===----------------------------------------------------------------------===//

#include "LegoAffineExtract.h"
#include "Lego/Passes.h"  // for createConvertLegoToLinalgPass declaration

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::lego;

namespace {

/// One affine memref access discovered in a loop body.
struct AccessInfo {
  Operation *op;       // memref::LoadOp or memref::StoreOp
  Value memref;        // the underlying memref Value
  AffineExpr indexExpr; // expression in (d0, s0..) form
  llvm::SmallVector<Value, 4> symbols; // values bound to symbols
  bool isWrite;
};

/// Walks the body of ``forOp`` and records every memref.load / memref.store
/// it finds. Returns nullopt if any access fails the affine test or if the
/// body contains structures we don't yet handle (nested loops, scf.if, etc.).
static std::optional<llvm::SmallVector<AccessInfo, 4>>
collectAffineAccesses(scf::ForOp forOp) {
  llvm::SmallVector<AccessInfo, 4> accesses;
  Value iv = forOp.getInductionVar();
  MLIRContext *ctx = forOp.getContext();

  // Reject non-trivial body structures up-front. We only handle linear bodies:
  // a sequence of arith / memref.load / memref.store / linalg.yield-equivalent
  // ops, terminated by an scf.yield with no operands (no iter_args yet).
  for (Operation &op : forOp.getBody()->without_terminator()) {
    // Reject any op that introduces nested control flow.
    if (op.getNumRegions() > 0)
      return std::nullopt;
  }

  // Reject loops with iter_args (reductions are a follow-up).
  if (!forOp.getRegionIterArgs().empty())
    return std::nullopt;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    Value memref;
    bool isWrite;
    Value addr;

    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (load.getIndices().size() != 1) return std::nullopt;
      memref = load.getMemRef();
      addr = load.getIndices().front();
      isWrite = false;
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (store.getIndices().size() != 1) return std::nullopt;
      memref = store.getMemRef();
      addr = store.getIndices().front();
      isWrite = true;
    } else {
      // Pure compute (arith) or other side-effect-free ops are OK; we copy
      // them verbatim into the linalg body.
      continue;
    }

    auto extracted = tryBuildAffineExpr(addr, iv, ctx);
    if (!extracted)
      return std::nullopt;

    // Every symbol must be defined outside the loop (we already check this
    // structurally inside tryBuildAffineExpr, but double-check for safety).
    for (Value sym : extracted->symbols) {
      if (forOp->isProperAncestor(sym.getDefiningOp() ? sym.getDefiningOp()
                                                     : nullptr))
        return std::nullopt;
    }

    AccessInfo info;
    info.op = &op;
    info.memref = memref;
    info.indexExpr = extracted->expr;
    info.symbols = std::move(extracted->symbols);
    info.isWrite = isWrite;
    accesses.push_back(std::move(info));
  }

  if (accesses.empty())
    return std::nullopt;

  return accesses;
}

/// Rewrite ``forOp`` to ``linalg.generic`` when all accesses are affine.
/// Returns success() iff the rewrite happened.
static LogicalResult tryConvertForToLinalgGeneric(scf::ForOp forOp,
                                                  RewriterBase &rewriter) {
  // Only convert step-1 loops with a constant zero lower bound (the simple
  // case). A more general implementation would normalise the loop first.
  auto lbConst = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepConst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lbConst || lbConst.value() != 0) return failure();
  if (!stepConst || stepConst.value() != 1) return failure();

  auto accessesOpt = collectAffineAccesses(forOp);
  if (!accessesOpt) return failure();
  auto &accesses = *accessesOpt;

  // Partition accesses into reads (inputs) and writes (outputs). A memref
  // that appears as both load and store becomes one input + one output
  // (linalg.generic permits the same memref in ins and outs).
  llvm::SmallVector<AccessInfo *, 4> reads;
  llvm::SmallVector<AccessInfo *, 4> writes;
  for (auto &acc : accesses) {
    if (acc.isWrite) writes.push_back(&acc);
    else             reads.push_back(&acc);
  }
  if (writes.empty()) return failure();  // pure-load loops aren't useful

  MLIRContext *ctx = forOp.getContext();

  // For each access, materialise the AffineExpr's symbols as a list of
  // operands suitable for the indexing map. linalg.generic indexing maps
  // are (d0..dN-1)[s0..sM-1] -> resultExpr; the symbols come from the maps'
  // metadata, not from operand lists. We need to compose each access's
  // symbol list into a single global symbol list for the linalg op.
  //
  // Simpler approach: compose each indexing map with explicit symbol values
  // by substituting them into the AffineExpr, leaving only d0. This works
  // when symbols are compile-time constants OR loop-invariant Values that
  // we capture as scalars in the linalg.generic body. For the first version,
  // we restrict to expressions with NO symbols (i.e. addr = c*i + const).
  // That's the most-common saxpy-/elementwise-style pattern.
  for (const auto &acc : accesses) {
    if (!acc.symbols.empty()) return failure();
  }

  // Build indexing_maps: one per (read_input ∪ write_output) operand, all
  // with the form (d0) -> (acc.indexExpr) which has no symbols by the check
  // above.
  llvm::SmallVector<AffineMap, 4> indexingMaps;
  llvm::SmallVector<Value, 4> insOperands;
  llvm::SmallVector<Value, 4> outsOperands;

  // Reads first, in order of appearance.
  for (auto *r : reads) {
    indexingMaps.push_back(AffineMap::get(/*dimCount=*/1, /*symCount=*/0,
                                          r->indexExpr, ctx));
    insOperands.push_back(r->memref);
  }
  for (auto *w : writes) {
    indexingMaps.push_back(AffineMap::get(/*dimCount=*/1, /*symCount=*/0,
                                          w->indexExpr, ctx));
    outsOperands.push_back(w->memref);
  }

  // Iteration domain: the loop's [0, ub) becomes a single "parallel" iter.
  llvm::SmallVector<utils::IteratorType, 1> iterTypes{
      utils::IteratorType::parallel};

  // Build the linalg.generic. Element types come from the memrefs.
  llvm::SmallVector<Type, 4> blockArgTypes;
  for (auto *r : reads)
    blockArgTypes.push_back(cast<MemRefType>(r->memref.getType()).getElementType());
  for (auto *w : writes)
    blockArgTypes.push_back(cast<MemRefType>(w->memref.getType()).getElementType());

  llvm::SmallVector<Location, 4> blockArgLocs(blockArgTypes.size(),
                                              forOp.getLoc());

  rewriter.setInsertionPoint(forOp);
  auto generic = linalg::GenericOp::create(
      rewriter, forOp.getLoc(), /*resultTypes=*/TypeRange{},
      /*inputs=*/insOperands, /*outputs=*/outsOperands,
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iterTypes,
      [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
        // Map original memref.load results to block args (in read order),
        // and remember the corresponding output block-arg index for each
        // store target.
        IRMapping bvm;
        unsigned readIdx = 0;
        for (auto *r : reads) {
          // Each load returned one Value — replace with the corresponding
          // block arg.
          bvm.map(cast<memref::LoadOp>(r->op).getResult(),
                  blockArgs[readIdx++]);
        }

        // Clone every body op except the loads, stores, and the terminator.
        // Loads are already replaced via the map; stores are converted into
        // linalg.yield of the value being stored.
        Value yieldVal;
        for (Operation &op : forOp.getBody()->without_terminator()) {
          if (isa<memref::LoadOp>(op))
            continue;
          if (auto store = dyn_cast<memref::StoreOp>(op)) {
            yieldVal = bvm.lookupOrDefault(store.getValueToStore());
            continue;
          }
          b.clone(op, bvm);
        }

        // Yield the stored value(s). For a single-write loop, that's just
        // the most-recent yieldVal.
        linalg::YieldOp::create(b, loc, ValueRange{yieldVal});
      });

  rewriter.eraseOp(forOp);
  (void)generic;
  return success();
}

struct ConvertLegoToLinalgPass
    : public PassWrapper<ConvertLegoToLinalgPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLegoToLinalgPass)

  StringRef getArgument() const override {
    return "convert-lego-to-linalg";
  }
  StringRef getDescription() const override {
    return "Raise scf.for loops with affine memref accesses to linalg.generic";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func.getContext());

    // Walk top-level scf.for loops in the function. We don't recurse into
    // nested scf.for because our current converter only handles single
    // loops — nested loops keep the outer scf.for around and only the
    // inner loop becomes linalg.generic (which is fine; subsequent passes
    // will tile it back if needed).
    llvm::SmallVector<scf::ForOp, 4> candidates;
    func.walk([&](scf::ForOp op) {
      // Skip loops that contain another scf.for inside their body — those
      // need multi-dim handling (follow-up).
      bool hasInnerFor = false;
      op.getBody()->walk([&](scf::ForOp inner) {
        if (inner != op) hasInnerFor = true;
      });
      if (!hasInnerFor) candidates.push_back(op);
    });

    for (scf::ForOp op : candidates) {
      // Try the conversion; ignore failure (loop just stays as scf.for).
      (void)tryConvertForToLinalgGeneric(op, rewriter);
    }
  }
};

}  // namespace

namespace mlir::lego {
std::unique_ptr<Pass> createConvertLegoToLinalgPass() {
  return std::make_unique<ConvertLegoToLinalgPass>();
}
}  // namespace mlir::lego
