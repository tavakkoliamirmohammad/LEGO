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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

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

/// Multi-IV variant: collect accesses inside ``innermost`` body, building
/// AffineExprs over the full ``ivs`` list (ivs[0]=outermost..ivs[N-1]=innermost).
static std::optional<llvm::SmallVector<AccessInfo, 4>>
collectAffineAccessesMultiIv(scf::ForOp innermost, ValueRange ivs);

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

    // Allowed forms:
    //   - identity:        d0
    //   - non-neg offset:  d0 + c  (c >= 0)  — for inputs ONLY
    //
    // Writes are restricted to identity (``d0``) so that the output
    // subview from linalg::tile doesn't go OOB at the last tile.
    // Non-affine (floordiv/mod), negative constants, symbols all fall
    // through to the custom lego-vectorize.
    auto isAcceptable = [&](AffineExpr e, bool forWrite) -> bool {
      if (!e) return false;
      if (auto dim = dyn_cast<AffineDimExpr>(e)) return dim.getPosition() == 0;
      if (forWrite) return false;  // writes must be identity
      // Inputs may have form ``d0 + c`` with c >= 0.
      auto bin = dyn_cast<AffineBinaryOpExpr>(e);
      if (!bin || bin.getKind() != AffineExprKind::Add) return false;
      auto dim = dyn_cast<AffineDimExpr>(bin.getLHS());
      auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS());
      if (!dim) {
        // Try the swapped order.
        dim = dyn_cast<AffineDimExpr>(bin.getRHS());
        cst = dyn_cast<AffineConstantExpr>(bin.getLHS());
      }
      if (!dim || !cst) return false;
      if (dim.getPosition() != 0) return false;
      return cst.getValue() >= 0;
    };
    if (!isAcceptable(extracted->expr, /*forWrite=*/isWrite))
      return std::nullopt;

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

/// Multi-IV access collection. Same shape as the 1-D version but uses every
/// IV in ``ivs`` as a distinct AffineDim. Has NO restriction on offset shape
/// — that's enforced per-access by the multi-D conversion based on whether
/// linalg can express it. Rejects any access that ``tryBuildAffineExpr``
/// can't represent (bitwise non-power-of-2, ``i*j`` of two IVs, etc.).
static std::optional<llvm::SmallVector<AccessInfo, 4>>
collectAffineAccessesMultiIv(scf::ForOp innermost, ValueRange ivs) {
  llvm::SmallVector<AccessInfo, 4> accesses;
  MLIRContext *ctx = innermost.getContext();

  // Reject control flow inside the innermost body.
  for (Operation &op : innermost.getBody()->without_terminator()) {
    if (op.getNumRegions() > 0) return std::nullopt;
  }

  for (Operation &op : innermost.getBody()->without_terminator()) {
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
      continue;
    }

    auto extracted = tryBuildAffineExpr(addr, ivs, ctx);
    if (!extracted) return std::nullopt;

    AccessInfo info;
    info.op = &op;
    info.memref = memref;
    info.indexExpr = extracted->expr;
    info.symbols = std::move(extracted->symbols);
    info.isWrite = isWrite;
    accesses.push_back(std::move(info));
  }
  if (accesses.empty()) return std::nullopt;
  return accesses;
}

/// SIMD lane multiplier × ILP unroll factor targeted by the linalg path.
/// Matches lego-vectorize's choice (kILPFactor=4 × 16-lane AVX-512). Loops
/// whose iteration count isn't a multiple of this number aren't converted —
/// they fall through to the existing custom lego-vectorize, which handles
/// tail/mask emission.
///
/// Using SIMD*ILP (=64 for AVX-512 f32) instead of just SIMD (=16) gives
/// the LLVM backend room to produce 4 independent vmovups+vfmadd chains per
/// iteration, restoring the ILP that lego-vectorize gets via its unroll-by-4
/// heuristic.
static constexpr int64_t kLinalgPathSimdWidth = 64;

// =========================================================================
// Multi-dimensional perfect-nest support.
//
// In addition to single-loop 1-D conversion, the pass detects perfect nests
// of ``scf.for`` (outer..innermost). For a perfect nest where every IV has
// constant bounds [0, ub), step=1, and every memref access in the innermost
// body extracts to an AffineExpr in d0..d(N-1), we emit a single multi-dim
// ``linalg.generic`` with one iterator type per dim.
//
// Iterator types: "reduction" for any dim that no WRITE access depends on
// (the canonical gemm pattern: C[i,j] = sum_k A[i,k] * B[k,j] — k is the
// reduction). All other dims are "parallel".
// =========================================================================

struct PerfectNestInfo {
  llvm::SmallVector<scf::ForOp, 4> loops;  // outer to inner
  llvm::SmallVector<Value, 4> ivs;          // ivs[0] = outermost
  llvm::SmallVector<int64_t, 4> ubs;        // static upper bounds
};

/// Walks down from ``outer`` to find the deepest perfectly-nested chain.
/// A nest is "perfect" iff each non-innermost loop's body contains only:
///   - one inner ``scf.for`` (or none — making this loop the innermost),
///   - any number of ``arith.constant`` ops (induction-arithmetic constants).
/// Each loop in the chain must have constant lb=0, step=1, ub=constant.
/// Returns nullopt for any imperfect nest.
static std::optional<PerfectNestInfo> detectPerfectNest(scf::ForOp outer) {
  PerfectNestInfo nest;
  scf::ForOp cur = outer;
  while (true) {
    auto lb = cur.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto step = cur.getStep().getDefiningOp<arith::ConstantIndexOp>();
    auto ub = cur.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    if (!lb || lb.value() != 0) return std::nullopt;
    if (!step || step.value() != 1) return std::nullopt;
    if (!ub) return std::nullopt;
    if (!cur.getRegionIterArgs().empty()) return std::nullopt;
    nest.loops.push_back(cur);
    nest.ivs.push_back(cur.getInductionVar());
    nest.ubs.push_back(ub.value());

    // Look for an inner scf.for. The loop body must have only constants and
    // possibly one inner scf.for; otherwise the nest isn't perfect.
    scf::ForOp innerFor;
    bool sawNonConstNonFor = false;
    for (Operation &op : cur.getBody()->without_terminator()) {
      if (auto inner = dyn_cast<scf::ForOp>(op)) {
        if (innerFor) { sawNonConstNonFor = true; break; }
        innerFor = inner;
        continue;
      }
      if (isa<arith::ConstantOp, arith::ConstantIndexOp>(op)) continue;
      // Anything else: this loop has real body work, can't go deeper.
      sawNonConstNonFor = true;
      break;
    }
    if (sawNonConstNonFor) {
      // ``cur`` is the innermost — it has actual body ops.
      break;
    }
    if (!innerFor) {
      // ``cur`` has empty body (no inner loop, no other ops). Degenerate.
      return std::nullopt;
    }
    cur = innerFor;
  }
  return nest;
}

/// Rewrite ``forOp`` to ``linalg.generic`` when all accesses are affine and
/// the iteration count is a multiple of ``kLinalgPathSimdWidth``.
/// Returns success() iff the rewrite happened.
static LogicalResult tryConvertForToLinalgGeneric(scf::ForOp forOp,
                                                  RewriterBase &rewriter) {
  // Only convert step-1 loops with a constant zero lower bound and a
  // constant upper bound divisible by the SIMD width. These constraints
  // ensure the subsequent linalg::tile + linalg::vectorize chain produces
  // statically-shaped vector ops without tail handling. Loops outside this
  // shape fall through to the custom lego-vectorize.
  auto lbConst = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepConst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  auto ubConst = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  if (!lbConst || lbConst.value() != 0) return failure();
  if (!stepConst || stepConst.value() != 1) return failure();
  if (!ubConst) return failure();
  if (ubConst.value() % kLinalgPathSimdWidth != 0) return failure();
  if (ubConst.value() < kLinalgPathSimdWidth) return failure();

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
  if (writes.size() > 1) return failure();  // multi-write needs more body work

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
        // Map original memref.load results to block args (in read order).
        IRMapping bvm;
        unsigned readIdx = 0;
        for (auto *r : reads) {
          bvm.map(cast<memref::LoadOp>(r->op).getResult(),
                  blockArgs[readIdx++]);
        }

        // The body of a linalg.generic is computational only — address
        // arithmetic that drove the original memref.load/store ops is
        // encoded in the indexing maps and must NOT be cloned (those ops
        // reference the now-out-of-scope outer IV). Compute the use-def
        // cone of the value that gets stored, restricted to ops INSIDE
        // forOp's body, and clone only that cone.
        Value yieldVal;
        memref::StoreOp theStore;
        for (Operation &op : forOp.getBody()->without_terminator()) {
          if (auto store = dyn_cast<memref::StoreOp>(op)) {
            theStore = store;
            yieldVal = store.getValueToStore();
            break;
          }
        }
        assert(theStore && "writes.empty() check should have caught this");

        // Backward closure: walk from yieldVal back through arith ops,
        // collecting any op defined inside forOp's body that isn't already
        // mapped (loads are mapped; constants and outer-scope values are
        // referenced as-is).
        SmallVector<Operation *> toClone;
        DenseSet<Operation *> visited;
        std::function<void(Value)> walk = [&](Value v) {
          if (bvm.contains(v)) return;
          Operation *def = v.getDefiningOp();
          if (!def) return;  // block arg, leave as-is
          if (def->getBlock() != forOp.getBody()) return;  // outside the loop
          if (!visited.insert(def).second) return;
          for (Value operand : def->getOperands()) walk(operand);
          toClone.push_back(def);
        };
        walk(yieldVal);

        // Clone in topological order (toClone is post-order).
        for (Operation *def : toClone) b.clone(*def, bvm);

        Value mappedYield = bvm.lookupOrDefault(yieldVal);
        linalg::YieldOp::create(b, loc, ValueRange{mappedYield});
      });

  rewriter.eraseOp(forOp);
  (void)generic;
  return success();
}

// =========================================================================
// Multi-dimensional perfect nest → linalg.generic
// =========================================================================

/// Attempt to convert a perfect nest into a single multi-dim
/// ``linalg.generic``. Returns ``success()`` iff the rewrite occurred. On
/// failure, the IR is unchanged.
///
/// Conservative scope (this commit):
///   - 2-D and 3-D perfect nests
///   - Every parallel iteration dim must be divisible by ``kLinalgPathSimdWidth``
///   - Every reduction dim must have a static bound (any size)
///   - Every memref access has ``AffineExpr`` of form
///         `Σ d_i * c_i + Σ s_j * c_j + c_const`
///     (i.e. the d0 coefficient check generalised: linear in each IV with
///     compile-time constant coefficient and non-negative offsets)
///   - Single write memref OR a memref that is BOTH read and written
///     (in-place reduction pattern as in gemm: ``C[i,j] += ...``)
///   - No symbols (loop-invariant offsets) — those need separate subview
///     materialisation; falls through to the custom path
///
/// Anything outside this scope falls through to either the 1-D conversion
/// or the custom ``lego-vectorize`` path.
static LogicalResult
tryConvertPerfectNestToLinalg(PerfectNestInfo &nest, RewriterBase &rewriter,
                              bool runVectorize) {
  if (nest.loops.size() < 2 || nest.loops.size() > 3) return failure();

  scf::ForOp innermost = nest.loops.back();
  MLIRContext *ctx = innermost.getContext();
  Location loc = nest.loops.front().getLoc();
  unsigned numDims = nest.ivs.size();

  auto accessesOpt = collectAffineAccessesMultiIv(innermost, nest.ivs);
  if (!accessesOpt) return failure();
  auto &accesses = *accessesOpt;

  // No symbols allowed in this scope — would need subview materialisation.
  for (const auto &acc : accesses) {
    if (!acc.symbols.empty()) return failure();
  }

  // Reject expressions that contain anything other than additions of
  // ``d_i * c`` / ``c_const`` / ``d_i`` terms with non-negative resulting
  // offsets per dim. The simpler way: walk the expression and check each
  // sub-expression's structure. We allow:
  //   - AffineDimExpr
  //   - AffineConstantExpr (>= 0 only)
  //   - Add(AffineExpr, AffineExpr) (operands recursively allowed)
  //   - Mul(AffineDim, AffineConstantExpr ≥ 0) or symmetric
  // This rules out floordiv/mod, negative constants, and d_i * d_j.
  std::function<bool(AffineExpr)> isShape = [&](AffineExpr e) -> bool {
    if (!e) return false;
    if (auto c = dyn_cast<AffineConstantExpr>(e)) return c.getValue() >= 0;
    if (isa<AffineDimExpr>(e)) return true;
    if (isa<AffineSymbolExpr>(e)) return false;  // we already checked symbols are empty
    auto bin = dyn_cast<AffineBinaryOpExpr>(e);
    if (!bin) return false;
    if (bin.getKind() == AffineExprKind::Add)
      return isShape(bin.getLHS()) && isShape(bin.getRHS());
    if (bin.getKind() == AffineExprKind::Mul) {
      // dim * positive-const, in either order.
      auto checkSide = [](AffineExpr a, AffineExpr b) -> bool {
        auto d = dyn_cast<AffineDimExpr>(a);
        auto c = dyn_cast<AffineConstantExpr>(b);
        return d && c && c.getValue() > 0;
      };
      return checkSide(bin.getLHS(), bin.getRHS()) ||
             checkSide(bin.getRHS(), bin.getLHS());
    }
    return false;  // floordiv, mod, ceildiv: reject
  };
  for (const auto &acc : accesses) {
    if (!isShape(acc.indexExpr)) return failure();
  }

  // Split into reads and writes. A memref appearing as both becomes ONE
  // entry in outs() (linalg-canonical for in-place / reduction patterns)
  // and the load result inside the body maps to that output's block arg.
  llvm::SmallVector<AccessInfo *, 4> reads;
  llvm::SmallVector<AccessInfo *, 4> writes;
  for (auto &acc : accesses) {
    if (acc.isWrite) writes.push_back(&acc);
    else             reads.push_back(&acc);
  }
  if (writes.empty()) return failure();
  if (writes.size() > 1) return failure();  // multi-write needs more body care

  AccessInfo *theWrite = writes.front();

  // Detect the in-place pattern: a load with the SAME memref AND SAME
  // indexing as the write. For gemm, that's the C[i,j] read+write inside
  // the k loop. The matching read becomes the output block arg.
  AccessInfo *inplaceRead = nullptr;
  for (auto *r : reads) {
    if (r->memref == theWrite->memref && r->indexExpr == theWrite->indexExpr) {
      inplaceRead = r;
      break;
    }
  }

  // Determine iterator types per dim. A dim is "reduction" iff the WRITE
  // operand's indexing map is independent of that dim AND there's an
  // in-place read+write on the same memref (true reduction signature).
  // Otherwise the dim is "parallel".
  //
  // Note: upstream ``linalg::vectorize`` has a bug for memref-form
  // ``linalg.generic`` with reductions — it emits a scalable destination
  // type incompatible with the fixed-size source, so the resulting
  // ``vector.multi_reduction`` fails to verify. Until that's fixed, we
  // bail out of multi-D conversion when any reduction is present. The
  // (working) custom ``lego-vectorize`` path then handles the nest.
  llvm::SmallVector<utils::IteratorType, 4> iterTypes(numDims,
                                                     utils::IteratorType::parallel);
  for (unsigned d = 0; d < numDims; ++d) {
    if (theWrite->indexExpr.isFunctionOfDim(d)) continue;
    // The output doesn't depend on d_d. Reduction iff in-place read+write.
    if (inplaceRead) {
      // Reduction case: fall through to custom path until upstream
      // linalg::vectorize gets the memref-form destination type right.
      return failure();
    }
    // Broadcast write — also bail (linalg memref-form doesn't model this
    // cleanly without a 0-rank input shim).
    return failure();
  }

  // All parallel dims must be divisible by SIMD*ILP for clean tile+vec.
  // Reduction dims can be any static size (we'll set their tile size to
  // the full bound).
  for (unsigned d = 0; d < numDims; ++d) {
    if (iterTypes[d] == utils::IteratorType::parallel) {
      if (nest.ubs[d] % kLinalgPathSimdWidth != 0) return failure();
      if (nest.ubs[d] < kLinalgPathSimdWidth) return failure();
    }
  }

  // For each access, decompose its AffineExpr into (involved dim
  // positions, their coefficients), validate the coefficients form a
  // valid row-major decomposition, and compute the desired N-D shape +
  // indexing map. We do this WITHOUT MUTATING IR — if any validation
  // fails, ``return failure()`` and the original nest is untouched.
  // The actual ``memref.expand_shape`` ops + ``linalg.generic`` are only
  // emitted at the end once every access is known to be convertible.

  // Helper: collect (dimPos, coeff) pairs from an Add/Mul/Dim/Const tree.
  // Returns nullopt if the expression has unsupported structure.
  std::function<std::optional<llvm::SmallVector<std::pair<unsigned, int64_t>, 4>>(AffineExpr)>
      collectTerms = [&](AffineExpr e)
      -> std::optional<llvm::SmallVector<std::pair<unsigned, int64_t>, 4>> {
    llvm::SmallVector<std::pair<unsigned, int64_t>, 4> terms;
    std::function<bool(AffineExpr)> walk = [&](AffineExpr sub) -> bool {
      if (auto c = dyn_cast<AffineConstantExpr>(sub)) {
        // Constant offset — ignored for shape; we'll capture it separately
        // if needed (we already required offsets >= 0 via isShape check).
        return true;
      }
      if (auto d = dyn_cast<AffineDimExpr>(sub)) {
        terms.push_back({d.getPosition(), 1});
        return true;
      }
      auto bin = dyn_cast<AffineBinaryOpExpr>(sub);
      if (!bin) return false;
      if (bin.getKind() == AffineExprKind::Add)
        return walk(bin.getLHS()) && walk(bin.getRHS());
      if (bin.getKind() == AffineExprKind::Mul) {
        // dim * const, in either order.
        auto dL = dyn_cast<AffineDimExpr>(bin.getLHS());
        auto cR = dyn_cast<AffineConstantExpr>(bin.getRHS());
        auto dR = dyn_cast<AffineDimExpr>(bin.getRHS());
        auto cL = dyn_cast<AffineConstantExpr>(bin.getLHS());
        if (dL && cR) {
          terms.push_back({dL.getPosition(), cR.getValue()});
          return true;
        }
        if (dR && cL) {
          terms.push_back({dR.getPosition(), cL.getValue()});
          return true;
        }
        return false;
      }
      return false;
    };
    if (!walk(e)) return std::nullopt;
    return terms;
  };

  // Phase 1: validate each access; compute shape, reassoc, and indexing
  // map without mutating IR.
  struct ValidatedAccess {
    AccessInfo *acc;
    llvm::SmallVector<int64_t, 4> shape;     // expanded N-D shape
    AffineMap indexingMap;                    // (d0..dN-1) -> projection
  };
  auto validate = [&](AccessInfo *acc) -> std::optional<ValidatedAccess> {
    auto termsOpt = collectTerms(acc->indexExpr);
    if (!termsOpt) return std::nullopt;
    auto &terms = *termsOpt;

    llvm::SmallDenseMap<unsigned, int64_t> dimCoef;
    for (auto [pos, c] : terms) dimCoef[pos] += c;
    for (auto [_, c] : dimCoef) if (c <= 0) return std::nullopt;

    llvm::SmallVector<std::pair<unsigned, int64_t>, 4> ordered(
        dimCoef.begin(), dimCoef.end());
    std::sort(ordered.begin(), ordered.end(),
              [](auto &a, auto &b) { return a.second > b.second; });

    for (size_t k = 0; k + 1 < ordered.size(); ++k) {
      int64_t expected = nest.ubs[ordered[k + 1].first] * ordered[k + 1].second;
      if (ordered[k].second != expected) return std::nullopt;
    }
    if (!ordered.empty() && ordered.back().second != 1) return std::nullopt;

    llvm::SmallVector<int64_t, 4> shape;
    for (auto [pos, _] : ordered) shape.push_back(nest.ubs[pos]);

    auto memrefTy = cast<MemRefType>(acc->memref.getType());
    if (memrefTy.getRank() != 1) return std::nullopt;
    int64_t totalSize = memrefTy.getShape()[0];
    int64_t expectedTotal = 1;
    for (int64_t s : shape) expectedTotal *= s;
    if (totalSize != expectedTotal) return std::nullopt;

    llvm::SmallVector<AffineExpr, 4> resultExprs;
    for (auto [pos, _] : ordered)
      resultExprs.push_back(getAffineDimExpr(pos, ctx));
    AffineMap map = AffineMap::get(numDims, /*symCount=*/0, resultExprs, ctx);

    return ValidatedAccess{acc, std::move(shape), map};
  };

  llvm::SmallVector<ValidatedAccess, 4> validatedReads;
  std::optional<ValidatedAccess> validatedWrite;
  for (auto *r : reads) {
    if (r == inplaceRead) continue;
    auto v = validate(r);
    if (!v) return failure();
    validatedReads.push_back(std::move(*v));
  }
  {
    auto v = validate(theWrite);
    if (!v) return failure();
    validatedWrite = std::move(*v);
  }

  // Phase 2: emit memref.expand_shape ops + linalg.generic.
  rewriter.setInsertionPoint(nest.loops.front());

  llvm::DenseMap<Value, Value> reshapeCache;
  auto emitReshape = [&](AccessInfo *acc, ArrayRef<int64_t> shape) -> Value {
    auto it = reshapeCache.find(acc->memref);
    if (it != reshapeCache.end()) return it->second;
    auto memrefTy = cast<MemRefType>(acc->memref.getType());
    ReassociationIndices reassocIndices;
    for (size_t k = 0; k < shape.size(); ++k)
      reassocIndices.push_back(static_cast<int64_t>(k));
    llvm::SmallVector<ReassociationIndices, 1> reassoc;
    reassoc.push_back(std::move(reassocIndices));
    auto resultTypeOr = memref::ExpandShapeOp::computeExpandedType(
        memrefTy, shape, reassoc);
    assert(succeeded(resultTypeOr) && "validated above");
    llvm::SmallVector<OpFoldResult> outputShape;
    for (int64_t s : shape) outputShape.push_back(rewriter.getIndexAttr(s));
    Value reshaped = memref::ExpandShapeOp::create(
                         rewriter, loc, *resultTypeOr, acc->memref, reassoc,
                         outputShape)
                         .getResult();
    reshapeCache[acc->memref] = reshaped;
    return reshaped;
  };

  llvm::SmallVector<AffineMap, 4> indexingMaps;
  llvm::SmallVector<Value, 4> insOperands;
  llvm::SmallVector<Value, 4> outsOperands;
  llvm::SmallVector<AccessInfo *, 4> orderedReads;
  for (auto &vr : validatedReads) {
    indexingMaps.push_back(vr.indexingMap);
    insOperands.push_back(emitReshape(vr.acc, vr.shape));
    orderedReads.push_back(vr.acc);
  }
  indexingMaps.push_back(validatedWrite->indexingMap);
  outsOperands.push_back(emitReshape(validatedWrite->acc, validatedWrite->shape));

  llvm::SmallVector<Type, 4> blockArgTypes;
  for (auto *r : orderedReads)
    blockArgTypes.push_back(cast<MemRefType>(r->memref.getType()).getElementType());
  blockArgTypes.push_back(
      cast<MemRefType>(theWrite->memref.getType()).getElementType());

  rewriter.setInsertionPoint(nest.loops.front());
  auto generic = linalg::GenericOp::create(
      rewriter, loc, /*resultTypes=*/TypeRange{},
      /*inputs=*/insOperands, /*outputs=*/outsOperands,
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iterTypes,
      [&](OpBuilder &b, Location bloc, ValueRange blockArgs) {
        IRMapping bvm;
        unsigned readIdx = 0;
        for (auto *r : orderedReads) {
          bvm.map(cast<memref::LoadOp>(r->op).getResult(), blockArgs[readIdx++]);
        }
        // The in-place read maps to the OUTPUT block arg (the existing
        // value of the output cell — the accumulator for reductions).
        Value outArg = blockArgs.back();
        if (inplaceRead) {
          bvm.map(cast<memref::LoadOp>(inplaceRead->op).getResult(), outArg);
        }

        // Find the value that gets stored.
        Value yieldVal;
        for (Operation &op : innermost.getBody()->without_terminator()) {
          if (auto store = dyn_cast<memref::StoreOp>(op)) {
            yieldVal = store.getValueToStore();
            break;
          }
        }
        assert(yieldVal);

        // Backward closure: clone only the use-def cone of yieldVal that
        // lives inside the innermost body. This excludes address-arithmetic
        // ops (those reference the IVs which are now dead) and the load /
        // store ops (those are mapped or removed).
        SmallVector<Operation *> toClone;
        DenseSet<Operation *> visited;
        std::function<void(Value)> walk = [&](Value v) {
          if (bvm.contains(v)) return;
          Operation *def = v.getDefiningOp();
          if (!def) return;
          if (def->getBlock() != innermost.getBody()) return;
          if (!visited.insert(def).second) return;
          for (Value operand : def->getOperands()) walk(operand);
          toClone.push_back(def);
        };
        walk(yieldVal);
        for (Operation *def : toClone) b.clone(*def, bvm);
        linalg::YieldOp::create(b, bloc, ValueRange{bvm.lookupOrDefault(yieldVal)});
      });
  (void)generic;

  // Erase the entire nest from outer to inner (each loop has only one inner
  // op left after we drop ``cur``: the inner scf.for, which we'll erase
  // next iteration).
  rewriter.eraseOp(nest.loops.front());
  return success();
}

struct ConvertLegoToLinalgPass
    : public PassWrapper<ConvertLegoToLinalgPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLegoToLinalgPass)

  ConvertLegoToLinalgPass() = default;
  ConvertLegoToLinalgPass(const ConvertLegoToLinalgPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "convert-lego-to-linalg";
  }
  StringRef getDescription() const override {
    return "Raise scf.for loops with affine memref accesses to linalg.generic";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, arith::ArithDialect,
                    vector::VectorDialect>();
  }

  /// Pass option: after raising to linalg.generic, immediately call
  /// upstream linalg::vectorize to produce vector-dialect code. This is
  /// the "do everything" path: scf.for -> linalg.generic -> vector ops.
  Option<bool> vectorize{*this, "vectorize",
                          llvm::cl::desc("Run upstream linalg::vectorize on the "
                                         "raised linalg.generic ops"),
                          llvm::cl::init(false)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func.getContext());

    // Step 1: try perfect-nest conversion on every OUTERMOST scf.for. A
    // successful conversion erases the whole nest; a failure leaves the
    // nest intact so the 1-D path (Step 2) or the custom lego-vectorize
    // can still process the innermost loop.
    llvm::SmallVector<scf::ForOp, 4> outerForOps;
    func.walk([&](scf::ForOp op) {
      // "Outermost" means parent is not scf.for.
      if (!isa<scf::ForOp>(op->getParentOp()))
        outerForOps.push_back(op);
    });
    llvm::DenseSet<Operation *> erased;
    for (scf::ForOp outer : outerForOps) {
      if (erased.count(outer)) continue;
      auto nest = detectPerfectNest(outer);
      if (!nest || nest->loops.size() < 2) continue;
      if (succeeded(tryConvertPerfectNestToLinalg(*nest, rewriter, vectorize))) {
        for (auto loop : nest->loops) erased.insert(loop);
      }
    }

    // Step 2: process remaining INNERMOST scf.for ops as 1-D loops.
    llvm::SmallVector<scf::ForOp, 4> candidates;
    func.walk([&](scf::ForOp op) {
      if (erased.count(op)) return;
      // Skip loops that contain another scf.for — those are not innermost.
      bool hasInnerFor = false;
      op.getBody()->walk([&](scf::ForOp inner) {
        if (inner != op) hasInnerFor = true;
      });
      if (!hasInnerFor) candidates.push_back(op);
    });

    llvm::SmallVector<linalg::GenericOp, 4> created;
    for (scf::ForOp op : candidates) {
      // Try the conversion; ignore failure (loop just stays as scf.for).
      auto insertPoint = op->getNextNode();
      if (succeeded(tryConvertForToLinalgGeneric(op, rewriter))) {
        // The newly-inserted linalg.generic is the previous-sibling of the
        // (now-deleted) scf.for. We can find it by walking back.
        // Easier: collect all linalg.generic ops at the end of this pass.
        (void)insertPoint;
      }
    }

    // Re-collect linalg.generic ops created by the conversion. We do this
    // AFTER all conversions to avoid iterator invalidation while rewriting.
    //
    // For each linalg.generic, we (a) tile by the SIMD width per dim,
    // producing scf.for with inner linalg.generic of static SIMD-width
    // shape, then (b) call linalg::vectorize on the inner op. This avoids
    // the trap of vectorising at full iteration size (vector<1Mxf32> for
    // saxpy) which makes LLVM lowering hang.
    if (vectorize) {
      func.walk([&](linalg::GenericOp gen) { created.push_back(gen); });
      for (linalg::GenericOp gen : created) {
        SmallVector<int64_t> staticRanges = gen.getStaticLoopRanges();
        bool dynamic = false;
        for (int64_t r : staticRanges) {
          if (ShapedType::isDynamic(r)) { dynamic = true; break; }
        }
        if (dynamic) continue;

        // Pick a tile size per dim. Default: kLinalgPathSimdWidth (16 f32
        // / 8 f64 on AVX-512). The conversion above guarantees that every
        // iteration dim is a multiple of this width.
        int64_t simdWidth = kLinalgPathSimdWidth;
        SmallVector<OpFoldResult> tileSizes;
        bool anyToTile = false;
        for (int64_t r : staticRanges) {
          int64_t t = (r >= simdWidth && r % simdWidth == 0) ? simdWidth : r;
          if (t < r) anyToTile = true;
          tileSizes.push_back(rewriter.getIndexAttr(t));
        }

        // Tile the op (only if at least one dim is being tiled — otherwise
        // tileLinalgOp degenerates to a no-op wrapper).
        linalg::LinalgOp innerOp = gen;
        if (anyToTile) {
          linalg::LinalgTilingOptions tilingOpts;
          tilingOpts.setTileSizes(
              llvm::map_to_vector(tileSizes, [&](OpFoldResult v) -> Value {
                return arith::ConstantIndexOp::create(
                           rewriter, gen.getLoc(),
                           cast<IntegerAttr>(cast<Attribute>(v)).getInt())
                    .getResult();
              }));
          tilingOpts.setLoopType(linalg::LinalgTilingLoopType::Loops);
          rewriter.setInsertionPoint(gen);
          FailureOr<linalg::TiledLinalgOp> tiled =
              linalg::tileLinalgOp(rewriter, gen, tilingOpts);
          if (succeeded(tiled)) {
            innerOp = tiled->op;
            rewriter.eraseOp(gen);
          } else {
            // Tiling failed; skip vectorization for this op.
            continue;
          }
        }

        // Vectorize the (possibly tiled) inner linalg op at the static
        // tile-size shape.
        SmallVector<int64_t> vecSizes;
        for (auto fold : tileSizes) {
          vecSizes.push_back(cast<IntegerAttr>(cast<Attribute>(fold)).getInt());
        }
        // Pin scalable-dims to false so we always get fixed-size vectors
        // (avoids inconsistent ``vector<[N]x[N]xT>`` for output operand
        // when input operands are ``vector<NxNxT>``).
        SmallVector<bool> scalableDims(vecSizes.size(), false);
        rewriter.setInsertionPoint(innerOp);
        FailureOr<linalg::VectorizationResult> result =
            linalg::vectorize(rewriter, innerOp, vecSizes, scalableDims);
        if (succeeded(result)) {
          if (result->replacements.empty())
            rewriter.eraseOp(innerOp);
          else
            rewriter.replaceOp(innerOp, result->replacements);
        }
      }
    }
  }
};

}  // namespace

namespace mlir::lego {
std::unique_ptr<Pass> createConvertLegoToLinalgPass() {
  return std::make_unique<ConvertLegoToLinalgPass>();
}

std::unique_ptr<Pass> createConvertLegoToLinalgPass(bool vectorize) {
  auto pass = std::make_unique<ConvertLegoToLinalgPass>();
  pass->vectorize = vectorize;
  return pass;
}
}  // namespace mlir::lego
