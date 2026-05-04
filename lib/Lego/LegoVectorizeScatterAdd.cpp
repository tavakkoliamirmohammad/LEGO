//===- LegoVectorizeScatterAdd.cpp - Vectorize scatter-add loops ----------===//
//
// Recognises the canonical *histogram* / scatter-add pattern that ``clang
// -O3`` scalarises entirely:
//
//     for i in 0..N:
//         b = bin[i]
//         count[b] = count[b] + delta(i)
//
// where ``delta(i)`` is either a constant (pure histogram) or a unit-stride
// load ``A[i]`` (scatter-add / segmented sum).  ``clang`` refuses to vectorise
// because it cannot prove the lanes are conflict-free at compile time, even
// though AVX-512 has the exact instructions needed (``vpgatherdps``,
// ``vpscatterdps``).
//
// LEGO recognises the shape and rewrites to a strip-mined vector loop that
// uses **upstream MLIR vector dialect ops only**, with portable cross-lane
// conflict detection:
//
//     scf.for %i = 0 to N step L {
//       %vbin  = vector.transfer_read bin[%i]      : vector<L x i32>
//       // Pairwise duplicate detection via L-1 lane rotations + cmp + reduce.
//       %any   = OR over k in 1..L-1 of
//                  reduce-or(cmpi eq, %vbin, shuffle(%vbin, k))
//       scf.if %any {
//         // Scalar fallback for this L-element chunk: original body, L times.
//       } else {
//         %vidx = arith.index_cast %vbin               : vector<L x i32>
//         %vcur = vector.gather count[%c0][%vidx]      : vector<L x f32>
//         %vdel = vector.transfer_read A[%i]           : vector<L x f32>
//         %vnew = arith.addf %vcur, %vdel
//         vector.scatter count[%c0][%vidx], <mask=true>, %vnew
//       }
//     }
//     // Tail loop: original scalar body for (N mod L) iterations.
//
// The conflict-detection cost is O(L) shuffle/cmp ops per chunk.  Empirically
// most real-world histograms have low collision rates (large bin space), so
// the fast path dominates.
//
// Uses ``vector.gather``, ``vector.scatter``, ``vector.shuffle`` — the
// portable upstream ops; lowers to ``vpgatherdps``/``vpscatterdps`` on x86
// and equivalent SVE ops on aarch64.  No architecture-specific intrinsics in
// the recogniser.
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LEGOVECTORIZESCATTERADDPASS
#include "Lego/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Lane width for the value type.  Scatter-add specifically uses HALF the
/// natural SIMD width (8 lanes for f32 on AVX-512) because:
///   (1) conflict detection is O(L) shuffles + cmps, so L=8 cuts that cost
///       in half vs L=16 (7 vs 15 rotations per chunk);
///   (2) AVX-512 gather/scatter are microcoded and run at ~1 element/cycle,
///       so wider vectors don't speed up the actual gather/scatter.
static int64_t getLanesForType(llvm::StringRef target, int64_t elemBytes) {
  if (target == "avx512") return elemBytes == 4 ? 8 : 4;
  if (target == "avx2")   return elemBytes == 4 ? 8 : 4;
  if (target == "neon" || target == "sve") return elemBytes == 4 ? 4 : 2;
  return 16 / std::max<int64_t>(elemBytes, 1);
}

/// Decoded shape of a scatter-add loop.  Holds the values the rewrite needs.
struct ScatterAddShape {
  // Index source: bin[i] → cast → use as index.
  memref::LoadOp     binLoad;        // load %bin[%i]   : memref<NxiK>
  arith::IndexCastOp idxCast;        // i32 → index
  // Read-modify-write target: count[b].
  memref::LoadOp     countLoad;      // load %count[%b]
  memref::StoreOp    countStore;     // store result, %count[%b]
  // The combine op: count[b] = update(count[b], delta).  Exactly one operand
  // is the countLoad result; the other is the "delta".
  Operation         *combineOp;      // arith.addf / arith.addi (add only for v1)
  Value              delta;          // the non-load operand of combineOp
};

/// Marker attribute placed on the vec-loop and tail-loop after a successful
/// rewrite to prevent re-matching the cloned scalar body.
static constexpr llvm::StringRef kScatterAddDoneAttr = "lego.scatter_add_done";

/// Inspect ``forOp`` and decide whether it matches the scatter-add pattern.
static std::optional<ScatterAddShape>
matchScatterAddPattern(scf::ForOp forOp) {
  if (forOp->hasAttr(kScatterAddDoneAttr)) return std::nullopt;

  // No iter_args (pure side-effect loop).
  if (forOp.getNumRegionIterArgs() != 0) return std::nullopt;

  // Step must be 1.
  APInt stepVal;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
      stepVal != 1)
    return std::nullopt;

  // Walk body: collect the loads + cast + store + combine op.
  Block *body = forOp.getBody();
  Value iv = forOp.getInductionVar();

  arith::IndexCastOp idxCast;
  memref::StoreOp countStore;
  for (Operation &op : body->getOperations()) {
    if (auto c = dyn_cast<arith::IndexCastOp>(&op)) {
      // Allow only one i32→index cast in the body.  More than one would mean
      // the kernel does extra index work we don't recognise.
      if (idxCast) return std::nullopt;
      if (!c.getOut().getType().isIndex()) return std::nullopt;
      idxCast = c;
    } else if (auto s = dyn_cast<memref::StoreOp>(&op)) {
      if (countStore) return std::nullopt;
      countStore = s;
    } else if (isa<memref::LoadOp, scf::YieldOp, arith::ConstantOp,
                   arith::AddFOp>(&op)) {
      // Allowed by structure check below.
    } else {
      return std::nullopt;                // unknown op rejects pattern
    }
  }
  if (!idxCast || !countStore) return std::nullopt;

  // The store address must be the index cast result.
  if (countStore.getIndices().size() != 1) return std::nullopt;
  if (countStore.getIndices().front() != idxCast.getOut()) return std::nullopt;

  // The cast input must be a memref.load on the IV (the "bin" load).
  auto binLoad =
      idxCast.getIn().getDefiningOp<memref::LoadOp>();
  if (!binLoad) return std::nullopt;
  if (binLoad.getIndices().size() != 1) return std::nullopt;
  if (binLoad.getIndices().front() != iv) return std::nullopt;

  // The store value must come from a combine op (arith.addf for v1).
  auto combineOp = countStore.getValueToStore().getDefiningOp<arith::AddFOp>();
  if (!combineOp) return std::nullopt;

  // Exactly one operand of the combine must be a memref.load on the SAME
  // address as the store (i.e. count[b] read-modify-write).  The other
  // operand is the delta — must be loop-invariant w.r.t. ``i`` only via a
  // unit-stride load on the IV (or any value defined outside the loop).
  Value lhs = combineOp.getLhs();
  Value rhs = combineOp.getRhs();
  Value countLoadVal, delta;
  auto lhsLoad = lhs.getDefiningOp<memref::LoadOp>();
  auto rhsLoad = rhs.getDefiningOp<memref::LoadOp>();
  auto isCountReadOf = [&](memref::LoadOp ld) -> bool {
    if (!ld) return false;
    if (ld.getMemRef() != countStore.getMemRef()) return false;
    if (ld.getIndices().size() != 1) return false;
    if (ld.getIndices().front() != idxCast.getOut()) return false;
    return true;
  };
  if (isCountReadOf(lhsLoad)) {
    countLoadVal = lhs;
    delta = rhs;
  } else if (isCountReadOf(rhsLoad)) {
    countLoadVal = rhs;
    delta = lhs;
  } else {
    return std::nullopt;
  }

  ScatterAddShape shape;
  shape.binLoad = binLoad;
  shape.idxCast = idxCast;
  shape.countLoad = countLoadVal.getDefiningOp<memref::LoadOp>();
  shape.countStore = countStore;
  shape.combineOp = combineOp;
  shape.delta = delta;
  return shape;
}

/// Build cross-lane conflict detection: returns an i1 that is true iff any
/// pair of lanes in ``vbin`` holds equal values.  Uses L-1 rotations of the
/// vector via ``vector.shuffle`` + ``arith.cmpi eq`` + ``arith.ori``,
/// followed by ONE ``vector.reduction <or>``.  Single reduction is far
/// cheaper than 15 reductions (each ~5 cycles on x86 AVX-512).
static Value buildConflictDetect(OpBuilder &b, Location loc, Value vbin,
                                 int64_t L) {
  auto i1VecTy = VectorType::get({L}, b.getI1Type());
  // Accumulate the OR of all (vbin == rotated) comparisons as a vector mask;
  // any lane set in the final mask means *some* pair of lanes was equal.
  Value anyEq;
  for (int64_t k = 1; k < L; ++k) {
    SmallVector<int64_t> mask(L);
    for (int64_t j = 0; j < L; ++j) mask[j] = (j + k) % L;
    Value rotated = vector::ShuffleOp::create(b, loc, vbin, vbin, mask);
    Value eq = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                     vbin, rotated);
    if (!anyEq)
      anyEq = eq;
    else
      anyEq = arith::OrIOp::create(b, loc, anyEq, eq);
  }
  // One reduction across the whole accumulated mask vector → i1.
  return vector::ReductionOp::create(b, loc, vector::CombiningKind::OR,
                                     anyEq, arith::FastMathFlags::none)
      .getResult();
  (void)i1VecTy;
}

/// Clone the original scalar body L times into the current block.  Used by
/// the conflict-fallback inner loop and by the tail loop.
static void cloneScalarBody(OpBuilder &b, Location loc, scf::ForOp origLoop,
                            Value newIv) {
  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), newIv);
  for (Operation &op : origLoop.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op)) continue;       // skip terminator
    b.clone(op, mapping);
  }
}

struct VectorizeScatterAddPattern : public OpRewritePattern<scf::ForOp> {
  llvm::StringRef target;
  VectorizeScatterAddPattern(MLIRContext *ctx, llvm::StringRef target)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2), target(target) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto shape = matchScatterAddPattern(forOp);
    if (!shape) return failure();

    Location loc = forOp.getLoc();
    Type elemTy = shape->countLoad.getType();
    if (!elemTy.isIntOrFloat()) return failure();
    int64_t elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    int64_t L = getLanesForType(target, elemBytes);
    if (L < 2) return failure();

    // bin element type (e.g. i32) determines the index vector type for gather.
    Type binElemTy = shape->binLoad.getType();
    if (!binElemTy.isInteger()) return failure();

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();

    // stripUb = lb + ((ub - lb) / L) * L  ;  tail runs from stripUb to ub.
    Value lenV   = arith::SubIOp::create(rewriter, loc, ub, lb);
    Value cL     = arith::ConstantIndexOp::create(rewriter, loc, L);
    Value chunks = arith::DivUIOp::create(rewriter, loc, lenV, cL);
    Value stripBody = arith::MulIOp::create(rewriter, loc, chunks, cL);
    Value stripUb = arith::AddIOp::create(rewriter, loc, lb, stripBody);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto binVecTy = VectorType::get({L}, binElemTy);
    auto idxVecTy = VectorType::get({L}, rewriter.getIndexType());
    auto valVecTy = VectorType::get({L}, elemTy);
    auto maskVecTy = VectorType::get({L}, rewriter.getI1Type());

    // Vector loop: scf.for %i = %lb to %stripUb step %L  (no iter_args).
    auto vecLoop = scf::ForOp::create(rewriter, loc, lb, stripUb, cL,
                                      /*iterArgs=*/ValueRange{});
    vecLoop->setAttr(kScatterAddDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(vecLoop.getBody());
      Value iv = vecLoop.getInductionVar();

      // Pad value (unused, mask=true everywhere) for transfer_read.
      Value binPad = arith::ConstantOp::create(
          rewriter, loc, binElemTy, rewriter.getZeroAttr(binElemTy));
      Value valPad = arith::ConstantOp::create(
          rewriter, loc, elemTy, rewriter.getZeroAttr(elemTy));

      Value vbin = vector::TransferReadOp::create(
                       rewriter, loc, binVecTy, shape->binLoad.getMemRef(),
                       ValueRange{iv}, binPad,
                       /*inBounds=*/ArrayRef<bool>{true}).getResult();

      Value anyConflict = buildConflictDetect(rewriter, loc, vbin, L);

      // scf.if anyConflict { scalar fallback } else { vectorized }
      auto ifOp = scf::IfOp::create(rewriter, loc, /*resultTypes=*/TypeRange{},
                                    anyConflict, /*withElseRegion=*/true);

      // Then-branch: scalar inner loop over [iv, iv+L).
      {
        OpBuilder::InsertionGuard g2(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        Value ivEnd = arith::AddIOp::create(rewriter, loc, iv, cL);
        auto inner = scf::ForOp::create(rewriter, loc, iv, ivEnd, c1,
                                        /*iterArgs=*/ValueRange{});
        inner->setAttr(kScatterAddDoneAttr, rewriter.getUnitAttr());
        {
          OpBuilder::InsertionGuard g3(rewriter);
          rewriter.setInsertionPointToStart(inner.getBody());
          cloneScalarBody(rewriter, loc, forOp, inner.getInductionVar());
        }
      }

      // Else-branch: vector gather → addf → scatter.
      {
        OpBuilder::InsertionGuard g2(rewriter);
        rewriter.setInsertionPointToStart(ifOp.elseBlock());

        // Cast bin vector (i32 lanes) to index vector for gather/scatter.
        Value vidx = arith::IndexCastOp::create(rewriter, loc, idxVecTy, vbin)
                         .getResult();

        // All-true mask.
        Value trueMask = arith::ConstantOp::create(
            rewriter, loc, maskVecTy,
            DenseElementsAttr::get(maskVecTy, rewriter.getBoolAttr(true)));

        // vector.gather count[%c0][%vidx]
        Value valPassThru = arith::ConstantOp::create(
            rewriter, loc, valVecTy, rewriter.getZeroAttr(valVecTy));
        Value vcur = vector::GatherOp::create(
                         rewriter, loc, valVecTy,
                         shape->countLoad.getMemRef(), ValueRange{c0},
                         vidx, trueMask, valPassThru).getResult();

        // Build the delta vector.
        //   - If delta is a memref.load on the IV → vector.transfer_read.
        //   - Else (loop-invariant scalar) → vector.broadcast.
        Value vdelta;
        if (auto deltaLoad = shape->delta.getDefiningOp<memref::LoadOp>()) {
          if (deltaLoad.getIndices().size() == 1 &&
              deltaLoad.getIndices().front() == forOp.getInductionVar()) {
            vdelta = vector::TransferReadOp::create(
                         rewriter, loc, valVecTy, deltaLoad.getMemRef(),
                         ValueRange{iv}, valPad,
                         /*inBounds=*/ArrayRef<bool>{true}).getResult();
          }
        }
        if (!vdelta) {
          vdelta = vector::BroadcastOp::create(rewriter, loc, valVecTy,
                                               shape->delta).getResult();
        }

        Value vnew = arith::AddFOp::create(rewriter, loc, vcur, vdelta)
                         .getResult();

        vector::ScatterOp::create(rewriter, loc, /*result=*/Type{},
                                  shape->countStore.getMemRef(),
                                  ValueRange{c0}, vidx, trueMask, vnew,
                                  /*alignment=*/IntegerAttr{});
      }
    }

    // Tail loop: clone the original scalar body for (N mod L) iterations.
    auto tailLoop = scf::ForOp::create(rewriter, loc, stripUb, ub,
                                       forOp.getStep(),
                                       /*iterArgs=*/ValueRange{});
    tailLoop->setAttr(kScatterAddDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      cloneScalarBody(rewriter, loc, forOp, tailLoop.getInductionVar());
    }

    rewriter.eraseOp(forOp);
    return success();
  }
};

struct LegoVectorizeScatterAddPass
    : public mlir::lego::impl::LegoVectorizeScatterAddPassBase<
          LegoVectorizeScatterAddPass> {
  using LegoVectorizeScatterAddPassBase::LegoVectorizeScatterAddPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeScatterAddPattern>(&getContext(), this->target);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVectorizeScatterAddPass() {
  return std::make_unique<LegoVectorizeScatterAddPass>();
}
std::unique_ptr<Pass>
createLegoVectorizeScatterAddPass(llvm::StringRef target) {
  LegoVectorizeScatterAddPassOptions opts;
  opts.target = target.str();
  return std::make_unique<LegoVectorizeScatterAddPass>(opts);
}
} // namespace lego
} // namespace mlir
