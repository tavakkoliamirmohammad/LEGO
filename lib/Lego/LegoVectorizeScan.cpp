//===- LegoVectorizeScan.cpp - Vectorize prefix-scan loops ---------------===//
//
// Recognises the canonical inclusive prefix-scan / cumulative-sum pattern
// that ``clang -O3`` scalarises:
//
//     acc = 0
//     for i in 0..N:
//         acc = acc + A[i]
//         B[i] = acc
//
// clang refuses to vectorise because of the loop-carried dependency on
// ``acc``.  But the scan can be vectorised via the **Hillis-Steele** in-
// vector prefix sum: lg(L) stages of shift-and-add inside each L-wide
// chunk, plus a scalar carry between chunks.
//
// LEGO recognises the shape and rewrites to a strip-mined vector loop
// using **upstream MLIR vector dialect ops only**:
//
//     scf.for %i = 0 to stripUb step L iter_args(%acc = 0.0) -> f32 {
//       %v       = vector.transfer_read A[%i]      : vector<L x T>
//       // log2(L) Hillis-Steele stages: shift right by 1, 2, 4, ..., L/2.
//       for k in 1..log2(L):
//         %vshift = vector.shuffle %v, %zero [L,L,..,0,1,..,L-1-2^k]
//         %v      = arith.addf %v, %vshift
//       // Add running carry.
//       %vbcast  = vector.broadcast %acc
//       %vout    = arith.addf %v, %vbcast
//       vector.transfer_write %vout, B[%i]
//       // Last lane is the new carry (full prefix sum of the chunk).
//       %newacc  = vector.extract %vout[L-1]
//       scf.yield %newacc
//     }
//     // Tail: scalar body for (N mod L) iterations seeded from %newacc.
//
// The combine op is restricted to ``arith.addf`` for v1 (other associative
// ops — ``arith.mulf``, ``arith.maximumf``, etc. — are a trivial extension).
// Strict IEEE-754 reproducibility: false; matches ``clang -O3 -ffast-math``
// (LEGO already opts in to ``reassociate-fp-reductions=true`` in the x86
// pipeline by default).
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LEGOVECTORIZESCANPASS
#include "Lego/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {

static int64_t getLanesForType(llvm::StringRef target, int64_t elemBytes) {
  if (target == "avx512") return elemBytes == 4 ? 16 : 8;
  if (target == "avx2")   return elemBytes == 4 ? 8  : 4;
  if (target == "neon" || target == "sve") return elemBytes == 4 ? 4 : 2;
  return 16 / std::max<int64_t>(elemBytes, 1);
}

/// Which associative combine the loop is folding with — drives identity
/// constant + emitted vector op.
enum class ScanKind { AddF, MulF, MaximumF, MinimumF };

/// Decoded shape of an inclusive prefix-scan loop.
struct ScanShape {
  memref::LoadOp   inLoad;     // A[i] — the streamed-in element
  Operation       *combineOp;  // arith.addf / mulf / maximumf / minimumf
  memref::StoreOp  outStore;   // B[i] = newAcc
  Value            initAcc;    // initial accumulator (iter_arg init)
  ScanKind         kind;
};

static constexpr llvm::StringRef kScanDoneAttr = "lego.scan_done";

/// Identify the combine op kind, or std::nullopt if not a supported scan op.
static std::optional<ScanKind> classifyCombine(Operation *op) {
  if (isa<arith::AddFOp>(op))     return ScanKind::AddF;
  if (isa<arith::MulFOp>(op))     return ScanKind::MulF;
  if (isa<arith::MaximumFOp>(op)) return ScanKind::MaximumF;
  if (isa<arith::MinimumFOp>(op)) return ScanKind::MinimumF;
  return std::nullopt;
}

static std::optional<ScanShape> matchScanPattern(scf::ForOp forOp) {
  if (forOp->hasAttr(kScanDoneAttr)) return std::nullopt;

  // One f32/f64 iter_arg.
  if (forOp.getNumRegionIterArgs() != 1) return std::nullopt;
  Value iaAcc = forOp.getRegionIterArgs()[0];
  if (!isa<FloatType>(iaAcc.getType())) return std::nullopt;

  APInt stepVal;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
      stepVal != 1)
    return std::nullopt;

  Block *body = forOp.getBody();
  Value iv = forOp.getInductionVar();

  Operation *combineOp = nullptr;
  std::optional<ScanKind> combineKind;
  memref::StoreOp outStore;
  for (Operation &op : body->getOperations()) {
    if (auto k = classifyCombine(&op)) {
      if (combineOp) return std::nullopt;       // only one combine
      combineOp   = &op;
      combineKind = k;
    } else if (auto s = dyn_cast<memref::StoreOp>(&op)) {
      if (outStore) return std::nullopt;
      outStore = s;
    } else if (isa<memref::LoadOp, scf::YieldOp, arith::ConstantOp>(&op)) {
      // Allowed.
    } else {
      return std::nullopt;
    }
  }
  if (!combineOp || !outStore || !combineKind) return std::nullopt;

  // The combine op must mix the iter_arg with a unit-stride load on iv.
  Value lhs = combineOp->getOperand(0);
  Value rhs = combineOp->getOperand(1);
  memref::LoadOp inLoad;
  if (lhs == iaAcc) {
    inLoad = rhs.getDefiningOp<memref::LoadOp>();
  } else if (rhs == iaAcc) {
    inLoad = lhs.getDefiningOp<memref::LoadOp>();
  } else {
    return std::nullopt;
  }
  if (!inLoad) return std::nullopt;
  if (inLoad.getIndices().size() != 1) return std::nullopt;
  if (inLoad.getIndices().front() != iv) return std::nullopt;

  // The store must write the combine's result to B[iv].
  Value combined = combineOp->getResult(0);
  if (outStore.getValueToStore() != combined) return std::nullopt;
  if (outStore.getIndices().size() != 1) return std::nullopt;
  if (outStore.getIndices().front() != iv) return std::nullopt;

  // The scf.for must yield the combine's result.
  auto y = cast<scf::YieldOp>(body->getTerminator());
  if (y.getNumOperands() != 1) return std::nullopt;
  if (y.getOperand(0) != combined) return std::nullopt;

  ScanShape shape;
  shape.inLoad    = inLoad;
  shape.combineOp = combineOp;
  shape.outStore  = outStore;
  shape.initAcc   = forOp.getInits().front();
  shape.kind      = *combineKind;
  return shape;
}

/// Emit the appropriate associative combine on two SSA values — used by
/// hillisSteele for the in-vector tree and by the final per-chunk merge.
static Value emitCombine(OpBuilder &b, Location loc, ScanKind kind,
                         Value lhs, Value rhs) {
  switch (kind) {
    case ScanKind::AddF:
      return arith::AddFOp::create(b, loc, lhs, rhs).getResult();
    case ScanKind::MulF:
      return arith::MulFOp::create(b, loc, lhs, rhs).getResult();
    case ScanKind::MaximumF:
      return arith::MaximumFOp::create(b, loc, lhs, rhs).getResult();
    case ScanKind::MinimumF:
      return arith::MinimumFOp::create(b, loc, lhs, rhs).getResult();
  }
  llvm_unreachable("unknown ScanKind");
}

/// Identity element of the combine op as a vector constant of the
/// matching shape (used to pad shifted-in lanes during Hillis-Steele).
///   AddF      → 0
///   MulF      → 1
///   MaximumF  → -inf
///   MinimumF  → +inf
static Value emitIdentityVector(OpBuilder &b, Location loc, ScanKind kind,
                                VectorType vecTy) {
  auto fpTy = cast<FloatType>(vecTy.getElementType());
  APFloat val(fpTy.getFloatSemantics());
  switch (kind) {
    case ScanKind::AddF:
      val = APFloat::getZero(fpTy.getFloatSemantics());
      break;
    case ScanKind::MulF:
      val = APFloat(fpTy.getFloatSemantics(), 1);
      break;
    case ScanKind::MaximumF:
      val = APFloat::getInf(fpTy.getFloatSemantics(), /*Negative=*/true);
      break;
    case ScanKind::MinimumF:
      val = APFloat::getInf(fpTy.getFloatSemantics(), /*Negative=*/false);
      break;
  }
  auto attr = DenseElementsAttr::get(vecTy, FloatAttr::get(fpTy, val));
  return arith::ConstantOp::create(b, loc, vecTy, attr).getResult();
}

/// Build a Hillis-Steele in-vector inclusive prefix scan with combine
/// op ``kind``: input ``v`` of width L, returns the prefix-combined vector
/// of the same width.  Uses ``vector.shuffle`` + the combine op for
/// log2(L) stages.  Shifted-in lanes are padded with the combine's
/// identity element so the partial result matches the algebraic spec.
static Value hillisSteele(OpBuilder &b, Location loc, ScanKind kind,
                          Value v, int64_t L) {
  auto vecTy = cast<VectorType>(v.getType());

  // Identity-padding vector for the shifted-in lanes.
  Value ident = emitIdentityVector(b, loc, kind, vecTy);

  // Need L to be a power of 2 so log2(L) is exact.
  assert(llvm::isPowerOf2_64(L) && "Hillis-Steele requires L a power of 2");
  int64_t stages = llvm::Log2_64(L);
  for (int64_t s = 0; s < stages; ++s) {
    int64_t shift = int64_t{1} << s;          // 1, 2, 4, ..., L/2
    // Shuffle indices: result[j] = (j < shift) ? ident[j] : v[j - shift].
    // Indices [0, L) refer to v; indices [L, 2L) refer to ident.
    SmallVector<int64_t> mask(L);
    for (int64_t j = 0; j < L; ++j) {
      if (j < shift)
        mask[j] = L + j;       // pull from identity-pad vector
      else
        mask[j] = j - shift;   // pull from v at j-shift
    }
    Value shifted = vector::ShuffleOp::create(b, loc, v, ident, mask);
    v = emitCombine(b, loc, kind, v, shifted);
  }
  return v;
}

/// Clone the original scalar body for the tail loop with iv and iter_arg
/// remapping; capture the cloned yield's operand for tail output.
static void cloneScalarBody(OpBuilder &b, Location loc, scf::ForOp origLoop,
                            Value newIv, Value newAcc) {
  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), newIv);
  mapping.map(origLoop.getRegionIterArgs().front(), newAcc);
  for (Operation &op : origLoop.getBody()->getOperations()) {
    if (auto y = dyn_cast<scf::YieldOp>(op)) {
      SmallVector<Value> mappedOperands;
      for (Value v : y.getOperands())
        mappedOperands.push_back(mapping.lookupOrDefault(v));
      scf::YieldOp::create(b, op.getLoc(), mappedOperands);
      break;
    }
    b.clone(op, mapping);
  }
  (void)loc;
}

struct VectorizeScanPattern : public OpRewritePattern<scf::ForOp> {
  llvm::StringRef target;
  VectorizeScanPattern(MLIRContext *ctx, llvm::StringRef target)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2), target(target) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto shape = matchScanPattern(forOp);
    if (!shape) return failure();

    Location loc = forOp.getLoc();
    auto fpTy = cast<FloatType>(shape->initAcc.getType());
    int64_t elemBytes = fpTy.getWidth() / 8;
    int64_t L = getLanesForType(target, elemBytes);
    if (!llvm::isPowerOf2_64(L) || L < 2) return failure();

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();

    Value lenV   = arith::SubIOp::create(rewriter, loc, ub, lb);
    Value cL     = arith::ConstantIndexOp::create(rewriter, loc, L);
    Value chunks = arith::DivUIOp::create(rewriter, loc, lenV, cL);
    Value stripBody = arith::MulIOp::create(rewriter, loc, chunks, cL);
    Value stripUb = arith::AddIOp::create(rewriter, loc, lb, stripBody);

    auto vecTy = VectorType::get({L}, fpTy);
    // Pad value for transfer_read of A[i:i+L].  Strip-mined to a multiple
    // of L so the read is always in-bounds — pad value is unused, just a
    // syntactic requirement.
    Value pad = arith::ConstantOp::create(rewriter, loc, fpTy,
                                          rewriter.getZeroAttr(fpTy));

    // Broadcast initAcc into a vector — used as the iter_arg of the vec
    // loop.  Carrying a vector (not a scalar fp) prevents the main
    // ``lego-vectorize`` pass from re-strip-mining this loop and
    // mangling our hand-rolled Hillis-Steele body.
    Value vecInitAcc =
        vector::BroadcastOp::create(rewriter, loc, vecTy, shape->initAcc);

    auto vecLoop = scf::ForOp::create(rewriter, loc, lb, stripUb, cL,
                                      ValueRange{vecInitAcc});
    vecLoop->setAttr(kScanDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(vecLoop.getBody());
      Value iv = vecLoop.getInductionVar();
      Value vAcc = vecLoop.getRegionIterArgs().front();

      // 1. Load chunk.
      Value v = vector::TransferReadOp::create(
                    rewriter, loc, vecTy, shape->inLoad.getMemRef(),
                    ValueRange{iv}, pad,
                    /*inBounds=*/ArrayRef<bool>{true}).getResult();

      // 2. In-vector inclusive prefix combine.
      v = hillisSteele(rewriter, loc, shape->kind, v, L);

      // 3. Combine the running carry.  All lanes of vAcc hold the same
      //    broadcast value (carried last lane from the previous chunk).
      Value vout = emitCombine(rewriter, loc, shape->kind, v, vAcc);

      // 4. Store the chunk.
      vector::TransferWriteOp::create(
          rewriter, loc, vout, shape->outStore.getMemRef(),
          ValueRange{iv}, /*inBounds=*/ArrayRef<bool>{true});

      // 5. New carry = broadcast of vout's last lane (becomes vAcc next iter).
      Value lastLane = vector::ExtractOp::create(rewriter, loc, vout,
                                                 ArrayRef<int64_t>{L - 1})
                           .getResult();
      Value newAcc =
          vector::BroadcastOp::create(rewriter, loc, vecTy, lastLane);

      scf::YieldOp::create(rewriter, loc, ValueRange{newAcc});
    }

    // After the vector loop, extract the final scalar carry for the tail.
    rewriter.setInsertionPointAfter(vecLoop);
    Value finalAcc = vector::ExtractOp::create(
                         rewriter, loc, vecLoop.getResult(0),
                         ArrayRef<int64_t>{L - 1})
                         .getResult();

    // Tail loop: scalar body for (N mod L) seeded from final scalar carry.
    auto tailLoop = scf::ForOp::create(rewriter, loc, stripUb, ub,
                                       forOp.getStep(),
                                       ValueRange{finalAcc});
    tailLoop->setAttr(kScanDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      cloneScalarBody(rewriter, loc, forOp,
                      tailLoop.getInductionVar(),
                      tailLoop.getRegionIterArgs().front());
    }

    rewriter.replaceOp(forOp, ValueRange{tailLoop.getResult(0)});
    return success();
  }
};

struct LegoVectorizeScanPass
    : public mlir::lego::impl::LegoVectorizeScanPassBase<
          LegoVectorizeScanPass> {
  using LegoVectorizeScanPassBase::LegoVectorizeScanPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeScanPattern>(&getContext(), this->target);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVectorizeScanPass() {
  return std::make_unique<LegoVectorizeScanPass>();
}
std::unique_ptr<Pass> createLegoVectorizeScanPass(llvm::StringRef target) {
  LegoVectorizeScanPassOptions opts;
  opts.target = target.str();
  return std::make_unique<LegoVectorizeScanPass>(opts);
}
} // namespace lego
} // namespace mlir
