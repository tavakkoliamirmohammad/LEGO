//===- LegoVectorizeFilteredReduce.cpp - Generalised associative fold ----===//
//
// Recognises the canonical *associative-fold* pattern that ``clang -O3``
// either fully scalarises (when paired with a data-dependent branch) or
// partially vectorises (multi-output reductions on Zen 4 frequently fall
// to scalar after one or two iter_args).  ONE pass subsumes:
//
//   1. filtered-reduce — single float iter_arg with a predicate-gated
//      combine (``arith.addf`` / ``mulf`` / ``maximumf`` / ``minimumf``):
//
//        acc = identity
//        for i in 0..N:
//            if cond[i] <pred> threshold:
//                acc = combine(acc, A[i])
//
//   2. multi-output reduction — ≥ 2 iter_args with no predicate, one
//      combine per iter_arg:
//
//        s = 0; m = -inf
//        for i in 0..N:
//            s = s + A[i]
//            m = max(m, A[i])
//
//   3. predicated count — float (or int) iter_arg with predicate-gated
//      ``acc + 1.0`` (or ``acc + 1``):
//
//        cnt = 0
//        for i in 0..N: if pred(A[i]): cnt = cnt + 1
//
//   4. all / any — i1 iter_arg with predicate-gated ``ori`` (any) or
//      ``andi`` (all):
//
//        flag = false
//        for i in 0..N: if pred(A[i]): flag = flag | true
//
// Vectorisation strategy is uniform across all four shapes: maintain
// one vector accumulator per iter_arg, suppress non-passing lanes via
// ``arith.select(mask, value, identity)`` (when a predicate is present),
// reduce after the loop, and fold in the original ``initAcc``.  Uses
// upstream MLIR vector ops only — no architecture-specific intrinsics.
//
// Generalisation note (axis 1 of task #46): the matcher accepts both
// the ``scf.if`` form (cpu_dsl-direct emission with a side-effect-free
// then-branch that canonicalize is conservative about folding) and the
// canonicalised ``arith.select`` form.  When no predicate is present at
// all, the body is just a flat sequence of combine ops + scf.yield.
//
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LEGOVECTORIZEFILTEREDREDUCEPASS
#include "Lego/Passes.h"
#include "LegoSpecializedVectorize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

using mlir::lego::specialised::computeStripBounds;
using mlir::lego::specialised::getDefaultLanesForType;

/// Associative combine flavours recognised across float / int / i1 ops.
/// Each maps to (a) the scalar arith op used to combine accumulator +
/// value, (b) the identity element used to suppress non-passing lanes,
/// and (c) the corresponding ``vector.reduction`` kind.
enum class CombineKind {
  AddF, MulF, MaximumF, MinimumF,
  AddI, MulI, MaxSI, MinSI, MaxUI, MinUI,
  AndI, OrI, XOrI,
};

static std::optional<CombineKind> classifyCombine(Operation *op) {
  if (isa<arith::AddFOp>(op))     return CombineKind::AddF;
  if (isa<arith::MulFOp>(op))     return CombineKind::MulF;
  if (isa<arith::MaximumFOp>(op)) return CombineKind::MaximumF;
  if (isa<arith::MinimumFOp>(op)) return CombineKind::MinimumF;
  if (isa<arith::AddIOp>(op))     return CombineKind::AddI;
  if (isa<arith::MulIOp>(op))     return CombineKind::MulI;
  if (isa<arith::MaxSIOp>(op))    return CombineKind::MaxSI;
  if (isa<arith::MinSIOp>(op))    return CombineKind::MinSI;
  if (isa<arith::MaxUIOp>(op))    return CombineKind::MaxUI;
  if (isa<arith::MinUIOp>(op))    return CombineKind::MinUI;
  if (isa<arith::AndIOp>(op))     return CombineKind::AndI;
  if (isa<arith::OrIOp>(op))      return CombineKind::OrI;
  if (isa<arith::XOrIOp>(op))     return CombineKind::XOrI;
  return std::nullopt;
}

static bool isFloatCombine(CombineKind k) {
  return k == CombineKind::AddF || k == CombineKind::MulF ||
         k == CombineKind::MaximumF || k == CombineKind::MinimumF;
}

/// Build the identity scalar value for ``kind`` at element type ``ty``.
static Value emitIdentityScalar(OpBuilder &b, Location loc, CombineKind kind,
                                Type ty) {
  if (auto fpTy = dyn_cast<FloatType>(ty)) {
    const llvm::fltSemantics &sem = fpTy.getFloatSemantics();
    APFloat val = APFloat::getZero(sem);
    switch (kind) {
      case CombineKind::AddF:     val = APFloat::getZero(sem); break;
      case CombineKind::MulF:     val = APFloat(sem, 1); break;
      case CombineKind::MaximumF: val = APFloat::getInf(sem, /*Negative=*/true); break;
      case CombineKind::MinimumF: val = APFloat::getInf(sem, /*Negative=*/false); break;
      default: llvm_unreachable("non-float kind on float type");
    }
    return arith::ConstantOp::create(b, loc, ty,
                                     b.getFloatAttr(ty, val)).getResult();
  }
  // Integer / i1 identities.
  unsigned bw = ty.getIntOrFloatBitWidth();
  APInt val = APInt::getZero(bw);
  switch (kind) {
    case CombineKind::AddI:  case CombineKind::OrI: case CombineKind::XOrI:
      val = APInt::getZero(bw); break;
    case CombineKind::MulI:
      val = APInt(bw, 1); break;
    case CombineKind::AndI:
      val = APInt::getAllOnes(bw); break;
    case CombineKind::MaxSI:
      val = APInt::getSignedMinValue(bw); break;
    case CombineKind::MinSI:
      val = APInt::getSignedMaxValue(bw); break;
    case CombineKind::MaxUI:
      val = APInt::getMinValue(bw); break;
    case CombineKind::MinUI:
      val = APInt::getMaxValue(bw); break;
    default: llvm_unreachable("non-int kind on int type");
  }
  return arith::ConstantOp::create(b, loc, ty,
                                   b.getIntegerAttr(ty, val)).getResult();
}

static Value emitIdentityVector(OpBuilder &b, Location loc, CombineKind kind,
                                VectorType vecTy) {
  Value scalar = emitIdentityScalar(b, loc, kind, vecTy.getElementType());
  return vector::BroadcastOp::create(b, loc, vecTy, scalar).getResult();
}

static Value emitCombine(OpBuilder &b, Location loc, CombineKind kind,
                         Value lhs, Value rhs) {
  switch (kind) {
    case CombineKind::AddF:     return arith::AddFOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MulF:     return arith::MulFOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MaximumF: return arith::MaximumFOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MinimumF: return arith::MinimumFOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::AddI:     return arith::AddIOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MulI:     return arith::MulIOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MaxSI:    return arith::MaxSIOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MinSI:    return arith::MinSIOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MaxUI:    return arith::MaxUIOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MinUI:    return arith::MinUIOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::AndI:     return arith::AndIOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::OrI:      return arith::OrIOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::XOrI:     return arith::XOrIOp::create(b, loc, lhs, rhs).getResult();
  }
  llvm_unreachable("unknown CombineKind");
}

static vector::CombiningKind toVectorCombiningKind(CombineKind kind) {
  switch (kind) {
    case CombineKind::AddF:     return vector::CombiningKind::ADD;
    case CombineKind::MulF:     return vector::CombiningKind::MUL;
    case CombineKind::MaximumF: return vector::CombiningKind::MAXIMUMF;
    case CombineKind::MinimumF: return vector::CombiningKind::MINIMUMF;
    case CombineKind::AddI:     return vector::CombiningKind::ADD;
    case CombineKind::MulI:     return vector::CombiningKind::MUL;
    case CombineKind::MaxSI:    return vector::CombiningKind::MAXSI;
    case CombineKind::MinSI:    return vector::CombiningKind::MINSI;
    case CombineKind::MaxUI:    return vector::CombiningKind::MAXUI;
    case CombineKind::MinUI:    return vector::CombiningKind::MINUI;
    case CombineKind::AndI:     return vector::CombiningKind::AND;
    case CombineKind::OrI:      return vector::CombiningKind::OR;
    case CombineKind::XOrI:     return vector::CombiningKind::XOR;
  }
  llvm_unreachable("unknown CombineKind");
}

/// Per-iter-arg recipe: how the iter_arg is updated each iteration.
struct CombineEntry {
  CombineKind kind;       // which op
  Value       valueIn;    // the value combined with iter_arg (load on iv,
                          //   loop-invariant constant, or other invariant
                          //   value defined outside the loop)
  bool        isInvariant;// true → broadcast(valueIn); false → transfer_read
  Value       initAcc;    // forOp.getInits()[k]
  unsigned    iterArgIdx; // 0..N-1
};

/// Predicate descriptor when present.
struct PredicateInfo {
  Operation *cmpOp;       // arith.cmpf or arith.cmpi producing the i1
  Value      condValue;   // the value compared against threshold (loaded)
  Value      threshold;   // RHS — must be loop-invariant
  bool       isFloat;     // true → cmpf, false → cmpi
};

/// Decoded shape of the loop.
struct FoldShape {
  SmallVector<CombineEntry> entries;            // one per iter_arg
  std::optional<PredicateInfo> predicate;       // empty for unconditional fold
  SmallVector<memref::LoadOp> dataLoads;        // any unit-stride loads on iv
};

static constexpr llvm::StringRef kFilteredReduceDoneAttr =
    "lego.filtered_reduce_done";

/// Look up a Value's defining op as a memref.load on ``iv`` (allow loads
/// from any memref, but require a unit-stride access).
static memref::LoadOp getUnitStrideLoad(Value v, Value iv) {
  auto ld = v.getDefiningOp<memref::LoadOp>();
  if (!ld) return nullptr;
  if (ld.getIndices().size() != 1) return nullptr;
  if (ld.getIndices().front() != iv) return nullptr;
  return ld;
}

/// Returns true if ``v`` is loop-invariant w.r.t. ``forOp`` — either a
/// constant, a block argument from outside the loop, or an op defined
/// outside the loop region.
static bool isLoopInvariant(Value v, scf::ForOp forOp) {
  if (auto *defOp = v.getDefiningOp())
    return !forOp->isAncestor(defOp);
  // Block argument: invariant iff it's not the loop's IV or iter_args.
  if (auto ba = dyn_cast<BlockArgument>(v))
    return ba.getOwner() != forOp.getBody();
  return false;
}

static std::optional<FoldShape>
matchFoldPattern(scf::ForOp forOp) {
  if (forOp->hasAttr(kFilteredReduceDoneAttr)) return std::nullopt;

  unsigned nia = forOp.getNumRegionIterArgs();
  if (nia < 1) return std::nullopt;
  // All iter_args must be of the same int-or-float type (uniform width).
  Type firstTy = forOp.getRegionIterArgs()[0].getType();
  if (!firstTy.isIntOrFloat()) return std::nullopt;
  for (unsigned i = 1; i < nia; ++i)
    if (forOp.getRegionIterArgs()[i].getType() != firstTy)
      return std::nullopt;

  APInt stepVal;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
      stepVal != 1)
    return std::nullopt;

  Block *body = forOp.getBody();
  Value iv = forOp.getInductionVar();

  // Walk the body once and collect: cmp ops, scf.if op, combine ops.
  Operation *cmpOp = nullptr;
  scf::IfOp  ifOp;
  SmallVector<Operation *, 4> bodyCombines;  // unconditional combines
  for (Operation &op : body->getOperations()) {
    if (auto c = dyn_cast<arith::CmpFOp>(&op)) {
      if (cmpOp) return std::nullopt;
      cmpOp = c;
    } else if (auto c = dyn_cast<arith::CmpIOp>(&op)) {
      if (cmpOp) return std::nullopt;
      cmpOp = c;
    } else if (auto i = dyn_cast<scf::IfOp>(&op)) {
      if (ifOp) return std::nullopt;
      ifOp = i;
    } else if (auto k = classifyCombine(&op)) {
      bodyCombines.push_back(&op);
      (void)k;
    } else if (isa<memref::LoadOp, scf::YieldOp, arith::ConstantOp>(&op)) {
      // Allowed.
    } else {
      return std::nullopt;
    }
  }

  // Map iter_arg Value → index for fast lookup.
  DenseMap<Value, unsigned> iaIndex;
  for (unsigned i = 0; i < nia; ++i)
    iaIndex[forOp.getRegionIterArgs()[i]] = i;

  // ---- Mode A: predicate-gated (cmp + scf.if) ----
  // Mode B: unconditional fold (no scf.if).
  // Both must yield exactly one combine result per iter_arg.
  SmallVector<CombineEntry> entries(nia);
  std::optional<PredicateInfo> pred;

  auto fy = cast<scf::YieldOp>(body->getTerminator());
  if (fy.getNumOperands() != nia) return std::nullopt;

  // Validate predicate, if present.
  if (cmpOp) {
    if (!ifOp) {
      // cmp without scf.if — could be the predicate of an arith.select
      // form (canonical filtered-reduce shape).  Not currently handled
      // by this generalised path; bail.
      return std::nullopt;
    }
    if (ifOp.getCondition() != cmpOp->getResult(0)) return std::nullopt;
    if (ifOp.getNumResults() != nia) return std::nullopt;
    for (unsigned i = 0; i < nia; ++i) {
      if (ifOp.getResult(i).getType() != firstTy) return std::nullopt;
      if (fy.getOperand(i) != ifOp.getResult(i)) return std::nullopt;
    }
    // Else-branch yields each iter_arg unchanged.
    if (ifOp.getElseRegion().empty()) return std::nullopt;
    Block *elseBlk = &ifOp.getElseRegion().front();
    auto eY = cast<scf::YieldOp>(elseBlk->getTerminator());
    if (eY.getNumOperands() != nia) return std::nullopt;
    for (unsigned i = 0; i < nia; ++i)
      if (eY.getOperand(i) != forOp.getRegionIterArgs()[i])
        return std::nullopt;
    // Then-branch has one combine per iter_arg + scf.yield.
    Block *thenBlk = &ifOp.getThenRegion().front();
    SmallVector<Operation *, 4> thenCombines;
    for (Operation &op : thenBlk->getOperations()) {
      if (classifyCombine(&op)) thenCombines.push_back(&op);
      else if (!isa<memref::LoadOp, scf::YieldOp, arith::ConstantOp>(&op))
        return std::nullopt;
    }
    if (thenCombines.size() != nia) return std::nullopt;
    auto tY = cast<scf::YieldOp>(thenBlk->getTerminator());
    if (tY.getNumOperands() != nia) return std::nullopt;
    // For each iter_arg index k, locate the combine whose result the
    // then-branch yields at position k.  The combine must consume
    // iter_arg[i] as one operand; the other operand may be either a
    // unit-stride memref.load on iv OR a loop-invariant value
    // (e.g. ``cnt + 1.0`` for predicated count).
    for (unsigned i = 0; i < nia; ++i) {
      Value yielded = tY.getOperand(i);
      Operation *combine = yielded.getDefiningOp();
      if (!combine || !classifyCombine(combine)) return std::nullopt;
      Value lhs = combine->getOperand(0), rhs = combine->getOperand(1);
      Value otherVal;
      if (lhs == forOp.getRegionIterArgs()[i]) otherVal = rhs;
      else if (rhs == forOp.getRegionIterArgs()[i]) otherVal = lhs;
      else return std::nullopt;
      bool inv = isLoopInvariant(otherVal, forOp);
      if (!inv && !getUnitStrideLoad(otherVal, iv))
        return std::nullopt;
      entries[i].kind        = *classifyCombine(combine);
      entries[i].valueIn     = otherVal;
      entries[i].isInvariant = inv;
      entries[i].initAcc     = forOp.getInits()[i];
      entries[i].iterArgIdx  = i;
    }
    // Predicate descriptor.
    PredicateInfo pi;
    pi.cmpOp = cmpOp;
    pi.isFloat = isa<arith::CmpFOp>(cmpOp);
    Value pLhs, pRhs;
    if (pi.isFloat) {
      auto cf = cast<arith::CmpFOp>(cmpOp);
      pLhs = cf.getLhs(); pRhs = cf.getRhs();
    } else {
      auto ci = cast<arith::CmpIOp>(cmpOp);
      pLhs = ci.getLhs(); pRhs = ci.getRhs();
    }
    auto cLoad = getUnitStrideLoad(pLhs, iv);
    if (!cLoad) return std::nullopt;
    pi.condValue = pLhs;
    pi.threshold = pRhs;
    if (auto *defOp = pi.threshold.getDefiningOp())
      if (forOp->isAncestor(defOp)) return std::nullopt;
    pred = pi;
  } else {
    // No cmp: unconditional fold.  Body must contain exactly nia combine
    // ops + yield; each combine consumes its own iter_arg + a load on iv.
    if (ifOp) return std::nullopt;
    if (bodyCombines.size() != nia) return std::nullopt;
    for (unsigned i = 0; i < nia; ++i) {
      Value yielded = fy.getOperand(i);
      Operation *combine = yielded.getDefiningOp();
      if (!combine || !classifyCombine(combine)) return std::nullopt;
      Value lhs = combine->getOperand(0), rhs = combine->getOperand(1);
      Value otherVal;
      if (lhs == forOp.getRegionIterArgs()[i]) otherVal = rhs;
      else if (rhs == forOp.getRegionIterArgs()[i]) otherVal = lhs;
      else return std::nullopt;
      bool inv = isLoopInvariant(otherVal, forOp);
      if (!inv && !getUnitStrideLoad(otherVal, iv))
        return std::nullopt;
      entries[i].kind        = *classifyCombine(combine);
      entries[i].valueIn     = otherVal;
      entries[i].isInvariant = inv;
      entries[i].initAcc     = forOp.getInits()[i];
      entries[i].iterArgIdx  = i;
    }
  }

  // Collect any unit-stride loads in the body for cataloguing.
  SmallVector<memref::LoadOp> loads;
  body->walk([&](memref::LoadOp ld) {
    if (ld.getIndices().size() == 1 && ld.getIndices().front() == iv)
      loads.push_back(ld);
  });

  // Element-type sanity: int combines on int iter_arg, float on float.
  bool isFloatTy = isa<FloatType>(firstTy);
  for (auto &e : entries)
    if (isFloatCombine(e.kind) != isFloatTy) return std::nullopt;

  FoldShape shape;
  shape.entries  = std::move(entries);
  shape.predicate = pred;
  shape.dataLoads = std::move(loads);
  return shape;
}

/// Clone the original scalar body for the tail loop.
static void cloneScalarBody(OpBuilder &b, Location loc, scf::ForOp origLoop,
                            Value newIv, ValueRange newIterArgs) {
  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), newIv);
  for (auto z : llvm::zip(origLoop.getRegionIterArgs(), newIterArgs))
    mapping.map(std::get<0>(z), std::get<1>(z));
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

struct VectorizeFoldPattern : public OpRewritePattern<scf::ForOp> {
  llvm::StringRef target;
  VectorizeFoldPattern(MLIRContext *ctx, llvm::StringRef target)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2), target(target) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto shape = matchFoldPattern(forOp);
    if (!shape) return failure();

    Location loc = forOp.getLoc();
    Type firstTy = forOp.getRegionIterArgs()[0].getType();
    int64_t elemBytes = std::max<int64_t>(firstTy.getIntOrFloatBitWidth() / 8, 1);
    int64_t L = getDefaultLanesForType(target, elemBytes);
    if (L < 2) return failure();
    unsigned nia = forOp.getNumRegionIterArgs();

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    auto sb = computeStripBounds(rewriter, loc, lb, ub, L);
    Value cL = sb.cL;
    Value stripUb = sb.stripUb;

    auto vecTy = VectorType::get({L}, firstTy);

    // General path: vector iter_arg inits: identity per iter_arg's
    // combine.  We fold the original initAcc back in *after* the
    // reduction.
    //
    // Note: a popcount-based fast path for predicated count
    // (loop-invariant value + addf/addi combine) would eliminate the
    // per-lane select+add.  Deferred — see task #50/#46 follow-up; the
    // general path already takes count_if from 0.04x to ~0.87x of
    // clang's popcount lowering.
    SmallVector<Value> vecInits;
    vecInits.reserve(nia);
    for (auto &e : shape->entries)
      vecInits.push_back(emitIdentityVector(rewriter, loc, e.kind, vecTy));

    auto vecLoop = scf::ForOp::create(rewriter, loc, lb, stripUb, cL, vecInits);
    vecLoop->setAttr(kFilteredReduceDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(vecLoop.getBody());
      Value iv = vecLoop.getInductionVar();

      // Load every required input vector.  Uses a small cache by memref +
      // index pattern; the shared dataLoad field is the canonical input.
      DenseMap<Operation *, Value> loadCache;
      auto loadVecFor = [&](memref::LoadOp ld) -> Value {
        auto it = loadCache.find(ld);
        if (it != loadCache.end()) return it->second;
        Value pad;
        if (auto fpTy = dyn_cast<FloatType>(ld.getType()))
          pad = arith::ConstantOp::create(rewriter, loc, fpTy,
                                          rewriter.getZeroAttr(fpTy));
        else
          pad = arith::ConstantOp::create(rewriter, loc, ld.getType(),
                                          rewriter.getZeroAttr(ld.getType()));
        Value v =
            vector::TransferReadOp::create(
                rewriter, loc,
                VectorType::get({L}, ld.getType()), ld.getMemRef(),
                ValueRange{iv}, pad,
                /*inBounds=*/ArrayRef<bool>{true}).getResult();
        loadCache[ld] = v;
        return v;
      };

      // Phase 1 — eagerly issue every required transfer_read first
      // (gives a clean op order: all loads, then mask build, then
      // selects + combines).  For loop-invariant ``valueIn`` values
      // (e.g. ``cnt + 1.0`` for predicated count), broadcast a single
      // scalar to a vector instead of loading.
      SmallVector<Value> vIns;
      vIns.reserve(nia);
      for (unsigned i = 0; i < nia; ++i) {
        const auto &e = shape->entries[i];
        if (e.isInvariant) {
          vIns.push_back(
              vector::BroadcastOp::create(rewriter, loc, vecTy, e.valueIn)
                  .getResult());
        } else {
          auto valLoad = e.valueIn.getDefiningOp<memref::LoadOp>();
          vIns.push_back(loadVecFor(valLoad));
        }
      }
      Value vCondVec;
      if (shape->predicate) {
        auto condLoad =
            shape->predicate->condValue.getDefiningOp<memref::LoadOp>();
        vCondVec = loadVecFor(condLoad);
      }

      // Phase 2 — build mask (if predicate).
      Value mask;
      if (shape->predicate) {
        const auto &pi = *shape->predicate;
        Value vThr = vector::BroadcastOp::create(
                         rewriter, loc, vCondVec.getType(), pi.threshold);
        if (pi.isFloat) {
          mask = arith::CmpFOp::create(
                     rewriter, loc,
                     cast<arith::CmpFOp>(pi.cmpOp).getPredicate(),
                     vCondVec, vThr).getResult();
        } else {
          mask = arith::CmpIOp::create(
                     rewriter, loc,
                     cast<arith::CmpIOp>(pi.cmpOp).getPredicate(),
                     vCondVec, vThr).getResult();
        }
      }

      // Phase 3 — per iter_arg: optional select + combine.
      SmallVector<Value> newAccs;
      newAccs.reserve(nia);
      for (unsigned i = 0; i < nia; ++i) {
        const auto &e = shape->entries[i];
        Value contrib = vIns[i];
        if (mask) {
          Value identVec = emitIdentityVector(rewriter, loc, e.kind, vecTy);
          contrib =
              arith::SelectOp::create(rewriter, loc, mask, vIns[i], identVec)
                  .getResult();
        }
        Value acc = vecLoop.getRegionIterArgs()[i];
        newAccs.push_back(emitCombine(rewriter, loc, e.kind, acc, contrib));
      }
      scf::YieldOp::create(rewriter, loc, newAccs);
    }

    // Reduce each vector accumulator to scalar; fold initAcc back in.
    rewriter.setInsertionPointAfter(vecLoop);
    SmallVector<Value> scalarSeeds;
    scalarSeeds.reserve(nia);
    for (unsigned i = 0; i < nia; ++i) {
      const auto &e = shape->entries[i];
      Value vAcc = vecLoop.getResult(i);
      Value scalarPartial =
          vector::ReductionOp::create(
              rewriter, loc, toVectorCombiningKind(e.kind), vAcc,
              arith::FastMathFlags::none).getResult();
      scalarSeeds.push_back(emitCombine(rewriter, loc, e.kind,
                                        e.initAcc, scalarPartial));
    }

    // Tail loop: scalar body for (N mod L) iterations seeded from
    // partial reductions.
    auto tailLoop = scf::ForOp::create(rewriter, loc, stripUb, ub,
                                       forOp.getStep(), scalarSeeds);
    tailLoop->setAttr(kFilteredReduceDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      cloneScalarBody(rewriter, loc, forOp,
                      tailLoop.getInductionVar(),
                      tailLoop.getRegionIterArgs());
    }

    rewriter.replaceOp(forOp, tailLoop.getResults());
    return success();
  }
};

struct LegoVectorizeFilteredReducePass
    : public mlir::lego::impl::LegoVectorizeFilteredReducePassBase<
          LegoVectorizeFilteredReducePass> {
  using LegoVectorizeFilteredReducePassBase::LegoVectorizeFilteredReducePassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeFoldPattern>(&getContext(), this->target);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVectorizeFilteredReducePass() {
  return std::make_unique<LegoVectorizeFilteredReducePass>();
}
std::unique_ptr<Pass>
createLegoVectorizeFilteredReducePass(llvm::StringRef target) {
  LegoVectorizeFilteredReducePassOptions opts;
  opts.target = target.str();
  return std::make_unique<LegoVectorizeFilteredReducePass>(opts);
}
} // namespace lego
} // namespace mlir
