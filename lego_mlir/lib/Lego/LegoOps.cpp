#include "Lego/LegoOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

using namespace mlir;
using namespace mlir::lego;
using namespace mlir::transform;

// ============================================================================
// Custom Parsers and Printers
// ============================================================================

static void printGenPRegion(OpAsmPrinter &printer, Operation *op, Region &region) {
  printer << " ";
  if (region.empty()) {
    printer << "{ }";
    return;
  }
  Block &entry = region.front();
  printer << "(";
  llvm::interleaveComma(entry.getArguments(), printer, [&](BlockArgument arg) {
    printer.printRegionArgument(arg);
  });
  printer << ") ";
  
  printer.printRegion(region, /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

static ParseResult parseGenPRegion(OpAsmParser &parser, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true))
    return failure();

  // Parse the region.
  // The parsed arguments are passed to the region parser to be used as the
  // entry block arguments.
  return parser.parseRegion(region, args, /*enableNameShadowing=*/false);
}

#define GET_OP_CLASSES
#include "Lego/LegoOps.cpp.inc"
#include "Lego/LegoUtils.h"
#include <numeric>

// ============================================================================
// RegPOp Verification
// ============================================================================

LogicalResult RegPOp::verify() {
  auto perm = extractI64Array(getPerm());
  auto dims = extractI64Array(getDims());

  if (perm.size() != dims.size()) {
    return emitOpError("Permutation rank " + std::to_string(perm.size()) +
                       " does not match dimensions rank " +
                       std::to_string(dims.size()));
  }

  for (int64_t d : dims) {
      if (d <= 0) return emitOpError("Dimension " + std::to_string(d) + " must be strictly positive");
  }

  // Verify perm is a valid permutation of 0..size-1
  SmallVector<int64_t> sortedPerm = perm;
  std::sort(sortedPerm.begin(), sortedPerm.end());
  for (size_t i = 0; i < sortedPerm.size(); ++i) {
    if (sortedPerm[i] != (int64_t)i) {
      return emitOpError("Invalid permutation: not a permutation of 0.." +
                         std::to_string(sortedPerm.size() - 1));
    }
  }

  return success();
}

// ============================================================================
// TileByOp Verification
// ============================================================================

LogicalResult TileByOp::verify() {
  auto info = extractNestedTileDims(getTileDims());
  if (!info.valid) {
    return emitOpError("Invalid tile dimensions structure. Expected nested list [[...], ...]");
  }

  for (auto d : info.flatDims) {
      if (d <= 0) return emitOpError("Tile dimension " + std::to_string(d) + " must be strictly positive");
  }

  int64_t d = info.d;
  int64_t q = info.q; // Unused for check, but part of structure

  // Get input (d, q) from OrderBy or other layout
  auto [inputD, inputQ] = getLayoutDQ(getInput());
  
  if (inputD != 0 || inputQ != 0) {
      if (d != inputD) {
          return emitOpError("Inner tile dimension " + std::to_string(d) + 
                             " does not match input layout dimension " + std::to_string(inputD));
      }

      // Verify global product of dimensions (volume preservation)
      int64_t tileProduct = 1;
      for (auto attr : getTileDims()) {
          auto tileGroup = extractI64Array(cast<ArrayAttr>(attr));
          for (auto x : tileGroup) tileProduct *= x;
      }

      int64_t inputProduct = 1;
      auto inputDims = getLayoutInputDims(getInput());
      for (auto x : inputDims) inputProduct *= x;

      if (tileProduct != inputProduct) {
           return emitOpError("Total product of tile dims (" + std::to_string(tileProduct) + 
                              ") does not match total product of input dims (" + 
                              std::to_string(inputProduct) + ")");
      }
  }

  return success();
}

// ============================================================================
// RowOp Verification
// ============================================================================

LogicalResult RowOp::verify() {
    auto dims = extractI64Array(getDims());
    for (int64_t d : dims) {
        if (d <= 0) return emitOpError("Dimension " + std::to_string(d) + " must be strictly positive");
    }
    return success();
}

// ============================================================================
// ColOp Verification
// ============================================================================

// ============================================================================
// Layout Algebraic Identity Patterns
// ============================================================================

namespace {
struct SimplifyApplyInverse : public OpRewritePattern<ApplyOp> {
  using OpRewritePattern<ApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyOp op,
                                PatternRewriter &rewriter) const override {
    // apply(L, apply_inverse(L, flat)) -> flat
    auto invOp = op.getIndices()[0].getDefiningOp<ApplyInverseOp>();
    if (!invOp) return failure();

    if (invOp.getLayout() != op.getLayout()) return failure();

    rewriter.replaceOp(op, invOp.getFlatIndex());
    return success();
  }
};

struct SimplifyInverseApply : public OpRewritePattern<ApplyInverseOp> {
  using OpRewritePattern<ApplyInverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyInverseOp op,
                                PatternRewriter &rewriter) const override {
    // apply_inverse(L, apply(L, indices)) -> indices
    auto applyOp = op.getFlatIndex().getDefiningOp<ApplyOp>();
    if (!applyOp) return failure();

    if (applyOp.getLayout() != op.getLayout()) return failure();

    rewriter.replaceOp(op, applyOp.getIndices());
    return success();
  }
};
} // namespace

void ApplyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SimplifyApplyInverse>(context);
}

void ApplyInverseOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<SimplifyInverseApply>(context);
}

namespace {

// Helper to extract a linear combination from a value
// Returns a map of block argument index -> stride
static std::optional<DenseMap<int, int64_t>> extractStrides(Value v, Block *entry) {
    DenseMap<int, int64_t> strides;
    
    // Base case: block argument
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
        if (blockArg.getOwner() == entry) {
            strides[blockArg.getArgNumber()] = 1;
            return strides;
        }
        return std::nullopt;
    }

    // Base case: constant (not allowed as a separate term for now, unless 0)
    APInt constValue;
    if (matchPattern(v, m_ConstantInt(&constValue))) {
        if (constValue.getSExtValue() == 0) return strides;
        return std::nullopt; // constant offset not supported in RegP
    }

    Operation *op = v.getDefiningOp();
    if (!op) return std::nullopt;

    if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        auto lhs = extractStrides(addOp.getLhs(), entry);
        auto rhs = extractStrides(addOp.getRhs(), entry);
        if (!lhs || !rhs) return std::nullopt;
        for (auto &it : *lhs) strides[it.first] += it.second;
        for (auto &it : *rhs) strides[it.first] += it.second;
        return strides;
    }

    if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
        Value val, multiplier;
        APInt m;
        if (matchPattern(mulOp.getLhs(), m_ConstantInt(&m))) {
            multiplier = mulOp.getLhs();
            val = mulOp.getRhs();
        } else if (matchPattern(mulOp.getRhs(), m_ConstantInt(&m))) {
            multiplier = mulOp.getRhs();
            val = mulOp.getLhs();
        } else {
            return std::nullopt;
        }

        auto innerStrides = extractStrides(val, entry);
        if (!innerStrides) return std::nullopt;
        int64_t mValue = m.getSExtValue();
        for (auto &it : *innerStrides) strides[it.first] += it.second * mValue;
        return strides;
    }

    return std::nullopt;
}

struct SimplifyLinearGenP : public OpRewritePattern<GenPOp> {
  using OpRewritePattern<GenPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenPOp op,
                                PatternRewriter &rewriter) const override {
    auto dims = extractI64Array(op.getDims());
    int rank = dims.size();
    
    Block &block = op.getBody().front();
    Operation *term = block.getTerminator();
    if (term->getNumOperands() != 1) return failure();

    auto stridesOpt = extractStrides(term->getOperand(0), &block);
    if (!stridesOpt) return failure();
    auto &detectedStrides = *stridesOpt;

    // We want to find a permutation P s.t. detectedStrides[P[i]] = product_{j=i+1..N-1} dims[P[j]]
    // This is equivalent to finding a permutation that reproduces the detected strides.
    
    SmallVector<int64_t> perm(rank, -1);
    SmallVector<bool> used(rank, false);
    
    // Greedily match strides. The smallest stride must be 1 (for the last dimension in perm).
    // The next smallest must be dims[P[last]].
    
    int64_t currentTargetStride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        bool found = false;
        for (int j = 0; j < rank; ++j) {
            if (!used[j] && detectedStrides[j] == currentTargetStride) {
                perm[i] = j;
                used[j] = true;
                currentTargetStride *= dims[j];
                found = true;
                break;
            }
        }
        if (!found) return failure();
    }

    rewriter.replaceOpWithNewOp<RegPOp>(op, op.getType(),
                                       rewriter.getI64ArrayAttr(perm),
                                       rewriter.getI64ArrayAttr(dims));
    return success();
  }
};
} // namespace

void GenPOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<SimplifyLinearGenP>(context);
}

LogicalResult ColOp::verify() {
    auto dims = extractI64Array(getDims());
    for (int64_t d : dims) {
        if (d <= 0) return emitOpError("Dimension " + std::to_string(d) + " must be strictly positive");
    }
    return success();
}

DiagnosedSilenceableFailure ApplyLayoutTransformOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results,
    transform::TransformState &state) {
  
  // Get the target payload operations
  auto targets = state.getPayloadOps(getTarget());
  
  // TODO: Retrieve the layout object from getLayout()
  // Since layout is an SSA value, we need to inspect what defined it.
  
  // For now, we just pass the targets through to the result.
  results.set(getOperation()->getResult(0), targets);
  
  return DiagnosedSilenceableFailure::success();
}

void ApplyLayoutTransformOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  transform::consumesHandle(getOperation()->getOpOperands().take_front(1), effects);
  transform::onlyReadsHandle(getOperation()->getOpOperands().drop_front(1), effects);
  transform::producesHandle(getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}
