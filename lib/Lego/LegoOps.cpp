#include "Lego/LegoOps.h"

using namespace mlir;
using namespace mlir::lego;

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

#include "Lego/LegoUtils.h"
#include <numeric>

static ParseResult parseNestedTileDims(OpAsmParser &parser,
                                       SmallVectorImpl<OpAsmParser::UnresolvedOperand> &tile_dims,
                                       ArrayAttr &tile_shape) {
  if (parser.parseLSquare()) return failure();
  
  SmallVector<int64_t> shape;
  int expectedD = -1;
  
  auto parseList = [&]() -> ParseResult {
      if (parser.parseLSquare()) return failure();
      int count = 0;
      if (succeeded(parser.parseOptionalRSquare())) {
          // empty
      } else {
          do {
              OpAsmParser::UnresolvedOperand operand;
              if (parser.parseOperand(operand)) return failure();
              tile_dims.push_back(operand);
              count++;
          } while (succeeded(parser.parseOptionalComma()));
          if (parser.parseRSquare()) return failure();
      }
      if (expectedD == -1) expectedD = count;
      else if (expectedD != count) return parser.emitError(parser.getCurrentLocation(), "all tile groups must have the same size");
      shape.push_back(count);
      return success();
  };

  if (succeeded(parser.parseOptionalRSquare())) {
      // empty outer list
  } else {
      do {
          if (parseList()) return failure();
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) return failure();
  }
  
  tile_shape = parser.getBuilder().getI64ArrayAttr(shape);
  return success();
}

static void printNestedTileDims(OpAsmPrinter &printer, Operation *op,
                                OperandRange tile_dims,
                                ArrayAttr tile_shape) {
  printer << "[";
  auto shape = extractI64Array(tile_shape);
  int offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
      int count = shape[i];
      printer << "[";
      for (int j = 0; j < count; ++j) {
          printer.printOperand(tile_dims[offset + j]);
          if (j < count - 1) printer << ", ";
      }
      printer << "]";
      offset += count;
      if (i < shape.size() - 1) printer << ", ";
  }
  printer << "]";
}

#define GET_OP_CLASSES
#include "Lego/LegoOps.cpp.inc"
#include <numeric>

// ============================================================================
// RegPOp Verification
// ============================================================================

LogicalResult RegPOp::verify() {
  auto perm = extractI64Array(getPerm());
  auto dims = getDims();

  if (perm.size() != dims.size()) {
    return emitOpError("Permutation rank " + std::to_string(perm.size()) +
                       " does not match dimensions rank " +
                       std::to_string(dims.size()));
  }

  for (Value v : dims) {
      APInt d;
      if (matchPattern(v, m_ConstantInt(&d)) && d.getSExtValue() <= 0) {
          return emitOpError("Dimension " + std::to_string(d.getSExtValue()) + " must be strictly positive");
      }
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
  // TileBy only accepts OrderBy as input.
  if (auto *defOp = getInput().getDefiningOp()) {
    if (!mlir::isa<OrderByOp>(defOp))
      return emitOpError("input must be an OrderBy layout");
  }

  auto tileShapeOpt = getTileShape();
  auto tileShapeAttr = getTileShape();
  auto tileShape = extractI64Array(tileShapeAttr);
  
  if (tileShape.empty()) return success();
  
  int64_t q = tileShape.size();
  int64_t d = tileShape[0];
  
  auto tileDims = getTileDims();
  int64_t totalExpected = 0;
  for (int64_t s : tileShape) {
      if (s != d) return emitOpError("Irregular tile grouping not perfectly rectangular in dimensions");
      totalExpected += s;
  }
  
  if ((int64_t)tileDims.size() != totalExpected) return emitOpError("Mismatch between tile_shape and tile_dims count");

  for (Value v : tileDims) {
      APInt val;
      if (matchPattern(v, m_ConstantInt(&val)) && val.getSExtValue() <= 0) {
          return emitOpError("Tile dimension " + std::to_string(val.getSExtValue()) + " must be strictly positive");
      }
  }

  // Get input (d, q) from OrderBy or other layout
  auto [inputD, inputQ] = getLayoutDQ(getInput());
  
  if (inputD != 0 || inputQ != 0) {
      if (d != inputD) {
          return emitOpError("Inner tile dimension " + std::to_string(d) + 
                             " does not match input layout dimension " + std::to_string(inputD));
      }

      // Verify global product of dimensions (volume preservation)
      bool allConstant = true;
      int64_t tileProduct = 1;
      for (Value v : tileDims) {
          APInt val;
          if (matchPattern(v, m_ConstantInt(&val))) {
              tileProduct *= val.getSExtValue();
          } else {
              allConstant = false;
              break;
          }
      }

      auto inputDims = getLayoutInputDims(getInput());
      bool inputConstant = true;
      int64_t inputProduct = 1;
      for (Value v : inputDims) {
          APInt val;
          if (matchPattern(v, m_ConstantInt(&val))) {
              inputProduct *= val.getSExtValue();
          } else {
              inputConstant = false;
              break;
          }
      }

      if (allConstant && inputConstant && tileProduct != inputProduct) {
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
    for (Value v : getDims()) {
        APInt d;
        if (matchPattern(v, m_ConstantInt(&d)) && d.getSExtValue() <= 0)
            return emitOpError("Dimension " + std::to_string(d.getSExtValue()) + " must be strictly positive");
    }
    return success();
}

// ============================================================================
// ColOp Verification
// ============================================================================

LogicalResult ColOp::verify() {
    for (Value v : getDims()) {
        APInt d;
        if (matchPattern(v, m_ConstantInt(&d)) && d.getSExtValue() <= 0)
            return emitOpError("Dimension " + std::to_string(d.getSExtValue()) + " must be strictly positive");
    }
    return success();
}

// ============================================================================
// CastViewOp Verification
// ============================================================================

LogicalResult CastViewOp::verify() {
    auto memrefType = cast<MemRefType>(getMemref().getType());
    if (!memrefType.hasStaticShape()) {
        // For now, only support static shapes for size checking.
        return success();
    }

    int64_t memrefElements = memrefType.getNumElements();
    
    auto layoutDims = getLayoutInputDims(getLayout());
    if (layoutDims.empty()) {
        // Might be a dynamic layout or something getLayoutInputDims doesn't handle yet.
        return success();
    }

    int64_t layoutVolume = 1;
    bool allConstant = true;
    for (Value v : layoutDims) {
        APInt d;
        if (matchPattern(v, m_ConstantInt(&d))) {
            layoutVolume *= d.getSExtValue();
        } else {
            allConstant = false;
            break;
        }
    }

    if (allConstant && memrefElements != layoutVolume) {
        return emitOpError("MemRef total number of elements (" + std::to_string(memrefElements) +
                           ") does not match layout volume (" + std::to_string(layoutVolume) + ")");
    }

    return success();
}

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

// ============================================================================
// ApplyOp / ApplyInverseOp
// ============================================================================

LogicalResult ApplyOp::verify() {
  auto layout = getLayout();
  auto indices = getIndices();
  auto expectedDims = getLayoutInputDims(layout);
  if (expectedDims.empty())
    return success();

  if (indices.size() != expectedDims.size()) {
    return emitOpError("number of indices (")
           << indices.size() << ") does not match layout rank ("
           << expectedDims.size() << ")";
  }
  return success();
}

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
    auto dimsVals = op.getDims();
    int rank = dimsVals.size();
    
    SmallVector<int64_t> dims;
    dims.reserve(rank);
    for (Value v : dimsVals) {
        APInt d;
        if (!matchPattern(v, m_ConstantInt(&d))) return failure();
        dims.push_back(d.getSExtValue());
    }
    
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
                                       dimsVals);
    return success();
  }
};
} // namespace

void GenPOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<SimplifyLinearGenP>(context);
}

