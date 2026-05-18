/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/Norm/FuseLayerNorm.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSELAYERNORM
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {
constexpr double kLayerNormEpsilon = 1e-5;

/** Check if a value is a double/float constant and return its value */
bool isValidEpsConst(Value v, double &outDouble) {
  auto constOp = v.getDefiningOp<mfuse::ConstantOp>();
  if (!constOp) {
    return false;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
  if (!dense || !dense.isSplat()) {
    return false;
  }
  auto elemType = dense.getElementType();
  if (!mlir::isa<mlir::FloatType>(elemType)) {
    return false;
  }
  if (elemType.isF64()) {
    outDouble = dense.getSplatValue<llvm::APFloat>().convertToDouble();
  } else if (elemType.isF32()) {
    outDouble = static_cast<double>(dense.getSplatValue<llvm::APFloat>().convertToFloat());
  } else {
    return false;
  }
  return true;
}

/** Extract epsilon value from AddOp */
bool extractEpsilonFromAdd(AddOp addOp, Value &variance, double &eps) {
  Value addLhs = addOp.getX();
  Value addRhs = addOp.getY();
  Operation *addLhsOp = addLhs.getDefiningOp();
  Operation *addRhsOp = addRhs.getDefiningOp();

  // Check if operand is AclnnVarMeanOp's variance output
  if (auto varMeanOp = dyn_cast<AclnnVarMeanOp>(addLhsOp)) {
    if (addLhs == varMeanOp.getVarianceOut() && isValidEpsConst(addRhs, eps)) {
      variance = addLhs;
      return true;
    }
  } else if (auto varMeanOp = dyn_cast<AclnnVarMeanOp>(addRhsOp)) {
    if (addRhs == varMeanOp.getVarianceOut() && isValidEpsConst(addLhs, eps)) {
      variance = addRhs;
      return true;
    }
  }
  return false;
}

/** Check if a value has only one user */
bool hasOnlyOneUser(Value v) { return v.hasOneUse(); }

/** Check if tensor shape matches the given normalized_shape */
bool checkShapeMatchesNormalizedShape(Value v, ArrayAttr normalizedShape) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return false;
  }
  const auto &shape = tensorType.getShape();
  if (shape.size() != normalizedShape.size()) {
    return false;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    auto dimAttr = mlir::dyn_cast<IntegerAttr>(normalizedShape[i]);
    if (!dimAttr) {
      return false;
    }
    int64_t normalizedDim = dimAttr.getInt();
    if (shape[i] != normalizedDim && normalizedDim != -1) {
      return false;
    }
  }
  return true;
}

/** LayerNormMatcher class for matching LayerNorm operation sequence */
class LayerNormMatcher {
 public:
  struct MatchResult {
    Value x;
    Value gamma;
    Value beta;
    double eps;
    ArrayAttr normalized_shape;
    bool matched;

    // Intermediate values
    Value gammaScaled;
    Value normalized;
    Value xMinusMean;
    Value rstd;
    Value variancePlusEps;
    Value variance;
    Value mean;
  };

  /** Match LayerNorm pattern starting from AddOp */
  static MatchResult match(AddOp addOp) {
    MatchResult result{nullptr, nullptr, nullptr, kLayerNormEpsilon, nullptr, false,  nullptr,
                       nullptr, nullptr, nullptr, nullptr,           nullptr, nullptr};
    // Step 1: Match the outermost AddOp (corresponds to + beta in LayerNorm)
    if (!matchAddBeta(addOp, result)) {
      MLOG(DEBUG) << "Failed to match AddBetaOp operation";
      return result;
    }
    // Step 2: Match MulOp (corresponds to * gamma in LayerNorm)
    if (!matchMulGamma(result)) {
      MLOG(DEBUG) << "Failed to match MulGammaOp operation";
      return result;
    }
    // Step 3: Match another MulOp (corresponds to * rstd in LayerNorm)
    if (!matchMulRstd(result)) {
      MLOG(DEBUG) << "Failed to match MulRstdOp operation";
      return result;
    }
    // Step 4: Match SubOp (corresponds to x - mean in LayerNorm)
    if (!matchSubXMean(result)) {
      MLOG(DEBUG) << "Failed to match SubOp operation";
      return result;
    }
    // Step 5: Match RsqrtOp (corresponds to 1/sqrt(var + eps) in LayerNorm)
    if (!matchRsqrt(result)) {
      MLOG(DEBUG) << "Failed to match RsqrtOp operation";
      return result;
    }
    // Step 6: Match variance + eps operation
    if (!matchVariancePlusEps(result)) {
      MLOG(DEBUG) << "Failed to match variance + eps operation";
      return result;
    }
    // Step 7: Match var_mean operation
    if (!matchVarMean(result)) {
      MLOG(DEBUG) << "Failed to match var_mean operation";
      return result;
    }
    // Step 8: Verify data dependency
    if (!verifyDataDependency(result)) {
      MLOG(DEBUG) << "Failed to verify data dependency";
      return result;
    }
    // Step 9: Verify all intermediate values have only one user
    if (!verifySingleUser(result)) {
      MLOG(DEBUG) << "Failed to verify all intermediate values have only one user";
      return result;
    }
    // Step 10: Verify gamma and beta shapes match normalized_shape
    if (!checkShapeMatchesNormalizedShape(result.gamma, result.normalized_shape)) {
      MLOG(DEBUG) << "Failed to verify gamma shape matches normalized_shape";
      return result;
    }
    if (!checkShapeMatchesNormalizedShape(result.beta, result.normalized_shape)) {
      MLOG(DEBUG) << "Failed to verify beta shape matches normalized_shape";
      return result;
    }
    result.matched = true;
    return result;
  }

 private:
  /** Match the outermost AddOp (corresponds to + beta in LayerNorm) */
  static bool matchAddBeta(AddOp addOp, MatchResult &result) {
    Value lhs = addOp.getX();
    Value rhs = addOp.getY();
    Operation *lhsOp = lhs.getDefiningOp();
    Operation *rhsOp = rhs.getDefiningOp();

    if (lhsOp && isa<MulOp>(lhsOp)) {
      result.gammaScaled = lhs;
      result.beta = rhs;
      return true;
    } else if (rhsOp && isa<MulOp>(rhsOp)) {
      result.gammaScaled = rhs;
      result.beta = lhs;
      return true;
    }
    return false;
  }

  /** Match MulOp (corresponds to * gamma in LayerNorm) */
  static bool matchMulGamma(MatchResult &result) {
    auto mul2 = result.gammaScaled.getDefiningOp<MulOp>();
    if (!mul2) {
      return false;
    }
    Value mul2Lhs = mul2.getLhs();
    Value mul2Rhs = mul2.getRhs();
    if (mul2Lhs.getDefiningOp<MulOp>()) {
      result.normalized = mul2Lhs;
      result.gamma = mul2Rhs;
      return true;
    } else if (mul2Rhs.getDefiningOp<MulOp>()) {
      result.normalized = mul2Rhs;
      result.gamma = mul2Lhs;
      return true;
    }
    return false;
  }

  /** Match another MulOp (corresponds to * rstd in LayerNorm) */
  static bool matchMulRstd(MatchResult &result) {
    auto mul1 = result.normalized.getDefiningOp<MulOp>();
    if (!mul1) {
      return false;
    }
    Value lhs = mul1.getLhs();
    Value rhs = mul1.getRhs();
    // Try both operand orders: (x - mean) * rstd or rstd * (x - mean)
    if (lhs.getDefiningOp<SubOp>()) {
      result.xMinusMean = lhs;
      result.rstd = rhs;
    } else if (rhs.getDefiningOp<SubOp>()) {
      result.xMinusMean = rhs;
      result.rstd = lhs;
    } else {
      return false;
    }
    return true;
  }

  /** Match SubOp (corresponds to x - mean in LayerNorm) */
  static bool matchSubXMean(MatchResult &result) { return result.xMinusMean.getDefiningOp<SubOp>() != nullptr; }

  /** Match RsqrtOp (corresponds to 1/sqrt(var + eps) in LayerNorm) */
  static bool matchRsqrt(MatchResult &result) {
    auto rsqrt = result.rstd.getDefiningOp<RsqrtOp>();
    if (!rsqrt) {
      return false;
    }
    result.variancePlusEps = rsqrt.getInput();
    return true;
  }

  /** Match variance + eps operation */
  static bool matchVariancePlusEps(MatchResult &result) {
    Operation *variancePlusEpsOp = result.variancePlusEps.getDefiningOp();
    if (auto addEps = mlir::dyn_cast<AddOp>(variancePlusEpsOp)) {
      if (extractEpsilonFromAdd(addEps, result.variance, result.eps)) {
        return true;
      }
    }
    return false;
  }

  /** Match var_mean operation */
  static bool matchVarMean(MatchResult &result) {
    auto varMean = result.variance.getDefiningOp<AclnnVarMeanOp>();
    if (!varMean) {
      return false;
    }
    result.x = varMean.getSelf();
    result.mean = varMean.getMeanOut();

    // Get the shape of input tensor x
    auto tensorType = mlir::dyn_cast<RankedTensorType>(result.x.getType());
    if (!tensorType) {
      return false;
    }
    const auto &shape = tensorType.getShape();

    // Collect normalized shape from x's shape at the specified dim positions
    llvm::SmallVector<int64_t> normalizedShapeValues;
    const size_t rank = shape.size();

    // Validate LayerNorm attributes first
    if (!varMean.getKeepdim() || varMean.getCorrection() != 0) {
      return false;  // LayerNorm requires keepdim=True and correction=0
    }

    auto dimAttr = varMean.getDim();

    if (!dimAttr || dimAttr.empty()) {
      normalizedShapeValues.assign(shape.begin(), shape.end());
    } else {
      llvm::SmallVector<int64_t> dims;
      const size_t numDims = dimAttr.size();
      dims.reserve(numDims);
      // Collect and validate dimensions
      for (auto attr : dimAttr.getValue()) {
        auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
        if (!intAttr) return false;  // Non-integer dimension

        int64_t dim = intAttr.getInt();
        int64_t actualDim = dim < 0 ? rank + dim : dim;
        if (actualDim < 0 || static_cast<size_t>(actualDim) >= rank) {
          return false;  // Invalid dimension
        }
        if (llvm::is_contained(dims, actualDim)) {
          return false;  // Duplicate dimension
        }
        dims.push_back(actualDim);
      }
      // Sort dimensions to ensure they are consecutive trailing dimensions
      std::sort(dims.begin(), dims.end());
      // Check for valid LayerNorm dimensions (must be consecutive trailing dimensions)
      const int64_t expectedStartDim = rank - numDims;
      for (size_t i = 0; i < numDims; ++i) {
        if (dims[i] != expectedStartDim + static_cast<int64_t>(i)) {
          return false;  // Not consecutive trailing dimensions
        }
      }
      // Collect normalized shape values
      std::transform(dims.begin(), dims.end(), std::back_inserter(normalizedShapeValues),
                     [&shape](int64_t dim) { return shape[dim]; });
    }

    // Create I64ArrayAttr from the collected shape values
    Builder builder(varMean.getContext());
    result.normalized_shape = builder.getI64ArrayAttr(normalizedShapeValues);
    return true;
  }

  /** Verify data dependency */
  static bool verifyDataDependency(const MatchResult &result) {
    auto sub = result.xMinusMean.getDefiningOp<SubOp>();
    return sub && sub.getX() == result.x && sub.getY() == result.mean;
  }

  /** Verify all intermediate values have only one user */
  static bool verifySingleUser(const MatchResult &result) {
    // Check intermediate values have only one user
    if (!hasOnlyOneUser(result.gammaScaled)) {
      return false;
    }
    if (!hasOnlyOneUser(result.normalized)) {
      return false;
    }
    if (!hasOnlyOneUser(result.xMinusMean)) {
      return false;
    }
    if (!hasOnlyOneUser(result.rstd)) {
      return false;
    }
    if (!hasOnlyOneUser(result.variancePlusEps)) {
      return false;
    }
    if (!hasOnlyOneUser(result.variance)) {
      return false;
    }
    if (!hasOnlyOneUser(result.mean)) {
      return false;
    }
    return true;
  }
};

/** FuseLayerNormPattern matches and fuses LayerNorm operation sequence */
class FuseLayerNormPattern : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    // Use LayerNormMatcher to check if the pattern matches
    auto matchResult = LayerNormMatcher::match(addOp);
    if (!matchResult.matched) {
      return failure();
    }
    MLOG(DEBUG) << "FuseLayerNormPattern: successfully matched LayerNorm pattern";
    // Create fused LayerNorm operation
    rewriter.setInsertionPoint(addOp);
    SmallVector<Type, 1> resultTypes = {addOp.getResult().getType()};
    auto layerNormOp =
      rewriter.create<AclnnLayerNormOp>(addOp.getLoc(), resultTypes, matchResult.x, matchResult.gamma, matchResult.beta,
                                        matchResult.normalized_shape, rewriter.getF64FloatAttr(matchResult.eps));
    // Replace original operation
    rewriter.replaceOp(addOp, layerNormOp.getYOut());
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseLayerNorm, FuseLayerNormPattern)

}  // namespace mfuse

}  // namespace mlir
