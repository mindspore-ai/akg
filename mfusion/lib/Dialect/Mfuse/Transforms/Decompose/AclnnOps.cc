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

#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "mfusion/Dialect/Mfuse/Analysis/BinaryOpCommonInfer.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/VarianceUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/BinaryScalarUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/ComputeOpBuilder.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/DecomposePatterns.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mfuse {

namespace {
static bool isSupportFloatType(Type type) { return type.isF32() || type.isF16() || type.isBF16(); }

/// Check if output type is float/int type
static bool isSupportType(Type type) { return isSupportFloatType(type) || type.isInteger(32); }

static FailureOr<RankedTensorType> inferBinaryResultType(Type lhsType, Type rhsType) {
  auto lhsRankedType = dyn_cast<RankedTensorType>(lhsType);
  auto rhsRankedType = dyn_cast<RankedTensorType>(rhsType);
  if (!lhsRankedType || !rhsRankedType) {
    return failure();
  }
  auto inferredType = dyn_cast_or_null<RankedTensorType>(
    BinaryOpCommonInfer::inferResultType(lhsRankedType, rhsRankedType, false));
  if (!inferredType) {
    return failure();
  }
  return inferredType;
}

static bool hasSameShapeAndElementType(RankedTensorType lhsType, RankedTensorType rhsType) {
  return lhsType.getShape() == rhsType.getShape() && lhsType.getElementType() == rhsType.getElementType();
}

static bool canDecomposeAclnnWithAlpha(Value x, Value y, bool hasUnitAlpha, Type resultElementType) {
  if (isScalarType(x.getType()) && isScalarType(y.getType())) {
    return false;
  }
  if (hasUnitAlpha) {
    return true;
  }
  return !isScalarType(y.getType()) && isSupportType(resultElementType);
}

static FailureOr<RankedTensorType> inferAclnnWithAlphaResultType(Value x, Value y, Value alpha,
                                                                bool hasUnitAlpha) {
  if (hasUnitAlpha) {
    return inferBinaryResultType(x.getType(), y.getType());
  }
  auto inferredMulType = inferBinaryResultType(y.getType(), alpha.getType());
  if (failed(inferredMulType)) {
    return failure();
  }
  return inferBinaryResultType(x.getType(), *inferredMulType);
}

static Value buildAclnnWithAlphaResult(ComputeOpBuilder &builder, Value x, Value y, Value alpha, bool hasUnitAlpha,
                                       bool isAdd) {
  Value rhs = y;
  if (!hasUnitAlpha) {
    rhs = builder.mul(y, alpha);
  }
  if (!isAdd) {
    return builder.sub(x, rhs);
  }
  if (isScalarType(x.getType())) {
    return builder.add(rhs, x);
  }
  return builder.add(x, rhs);
}

/// Helper function to decompose Aclnn operations with alpha parameter
/// This handles the common logic for both AclnnAdd and AclnnSub
/// op The Aclnn operation (AclnnAddOp or AclnnSubOp)
/// rewriter PatternRewriter instance
/// return Whether the operation was decomposed
template <typename OpType>
static LogicalResult decomposeAclnnWithAlpha(OpType op, PatternRewriter &rewriter) {
  auto xScalar = getRecoverableScalar(op.getX());
  auto yScalar = getRecoverableScalar(op.getY());
  Value x = xScalar ? xScalar->scalar : op.getX();
  Value y = yScalar ? yScalar->scalar : op.getY();
  Value alpha = op.getAlpha();
  Type resultType = op.getResult().getType();
  auto resultElementType = mlir::cast<RankedTensorType>(resultType).getElementType();
  bool isAdd = isa<mfuse::AclnnAddOp>(op);
  bool hasUnitAlpha = isConstOne(alpha);

  if (!canDecomposeAclnnWithAlpha(x, y, hasUnitAlpha, resultElementType)) {
    return failure();
  }

  auto inferredResultType = inferAclnnWithAlphaResultType(x, y, alpha, hasUnitAlpha);
  auto expectedResultType = dyn_cast<RankedTensorType>(resultType);
  if (failed(inferredResultType) || !expectedResultType ||
      !hasSameShapeAndElementType(*inferredResultType, expectedResultType)) {
    return rewriter.notifyMatchFailure(op, "decomposed result type differs from the original result type");
  }

  mlir::mfuse::ComputeOpBuilder builder(rewriter, op.getLoc());
  Value result = buildAclnnWithAlphaResult(builder, x, y, alpha, hasUnitAlpha, isAdd);
  rewriter.replaceOp(op, result);
  eraseDeadNumToTensor(rewriter, xScalar);
  eraseDeadNumToTensor(rewriter, yScalar);
  return success();
}

}  // namespace

/// OpRewritePattern for decomposing AclnnAdd operations (x + y * alpha) into mul and add
class AclnnAddDecomposePattern : public OpRewritePattern<mfuse::AclnnAddOp> {
 public:
  using OpRewritePattern<mfuse::AclnnAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnAddOp addOp, PatternRewriter &rewriter) const override {
    return decomposeAclnnWithAlpha(addOp, rewriter);
  }
};

/// All aclnn matmul-like ops (mm/matmul/batch_matmul) use getOut().
template <typename Op>
static Value getMatmulResult(Op op) {
  return op.getOut();
}

//===----------------------------------------------------------------------===//
// ConvertAclnnMatmulToMatmul: aclnn.mm / aclnn.matmul / aclnn.batch_matmul =>
// mfuse.matmul (trans_x1/trans_x2 forwarded from aclnn)
//===----------------------------------------------------------------------===//

/// Template pattern to convert aclnn matmul-like ops (mm/matmul/batch_matmul)
/// to mfuse.matmul. This allows aclnn-specific operations to be converted to
/// generic mfuse operations for further fusion.
template <typename AclnnMatmulLikeOp>
class ConvertAclnnMatmulLikeToMatMulPattern : public OpRewritePattern<AclnnMatmulLikeOp> {
 public:
  using OpRewritePattern<AclnnMatmulLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AclnnMatmulLikeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value mat2 = op.getMat2();
    Type resultType = getMatmulResult(op).getType();
    auto newMatmul = rewriter.create<MatmulOp>(loc, resultType, self, mat2, rewriter.getBoolAttr(op.getTransX1()),
                                               rewriter.getBoolAttr(op.getTransX2()));
    rewriter.replaceOp(op, newMatmul.getResult());
    return success();
  }
};

using ConvertAclnnMmToMatMulPattern = ConvertAclnnMatmulLikeToMatMulPattern<AclnnMmOp>;
using ConvertAclnnMatmulToMatMulPattern = ConvertAclnnMatmulLikeToMatMulPattern<AclnnMatmulOp>;
using ConvertAclnnBatchMatmulToMatMulPattern = ConvertAclnnMatmulLikeToMatMulPattern<AclnnBatchMatmulOp>;

/// OpRewritePattern for decomposing AclnnSub operations (x - y * alpha) into mul and sub
class AclnnSubDecomposePattern : public OpRewritePattern<mfuse::AclnnSubOp> {
 public:
  using OpRewritePattern<mfuse::AclnnSubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnSubOp subOp, PatternRewriter &rewriter) const override {
    return decomposeAclnnWithAlpha(subOp, rewriter);
  }
};

/// Decompose mfuse.aclnn.var into meta ops (reuse sibling reduce_mean when present).
class AclnnVarDecomposePattern : public OpRewritePattern<mfuse::AclnnVarOp> {
 public:
  using OpRewritePattern<mfuse::AclnnVarOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnVarOp op, PatternRewriter &rewriter) const override {
    auto varOr = decomposeAclnnVar(op, rewriter);
    if (failed(varOr)) {
      return failure();
    }
    rewriter.replaceOp(op, *varOr);
    return success();
  }
};

/// Decompose mfuse.aclnn.var_mean after manual fusion when layer_norm fusion did not match.
class AclnnVarMeanDecomposePattern : public OpRewritePattern<mfuse::AclnnVarMeanOp> {
 public:
  using OpRewritePattern<mfuse::AclnnVarMeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnVarMeanOp op, PatternRewriter &rewriter) const override {
    auto resultsOr = decomposeAclnnVarMean(op, rewriter);
    if (failed(resultsOr)) {
      return failure();
    }
    rewriter.replaceOp(op, {resultsOr->first, resultsOr->second});
    return success();
  }
};

/// Register decomposition patterns for Aclnn operations
void registerAclnnDecomposePatterns(RewritePatternSet &patterns, const std::vector<std::string> &opList) {
  MLIRContext *ctx = patterns.getContext();

  // Map of operation names to their pattern registration functions
  std::map<std::string, PatternFunc> patternMap = {
    {"aclnnadd", [](RewritePatternSet &p, MLIRContext *c) { p.add<AclnnAddDecomposePattern>(c); }},
    {"aclnnsub", [](RewritePatternSet &p, MLIRContext *c) { p.add<AclnnSubDecomposePattern>(c); }},
    {"aclnnmm", [](RewritePatternSet &p, MLIRContext *c) { p.add<ConvertAclnnMmToMatMulPattern>(c); }},
    {"aclnnmatmul", [](RewritePatternSet &p, MLIRContext *c) { p.add<ConvertAclnnMatmulToMatMulPattern>(c); }},
    {"aclnnbatchmatmul",
     [](RewritePatternSet &p, MLIRContext *c) { p.add<ConvertAclnnBatchMatmulToMatMulPattern>(c); }}};

  // Register patterns using the common function
  registerPatternsByOpList(patterns, ctx, patternMap, opList);
}

void registerAclnnPostFusionDecomposePatterns(RewritePatternSet &patterns,
                                              const std::vector<std::string> &opList) {
  MLIRContext *ctx = patterns.getContext();
  std::map<std::string, PatternFunc> patternMap = {
    {"aclnnvar", [](RewritePatternSet &p, MLIRContext *c) { p.add<AclnnVarDecomposePattern>(c); }},
    {"aclnnvarmean", [](RewritePatternSet &p, MLIRContext *c) { p.add<AclnnVarMeanDecomposePattern>(c); }}};
  registerPatternsByOpList(patterns, ctx, patternMap, opList);
}

}  // namespace mfuse
}  // namespace mlir
