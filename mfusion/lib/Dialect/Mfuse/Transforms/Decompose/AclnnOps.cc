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
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/VarianceUtils.h"
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

/// Helper function to decompose Aclnn operations with alpha parameter
/// This handles the common logic for both AclnnAdd and AclnnSub
/// op The Aclnn operation (AclnnAddOp or AclnnSubOp)
/// builder ComputeOpBuilder instance
/// rewriter PatternRewriter instance
/// return The decomposed operation result
template <typename OpType>
static Value decomposeAclnnWithAlpha(OpType op, mlir::mfuse::ComputeOpBuilder &builder, PatternRewriter &rewriter) {
  // Get the inputs from the op
  Value x = op.getX();
  Value y = op.getY();
  Value alpha = op.getAlpha();
  Type resultType = op.getResult().getType();
  auto resultElementType = mlir::cast<RankedTensorType>(resultType).getElementType();
  // Determine if the operation is an add or sub
  bool isAdd = isa<mfuse::AclnnAddOp>(op);
  if (isConstOne(alpha)) {
    return isAdd ? builder.add(x, y) : builder.sub(x, y);
  }

  // Do not decompose add/sub.Scalar to mul + add/sub
  if (auto rhsEnc = mlir::dyn_cast<mlir::RankedTensorType>(y.getType()).getEncoding()) {
    if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(rhsEnc)) {
      if (dictAttr.contains(mlir::mfuse::kScalarMarkerAttr)) {
        return nullptr;
      }
    }
  }

  if (!isSupportType(resultElementType)) {
    return nullptr;
  }

  Value mulResult = builder.mul(y, alpha);
  // Perform the final add or sub operation
  return isAdd ? builder.add(x, mulResult) : builder.sub(x, mulResult);
}

}  // namespace

/// OpRewritePattern for decomposing AclnnAdd operations (x + y * alpha) into mul and add
class AclnnAddDecomposePattern : public OpRewritePattern<mfuse::AclnnAddOp> {
 public:
  using OpRewritePattern<mfuse::AclnnAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnAddOp addOp, PatternRewriter &rewriter) const override {
    // Create ComputeOpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, addOp.getLoc());

    // Decompose using the helper function
    Value addResult = decomposeAclnnWithAlpha(addOp, builder, rewriter);
    if (!addResult) {
      return failure();
    }

    // Replace the original AclnnAdd operation with the decomposed computation
    rewriter.replaceOp(addOp, addResult);
    return success();
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
    // Create ComputeOpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, subOp.getLoc());

    // Decompose using the helper function
    Value subResult = decomposeAclnnWithAlpha(subOp, builder, rewriter);
    if (!subResult) {
      return failure();
    }

    // Replace the original AclnnSub operation with the decomposed computation
    rewriter.replaceOp(subOp, subResult);
    return success();
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
