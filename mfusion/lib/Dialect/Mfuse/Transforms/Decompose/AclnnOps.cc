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

#include "llvm/Support/FormatVariadic.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/ComputeOpBuilder.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/DecomposePatterns.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"
#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mfuse {

namespace {
/// Helper function to convert AnyScalar to RankedTensorType (0D tensor)
static Value convertAnyScalarToRankedTensor(PatternRewriter &rewriter, Location loc, Value scalar) {
  Type scalarType = scalar.getType();
  // If already a RankedTensorType, return as is
  if (isa<RankedTensorType>(scalarType)) {
    return scalar;
  }
  // For other scalar types (FloatType, IntegerType, IndexType), create a 0D tensor
  Type elementType = scalarType;
  auto tensorType = RankedTensorType::get({}, elementType);
  // Create a CastOp to convert the scalar to 0D tensor
  return rewriter.create<CastOp>(loc, tensorType, scalar);
}
}  // namespace

/// OpRewritePattern for decomposing AclnnAdd operations (x + y * alpha) into mul and add
class AclnnAddDecomposePattern : public OpRewritePattern<mfuse::AclnnAddOp> {
 public:
  using OpRewritePattern<mfuse::AclnnAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnAddOp addOp, PatternRewriter &rewriter) const override {
    // Get the inputs
    Value x = addOp.getX();
    Value y = addOp.getY();
    Value alpha = addOp.getAlpha();
    Location loc = addOp.getLoc();
    Type resultType = addOp.getResult().getType();

    // Create ComputeOpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Check if alpha is 1.0, if so, skip the mul operation
    Value addResult;
    if (isConstOne(alpha)) {
      // If alpha is 1.0, just add x and y
      addResult = builder.add(x, y, resultType);
    } else {
      // Check if alpha is a constant
      auto constantOp = alpha.getDefiningOp<mlir::arith::ConstantOp>();
      if (constantOp) {
        // If alpha is a constant, extract its value
        auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(constantOp.getValue());
        auto floatValue = denseAttr.getSplatValue<float>();
        // Decompose AclnnAdd (x + y * alpha) into mul and add
        // Ensure alpha is float32
        Value mulResult = builder.mul(y, floatValue, y.getType());
        addResult = builder.add(x, mulResult, resultType);
      } else {
        // If alpha is not a constant, convert it to RankedTensorType and use directly
        Value alphaTensor = convertAnyScalarToRankedTensor(rewriter, loc, alpha);
        // Decompose AclnnAdd (x + y * alpha) into mul and add
        Value mulResult = builder.mul(y, alphaTensor, y.getType());
        addResult = builder.add(x, mulResult, resultType);
      }
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

/// Try to find aclnn matmul-like op (mm/matmul/batch_matmul) from add operands.
/// Returns the matmul op and sets bias to the other operand.
template <typename AclnnMatmulOp>
static Operation *findAclnnMatmulFromAddOperands(Value x, Value y, Value &bias) {
  if (auto mm = x.getDefiningOp<AclnnMatmulOp>()) {
    bias = y;
    return mm.getOperation();
  }
  if (auto mm = y.getDefiningOp<AclnnMatmulOp>()) {
    bias = x;
    return mm.getOperation();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConvertAclnnMatmulToMatmul: aclnn.mm / aclnn.matmul / aclnn.batch_matmul =>
// mfuse.matmul (trans_x1=false, trans_x2=false)
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
    auto newMatmul =
      rewriter.create<MatmulOp>(loc, resultType, self, mat2, rewriter.getBoolAttr(false), rewriter.getBoolAttr(false));
    rewriter.replaceOp(op, newMatmul.getResult());
    return success();
  }
};

using ConvertAclnnMmToMatMulPattern = ConvertAclnnMatmulLikeToMatMulPattern<AclnnMmOp>;
using ConvertAclnnMatmulToMatMulPattern = ConvertAclnnMatmulLikeToMatMulPattern<AclnnMatmulOp>;
using ConvertAclnnBatchMatmulToMatMulPattern = ConvertAclnnMatmulLikeToMatMulPattern<AclnnBatchMatmulOp>;

/// Pattern to fuse aclnn.matmul + add into mfuse.matmul_with_bias.
/// Converts aclnn.mm/matmul/batch_matmul + add to mfuse.matmul_with_bias.
/// Higher benefit pattern, so it runs before converting standalone matmul operations.
/// Processed after AclnnAddDecomposePattern.
class ConvertAclnnMatmulAddToMatMulWithBiasPattern : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    Value x = addOp.getX();
    Value y = addOp.getY();
    Value bias;
    Operation *matmulOp = nullptr;

    if (!matmulOp) {
      matmulOp = findAclnnMatmulFromAddOperands<AclnnMmOp>(x, y, bias);
    }
    if (!matmulOp) {
      matmulOp = findAclnnMatmulFromAddOperands<AclnnMatmulOp>(x, y, bias);
    }
    if (!matmulOp) {
      matmulOp = findAclnnMatmulFromAddOperands<AclnnBatchMatmulOp>(x, y, bias);
    }
    if (!matmulOp) {
      return failure();
    }

    Value self = matmulOp->getOperand(0);
    Value other = matmulOp->getOperand(1);
    Type resultType = addOp.getResult().getType();
    Location loc = addOp.getLoc();
    auto newOp = rewriter.create<MatmulWithBiasOp>(loc, resultType, self, other, bias, rewriter.getBoolAttr(false),
                                                   rewriter.getBoolAttr(false));
    rewriter.replaceOp(addOp, newOp.getResult());
    return success();
  }
};

/// OpRewritePattern for decomposing AclnnSub operations (x - y * alpha) into mul and sub
class AclnnSubDecomposePattern : public OpRewritePattern<mfuse::AclnnSubOp> {
 public:
  using OpRewritePattern<mfuse::AclnnSubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnSubOp subOp, PatternRewriter &rewriter) const override {
    // Get the inputs
    Value x = subOp.getX();
    Value y = subOp.getY();
    Value alpha = subOp.getAlpha();
    Location loc = subOp.getLoc();
    Type resultType = subOp.getResult().getType();

    // Create ComputeOpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Check if alpha is 1.0, if so, skip the mul operation
    Value subResult;
    if (isConstOne(alpha)) {
      // If alpha is 1.0, just subtract x and y
      subResult = builder.sub(x, y, resultType);
    } else {
      // Check if alpha is a constant
      auto constantOp = alpha.getDefiningOp<mlir::arith::ConstantOp>();
      if (constantOp) {
        // If alpha is a constant, extract its value
        auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(constantOp.getValue());
        auto floatValue = denseAttr.getSplatValue<float>();
        // Decompose AclnnSub (x - y * alpha) into mul and sub
        // Ensure alpha is float32
        Value mulResult = builder.mul(y, floatValue, y.getType());
        subResult = builder.sub(x, mulResult, resultType);
      } else {
        // If alpha is not a constant, convert it to RankedTensorType and use directly
        Value alphaTensor = convertAnyScalarToRankedTensor(rewriter, loc, alpha);
        // Decompose AclnnSub (x - y * alpha) into mul and sub
        Value mulResult = builder.mul(y, alphaTensor, y.getType());
        subResult = builder.sub(x, mulResult, resultType);
      }
    }

    // Replace the original AclnnSub operation with the decomposed computation
    rewriter.replaceOp(subOp, subResult);
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
    {"aclnnmatmuladd",
     [](RewritePatternSet &p, MLIRContext *c) { p.add<ConvertAclnnMatmulAddToMatMulWithBiasPattern>(c); }},
    {"aclnnmm", [](RewritePatternSet &p, MLIRContext *c) { p.add<ConvertAclnnMmToMatMulPattern>(c); }},
    {"aclnnmatmul", [](RewritePatternSet &p, MLIRContext *c) { p.add<ConvertAclnnMatmulToMatMulPattern>(c); }},
    {"aclnnbatchmatmul",
     [](RewritePatternSet &p, MLIRContext *c) { p.add<ConvertAclnnBatchMatmulToMatMulPattern>(c); }}};

  // Register patterns using the common function
  registerPatternsByOpList(patterns, ctx, patternMap, opList);
}

}  // namespace mfuse
}  // namespace mlir
