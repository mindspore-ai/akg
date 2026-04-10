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
/// Helper function to convert AnyScalar to RankedTensorType (0D tensor)
static Value convertAnyScalarToRankedTensor(PatternRewriter &rewriter, Location loc, Value scalar) {
  Type scalarType = scalar.getType();
  // If already a RankedTensorType, return as is
  if (isa<RankedTensorType>(scalarType)) {
    return scalar;
  }
  // For other scalar types (FloatType, IntegerType, IndexType), create a 0D tensor
  Type elementType = scalarType;
  return rewriter.create<mlir::mfuse::CastOp>(loc, scalar, elementType);
}

/// Helper function to create alphaValue from DenseElementsAttr
static Value createAlphaValueFromConstant(PatternRewriter &rewriter, Location loc, DenseElementsAttr denseAttr,
                                          RankedTensorType targetType) {
  auto elementType = denseAttr.getElementType();
  if (isa<FloatType>(elementType)) {
    auto floatVal = denseAttr.getSplatValue<APFloat>();
    auto floatAttr = rewriter.getFloatAttr(targetType.getElementType(), floatVal.convertToDouble());
    return rewriter.create<mfuse::ConstantOp>(loc, targetType, DenseElementsAttr::get(targetType, floatAttr));
  } else {
    auto intVal = denseAttr.getSplatValue<APInt>();
    auto intAttr = rewriter.getIntegerAttr(targetType.getElementType(), intVal.getSExtValue());
    return rewriter.create<mfuse::ConstantOp>(loc, targetType, DenseElementsAttr::get(targetType, intAttr));
  }
}

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
  Location loc = op.getLoc();
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

  // Check if alpha is a constant
  auto constantOp = alpha.getDefiningOp<mlir::mfuse::ConstantOp>();
  Value mulResult;

  if (constantOp) {
    // If alpha is a constant, extract its value
    auto denseTensor = mlir::dyn_cast<DenseElementsAttr>(constantOp.getValue());
    auto supposedAlphaType = RankedTensorType::get({}, resultElementType);
    Value alphaValue = createAlphaValueFromConstant(rewriter, loc, denseTensor, supposedAlphaType);
    mulResult = builder.mul(y, alphaValue);
  } else {
    // If alpha is dynamic, convert it to RankedTensorType (0D tensor)
    Value alphaTensor = convertAnyScalarToRankedTensor(rewriter, loc, alpha);
    mulResult = builder.mul(y, alphaTensor);
  }

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

/// Read trans_x1/trans_x2 from an aclnn matmul-like op (mm, matmul, batch_matmul).
/// Returns (false, false) if \p op is not one of these (caller should only pass matched ops).
static std::pair<bool, bool> getAclnnMatmulTransposeFlags(Operation *op) {
  if (auto mm = dyn_cast<AclnnMmOp>(op)) {
    return {mm.getTransX1(), mm.getTransX2()};
  }
  if (auto m = dyn_cast<AclnnMatmulOp>(op)) {
    return {m.getTransX1(), m.getTransX2()};
  }
  if (auto bm = dyn_cast<AclnnBatchMatmulOp>(op)) {
    return {bm.getTransX1(), bm.getTransX2()};
  }
  return {false, false};
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

    matmulOp = findAclnnMatmulFromAddOperands<AclnnMmOp>(x, y, bias);
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
    const auto transFlags = getAclnnMatmulTransposeFlags(matmulOp);
    auto newOp = rewriter.create<MatmulWithBiasOp>(loc, resultType, self, other, bias,
                                                   rewriter.getBoolAttr(transFlags.first),
                                                   rewriter.getBoolAttr(transFlags.second));
    rewriter.replaceOp(addOp, newOp.getResult());
    return success();
  }
};

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
