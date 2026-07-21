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

#include <limits>

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/ComputeOpBuilder.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/DecomposePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace mfuse {
namespace {
FailureOr<int64_t> getStaticReductionSize(RankedTensorType inputType, ArrayAttr dimensions) {
  int64_t reductionSize = 1;
  for (auto dimAttr : dimensions.getValue()) {
    int64_t dim = cast<IntegerAttr>(dimAttr).getInt();
    int64_t dimSize = inputType.getDimSize(dim);
    if (dimSize == ShapedType::kDynamic || dimSize <= 0) {
      return failure();
    }
    if (reductionSize > std::numeric_limits<int64_t>::max() / dimSize) {
      return failure();
    }
    reductionSize *= dimSize;
  }
  return reductionSize;
}

FailureOr<Value> materializePositiveIntScalarTensorConstant(PatternRewriter &rewriter, Location loc,
                                                            int64_t positiveIntValue) {
  if (positiveIntValue <= 0) {
    return failure();
  }
  auto scalarType = RankedTensorType::get({}, rewriter.getI64Type());
  auto elementAttr = rewriter.getI64IntegerAttr(positiveIntValue);
  auto denseAttr = DenseElementsAttr::get(scalarType, elementAttr);
  auto constantOp = mfuse::ConstantOp::materialize(rewriter, denseAttr, scalarType, loc);
  if (!constantOp) {
    return failure();
  }
  return constantOp.getResult();
}

static bool shouldComputeMeanInF32(Type elementType) {
  auto floatType = dyn_cast<FloatType>(elementType);
  return floatType && (floatType.isF16() || floatType.isBF16());
}

static RankedTensorType getSameShapeWithElementType(RankedTensorType tensorType, Type elementType) {
  return RankedTensorType::get(tensorType.getShape(), elementType, tensorType.getEncoding());
}
}  // namespace

/// OpRewritePattern for decomposing GELU operations
class GeluDecomposePattern : public OpRewritePattern<mfuse::AclnnGeluOp> {
 public:
  using OpRewritePattern<mfuse::AclnnGeluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnGeluOp geluOp, PatternRewriter &rewriter) const override {
    // Get the input tensor
    Value input = geluOp.getInput();
    Location loc = geluOp.getLoc();
    auto approximate = geluOp.getApproximateAttr().str();
    constexpr auto tanh_appro_str = "tanh";
    constexpr auto default_appro_str = "none";
    if (approximate != tanh_appro_str && approximate != default_appro_str) {
      return failure();
    }

    Type resultType = geluOp.getOutput().getType();
    auto tensorType = dyn_cast<mlir::RankedTensorType>(input.getType());

    // Create OpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Convert to float32 if not already
    if (!tensorType.getElementType().isF32()) {
      input = builder.cast(input, rewriter.getF32Type());
    }
    float kSqrtTwo = 1.41421356237309504880;
    float kTwoSqrtPI = 1.12837916709551257390;
    float kBeta = kSqrtTwo * kTwoSqrtPI;
    float kKappa = 0.044715f;
    auto x = builder.buildExpr(input);

    // y = -sqrt(8/pi) * (x + 0.044715 * x^3) sqrt(2/π) * 0.5
    auto y = kNegOne * kBeta * (x + kKappa * x * x * x);
    // Gelu(x) = x / (1 + exp(y))
    auto exp_y = builder.buildExpr(builder.exp(y.getValue()));
    auto result = x / (kOne + exp_y);
    Value output = result.getValue();
    auto resultTensorType = dyn_cast<mlir::RankedTensorType>(resultType);
    if (resultTensorType && !resultTensorType.getElementType().isF32()) {
      output = builder.cast(output, resultTensorType.getElementType());
    }
    rewriter.replaceOp(geluOp, output);

    return success();
  }
};

/// OpRewritePattern for decomposing Tanh operations
class TanhDecomposePattern : public OpRewritePattern<mfuse::AclnnTanhOp> {
 public:
  using OpRewritePattern<mfuse::AclnnTanhOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnTanhOp tanhOp, PatternRewriter &rewriter) const override {
    // Get the input tensor
    Value input = tanhOp.getInput();
    Location loc = tanhOp.getLoc();
    auto tensorType = dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!tensorType || !tensorType.getElementType().isF32()) {
      llvm::errs() << "Invalid input type for mfuse.tanh\n";
      return failure();
    }

    // Create OpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Tanh(x) = (exp(2x) - 1) / (exp(2x) + 1), clamp via maximum/minimum (not aclnn.clamp).
    auto x = builder.clampMinMax(input, kMinTanhClampValue, kMaxTanhClampValue);
    auto two_x = builder.mul(x, kTwo);
    auto exp_2x = builder.buildExpr(builder.exp(two_x));
    auto result = (exp_2x - kOne) / (exp_2x + kOne);
    rewriter.replaceOp(tanhOp, result.getValue());

    return success();
  }
};

/// OpRewritePattern for decomposing aclnn.clamp to maximum/minimum.
class ClampDecomposePattern : public OpRewritePattern<mfuse::AclnnClampOp> {
 public:
  using OpRewritePattern<mfuse::AclnnClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnClampOp clampOp, PatternRewriter &rewriter) const override {
    Value input = clampOp.getInput();
    Value minVal = clampOp.getMin();
    Value maxVal = clampOp.getMax();
    Location loc = clampOp.getLoc();
    auto resultType = clampOp.getResult().getType();
    Value afterMin = rewriter.create<mfuse::MaximumOp>(loc, resultType, input, minVal);
    Value output = rewriter.create<mfuse::MinimumOp>(loc, resultType, afterMin, maxVal);
    rewriter.replaceOp(clampOp, output);
    return success();
  }
};

/// OpRewritePattern for decomposing GELU Backward operations
class GeluBackwardDecomposePattern : public OpRewritePattern<mfuse::AclnnGeluBackwardOp> {
 public:
  using OpRewritePattern<mfuse::AclnnGeluBackwardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnGeluBackwardOp geluBackwardOp, PatternRewriter &rewriter) const override {
    // Get the input tensors
    Value grad = geluBackwardOp.getGradOutput();
    Value self = geluBackwardOp.getSelf();
    auto approximate = geluBackwardOp.getApproximateAttr().str();
    constexpr auto tanh_appro_str = "tanh";
    constexpr auto default_appro_str = "none";
    if (approximate != tanh_appro_str && approximate != default_appro_str) {
      return failure();
    }

    Location loc = geluBackwardOp.getLoc();
    Type resultType = geluBackwardOp.getOutput().getType();
    auto gradTensorType = dyn_cast<mlir::RankedTensorType>(grad.getType());
    auto selfTensorType = dyn_cast<mlir::RankedTensorType>(self.getType());
    // Check if input types are valid
    if (!gradTensorType || !selfTensorType) {
      return failure();
    }

    // Create OpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Convert to float32 if not already
    Value processedGrad = grad;
    Value processedSelf = self;
    if (!gradTensorType.getElementType().isF32()) {
      processedGrad = builder.cast(grad, rewriter.getF32Type());
    }
    if (!selfTensorType.getElementType().isF32()) {
      processedSelf = builder.cast(self, rewriter.getF32Type());
    }

    // sqrt(2.0 / pi)
    float kBeta = 0.7978845608028564;
    float kKappa = 0.044715;

    // Create Expr objects for easier operations
    auto x = builder.buildExpr(processedSelf);
    auto grad_expr = builder.buildExpr(processedGrad);

    // gelu_grad of dy and x is dy * y'
    // y' = 0.5 * (1.0 + tanh(tanh_para)) + 0.5 * x * (1.0 - tanh(tanh_para) * tanh(tanh_para)) * mul_right
    // tanh_para is 'sqrt(2.0 / pi) * (x + 0.044715 * x * x * x)'
    // mul_right is 'sqrt(2.0 / pi) * (1 + 3 * 0.044715 * x * x)'
    auto x_sq = x * x;
    auto tanh_para = kBeta * (x + kKappa * x_sq * x);
    auto tanh_para_val = builder.tanh(tanh_para.getValue());
    auto tanh_para_expr = builder.buildExpr(tanh_para_val);

    // 0.5 * (1.0 + tanh(tanh_para))
    auto left_derivative = kHalf * (tanh_para_expr + kOne);

    // 0.5 * x * (1.0 - tanh(tanh_para) * tanh(tanh_para)) * mul_right
    auto mul_right = kBeta * (kOne + kThree * kKappa * x_sq);
    auto right_derivative = kHalf * x * (kOne - tanh_para_expr * tanh_para_expr) * mul_right;
    auto out = grad_expr * (left_derivative + right_derivative);

    // Convert back to original type if needed
    Value finalResult = out.getValue();
    auto resultTensorType = dyn_cast<mlir::RankedTensorType>(resultType);
    if (resultTensorType && !resultTensorType.getElementType().isF32()) {
      finalResult = builder.cast(finalResult, resultTensorType.getElementType());
    }

    rewriter.replaceOp(geluBackwardOp, finalResult);

    return success();
  }
};

/// OpRewritePattern for decomposing Sigmoid operations
class SigmoidDecomposePattern : public OpRewritePattern<mfuse::AclnnSigmoidOp> {
 public:
  using OpRewritePattern<mfuse::AclnnSigmoidOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::AclnnSigmoidOp sigmoidOp, PatternRewriter &rewriter) const override {
    // Get the input tensor
    Value input = sigmoidOp.getInput();
    Location loc = sigmoidOp.getLoc();
    Type resultType = sigmoidOp.getResult().getType();
    auto tensorType = dyn_cast<mlir::RankedTensorType>(input.getType());

    // Check if input is a valid tensor type
    if (!tensorType) {
      return failure();
    }

    // Create OpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Convert to float32 if not already
    Value processedInput = input;
    if (!tensorType.getElementType().isF32()) {
      processedInput = builder.cast(input, rewriter.getF32Type());
    }

    // out = 1 / (1.0 + exp(-x))
    auto neg_x = builder.mul(processedInput, kNegOne);
    auto exp_neg_x = builder.exp(neg_x);
    auto denominator = builder.add(exp_neg_x, kOne);
    Value sigmoid_result = builder.reciprocal(denominator);

    // Convert back to original type if needed
    Value finalResult = sigmoid_result;
    auto resultTensorType = dyn_cast<mlir::RankedTensorType>(resultType);
    if (resultTensorType && !resultTensorType.getElementType().isF32()) {
      finalResult = builder.cast(sigmoid_result, resultTensorType.getElementType());
    }

    rewriter.replaceOp(sigmoidOp, finalResult);

    return success();
  }
};

LogicalResult decomposeReduceMean(ReduceMeanOp meanOp, PatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(meanOp.getInput().getType());
  auto resultType = dyn_cast<RankedTensorType>(meanOp.getResult().getType());
  if (!inputType || !resultType) {
    return failure();
  }

  auto reductionSizeOr = getStaticReductionSize(inputType, meanOp.getDimensions());
  if (failed(reductionSizeOr)) {
    return rewriter.notifyMatchFailure(meanOp, "reduced dimensions must be statically known and positive");
  }

  Type resultElementType = resultType.getElementType();
  auto floatElementType = dyn_cast<FloatType>(resultElementType);
  if (!floatElementType) {
    return rewriter.notifyMatchFailure(meanOp, "result element type must be floating point");
  }

  // Compute f16/bf16 means in f32 and cast back. Cast the input directly to
  // the compute type to avoid overflowing large values through f16 first.
  const bool computeInF32 = shouldComputeMeanInF32(resultElementType);
  Type computeElementType = computeInF32 ? rewriter.getF32Type() : resultElementType;
  Value reduceInput = meanOp.getInput();
  RankedTensorType sumResultType =
      computeInF32 ? getSameShapeWithElementType(resultType, computeElementType) : resultType;

  if (inputType.getElementType() != computeElementType) {
    RankedTensorType reduceInputType = getSameShapeWithElementType(inputType, computeElementType);
    reduceInput = rewriter.create<mfuse::CastOp>(meanOp.getLoc(), reduceInputType, reduceInput).getResult();
  }

  auto reduceSum = rewriter.create<mfuse::ReduceSumOp>(meanOp.getLoc(), sumResultType, reduceInput,
                                                       meanOp.getDimensions(), meanOp.getKeepdimAttr());
  auto divisorOr = materializePositiveIntScalarTensorConstant(rewriter, meanOp.getLoc(), *reductionSizeOr);
  if (failed(divisorOr)) {
    return rewriter.notifyMatchFailure(meanOp, "failed to materialize mean divisor constant");
  }

  auto mean = rewriter.create<mfuse::DivOp>(meanOp.getLoc(), sumResultType, reduceSum.getResult(), *divisorOr);
  Value output = mean.getResult();
  if (computeInF32) {
    output = rewriter.create<mfuse::CastOp>(meanOp.getLoc(), resultType, output).getResult();
  }
  rewriter.replaceOp(meanOp, output);
  return success();
}

class ReduceMeanDecomposePattern : public OpRewritePattern<mfuse::ReduceMeanOp> {
 public:
  using OpRewritePattern<mfuse::ReduceMeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::ReduceMeanOp meanOp, PatternRewriter &rewriter) const override {
    return decomposeReduceMean(meanOp, rewriter);
  }
};

/// OpRewritePattern for decomposing MatmulWithBias into matmul + add
class MatMulWithBiasDecomposePattern : public OpRewritePattern<mfuse::MatmulWithBiasOp> {
 public:
  using OpRewritePattern<mfuse::MatmulWithBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::MatmulWithBiasOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    Value other = op.getOther();
    Value bias = op.getBias();
    Type resultType = op.getResult().getType();

    auto matmulResult = rewriter.create<mfuse::MatmulOp>(loc, resultType, self, other,
                                                         op.getTransX1Attr(), op.getTransX2Attr());
    auto addResult = rewriter.create<mfuse::AddOp>(loc, resultType, matmulResult.getResult(), bias);
    rewriter.replaceOp(op, addResult.getResult());
    return success();
  }
};

/// Populate the given pattern set with decompose patterns.
/// When \p opList is empty, matmul_with_bias is registered only if
/// \p includeMatMulWithBiasByDefault is true (AFTER_MANUAL_FUSION passes false;
/// DVM adds it back via extra-op-list=matmul_with_bias).
void registerDecomposeMathOpPatterns(RewritePatternSet &patterns, const std::vector<std::string> &opList,
                                     bool includeMatMulWithBiasByDefault) {
  MLIRContext *ctx = patterns.getContext();

  // Map of operation names to their pattern registration functions
  std::map<std::string, PatternFunc> patternMap = {
    {"clamp", [](RewritePatternSet &p, MLIRContext *c) { p.add<ClampDecomposePattern>(c); }},
    {"gelu", [](RewritePatternSet &p, MLIRContext *c) { p.add<GeluDecomposePattern>(c); }},
    {"gelubackward", [](RewritePatternSet &p, MLIRContext *c) { p.add<GeluBackwardDecomposePattern>(c); }},
    {"reducemean", [](RewritePatternSet &p, MLIRContext *c) { p.add<ReduceMeanDecomposePattern>(c); }},
    {"tanh", [](RewritePatternSet &p, MLIRContext *c) { p.add<TanhDecomposePattern>(c); }},
    {"sigmoid", [](RewritePatternSet &p, MLIRContext *c) { p.add<SigmoidDecomposePattern>(c); }},
    {"matmulwithbias", [](RewritePatternSet &p, MLIRContext *c) { p.add<MatMulWithBiasDecomposePattern>(c); }}};

  if (opList.empty() && !includeMatMulWithBiasByDefault) {
    patternMap.erase("matmulwithbias");
  }

  // Register patterns using the common function
  registerPatternsByOpList(patterns, ctx, patternMap, opList);
}

}  // namespace mfuse
}  // namespace mlir
