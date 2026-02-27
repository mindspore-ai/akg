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

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/ComputeOpBuilder.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/DecomposePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace mfuse {
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
    Type float32Ty = mlir::RankedTensorType::get(tensorType.getShape(), rewriter.getF32Type());

    // Create OpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Convert to float32 if not already
    if (!tensorType.getElementType().isF32()) {
      input = builder.cast(input, float32Ty);
    }
    float kSqrtTwo = 1.41421356237309504880;
    float kTwoSqrtPI = 1.12837916709551257390;
    float kBeta = kSqrtTwo * kTwoSqrtPI;
    float kKappa = 0.044715f;
    auto x = builder.buildExpr(input, float32Ty);

    // y = -sqrt(8/pi) * (x + 0.044715 * x^3) sqrt(2/π) * 0.5
    auto y = kNegOne * kBeta * (x + kKappa * x * x * x);
    // Gelu(x) = x / (1 + exp(y))
    auto exp_y = builder.buildExpr(builder.exp(y.getValue(), float32Ty), float32Ty);
    auto result = x / (kOne + exp_y);
    Value output = result.getValue();
    if (resultType != float32Ty) {
      output = builder.cast(output, resultType);
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
    Type resultType = tanhOp.getOutput().getType();
    auto tensorType = dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!tensorType || !tensorType.getElementType().isF32()) {
      llvm::errs() << "Invalid input type for mfuse.tanh\n";
      return failure();
    }

    // Create OpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Tanh(x) = 1 - 2/(e^(2x) + 1)
    auto two_x = builder.mul(input, kTwo, resultType);
    auto exp_2x = builder.buildExpr(builder.exp(two_x, resultType), resultType);
    auto result = (kOne - exp_2x) / (kOne + exp_2x);

    // Replace the original Tanh operation with the decomposed computation
    rewriter.replaceOp(tanhOp, result.getValue());

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
    auto float32Ty = mlir::RankedTensorType::get(selfTensorType.getShape(), rewriter.getF32Type());

    // Create OpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Convert to float32 if not already
    Value processedGrad = grad;
    Value processedSelf = self;
    if (!gradTensorType.getElementType().isF32()) {
      processedGrad = builder.cast(grad, float32Ty);
    }
    if (!selfTensorType.getElementType().isF32()) {
      processedSelf = builder.cast(self, float32Ty);
    }

    // sqrt(2.0 / pi)
    float kBeta = 0.7978845608028564;
    float kKappa = 0.044715;

    // Create Expr objects for easier operations
    auto x = builder.buildExpr(processedSelf, float32Ty);
    auto grad_expr = builder.buildExpr(processedGrad, float32Ty);

    // gelu_grad of dy and x is dy * y'
    // y' = 0.5 * (1.0 + tanh(tanh_para)) + 0.5 * x * (1.0 - tanh(tanh_para) * tanh(para)) * mul_right
    // tanh_para is 'sqrt(2.0 / pi) * (x + 0.044715 * x * x * x)'
    // mul_right is 'sqrt(2.0 / pi) * (1 + 3 * 0.044715 * x * x)'
    auto x_sq = x * x;
    auto tanh_para = kBeta * (x + kKappa * x_sq * x);
    auto tanh_para_val = builder.tanh(tanh_para.getValue(), float32Ty);
    auto tanh_para_expr = builder.buildExpr(tanh_para_val, float32Ty);

    // 0.5 * (1.0 + tanh(tanh_para))
    auto left_derivative = kHalf * (tanh_para_expr + kOne);

    // 0.5 * x * (1.0 - tanh(tanh_para) * tanh(para)) * mul_right
    auto tanh_x = builder.tanh(x.getValue(), float32Ty);
    auto tanh_x_expr = builder.buildExpr(tanh_x, float32Ty);
    auto mul_right = kBeta * (kOne + kThree * kKappa * x_sq);
    auto right_derivative = kHalf * x * (kOne - tanh_para_expr * tanh_x_expr) * mul_right;
    auto out = grad_expr * (left_derivative + right_derivative);

    // Convert back to original type if needed
    Value finalResult = out.getValue();
    auto resultTensorType = dyn_cast<mlir::RankedTensorType>(resultType);
    if (resultTensorType && !resultTensorType.getElementType().isF32()) {
      finalResult = builder.cast(finalResult, resultType);
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

    auto float32Ty = mlir::RankedTensorType::get(tensorType.getShape(), rewriter.getF32Type());

    // Create OpBuilder instance
    mlir::mfuse::ComputeOpBuilder builder(rewriter, loc);

    // Convert to float32 if not already
    Value processedInput = input;
    if (!tensorType.getElementType().isF32()) {
      processedInput = builder.cast(input, float32Ty);
    }

    // out = 1 / (1.0 + exp(-x))
    auto neg_x = builder.mul(processedInput, kNegOne, float32Ty);
    auto exp_neg_x = builder.exp(neg_x, float32Ty);
    auto denominator = builder.add(exp_neg_x, kOne, float32Ty);
    Value sigmoid_result = builder.reciprocal(denominator, float32Ty);

    // Convert back to original type if needed
    Value finalResult = sigmoid_result;
    if (resultType != float32Ty) {
      finalResult = builder.cast(sigmoid_result, resultType);
    }

    rewriter.replaceOp(sigmoidOp, finalResult);

    return success();
  }
};

/// Populate the given pattern set with decompose patterns.
/// This function registers decompose patterns based on the provided op list.
void registerDecomposeMathOpPatterns(RewritePatternSet &patterns, const std::vector<std::string> &opList) {
  MLIRContext *ctx = patterns.getContext();

  // Map of operation names to their pattern registration functions
  std::map<std::string, PatternFunc> patternMap = {
    {"gelu", [](RewritePatternSet &p, MLIRContext *c) { p.add<GeluDecomposePattern>(c); }},
    {"gelubackward", [](RewritePatternSet &p, MLIRContext *c) { p.add<GeluBackwardDecomposePattern>(c); }},
    {"tanh", [](RewritePatternSet &p, MLIRContext *c) { p.add<TanhDecomposePattern>(c); }},
    {"sigmoid", [](RewritePatternSet &p, MLIRContext *c) { p.add<SigmoidDecomposePattern>(c); }}};

  // Register patterns using the common function
  registerPatternsByOpList(patterns, ctx, patternMap, opList);
}

}  // namespace mfuse
}  // namespace mlir
