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

#include "mfusion/Conversion/MfuseToTorch/MfuseAclnnToTorch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "mfusion/Conversion/MfuseToTorch/Utils.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

namespace {

// ============================================================================
// =   Please keep the patterns in alphabetical order by operator name   =
// ============================================================================

/// Converts mfuse.aclnn.add -> torch.aten.add.Tensor, materializing alpha (rank-0 tensor) as Torch scalar.
class ConvertMfuseAclnnAdd : public mlir::OpConversionPattern<mlir::mfuse::AclnnAddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AclnnAddOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value x = adaptor.getX();
    mlir::Value y = adaptor.getY();
    mlir::Value alphaScalar = materializeConstValueToTorchScalar(op.getOperation(), op.getAlpha(), rewriter);

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<TorchD::AtenAddTensorOp>(op, resultType, x, y, alphaScalar);
    return mlir::success();
  }
};

/// Converts mfuse.aclnn.add_rms_norm -> torch.npu.npu_add_rms_norm, materializing epsilon as Torch scalar.
class ConvertMfuseAclnnAddRmsNorm : public mlir::OpConversionPattern<mlir::mfuse::AclnnAddRmsNormOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AclnnAddRmsNormOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value x1 = adaptor.getX1();
    mlir::Value x2 = adaptor.getX2();
    mlir::Value gamma = adaptor.getGamma();
    // Create epsilon scalar as Torch constant directly from the existing FloatAttr
    mlir::Value epsilonScalar = rewriter.create<TorchD::ConstantFloatOp>(
      op.getLoc(), mlir::FloatAttr::get(mlir::Float64Type::get(rewriter.getContext()), op.getEpsilon()));
    // Get all result types
    mlir::SmallVector<mlir::Type> resultTypes;
    resultTypes.reserve(op.getNumResults());
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mlir::Type convertedType = getTypeConverter()->convertType(op.getResult(i).getType());
      if (!convertedType) {
        return mlir::failure();
      }
      resultTypes.push_back(convertedType);
    }

    // Create torch.operator for npu_add_rms_norm
    mlir::SmallVector<mlir::Value> operands = {x1, x2, gamma, epsilonScalar};
    unsigned numRegions = 0;
    rewriter.replaceOpWithNewOp<TorchD::OperatorOp>(
      op, resultTypes, rewriter.getStringAttr("torch.npu.npu_add_rms_norm"), operands, numRegions);
    return mlir::success();
  }
};

/// Converts mfuse.aclnn.rms_norm -> torch.npu.npu_rms_norm, materializing epsilon as Torch scalar.
class ConvertMfuseAclnnRmsNorm : public mlir::OpConversionPattern<mlir::mfuse::AclnnRmsNormOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AclnnRmsNormOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value x = adaptor.getX();
    mlir::Value gamma = adaptor.getGamma();
    double epsilonVal = op.getEpsilonAttr().getValueAsDouble();
    mlir::Value epsilonScalar = rewriter.create<TorchD::ConstantFloatOp>(
        op.getLoc(),
        mlir::FloatAttr::get(mlir::Float64Type::get(rewriter.getContext()), epsilonVal));
    mlir::SmallVector<mlir::Type> resultTypes;
    resultTypes.reserve(op.getNumResults());
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mlir::Type convertedType = getTypeConverter()->convertType(op.getResult(i).getType());
      if (!convertedType) {
        return rewriter.notifyMatchFailure(
            op, "failed to convert result type at index " + std::to_string(i));
      }
      resultTypes.push_back(convertedType);
    }
    mlir::SmallVector<mlir::Value> operands = {x, gamma, epsilonScalar};
    rewriter.replaceOpWithNewOp<TorchD::OperatorOp>(
        op, resultTypes, rewriter.getStringAttr("torch.npu.npu_rms_norm"), operands, 0);
    return mlir::success();
  }
};

/// Converts mfuse.aclnn.gelu -> torch.aten.gelu, materializing attributes as inputs.
class ConvertMfuseAclnnGelu : public mlir::OpConversionPattern<mlir::mfuse::AclnnGeluOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AclnnGeluOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value input = adaptor.getInput();

    // Convert result type
    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }

    // For torch.aten.gelu, the 'approximate' parameter is a string input
    // Create a string constant for 'approximate' parameter
    mlir::Value approximate =
      rewriter.create<TorchD::ConstantStrOp>(op.getLoc(), rewriter.getStringAttr(op.getApproximateAttr().str()));

    // Create torch.aten.gelu op with input and approximate parameter
    rewriter.replaceOpWithNewOp<TorchD::AtenGeluOp>(op, resultType, input, approximate);
    return mlir::success();
  }
};

/// Converts mfuse.aclnn.gelu_backward -> torch.aten.gelu_backward, materializing attributes as inputs.
class ConvertMfuseAclnnGeluBackward : public mlir::OpConversionPattern<mlir::mfuse::AclnnGeluBackwardOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AclnnGeluBackwardOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value gradOutput = adaptor.getGradOutput();
    mlir::Value self = adaptor.getSelf();

    // Convert result type
    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }

    // For torch.aten.gelu_backward, the 'approximate' parameter is a string input
    // Create a string constant for 'approximate' parameter
    mlir::Value approximate =
      rewriter.create<TorchD::ConstantStrOp>(op.getLoc(), rewriter.getStringAttr(op.getApproximateAttr().str()));

    // Create torch.aten.gelu_backward op with all inputs
    rewriter.replaceOpWithNewOp<TorchD::AtenGeluBackwardOp>(op, resultType, gradOutput, self, approximate);
    return mlir::success();
  }
};

/// Converts mfuse.aclnn.sub -> torch.aten.sub.Tensor, materializing alpha (rank-0 tensor) as Torch scalar.
class ConvertMfuseAclnnSub : public mlir::OpConversionPattern<mlir::mfuse::AclnnSubOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AclnnSubOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value x = adaptor.getX();
    mlir::Value y = adaptor.getY();
    mlir::Value alphaScalar = materializeConstValueToTorchScalar(op.getOperation(), op.getAlpha(), rewriter);

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<TorchD::AtenSubTensorOp>(op, resultType, x, y, alphaScalar);
    return mlir::success();
  }
};

}  // namespace

// ============================================================================
// =   Please keep the patterns in alphabetical order by operator name   =
// ============================================================================

void populateMfuseAclnnToTorchConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertMfuseAclnnAdd>(converter, context);
  patterns.add<ConvertMfuseAclnnAddRmsNorm>(converter, context);
  patterns.add<ConvertMfuseAclnnGelu>(converter, context);
  patterns.add<ConvertMfuseAclnnGeluBackward>(converter, context);
  patterns.add<ConvertMfuseAclnnRmsNorm>(converter, context);
  patterns.add<ConvertMfuseAclnnSub>(converter, context);
}

}  // namespace mlir
