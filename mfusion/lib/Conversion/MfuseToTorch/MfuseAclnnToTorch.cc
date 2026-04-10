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
#include "llvm/ADT/StringRef.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

namespace {

static bool isDvmKernelGenerator(llvm::StringRef kernelGenerator) {
  return kernelGenerator == "dvm";
}

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
    mlir::Value alphaScalar = adaptor.getAlpha();

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }
    bool isRhsScalar = mlir::isa<TorchD::FloatType>(y.getType()) || mlir::isa<TorchD::IntType>(y.getType());
    if (isRhsScalar) {
      rewriter.replaceOpWithNewOp<TorchD::AtenAddScalarOp>(op, resultType, x, y, alphaScalar);
    } else {
      rewriter.replaceOpWithNewOp<TorchD::AtenAddTensorOp>(op, resultType, x, y, alphaScalar);
    }
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
    mlir::Value alphaScalar = adaptor.getAlpha();

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }
    bool isRhsScalar = mlir::isa<TorchD::FloatType>(y.getType()) || mlir::isa<TorchD::IntType>(y.getType());
    if (isRhsScalar) {
      rewriter.replaceOpWithNewOp<TorchD::AtenSubScalarOp>(op, resultType, x, y, alphaScalar);
    } else {
      rewriter.replaceOpWithNewOp<TorchD::AtenSubTensorOp>(op, resultType, x, y, alphaScalar);
    }
    return mlir::success();
  }
};

class ConvertMfuseAclnnClamp : public mlir::OpConversionPattern<mlir::mfuse::AclnnClampOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AclnnClampOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResult().getType(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert result types");
    }
    auto input = adaptor.getInput();
    auto inputType = input.getType();
    mlir::Value minValue = adaptor.getMin();
    mlir::Value maxValue = adaptor.getMax();
    Value final_result = input;
    auto inputValueTensorType = mlir::dyn_cast<TorchD::ValueTensorType>(inputType);
    auto boolResultType = TorchD::ValueTensorType::get(op.getResult().getType().getContext(),
                                                       inputValueTensorType.getSizes(), rewriter.getI1Type());
    if (!mlir::isa<TorchD::NoneType>(minValue.getType())) {
      auto min_isnan = rewriter.create<TorchD::AtenIsnanOp>(op.getLoc(), boolResultType, input);
      auto min_ge = rewriter.create<TorchD::AtenGeScalarOp>(op.getLoc(), boolResultType, input, minValue);
      auto min_condition =
        rewriter.create<TorchD::AtenBitwiseOrTensorOp>(op.getLoc(), boolResultType, min_ge, min_isnan);
      final_result =
        rewriter.create<TorchD::AtenWhereScalarOtherOp>(op.getLoc(), inputType, min_condition, input, minValue);
    }
    if (!mlir::isa<TorchD::NoneType>(maxValue.getType())) {
      auto max_isnan = rewriter.create<TorchD::AtenIsnanOp>(op.getLoc(), boolResultType, final_result);
      auto max_le = rewriter.create<TorchD::AtenLeScalarOp>(op.getLoc(), boolResultType, final_result, maxValue);
      auto max_condition =
        rewriter.create<TorchD::AtenBitwiseOrTensorOp>(op.getLoc(), boolResultType, max_le, max_isnan);
      final_result =
        rewriter.create<TorchD::AtenWhereScalarOtherOp>(op.getLoc(), inputType, max_condition, final_result, maxValue);
    }
    rewriter.replaceOp(op, final_result);
    return mlir::success();
  }
};

/// Converts mfuse.aclnn.mm -> torch.aten.mm. For kernel-generator dvm, trans_x1/trans_x2 are
/// preserved as discardable attrs; otherwise swap last two dims via torch.aten.permute.
class ConvertMfuseAclnnMm : public mlir::OpConversionPattern<mlir::mfuse::AclnnMmOp> {
 public:
  ConvertMfuseAclnnMm(mlir::TypeConverter &converter, mlir::MLIRContext *context,
                      llvm::StringRef kernelGenerator)
      : OpConversionPattern(converter, context), kernelGenerator_(kernelGenerator.str()) {}

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AclnnMmOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value self = adaptor.getSelf();
    mlir::Value mat2 = adaptor.getMat2();
    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }
    mlir::Location loc = op.getLoc();
    if (isDvmKernelGenerator(kernelGenerator_)) {
      auto newMm = rewriter.create<TorchD::AtenMmOp>(loc, resultType, self, mat2);
      newMm->setAttr("dvm_trans_a", rewriter.getBoolAttr(op.getTransX1()));
      newMm->setAttr("dvm_trans_b", rewriter.getBoolAttr(op.getTransX2()));
      rewriter.replaceOp(op, newMm.getResult());
      return mlir::success();
    }
    if (op.getTransX1()) {
      auto permOr = buildSwapLastTwoDimsPermute(loc, self, rewriter);
      if (mlir::failed(permOr)) {
        return mlir::failure();
      }
      self = *permOr;
    }
    if (op.getTransX2()) {
      auto permOr = buildSwapLastTwoDimsPermute(loc, mat2, rewriter);
      if (mlir::failed(permOr)) {
        return mlir::failure();
      }
      mat2 = *permOr;
    }
    rewriter.replaceOpWithNewOp<TorchD::AtenMmOp>(op, resultType, self, mat2);
    return mlir::success();
  }

 private:
  static mlir::FailureOr<mlir::Value> buildSwapLastTwoDimsPermute(mlir::Location loc, mlir::Value v,
                                                                mlir::ConversionPatternRewriter &rewriter) {
    auto vtt = mlir::dyn_cast<TorchD::ValueTensorType>(v.getType());
    if (!vtt || !vtt.hasSizes()) {
      return mlir::failure();
    }
    auto sizes = vtt.getSizes();
    int64_t rank = static_cast<int64_t>(sizes.size());
    if (rank < 2) {
      return mlir::failure();
    }
    llvm::SmallVector<int64_t> newSizes(sizes.begin(), sizes.end());
    std::swap(newSizes[rank - 2], newSizes[rank - 1]);
    mlir::Type permResultType = vtt.getWithSizesAndDtype(newSizes, vtt.getOptionalDtype());
    llvm::SmallVector<mlir::Value> permDims;
    permDims.reserve(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank - 2; ++i) {
      permDims.push_back(rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i)));
    }
    permDims.push_back(rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(rank - 1)));
    permDims.push_back(rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(rank - 2)));
    mlir::MLIRContext *ctx = rewriter.getContext();
    auto listType = TorchD::ListType::get(ctx, TorchD::IntType::get(ctx));
    mlir::Value permList = rewriter.create<TorchD::PrimListConstructOp>(loc, listType, permDims);
    return rewriter.create<TorchD::AtenPermuteOp>(loc, permResultType, v, permList).getResult();
  }

  std::string kernelGenerator_;
};

}  // namespace

// ============================================================================
// =   Please keep the patterns in alphabetical order by operator name   =
// ============================================================================

void populateMfuseAclnnToTorchConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns,
                                                 llvm::StringRef kernelGenerator) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertMfuseAclnnAdd>(converter, context);
  patterns.add<ConvertMfuseAclnnAddRmsNorm>(converter, context);
  patterns.add<ConvertMfuseAclnnClamp>(converter, context);
  patterns.add<ConvertMfuseAclnnGelu>(converter, context);
  patterns.add<ConvertMfuseAclnnGeluBackward>(converter, context);
  patterns.add<ConvertMfuseAclnnMm>(converter, context, kernelGenerator);
  patterns.add<ConvertMfuseAclnnRmsNorm>(converter, context);
  patterns.add<ConvertMfuseAclnnSub>(converter, context);
}

}  // namespace mlir
