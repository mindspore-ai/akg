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

#include "mfusion/Conversion/MfuseToTorch/MfuseMetaToTorch.h"

#include <algorithm>
#include <iterator>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

namespace {

std::optional<int64_t> getTorchScalarTypeInt(mlir::Type type) {
  if (mlir::isa<mlir::NoneType, mlir::mfuse::NoneType, TorchD::NoneType>(type)) {
    return std::nullopt;
  }
  if (type.isSignlessInteger() && !type.isSignlessInteger(1)) {
    if (type.isSignlessInteger(8)) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Char);
    if (type.isSignlessInteger(16)) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Short);
    if (type.isSignlessInteger(32)) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Int);
    if (type.isSignlessInteger(64)) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Long);
    return std::nullopt;
  }
  if (type.isUnsignedInteger() && !type.isUnsignedInteger(8)) {
    return std::nullopt;
  }
  // Refer to: torch-mlir/Dialect/Torch/Utils/Utils.cpp
  auto scalarType = TorchD::getScalarTypeForType(type);
  return static_cast<int64_t>(scalarType);
}

mlir::FailureOr<mlir::Value> buildTorchDtypeValue(mlir::Type dtypeType, mlir::Location loc,
                                                  mlir::ConversionPatternRewriter &rewriter) {
  auto maybeDtypeInt = getTorchScalarTypeInt(dtypeType);
  if (!maybeDtypeInt) {
    return mlir::failure();
  }
  return rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(*maybeDtypeInt)).getResult();
}

mlir::Value buildTorchIntListFromI64ArrayAttr(mlir::ArrayAttr attr, mlir::Location loc,
                                              mlir::ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Value> values;
  values.reserve(attr.size());
  for (auto element : attr) {
    int64_t v = mlir::cast<mlir::IntegerAttr>(element).getInt();
    values.push_back(rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(v)));
  }
  return rewriter.create<TorchD::PrimListConstructOp>(
    loc, TorchD::ListType::get(rewriter.getContext(), TorchD::IntType::get(rewriter.getContext())), values);
}

// ============================================================================
// =   Please keep the patterns in alphabetical order by operator name   =
// ============================================================================

/// Default 2D convolution parameter values for torch.aten.convolution.
constexpr int64_t kConv2DStrideVal = 1;
constexpr int64_t kConv2DPaddingVal = 0;
constexpr int64_t kConv2DDilationVal = 1;
constexpr int64_t kConv2DOutputPaddingVal = 0;
constexpr int64_t kConv2DGroupsVal = 1;

/// Builds a torch list of two constant ints [a, b] for convolution spatial params.
static mlir::Value buildConvIntList2(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                                     int64_t a, int64_t b) {
  return buildTorchIntListFromI64ArrayAttr(rewriter.getI64ArrayAttr({a, b}), loc, rewriter);
}

/// Helper to build torch.aten.convolution with default stride/padding/dilation/groups.
/// stride=[1,1], padding=[0,0], dilation=[1,1], output_padding=[0,0], transposed=false, groups=1.
static mlir::Value buildAtenConvolution(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                                        mlir::Type resultType, mlir::Value input, mlir::Value weight,
                                        mlir::Value bias) {
  mlir::Value strideList = buildConvIntList2(rewriter, loc, kConv2DStrideVal, kConv2DStrideVal);
  mlir::Value paddingList = buildConvIntList2(rewriter, loc, kConv2DPaddingVal, kConv2DPaddingVal);
  mlir::Value dilationList = buildConvIntList2(rewriter, loc, kConv2DDilationVal, kConv2DDilationVal);
  mlir::Value outputPaddingList = buildConvIntList2(rewriter, loc, kConv2DOutputPaddingVal, kConv2DOutputPaddingVal);
  mlir::Value falseVal = rewriter.create<TorchD::ConstantBoolOp>(loc, false);
  mlir::Value groupsVal = rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(kConv2DGroupsVal));
  return rewriter
      .create<TorchD::AtenConvolutionOp>(loc, resultType, input, weight, bias, strideList, paddingList,
                                        dilationList, falseVal, outputPaddingList, groupsVal)
      .getResult();
}

/// Converts mfuse.conv2d -> torch.aten.convolution (with bias=None).
class ConvertMfuseConv2D : public mlir::OpConversionPattern<mlir::mfuse::Conv2DOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::Conv2DOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value input = adaptor.getInput();
    mlir::Value weight = adaptor.getWeight();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    mlir::Value noneBias = rewriter.create<TorchD::ConstantNoneOp>(op.getLoc());
    mlir::Value conv = buildAtenConvolution(rewriter, op.getLoc(), resultType, input, weight, noneBias);
    rewriter.replaceOp(op, conv);
    return mlir::success();
  }
};

/// Converts mfuse.conv2d_with_bias -> torch.aten.convolution (with bias operand).
/// This avoids emitting a separate torch.aten.add.Tensor after conv.
class ConvertMfuseConv2DWithBias : public mlir::OpConversionPattern<mlir::mfuse::Conv2DWithBiasOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::Conv2DWithBiasOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value input = adaptor.getInput();
    mlir::Value weight = adaptor.getWeight();
    mlir::Value bias = adaptor.getBias();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    mlir::Value conv = buildAtenConvolution(rewriter, op.getLoc(), resultType, input, weight, bias);
    rewriter.replaceOp(op, conv);
    return mlir::success();
  }
};

/// Converts mfuse.matmul -> torch.aten.mm (for 2D matrices) or torch.aten.matmul (for ND or transposed).
/// For simple 2D case (trans_x1=false, trans_x2=false), uses torch.aten.mm.
/// For other cases (ND or transposed), uses torch.aten.matmul.
class ConvertMfuseMatmul : public mlir::OpConversionPattern<mlir::mfuse::MatmulOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::MatmulOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value self = adaptor.getSelf();
    mlir::Value other = adaptor.getOther();
    bool transX1 = op.getTransX1();
    bool transX2 = op.getTransX2();

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }

    // Check if inputs are 2D and no transpose is needed
    auto selfType = mlir::dyn_cast<mlir::RankedTensorType>(self.getType());
    auto otherType = mlir::dyn_cast<mlir::RankedTensorType>(other.getType());

    // For simple 2D case without transpose, use torch.aten.mm
    if (!transX1 && !transX2 && selfType && otherType && selfType.getRank() == 2 && otherType.getRank() == 2) {
      rewriter.replaceOpWithNewOp<TorchD::AtenMmOp>(op, resultType, self, other);
      return mlir::success();
    }

    // For other cases (ND or transposed), use torch.aten.matmul
    // Note: torch.aten.matmul doesn't support transpose attributes directly,
    // so we would need to insert transpose ops if transX1 or transX2 is true.
    // For now, we'll use torch.aten.matmul and let downstream passes handle transpose.
    rewriter.replaceOpWithNewOp<TorchD::AtenMatmulOp>(op, resultType, self, other);
    return mlir::success();
  }
};

/// Converts mfuse.matmul_with_bias -> torch.aten.mm/matmul + torch.aten.add.Tensor.
/// Since torch.aten.mm/matmul don't support bias directly, we decompose it into
/// matmul followed by add.
class ConvertMfuseMatmulWithBias : public mlir::OpConversionPattern<mlir::mfuse::MatmulWithBiasOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::MatmulWithBiasOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value self = adaptor.getSelf();
    mlir::Value other = adaptor.getOther();
    mlir::Value bias = adaptor.getBias();
    bool transX1 = op.getTransX1();
    bool transX2 = op.getTransX2();

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return mlir::failure();
    }

    // Check if inputs are 2D and no transpose is needed
    auto selfType = mlir::dyn_cast<mlir::RankedTensorType>(self.getType());
    auto otherType = mlir::dyn_cast<mlir::RankedTensorType>(other.getType());

    mlir::Value matmulResult;
    // For simple 2D case without transpose, use torch.aten.mm
    if (!transX1 && !transX2 && selfType && otherType && selfType.getRank() == 2 && otherType.getRank() == 2) {
      matmulResult = rewriter.create<TorchD::AtenMmOp>(op.getLoc(), resultType, self, other);
    } else {
      // For other cases (ND or transposed), use torch.aten.matmul
      matmulResult = rewriter.create<TorchD::AtenMatmulOp>(op.getLoc(), resultType, self, other);
    }

    // Add bias: torch.aten.add.Tensor(matmul_result, bias, alpha=1)
    constexpr double kAlphaOne = 1.0;
    mlir::FloatAttr alphaAttr = rewriter.getFloatAttr(rewriter.getF64Type(), kAlphaOne);
    mlir::Value alphaOne = rewriter.create<TorchD::ConstantFloatOp>(op.getLoc(), alphaAttr);
    mlir::Value addResult =
      rewriter.create<TorchD::AtenAddTensorOp>(op.getLoc(), resultType, matmulResult, bias, alphaOne);

    rewriter.replaceOp(op, addResult);
    return mlir::success();
  }
};

struct ConvertMfuseCast : public mlir::OpConversionPattern<mlir::mfuse::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::CastOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value input = adaptor.getInput();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    mlir::Type dtypeType = op.getDtypeAttr().getValue();
    if (mlir::isa<mlir::NoneType, mlir::mfuse::NoneType, TorchD::NoneType>(dtypeType)) {
      return rewriter.notifyMatchFailure(op, "cast op requires a concrete dtype, but got NoneType");
    }
    auto dtypeValOrFailure = buildTorchDtypeValue(dtypeType, op.getLoc(), rewriter);
    if (mlir::failed(dtypeValOrFailure)) {
      return rewriter.notifyMatchFailure(op, "unsupported dtype for torch scalar type");
    }
    mlir::Value dtypeVal = *dtypeValOrFailure;

    mlir::Value nonBlockingVal = rewriter.create<TorchD::ConstantBoolOp>(op.getLoc(), false);
    mlir::Value copyVal = rewriter.create<TorchD::ConstantBoolOp>(op.getLoc(), false);
    mlir::Value memoryFormatVal = rewriter.create<TorchD::ConstantNoneOp>(op.getLoc());

    rewriter.replaceOpWithNewOp<TorchD::AtenToDtypeOp>(op, resultType, input, dtypeVal, nonBlockingVal, copyVal,
                                                       memoryFormatVal);
    return mlir::success();
  }
};

/// Converts mfuse.permute -> torch.aten.permute.
/// Performs minimal structural validation and relies on upstream passes for
/// semantic validity of perm values.
class ConvertMfusePermute : public mlir::OpConversionPattern<mlir::mfuse::PermuteOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::PermuteOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto permAttr = op.getPermAttr();
    if (!permAttr) {
      return rewriter.notifyMatchFailure(op, "perm attribute must be present");
    }

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    int64_t rank = inputType.getRank();
    auto permValues = permAttr.getValue();
    if (permValues.size() != static_cast<size_t>(rank)) {
      return rewriter.notifyMatchFailure(op, "perm size must match input rank");
    }

    // Minimal structural validation: all elements must be integer attributes.
    llvm::SmallVector<mlir::Value> permDims;
    permDims.reserve(permValues.size());
    for (auto attr : permValues) {
      auto dimAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      if (!dimAttr) {
        return rewriter.notifyMatchFailure(op, "perm values must be integers");
      }
      permDims.push_back(rewriter.create<TorchD::ConstantIntOp>(op.getLoc(), dimAttr));
    }

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) return mlir::failure();

    mlir::Value input = adaptor.getInput();
    auto listType = TorchD::ListType::get(op.getContext(), TorchD::IntType::get(op.getContext()));
    mlir::Value permList = rewriter.create<TorchD::PrimListConstructOp>(op.getLoc(), listType, permDims);
    rewriter.replaceOpWithNewOp<TorchD::AtenPermuteOp>(op, resultType, input, permList);
    return mlir::success();
  }
};

struct ConvertMfuseReduceSum : public mlir::OpConversionPattern<mlir::mfuse::ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::ReduceSumOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value input = adaptor.getInput();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    mlir::Value dimList = buildTorchIntListFromI64ArrayAttr(op.getDimensions(), op.getLoc(), rewriter);

    bool keepdim = op.getKeepdim();
    mlir::Value keepdimVal = rewriter.create<TorchD::ConstantBoolOp>(op.getLoc(), keepdim);

    mlir::Type dtypeType = op.getDtypeAttr().getValue();
    mlir::Value dtypeVal;
    if (mlir::isa<mlir::NoneType, mlir::mfuse::NoneType, TorchD::NoneType>(dtypeType)) {
      dtypeVal = rewriter.create<TorchD::ConstantNoneOp>(op.getLoc());
    } else {
      auto dtypeValOrFailure = buildTorchDtypeValue(dtypeType, op.getLoc(), rewriter);
      if (mlir::failed(dtypeValOrFailure)) {
        return rewriter.notifyMatchFailure(op, "unsupported dtype for torch scalar type");
      }
      dtypeVal = *dtypeValOrFailure;
    }

    rewriter.replaceOpWithNewOp<TorchD::AtenSumDimIntListOp>(op, resultType, input, dimList, keepdimVal, dtypeVal);
    return mlir::success();
  }
};

/// Converts mfuse.reshape -> torch.aten.reshape.
/// Shape is derived from reshape result type. A dynamic dim is mapped to -1.
class ConvertMfuseReshape : public mlir::OpConversionPattern<mlir::mfuse::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::ReshapeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> shapeValues;
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    }
    shapeValues.reserve(resultType.getShape().size());
    std::transform(
      resultType.getShape().begin(), resultType.getShape().end(), std::back_inserter(shapeValues), [&](int64_t d) {
        return rewriter.create<TorchD::ConstantIntOp>(op.getLoc(), d == mlir::ShapedType::kDynamic ? -1 : d);
      });

    mlir::Type torchResultType = getTypeConverter()->convertType(resultType);
    if (!torchResultType) return mlir::failure();

    mlir::Value input = adaptor.getInput();
    auto listType = TorchD::ListType::get(op.getContext(), TorchD::IntType::get(op.getContext()));
    auto shapeList = rewriter.create<TorchD::PrimListConstructOp>(op.getLoc(), listType, shapeValues);
    rewriter.replaceOpWithNewOp<TorchD::AtenReshapeOp>(op, torchResultType, input, shapeList);
    return mlir::success();
  }
};

// ============================================================================
// =   Please keep the patterns in alphabetical order by operator name   =
// ============================================================================

static void populateMfuseMetaToTorchCustomPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertMfuseCast>(converter, context);
  patterns.add<ConvertMfuseConv2D>(converter, context);
  patterns.add<ConvertMfuseConv2DWithBias>(converter, context);
  patterns.add<ConvertMfuseMatmul>(converter, context);
  patterns.add<ConvertMfuseMatmulWithBias>(converter, context);
  patterns.add<ConvertMfusePermute>(converter, context);
  patterns.add<ConvertMfuseReduceSum>(converter, context);
  patterns.add<ConvertMfuseReshape>(converter, context);
}

}  // namespace

void populateMfuseMetaToTorchConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  populateMfuseMetaToTorchCustomPatterns(converter, patterns);
}

}  // namespace mlir
