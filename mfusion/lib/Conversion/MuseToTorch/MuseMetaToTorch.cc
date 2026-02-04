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

#include "mfusion/Conversion/MuseToTorch/MuseMetaToTorch.h"

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
#include "mfusion/Dialect/Muse/Muse.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

namespace {

std::optional<int64_t> getTorchScalarTypeInt(mlir::Type type) {
  if (mlir::isa<mlir::NoneType, mlir::muse::NoneType, TorchD::NoneType>(type)) {
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

struct ConvertMuseCast : public mlir::OpConversionPattern<mlir::muse::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::muse::CastOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value input = adaptor.getInput();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    mlir::Type dtypeType = op.getDtypeAttr().getValue();
    if (mlir::isa<mlir::NoneType, mlir::muse::NoneType, TorchD::NoneType>(dtypeType)) {
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

/// Converts muse.permute -> torch.aten.transpose.Int.
/// Handles permute operations that swap two dimensions (typically the last two).
class ConvertMusePermute : public mlir::OpConversionPattern<mlir::muse::PermuteOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::muse::PermuteOp op, OpAdaptor adaptor,
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

    // Extract permutation values
    llvm::SmallVector<int64_t> perm;
    for (auto attr : permValues) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        perm.push_back(intAttr.getInt());
      } else {
        return rewriter.notifyMatchFailure(op, "perm values must be integers");
      }
    }

    // Check if this is a simple two-dimension swap (identity except for swapping two dims).
    // Find the two dimensions that are swapped.
    int64_t dim0 = -1, dim1 = -1;
    for (int64_t i = 0; i < rank; ++i) {
      if (perm[i] != i) {
        if (dim0 == -1) {
          dim0 = i;
        } else if (dim1 == -1) {
          dim1 = i;
        } else {
          // More than two dimensions are swapped, cannot use transpose.Int
          return rewriter.notifyMatchFailure(op, "permute swaps more than two dimensions");
        }
      }
    }

    if (dim0 == -1 || dim1 == -1) {
      // Identity permutation, no conversion needed (should be eliminated by canonicalize)
      return rewriter.notifyMatchFailure(op, "identity permutation");
    }

    // Verify that perm[dim0] == dim1 and perm[dim1] == dim0 (swapped)
    if (perm[dim0] != dim1 || perm[dim1] != dim0) {
      return rewriter.notifyMatchFailure(op, "permute is not a simple two-dimension swap");
    }

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) return mlir::failure();

    mlir::Value input = adaptor.getInput();
    auto dim0Attr = rewriter.getI64IntegerAttr(dim0);
    auto dim1Attr = rewriter.getI64IntegerAttr(dim1);
    auto dim0Const = rewriter.create<TorchD::ConstantIntOp>(op.getLoc(), dim0Attr);
    auto dim1Const = rewriter.create<TorchD::ConstantIntOp>(op.getLoc(), dim1Attr);

    rewriter.replaceOpWithNewOp<TorchD::AtenTransposeIntOp>(op, resultType, input, dim0Const, dim1Const);
    return mlir::success();
  }
};

struct ConvertMuseReduceSum : public mlir::OpConversionPattern<mlir::muse::ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::muse::ReduceSumOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value input = adaptor.getInput();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    mlir::Value dimList = buildTorchIntListFromI64ArrayAttr(op.getDimensions(), op.getLoc(), rewriter);

    bool keepdim = op.getKeepdim();
    mlir::Value keepdimVal = rewriter.create<TorchD::ConstantBoolOp>(op.getLoc(), keepdim);

    mlir::Type dtypeType = op.getDtypeAttr().getValue();
    mlir::Value dtypeVal;
    if (mlir::isa<mlir::NoneType, mlir::muse::NoneType, TorchD::NoneType>(dtypeType)) {
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

/// Converts muse.reshape -> torch.aten.view.
/// Shape is a Value (1D tensor of i64); if constant, extract dims and build list for view.
class ConvertMuseReshape : public mlir::OpConversionPattern<mlir::muse::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::muse::ReshapeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value shapeVal = adaptor.getShape();
    llvm::SmallVector<mlir::Value> shapeValues;

    auto constOp = shapeVal.getDefiningOp<mlir::arith::ConstantOp>();
    if (constOp) {
      auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
      if (denseAttr && denseAttr.getElementType().isInteger(64)) {
        for (auto apInt : denseAttr.getValues<mlir::APInt>()) {
          shapeValues.push_back(rewriter.create<TorchD::ConstantIntOp>(op.getLoc(), apInt.getSExtValue()));
        }
      }
    }
    if (shapeValues.empty()) {
      return rewriter.notifyMatchFailure(op, "shape must be a constant 1D i64 tensor");
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    }

    mlir::Type torchResultType = getTypeConverter()->convertType(resultType);
    if (!torchResultType) return mlir::failure();

    mlir::Value input = adaptor.getInput();
    auto listType = TorchD::ListType::get(op.getContext(), TorchD::IntType::get(op.getContext()));
    auto shapeList = rewriter.create<TorchD::PrimListConstructOp>(op.getLoc(), listType, shapeValues);
    rewriter.replaceOpWithNewOp<TorchD::AtenViewOp>(op, torchResultType, input, shapeList);
    return mlir::success();
  }
};

// ============================================================================
// =   Please keep the patterns in alphabetical order by operator name   =
// ============================================================================

static void populateMuseMetaToTorchCustomPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertMuseCast>(converter, context);
  patterns.add<ConvertMusePermute>(converter, context);
  patterns.add<ConvertMuseReduceSum>(converter, context);
  patterns.add<ConvertMuseReshape>(converter, context);
}

}  // namespace

void populateMuseMetaToTorchConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  populateMuseMetaToTorchCustomPatterns(converter, patterns);
}

}  // namespace mlir
