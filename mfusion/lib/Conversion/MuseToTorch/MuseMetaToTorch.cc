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

static void populateMuseMetaToTorchCustomPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertMuseCast>(converter, context);
  patterns.add<ConvertMuseReduceSum>(converter, context);
}

}  // namespace

void populateMuseMetaToTorchConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  populateMuseMetaToTorchCustomPatterns(converter, patterns);
}

}  // namespace mlir
