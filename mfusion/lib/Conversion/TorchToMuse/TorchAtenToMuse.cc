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

#include "mfusion/Conversion/TorchToMuse/TorchAtenToMuse.h"

#include <limits>
#include <numeric>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mfusion/Dialect/Muse/Muse.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// Custom conversion patterns
// (the pattern list should be alphabetically sorted)
//===----------------------------------------------------------------------===//

struct ConvertAtenDivTensorMode : public OpConversionPattern<TorchD::AtenDivTensorModeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenDivTensorModeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto outType = cast<mlir::muse::TensorType>(getTypeConverter()->convertType(op.getType()));

    std::string roundingMode;
    if (!matchPattern(op.getRoundingMode(), TorchD::m_TorchConstantStr(roundingMode)))
      return rewriter.notifyMatchFailure(op, "rounding_mode must be a constant string");

    int64_t mode = 0;
    if (roundingMode == "floor") {
      mode = 2;
    } else if (roundingMode == "trunc") {
      mode = 1;
    }

    auto modeValue = rewriter.create<mlir::muse::CreateI64Op>(op.getLoc(), mode);
    rewriter.replaceOpWithNewOp<mlir::muse::DivModOp>(op, outType, adaptor.getSelf(), adaptor.getOther(), modeValue);
    return success();
  }
};

struct ConvertAtenEmptyMemoryFormat : public OpConversionPattern<TorchD::AtenEmptyMemoryFormatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenEmptyMemoryFormatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Extract device - automatically converted from constant.device to constant.string
    // If device is None, create a default "cpu" string
    Value deviceValue = adaptor.getDevice();
    if (isa<mlir::muse::NoneType>(deviceValue.getType())) {
      deviceValue = rewriter.create<mlir::muse::CreateStringOp>(loc, rewriter.getStringAttr("cpu"));
    }

    // use the dtype from the result type
    auto outType = cast<mlir::muse::TensorType>(getTypeConverter()->convertType(op.getType()));
    auto dtypeValue = rewriter.create<mlir::muse::CreateDtypeOp>(loc, outType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::muse::EmptyOp>(op, adaptor.getSize(), dtypeValue, deviceValue);
    return success();
  }
};

struct ConvertAtenSliceScatter : public OpConversionPattern<TorchD::AtenSliceScatterOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenSliceScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    Value src = adaptor.getSrc();
    auto selfType = cast<mlir::muse::TensorType>(self.getType());
    int64_t rank = selfType.getRank();

    int64_t dim;
    if (!matchPattern(op.getDim(), TorchD::m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    dim = TorchD::toPositiveDim(dim, rank);
    if (!TorchD::isValidDim(dim, rank)) return rewriter.notifyMatchFailure(op, "dim out of range");

    int64_t step;
    if (!matchPattern(op.getStep(), TorchD::m_TorchConstantInt(&step)))
      return rewriter.notifyMatchFailure(op, "step must be constant");

    int64_t start = 0;
    if (!isa<TorchD::NoneType>(op.getStart().getType())) {
      if (!matchPattern(op.getStart(), TorchD::m_TorchConstantInt(&start)))
        return rewriter.notifyMatchFailure(op, "start must be constant");
    }

    int64_t end = std::numeric_limits<int64_t>::max();
    if (!isa<TorchD::NoneType>(op.getEnd().getType())) {
      if (!matchPattern(op.getEnd(), TorchD::m_TorchConstantInt(&end)))
        return rewriter.notifyMatchFailure(op, "end must be constant");
    }

    if (end == std::numeric_limits<int64_t>::max() && selfType.hasStaticShape()) {
      end = selfType.getDimSize(dim);
    }

    Value clonedSelf = rewriter.create<mlir::muse::CloneOp>(op.getLoc(), selfType, self);

    Value beginVal = rewriter.create<mlir::muse::CreateI64ArrayOp>(op.getLoc(), ArrayRef<int64_t>{start});
    Value endVal = rewriter.create<mlir::muse::CreateI64ArrayOp>(op.getLoc(), ArrayRef<int64_t>{end});
    Value stridesVal = rewriter.create<mlir::muse::CreateI64ArrayOp>(op.getLoc(), ArrayRef<int64_t>{step});
    Value axesVal = rewriter.create<mlir::muse::CreateI64ArrayOp>(op.getLoc(), ArrayRef<int64_t>{dim});

    rewriter.create<mlir::muse::StridedSliceAssignOp>(op.getLoc(), clonedSelf, src, beginVal, endVal, stridesVal,
                                                      axesVal);
    rewriter.replaceOp(op, clonedSelf);
    return success();
  }
};

struct ConvertAtenSumDimIntList : public OpConversionPattern<TorchD::AtenSumDimIntListOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenSumDimIntListOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = adaptor.getSelf();

    auto inType = cast<mlir::muse::TensorType>(self.getType());

    // convert dim to i64 array if dim is none or empty list
    Value dimsValue = adaptor.getDim();
    bool reduceAll = false;
    if (isa<mlir::muse::NoneType>(dimsValue.getType())) {
      reduceAll = true;
    } else if (auto makeList = dimsValue.getDefiningOp<mlir::muse::MakeListOp>()) {
      if (makeList.getElements().empty()) {
        reduceAll = true;
      }
    }
    if (reduceAll) {
      int64_t inputRank = inType.getRank();
      llvm::SmallVector<int64_t, 4> dims(inputRank);
      std::iota(dims.begin(), dims.end(), 0);
      dimsValue = rewriter.create<mlir::muse::CreateI64ArrayOp>(loc, dims);
    }

    Value keepdim = adaptor.getKeepdim();

    auto outType = cast<mlir::muse::TensorType>(getTypeConverter()->convertType(op.getType()));
    Value dtypeValue = rewriter.create<mlir::muse::CreateDtypeOp>(loc, outType.getElementType());

    rewriter.replaceOpWithNewOp<mlir::muse::ReduceSumOp>(op, outType, self, dimsValue, keepdim, dtypeValue);
    return success();
  }
};

struct ConvertAtenToDtype : public OpConversionPattern<TorchD::AtenToDtypeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenToDtypeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto outType = cast<mlir::muse::TensorType>(getTypeConverter()->convertType(op.getType()));

    auto dtypeValue = rewriter.create<mlir::muse::CreateDtypeOp>(op.getLoc(), outType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::muse::CastOp>(op, outType, self, dtypeValue);
    return success();
  }
};

struct ConvertAtenTransposeInt : public OpConversionPattern<TorchD::AtenTransposeIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenTransposeIntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    int64_t dim0, dim1;

    if (!matchPattern(op.getDim0(), TorchD::m_TorchConstantInt(&dim0)))
      return rewriter.notifyMatchFailure(op, "dim0 must be constant");
    if (!matchPattern(op.getDim1(), TorchD::m_TorchConstantInt(&dim1)))
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");

    auto inType = cast<mlir::muse::TensorType>(self.getType());
    int64_t inputRank = inType.getRank();
    auto outType = cast<mlir::muse::TensorType>(getTypeConverter()->convertType(op->getResult(0).getType()));

    dim0 = TorchD::toPositiveDim(dim0, inputRank);
    if (!TorchD::isValidDim(dim0, inputRank)) return rewriter.notifyMatchFailure(op, "dim0 out of range");

    dim1 = TorchD::toPositiveDim(dim1, inputRank);
    if (!TorchD::isValidDim(dim1, inputRank)) return rewriter.notifyMatchFailure(op, "dim1 out of range");

    llvm::SmallVector<int64_t, 4> permValues(inputRank);
    std::iota(std::begin(permValues), std::end(permValues), 0);
    std::swap(permValues[dim0], permValues[dim1]);

    auto permValue = rewriter.create<mlir::muse::CreateI64ArrayOp>(op.getLoc(), permValues);
    rewriter.replaceOpWithNewOp<mlir::muse::PermuteOp>(op, outType, self, permValue);
    return success();
  }
};

struct ConvertAtenZeros : public OpConversionPattern<TorchD::AtenZerosOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenZerosOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Extract device - automatically converted from constant.device to constant.string
    // If device is None, create a default "cpu" string
    Value deviceValue = adaptor.getDevice();
    if (isa<mlir::muse::NoneType>(deviceValue.getType())) {
      deviceValue = rewriter.create<mlir::muse::CreateStringOp>(loc, rewriter.getStringAttr("cpu"));
    }

    auto outType = cast<mlir::muse::TensorType>(getTypeConverter()->convertType(op.getType()));
    auto dtypeValue = rewriter.create<mlir::muse::CreateDtypeOp>(loc, outType.getElementType());

    rewriter.replaceOpWithNewOp<mlir::muse::ZerosOp>(op, adaptor.getSize(), dtypeValue, deviceValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

// Populate custom (hand-written) Aten ops to Muse conversion patterns
static void populateAtenToMuseCustomPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertAtenDivTensorMode>(converter, context);
  patterns.add<ConvertAtenEmptyMemoryFormat>(converter, context);
  patterns.add<ConvertAtenSliceScatter>(converter, context);
  patterns.add<ConvertAtenSumDimIntList>(converter, context);
  patterns.add<ConvertAtenToDtype>(converter, context);
  patterns.add<ConvertAtenTransposeInt>(converter, context);
  patterns.add<ConvertAtenZeros>(converter, context);
}

// Populate all Aten ops to Muse conversion patterns
void populateAtenToMuseConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  populateAtenToMuseCustomPatterns(converter, patterns);
}

}  // namespace mlir
