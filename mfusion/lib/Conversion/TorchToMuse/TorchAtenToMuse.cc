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

#include <numeric>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mfusion/Dialect/Muse/Muse.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// Aten ops to Muse conversion patterns
// (the pattern list should be alphabetically sorted)
//===----------------------------------------------------------------------===//

struct ConvertAtenSumDimIntList : public OpConversionPattern<TorchD::AtenSumDimIntListOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenSumDimIntListOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();

    auto inType = cast<RankedTensorType>(self.getType());

    // convert dim to i64 array if dim is none or empty list
    Value dimsValue = op.getDim();
    bool reduceAll = false;
    if (isa<TorchD::NoneType>(dimsValue.getType())) {
      reduceAll = true;
    }

    llvm::SmallVector<int64_t, 4> dims;
    if (!reduceAll) {
      llvm::SmallVector<Value, 4> dimValues;
      if (!TorchD::getListConstructElements(dimsValue, dimValues)) {
        return rewriter.notifyMatchFailure(op, "dim must come from list construct");
      }
      if (dimValues.empty()) {
        reduceAll = true;
      } else {
        for (Value dimValue : dimValues) {
          int64_t dim = 0;
          if (!matchPattern(dimValue, TorchD::m_TorchConstantInt(&dim))) {
            return rewriter.notifyMatchFailure(op, "dim list must be constant ints");
          }
          dims.push_back(dim);
        }
      }
    }
    if (reduceAll) {
      int64_t inputRank = inType.getRank();
      dims.resize(inputRank);
      std::iota(dims.begin(), dims.end(), 0);
    }

    bool keepdimValue = false;
    if (!matchPattern(op.getKeepdim(), TorchD::m_TorchConstantBool(&keepdimValue))) {
      return rewriter.notifyMatchFailure(op, "keepdim must be constant bool");
    }

    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    auto dimsAttr = rewriter.getI64ArrayAttr(dims);
    auto keepdimAttr = rewriter.getBoolAttr(keepdimValue);
    auto dtypeAttr = mlir::TypeAttr::get(outType.getElementType());

    rewriter.replaceOpWithNewOp<mlir::muse::ReduceSumOp>(op, outType, self, dimsAttr, keepdimAttr, dtypeAttr);
    return success();
  }
};

struct ConvertAtenToDtype : public OpConversionPattern<TorchD::AtenToDtypeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenToDtypeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    auto dtypeAttr = mlir::TypeAttr::get(outType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::muse::CastOp>(op, outType, self, dtypeAttr);
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

    auto inType = cast<RankedTensorType>(self.getType());
    int64_t inputRank = inType.getRank();
    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op->getResult(0).getType()));

    dim0 = TorchD::toPositiveDim(dim0, inputRank);
    if (!TorchD::isValidDim(dim0, inputRank)) return rewriter.notifyMatchFailure(op, "dim0 out of range");

    dim1 = TorchD::toPositiveDim(dim1, inputRank);
    if (!TorchD::isValidDim(dim1, inputRank)) return rewriter.notifyMatchFailure(op, "dim1 out of range");

    llvm::SmallVector<int64_t, 4> permValues(inputRank);
    std::iota(std::begin(permValues), std::end(permValues), 0);
    std::swap(permValues[dim0], permValues[dim1]);

    auto permAttr = rewriter.getI64ArrayAttr(permValues);
    rewriter.replaceOpWithNewOp<mlir::muse::PermuteOp>(op, outType, self, permAttr);
    return success();
  }
};

struct ConvertAtenSliceTensor : public OpConversionPattern<TorchD::AtenSliceTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenSliceTensorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto inType = cast<RankedTensorType>(self.getType());
    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    int64_t dim, start, end, step;
    if (!matchPattern(op.getDim(), TorchD::m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    }
    if (!matchPattern(op.getStart(), TorchD::m_TorchConstantInt(&start))) {
      return rewriter.notifyMatchFailure(op, "start must be constant");
    }
    if (!matchPattern(op.getEnd(), TorchD::m_TorchConstantInt(&end))) {
      return rewriter.notifyMatchFailure(op, "end must be constant");
    }
    if (!matchPattern(op.getStep(), TorchD::m_TorchConstantInt(&step))) {
      return rewriter.notifyMatchFailure(op, "step must be constant");
    }

    int64_t inputRank = inType.getRank();
    dim = TorchD::toPositiveDim(dim, inputRank);
    if (!TorchD::isValidDim(dim, inputRank)) {
      return rewriter.notifyMatchFailure(op, "dim out of range");
    }

    int64_t dimSize = inType.getDimSize(dim);

    start = TorchD::toPositiveDim(start, dimSize);
    if (start < 0) start = 0;
    if (start > dimSize) start = dimSize;

    end = TorchD::toPositiveDim(end, dimSize);
    if (end < start) end = start;
    if (end > dimSize) end = dimSize;

    rewriter.replaceOpWithNewOp<mlir::muse::SliceOp>(op, outType, self, rewriter.getI64IntegerAttr(dim),
                                                     rewriter.getI64IntegerAttr(start), rewriter.getI64IntegerAttr(end),
                                                     rewriter.getI64IntegerAttr(step));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

// Populate custom (hand-written) Aten ops to Muse conversion patterns
static void populateAtenToMuseCustomPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertAtenSumDimIntList>(converter, context);
  patterns.add<ConvertAtenToDtype>(converter, context);
  patterns.add<ConvertAtenTransposeInt>(converter, context);
  patterns.add<ConvertAtenSliceTensor>(converter, context);
}

// Populate all Aten ops to Muse conversion patterns
void populateAtenToMuseConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  populateAtenToMuseCustomPatterns(converter, patterns);
}

}  // namespace mlir
