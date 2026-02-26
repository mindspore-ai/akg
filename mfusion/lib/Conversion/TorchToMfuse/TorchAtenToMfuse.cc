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

#include "mfusion/Conversion/TorchToMfuse/TorchAtenToMfuse.h"

#include <algorithm>
#include <numeric>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/OpConstants.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

namespace {
// Check if the shape has at most one dynamic dimension.
bool isSemiStaticShape(RankedTensorType type) {
  int64_t dynamicDims = std::count_if(type.getShape().begin(), type.getShape().end(),
                                      [](int64_t dim) { return dim == ShapedType::kDynamic; });
  return dynamicDims <= 1;
}
}  // namespace

//===----------------------------------------------------------------------===//
// Aten ops to Mfuse conversion patterns
// (the pattern list should be alphabetically sorted)
//===----------------------------------------------------------------------===//

struct ConvertAtenReshape : public OpConversionPattern<TorchD::AtenReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto outType = dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    if (!outType) {
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    }
    if (!isSemiStaticShape(outType)) {
      return rewriter.notifyMatchFailure(op, "result has more than one dynamic dimension");
    }

    rewriter.replaceOpWithNewOp<mlir::mfuse::ReshapeOp>(op, outType, adaptor.getSelf());
    return success();
  }
};

/// Default convolution parameters for 2D conv (spatial lists have 2 elements).
constexpr int64_t kConvGroupsDefault = 1;
constexpr int64_t kConvStrideDefault = 1;
constexpr int64_t kConvPaddingDefault = 0;
constexpr int64_t kConvDilationDefault = 1;
constexpr int64_t kConvOutputPaddingDefault = 0;

/// Returns failure and notifies if listVal is not a 2-element list of constant ints [e0, e1].
static LogicalResult checkConvParamList2(TorchD::AtenConvolutionOp op, ConversionPatternRewriter &rewriter,
                                        Value listVal, int64_t e0, int64_t e1, StringRef failureMsg) {
  llvm::SmallVector<Value, 2> elts;
  if (!TorchD::getListConstructElements(listVal, elts) || elts.size() != mfuse::kDim2) {
    return rewriter.notifyMatchFailure(op, failureMsg);
  }
  int64_t v0 = 0, v1 = 0;
  if (!matchPattern(elts[0], TorchD::m_TorchConstantInt(&v0)) ||
      !matchPattern(elts[1], TorchD::m_TorchConstantInt(&v1)) || v0 != e0 || v1 != e1) {
    return rewriter.notifyMatchFailure(op, failureMsg);
  }
  return success();
}


/// Convert torch.aten.convolution (no bias, default stride/padding/dilation)
/// to mfuse.conv2d. Only matches when bias is None and stride=[1,1],
/// padding=[0,0], dilation=[1,1], output_padding=[0,0], groups=1, transposed=false.
struct ConvertAtenConvolution : public OpConversionPattern<TorchD::AtenConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenConvolutionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<TorchD::NoneType>(op.getBias().getType())) {
      return rewriter.notifyMatchFailure(op, "convolution must have no bias (None)");
    }
    int64_t groups = 0;
    if (!matchPattern(op.getGroups(), TorchD::m_TorchConstantInt(&groups)) || groups != kConvGroupsDefault) {
      return rewriter.notifyMatchFailure(op, "groups must be 1");
    }
    bool transposed = true;
    if (!matchPattern(op.getTransposed(), TorchD::m_TorchConstantBool(&transposed)) || transposed) {
      return rewriter.notifyMatchFailure(op, "transposed must be false");
    }
    if (failed(checkConvParamList2(op, rewriter, op.getStride(), kConvStrideDefault, kConvStrideDefault,
                                  "stride must be [1, 1]"))) {
      return failure();
    }
    if (failed(checkConvParamList2(op, rewriter, op.getPadding(), kConvPaddingDefault, kConvPaddingDefault,
                                  "padding must be [0, 0]"))) {
      return failure();
    }
    if (failed(checkConvParamList2(op, rewriter, op.getDilation(), kConvDilationDefault, kConvDilationDefault,
                                  "dilation must be [1, 1]"))) {
      return failure();
    }
    if (failed(checkConvParamList2(op, rewriter, op.getOutputPadding(), kConvOutputPaddingDefault,
                                  kConvOutputPaddingDefault, "output_padding must be [0, 0]"))) {
      return failure();
    }

    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    auto resultType = dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    rewriter.replaceOpWithNewOp<mlir::mfuse::Conv2DOp>(op, resultType, input, weight);
    return success();
  }
};

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

    rewriter.replaceOpWithNewOp<mlir::mfuse::ReduceSumOp>(op, outType, self, dimsAttr, keepdimAttr, dtypeAttr);
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
    rewriter.replaceOpWithNewOp<mlir::mfuse::CastOp>(op, outType, self, dtypeAttr);
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
    rewriter.replaceOpWithNewOp<mlir::mfuse::PermuteOp>(op, outType, self, permAttr);
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

    rewriter.replaceOpWithNewOp<mlir::mfuse::SliceOp>(
      op, outType, self, rewriter.getI64IntegerAttr(dim), rewriter.getI64IntegerAttr(start),
      rewriter.getI64IntegerAttr(end), rewriter.getI64IntegerAttr(step));
    return success();
  }
};

struct ConvertAtenView : public OpConversionPattern<TorchD::AtenViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto outType = dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
    if (!outType) {
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    }
    if (!isSemiStaticShape(outType)) {
      return rewriter.notifyMatchFailure(op, "result has more than one dynamic dimension");
    }

    rewriter.replaceOpWithNewOp<mlir::mfuse::ReshapeOp>(op, outType, adaptor.getSelf());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

// Populate custom (hand-written) Aten ops to Mfuse conversion patterns
static void populateAtenToMfuseCustomPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertAtenReshape>(converter, context);
  patterns.add<ConvertAtenSliceTensor>(converter, context);
  patterns.add<ConvertAtenConvolution>(converter, context);
  patterns.add<ConvertAtenSumDimIntList>(converter, context);
  patterns.add<ConvertAtenToDtype>(converter, context);
  patterns.add<ConvertAtenTransposeInt>(converter, context);
  patterns.add<ConvertAtenView>(converter, context);
}

// Populate all Aten ops to Mfuse conversion patterns
void populateAtenToMfuseConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  populateAtenToMfuseCustomPatterns(converter, patterns);
}

}  // namespace mlir
