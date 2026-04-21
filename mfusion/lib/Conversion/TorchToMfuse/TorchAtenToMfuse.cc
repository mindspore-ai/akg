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
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

namespace {
// Check that `listVal` is a list construct of constant ints and
// collect the integer values into `out`.
bool isConstantListInt(Value listVal, llvm::SmallVectorImpl<int64_t> &out) {
  llvm::SmallVector<Value, 4> elems;
  if (!TorchD::getListConstructElements(listVal, elems)) {
    return false;
  }

  out.clear();
  out.reserve(elems.size());
  for (Value v : elems) {
    int64_t dim = 0;
    if (!matchPattern(v, TorchD::m_TorchConstantInt(&dim))) {
      return false;
    }
    out.push_back(dim);
  }
  return true;
}
}  // namespace

//===----------------------------------------------------------------------===//
// Generic binary op conversion with type promotion
//===----------------------------------------------------------------------===//
// Converts Torch Aten binary tensor ops to Mfuse meta ops using the target's
// custom builder, which calls promoteBinaryOperands to unify operand element
// types (e.g. f16, f32 -> f32) and satisfy SameOperandsAndResultElementType.
template <typename SourceOp, typename TargetOp>
struct ConvertBinaryOpPattern : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    if (operands.size() < 2) {
      return rewriter.notifyMatchFailure(op, "binary op requires at least 2 operands");
    }
    Value lhs = operands[0];
    Value rhs = operands[1];
    Type resType = this->getTypeConverter()->convertType(op.getType());
    if (!resType) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    auto targetOp = rewriter.create<TargetOp>(op.getLoc(), resType, lhs, rhs);
    rewriter.replaceOp(op, targetOp.getResult());
    return success();
  }
};

// Convert reshape like ops to mfuse.reshape
template <typename SourceOp>
struct ConvertReshapeLikeOp : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto outType = mlir::dyn_cast<RankedTensorType>(this->getTypeConverter()->convertType(op.getType()));
    if (!outType) {
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    }
    auto dynamic_dim_count = std::count_if(outType.getShape().begin(), outType.getShape().end(),
                                           [](int64_t dim) { return dim == ShapedType::kDynamic; });
    if (dynamic_dim_count > 1) {
      return rewriter.notifyMatchFailure(op, "result has more than one dynamic dimension");
    }
    rewriter.replaceOpWithNewOp<mlir::mfuse::ReshapeOp>(op, outType, adaptor.getSelf());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// (the pattern list should be alphabetically sorted)
//===----------------------------------------------------------------------===//

/// Convert torch.aten.broadcast_to -> mfuse.broadcast_to.
/// The size must be a constant list (all elements are constant ints),
/// and not computed by other operators.
struct ConvertAtenBroadcastTo : public OpConversionPattern<TorchD::AtenBroadcastToOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenBroadcastToOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();

    // Check size is a constant int list.
    Value sizeVal = op.getSize();
    llvm::SmallVector<int64_t, 4> sizeInts;
    if (!isConstantListInt(sizeVal, sizeInts)) {
      return rewriter.notifyMatchFailure(op, "size must be a list construct of constant ints for mfuse.broadcast_to");
    }

    rewriter.replaceOpWithNewOp<mlir::mfuse::BroadcastToOp>(op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  }
};

/// Default convolution parameters for 2D conv (spatial lists have 2 elements).
constexpr int64_t kConvGroupsDefault = 1;
constexpr int64_t kConvStrideDefault = 1;
constexpr int64_t kConvPaddingDefault = 0;
constexpr int64_t kConvDilationDefault = 1;
constexpr int64_t kConvOutputPaddingDefault = 0;

/// Returns failure and notifies if listVal is not a 2-element list of constant ints; on success
/// writes the two values to out0/out1 (2D spatial conv lists).
static LogicalResult extractConvParamList2(TorchD::AtenConvolutionOp op, ConversionPatternRewriter &rewriter,
                                           Value listVal, int64_t &out0, int64_t &out1) {
  llvm::SmallVector<Value, 2> elts;
  if (!TorchD::getListConstructElements(listVal, elts) || elts.size() != mfuse::kDim2) {
    return rewriter.notifyMatchFailure(op, "conv spatial param list must be a 2-element constant int list");
  }
  if (!matchPattern(elts[0], TorchD::m_TorchConstantInt(&out0)) ||
      !matchPattern(elts[1], TorchD::m_TorchConstantInt(&out1))) {
    return rewriter.notifyMatchFailure(op, "conv spatial param list must be constant ints");
  }
  return success();
}

static LogicalResult extractAtenConvolutionHyperParams(TorchD::AtenConvolutionOp op,
                                                       ConversionPatternRewriter &rewriter, int64_t &groups,
                                                       bool &transposed, int64_t &s0, int64_t &s1, int64_t &p0,
                                                       int64_t &p1, int64_t &d0, int64_t &d1, int64_t &o0,
                                                       int64_t &o1) {
  groups = 0;
  if (!matchPattern(op.getGroups(), TorchD::m_TorchConstantInt(&groups)) || groups < 1) {
    return rewriter.notifyMatchFailure(op, "groups must be a constant positive int");
  }
  transposed = false;
  if (!matchPattern(op.getTransposed(), TorchD::m_TorchConstantBool(&transposed))) {
    return rewriter.notifyMatchFailure(op, "transposed must be a constant bool");
  }
  s0 = s1 = p0 = p1 = d0 = d1 = o0 = o1 = 0;
  if (failed(extractConvParamList2(op, rewriter, op.getStride(), s0, s1))) {
    return failure();
  }
  if (failed(extractConvParamList2(op, rewriter, op.getPadding(), p0, p1))) {
    return failure();
  }
  if (failed(extractConvParamList2(op, rewriter, op.getDilation(), d0, d1))) {
    return failure();
  }
  if (failed(extractConvParamList2(op, rewriter, op.getOutputPadding(), o0, o1))) {
    return failure();
  }
  return success();
}

static bool isNarrowDefaultMetaConv2d(int64_t groups, bool transposed, int64_t s0, int64_t s1, int64_t p0, int64_t p1,
                                      int64_t d0, int64_t d1, int64_t o0, int64_t o1, bool noBias) {
  return noBias && groups == kConvGroupsDefault && !transposed && s0 == kConvStrideDefault &&
         s1 == kConvStrideDefault &&
         p0 == kConvPaddingDefault && p1 == kConvPaddingDefault && d0 == kConvDilationDefault &&
         d1 == kConvDilationDefault && o0 == kConvOutputPaddingDefault && o1 == kConvOutputPaddingDefault;
}

template <typename ConvolutionAdaptor>
static LogicalResult replaceAtenConvolutionWithAclnn(TorchD::AtenConvolutionOp op, ConvolutionAdaptor adaptor,
                                                     ConversionPatternRewriter &rewriter, RankedTensorType resultType,
                                                     int64_t groups, bool transposed, int64_t s0, int64_t s1,
                                                     int64_t p0, int64_t p1, int64_t d0, int64_t d1, int64_t o0,
                                                     int64_t o1, bool noBias) {
  Value input = adaptor.getInput();
  Value weight = adaptor.getWeight();
  if (isNarrowDefaultMetaConv2d(groups, transposed, s0, s1, p0, p1, d0, d1, o0, o1, noBias)) {
    rewriter.replaceOpWithNewOp<mlir::mfuse::AclnnConv2DOp>(op, resultType, input, weight);
    return success();
  }
  auto strideAttr = rewriter.getI64ArrayAttr({s0, s1});
  auto paddingAttr = rewriter.getI64ArrayAttr({p0, p1});
  auto dilationAttr = rewriter.getI64ArrayAttr({d0, d1});
  auto outputPaddingAttr = rewriter.getI64ArrayAttr({o0, o1});
  auto transposedAttr = rewriter.getBoolAttr(transposed);
  auto groupsAttr = rewriter.getI64IntegerAttr(groups);
  if (noBias) {
    rewriter.replaceOpWithNewOp<mlir::mfuse::AclnnConv2DOp>(op, resultType, input, weight, strideAttr, paddingAttr,
                                                             dilationAttr, transposedAttr, outputPaddingAttr,
                                                             groupsAttr);
    return success();
  }
  Value bias = adaptor.getBias();
  rewriter.replaceOpWithNewOp<mlir::mfuse::AclnnConv2DWithBiasOp>(
    op, resultType, input, weight, bias, strideAttr, paddingAttr, dilationAttr, transposedAttr, outputPaddingAttr,
    groupsAttr);
  return success();
}

/// Convert torch.aten.convolution to mfuse.aclnn.conv2d / mfuse.aclnn.conv2d_with_bias (rank-4, 2-element
/// spatial lists). Narrow default hyper-parameters use the 2/3-operand builders; otherwise full
/// attributes are set (aligned with CANN aclnnConvolution / torch.aten.convolution).
struct ConvertAtenConvolution : public OpConversionPattern<TorchD::AtenConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenConvolutionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    int64_t groups = 0;
    bool transposed = false;
    int64_t s0 = 0, s1 = 0, p0 = 0, p1 = 0, d0 = 0, d1 = 0, o0 = 0, o1 = 0;
    if (failed(extractAtenConvolutionHyperParams(op, rewriter, groups, transposed, s0, s1, p0, p1, d0, d1, o0, o1))) {
      return failure();
    }

    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    auto resultType = dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    auto inTy = dyn_cast<RankedTensorType>(input.getType());
    auto wTy = dyn_cast<RankedTensorType>(weight.getType());
    if (!inTy || inTy.getRank() != 4 || !wTy || wTy.getRank() != 4) {
      return rewriter.notifyMatchFailure(op, "only rank-4 input and weight are supported for mfuse conv lowering");
    }

    const bool noBias = isa<TorchD::NoneType>(op.getBias().getType());
    return replaceAtenConvolutionWithAclnn(op, adaptor, rewriter, resultType, groups, transposed, s0, s1, p0, p1, d0,
                                           d1, o0, o1, noBias);
  }
};

/// Convert torch.aten.expand -> mfuse.broadcast_to.
/// The size must be a constant list (all elements are constant ints),
/// and not computed by other operators.
struct ConvertAtenExpand : public OpConversionPattern<TorchD::AtenExpandOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();

    // Check size is a constant int list.
    llvm::SmallVector<int64_t, 4> sizeInts;
    if (!isConstantListInt(op.getSize(), sizeInts)) {
      return rewriter.notifyMatchFailure(op, "size must be a list construct of constant ints for mfuse.broadcast_to");
    }

    rewriter.replaceOpWithNewOp<mlir::mfuse::BroadcastToOp>(op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  }
};

struct ConvertAtenPermute : public OpConversionPattern<TorchD::AtenPermuteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenPermuteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto inType = mlir::dyn_cast<RankedTensorType>(self.getType());
    if (!inType) {
      return rewriter.notifyMatchFailure(op, "input type must be ranked tensor");
    }
    int64_t inputRank = inType.getRank();

    llvm::SmallVector<int64_t, 4> permValues;
    if (!isConstantListInt(op.getDims(), permValues)) {
      return rewriter.notifyMatchFailure(op, "dims must be a list construct of constant ints");
    }

    if (permValues.size() != static_cast<size_t>(inputRank)) {
      return rewriter.notifyMatchFailure(op, "dims size must match input rank");
    }

    for (int64_t &dim : permValues) {
      dim = TorchD::toPositiveDim(dim, inputRank);
      if (!TorchD::isValidDim(dim, inputRank)) {
        return rewriter.notifyMatchFailure(op, "dim out of range");
      }
    }

    auto outType = cast<RankedTensorType>(getTypeConverter()->convertType(op->getResult(0).getType()));
    auto permAttr = rewriter.getI64ArrayAttr(permValues);
    rewriter.replaceOpWithNewOp<mlir::mfuse::PermuteOp>(op, outType, self, permAttr);
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
    int64_t inputRank = inType.getRank();
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
          dims.push_back(dim < 0 ? dim + inputRank : dim);
        }
      }
    }
    if (reduceAll) {
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
    rewriter.replaceOpWithNewOp<mlir::mfuse::ReduceSumOp>(op, outType, self, dimsAttr, keepdimAttr);
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

// Default epsilon for RmsNorm; kept in sync with PyTorch/HuggingFace defaults
// so that ConvertAtenRmsNorm preserves the original F.rms_norm behaviour when
// eps is not explicitly provided.
constexpr double kDefaultRmsNormEpsilon = 1e-6;

/// Convert torch.aten.rms_norm -> mfuse.aclnn.rms_norm (direct path). Requires weight to be present and eps to be a
/// constant float. aten.rms_norm produces 1 result; aclnn.rms_norm produces 2 (yOut, rstdOut). Only yOut is used to
/// replace the original op; rstdOut is left for DCE.
struct ConvertAtenRmsNorm : public OpConversionPattern<TorchD::AtenRmsNormOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenRmsNormOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (isa<TorchD::NoneType>(op.getWeight().getType())) {
      return rewriter.notifyMatchFailure(op, "weight must not be None");
    }

    double epsVal = kDefaultRmsNormEpsilon;
    if (!isa<TorchD::NoneType>(op.getEps().getType())) {
      if (!matchPattern(op.getEps(), TorchD::m_TorchConstantFloat(&epsVal))) {
        return rewriter.notifyMatchFailure(op, "eps must be a constant float");
      }
    }

    llvm::SmallVector<int64_t, 4> normalizedShape;
    if (!isConstantListInt(op.getNormalizedShape(), normalizedShape)) {
      return rewriter.notifyMatchFailure(op, "normalized_shape must be constant int list");
    }

    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();
    int64_t numNorm = static_cast<int64_t>(normalizedShape.size());
    if (numNorm > inputRank) {
      return rewriter.notifyMatchFailure(op, "normalized_shape rank exceeds input rank");
    }

    auto yOutType = cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    // rstd shape: input shape with last N dims (N = len(normalized_shape))
    // reduced to 1, matching keepdim=true semantics.
    auto inputShape = inputType.getShape();
    SmallVector<int64_t> rstdShape(inputShape.begin(), inputShape.end());
    for (int64_t i = inputRank - numNorm; i < inputRank; ++i) {
      rstdShape[i] = 1;
    }
    auto rstdType = RankedTensorType::get(rstdShape, inputType.getElementType());

    auto epsilonAttr = rewriter.getF64FloatAttr(epsVal);
    SmallVector<Type, 2> resultTypes = {yOutType, rstdType};
    auto rmsNormOp = rewriter.create<mfuse::AclnnRmsNormOp>(op.getLoc(), resultTypes, input, weight, epsilonAttr);

    rewriter.replaceOp(op, rmsNormOp.getYOut());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

// Populate custom (hand-written) Aten ops to Mfuse conversion patterns
static void populateAtenToMfuseCustomPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertAtenBroadcastTo>(converter, context);
  patterns.add<ConvertAtenConvolution>(converter, context);
  patterns.add<ConvertAtenExpand>(converter, context);
  patterns.add<ConvertAtenRmsNorm>(converter, context);
  patterns.add<ConvertAtenSumDimIntList>(converter, context);
  patterns.add<ConvertAtenTransposeInt>(converter, context);
  // aten.permute -> mfuse.permute so fuse-batch-matmul can fold swap-last-two-dims + matmul into trans_x*.
  // (transpose.int is handled by ConvertAtenTransposeInt; graphs that use permute need this too.)
  patterns.add<ConvertAtenPermute>(converter, context);
}

// Populate reshape-like Aten ops to Mfuse conversion patterns
static void populateAtenToMfuseReshapeLikeOpPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertReshapeLikeOp<TorchD::AtenReshapeOp>>(converter, context);
  patterns.add<ConvertReshapeLikeOp<TorchD::AtenSqueezeDimOp>>(converter, context);
  patterns.add<ConvertReshapeLikeOp<TorchD::AtenSqueezeOp>>(converter, context);
  patterns.add<ConvertReshapeLikeOp<TorchD::AtenUnsqueezeOp>>(converter, context);
  patterns.add<ConvertReshapeLikeOp<TorchD::AtenViewOp>>(converter, context);
}

// Populate binary Aten ops using the generic pattern so conversion goes through
// the Mfuse op's builder (promoteBinaryOperands), ensuring consistent operand types.
// Add and sub are converted via PDLL to mfuse.aclnn.add / mfuse.aclnn.sub.
static void populateAtenToMfuseBinaryOpPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertBinaryOpPattern<TorchD::AtenDivTensorOp, mfuse::DivOp>,
               ConvertBinaryOpPattern<TorchD::AtenDivScalarOp, mfuse::DivOp>,
               ConvertBinaryOpPattern<TorchD::AtenEqTensorOp, mfuse::EqOp>,
               ConvertBinaryOpPattern<TorchD::AtenEqScalarOp, mfuse::EqOp>,
               ConvertBinaryOpPattern<TorchD::AtenGeTensorOp, mfuse::GeOp>,
               ConvertBinaryOpPattern<TorchD::AtenGeScalarOp, mfuse::GeOp>,
               ConvertBinaryOpPattern<TorchD::AtenGtTensorOp, mfuse::GtOp>,
               ConvertBinaryOpPattern<TorchD::AtenGtScalarOp, mfuse::GtOp>,
               ConvertBinaryOpPattern<TorchD::AtenLeTensorOp, mfuse::LeOp>,
               ConvertBinaryOpPattern<TorchD::AtenLeScalarOp, mfuse::LeOp>,
               ConvertBinaryOpPattern<TorchD::AtenLogicalAndOp, mfuse::LogicalAndOp>,
               ConvertBinaryOpPattern<TorchD::AtenLogicalOrOp, mfuse::LogicalOrOp>,
               ConvertBinaryOpPattern<TorchD::AtenLtTensorOp, mfuse::LtOp>,
               ConvertBinaryOpPattern<TorchD::AtenLtScalarOp, mfuse::LtOp>,
               ConvertBinaryOpPattern<TorchD::AtenMaximumOp, mfuse::MaximumOp>,
               ConvertBinaryOpPattern<TorchD::AtenMinimumOp, mfuse::MinimumOp>,
               ConvertBinaryOpPattern<TorchD::AtenMulTensorOp, mfuse::MulOp>,
               ConvertBinaryOpPattern<TorchD::AtenMulScalarOp, mfuse::MulOp>,
               ConvertBinaryOpPattern<TorchD::AtenNeTensorOp, mfuse::NeOp>,
               ConvertBinaryOpPattern<TorchD::AtenNeScalarOp, mfuse::NeOp>,
               ConvertBinaryOpPattern<TorchD::AtenPowTensorTensorOp, mfuse::PowOp>,
               ConvertBinaryOpPattern<TorchD::AtenPowTensorScalarOp, mfuse::PowOp>>(converter, ctx);
}

// Populate all Aten ops to Mfuse conversion patterns
void populateAtenToMfuseConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  populateAtenToMfuseCustomPatterns(converter, patterns);
  populateAtenToMfuseBinaryOpPatterns(converter, patterns);
  populateAtenToMfuseReshapeLikeOpPatterns(converter, patterns);
}

}  // namespace mlir
