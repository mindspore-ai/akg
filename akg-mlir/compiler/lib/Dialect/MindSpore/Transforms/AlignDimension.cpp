//===------------------ AlignDimension.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert reshape to binary op's input if needed to match rank
//
//===----------------------------------------------------------------------===//

#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ALIGNDIMENSION
#include "akg/Dialect/MindSpore/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mindspore;

namespace {

static LogicalResult
computeReshapeOutput(ArrayRef<int64_t> higherRankShape,
                     ArrayRef<int64_t> lowerRankShape,
                     SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      return failure();
  }
  return success();
}

static LogicalResult EqualizeRanks(PatternRewriter &rewriter, Location loc,
                                   Value &input1, Value &input2) {
  auto input1Ty = llvm::dyn_cast<RankedTensorType>(input1.getType());
  auto input2Ty = llvm::dyn_cast<RankedTensorType>(input2.getType());

  if (!input1Ty || !input2Ty) {
    return failure();
  }

  int64_t input1Rank = input1Ty.getRank();
  int64_t input2Rank = input2Ty.getRank();

  if (input1Rank == input2Rank)
    return success();

  Value higherTensorValue, lowerTensorValue;
  if (input1Rank > input2Rank) {
    higherTensorValue = input1;
    lowerTensorValue = input2;
  } else {
    higherTensorValue = input2;
    lowerTensorValue = input1;
  }

  ArrayRef<int64_t> higherRankShape =
      llvm::cast<RankedTensorType>(higherTensorValue.getType()).getShape();
  ArrayRef<int64_t> lowerRankShape =
      llvm::cast<RankedTensorType>(lowerTensorValue.getType()).getShape();

  SmallVector<int64_t, 4> reshapeOutputShape;

  if (computeReshapeOutput(higherRankShape, lowerRankShape, reshapeOutputShape)
          .failed())
    return failure();

  auto reshapeInputType =
      llvm::cast<RankedTensorType>(lowerTensorValue.getType());
  auto reshapeOutputType = RankedTensorType::get(
      ArrayRef<int64_t>(reshapeOutputShape), reshapeInputType.getElementType());

  auto reshapeLower = rewriter.create<mindspore::ReshapeOp>(
      loc, reshapeOutputType, lowerTensorValue,
      rewriter.getDenseI64ArrayAttr(reshapeOutputShape));

  if (input1Rank > input2Rank) {
    input1 = higherTensorValue;
    input2 = reshapeLower.getResult();
  } else {
    input1 = reshapeLower.getResult();
    input2 = higherTensorValue;
  }

  return success();
}

/// Common code to create the reshape op where necessary to make the rank of the
/// operations equal. input1 and input2 will be updated when the rank has
/// changed. The caller is expected to use these to rewrite the original
/// operator with the RESHAPE now in the graph.
/// return failure when (1) no reshape needed, or (2) output_type is specified
/// and it has different rank
LogicalResult reshapeLowerToHigher(PatternRewriter &rewriter, Location loc,
                                   RankedTensorType outputType, Value &input1,
                                   Value &input2) {
  auto input1Ty = dyn_cast<RankedTensorType>(input1.getType());
  auto input2Ty = dyn_cast<RankedTensorType>(input2.getType());

  if (!input1Ty || !input2Ty) {
    return rewriter.notifyMatchFailure(loc, "input not a ranked tensor");
  }

  int64_t input1Rank = input1Ty.getRank();
  int64_t input2Rank = input2Ty.getRank();

  if (input1Rank == input2Rank)
    return rewriter.notifyMatchFailure(loc,
                                       "cannot rewrite as its already correct");

  Value input1Copy = input1;
  Value input2Copy = input2;
  if (EqualizeRanks(rewriter, loc, input1Copy, input2Copy).failed()) {
    return rewriter.notifyMatchFailure(loc, "failed to reshape inputs");
  }

  // Verify the rank agrees with the output type if the output type is ranked.
  if (outputType) {
    if (outputType.getRank() !=
            llvm::cast<RankedTensorType>(input1Copy.getType()).getRank() ||
        outputType.getRank() !=
            llvm::cast<RankedTensorType>(input2Copy.getType()).getRank())
      return rewriter.notifyMatchFailure(
          loc, "the reshaped type doesn't agrees with the ranked output type");
  }

  input1 = input1Copy;
  input2 = input2Copy;

  return success();
}

template <typename OpTy>
struct ConvertMindsporeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy mindsporeBinaryOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = mindsporeBinaryOp.getInput1();
    Value input2 = mindsporeBinaryOp.getInput2();
    Value output = mindsporeBinaryOp.getResult();

    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return failure();

    if (reshapeLowerToHigher(rewriter, mindsporeBinaryOp.getLoc(), outputType,
                             input1, input2)
            .failed())
      return failure();

    rewriter.replaceOpWithNewOp<OpTy>(mindsporeBinaryOp, outputType, input1, input2);

    return success();
  }
};

template <>
struct ConvertMindsporeOp<mindspore::SelectOp> : public OpRewritePattern<mindspore::SelectOp> {
  using OpRewritePattern<mindspore::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::SelectOp mindsporeOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = mindsporeOp.getPred();
    Value input2 = mindsporeOp.getOnTrue();
    Value input3 = mindsporeOp.getOnFalse();
    Value output = mindsporeOp.getResult();

    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(mindsporeOp, "output not a ranked tensor");

    // Apply broadcasting to each pair of inputs separately, and chain them as
    // compound as below so that the broadcasting happens all at once.
    bool reshaped1 = reshapeLowerToHigher(rewriter, mindsporeOp.getLoc(), outputType,
                                          input1, input2)
                         .succeeded();

    bool reshaped2 = reshapeLowerToHigher(rewriter, mindsporeOp.getLoc(), outputType,
                                          input1, input3)
                         .succeeded();

    bool reshaped3 = reshapeLowerToHigher(rewriter, mindsporeOp.getLoc(), outputType,
                                          input2, input3)
                         .succeeded();

    if (!reshaped1 && !reshaped2 && !reshaped3)
      return rewriter.notifyMatchFailure(
          mindsporeOp,
          "cannot rewrite as the rank of all operands is already aligned");

    int32_t result1Rank = cast<RankedTensorType>(input1.getType()).getRank();
    int32_t result2Rank = cast<RankedTensorType>(input2.getType()).getRank();
    int32_t result3Rank = cast<RankedTensorType>(input3.getType()).getRank();
    int32_t outputRank = outputType.getRank();

    if ((result1Rank != result2Rank) || (result2Rank != result3Rank) ||
        (result1Rank != outputRank))
      return rewriter.notifyMatchFailure(
          mindsporeOp, "not all ranks are aligned with each other");

    rewriter.replaceOpWithNewOp<mindspore::SelectOp>(mindsporeOp, outputType, input1,
                                                		 input2, input3);

    return success();
  }
};

} // namespace

namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct AlignDimension
    : public impl::AlignDimensionBase<AlignDimension> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    MLIRContext *ctx = func.getContext();
    // Add the generated patterns to the list.
    patterns.add<ConvertMindsporeOp<mindspore::AddOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::SubOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::MulOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::DivOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::MaximumOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::MinimumOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::EqualOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::GreaterOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::GreaterEqualOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::LogicalAndOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::LogicalOrOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::SelectOp>>(ctx);
    patterns.add<ConvertMindsporeOp<mindspore::PowOp>>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createAlignDimensionPass() {
  return std::make_unique<AlignDimension>();
}
