//===------------ MindSporeMakeBroadcastable.cpp --------------------------===//
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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>

namespace mlir {
#define GEN_PASS_DEF_MINDSPOREMAKEBROADCASTABLE
#include "akg/Dialect/MindSpore/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mindspore;

namespace {

DictionaryAttr addOpSymShapeAttr(Type inputType, MLIRContext *context) {
  ShapedType shapedType = cast<ShapedType>(inputType);
  ArrayRef<int64_t> typeShapes = shapedType.getShape();
  SmallVector<Attribute> symAttr;
  for (size_t i = 0; i < typeShapes.size(); i++) {
    symAttr.emplace_back(StringAttr::get(context, std::to_string(typeShapes[i])));
  }

  SmallVector<NamedAttribute> opSymbol;
  opSymbol.emplace_back(StringAttr::get(context, "input_0"),
                        ArrayAttr::get(context, symAttr));
  return DictionaryAttr::get(context, ArrayRef<NamedAttribute>(opSymbol));;
}

static void createBroadCastOp(PatternRewriter &rewriter, 
                              Value input, Value output) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
	if (inputType.getShape() == outputType.getShape())
		return;

  auto context = input.getContext();
  SmallVector<NamedAttribute> allAttrs;
	auto newShapeAttr = DenseI64ArrayAttr::get(
		context, ArrayRef<int64_t>(outputType.getShape()));
  allAttrs.emplace_back(
    NamedAttribute(StringAttr::get(context, "new_shape"), newShapeAttr));
  DictionaryAttr symbolAttrs = addOpSymShapeAttr(input.getType(), context);
  allAttrs.emplace_back(
    StringAttr::get(context, getFrontendSymbolAttrName()), symbolAttrs);
  
  auto elemTy = getElementTypeOrSelf(input.getType());
  auto resultType = RankedTensorType::get(outputType.getShape(), elemTy);
	auto brcOp = rewriter.create<mindspore::BroadcastToOp>(
		input.getLoc(), resultType, input, allAttrs);
	input.replaceAllUsesExcept(brcOp.getResult(), brcOp);
	return;
}

template <typename OpTy>
struct ConvertMindsporeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy msOp,
                                PatternRewriter &rewriter) const override {
    Value input1 = msOp.getInput1();
    Value input2 = msOp.getInput2();
    Value output = msOp.getResult();

    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return failure();

		rewriter.setInsertionPoint(msOp);
		createBroadCastOp(rewriter, input1, output);
		createBroadCastOp(rewriter, input2, output);

    return success();
  }
};

template <>
struct ConvertMindsporeOp<mindspore::SelectOp> : public OpRewritePattern<mindspore::SelectOp> {
  using OpRewritePattern<mindspore::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {

    Value input1 = selectOp.getPred();
    Value input2 = selectOp.getOnTrue();
    Value input3 = selectOp.getOnFalse();
    Value output = selectOp.getResult();

    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!outputType)
      return failure();

		rewriter.setInsertionPoint(selectOp);
		createBroadCastOp(rewriter, input1, output);
		createBroadCastOp(rewriter, input2, output);
		createBroadCastOp(rewriter, input3, output);

    return success();
  }
};

} // namespace

namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct MindsporeMakeBroadcastable
    : public impl::MindsporeMakeBroadcastableBase<MindsporeMakeBroadcastable> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
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
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), grc);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createMindsporeMakeBroadcastablePass() {
  return std::make_unique<MindsporeMakeBroadcastable>();
}
