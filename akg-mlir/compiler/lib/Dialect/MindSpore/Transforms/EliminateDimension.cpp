//===------------------ EliminateDimension.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert reshape to broadcast op's input if needed to match rank
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

namespace mlir {
#define GEN_PASS_DEF_ELIMINATEDIMENSION
#include "akg/Dialect/MindSpore/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mindspore;

namespace {

static DictionaryAttr addOpSymShapeAttr(Type inputType, MLIRContext *context) {
  ShapedType shapedType = inputType.cast<ShapedType>();
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

struct ConvertBroadcastToOp : public OpRewritePattern<mindspore::BroadcastToOp> {
  using OpRewritePattern<mindspore::BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::BroadcastToOp brcOp,
                                PatternRewriter &rewriter) const override {
    Value input = brcOp.getInput();
    auto inputType = input.getType().cast<ShapedType>();
    auto inputShape = inputType.getShape();
    if (!brcOp.getOperation()->hasAttr("frontend_symbol")) {
      DictionaryAttr symbolAttrs = addOpSymShapeAttr(input.getType(), brcOp.getContext());
      brcOp.getOperation()->setAttr(StringAttr::get(brcOp.getContext(), getFrontendSymbolAttrName()), symbolAttrs);
    }

    SmallVector<int64_t> newShape;
    for(auto dim : inputShape) {
      if (dim == 1)
        continue;
      newShape.push_back(dim);
    }
    auto newType = inputType.clone(newShape);
    if (newType == inputType) 
      return failure();

    rewriter.setInsertionPoint(brcOp);
    auto reshapeOp = rewriter.create<mindspore::ReshapeOp>(
      brcOp.getLoc(), newType, input,
      rewriter.getDenseI64ArrayAttr(newShape));

    input.replaceAllUsesExcept(reshapeOp.getResult(), reshapeOp);
    return success();
  }
};

} // namespace

namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct EliminateDimension
    : public impl::EliminateDimensionBase<EliminateDimension> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    MLIRContext *ctx = func.getContext();
    // Add the generated patterns to the list.
    patterns.add<ConvertBroadcastToOp>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createEliminateDimensionPass() {
  return std::make_unique<EliminateDimension>();
}
