/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Utils/GlobalVars.hpp"
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
}  // namespace mlir

namespace {
using namespace mlir;             // NOLINT(build/namespaces)
using namespace mlir::mindspore;  // NOLINT(build/namespaces)

static DictionaryAttr addOpSymShapeAttr(Type inputType, MLIRContext *context) {
  ShapedType shapedType = cast<ShapedType>(inputType);
  ArrayRef<int64_t> typeShapes = shapedType.getShape();
  SmallVector<Attribute> symAttr;
  std::transform(typeShapes.begin(), typeShapes.end(), std::back_inserter(symAttr),
                 [context](int64_t typeShape) { return StringAttr::get(context, std::to_string(typeShape)); });

  SmallVector<NamedAttribute> opSymbol;
  opSymbol.emplace_back(StringAttr::get(context, "input_0"), ArrayAttr::get(context, symAttr));
  return DictionaryAttr::get(context, ArrayRef<NamedAttribute>(opSymbol));
}

struct ConvertBroadcastToOp : public OpRewritePattern<mindspore::BroadcastToOp> {
  using OpRewritePattern<mindspore::BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::BroadcastToOp brcOp, PatternRewriter &rewriter) const override {
    Value input = brcOp.getInput();
    auto inputType = cast<ShapedType>(input.getType());
    auto inputShape = inputType.getShape();
    if (!brcOp.getOperation()->hasAttr("frontend_symbol")) {
      DictionaryAttr symbolAttrs = addOpSymShapeAttr(input.getType(), brcOp.getContext());
      brcOp.getOperation()->setAttr(StringAttr::get(brcOp.getContext(), getFrontendSymbolAttrName()), symbolAttrs);
    }

    SmallVector<int64_t> newShape;
    for (auto dim : inputShape) {
      if (dim == 1) {
        continue;
      }
      newShape.push_back(dim);
    }
    auto newType = inputType.clone(newShape);
    if (newType == inputType) {
      return failure();
    }

    rewriter.setInsertionPoint(brcOp);
    auto reshapeOp =
      rewriter.create<mindspore::ReshapeOp>(brcOp.getLoc(), newType, input, rewriter.getDenseI64ArrayAttr(newShape));

    input.replaceAllUsesExcept(reshapeOp.getResult(), reshapeOp);
    return success();
  }
};

}  // namespace

namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct EliminateDimension : public impl::EliminateDimensionBase<EliminateDimension> {
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
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createEliminateDimensionPass() { return std::make_unique<EliminateDimension>(); }
}  // namespace mlir
