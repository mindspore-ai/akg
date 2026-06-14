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
#define GEN_PASS_DEF_LEGALIZETYPE
#include "akg/Dialect/MindSpore/Passes.h.inc"
}  // namespace mlir

using namespace mlir;             // NOLINT(build/namespaces)
using namespace mlir::mindspore;  // NOLINT(build/namespaces)

namespace {

static Value getCastedValue(PatternRewriter &rewriter, Value oper, Type toElemTy) {
  ShapedType shapedType = cast<ShapedType>(oper.getType());
  if (shapedType.getElementType() == toElemTy) {
    return oper;
  }

  auto loc = oper.getLoc();
  auto resultType = RankedTensorType::get(shapedType.getShape(), toElemTy);

  auto castOp = rewriter.create<mindspore::CastOp>(loc, resultType, oper);
  return castOp->getResult(0);
}

template <typename Op>
static Operation *createNewOp(PatternRewriter &rewriter, Op subOp, SmallVector<Value> &castedOperands) {
  IRMapping mapper;
  Operation *op = subOp.getOperation();
  for (const auto &[idx, oper] : llvm::enumerate(op->getOperands())) {
    mapper.map(oper, castedOperands[idx]);
  }

  auto newOp = rewriter.clone(*op, mapper);
  for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
    ShapedType shapedType = cast<ShapedType>(res.getType());
    auto newResTy = shapedType.clone(rewriter.getI64Type());
    newOp->getResult(idx).setType(newResTy);
  }

  return newOp;
}

template <typename Op>
static void createI64ElementTypeOp(Op subOp, PatternRewriter &rewriter) {
  Operation *op = subOp.getOperation();
  Type i64Ty = rewriter.getI64Type();
  Type ui8Ty = rewriter.getIntegerType(8, false);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> castedOperands;
  for (auto oper : op->getOperands()) {
    auto castedOperand = getCastedValue(rewriter, oper, i64Ty);
    castedOperands.push_back(castedOperand);
  }

  auto newOp = createNewOp<Op>(rewriter, subOp, castedOperands);

  rewriter.setInsertionPointAfter(newOp);
  SmallVector<Value> castedResults;
  for (auto res : newOp->getResults()) {
    auto castedResult = getCastedValue(rewriter, res, ui8Ty);
    castedResults.push_back(castedResult);
  }

  rewriter.replaceOp(op, castedResults);
  return;
}

static bool isUI8ElemType(Operation *op, Type ui8Ty) {
  auto elemTy = getElementTypeOrSelf(op->getResultTypes()[0]);
  return elemTy == ui8Ty;
}

struct ConvertSubOp : public OpRewritePattern<mindspore::SubOp> {
  using OpRewritePattern<mindspore::SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::SubOp subOp, PatternRewriter &rewriter) const override {
    Type ui8Ty = rewriter.getIntegerType(8, false);
    if (!isUI8ElemType(subOp, ui8Ty)) {
      return failure();
    }

    createI64ElementTypeOp(subOp, rewriter);
    return success();
  }
};

}  // namespace

namespace {
/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct LegalizeType : public impl::LegalizeTypeBase<LegalizeType> {
 public:
  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    MLIRContext *ctx = func.getContext();
    auto moduleOp = func.getOperation()->getParentOp();
    moduleOp->removeAttr("mindspore.symbol_calc_expr");
    // Add the generated patterns to the list.
    patterns.add<ConvertSubOp>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::createLegalizeTypePass() { return std::make_unique<LegalizeType>(); }
