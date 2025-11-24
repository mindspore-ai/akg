/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGLOWERTEMPLATEOP
#define GEN_PASS_DEF_LINALGLOWERTEMPLATEOP
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalgExt;

namespace {
struct LowerTemplateOp : public OpRewritePattern<TemplateOp> {
 public:
  using OpRewritePattern<TemplateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TemplateOp templateOp, PatternRewriter &rewriter) const override {
    // find func Op
    auto fnSym = cast<SymbolRefAttr>(templateOp->getAttr(TemplateFuncAttrName));
    auto funcOp = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(templateOp, fnSym));

    // cast linalg op operands if they are not all dynamic sizes

    SmallVector<Value> castedOperands =
      castOperands(templateOp.getOperands(), funcOp.getArgumentTypes(), templateOp->getLoc(), rewriter);

    SmallVector<Range> loopRanges =
      cast<LinalgOp>(templateOp.getOperation()).createLoopRanges(rewriter, templateOp.getLoc());
    // insert loop size to operands
    llvm::transform(loopRanges, std::back_inserter(castedOperands), [&](Range range) {
      return getValueOrCreateConstantIndexOp(rewriter, templateOp->getLoc(), range.size);
    });

    // create function call op by template op operands
    auto newOp = rewriter.create<func::CallOp>(templateOp->getLoc(), funcOp, castedOperands);

    // replace
    rewriter.replaceOp(templateOp, newOp.getResults());
    return success();
  }

 private:
  SmallVector<Value> castOperands(SmallVector<Value> operands, ArrayRef<Type> dstTypes, Location loc,
                                  PatternRewriter &rewriter) const;
};

struct LinalgLowerTemplateOpPass : public impl::LinalgLowerTemplateOpBase<LinalgLowerTemplateOpPass> {
  LinalgLowerTemplateOpPass() = default;
  LinalgLowerTemplateOpPass(const LinalgLowerTemplateOpPass &) = default;
  LinalgLowerTemplateOpPass &operator=(const LinalgLowerTemplateOpPass &) = delete;

  void runOnOperation() override {
    Operation *op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateLinalgTemplateOpLowerPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

SmallVector<Value> LowerTemplateOp::castOperands(SmallVector<Value> operands, ArrayRef<Type> dstTypes, Location loc,
                                                 PatternRewriter &rewriter) const {
  SmallVector<Value> newOperands;
  for (auto item : llvm::zip(operands, dstTypes)) {
    auto oper = std::get<0>(item);
    auto dstType = std::get<1>(item);
    if (!isa<ShapedType>(oper.getType()) || oper.getType() == dstType) {
      newOperands.push_back(oper);
      continue;
    }

    assert(isa<MemRefType>(oper.getType()) && "currently only support memref");
    auto memType = cast<MemRefType>(oper.getType());
    bool allDynamicSize = llvm::all_of(memType.getShape(), mlir::ShapedType::isDynamic);
    if (allDynamicSize) {
      newOperands.push_back(oper);
    } else {
      // add castOp to cast memref type to all dynamic sizes
      auto dynType = memType.clone(std::vector<int64_t>(memType.getShape().size(), mlir::ShapedType::kDynamic));
      auto dynOper = rewriter.create<memref::CastOp>(loc, dynType, oper);
      newOperands.push_back(dynOper.getResult());
    }
  }
  return newOperands;
}

void mlir::populateLinalgTemplateOpLowerPatterns(RewritePatternSet &patterns) {
  (void)patterns.add<LowerTemplateOp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgLowerTemplateOpPass() {
  return std::make_unique<LinalgLowerTemplateOpPass>();
}
