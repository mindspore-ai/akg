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

#include "akg/Dialect/MindSpore/Transforms/RemoveRedundantReduce.h"

#include <algorithm>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Dialect/MindSpore/Passes.h"

namespace mlir {
#ifndef GEN_PASS_DECL_REMOVEREDUNDANTREDUCE
#define GEN_PASS_DECL_REMOVEREDUNDANTREDUCE
#ifndef GEN_PASS_DEF_REMOVEREDUNDANTREDUCE
#define GEN_PASS_DEF_REMOVEREDUNDANTREDUCE
#include "akg/Dialect/MindSpore/Passes.h.inc"
#endif
#endif
}  // namespace mlir

using namespace mlir;
using namespace mlir::mindspore;

namespace {
template <typename SourceOp>
class RemoveRedundantReduceOp : public OpRewritePattern<SourceOp> {
 public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp mindsporeOp, PatternRewriter &rewriter) const override {
    Operation *op = mindsporeOp;
    Value opnd = op->getOperand(0);
    auto inputTy = opnd.getType().cast<ShapedType>();
    llvm::ArrayRef<int64_t> axesAttr = op->getAttr("axis").dyn_cast<DenseI64ArrayAttr>();
    llvm::SmallVector<int64_t> axes;
    (void)std::transform(axesAttr.begin(), axesAttr.end(), std::back_inserter(axes), [](int64_t axis) { return axis; });
    bool keepDims = false;
    if (op->getAttr("keepdims")) {
      keepDims = op->getAttr("keepdims").dyn_cast<BoolAttr>().getValue();
    }
    bool isRedundantReduce = true;
    for (int64_t i = 0; i < inputTy.getRank(); i++) {
      if (llvm::is_contained(axes, i) && (inputTy.getShape()[i]) != 1) {
        isRedundantReduce = false;
      }
    }
    if (isRedundantReduce) {
      if (keepDims) {
        rewriter.replaceOp(op, opnd);
        return success();
      }
    }
    return success();
  }
};

}  // namespace

namespace {
struct RemoveRedundantReduce : public impl::RemoveRedundantReduceBase<RemoveRedundantReduce> {
 public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    (void)
      patterns.add<RemoveRedundantReduceOp<mindspore::ReduceSumOp>, RemoveRedundantReduceOp<mindspore::ReduceAllOp>,
                   RemoveRedundantReduceOp<mindspore::ReduceAnyOp>, RemoveRedundantReduceOp<mindspore::ReduceMaxOp>,
                   RemoveRedundantReduceOp<mindspore::ReduceMinOp>, RemoveRedundantReduceOp<mindspore::ReduceProdOp>>(
        patterns.getContext());
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    func::FuncOp func = getOperation();
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), grc);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createRemoveRedundantReducePass() {
  return std::make_unique<RemoveRedundantReduce>();
}
