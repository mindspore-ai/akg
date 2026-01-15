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

#include "akg/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_CLONETENSOREMPTY_
#define GEN_PASS_DEF_CLONETENSOREMPTY
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace {
/// Clone tensor.empty operations that are used as outputs of linalg.generic.
/// This pass clones different tensor.empty operations to linalg.generic outputs
/// to enable better optimization opportunities.
bool CloneNewTensorEmpty(linalg::GenericOp op, PatternRewriter &rewriter) {
  bool needUpdate = false;
  for (Value dst : op.getDpsInits()) {
    auto dstDefiningOp = dst.getDefiningOp();
    if (!dstDefiningOp) continue;
    if (!isa<TensorType>(dst.getType())) continue;
    if (std::distance(dst.use_begin(), dst.use_end()) <= 1) continue;
    if (isa<tensor::EmptyOp>(dstDefiningOp)) {
      // Clone the tensor.empty operation before the linalg.generic
      rewriter.setInsertionPoint(op);
      auto clonedOp = rewriter.clone(*dstDefiningOp);
      // Replace the use of the original tensor.empty with the cloned one
      op->replaceUsesOfWith(dst, clonedOp->getResult(0));
      needUpdate = true;
    }
  }
  return needUpdate;
}

struct CloneTensorEmptyPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    if (CloneNewTensorEmpty(op, rewriter)) {
      return success();
    }
    return failure();
  }
};

/// This pass clones tensor.empty operations used as outputs of linalg.generic
/// to enable better optimization opportunities.
class CloneTensorEmpty : public impl::CloneTensorEmptyBase<CloneTensorEmpty> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<CloneTensorEmptyPattern>(context);
    // Apply patterns using greedy pattern rewrite driver
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCloneTensorEmptyPass() {
  return std::make_unique<CloneTensorEmpty>();
}
}  // namespace mlir
