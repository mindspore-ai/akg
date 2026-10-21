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

#include "akg/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#ifndef GEN_PASS_DECL_FOLDMEMREFSUBVIEWONLY
#define GEN_PASS_DECL_FOLDMEMREFSUBVIEWONLY
#ifndef GEN_PASS_DEF_FOLDMEMREFSUBVIEWONLY
#define GEN_PASS_DEF_FOLDMEMREFSUBVIEWONLY
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "fold-memref-subview-only"

namespace mlir {
namespace {

struct FoldMemRefSubViewOnlyPass : public FoldMemRefSubViewOnlyBase<FoldMemRefSubViewOnlyPass> {
  void runOnOperation() override;
};

}  // namespace

void FoldMemRefSubViewOnlyPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Reuse the upstream pattern set as-is; subview-only restriction is enforced
  // at the driver level by limiting the worklist.
  RewritePatternSet patterns(&getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  FrozenRewritePatternSet frozen(std::move(patterns));

  // Start from every memref.subview and add only the subview itself, nested
  // subviews, and their direct non-view users. Do not add collapse_shape or
  // expand_shape operations: the upstream pattern set also contains folders
  // for those operations, and running them here would fold reshapes as a side
  // effect of folding a subview.
  llvm::SetVector<Operation *> targetSet;
  auto isViewOp = [](Operation *op) {
    return isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp, memref::MemorySpaceCastOp,
               memref::ReinterpretCastOp, memref::ReshapeOp, memref::SubViewOp>(op);
  };
  func.walk([&targetSet, &isViewOp](memref::SubViewOp subview) {
    SmallVector<Operation *> queue{subview};
    while (!queue.empty()) {
      Operation *op = queue.pop_back_val();
      if (!targetSet.insert(op)) {
        continue;
      }
      for (Operation *user : op->getUsers()) {
        if (isa<memref::SubViewOp>(user)) {
          queue.push_back(user);
        } else if (!isViewOp(user)) {
          targetSet.insert(user);
        }
      }
    }
  });

  if (targetSet.empty()) {
    return;
  }

  SmallVector<Operation *> targets(targetSet.begin(), targetSet.end());
  GreedyRewriteConfig config;
  // A subview folder replaces a consumer with a new operation. Keep that
  // replacement out of this worklist so reshape folders cannot run on it.
  config.strictMode = GreedyRewriteStrictness::ExistingOps;
  (void)applyOpPatternsAndFold(targets, frozen, config);
}

std::unique_ptr<Pass> createFoldMemRefSubViewOnlyPass() { return std::make_unique<FoldMemRefSubViewOnlyPass>(); }

}  // namespace mlir
