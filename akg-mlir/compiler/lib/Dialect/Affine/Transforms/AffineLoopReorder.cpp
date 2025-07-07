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

#include "akg/Dialect/Affine/Transforms/AffineLoopReorder.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h.inc"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_AFFINELOOPREORDER
#define GEN_PASS_DECL_AFFINELOOPREORDER
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace akgglobal;

namespace {

constexpr int kDoubleTileNums = 3;  // two-tile split one loop to three parts

// AffineLoopReorder is used to reorder loops based on handwrite settings. It can reorder
// nest loops and keep logic correct. Currently it used on reduction mappings reorder.
struct AffineLoopReorder : public impl::AffineLoopReorderBase<AffineLoopReorder> {
  AffineLoopReorder() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mindspore::MindSporeDialect>();
  }
  void runOnOperation() override;

 private:
  // Validate the setting is a permutation
  bool isPermutation(const std::vector<int> &nums);
  // Validate whether we need to do reorder
  bool needToReorder(const std::vector<int> &nums);
  // Sink apply ops to keep the correctness
  void SinkApplyOps(OpBuilder &builder, SmallVector<Operation *, 8> &opList);
  // Check the op0 is inside in op1
  bool isInsideIn(Operation *const op0, const Operation *const op1);
};

bool AffineLoopReorder::isPermutation(const std::vector<int> &nums) {
  int size = nums.size();
  std::vector<bool> seen(size, false);
  for (int num : nums) {
    if (num < 0 || num >= size || seen[num]) {
      return false;
    }
    seen[num] = true;
  }
  return true;
}

bool AffineLoopReorder::needToReorder(const std::vector<int> &nums) {
  for (int i = 0; i < static_cast<int>(nums.size()); i++) {
    if (nums[i] != i) {
      return true;
    }
  }
  return false;
}

bool AffineLoopReorder::isInsideIn(Operation *const op0, const Operation *const op1) {
  Operation *curOp = op0;
  while (curOp) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(curOp)) {
      if (curOp == op1) {
        return true;
      }
    }
    curOp = curOp->getParentOp();
  }
  return false;
}

void AffineLoopReorder::SinkApplyOps(OpBuilder &builder, SmallVector<Operation *, 8> &opList) {
  auto funcOp = getOperation();
  Operation *keepArgs = nullptr;
  funcOp->walk([&](mindspore::KeepArgsOp op) {
    if (op->hasAttr("BoundaryIf")) {
      keepArgs = op;
      WalkResult::interrupt();
    }
    WalkResult::advance();
  });
  auto firstOp = &*dyn_cast<affine::AffineForOp>(opList[opList.size() - 1]).getRegion().front().getOperations().begin();
  builder.setInsertionPoint(firstOp);
  SmallVector<Operation *, 8> ops;
  funcOp->walk([&](affine::AffineApplyOp op) {
    if (!isInsideIn(op.getOperation(), opList[opList.size() - 1])) {
      ops.push_back(op.getOperation());
    }
  });
  for (auto op : ops) {
    auto newOp = builder.clone(*op);
    op->replaceAllUsesWith(newOp);
    op->erase();
  }
}

void AffineLoopReorder::runOnOperation() {
  auto funcOp = getOperation();

  if (!funcOp->hasAttr("OperatorType") ||
      dyn_cast<StringAttr>(funcOp->getAttr("OperatorType")).getValue().str() != "Reduce") {
    return;
  }

  SmallVector<Operation *, 8> opList;
  funcOp->walk([&](affine::AffineForOp op) { opList.push_back(op.getOperation()); });
  std::reverse(opList.begin(), opList.end());

  // Get the order and check the validations
  auto order = GpuScheduleTool::getInstance().getUpdatedOrder();
  auto start = 0;
  if (order.size() == 0) {
    if (newOrder.size() == 0) {
      return;
    }
    if (newOrder.size() != opList.size()) {
      auto msg = "mismatch between order and afffine.for ops numbers: " + std::to_string(newOrder.size()) + " vs " +
                 std::to_string(opList.size()) + ".";
      funcOp.emitError(msg);
      signalPassFailure();
    }
    if (!isPermutation(newOrder)) {
      funcOp.emitError("new-order is not a permutation of afffine.for ops, skip.");
      signalPassFailure();
    }
    order = newOrder;
  } else {
    // have a fake loop that cover nest-loops
    if (opList.size() % kDoubleTileNums == 1 && opList.size() == order.size() + 1) {
      start = 1;
    }
  }
  if (order.size() == 0 || !needToReorder(order)) {
    return;
  }

  // Start to transforms the mlir file by settings.
  mlir::OpBuilder builder(funcOp);
  mlir::IRMapping mapper;
  SmallVector<Operation *, 8> newOpList;
  // sink affine.apply ops to deal with non perfect nest
  SinkApplyOps(builder, opList);
  builder.setInsertionPointAfter(opList[start]);
  // create new nest-loop
  for (size_t i = start; i < opList.size(); i++) {
    auto matchedLoop = dyn_cast<affine::AffineForOp>(opList[start + order[i - start]]);
    auto newLoop = builder.create<mlir::affine::AffineForOp>(
      matchedLoop.getLoc(), matchedLoop.getLowerBoundOperands(), matchedLoop.getLowerBoundMap(),
      matchedLoop.getUpperBoundOperands(), matchedLoop.getUpperBoundMap(), matchedLoop.getStepAsInt());
    Operation *newOp = newLoop.getOperation();
    newOp->setAttrs(matchedLoop.getOperation()->getAttrs());
    newOpList.push_back(newOp);
    mapper.map(matchedLoop.getInductionVar(), newLoop.getInductionVar());
    builder.setInsertionPointToStart(newLoop.getBody());
  }
  for (auto &op : dyn_cast<affine::AffineForOp>(opList[opList.size() - 1]).getBody()->without_terminator()) {
    builder.clone(op, mapper);
  }
  opList[start]->erase();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAffineLoopReorderPass() {
  return std::make_unique<AffineLoopReorder>();
}
