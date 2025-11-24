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

#include <algorithm>

#include "akg/Dialect/MindSpore/Transforms/MoveDownReductionOps.h"

#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
#ifndef GEN_PASS_DECL_MOVEDOWNREDUCTIONOPS
#define GEN_PASS_DECL_MOVEDOWNREDUCTIONOPS
#ifndef GEN_PASS_DEF_MOVEDOWNREDUCTIONOPS
#define GEN_PASS_DEF_MOVEDOWNREDUCTIONOPS
#include "akg/Dialect/MindSpore/Passes.h.inc"
#endif
#endif
}  // namespace mlir

using namespace mlir;
using namespace mlir::mindspore;

namespace {

bool IsReduceOp(Operation *op) {
  if (isa<mindspore::ReduceSumOp>(op) || isa<mindspore::ReduceAllOp>(op) || isa<mindspore::ReduceAnyOp>(op) ||
      isa<mindspore::ReduceMaxOp>(op) || isa<mindspore::ReduceMinOp>(op) || isa<mindspore::ReduceProdOp>(op)) {
    return true;
  }
  return false;
}

struct MoveDownReductionOps : public impl::MoveDownReductionOpsBase<MoveDownReductionOps> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    Operation *redOp = nullptr;
    (void)funcOp->walk([&](Operation *op) {
      if (IsReduceOp(op)) {
        redOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!redOp) {
      return;
    }

    SmallVector<Operation *, 8> relatedOps;
    SmallVector<mlir::Value, 8> usedValues;

    CommonUtils::getAllNextRelatedOps(redOp, relatedOps, usedValues);
    Operation *curOp = redOp->getNextNode();
    Operation *nextOp = nullptr;
    while (curOp) {
      bool flag = false;
      if (std::any_of(relatedOps.begin(), relatedOps.end(), [curOp](const Operation *op) { return op == curOp; })) {
        flag = true;
      }
      nextOp = curOp->getNextNode();
      if (!flag) {
        curOp->moveBefore(redOp);
      }
      curOp = nextOp;
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createMoveDownReductionOpsPass() {
  return std::make_unique<MoveDownReductionOps>();
}
