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

#include "akg/Dialect/Affine/Transforms/AffineHandleBoundaryIfRestore.h"
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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"

#include <string>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_AFFINEHANDLEBOUNDARYIFRESTORE
#define GEN_PASS_DECL_AFFINEHANDLEBOUNDARYIFRESTORE
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace akgglobal;

namespace {

struct AffineHandleBoundaryIfRestore : public impl::AffineHandleBoundaryIfRestoreBase<AffineHandleBoundaryIfRestore> {
  AffineHandleBoundaryIfRestore() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mindspore::MindSporeDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }
  void runOnOperation() override;
  std::pair<Operation *, Operation *> getBoundaryIf();
  Operation *getInnerForOp(Operation *boundaryIf, SmallVector<Value, 8> &relatedVars);
  Operation *getInnerApplyOp(Operation *funcOp, SmallVector<Operation *, 8> &relatedOps);
  void collectRelatedOpsAndVars(Operation *op, SmallVector<Operation *, 8> &relatedOps,
                                SmallVector<Value, 8> &relatedVars);
  bool isDeeper(Operation *const op1, const Operation *const op2);
  Operation *createFakeOuterLoop(OpBuilder &builder, Operation *funcOp);
};

std::pair<Operation *, Operation *> AffineHandleBoundaryIfRestore::getBoundaryIf() {
  Operation *keepArgs = nullptr;
  getOperation()->walk([&](mindspore::KeepArgsOp op) {
    if (op->hasAttr("BoundaryIf")) {
      keepArgs = op;
      WalkResult::interrupt();
    }
    WalkResult::advance();
  });
  if (!keepArgs) {
    return std::make_pair(nullptr, nullptr);
  }
  Operation *curOp = keepArgs;
  while (curOp && !isa<scf::IfOp>(curOp)) {
    curOp = curOp->getParentOp();
  }
  return std::make_pair(curOp, keepArgs);
}

Operation *AffineHandleBoundaryIfRestore::getInnerForOp(Operation *boundaryIf, SmallVector<Value, 8> &relatedVars) {
  Operation *curOp = boundaryIf;
  while (curOp) {
    if (auto forOp = dyn_cast<scf::ForOp>(curOp)) {
      bool flag = false;
      for (auto arg : relatedVars) {
        if (arg == forOp.getInductionVar()) {
          flag = true;
          break;
        }
      }
      if (flag) {
        return curOp;
      }
    }
    curOp = curOp->getParentOp();
  }
  return nullptr;
}

Operation *AffineHandleBoundaryIfRestore::getInnerApplyOp(Operation *funcOp, SmallVector<Operation *, 8> &relatedOps) {
  SmallVector<Operation *, 8> applyOps;
  for (auto op : relatedOps) {
    for (auto operand : op->getOperands()) {
      Operation *prev = operand.getDefiningOp();
      if (prev && isa<affine::AffineApplyOp>(prev)) {
        applyOps.push_back(prev);
      }
    }
  }

  Operation *res = nullptr;
  funcOp->walk([&](affine::AffineApplyOp apply) {
    bool flag = false;
    for (auto op : applyOps) {
      if (apply.getOperation() == op) {
        flag = true;
        break;
      }
    }
    if (flag) {
      res = apply.getOperation();
    }
  });
  return res;
}

bool AffineHandleBoundaryIfRestore::isDeeper(Operation *const op1, const Operation *const op2) {
  if (!op1) {
    return false;
  }
  if (!op2) {
    return true;
  }
  Operation *curOp = op1;
  while (curOp) {
    if (curOp == op2) {
      return true;
    }
    curOp = curOp->getParentOp();
  }
  return false;
}

void AffineHandleBoundaryIfRestore::collectRelatedOpsAndVars(Operation *op, SmallVector<Operation *, 8> &relatedOps,
                                                             SmallVector<Value, 8> &relatedVars) {
  for (auto operand : op->getOperands()) {
    if (auto prevOp = operand.getDefiningOp()) {
      if (isa<arith::MulIOp>(prevOp) || isa<arith::AddIOp>(prevOp) || isa<arith::AndIOp>(prevOp) ||
          isa<arith::CmpIOp>(prevOp)) {
        relatedOps.push_back(prevOp);
        collectRelatedOpsAndVars(prevOp, relatedOps, relatedVars);
      }
    } else if (isa<BlockArgument>(operand)) {
      relatedVars.push_back(operand);
    }
  }
}

Operation *AffineHandleBoundaryIfRestore::createFakeOuterLoop(OpBuilder &builder, Operation *funcOp) {
  auto func = dyn_cast<gpu::GPUFuncOp>(funcOp);
  auto iter = func.getBlocks().begin();
  std::advance(iter, 1);
  auto block = &(*iter);

  auto firstOp = &(block->front());
  auto loc = firstOp->getLoc();
  builder.setInsertionPoint(firstOp);
  auto lb = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                              builder.getIntegerAttr(builder.getIndexType(), 0));  // lower bound
  auto ub = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                              builder.getIntegerAttr(builder.getIndexType(), 1));  // upper bound
  auto step = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                builder.getIntegerAttr(builder.getIndexType(), 1));  // step
  auto fakeLoop = builder.create<mlir::scf::ForOp>(loc, lb, ub, step);
  Operation *end = nullptr;
  fakeLoop.walk([&](scf::YieldOp op) { end = op.getOperation(); });
  auto curOp = fakeLoop.getOperation()->getNextNode();
  while (curOp && !isa<scf::YieldOp>(curOp) && !isa<gpu::ReturnOp>(curOp)) {
    auto next = curOp->getNextNode();
    curOp->moveBefore(end);
    curOp = next;
  }
  return fakeLoop.getOperation();
}

void AffineHandleBoundaryIfRestore::runOnOperation() {
  auto m = getOperation();
  Operation *funcOp = nullptr;

  m->walk([&](gpu::GPUFuncOp op) { funcOp = op.getOperation(); });

  auto [boundaryIf, keepArgs] = getBoundaryIf();
  if (!boundaryIf) {
    return;
  }
  keepArgs->erase();
  if (!isa<scf::IfOp>(boundaryIf)) {
    funcOp->emitError("AffineHandleBoundaryIfRestore cannot get proper BoundaryIf");
    return;
  }

  SmallVector<Operation *, 8> relatedOps;
  SmallVector<mlir::Value, 8> relatedVars;
  collectRelatedOpsAndVars(boundaryIf, relatedOps, relatedVars);
  auto innerFor = getInnerForOp(boundaryIf, relatedVars);
  auto innerApply = getInnerApplyOp(funcOp, relatedOps);
  OpBuilder builder(funcOp);
  if (!innerFor) {
    innerFor = createFakeOuterLoop(builder, funcOp);
  }
  bool underApply = isDeeper(innerApply, innerFor);
  if (underApply) {
    innerFor = innerApply->getParentOp();
  }

  SmallVector<Operation *, 16> moveOps;
  for (auto &op : dyn_cast<scf::ForOp>(boundaryIf->getParentOp()).getRegion().front()) {
    if ((&op) == boundaryIf || isa<scf::YieldOp>(op)) {
      continue;
    }
    bool isRelated = false;
    for (auto relatedOp : relatedOps) {
      if (&op == relatedOp) {
        isRelated = true;
        break;
      }
    }
    if (isRelated) {
      moveOps.push_back(&op);
    }
  }

  auto ifOp = dyn_cast<scf::IfOp>(boundaryIf);
  auto forOp = dyn_cast<scf::ForOp>(innerFor);
  auto block = &forOp.getRegion().front();
  if (underApply) {
    builder.setInsertionPointAfter(innerApply);
  } else {
    builder.setInsertionPointToStart(block);
  }
  auto newIfOp = builder.create<scf::IfOp>(innerFor->getLoc(), ifOp.getOperand());
  builder.setInsertionPoint(newIfOp);
  for (auto op : moveOps) {
    op->moveBefore(newIfOp);
  }
  ifOp.erase();
  bool startMove = false;
  builder.setInsertionPointToStart(newIfOp.thenBlock());

  Operation *end = &*newIfOp.thenBlock()->getOperations().rbegin();
  moveOps.clear();
  for (mlir::Operation &op : block->getOperations()) {
    Operation *opPtr = &op;
    if (!startMove) {
      if (opPtr == newIfOp.getOperation()) {
        startMove = true;
      }
      continue;
    }
    if (!isa<scf::YieldOp>(opPtr)) {
      moveOps.push_back(opPtr);
    }
  }
  for (auto op : moveOps) {
    op->moveBefore(end);
  }
}

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> mlir::createAffineHandleBoundaryIfRestore() {
  return std::make_unique<AffineHandleBoundaryIfRestore>();
}
