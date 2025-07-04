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

#include "akg/Dialect/Affine/Transforms/ExtractIfOp.h"
#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir {
#ifndef GEN_PASS_DEF_EXTRACTIFOP
#define GEN_PASS_DEF_EXTRACTIFOP
#ifndef GEN_PASS_DECL_EXTRACTIFOP
#define GEN_PASS_DECL_EXTRACTIFOP
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "redundant-if"

using namespace mlir;
using namespace llvm;
using namespace akg;

namespace {
// To prevent repeated data reading or unnecessary branch judgment, extract the statements that are lifted or sunk in
// MergeFusionOp to the corresponding for loop, thereby improving operator performance.
class ExtractIfOpPass : public impl::ExtractIfOpBase<ExtractIfOpPass> {
 public:
  ExtractIfOpPass() {}
  explicit ExtractIfOpPass(const std::string &target) : target(target) {}

  void runOnBlock(Block *block);
  void runOnOperation() override;

 private:
  Operation *getInsertPoint(mlir::Operation *op, bool isForward = true);
  void removeUselessIf(affine::AffineIfOp ifOp) const;
  void extractIfOp(affine::AffineIfOp ifOp);
  void extractBroadcastForwardOp(affine::AffineIfOp ifOp) const;
  bool extractIfOpPrejudgment(affine::AffineIfOp ifOp) const;

  SmallSet<Operation *, 8> unextractableOp;
  MemRefDependenceGraph dependenceGraph{nullptr};
  std::string target = kTargetCpu;
};

}  // namespace

// If all if conditions are constants, delete if op.
// Determines whether the statements in thenBlock need to be retained.
// affine.if affine_set<() : (1 == 0)>()
// affine.if affine_set<() : (1 >= 0)>()
static uint64_t isInConstantEqualityRange(IntegerSet set) {
  if (set.getNumInputs() != 0) {
    return -1;
  }

  uint64_t needRetained = 0;
  for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
    auto constraint = set.getConstraint(i);
    if (constraint.getKind() == AffineExprKind::Constant) {
      int64_t constraintValue = llvm::dyn_cast<AffineConstantExpr>(constraint).getValue();
      if (set.isEq(i)) {
        // affine.if affine_set<() : (1 == 0)>()
        needRetained |= (uint64_t)(constraintValue == 0);
      } else {
        // affine.if affine_set<() : (1 >= 0)>()
        needRetained |= (uint64_t)(constraintValue >= 0);
      }
    }
  }

  return static_cast<uint64_t>(needRetained);
}

Operation *ExtractIfOpPass::getInsertPoint(mlir::Operation *op, bool isForward) {
  Operation *innerOp = nullptr;
  for (auto operand : op->getOperands()) {
    SmallVector<Operation *, 8> opAxes;
    CommonUtils::collectRelatedAxes(operand, opAxes);
    // The current op does not have a related axis. Searches for the position based on the if condition.
    // affine.store %cst, %arg1[] : memref<f32>
    if (opAxes.empty() && op->getParentOp()) {
      Operation *parentOp = op->getParentOp();
      SmallVector<affine::AffineIfOp, 8> ifOpVec;
      // Gets all nested if statements.
      while (parentOp && isa<affine::AffineIfOp>(parentOp)) {
        ifOpVec.push_back(dyn_cast<affine::AffineIfOp>(parentOp));
        parentOp = parentOp->getParentOp();
      }
      // Gets the associated axis in the if statement.
      for (auto ifOp : ifOpVec) {
        for (auto value : ifOp.getOperands()) {
          CommonUtils::collectRelatedAxes(value, opAxes);
        }
      }

      Operation *outerOp = nullptr;
      // Since the current op does not have an associated axis, we should try to extract it to the outermost.
      for (auto axes : opAxes) {
        outerOp = CommonUtils::getInnerOrOuterOp(outerOp, axes, false);
      }
      if (Operation *parentOp = outerOp->getParentOp()) {
        if (isa<affine::AffineForOp>(parentOp)) {
          outerOp = parentOp;
        } else {
          outerOp = outerOp->getPrevNode();
        }
      }
      innerOp = CommonUtils::getInnerOrOuterOp(innerOp, outerOp);
    } else {
      for (auto axes : opAxes) {
        innerOp = CommonUtils::getInnerOrOuterOp(innerOp, axes);
      }
    }
  }

  // To ensure the correctness of the result, we need to insert the current op after the dependent op.
  // If the current op and the dependent op are in the same block, they are in the same if statement, and no
  // additional judgment is required.
  int nodeId = dependenceGraph.getNodeId(op);
  if (nodeId == -1) {
    return innerOp;
  }
  llvm::DenseSet<unsigned> dependentIds;
  dependenceGraph.getDirectlyDependentNodes(nodeId, dependentIds);
  Operation *dependentInnerOp = nullptr;
  for (auto id : dependentIds) {
    Operation *dependenceOp = dependenceGraph.getNode(id)->op;
    if (dependenceOp->getBlock() != op->getBlock()) {
      dependentInnerOp = CommonUtils::getInnerOrOuterOp(dependentInnerOp, dependenceOp);
    }
  }

  return CommonUtils::getInnerOrOuterOp(dependentInnerOp, innerOp, isForward);
}

void ExtractIfOpPass::removeUselessIf(affine::AffineIfOp ifOp) const {
  // If the then block is empty or the if condition is invalid, delete the if statement.
  if (isa<affine::AffineYieldOp>(ifOp.getThenRegion().front().front()) || !CommonUtils::isInRange(ifOp)) {
    ifOp.erase();
  }

  auto set = ifOp.getIntegerSet();
  int64_t inRange = isInConstantEqualityRange(set);
  if (inRange == 1) {
    OpBuilder b(ifOp);
    b.setInsertionPointAfter(ifOp);
    for (auto &op : ifOp.getThenBlock()->without_terminator()) {
      (void)b.clone(op);
    }
    ifOp.erase();
  } else if (inRange == 0) {
    ifOp.erase();
  }
}

bool ExtractIfOpPass::extractIfOpPrejudgment(affine::AffineIfOp ifOp) const {
  // If the then block is empty or the if condition is invalid, delete the if statement.
  if (isa<affine::AffineYieldOp>(ifOp.getThenRegion().front().front())) {
    ifOp.erase();
    return true;
  }

  auto set = ifOp.getIntegerSet();
  // Currently, only the statements that are lifted or sunk in MergeFusionOp are processed.
  if (set.getNumInequalities() != 0 || ifOp.hasElse()) {
    return true;
  }

  // only support:
  // (d0) : (d0 == 0)
  // (d0) : (-d0 + 32 == 0)
  for (auto constraint : set.getConstraints()) {
    if (constraint.getKind() != AffineExprKind::DimId && constraint.getKind() != AffineExprKind::Add) {
      return true;
    }
  }

  return false;
}

void ExtractIfOpPass::extractIfOp(affine::AffineIfOp ifOp) {
  if (extractIfOpPrejudgment(ifOp)) {
    return;
  }

  auto constraint = ifOp.getIntegerSet().getConstraint(0);
  bool isForward = false;
  if (constraint.getKind() == AffineExprKind::DimId) {
    isForward = true;
  }

  Operation *innermostOp = nullptr;
  SmallVector<Operation *, 8> opVec;
  // Find the innermost for loop associated with the current if statement.
  ifOp.getThenRegion().walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<affine::AffineYieldOp>(op)) {
      return;
    }
    Operation *insertOp = getInsertPoint(op, isForward);
    innermostOp = CommonUtils::getInnerOrOuterOp(innermostOp, insertOp);
    if (unextractableOp.count(op) == 1) {
      return;
    }
    opVec.push_back(op);
  });

  if (!innermostOp) {
    return;
  }

  // The location of the extraction has not changed.
  Operation *parentOp = ifOp.getOperation()->getParentOp();
  while (parentOp && isa<affine::AffineIfOp>(parentOp)) {
    parentOp = parentOp->getParentOp();
  }
  if (innermostOp == parentOp) {
    return;
  }

  OpBuilder b(ifOp);
  // Determines the position where the if statement is inserted based on the if condition.
  if (auto loopOp = dyn_cast<affine::AffineForOp>(innermostOp)) {
    if (isForward) {
      b.setInsertionPoint(&(loopOp.getBody()->front()));
    } else {
      b.setInsertionPoint(&(loopOp.getBody()->back()));
    }
  } else {
    b.setInsertionPointAfter(innermostOp);
  }

  for (Operation *op : opVec) {
    mlir::Operation *clonedOp = b.clone(*op);
    op->replaceAllUsesWith(clonedOp);
    dependenceGraph.updateNodeOp(op, clonedOp);
    op->erase();
  }

  if (isa<affine::AffineYieldOp>(ifOp.getThenRegion().front().front())) {
    ifOp.erase();
  }
}

void ExtractIfOpPass::extractBroadcastForwardOp(affine::AffineIfOp ifOp) const {
  // Obtains the position where the forward fusion operator needs to be inserted.
  Operation *outermostOp = nullptr;
  Operation *innermostOp = nullptr;
  for (auto value : ifOp.getOperands()) {
    if (auto blockArg = value.dyn_cast<BlockArgument>()) {
      if (blockArg.getType().isa<IndexType>()) {
        Block *block = blockArg.getOwner();
        Operation *parentOp = block->getParentOp();
        if (!parentOp->hasAttr("broadcastLoop")) {
          return;
        }
        outermostOp = CommonUtils::getInnerOrOuterOp(outermostOp, parentOp, false);
        innermostOp = CommonUtils::getInnerOrOuterOp(innermostOp, parentOp);
      }
    }
  }

  if (!outermostOp || !innermostOp) {
    return;
  }

  SmallVector<Operation *, 8> forwardFusionOp;
  // Collect the for loops of all forward fusion operators.
  auto opResult = ifOp.getThenRegion().walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    mlir::ValueRange indices;
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      indices = load.getIndices();
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      indices = store.getIndices();
    }

    llvm::SmallSet<Operation *, 8> relatedAxes;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (auto blockArg = indices[i].dyn_cast<BlockArgument>()) {
        if (blockArg.getType().isa<IndexType>()) {
          Block *block = blockArg.getOwner();
          Operation *parentOp = block->getParentOp();
          if (parentOp->hasAttr("broadcastLoop")) {
            return WalkResult::interrupt();
          }
          if (dyn_cast<affine::AffineForOp>(innermostOp).getBody()->findAncestorOpInBlock(*parentOp) &&
              std::find(forwardFusionOp.begin(), forwardFusionOp.end(), parentOp) == forwardFusionOp.end()) {
            forwardFusionOp.push_back(parentOp);
          }
        }
      }
    }
    return WalkResult::advance();
  });
  if (opResult.wasInterrupted()) {
    return;
  }

  OpBuilder b(ifOp);
  b.setInsertionPoint(outermostOp);

  mlir::IRMapping mapper;
  for (size_t i = 0; i < forwardFusionOp.size(); i++) {
    auto matchedLoop = dyn_cast<affine::AffineForOp>(forwardFusionOp[i]);
    auto newLoop = b.create<mlir::affine::AffineForOp>(
      matchedLoop.getLoc(), matchedLoop.getLowerBoundOperands(), matchedLoop.getLowerBoundMap(),
      matchedLoop.getUpperBoundOperands(), matchedLoop.getUpperBoundMap(), matchedLoop.getStepAsInt());
    Operation *newOp = newLoop.getOperation();
    newOp->setAttrs(matchedLoop.getOperation()->getAttrs());
    mapper.map(matchedLoop.getInductionVar(), newLoop.getInductionVar());
    b.setInsertionPointToStart(newLoop.getBody());
  }

  for (auto &op : ifOp.getThenBlock()->without_terminator()) {
    (void)b.clone(op, mapper);
  }
  ifOp.erase();
}

void ExtractIfOpPass::runOnBlock(Block *block) {
  block->walk([&](affine::AffineIfOp ifOp) { removeUselessIf(ifOp); });
  // build dependence graph
  dependenceGraph = MemRefDependenceGraph(block);
  if (!dependenceGraph.init()) {
    return;
  }

  block->walk([&](affine::AffineIfOp ifOp) { extractIfOp(ifOp); });
}

void ExtractIfOpPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  for (Region &region : funcOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      runOnBlock(&block);
    }
  }

  if (target == kTargetCpu) {
    OperatorTemplate opType = CommonUtils::getOperatorType(funcOp);
    // To enable parallel, the forward fusion of the broadcast operator is extracted.
    if (opType == OperatorTemplate::Broadcast) {
      funcOp->walk([&](affine::AffineIfOp ifOp) { extractBroadcastForwardOp(ifOp); });
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createExtractIfOpPass() {
  return std::make_unique<ExtractIfOpPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createExtractIfOpPass(const std::string &target) {
  return std::make_unique<ExtractIfOpPass>(target);
}
