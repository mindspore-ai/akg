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

#include "akg/Dialect/Affine/Transforms/MergeFusionOp.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir {
#ifndef GEN_PASS_DEF_MERGEFUSIONOP
#define GEN_PASS_DEF_MERGEFUSIONOP
#ifndef GEN_PASS_DECL_MERGEFUSIONOP
#define GEN_PASS_DECL_MERGEFUSIONOP
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "merge-fusion-op"

using namespace mlir;
using namespace llvm;

namespace {

// Move all operators between two for loops up or down to the innermost for loop and add if statements
// to ensure the correctness of the result, so that the subsequent pass can be enabled correctly.
class MergeFusionOpPass : public impl::MergeFusionOpBase<MergeFusionOpPass> {
 public:
  MergeFusionOpPass() {}
  explicit MergeFusionOpPass(const std::string &target) : target(target) {}

  void runOnOperation() override;

 private:
  void getFusionOpBetweenOp(Block *beforeBlock, AffineForOp after);
  void clearFusionOp();
  AffineIfOp createAffineIfOp(OpBuilder &builder, const bool isBackward = false);
  void createIfThenBlock(Operation *op);

  SmallVector<Operation *, 8> forwardFusionOp;
  SmallVector<Operation *, 8> backwardFusionOp;
  SmallVector<AffineForOp, 4> forwardBetweenLoops;
  SmallVector<AffineForOp, 4> backwardBetweenLoops;
  SmallSet<Operation *, 8> reduceInitOp;
  std::string target = kTargetCpu;
};

}  // namespace

void MergeFusionOpPass::clearFusionOp() {
  forwardFusionOp.clear();
  backwardFusionOp.clear();
  forwardBetweenLoops.clear();
  backwardBetweenLoops.clear();
}

// Get all AffineForOps from the current op to the ancestor block and return the top-level ancestor Op.
static Operation *getAncestorForOp(Operation &op, Block *block, SmallVectorImpl<AffineForOp> *betweenLoops = nullptr) {
  auto *currOp = &op;
  // Returns nullptr if the current op doesn't lie in this block.
  if (!block->findAncestorOpInBlock(op)) {
    return nullptr;
  }

  if (auto affineForOp = dyn_cast<AffineForOp>(op)) {
    betweenLoops->push_back(affineForOp);
  }
  while (currOp && currOp->getBlock() != block) {
    currOp = currOp->getParentOp();
    if (auto affineForOp = dyn_cast<AffineForOp>(currOp)) {
      betweenLoops->push_back(affineForOp);
    }
  }
  return currOp;
}

static void getAllAffineFor(func::FuncOp f, std::vector<SmallVector<AffineForOp, 6>> *bands) {
  // multi-filter: the outermost layer consists of multiple AffineForOp
  for (AffineForOp forOp : f.getOps<AffineForOp>()) {
    SmallVector<AffineForOp, 6> band;
    // From the inside to outside
    forOp.walk([&band](const AffineForOp op) { band.push_back(op); });
    bands->push_back(band);
  }
}

static void insertUniqueValue(SmallVectorImpl<AffineForOp> *loopA, SmallVector<AffineForOp, 16> loopB) {
  for (auto loop : loopB) {
    if (std::find(loopA->begin(), loopA->end(), loop) != loopA->end()) {
      continue;
    }
    loopA->push_back(loop);
  }
}

static bool isSeparatedByForLoop(Operation *opA, Operation *opB) {
  Block *block = opA->getBlock();
  if (block != opB->getBlock() || isa<AffineForOp>(opB)) {
    return true;
  }

  for (auto iter = Block::iterator(opA); iter != Block::iterator(opB); ++iter) {
    if (isa<AffineForOp>(*iter)) {
      return true;
    }
  }
  return false;
}

// Get all operators between any two ops and record them in a global variable.
void MergeFusionOpPass::getFusionOpBetweenOp(Block *beforeBlock, AffineForOp after) {
  clearFusionOp();
  if (!beforeBlock->findAncestorOpInBlock(*after.getOperation())) {
    return;
  }
  beforeBlock->walk([this, &after](Operation *op) {
    // Because AffineYieldOp can only be placed at the end of the current
    // block, no Op can be inserted before AffineYieldOp.
    // The current solution only handles non-AffineForOp between two loops, so AffineForOp needs to be skipped.
    // If the inner for loop is an ancestor of the current operator, it means that the
    // current operator is not between the two for loops.
    // todo: isSeparatedByForLoop
    if (isa<AffineYieldOp, AffineForOp, arith::ConstantOp, memref::AllocOp, memref::DeallocOp, memref::CopyOp,
            memref::DimOp, func::ReturnOp>(op) ||
        op == after.getOperation() || after.getBody()->findAncestorOpInBlock(*op) != nullptr) {
      return;
    }

    // nested statements
    if (!isa<AffineForOp, func::FuncOp>(op->getParentOp())) {
      return;
    }

    // Locate the same block to determine whether it is a forward fusion operator or a
    // backward fusion operator.
    Block *commBlock = CommonUtils::getCommonBlock(op, after);
    if (!commBlock) {
      return;
    }

    // Collect all affine.for between two for loops to facilitate the generation of
    // conditions in if statements.
    SmallVector<AffineForOp, 16> betweenLoops;
    Operation *opAncestor = getAncestorForOp(*op, commBlock, &betweenLoops);
    Operation *afterAncestor = getAncestorForOp(*(after.getOperation()), commBlock, &betweenLoops);
    // The current op is the ancestor of after op, that is, the current op contains after op, which is not supported.
    // reduce_x and all reduce: init op do not need to sink.
    if (opAncestor == afterAncestor || reduceInitOp.count(op) == 1) {
      return;
    }
    if (opAncestor->isBeforeInBlock(afterAncestor)) {
      forwardFusionOp.push_back(op);
      insertUniqueValue(&forwardBetweenLoops, betweenLoops);
    } else {
      backwardFusionOp.push_back(op);
      insertUniqueValue(&backwardBetweenLoops, betweenLoops);
    }
  });
}

AffineIfOp MergeFusionOpPass::createAffineIfOp(OpBuilder &builder, const bool isBackward) {
  SmallVector<AffineForOp, 4> loops = isBackward ? backwardBetweenLoops : forwardBetweenLoops;
  if (loops.empty()) {
    return nullptr;
  }

  auto *context = loops[0]->getContext();

  FlatAffineValueConstraints cst;
  SmallVector<Operation *, 8> ops;
  llvm::append_range(ops, loops);
  (void)getIndexSet(ops, &cst);

  IntegerSet allCondSet = cst.getAsIntegerSet(context);
  // left side of a constraint in the if statement
  SmallVector<AffineExpr, 4> exprs;
  SmallVector<bool, 4> eqFlags;
  auto numVar = cst.getNumDimVars();
  // Determine the number of conditions in the if statement based on the vars of each
  // AffineForOp in loops.
  for (size_t i = 0; i < numVar; ++i) {
    // variables in each constraint(variables in each AffineForOp)
    auto insertDimExpr = allCondSet.getConstraint(i * (unsigned int)2 + (size_t)isBackward);
    exprs.push_back(insertDimExpr);
    // each of the constraints is an equality
    eqFlags.push_back(true);
  }
  IntegerSet ifCondSet = IntegerSet::get(numVar, 0, exprs, eqFlags);
  // ifCondSet can be null if cst was empty -- this can happen if all loops
  // in the nest have constant trip counts.
  if (!ifCondSet) {
    return nullptr;
  }

  // right side of a constraint in the if statement
  SmallVector<mlir::Value, 4> setOperands;
  cst.getValues(0, numVar, &setOperands);
  canonicalizeSetAndOperands(&ifCondSet, &setOperands);

  return builder.create<AffineIfOp>(loops[0]->getLoc(), ifCondSet, setOperands, false);
}

void MergeFusionOpPass::createIfThenBlock(Operation *op) {
  OpBuilder builder(op);
  auto body = dyn_cast<AffineForOp>(op).getBody();
  if (!forwardFusionOp.empty()) {
    builder.setInsertionPoint(&(body->front()));
    if (forwardBetweenLoops.empty()) {
      llvm::errs() << "Forward: Failed to obtain the conditional control variable in the if statement.\n";
    }
    AffineIfOp ifOp = createAffineIfOp(builder, false);
    if (!ifOp) {
      return;
    }
    Block *thenBlock = ifOp.getThenBlock();
    for (auto fusionOp : forwardFusionOp) {
      thenBlock->getOperations().splice(std::prev(thenBlock->end()), fusionOp->getBlock()->getOperations(),
                                        Block::iterator(fusionOp));
    }
  }

  if (!backwardFusionOp.empty()) {
    builder.setInsertionPoint(&(body->back()));
    if (backwardBetweenLoops.empty()) {
      llvm::errs() << "Backward: Failed to obtain the conditional control variable in the if statement.\n";
    }
    AffineIfOp ifOp = createAffineIfOp(builder, true);
    if (!ifOp) {
      return;
    }
    Block *thenBlock = ifOp.getThenBlock();
    for (auto fusionOp : backwardFusionOp) {
      thenBlock->getOperations().splice(std::prev(thenBlock->end()), fusionOp->getBlock()->getOperations(),
                                        Block::iterator(fusionOp));
    }
  }
}

void MergeFusionOpPass::runOnOperation() {
  std::vector<SmallVector<AffineForOp, 6>> bands;
  auto funcOp = getOperation();
  getAllAffineFor(funcOp, &bands);

  if (target == kTargetCuda && bands.size() > 1) {
    (void)funcOp.emitError(
      "Error: akg cannot support multi-fiters cases currently on gpu backend. it will cause unexpected errors.");
    return;
  }

  if (target == kTargetCuda) {
    mergeSpecificOp = true;
  }

  // multi-filter
  for (auto band : bands) {
    size_t num = band.size();
    if (num < 1) {
      continue;
    }
    Operation *firstOp = band[num - 1];
    ReduceDirection reduceDirection = CommonUtils::getReduceDirection(firstOp);
    // reduce_x and all reduce: init op do not need to sink.
    if (!mergeSpecificOp && (reduceDirection == ReduceDirection::X || reduceDirection == ReduceDirection::ALL)) {
      firstOp->walk([&](Operation *op) {
        if (op->getAttr(kReductionTypeStr)) {
          auto initOp = CommonUtils::getReduceInitOp(op, &funcOp.getBody().front());
          if (initOp) {
            (void)reduceInitOp.insert(initOp.getOperation());
          }
        }
      });
    }

    for (auto forOp : band) {
      auto ubMap = forOp.getUpperBoundMap();
      auto lbMap = forOp.getLowerBoundMap();
      if (ubMap.getNumResults() > 1 || lbMap.getNumResults() > 1) {
        (void)forOp.emitError("Error: The boundary of the for loop contains an inequality calculation.");
        return;
      }
    }

    for (int i = num - 1; i >= 0; --i) {
      Operation *curOp = band[(unsigned int)i];
      if (mergeSpecificOp) {
        getFusionOpBetweenOp(firstOp->getBlock(), dyn_cast<AffineForOp>(curOp));
      } else {
        getFusionOpBetweenOp(dyn_cast<AffineForOp>(firstOp).getBody(), dyn_cast<AffineForOp>(curOp));
      }
      if (forwardFusionOp.empty() && backwardFusionOp.empty()) {
        continue;
      }
      createIfThenBlock(curOp);
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createMergeFusionOpPass() {
  return std::make_unique<MergeFusionOpPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createMergeFusionOpPass(const std::string &target) {
  return std::make_unique<MergeFusionOpPass>(target);
}
