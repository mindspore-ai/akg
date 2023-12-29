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
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGELEMENTWISEFUSIONEXT
#define GEN_PASS_DEF_LINALGELEMENTWISEFUSIONEXT
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;

namespace {
static bool checkFusedOpDominateAllProducerUsers(Operation *fusedOp, Operation *producer, DominanceInfo &domInfo) {
  for (auto res : producer->getResults()) {
    for (auto user : res.getUsers()) {
      if (!domInfo.properlyDominates(fusedOp, user)) {
        return false;
      }
    }
  }
  return true;
}

static bool CheckIfMatchDominateInSimplePattern0(Operation *fusedOp, Operation *op, DominanceInfo &domInfo) {
  // if op has more than one Result, give up
  if (op->getNumResults() > 1) {
    return false;
  }
  // if op has more than one User, give up
  if (op->getNumResults() == 1 && !op->getResults()[0].hasOneUse()) {
    return false;
  }
  // if op is "ReturnOp" or dominated by fusedOp, order-preserving can be tried here.
  if (isa<func::ReturnOp>(op) || domInfo.properlyDominates(fusedOp, op)) {
    return true;
  }
  // check op's user
  Operation *userOp = *(op->getResults()[0].getUsers().begin());
  return CheckIfMatchDominateInSimplePattern0(fusedOp, userOp, domInfo);
}

// convert:
//  "A-------->B-------->C-------->ReturnOp"
//                            |
//                         fusedOp
// to:
//  "A-------->B-------->C-------->ReturnOp"
//       |
//    fusedOp
static bool TryingtToPreserveOrderInSimplePattern0(Operation *fusedOp, Operation *op, DominanceInfo &domInfo) {
  Operation *userOp = *(op->getResults()[0].getUsers().begin());
  if (isa<func::ReturnOp>(userOp) || domInfo.properlyDominates(fusedOp, userOp)) {
    op->moveBefore(userOp);
    return true;
  }
  if (TryingtToPreserveOrderInSimplePattern0(fusedOp, userOp, domInfo)) {
    op->moveBefore(userOp);
    return true;
  }
  return false;
}

class FuseElementwiseOpsExt : public OpRewritePattern<GenericOp> {
 public:
  FuseElementwiseOpsExt(MLIRContext *context, ControlFusionFn fun, DominanceInfo &domInfo, PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), controlFn(std::move(fun)), domInfo(domInfo) {}

  LogicalResult matchAndRewrite(GenericOp genericOp, PatternRewriter &rewriter) const override {
    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (!areElementwiseOpsFusable(&opOperand))
        continue;
      if (!controlFn(&opOperand))
        continue;

      FailureOr<Operation *> fusionResult = fuseElementwiseOps(rewriter, &opOperand);
      if (succeeded(fusionResult)) {
        Operation *fusedOp = *(fusionResult);
        // 1 replace old consumer: Risk-free replacement
        auto replacements = fusedOp->getResults().take_back(genericOp.getNumResults());
        rewriter.replaceOp(genericOp, replacements);
        // 2 replace old producer: Windy replacement
        Operation *producer = opOperand.get().getDefiningOp();
        // 2.1 Pattern0: producer and it's user have only one user.
        if (CheckIfMatchDominateInSimplePattern0(fusedOp, producer, domInfo)) {
          (void)TryingtToPreserveOrderInSimplePattern0(fusedOp, producer, domInfo);
        }
        // final check: whether fusedOp dominates all producer's users
        // todo: Order-preserving, make fusedOp dominates all of the producer's users.
        if (!checkFusedOpDominateAllProducerUsers(fusedOp, producer, domInfo)) {
          return success();
        }
        replacements = fusedOp->getResults().take_front(producer->getNumResults());
        rewriter.replaceOp(producer, replacements);
        return success();
      }
    }

    return failure();
  }

 private:
  ControlFusionFn controlFn;
  DominanceInfo &domInfo;
};

struct LinalgElementwiseFusionExtPass : public impl::LinalgElementwiseFusionExtBase<LinalgElementwiseFusionExtPass> {
  LinalgElementwiseFusionExtPass() : LinalgElementwiseFusionExtBase() {
    // alwayTrueControlFn as default ControlFn
    controlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer != nullptr;
    };
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;

    RewritePatternSet patterns(context);
    DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
    ;
    (void)patterns.add<FuseElementwiseOpsExt>(context, controlFn, domInfo);
    // Add the patterns that clean up dead operands and results.
    populateEraseUnusedOperandsAndResultsPatterns(patterns);
    populateFoldReshapeOpsByExpansionPatterns(patterns, controlFn);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
  }

 private:
  ControlFusionFn controlFn;
};

}  // namespace

std::unique_ptr<mlir::Pass> mlir::createLinalgElementwiseFusionExtPass() {
  return std::make_unique<LinalgElementwiseFusionExtPass>();
}
