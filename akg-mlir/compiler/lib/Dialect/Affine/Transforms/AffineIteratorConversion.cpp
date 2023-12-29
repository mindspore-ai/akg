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

#include "akg/Dialect/Affine/Transforms/AffineIteratorConversion.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DECL_AFFINEITERATORCONVERSION
#define GEN_PASS_DEF_AFFINEITERATORCONVERSION
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "affine-load-removal"

using namespace mlir;
using namespace llvm;

namespace mlir {

struct AffineIteratorConversion : public impl::AffineIteratorConversionBase<AffineIteratorConversion> {
  AffineIteratorConversion() = default;

  void runOnOperation() override;
  void loadRemoveEachBand(Operation *curOp);
  void removeInitMemoryCopy(func::FuncOp func);
};
}  // namespace mlir

class CreateArithOp {
 public:
  CreateArithOp(OpBuilder b, AffineForOp newLoop, AffineLoadOp loadOp, AffineStoreOp storeOp, Operation *arithOp)
      : b(b), newLoop(newLoop), loadOp(loadOp), storeOp(storeOp), arithOp(arithOp) {}

  template <typename opType>
  Operation *create() {
    auto newLoadOp = b.create<AffineLoadOp>(loadOp.getLoc(), loadOp.getMemRef(), loadOp.getAffineMapAttr().getValue(),
                                            loadOp.getIndices());
    auto newArithOp = b.create<opType>(newLoadOp.getLoc(), newLoadOp.getResult().getType(),
                                       ValueRange{newLoop.getResults().back(), newLoadOp.getResult()});
    b.create<AffineStoreOp>(storeOp.getLoc(), newArithOp.getResult(), storeOp.getMemRef(),
                            storeOp.getAffineMapAttr().getValue(), storeOp.getIndices());
    return newArithOp.getOperation();
  }
  ~CreateArithOp() {}
  Operation *arithOp;

 private:
  OpBuilder b;
  AffineForOp newLoop;
  AffineLoadOp loadOp;
  AffineStoreOp storeOp;
};

static Operation *identifyAndCreateArithOp(CreateArithOp &rewriter) {
  Operation *newArithOp;
  llvm::TypeSwitch<Operation *>(rewriter.arithOp)
    .Case([&](arith::AddFOp) { newArithOp = rewriter.create<arith::AddFOp>(); })
    .Case([&](arith::MulFOp) { newArithOp = rewriter.create<arith::MulFOp>(); })
    .Case([&](arith::AddIOp) { newArithOp = rewriter.create<arith::AddIOp>(); })
    .Case([&](arith::AndIOp) { newArithOp = rewriter.create<arith::AndIOp>(); })
    .Case([&](arith::OrIOp) { newArithOp = rewriter.create<arith::OrIOp>(); })
    .Case([&](arith::MulIOp) { newArithOp = rewriter.create<arith::MulIOp>(); })
    .Case([&](arith::MinFOp) { newArithOp = rewriter.create<arith::MinFOp>(); })
    .Case([&](arith::MaxFOp) { newArithOp = rewriter.create<arith::MaxFOp>(); })
    .Case([&](arith::MinSIOp) { newArithOp = rewriter.create<arith::MinSIOp>(); })
    .Case([&](arith::MaxSIOp) { newArithOp = rewriter.create<arith::MaxSIOp>(); })
    .Case([&](arith::MinUIOp) { newArithOp = rewriter.create<arith::MinUIOp>(); })
    .Case([&](arith::MaxUIOp) { newArithOp = rewriter.create<arith::MaxUIOp>(); })
    .Default([](Operation *) {});

  return newArithOp;
}

void AffineIteratorConversion::removeInitMemoryCopy(func::FuncOp func) {
  memref::CopyOp copyOp;
  func.walk([&](memref::CopyOp op) {
    if (op.getTarget().getDefiningOp() && isa<memref::AllocOp>(op.getSource().getDefiningOp()) &&
        isa<memref::AllocOp>(op.getTarget().getDefiningOp())) {
      copyOp = op;
    }
  });
  if (!copyOp) {
    return;
  }
  auto source = copyOp.getSource();
  auto target = copyOp.getTarget();
  auto allocOp = target.getDefiningOp();
  target.replaceAllUsesWith(source);
  allocOp->erase();
  copyOp.erase();
}

static AffineLoadOp getLoadOp(AffineForOp reduceForOp, Operation *arithOp) {
  auto lhs = arithOp->getOperands()[0];
  auto rhs = arithOp->getOperands()[1];
  auto iv = reduceForOp.getInductionVar();
  AffineLoadOp loadOp;
  reduceForOp.walk([&](AffineLoadOp op) {
    if (op != lhs.getDefiningOp() && op != rhs.getDefiningOp()) {
      return;
    }
    auto indices = op.getIndices();
    bool flag = false;
    for (auto index : indices) {
      if (index == iv) {
        flag = true;
        break;
      }
    }
    if (!flag) {
      loadOp = op;
    }
  });
  return loadOp;
}

static Operation *getInnermostReduceOp(Operation *curOp) {
  Operation *innermostReduceOp = nullptr;
  curOp->walk([&](AffineForOp op) -> WalkResult {
    if (op->getAttr("reduceLoop")) {
      innermostReduceOp = op.getOperation();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return innermostReduceOp;
}

void AffineIteratorConversion::loadRemoveEachBand(Operation *curOp) {
  OpBuilder b(curOp);
  Operation *reduceArithOp = nullptr;
  curOp->walk([&](Operation *op) -> WalkResult {
    // The tail block does not need to be processed.
    if (op->getAttr(kReductionTypeStr)) {
      reduceArithOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!reduceArithOp) {
    return;
  }

  Operation *reduceLoopOp = getInnermostReduceOp(curOp);
  if (!reduceLoopOp) {
    return;
  }
  AffineStoreOp initStoreOp = nullptr;
  while (isa<AffineForOp>(reduceLoopOp) && reduceLoopOp->getAttr("reduceLoop")) {
    AffineForOp reduceLoop = cast<AffineForOp>(reduceLoopOp);
    // init load statement
    AffineLoadOp loadOp = getLoadOp(reduceLoop, reduceArithOp);
    // init statement
    auto initOp = CommonUtils::getReduceInitOp(reduceArithOp, curOp->getBlock());
    if (initOp) {
      initStoreOp = initOp;
    }

    // reduce result store statement
    AffineStoreOp storeOp = nullptr;
    reduceLoop.walk([&](AffineStoreOp op) {
      if (op.getMemref() == loadOp.getMemref()) {
        storeOp = op;
      }
    });
    if (!storeOp || !initStoreOp || !loadOp) {
      reduceArithOp->emitError("Error: the statement associated with reduce is not found. \n");
      return;
    }
    auto definingOp = initStoreOp.getValue().getDefiningOp();
    if (!isa<arith::ConstantOp>(definingOp)) {
      return;
    }
    arith::ConstantOp constOp = cast<arith::ConstantOp>(definingOp);
    auto newLoop = replaceForOpWithNewYields(b, reduceLoop, constOp.getResult(), SmallVector<Value>{storeOp.getValue()},
                                             storeOp.getValue());
    loadOp.getResult().replaceUsesWithIf(newLoop.getLoopBody().getArguments().back(), [&](OpOperand &use) {
      Operation *user = use.getOwner();
      return newLoop->isProperAncestor(user);
    });
    b.setInsertionPoint(reduceLoop);
    auto parentOp = newLoop.getOperation()->getParentOp();
    if (isa<AffineForOp>(parentOp) && parentOp->getAttr("reduceLoop")) {
      CreateArithOp rewriter(b, newLoop, loadOp, storeOp, reduceArithOp);
      reduceArithOp = identifyAndCreateArithOp(rewriter);
    } else {
      b.create<AffineStoreOp>(storeOp.getLoc(), newLoop.getResults().back(), storeOp.getMemRef(),
                              storeOp.getAffineMapAttr().getValue(), storeOp.getIndices());
    }
    reduceLoop.erase();
    loadOp.erase();
    storeOp.erase();
    reduceLoopOp = newLoop.getOperation()->getParentOp();
  }
  AffineIfOp ifOp = dyn_cast<AffineIfOp>(initStoreOp.getOperation()->getParentOp());
  if (initStoreOp) {
    initStoreOp.erase();
  }
  if (ifOp) {
    ifOp.erase();
  }
}

void AffineIteratorConversion::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder b(func);
  OperatorTemplate opType = CommonUtils::getOperatorType(func);
  ReduceDirection reduceDirection = CommonUtils::getReduceDirection(func);
  if (opType != OperatorTemplate::Reduce || reduceDirection == ReduceDirection::Y) {
    return;
  }

  removeInitMemoryCopy(func);
  // todo(yanzhi): bugfix this function
  SmallVector<Operation *, 8> reduceLoops = CommonUtils::collectReductionAxes(func);
  for (auto reduceLoop : reduceLoops) {
    reduceLoop->setAttr("reduceLoop", b.getUnitAttr());
  }

  SmallVector<AffineForOp, 6> bands;
  for (auto band : func.getOps<AffineForOp>()) {
    bands.push_back(band);
  }
  for (auto band : bands) {
    loadRemoveEachBand(band);
  }

  // remove empty for
  func->walk([&](AffineForOp forOp) {
    if (isa<AffineYieldOp>(forOp.getBody()->front())) {
      forOp.erase();
    }
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAffineIteratorConversionPass() {
  return std::make_unique<AffineIteratorConversion>();
}
