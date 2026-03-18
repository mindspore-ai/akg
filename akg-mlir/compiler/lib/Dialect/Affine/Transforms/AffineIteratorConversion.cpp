/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
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

namespace mlir {

struct AffineIteratorConversion : public impl::AffineIteratorConversionBase<AffineIteratorConversion> {
  AffineIteratorConversion() = default;

  void runOnOperation() override;
  void loadRemoveEachBand(Operation *curOp);
  void removeInitMemoryCopy(func::FuncOp func);
};

class CreateArithOp {
 private:
  OpBuilder b;
  affine::AffineForOp newLoop;
  affine::AffineLoadOp loadOp;
  affine::AffineStoreOp storeOp;

 public:
  CreateArithOp(OpBuilder b, affine::AffineForOp newLoop, affine::AffineLoadOp loadOp, affine::AffineStoreOp storeOp,
                Operation *arithOp)
      : b(b), newLoop(newLoop), loadOp(loadOp), storeOp(storeOp), arithOp(arithOp) {}

  template <typename opType>
  Operation *create() {
    auto newLoadOp = b.create<affine::AffineLoadOp>(loadOp.getLoc(), loadOp.getMemRef(),
                                                    loadOp.getAffineMapAttr().getValue(), loadOp.getIndices());
    auto newArithOp = b.create<opType>(newLoadOp.getLoc(), newLoadOp.getResult().getType(),
                                       ValueRange{newLoop.getResults().back(), newLoadOp.getResult()});
    b.create<affine::AffineStoreOp>(storeOp.getLoc(), newArithOp.getResult(), storeOp.getMemRef(),
                                    storeOp.getAffineMapAttr().getValue(), storeOp.getIndices());
    return newArithOp.getOperation();
  }
  ~CreateArithOp() {}
  Operation *arithOp;
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
    .Case([&](arith::MinNumFOp) { newArithOp = rewriter.create<arith::MinNumFOp>(); })
    .Case([&](arith::MaxNumFOp) { newArithOp = rewriter.create<arith::MaxNumFOp>(); })
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

static affine::AffineLoadOp getLoadOp(affine::AffineForOp reduceForOp, Operation *arithOp) {
  auto lhs = arithOp->getOperands()[0];
  auto rhs = arithOp->getOperands()[1];
  auto iv = reduceForOp.getInductionVar();
  affine::AffineLoadOp loadOp;
  reduceForOp.walk([&](affine::AffineLoadOp op) {
    if (op != lhs.getDefiningOp() && op != rhs.getDefiningOp()) {
      return;
    }
    if (llvm::find(op.getIndices(), iv) == op.getIndices().end()) {
      loadOp = op;
    }
  });
  return loadOp;
}

static Operation *findReduceArithOp(Operation *curOp) {
  Operation *reduceArithOp = nullptr;
  curOp->walk([&](Operation *op) -> WalkResult {
    if (op->getAttr(kReductionTypeStr)) {
      bool alreadyConverted = llvm::any_of(op->getOperands(), [](Value v) {
        return !v.getDefiningOp();
      });
      if (!alreadyConverted) {
        reduceArithOp = op;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return reduceArithOp;
}

static arith::ConstantOp resolveInitConstant(affine::AffineStoreOp initStoreOp) {
  auto definingOp = initStoreOp.getValue().getDefiningOp();
  arith::ConstantOp constOp = dyn_cast<arith::ConstantOp>(definingOp);
  if (!constOp) {
    if (auto initLoad = dyn_cast<affine::AffineLoadOp>(definingOp)) {
      Value loadMemref = initLoad.getMemRef();
      for (auto it = Block::reverse_iterator(initLoad.getOperation());
           it != initLoad->getBlock()->rend(); ++it) {
        if (auto precedingStore = dyn_cast<affine::AffineStoreOp>(&*it)) {
          if (precedingStore.getMemRef() == loadMemref) {
            constOp = dyn_cast<arith::ConstantOp>(precedingStore.getValue().getDefiningOp());
            break;
          }
        }
      }
    }
  }
  return constOp;
}

static affine::AffineStoreOp findMatchingStoreOp(affine::AffineForOp reduceLoop, affine::AffineLoadOp loadOp) {
  affine::AffineStoreOp storeOp = nullptr;
  reduceLoop.walk([&](affine::AffineStoreOp op) {
    if (op.getMemref() == loadOp.getMemref()) {
      storeOp = op;
    }
  });
  return storeOp;
}

void AffineIteratorConversion::loadRemoveEachBand(Operation *curOp) {
  OpBuilder b(curOp);
  Operation *reduceArithOp = findReduceArithOp(curOp);
  if (!reduceArithOp) {
    return;
  }

  auto axesAttr = reduceArithOp->getAttrOfType<ArrayAttr>(kReductionAxesStr);
  if (!axesAttr || axesAttr.empty()) {
    return;
  }
  SmallVector<affine::AffineForOp, 4> enclosingLoops;
  affine::getAffineForIVs(*reduceArithOp, &enclosingLoops);
  auto idx = cast<IntegerAttr>(axesAttr[0]).getInt();
  if (idx < 0 || idx >= static_cast<int64_t>(enclosingLoops.size())) {
    return;
  }
  Operation *reduceLoopOp = enclosingLoops[idx].getOperation();
  affine::AffineStoreOp initStoreOp = nullptr;
  while (isa<affine::AffineForOp>(reduceLoopOp) && reduceLoopOp->getAttr(kReductionLoopAttr)) {
    affine::AffineForOp reduceLoop = cast<affine::AffineForOp>(reduceLoopOp);
    affine::AffineLoadOp loadOp = getLoadOp(reduceLoop, reduceArithOp);
    auto initOp = CommonUtils::getReduceInitOp(reduceArithOp, curOp->getBlock());
    if (initOp) {
      initStoreOp = initOp;
    }

    affine::AffineStoreOp storeOp = findMatchingStoreOp(reduceLoop, loadOp);
    if (!storeOp || !initStoreOp || !loadOp) {
      reduceArithOp->emitError("Error: the statement associated with reduce is not found. \n");
      return;
    }
    arith::ConstantOp constOp = resolveInitConstant(initStoreOp);
    if (!constOp) {
      return;
    }
    IRRewriter rewriter(curOp->getContext());
    auto newLoop = cast<affine::AffineForOp>(*reduceLoop.replaceWithAdditionalYields(
      rewriter, constOp.getResult(),
      /*replaceInitOperandUsesInLoop=*/false, [&](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBbArgs) {
        return SmallVector<Value>{storeOp.getValue()};
      }));
    newLoop->setAttr(kReductionLoopAttr, b.getUnitAttr());
    loadOp.getResult().replaceUsesWithIf(newLoop.getBody()->getArguments().back(), [&](OpOperand &use) {
      Operation *user = use.getOwner();
      return newLoop->isProperAncestor(user);
    });
    b.setInsertionPointAfter(newLoop.getOperation());
    auto parentOp = newLoop.getOperation()->getParentOp();
    if (isa<affine::AffineForOp>(parentOp) && parentOp->getAttr(kReductionLoopAttr)) {
      CreateArithOp arithOpCreater(b, newLoop, loadOp, storeOp, reduceArithOp);
      reduceArithOp = identifyAndCreateArithOp(arithOpCreater);
    } else {
      b.create<affine::AffineStoreOp>(storeOp.getLoc(), newLoop.getResults().back(), storeOp.getMemRef(),
                                      storeOp.getAffineMapAttr().getValue(), storeOp.getIndices());
    }
    loadOp.erase();
    storeOp.erase();
    reduceLoopOp = newLoop.getOperation()->getParentOp();
  }
  affine::AffineIfOp ifOp = dyn_cast<affine::AffineIfOp>(initStoreOp.getOperation()->getParentOp());
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
  if (opType != OperatorTemplate::Reduction) {
    return;
  }

  // The reduce axis sinks to the innermost layer.
  func.walk([&](mlir::affine::AffineForOp inner) {
    if (auto outer = mlir::dyn_cast<mlir::affine::AffineForOp>(inner->getParentOp())) {
      if (mlir::CommonUtils::isReduceAxis(func, inner->getParentOp())) {
        mlir::affine::interchangeLoops(outer, inner);
      }
    }
  });

  removeInitMemoryCopy(func);

  SmallVector<Operation *, 8> reduceLoops = CommonUtils::collectReductionAxes(func);
  for (auto reduceLoop : reduceLoops) {
    reduceLoop->setAttr(kReductionLoopAttr, b.getUnitAttr());
  }

  SmallVector<affine::AffineForOp, 6> bands;
  (void)std::copy(func.getOps<affine::AffineForOp>().begin(), func.getOps<affine::AffineForOp>().end(),
                  std::back_inserter(bands));
  for (auto band : bands) {
    int reductionCount = 0;
    band.walk([&](Operation *op) {
      if (op->getAttr(kReductionTypeStr)) {
        reductionCount++;
      }
    });
    for (int i = 0; i < reductionCount; ++i) {
      bool hasReduction = false;
      band.walk([&](Operation *op) {
        if (op->getAttr(kReductionTypeStr)) {
          hasReduction = true;
        }
      });
      if (!hasReduction) {
        break;
      }
      loadRemoveEachBand(band);
    }
  }

  // remove empty for
  func->walk([&](affine::AffineForOp forOp) {
    if (isa<affine::AffineYieldOp>(forOp.getBody()->front())) {
      forOp.erase();
    }
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> createAffineIteratorConversionPass() {
  return std::make_unique<AffineIteratorConversion>();
}

}  // namespace mlir
