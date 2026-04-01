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
#include <iterator>

#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DECL_AFFINEITERATORCONVERSION
#define GEN_PASS_DEF_AFFINEITERATORCONVERSION
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "affine-iterator-conversion"

namespace mlir {

struct AffineIteratorConversion : public impl::AffineIteratorConversionBase<AffineIteratorConversion> {
  void runOnOperation() override;

 private:
  void convertReduction(Operation *band);
  void removeInitMemoryCopy(func::FuncOp func);
};

// Creates a new arith op matching the original reduction type, using the
// loop's iter_arg result as the accumulator. Used for nested reduction axes.
struct ReductionArithCreator {
  OpBuilder b;
  affine::AffineForOp newLoop;
  affine::AffineLoadOp loadOp;
  affine::AffineStoreOp storeOp;
  Operation *arithOp;

  template <typename OpTy>
  Operation *create() {
    auto newLoad = b.create<affine::AffineLoadOp>(loadOp.getLoc(), loadOp.getMemRef(),
                                                  loadOp.getAffineMapAttr().getValue(), loadOp.getIndices());
    auto newArith = b.create<OpTy>(newLoad.getLoc(), newLoad.getResult().getType(),
                                   ValueRange{newLoop.getResults().back(), newLoad.getResult()});
    b.create<affine::AffineStoreOp>(storeOp.getLoc(), newArith.getResult(), storeOp.getMemRef(),
                                    storeOp.getAffineMapAttr().getValue(), storeOp.getIndices());
    return newArith.getOperation();
  }

  Operation *createMatchingOp() {
    Operation *result = nullptr;
    llvm::TypeSwitch<Operation *>(arithOp)
      .Case([&](arith::AddFOp) { result = create<arith::AddFOp>(); })
      .Case([&](arith::MulFOp) { result = create<arith::MulFOp>(); })
      .Case([&](arith::AddIOp) { result = create<arith::AddIOp>(); })
      .Case([&](arith::AndIOp) { result = create<arith::AndIOp>(); })
      .Case([&](arith::OrIOp) { result = create<arith::OrIOp>(); })
      .Case([&](arith::MulIOp) { result = create<arith::MulIOp>(); })
      .Case([&](arith::MinNumFOp) { result = create<arith::MinNumFOp>(); })
      .Case([&](arith::MaxNumFOp) { result = create<arith::MaxNumFOp>(); })
      .Case([&](arith::MinSIOp) { result = create<arith::MinSIOp>(); })
      .Case([&](arith::MaxSIOp) { result = create<arith::MaxSIOp>(); })
      .Case([&](arith::MinUIOp) { result = create<arith::MinUIOp>(); })
      .Case([&](arith::MaxUIOp) { result = create<arith::MaxUIOp>(); })
      .Default([](Operation *) {});
    return result;
  }
};

static Operation *findReduceArithOp(Operation *root) {
  Operation *result = nullptr;
  root->walk([&](Operation *op) -> WalkResult {
    if (op->getAttr(kReductionTypeStr)) {
      bool converted = llvm::any_of(op->getOperands(), [](Value v) { return !v.getDefiningOp(); });
      if (!converted) {
        result = op;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

// Find the accumulator load: the operand of the arith op whose indices
// do not reference the reduction loop's induction variable.
static affine::AffineLoadOp getAccumulatorLoad(affine::AffineForOp reduceLoop, Operation *arithOp) {
  Value lhs = arithOp->getOperand(0), rhs = arithOp->getOperand(1);
  Value iv = reduceLoop.getInductionVar();
  affine::AffineLoadOp result;
  reduceLoop.walk([&](affine::AffineLoadOp op) {
    if (op != lhs.getDefiningOp() && op != rhs.getDefiningOp()) {
      return;
    }
    if (llvm::find(op.getIndices(), iv) == op.getIndices().end()) {
      result = op;
    }
  });
  return result;
}

// Resolve the init constant from a reduction init store. Handles both direct
// constants and constants accessed through a load chain (e.g. when one
// reduction's init is loaded from another reduction's accumulator).
static arith::ConstantOp resolveInitConstant(affine::AffineStoreOp initStore) {
  auto *defOp = initStore.getValue().getDefiningOp();
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    return constOp;
  }
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(defOp)) {
    Value memref = loadOp.getMemRef();
    for (auto it = Block::reverse_iterator(loadOp.getOperation()); it != loadOp->getBlock()->rend(); ++it) {
      if (auto store = dyn_cast<affine::AffineStoreOp>(&*it)) {
        if (store.getMemRef() == memref) {
          return dyn_cast<arith::ConstantOp>(store.getValue().getDefiningOp());
        }
      }
    }
  }
  return nullptr;
}

static affine::AffineStoreOp findMatchingStore(affine::AffineForOp loop, affine::AffineLoadOp loadOp) {
  affine::AffineStoreOp result;
  loop.walk([&](affine::AffineStoreOp op) {
    if (op.getMemref() == loadOp.getMemref()) {
      result = op;
    }
  });
  return result;
}

// Find the init store for a reduction accumulator. First searches by
// {reduction_init} attribute across the function, then falls back to a
// structural search for a sibling store preceding the reduction loop.
static affine::AffineStoreOp findInitStore(Operation *band, Operation *reduceLoop, Value accumMemRef) {
  auto func = band->getParentOfType<func::FuncOp>();
  affine::AffineStoreOp result;
  func.walk([&](affine::AffineStoreOp store) {
    if (!result && store->getAttr(kReductionInitAttr) && store.getMemRef() == accumMemRef) {
      result = store;
    }
  });
  if (result) {
    return result;
  }
  if (Block *block = reduceLoop->getBlock()) {
    for (auto it = Block::reverse_iterator(reduceLoop); it != block->rend(); ++it) {
      if (auto store = dyn_cast<affine::AffineStoreOp>(&*it)) {
        if (store.getMemRef() == accumMemRef) {
          return store;
        }
      }
    }
  }
  return nullptr;
}

static affine::AffineForOp replaceWithIterArgs(OpBuilder &b, Operation *ctx, affine::AffineForOp loop,
                                               affine::AffineLoadOp loadOp, affine::AffineStoreOp storeOp,
                                               arith::ConstantOp initVal) {
  IRRewriter rewriter(ctx->getContext());
  auto newLoop = cast<affine::AffineForOp>(*loop.replaceWithAdditionalYields(
    rewriter, initVal.getResult(),
    /*replaceInitOperandUsesInLoop=*/false,
    [&](OpBuilder &, Location, ArrayRef<BlockArgument>) { return SmallVector<Value>{storeOp.getValue()}; }));
  newLoop->setAttr(kReductionLoopAttr, b.getUnitAttr());
  loadOp.getResult().replaceUsesWithIf(newLoop.getBody()->getArguments().back(),
                                       [&](OpOperand &use) { return newLoop->isProperAncestor(use.getOwner()); });
  return newLoop;
}

static void eraseInitStore(affine::AffineStoreOp initStore) {
  if (!initStore) {
    return;
  }
  auto ifOp = dyn_cast<affine::AffineIfOp>(initStore->getParentOp());
  // Save the defining op of the stored value before erasing the store,
  // so we can clean up dead intermediate loads from the init chain
  // (e.g. store 0->A, load A->%0, store %0->B: after erasing the B store,
  //  the load becomes dead and should also be removed).
  auto *defOp = initStore.getValue().getDefiningOp();
  initStore.erase();
  if (ifOp) {
    ifOp.erase();
  }
  if (defOp && isa<affine::AffineLoadOp>(defOp) && defOp->use_empty()) {
    defOp->erase();
  }
}

void AffineIteratorConversion::removeInitMemoryCopy(func::FuncOp func) {
  memref::CopyOp initCopy;
  func.walk([&](memref::CopyOp copyOp) {
    if (copyOp.getTarget().getDefiningOp() && isa<memref::AllocOp>(copyOp.getSource().getDefiningOp()) &&
        isa<memref::AllocOp>(copyOp.getTarget().getDefiningOp())) {
      initCopy = copyOp;
    }
  });
  if (!initCopy) {
    return;
  }
  Value target = initCopy.getTarget();
  target.replaceAllUsesWith(initCopy.getSource());
  target.getDefiningOp()->erase();
  initCopy.erase();
}

void AffineIteratorConversion::convertReduction(Operation *band) {
  OpBuilder b(band);
  Operation *arithOp = findReduceArithOp(band);
  if (!arithOp) {
    return;
  }

  auto axesAttr = arithOp->getAttrOfType<ArrayAttr>(kReductionAxesStr);
  if (!axesAttr || axesAttr.empty()) {
    return;
  }

  SmallVector<affine::AffineForOp, 4> enclosingLoops;
  affine::getAffineForIVs(*arithOp, &enclosingLoops);

  Operation *reduceLoopOp = nullptr;
  for (auto it = enclosingLoops.rbegin(); it != enclosingLoops.rend(); ++it) {
    if ((*it)->getAttr(kReductionLoopAttr)) {
      reduceLoopOp = it->getOperation();
      break;
    }
  }
  if (!reduceLoopOp) {
    return;
  }

  affine::AffineStoreOp initStoreOp;
  while (isa<affine::AffineForOp>(reduceLoopOp) && reduceLoopOp->getAttr(kReductionLoopAttr)) {
    auto reduceLoop = cast<affine::AffineForOp>(reduceLoopOp);
    auto loadOp = getAccumulatorLoad(reduceLoop, arithOp);
    if (!loadOp) {
      arithOp->emitError("reduction accumulator load not found");
      return;
    }

    if (!initStoreOp) {
      initStoreOp = findInitStore(band, reduceLoopOp, loadOp.getMemRef());
    }
    if (!initStoreOp) {
      arithOp->emitError("reduction init store not found");
      return;
    }

    auto storeOp = findMatchingStore(reduceLoop, loadOp);
    if (!storeOp) {
      arithOp->emitError("reduction accumulator store not found");
      return;
    }

    auto constOp = resolveInitConstant(initStoreOp);
    if (!constOp) {
      return;
    }

    auto newLoop = replaceWithIterArgs(b, band, reduceLoop, loadOp, storeOp, constOp);
    b.setInsertionPointAfter(newLoop);
    auto parentOp = newLoop->getParentOp();
    if (isa<affine::AffineForOp>(parentOp) && parentOp->getAttr(kReductionLoopAttr)) {
      ReductionArithCreator creator{b, newLoop, loadOp, storeOp, arithOp};
      arithOp = creator.createMatchingOp();
    } else {
      b.create<affine::AffineStoreOp>(storeOp.getLoc(), newLoop.getResults().back(), storeOp.getMemRef(),
                                      storeOp.getAffineMapAttr().getValue(), storeOp.getIndices());
    }
    loadOp.erase();
    storeOp.erase();
    reduceLoopOp = newLoop->getParentOp();
  }
  eraseInitStore(initStoreOp);
}

void AffineIteratorConversion::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder b(func);
  if (CommonUtils::getOperatorType(func) != OperatorTemplate::Reduction) {
    return;
  }

  // Sink reduce axes to innermost. interchangeLoops moves the Operation
  // objects themselves (not just bounds/IVs), so {reduction} attributes
  // follow the original Operation to its new (inner) position automatically.
  func.walk([&](affine::AffineForOp inner) {
    if (auto outer = dyn_cast<affine::AffineForOp>(inner->getParentOp())) {
      if (CommonUtils::isReduceAxis(func, inner->getParentOp())) {
        affine::interchangeLoops(outer, inner);
      }
    }
  });

  removeInitMemoryCopy(func);

  // Supplement: collectReductionAxes can discover reduction loops for IR
  // that lacks explicit {reduction} attributes on loops.
  for (auto *loop : CommonUtils::collectReductionAxes(func)) {
    loop->setAttr(kReductionLoopAttr, b.getUnitAttr());
  }

  SmallVector<affine::AffineForOp, 6> bands;
  std::copy(func.getOps<affine::AffineForOp>().begin(), func.getOps<affine::AffineForOp>().end(),
            std::back_inserter(bands));
  for (auto band : bands) {
    int reductionCount = 0;
    band.walk([&](Operation *op) {
      if (op->getAttr(kReductionTypeStr)) {
        reductionCount++;
      }
    });
    for (int i = 0; i < reductionCount; ++i) {
      convertReduction(band);
    }
  }

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
