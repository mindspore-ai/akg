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

#include "akg/Dialect/SCF/Transforms/RewriteReduceInMultiLevelMemory.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace akgglobal;

namespace mlir {
#define GEN_PASS_DECL_REWRITEREDUCEINMULTILEVELMEMORY
#define GEN_PASS_DEF_REWRITEREDUCEINMULTILEVELMEMORY
#include "akg/Dialect/SCF/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace mlir {
namespace scf {
namespace {

void cloneAndReplaceOps(SmallVector<Operation *, 8> &ops, OpBuilder &builder) {
  for (Operation *op : ops) {
    if (op) {
      Operation *clonedOp = builder.clone(*op);
      op->replaceAllUsesWith(clonedOp);
      op->erase();
    }
  }
}

Value createInitialValue(Operation *op, mlir::Location loc, OpBuilder &builder) {
  Type elementType = op->getResultTypes()[0];
  Value initialValue;
  if (isa<arith::AddFOp>(op)) {
    initialValue =
      builder.create<arith::ConstantFloatOp>(loc, APFloat(APFloat::IEEEsingle(), "0.0"), cast<FloatType>(elementType));
  } else if (isa<arith::MulFOp>(op)) {
    initialValue =
      builder.create<arith::ConstantFloatOp>(loc, APFloat(APFloat::IEEEsingle(), "1.0"), cast<FloatType>(elementType));
  } else if (isa<arith::AddIOp>(op)) {
    initialValue = builder.create<arith::ConstantIntOp>(loc, 0, cast<IntegerType>(elementType));
  } else if (isa<arith::AndIOp>(op)) {
    initialValue = builder.create<arith::ConstantIntOp>(loc, 1, cast<IntegerType>(elementType));
  } else if (isa<arith::OrIOp>(op)) {
    initialValue = builder.create<arith::ConstantIntOp>(loc, 0, cast<IntegerType>(elementType));
  } else if (isa<arith::MulIOp>(op)) {
    initialValue = builder.create<arith::ConstantIntOp>(loc, 1, cast<IntegerType>(elementType));
  } else if (isa<arith::MinNumFOp>(op)) {
    initialValue = builder.create<arith::ConstantFloatOp>(
      loc, APFloat::getLargest(cast<FloatType>(elementType).getFloatSemantics()), cast<FloatType>(elementType));
  } else if (isa<arith::MaxNumFOp>(op)) {
    initialValue = builder.create<arith::ConstantFloatOp>(
      loc, APFloat::getSmallest(cast<FloatType>(elementType).getFloatSemantics()), cast<FloatType>(elementType));
  } else if (isa<arith::MinSIOp>(op)) {
    initialValue =
      builder.create<arith::ConstantIntOp>(loc, std::numeric_limits<int64_t>::max(), cast<IntegerType>(elementType));
  } else if (isa<arith::MaxSIOp>(op)) {
    initialValue = builder.create<arith::ConstantIntOp>(loc, std::numeric_limits<int64_t>::lowest(),
                                                        cast<IntegerType>(elementType));
  } else if (isa<arith::MinUIOp>(op)) {
    initialValue =
      builder.create<arith::ConstantIntOp>(loc, std::numeric_limits<uint64_t>::max(), cast<IntegerType>(elementType));
  } else if (isa<arith::MaxUIOp>(op)) {
    initialValue = builder.create<arith::ConstantIntOp>(loc, 0, cast<IntegerType>(elementType));
  } else {
    (void)op->emitError("Unsupported operation type\n");
    return nullptr;
  }
  return initialValue;
}

// Try to match this pattern in sequential reduction loop:
static std::tuple<Operation *, Operation *, Operation *, Operation *, Operation *> matchReductionRelatedOps(
  Operation *funcOp, Operation *redOp) {
  Operation *allocLocalA = nullptr, *initLoadLocalA = nullptr, *initStoreLocalA = nullptr, *loadLocalA = nullptr,
            *storeLocalA = nullptr;
  Value localA = redOp->getOperands()[1];
  loadLocalA = localA.getDefiningOp();
  storeLocalA = *(redOp->getUsers().begin());
  allocLocalA = loadLocalA->getOperand(0).getDefiningOp();
  (void)funcOp->walk([&](memref::StoreOp storeOp) -> WalkResult {
    if (storeOp.getMemref() == allocLocalA->getResult(0)) {
      initStoreLocalA = storeOp.getOperation();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  initLoadLocalA = initStoreLocalA->getOperand(0).getDefiningOp();
  if (!allocLocalA || !initLoadLocalA || !initStoreLocalA || !loadLocalA || !storeLocalA) {
    (void)redOp->emitError("matchReductionRelatedOps cannot match all ops, plz check this pattern.");
  }
  return std::make_tuple(allocLocalA, initLoadLocalA, initStoreLocalA, loadLocalA, storeLocalA);
}

static void removeInitOuput(Operation *funcOp, mlir::Value v) {
  funcOp->walk([&](memref::StoreOp storeOp) {
    // constant init like :memref.store %cst, %arg1[] : memref<f32>
    if (storeOp.getOperand(1) == v && isa<arith::ConstantOp>(storeOp.getOperand(0).getDefiningOp())) {
      storeOp.erase();
    }
  });
}

static Operation *getOutermostSeqLoop(Operation *redOp) {
  Operation *outerSeqReduceLoop = nullptr;
  auto curOp = redOp;
  while (curOp) {
    if (isa<scf::ParallelOp>(curOp) && curOp->getAttr("reduceLoop")) {
      if (gpu::GpuAttrUtils::getProcessorFromParallelOp(curOp) == gpu::Processor::Sequential) {
        outerSeqReduceLoop = curOp;
      } else {
        break;
      }
    }
    curOp = curOp->getParentOp();
  }
  return outerSeqReduceLoop;
}

static Operation *getPostLoadLocal(Operation *outerSeqReduceLoop, memref::LoadOp loadLocalAOp) {
  Operation *loadLocalAPost = nullptr;
  auto curOp = outerSeqReduceLoop->getNextNode();

  while (curOp) {
    if (auto op = dyn_cast<memref::LoadOp>(curOp)) {
      if (op.getMemref() == loadLocalAOp.getMemref() && op.getIndices() == loadLocalAOp.getIndices()) {
        loadLocalAPost = curOp;
        break;
      }
    }
    curOp = curOp->getNextNode();
  }
  return loadLocalAPost;
}

/*
  This pass change the logic, from reduction on single memory level to multi memory level.
*/
struct RewriteReduceInMultiLevelMemory
    : public impl::RewriteReduceInMultiLevelMemoryBase<RewriteReduceInMultiLevelMemory> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<Operation *, 2> redOps;
    bool isReduceY = GpuScheduleTool::getInstance().getReduceDirection() == (unsigned long)ReduceDirection::Y;
    funcOp.walk([&](Operation *op) {
      if (!isa<mlir::func::FuncOp>(op)) {
        bool parallelReduce = (op->hasAttr(akg::utils::kEnableParallelReduce) &&
                               op->getAttrOfType<BoolAttr>(akg::utils::kEnableParallelReduce).getValue());
        bool atomicAdd = (op->hasAttr(akg::utils::kEnableAtomicAdd) &&
                          op->getAttrOfType<BoolAttr>(akg::utils::kEnableAtomicAdd).getValue());
        if (parallelReduce || (atomicAdd && isReduceY)) {
          int cacheLevelB = CommonUtils::getCacheLevel(funcOp, op->getOperands()[1]);
          if (cacheLevelB != 1) {
            redOps.push_back(op);
          } else {
            (void)op->emitError(
              "reduction related operands are not promoted, can't apply "
              "RewriteReduceInMultiLevelMemory.");
          }
        }
      }
    });

    OpBuilder builder(funcOp);

    for (auto redOp : redOps) {
      Operation *outerSeqReduceLoop = getOutermostSeqLoop(redOp);
      if (!outerSeqReduceLoop) {
        continue;
      }

      Operation *allocLocalA = nullptr, *initLoadLocalA = nullptr, *initStoreLocalA = nullptr, *loadLocalA = nullptr,
                *storeLocalA = nullptr;
      std::tie(allocLocalA, initLoadLocalA, initStoreLocalA, loadLocalA, storeLocalA) =
        matchReductionRelatedOps(funcOp, redOp);

      // Rewrite logic:
      // (1) On gpu backend, the temp buffer initialization should be set properly, but not by output data;
      // (2) And it is invalid to initialize reduction output in every thread. In fact, we do not need to read
      // the reduction output data in AI dataflow.
      builder.setInsertionPoint(allocLocalA);
      mlir::Location loc = allocLocalA->getLoc();
      auto initValue = createInitialValue(redOp, loc, builder);
      if (!initValue) {
        return signalPassFailure();
      }
      initStoreLocalA->setOperand(0, initValue);
      if (isa<memref::LoadOp>(initLoadLocalA)) {
        removeInitOuput(funcOp, initLoadLocalA->getOperand(0));
        initLoadLocalA->erase();
      }

      // NOTE:reduce op should be the last op in sequential loop. if there is any other op, move
      // it below thread-loop; also if some of alloc/deallocs break the relationship, move them
      // either.
      SmallVector<Operation *, 8> needMoveBeforeOps;
      SmallVector<mlir::Value, 8> usedValues;

      CommonUtils::getAllPreviousRelatedOpsV2(loadLocalA, needMoveBeforeOps, usedValues);
      std::reverse(needMoveBeforeOps.begin(), needMoveBeforeOps.end());
      builder.setInsertionPoint(outerSeqReduceLoop);
      cloneAndReplaceOps(needMoveBeforeOps, builder);

      SmallVector<Operation *, 8> needMoveAfterOps;
      usedValues.clear();
      CommonUtils::getAllNextRelatedOps(storeLocalA, needMoveAfterOps, usedValues);
      builder.setInsertionPointAfter(outerSeqReduceLoop);
      cloneAndReplaceOps(needMoveAfterOps, builder);

      auto loadLocalAOp = dyn_cast<memref::LoadOp>(loadLocalA);
      Operation *loadLocalAPost = getPostLoadLocal(outerSeqReduceLoop, loadLocalAOp);

      if (!loadLocalAPost) {
        (void)funcOp->emitError(
          "memref.load in thread-level for reduction op does not exist, please check the .mlir file.");
        signalPassFailure();
      }
      builder.setInsertionPoint(loadLocalAPost);
      loc = loadLocalAPost->getLoc();

      auto loadLocalTemp = builder.create<memref::LoadOp>(loc, loadLocalAOp.getMemref(), loadLocalAOp.getIndices());
      auto initValue2 = createInitialValue(redOp, loc, builder);
      if (!initValue2) {
        return signalPassFailure();
      }
      Value reducePost =
        CommonUtils::cloneOpWithNetOperands(builder, loc, redOp, loadLocalTemp.getResult(), initValue2);
      (void)builder.create<memref::StoreOp>(loc, reducePost, loadLocalTemp.getMemref(), loadLocalTemp.getIndices());
      Operation *reducePostOp = reducePost.getDefiningOp();

      for (const auto &attr : redOp->getAttrs()) {
        reducePostOp->setAttr(attr.getName(), attr.getValue());
      }
      (void)redOp->removeAttr(mlir::akg::utils::kEnableParallelReduce);
      (void)redOp->removeAttr(kReductionTypeStr);
      (void)redOp->removeAttr(kReductionAxesStr);
      (void)redOp->removeAttr(mlir::akg::utils::kEnableAtomicAdd);
    }
  }
};

}  // namespace
}  // namespace scf
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createRewriteReduceInMultiLevelMemoryPass() {
  return std::make_unique<scf::RewriteReduceInMultiLevelMemory>();
}
