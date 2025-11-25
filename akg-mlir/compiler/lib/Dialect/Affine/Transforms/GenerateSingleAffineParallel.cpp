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

#include "akg/Dialect/Affine/Transforms/GenerateSingleAffineParallel.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DEF_GENERATESINGLEAFFINEPARALLEL
#define GEN_PASS_DECL_GENERATESINGLEAFFINEPARALLEL
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace llvm;

namespace {

class GenerateSingleAffineParallelPass
    : public impl::GenerateSingleAffineParallelBase<GenerateSingleAffineParallelPass> {
 public:
  explicit GenerateSingleAffineParallelPass() {}
  explicit GenerateSingleAffineParallelPass(bool isDynamicShape) {
    if (isDynamicShape) {
      forceGen = 1;
    }
  }

  void runOnOperation() override;

  void generateWithoutLoop();
  void generateWithLoop(affine::AffineForOp outerLoop);
  void updateReductionLoop();

 private:
  void sinkOps(affine::AffineForOp pointLoop);
};
}  // namespace

void GenerateSingleAffineParallelPass::generateWithoutLoop() {
  func::FuncOp funcOp = getOperation();
  Operation *topLoop = nullptr;
  SmallVector<Operation *> restOp;
  auto &currentBlock = funcOp.front();
  auto terminator = currentBlock.getTerminator();
  funcOp->walk([&](Operation *op) {
    if (op == terminator) {
      return;
    }
    if (topLoop == nullptr) {
      topLoop = op;
    }
    restOp.push_back(op);
  });
  if (topLoop != nullptr) {
    Location loc = topLoop->getLoc();
    OpBuilder b(topLoop);
    affine::AffineForOp pointLoop = b.create<affine::AffineForOp>(loc, 0, 1);
    AffineMap lowerBoundMap = pointLoop.getLowerBoundMap();
    ValueRange lowerBoundOperands = pointLoop.getLowerBoundOperands();
    AffineMap upperBoundMap = pointLoop.getUpperBoundMap();
    ValueRange upperBoundOperands = pointLoop.getUpperBoundOperands();

    affine::AffineParallelOp parallelLoop = b.create<affine::AffineParallelOp>(
      loc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), llvm::ArrayRef(lowerBoundMap), lowerBoundOperands,
      llvm::ArrayRef(upperBoundMap), upperBoundOperands, llvm::ArrayRef(pointLoop.getStepAsInt()));
    for (auto it = restOp.begin(); it != std::prev(restOp.end()); ++it) {
      auto op = *it;
      parallelLoop.getBody()->getOperations().splice(std::prev(parallelLoop.getBody()->end()),
                                                     op->getBlock()->getOperations(), op);
    }
    pointLoop->erase();
  }
}

void GenerateSingleAffineParallelPass::generateWithLoop(affine::AffineForOp outerLoop) {
  Location loc = outerLoop.getLoc();
  Operation *topLoop = outerLoop.getOperation();
  OpBuilder b(topLoop);
  affine::AffineForOp pointLoop = b.create<affine::AffineForOp>(loc, 0, 1);
  pointLoop.getBody()->getOperations().splice(pointLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
                                              topLoop);
  if (forceGen == 1) {
    sinkOps(pointLoop);
  }
}

// This func will sink all the DimOp as well as the ExpandShapeOps users.
// Note that it is designed for gpu kernel outlining and should be invoked carefully.
void GenerateSingleAffineParallelPass::sinkOps(affine::AffineForOp pointLoop) {
  func::FuncOp funcOp = getOperation();
  SmallVector<Operation *> toSink;
  funcOp->walk([&](Operation *op) {
    if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
      for (auto user : op->getUsers()) {
        // These ops are not allowed in gpu func.
        if (isa<memref::AllocOp>(user)) {
          return;
        }
      }
      auto src = dimOp.getSource();
      for (auto user : src.getUsers()) {
        if (isa<memref::ExpandShapeOp>(user)) {
          toSink.push_back(user);
          toSink.push_back(op);
          return;
        }
      }
      toSink.push_back(op);
    }
  });

  if (toSink.empty()) {
    return;
  }
  for (auto it = toSink.begin(); it != toSink.end(); ++it) {
    auto op = *it;
    pointLoop.getBody()->getOperations().splice(pointLoop.getBody()->begin(), op->getBlock()->getOperations(), op);
  }
}

void GenerateSingleAffineParallelPass::updateReductionLoop() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(funcOp);
  (void)funcOp->walk([&](Operation *redOp) {
    if (!isa<mlir::func::FuncOp>(redOp) && redOp->hasAttr(kReductionAxesStr)) {
      ArrayAttr axesArrayAttr = cast<ArrayAttr>(redOp->getAttr(kReductionAxesStr));
      ;
      SmallVector<mlir::Attribute> intAttrs;
      SmallVector<int, 8> flags;
      for (auto axisAttr : axesArrayAttr) {
        auto value = cast<IntegerAttr>(axisAttr).getInt() + 1;
        flags.push_back(value);
        auto intAttr = builder.getIntegerAttr(builder.getIndexType(), value);
        intAttrs.push_back(intAttr);
      }
      ArrayAttr axesAttr = builder.getArrayAttr(intAttrs);
      redOp->setAttr(kReductionAxesStr, axesAttr);
    }
  });
}

void GenerateSingleAffineParallelPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  bool hasLoop = false;
  affine::AffineForOp outerLoop;
  funcOp->walk([&](Operation *op) {
    if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
      outerLoop = loop;
    }
    if (hasLoop) {
      return;
    }
    if (dyn_cast<affine::AffineForOp>(op)) {
      hasLoop = true;
    }
  });
  OperatorTemplate opType = CommonUtils::getOperatorType(getOperation());
  if (!hasLoop) {
    generateWithoutLoop();
  } else if ((forceGen == 1 || opType == OperatorTemplate::Reduce) && outerLoop) {
    generateWithLoop(outerLoop);
  }
  if (opType == OperatorTemplate::Reduce) {
    updateReductionLoop();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createGenerateSingleAffineParallelPass() {
  return std::make_unique<GenerateSingleAffineParallelPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createGenerateSingleAffineParallelPass(bool isDynamicShape) {
  return std::make_unique<GenerateSingleAffineParallelPass>(isDynamicShape);
}
