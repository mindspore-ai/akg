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

#include "akg/Dialect/Affine/Transforms/AffineHandleBoundaryIfExtract.h"
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
#define GEN_PASS_DEF_AFFINEHANDLEBOUNDARYIFEXTRACT
#define GEN_PASS_DECL_AFFINEHANDLEBOUNDARYIFEXTRACT
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace akgglobal;

namespace {

struct AffineHandleBoundaryIfExtract : public impl::AffineHandleBoundaryIfExtractBase<AffineHandleBoundaryIfExtract> {
  AffineHandleBoundaryIfExtract() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mindspore::MindSporeDialect>();
  }
  void runOnOperation() override;
  bool isBoundaryIf(Operation *outerIfOp);
};

bool AffineHandleBoundaryIfExtract::isBoundaryIf(Operation *outerIfOp) {
  if (outerIfOp->getPrevNode()) {
    return false;
  }
  if (outerIfOp->getNextNode() && !isa<AffineYieldOp>(outerIfOp->getNextNode())) {
    return false;
  }
  if (!isa<AffineForOp>(outerIfOp->getParentOp())) {
    return false;
  }
  for (auto operand : outerIfOp->getOperands()) {
    if (isa<BlockArgument>(operand)) {
      continue;
    }
    if (auto op = operand.getDefiningOp()) {
      if (isa<memref::DimOp>(op)) {
        continue;
      }
    }
    return false;
  }
  return true;
}

void AffineHandleBoundaryIfExtract::runOnOperation() {
  auto funcOp = getOperation();
  Operation *outerIfOp = nullptr;
  (void)funcOp->walk([&](Operation *op) {
    if (isa<AffineIfOp>(op) && isBoundaryIf(op)) {
      outerIfOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!outerIfOp) {
    return;
  }
  OpBuilder builder(funcOp);
  builder.setInsertionPoint(outerIfOp);
  auto ifOp = dyn_cast<AffineIfOp>(outerIfOp);
  for (auto &op : llvm::make_early_inc_range(ifOp.getThenRegion().front())) {
    if (!isa<AffineYieldOp>(op)) {
      mlir::Operation *clonedOp = builder.clone(op);
      op.replaceAllUsesWith(clonedOp);
    }
  }
  auto &thenRegion = ifOp.getThenRegion();
  auto *newBlock = new Block();
  thenRegion.getBlocks().clear();
  thenRegion.getBlocks().push_back(newBlock);
  builder.setInsertionPointToEnd(newBlock);
  auto loc = ifOp.getLoc();
  mlir::Attribute constAttr = builder.getIndexAttr(0);
  auto constant = builder.create<mlir::arith::ConstantOp>(loc, constAttr);
  auto keep = builder.create<mindspore::KeepArgsOp>(loc, constant, constant);
  keep.getOperation()->setAttr("BoundaryIf", builder.getUnitAttr());
  builder.create<mlir::AffineYieldOp>(loc);
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAffineHandleBoundaryIfExtract() {
  return std::make_unique<AffineHandleBoundaryIfExtract>();
}
