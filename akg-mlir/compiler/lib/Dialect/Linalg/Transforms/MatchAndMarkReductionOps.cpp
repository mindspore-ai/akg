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

#include "akg/Dialect/Linalg/Transforms/MatchAndMarkReductionOps.h"

#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

namespace mlir {
#define GEN_PASS_DECL_MATCHANDMARKREDUCTIONOPS
#define GEN_PASS_DEF_MATCHANDMARKREDUCTIONOPS
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace linalg {
namespace {

static void MatchAndMarkRedOpInLinalg(Operation *funcOp) {
  OpBuilder builder(funcOp);
  (void)funcOp->walk([&](linalg::GenericOp genericOp) {
    auto iteratorTypes = genericOp.getIteratorTypesArray();
    SmallVector<mlir::Attribute> intAttrs;
    int axis = 0;
    int reduceAxis = 0;
    bool is_reduction_x = false;
    for (auto it : iteratorTypes) {
      if (it == utils::IteratorType::reduction) {
        reduceAxis++;
        auto intAttr = builder.getIntegerAttr(builder.getIndexType(), axis);
        intAttrs.push_back(intAttr);
        if (static_cast<size_t>(axis) == iteratorTypes.size() - 1) {
          is_reduction_x = true;
        }
      }
      axis++;
    }
    ArrayAttr axesAttr = builder.getArrayAttr(intAttrs);
    // if has reduction axis
    if (axesAttr.size() >= 1) {
      Operation *yield_op = &genericOp.getRegion().front().getOperations().back();
      Operation *op = yield_op->getOperand(0).getDefiningOp();
      op->setAttr(kReductionAxesStr, axesAttr);
      ReduceDirection reduceDirection = ReduceDirection::UNKNOWN;
      if (reduceAxis == axis) {
        reduceDirection = ReduceDirection::ALL;
      } else if (is_reduction_x) {
        reduceDirection = ReduceDirection::X;
      } else {
        reduceDirection = ReduceDirection::Y;
      }
      auto strAttr = builder.getStringAttr(reduceDirectionMap.at(reduceDirection));
      op->setAttr(kReductionTypeStr, strAttr);
      akgglobal::GpuScheduleTool::getInstance().setReduceDirection((size_t)reduceDirection);
    }
  });
}

// Update reduction_axes in affine dialect
static void MatchAndMarkRedOpInAffine(Operation *funcOp) {
  OpBuilder builder(funcOp);
  SmallVector<Operation *, 8> reduceLoops = CommonUtils::collectReductionAxes(funcOp);
  for (auto reduceLoop : reduceLoops) {
    reduceLoop->setAttr(kReductionLoopAttr, builder.getUnitAttr());
  }
  (void)funcOp->walk([&](Operation *redOp) {
    if (!isa<mlir::func::FuncOp>(redOp) && redOp->hasAttr(kReductionAxesStr)) {
      SmallVector<bool, 8> redFlags(false);
      auto curOp = redOp;
      while (curOp) {
        if (isa<affine::AffineForOp>(curOp)) {
          if (curOp->hasAttr(kReductionLoopAttr)) {
            redFlags.push_back(true);
          } else {
            redFlags.push_back(false);
          }
        }
        curOp = curOp->getParentOp();
      }
      std::reverse(redFlags.begin(), redFlags.end());

      // re-set reduction_axes properly
      SmallVector<mlir::Attribute> intAttrs;
      for (size_t i = 0; i < redFlags.size(); i++) {
        if (redFlags[i]) {
          auto intAttr = builder.getIntegerAttr(builder.getIndexType(), i);
          intAttrs.push_back(intAttr);
        }
      }
      ArrayAttr axesAttr = builder.getArrayAttr(intAttrs);
      redOp->setAttr(kReductionAxesStr, axesAttr);
    }
  });
}

struct MatchAndMarkReductionOps : public impl::MatchAndMarkReductionOpsBase<MatchAndMarkReductionOps> {
  MatchAndMarkReductionOps() = default;
  explicit MatchAndMarkReductionOps(const std::string &dialect) { this->dialect = dialect; }
  void runOnOperation() override {
    Operation *funcOp = getOperation();

    if (!(funcOp->hasAttr(kOperatorTypeStr) && dyn_cast<StringAttr>(funcOp->getAttr(kOperatorTypeStr)) == kReduceStr)) {
      return;
    }
    if (this->dialect == "linalg") {
      MatchAndMarkRedOpInLinalg(funcOp);
    } else if (this->dialect == "affine") {
      MatchAndMarkRedOpInAffine(funcOp);
    } else {
      std::string errorMsg = "MatchAndMarkReductionOps got a unknown dialect = " + this->dialect + ", pass failed.";
      funcOp->emitError(errorMsg);
    }
  }
};

}  // namespace
}  // namespace linalg
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createMatchAndMarkReductionOpsPass() {
  return std::make_unique<linalg::MatchAndMarkReductionOps>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createMatchAndMarkReductionOpsPass(std::string dialect) {
  return std::make_unique<linalg::MatchAndMarkReductionOps>(dialect);
}
