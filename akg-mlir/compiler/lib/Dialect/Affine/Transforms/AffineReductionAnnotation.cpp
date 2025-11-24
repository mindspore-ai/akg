/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Transforms/AffineReductionAnnotation.h"

#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
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
namespace affine {
#define GEN_PASS_DECL_AFFINEREDUCTIONANNOTATION
#define GEN_PASS_DEF_AFFINEREDUCTIONANNOTATION
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace affine
}  // namespace mlir

using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::MemRefType;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::SmallVector;
using mlir::StringAttr;
using mlir::Value;
using mlir::affine::AffineForOp;
using mlir::affine::AffineLoadOp;
using mlir::affine::AffineStoreOp;
using mlir::cast;
using mlir::dyn_cast;
using mlir::func::FuncOp;
using mlir::isa;
using mlir::kOperatorTypeStr;
using mlir::kReduceStr;
using mlir::kReductionAxesStr;
using mlir::kReductionTypeStr;
using mlir::memref::LoadOp;
using mlir::memref::StoreOp;
using mlir::CommonUtils;
using mlir::ReduceDirection;
using mlir::reduceDirectionMap;
using akgglobal::GpuScheduleTool;

namespace {

struct AffineReductionAnnotation : public affine::impl::AffineReductionAnnotationBase<AffineReductionAnnotation> {
  AffineReductionAnnotation() {}
  void runOnOperation() override {
    Operation *funcOp = getOperation();

    if (!(funcOp->hasAttr(kOperatorTypeStr) &&
          dyn_cast<StringAttr>(funcOp->getAttr(kOperatorTypeStr)) == kReduceStr)) {
      return;
    }
    annotateReductionOps(funcOp);
  }
  void annotateReductionOps(Operation *funcOp);
  bool isReductionPattern(Operation *op);
};
}  // namespace

bool AffineReductionAnnotation::isReductionPattern(Operation *op) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return false;

  // detect loop
  Operation *parentLoop = nullptr;
  Operation *current = op->getParentOp();
  while (current) {
    if (isa<affine::AffineForOp>(current)) {
      parentLoop = current;
      break;
    }
    current = current->getParentOp();
  }
  if (!parentLoop) return false;

  // detect store
  affine::AffineStoreOp storeOp = nullptr;
  for (auto *user : op->getResult(0).getUsers()) {
    if (auto s = dyn_cast<affine::AffineStoreOp>(user)) {
      storeOp = s;
      break;
    }
  }
  if (!storeOp) {
    return false;
  }

  auto outMemrefType = dyn_cast<MemRefType>(storeOp.getMemref().getType());
  if (!outMemrefType) return false;
  int32_t outRank = outMemrefType.getRank();

  auto getLoadRank = [](Value v) -> std::optional<int64_t> {
    if (auto load = v.getDefiningOp<affine::AffineLoadOp>()) {
      if (auto mt = dyn_cast<MemRefType>(load.getMemref().getType()))
        return mt.getRank();
    }
    if (auto mload = v.getDefiningOp<memref::LoadOp>()) {
      if (auto mt = dyn_cast<MemRefType>(mload.getMemref().getType()))
        return mt.getRank();
    }
    return std::nullopt;
  };

  Value a = op->getOperand(0);
  Value b = op->getOperand(1);
  auto aRank = getLoadRank(a);
  auto bRank = getLoadRank(b);

  if (!aRank.has_value() || !bRank.has_value()) {
    return false;
  }

  bool higherThenOutA = (*aRank > outRank);
  bool higherThenOutB = (*bRank > outRank);
  bool equalOutA = (*aRank == outRank);
  bool equalOutB = (*bRank == outRank);

  if ((higherThenOutA && equalOutB) || (higherThenOutB && equalOutA)) {
    return true;
  }
  return false;
}

// Update reduction_axes in affine dialect
void AffineReductionAnnotation::annotateReductionOps(Operation *funcOp) {
  OpBuilder builder(funcOp);

  // identify reduction operations and mark them
  (void)funcOp->walk([&](Operation *op) {
    if (isReductionPattern(op)) {
      // Mark this operation as a reduction
      op->setAttr(kReductionAxesStr, builder.getArrayAttr({}));
    }
  });

  // collect reduction axes and update attributes
  SmallVector<Operation *, 8> reduceLoops = CommonUtils::collectReductionAxes(funcOp);
  for (auto reduceLoop : reduceLoops) {
    reduceLoop->setAttr("reduceLoop", builder.getUnitAttr());
  }

  (void)funcOp->walk([&](Operation *redOp) {
    if (!isa<mlir::func::FuncOp>(redOp) && redOp->hasAttr(kReductionAxesStr)) {
      SmallVector<bool, 8> redFlags;
      SmallVector<bool, 8> sizeOneFlags;
      auto curOp = redOp;
      while (curOp) {
        if (isa<affine::AffineForOp>(curOp)) {
          bool isSizeOne = false;
          if (auto forOp = dyn_cast<affine::AffineForOp>(curOp)) {
            if (forOp.hasConstantBounds()) {
              int64_t lb = forOp.getConstantLowerBound();
              int64_t ub = forOp.getConstantUpperBound();
              if (ub - lb == 1) {
                isSizeOne = true;
              }
            }
          }
          if (curOp->hasAttr("reduceLoop")) {
            redFlags.push_back(true);
          } else {
            redFlags.push_back(false);
          }
          sizeOneFlags.push_back(isSizeOne);
        }
        curOp = curOp->getParentOp();
      }
      std::reverse(redFlags.begin(), redFlags.end());
      std::reverse(sizeOneFlags.begin(), sizeOneFlags.end());

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

      // Determine reduction direction
      ReduceDirection reduceDirection = ReduceDirection::UNKNOWN;

      bool allEffectiveReduce = !redFlags.empty();
      for (size_t i = 0; i < redFlags.size(); ++i) {
        if (!(redFlags[i] || sizeOneFlags[i])) {
          allEffectiveReduce = false;
          break;
        }
      }
      if (allEffectiveReduce) {
        reduceDirection = ReduceDirection::ALL;
      } else if (!redFlags.empty() && redFlags.back()) {
        reduceDirection = ReduceDirection::X;
      } else {
        bool anyReduce = llvm::any_of(redFlags, [](bool v) { return v; });
        if (anyReduce) {
          reduceDirection = ReduceDirection::Y;
        }
      }

      auto strAttr = builder.getStringAttr(reduceDirectionMap.at(reduceDirection));
      redOp->setAttr(kReductionTypeStr, strAttr);
      GpuScheduleTool::getInstance().setReduceDirection((size_t)reduceDirection);
    }
  });
}


std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::affine::createAffineReductionAnnotationPass() {
  return std::make_unique<AffineReductionAnnotation>();
}
