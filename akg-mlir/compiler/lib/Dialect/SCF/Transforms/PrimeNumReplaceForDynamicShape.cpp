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

#include "akg/Dialect/SCF/Transforms/PrimeNumReplaceForDynamicShape.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
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
#define GEN_PASS_DECL_PRIMENUMREPLACEFORDYNAMICSHAPE
#define GEN_PASS_DEF_PRIMENUMREPLACEFORDYNAMICSHAPE
#include "akg/Dialect/SCF/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace akgglobal;

namespace mlir {
namespace scf {
namespace {
constexpr auto kPrimeReplace = "PrimeReplace";

// A struct to collect the keep args informations
struct ConvertInfo {
  mlir::Value realUpperBound;
  Operation *primeOp;
  mindspore::KeepArgsOp keepArgsOp;
};

// Convert Value to Int
static int getIntConst(mlir::Value value) {
  auto constValueAttr = value.getDefiningOp()->getAttr("value");
  if (isa<IntegerAttr>(constValueAttr)) {
    return dyn_cast<IntegerAttr>(constValueAttr).getInt();
  }
  return 0;
}

// Get the gpu iter attrs on gpu::LaunchOp
static Value getReplacedGpuIter(Operation *primeOp, gpu::LaunchOp gpuLaunch, Operation *mainFunc) {
  auto primeConst = dyn_cast<mlir::arith::ConstantOp>(primeOp);
  if (!primeConst) {
    return Value();
  }

  auto replacedPrime = getIntConst(primeConst);
  auto attrs = mainFunc->getAttrs();

  // Mapping to gpu dim info
  static const std::map<std::string, Value> keyToValueMap = {
    {"block_x", gpuLaunch.getBlockIds().x},   {"block_y", gpuLaunch.getBlockIds().y},
    {"block_z", gpuLaunch.getBlockIds().z},   {"thread_x", gpuLaunch.getThreadIds().x},
    {"thread_y", gpuLaunch.getThreadIds().y}, {"thread_z", gpuLaunch.getThreadIds().z},
  };

  for (auto attr : attrs) {
    auto attrNameStr = dyn_cast<StringAttr>(attr.getName());
    auto attrValueInt = dyn_cast_or_null<IntegerAttr>(attr.getValue());
    if (!attrNameStr || !attrValueInt || attrValueInt.getInt() != replacedPrime) {
      continue;
    }
    auto it = keyToValueMap.find(attrNameStr.getValue().str());
    if (it != keyToValueMap.end()) {
      return it->second;
    }
  }

  return Value();
}

// PrimeNumReplace finds the unknown shape op, and create a new keep args op with `PrimeReplace` attr.
// KeepArgsOp cannot remove by `--canonicalize`, so that we can keep the information during transformations.
static mlir::LogicalResult PrimeNumReplace(Operation *funcOp) {
  OpBuilder builder(funcOp);
  auto ctx = funcOp->getContext();
  auto &tool = PrimeNumTool::getInstance();
  funcOp->walk([&](scf::ParallelOp parallelOp) {
    auto upperBound = parallelOp.getUpperBound().front();
    if (!isa<mlir::arith::ConstantOp>(upperBound.getDefiningOp())) {
      auto primeNum = tool.getOnePrimeWithIdxUpdate();
      auto unknownUpperBound = upperBound.getDefiningOp();
      builder.setInsertionPointAfter(unknownUpperBound);

      auto loc = unknownUpperBound->getLoc();
      auto constAttr = builder.getIndexAttr((long)primeNum);
      auto newPrimeConstant = builder.create<mlir::arith::ConstantOp>(loc, constAttr);
      unknownUpperBound->replaceAllUsesWith(newPrimeConstant);
      auto keepArgs = builder.create<mindspore::KeepArgsOp>(loc, upperBound, newPrimeConstant.getResult());
      keepArgs.getOperation()->setAttr(kPrimeReplace, builder.getUnitAttr());

      // update grid/block mapping
      auto processor = stringifyProcessor(gpu::GpuAttrUtils::getProcessorFromParallelOp(parallelOp));
      mlir::IntegerAttr int32Attr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), (long)primeNum);
      funcOp->setAttr(processor, int32Attr);
    }
  });
  return mlir::success();
}

// Update arguments on gpu::LaunchOp
static void UpdateRuntimeVars(func::FuncOp &mainFunc, std::map<int, mlir::Value> &allConstsMap) {
  mlir::MLIRContext *context = mainFunc.getContext();
  mlir::TypeRange originalArgTypes = mainFunc.getFunctionType().getInputs();
  llvm::SmallVector<mlir::Type, 4> newArgTypes(originalArgTypes.begin(), originalArgTypes.end());
  size_t currArgSize = mainFunc.getArguments().size();
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();

  for (auto it : GpuScheduleTool::getInstance().getRuntimeVars()) {
    mlir::Region &bodyRegion = mainFunc.getBody();
    auto doReplace = [&](int64_t primeNum) {
      if (allConstsMap.find(primeNum) == allConstsMap.end()) {
        return;
      }
      mlir::Type newArgType = mlir::IndexType::get(context);
      mlir::Value newArg = bodyRegion.addArgument(newArgType, mainFunc->getLoc());
      allConstsMap[primeNum].replaceAllUsesWith(newArg);
      if (gpuTool.isRuntimeVar(primeNum)) {
        auto oldVar = gpuTool.getRuntimeArgument(primeNum);
        oldVar.argIndex = currArgSize;
        gpuTool.updateRuntimeArgument(oldVar);
      } else {
        auto newVar = gpuTool.addRuntimeArgument(primeNum);
        newVar.argIndex = currArgSize;
        gpuTool.updateRuntimeArgument(newVar);
      }
      currArgSize++;
      newArgTypes.push_back(newArgType);
    };
    auto posPrime = it.first;
    auto negPrime = -it.first;
    doReplace(posPrime);
    doReplace(negPrime);
  }
  mlir::FunctionType newFuncType =
    mlir::FunctionType::get(context, mlir::TypeRange(newArgTypes), mainFunc.getFunctionType().getResults());
  mainFunc.setType(newFuncType);
}

// Restore dynamic values and remove keepargs ops
static void RestoreValueAndRemoveKeepArgs(Operation *funcOp, gpu::LaunchOp &gpuLaunch,
                                          std::vector<ConvertInfo> &convertInfos) {
  OpBuilder builder(gpuLaunch);
  for (auto currInfo : convertInfos) {
    auto primeOp = currInfo.primeOp;
    auto realUpperBound = currInfo.realUpperBound;
    auto gpuIter = getReplacedGpuIter(primeOp, gpuLaunch, funcOp);
    // this prime replaces gpu.block/thread
    if (gpuIter) {
      auto loc = primeOp->getLoc();
      builder.setInsertionPoint(primeOp);
      auto origLoopCond =
        builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, gpuIter, realUpperBound);
      auto ifVarLessThanCond = builder.create<mlir::scf::IfOp>(loc, origLoopCond.getResult());
      builder.setInsertionPointToStart(&ifVarLessThanCond.getThenRegion().front());
      Block *srcBlock = primeOp->getBlock();
      Block *thenBlock = ifVarLessThanCond.thenBlock();
      thenBlock->getOperations().splice(std::prev(thenBlock->end()), srcBlock->getOperations(), primeOp->getIterator(),
                                        std::prev(srcBlock->end()));
    } else {
      // cases that keep the scf.for
      primeOp->replaceAllUsesWith(realUpperBound.getDefiningOp());
    }
    currInfo.keepArgsOp->erase();
    primeOp->erase();
  }
}

// PrimeNumReStore finds the keep args op with `PrimeReplace` attr, and try to restore those prime number
// with those unknown values.
static mlir::LogicalResult PrimeNumReStore(Operation *funcOp) {
  gpu::LaunchOp gpuLaunch;
  func::FuncOp mainFunc;
  std::vector<ConvertInfo> convertInfos;
  std::map<int, mlir::Value> allConstsMap;

  // collect dynamic shape related informations
  funcOp->walk([&](Operation *op) {
    if (auto f = dyn_cast<gpu::LaunchOp>(op)) {
      gpuLaunch = f;
    }
    if (auto f = dyn_cast<func::FuncOp>(op)) {
      mainFunc = f;
    }
    if (op->hasAttr(kPrimeReplace)) {
      if (auto keepArgsOp = dyn_cast<mindspore::KeepArgsOp>(op)) {
        auto operands = keepArgsOp.getOperands();
        ConvertInfo currInfo = ConvertInfo();
        currInfo.realUpperBound = operands[0];
        currInfo.primeOp = operands[1].getDefiningOp();
        currInfo.keepArgsOp = keepArgsOp;
        convertInfos.push_back(currInfo);
      }
    }
    if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
      allConstsMap[getIntConst(constOp)] = constOp;
    }
  });

  if (!gpuLaunch || !mainFunc) {
    llvm::errs() << "No gpu launch op, please invoke after `--convert-parallel-loops-to-gpu`.\n";
    return mlir::failure();
  }

  UpdateRuntimeVars(mainFunc, allConstsMap);

  RestoreValueAndRemoveKeepArgs(funcOp, gpuLaunch, convertInfos);

  return mlir::success();
}

// PrimeNumReplaceForDynamicShape is the pass for prime number replacement. Prime number replacement is
// used for dynamic shape passing by. For example, some of passes may not support unknown shapes since
// they need to analyze boxes or just under develop. This pass use prime numbers to represent them, so that
// we can move on. When left passes can handle dyanmic shape, we restore those prime numbers.
struct PrimeNumReplaceForDynamicShape
    : public impl::PrimeNumReplaceForDynamicShapeBase<PrimeNumReplaceForDynamicShape> {
  PrimeNumReplaceForDynamicShape() = default;
  explicit PrimeNumReplaceForDynamicShape(const std::string action) { this->action = action; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mindspore::MindSporeDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (this->action == "replace") {
      if (mlir::failed(PrimeNumReplace(funcOp))) {
        signalPassFailure();
      }
    } else if (this->action == "restore") {
      if (mlir::failed(PrimeNumReStore(funcOp))) {
        signalPassFailure();
      }
    } else {
      std::string errorMsg = "action should be \"replace\" or \"restore\", while got \"" + this->action + "\"";
      (void)funcOp->emitError(errorMsg);
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace scf
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createPrimeNumReplaceForDynamicShapePass() {
  return std::make_unique<scf::PrimeNumReplaceForDynamicShape>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createPrimeNumReplaceForDynamicShapePass(
  std::string action) {
  return std::make_unique<scf::PrimeNumReplaceForDynamicShape>(action);
}
