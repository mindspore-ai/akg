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
#include "akg/Transforms/AKGParallelLaunch.h"
#include "akg/Dialect/CPU/IR/CPUOps.h"
#include "akg/Pipelines/CPUOpt.h"
#include "akg/Transforms/AKGFuncOutlining.h"
#include "akg/Transforms/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#ifndef GEN_PASS_DECL_AKGPARALLELLAUNCH
#define GEN_PASS_DECL_AKGPARALLELLAUNCH
#ifndef GEN_PASS_DEF_AKGPARALLELLAUNCH
#define GEN_PASS_DEF_AKGPARALLELLAUNCH
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "akg-parallel-launch"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::scf;
using namespace mlir::CPU;

namespace {
class AKGParallelLaunch : public impl::AKGParallelLaunchBase<AKGParallelLaunch> {
 public:
  AKGParallelLaunch() {}
  AKGParallelLaunch(bool isMindSpore, bool isOutlining) : isMindSpore(isMindSpore), isOutlining(isOutlining) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<CPU::CPUDialect>();
  }

  void runOnOperation() override {
    if (!isOutlining) {
      return;
    }
    ModuleOp moduleOp = getOperation();
    LLVM::LLVMFuncOp mainFunc;
    SmallVector<LLVM::LLVMFuncOp> calculateFuncs;
    context = &getContext();
    // try to find the calculatefunc and mainFunc;
    identifyFuncs(moduleOp, mainFunc, calculateFuncs);
    if (calculateFuncs.empty()) {
      return;
    }

    addFuncConversion(mainFunc, calculateFuncs);
  }

  bool isMindSpore = false;
  bool isOutlining = false;
  MLIRContext *context = nullptr;
  void identifyFuncs(ModuleOp &moduleOp, LLVM::LLVMFuncOp &mainFunc, SmallVector<LLVM::LLVMFuncOp> &calculateFuncs);
  void addFuncConversion(LLVM::LLVMFuncOp &mainFunc, SmallVector<LLVM::LLVMFuncOp> &calculateFuncs) const;
};
}  // namespace

void AKGParallelLaunch::addFuncConversion(LLVM::LLVMFuncOp &mainFunc,
                                          SmallVector<LLVM::LLVMFuncOp> &calculateFuncs) const {
  SmallVector<CPU::ParallelLaunchOp> CPUParallelLaunchOps;
  mainFunc.walk([&](CPU::ParallelLaunchOp op) { CPUParallelLaunchOps.push_back(op); });

  if (CPUParallelLaunchOps.empty()) {
    return;
  }
  assert(CPUParallelLaunchOps.size() == calculateFuncs.size());

  SmallVector<std::pair<CPU::ParallelLaunchOp, LLVM::LLVMFuncOp>> toBeHandled;
  for (LLVM::LLVMFuncOp lambdaFunc : calculateFuncs) {
    for (CPU::ParallelLaunchOp launchOp : CPUParallelLaunchOps) {
      auto funcName = lambdaFunc.getSymNameAttr();
      auto CPULaunchFuncName = launchOp.getCalleeAttr().getAttr();
      if (funcName == CPULaunchFuncName) {
        toBeHandled.push_back(std::make_pair(launchOp, lambdaFunc));
      }
    }
  }
  return;
}

void AKGParallelLaunch::identifyFuncs(ModuleOp &moduleOp, LLVM::LLVMFuncOp &mainFunc,
                                      SmallVector<LLVM::LLVMFuncOp> &calculateFuncs) {
  for (LLVM::LLVMFuncOp funcOp : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    auto attrs = funcOp.getOperation()->getAttrs();
    for (auto attr : attrs) {
      if (attr.getName().str() == kFuncType) {
        auto val = attr.getValue();
        StringAttr valStr = cast<StringAttr>(val);
        if (valStr.str() == kCpuCalcFunc) {
          calculateFuncs.push_back(funcOp);
          break;
        } else if (valStr.str() == kCpuMainFunc) {
          mainFunc = funcOp;
          break;
        }
      }
    }
  }
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> mlir::createAKGParallelLaunchPass() {
  return std::make_unique<AKGParallelLaunch>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> mlir::createAKGParallelLaunchPass(bool isMindSpore, bool isOutlining) {
  return std::make_unique<AKGParallelLaunch>(isMindSpore, isOutlining);
}
