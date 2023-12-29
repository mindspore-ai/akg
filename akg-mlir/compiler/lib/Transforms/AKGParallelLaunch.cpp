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

  void MindSporeParallelLaunch(SmallVector<std::pair<CPU::ParallelLaunchOp, LLVM::LLVMFuncOp>> &toBeHandled,
                               const LLVM::LLVMFuncOp &mainFunc, const ArrayRef<BlockArgument> &args) const;
};
}  // namespace

void AKGParallelLaunch::MindSporeParallelLaunch(
  SmallVector<std::pair<CPU::ParallelLaunchOp, LLVM::LLVMFuncOp>> &toBeHandled, const LLVM::LLVMFuncOp &mainFunc,
  const ArrayRef<BlockArgument> &args) const {
  assert(args.size() >= (1 + int(isMindSpore)));

  // first convert to int32ptr;
  Type typeInt32 = IntegerType::get(context, 32);
  mlir::Type i32Ptr = LLVM::LLVMPointerType::get(typeInt32);
  for (auto &[launchOp, lambdaFunc] : toBeHandled) {
    LLVM::LLVMFunctionType ftypeMLIRParallelLambda = lambdaFunc.getFunctionType();
    LLVM::LLVMPointerType ftypeMLIRParallelLambdaPtr = LLVM::LLVMPointerType::get(ftypeMLIRParallelLambda);
    SmallVector<Type, 4> parallelLaunchFuncOperandTypes;
    parallelLaunchFuncOperandTypes.push_back(ftypeMLIRParallelLambdaPtr);
    auto launchArgs = launchOp.getArgs();
    // we should drop the first args in federated mode
    for (size_t i = 0; i < launchArgs.size(); i++) {
      parallelLaunchFuncOperandTypes.push_back(launchArgs[i].getType());
    }
    parallelLaunchFuncOperandTypes.push_back(typeInt32);
    LLVM::LLVMFunctionType ftypeMLIRParallelLaunch = LLVM::LLVMFunctionType::get(
      ftypeMLIRParallelLambda.getContext(), typeInt32, parallelLaunchFuncOperandTypes, false);
    OpBuilder builder(launchOp);
    auto loc = launchOp.getLoc();
    Value parallelLaunchFuncPtr;
    LLVM::LLVMFuncOp parallelLaunchFunc;
    auto lambdaFuncAddr = builder.create<LLVM::AddressOfOp>(loc, lambdaFunc);
    auto attrs = lambdaFunc.getOperation()->getAttrs();
    if (!isMindSpore) {
      auto module = mainFunc->getParentOfType<ModuleOp>();
      parallelLaunchFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(kParallelLaunchFunc);
      if (!parallelLaunchFunc) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        parallelLaunchFunc =
          builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), kParallelLaunchFunc, ftypeMLIRParallelLaunch);
      }
    } else {
      LLVM::LLVMPointerType ftypeMLIRParallelLaunchPtr = LLVM::LLVMPointerType::get(ftypeMLIRParallelLaunch);
      LLVM::LLVMPointerType ftypeMLIRParallelLaunchPtrPtr = LLVM::LLVMPointerType::get(ftypeMLIRParallelLaunchPtr);
      auto firstArgs = args.front();
      auto parallelLaunchFuncPtrPtr = builder.create<LLVM::BitcastOp>(loc, ftypeMLIRParallelLaunchPtrPtr, firstArgs);
      parallelLaunchFuncPtr = builder.create<LLVM::LoadOp>(loc, ftypeMLIRParallelLaunchPtr, parallelLaunchFuncPtrPtr);
    }

    // create func args
    SmallVector<Value, 4> callFuncArgs;
    // 1.add indirect call func as first operands, and the call arguments as the remaining operands
    if (isMindSpore) {
      callFuncArgs.push_back(parallelLaunchFuncPtr);
    }
    callFuncArgs.push_back(lambdaFuncAddr);
    for (size_t i = 0; i < launchArgs.size(); i++) {
      callFuncArgs.push_back(launchArgs[i]);
    }
    if (!isMindSpore) {
      int64_t upperBoundVal = 1;
      for (auto attr : attrs) {
        if (attr.getName().str() == kUpperBound) {
          upperBoundVal = attr.getValue().cast<IntegerAttr>().getInt();
        }
      }
      auto upperBoundConstant = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), (int32_t)upperBoundVal);
      callFuncArgs.push_back(upperBoundConstant);
      builder.create<LLVM::CallOp>(loc, parallelLaunchFunc, callFuncArgs);
    } else {
      auto secondArgs = args[1];
      auto int32PtrVal = builder.create<LLVM::BitcastOp>(loc, i32Ptr, secondArgs);
      auto taskNums = builder.create<LLVM::LoadOp>(loc, typeInt32, int32PtrVal);
      callFuncArgs.push_back(taskNums);
      builder.create<LLVM::CallOp>(loc, TypeRange(typeInt32), callFuncArgs);
    }

    launchOp->erase();
  }
}

void AKGParallelLaunch::addFuncConversion(LLVM::LLVMFuncOp &mainFunc,
                                          SmallVector<LLVM::LLVMFuncOp> &calculateFuncs) const {
  //  try to bitcast the first arg
  ArrayRef<BlockArgument> args = mainFunc.getArguments();
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

  MindSporeParallelLaunch(toBeHandled, mainFunc, args);

  return;
}

void AKGParallelLaunch::identifyFuncs(ModuleOp &moduleOp, LLVM::LLVMFuncOp &mainFunc,
                                      SmallVector<LLVM::LLVMFuncOp> &calculateFuncs) {
  for (LLVM::LLVMFuncOp funcOp : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    auto attrs = funcOp.getOperation()->getAttrs();
    for (auto attr : attrs) {
      if (attr.getName().str() == kFuncType) {
        auto val = attr.getValue();
        StringAttr valStr = val.cast<StringAttr>();
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
