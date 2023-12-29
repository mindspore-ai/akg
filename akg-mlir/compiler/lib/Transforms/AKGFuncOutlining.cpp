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
#include "akg/Transforms/AKGFuncOutlining.h"

#include <algorithm>
#include <iterator>
#include "akg/Dialect/CPU/IR/CPUOps.h"
#include "akg/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
#ifndef GEN_PASS_DECL_AKGFUNCOUTLINING
#define GEN_PASS_DECL_AKGFUNCOUTLINING
#ifndef GEN_PASS_DEF_AKGFUNCOUTLINING
#define GEN_PASS_DEF_AKGFUNCOUTLINING
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "akg-func-outlining"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::scf;
using namespace mlir::LLVM;
using namespace mlir::CPU;

namespace mlir {

std::string getLambdaName(const StringRef funcName) {
  static uint32_t cnt = 0;
  auto str = (funcName + llvm::Twine("_lambda")).str();
  if (cnt == 0) {
    cnt++;
    return str;
  }
  return str + std::to_string(cnt++);
}

static bool isSinkOps(Operation *op) {
  if (isa<memref::AllocOp>(op)) {
    return false;
  }

  return true;
}

static bool getSunkOps(Operation *op, const SetVector<Value> sinkCandidatesOps, SetVector<Operation *> &toBeSunk,
                       llvm::SmallPtrSetImpl<Value> &availableValues) {
  if (toBeSunk.count(op) != 0) {
    return true;
  }
  if (isSinkOps(op) == 0) {
    return false;
  }

  for (Value opnd : op->getOperands()) {
    if (availableValues.count(opnd) != 0) {
      continue;
    }
    auto defOp = opnd.getDefiningOp();
    if (!defOp) {
      continue;
    }
    if ((!getSunkOps(defOp, sinkCandidatesOps, toBeSunk, availableValues)) && sinkCandidatesOps.count(opnd) == 0) {
      return false;
    }
  }
  (void)toBeSunk.insert(op);
  for (auto val : op->getResults()) {
    (void)availableValues.insert(val);
  }
  return true;
}

void tryToSinkOps(Region &parallelRegion) {
  SetVector<Value> sinkCandidatesOps;
  getUsedValuesDefinedAbove(parallelRegion, sinkCandidatesOps);
  SetVector<Operation *> toBeSunk;
  llvm::SmallPtrSet<Value, 4> availableValues;
  for (auto opnd : sinkCandidatesOps) {
    Operation *op = opnd.getDefiningOp();
    if (!op) {
      continue;
    }
    (void)getSunkOps(op, sinkCandidatesOps, toBeSunk, availableValues);
  }

  IRMapping maps;
  OpBuilder builder(parallelRegion);
  for (Operation *op : toBeSunk) {
    Operation *cloned = builder.clone(*op, maps);
    for (auto pair : llvm::zip(op->getResults(), cloned->getResults())) {
      replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair), parallelRegion);
    }
  }

  return;
}

void addCallBackPtrToFunc(SmallVector<func::FuncOp> &funcOps) {
  for (auto &func : funcOps) {
    auto context = func->getContext();
    OpBuilder builder(func);
    auto loc = func.getLoc();
    SmallVector<Type, 4> newFuncOperandTypes;
    LLVM::LLVMPointerType llvmPointerType = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    const int32_t firstArgPos = 0;
    func.insertArgument(firstArgPos, llvmPointerType, {}, loc);  // callback func
    func.insertArgument(firstArgPos, llvmPointerType, {}, loc);  // task nums

    for (auto type : func.getArgumentTypes()) {
      (void)newFuncOperandTypes.emplace_back(type);
    }
    auto newFuncTypes = FunctionType::get(context, newFuncOperandTypes, {});
    func.setType(newFuncTypes);
    auto origFuncAttributes = func.getOperation()->getAttrs();
    SmallVector<NamedAttribute> newFuncAttrs;
    for (auto attr : origFuncAttributes) {
      (void)newFuncAttrs.emplace_back(attr);
    }
    (void)newFuncAttrs.emplace_back(
      NamedAttribute(StringAttr::get(context, kFuncType), StringAttr::get(context, kCpuMainFunc)));
    func->setAttrs(newFuncAttrs);
  }
  return;
}

static func::FuncOp parallelRegionOutLiningImpl(const func::FuncOp &mainFunc, Operation *outLiningOp,
                                                const StringRef lambdaName, SetVector<Value> &operands,
                                                const Operation *origParallelUpperBoundDefOp) {
  auto loc = outLiningOp->getLoc();
  OpBuilder builder(outLiningOp->getContext());
  Region &outliningOpBody = outLiningOp->getRegion(0);
  // Identify uses from values defined outside of the scope of parallel region
  getUsedValuesDefinedAbove(outliningOpBody, operands);
  // create the CPU.parallelLaunch func operandsType
  SmallVector<Type, 4> newLambdaFuncTypes;
  // the first two types is (int32, int32)
  Type llvmInt32Type = IntegerType::get(outLiningOp->getContext(), 32);
  (void)newLambdaFuncTypes.emplace_back(llvmInt32Type);
  (void)newLambdaFuncTypes.emplace_back(llvmInt32Type);
  (void)std::transform(operands.begin(), operands.end(), std::back_inserter(newLambdaFuncTypes),
                       [](const Value val) { return val.getType(); });

  FunctionType lambdaFuncType = FunctionType::get(outLiningOp->getContext(), newLambdaFuncTypes, {});
  auto lambdaFunc = builder.create<func::FuncOp>(loc, lambdaName, lambdaFuncType);
  SmallVector<NamedAttribute> lambdaFuncAttrs;
  auto mainFuncAttributes = mainFunc->getAttrs();
  for (auto attr : mainFuncAttributes) {
    if (attr.getName().str() == "function_type" || attr.getName().str() == "sym_name" ||
        attr.getName().str() == kFuncType) {
      continue;
    }
    (void)lambdaFuncAttrs.emplace_back(attr);
  }

  for (auto attr : lambdaFunc->getAttrs()) {
    if (attr.getName().str() == "function_type" || attr.getName().str() == "sym_name") {
      (void)lambdaFuncAttrs.emplace_back(attr);
    }
  }
  (void)lambdaFuncAttrs.emplace_back(NamedAttribute(StringAttr::get(outLiningOp->getContext(), kFuncType),
                                                    StringAttr::get(outLiningOp->getContext(), kCpuCalcFunc)));
  if (origParallelUpperBoundDefOp) {
    if (auto constantOp = dyn_cast<arith::ConstantOp>(origParallelUpperBoundDefOp)) {
      auto val = constantOp.getValue().cast<IntegerAttr>().getInt();
      auto attr = NamedAttribute(StringAttr::get(lambdaFunc->getContext(), kUpperBound),
                                 IntegerAttr::get(builder.getI64Type(), val));
      (void)lambdaFuncAttrs.emplace_back(attr);
    }
  }

  IRMapping maps;
  Block &entryBlock = *lambdaFunc.addEntryBlock();
  Region &lambdaFuncBody = lambdaFunc.getBody();

  const uint32_t firstTwoReservedArgs = 2;
  for (auto opnd : enumerate(operands)) {
    maps.map(opnd.value(), entryBlock.getArgument((uint32_t)opnd.index() + firstTwoReservedArgs));
  }
  outliningOpBody.cloneInto(&lambdaFuncBody, maps);
  Block &outliningOpEntry = outliningOpBody.front();
  Block *clonedOutliningOpEntry = maps.lookup(&outliningOpEntry);
  builder.setInsertionPointToEnd(&entryBlock);
  (void)builder.create<cf::BranchOp>(loc, clonedOutliningOpEntry);

  SetVector<Operation *> toBeRemoved;
  lambdaFunc.walk([&](const scf::YieldOp op) {
    if (op->getParentOfType<scf::IfOp>() || op->getParentOfType<scf::ParallelOp>() ||
        op->getParentOfType<scf::ForOp>()) {
      return;
    }
    (void)toBeRemoved.insert(op);
  });

  for (Operation *op : toBeRemoved) {
    OpBuilder newBuilder(op);
    assert(op->getNumResults() == 0);
    (void)newBuilder.create<func::ReturnOp>(op->getLoc());
    op->erase();
  }

  lambdaFunc->setAttrs(lambdaFuncAttrs);
  return lambdaFunc;
}

func::FuncOp tryToRewriteCPUParallelLaunchOp(func::FuncOp &mainFunc, scf::ParallelOp parallelOp,
                                             SmallVector<Value> &operands, const std::string &lambdaName,
                                             Operation *&outLiningOp) {
  auto loc = parallelOp.getLoc();
  OpBuilder builder(parallelOp);
  // try to add scf.if for helper
  Value trueCond = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, trueCond, false);
  Block *ifThenBlock = &ifOp.getThenRegion().getBlocks().front();
  // move parallel region into ifThenBlock
  parallelOp->moveBefore(ifThenBlock, ifThenBlock->begin());
  outLiningOp = ifOp;

  tryToSinkOps(outLiningOp->getRegion(0));

  DenseSet<Value> parallelRegionOperandSet;
  parallelRegionOperandSet.insert(operands.begin(), operands.end());
  SetVector<Value> operandSet(operands.begin(), operands.end());

  auto upperVal = parallelOp.getUpperBound()[0];
  auto origParallelUpperBoundDefOp = upperVal.getDefiningOp();
  assert(origParallelUpperBoundDefOp != nullptr);

  auto lambdaFunc =
    parallelRegionOutLiningImpl(mainFunc, outLiningOp, lambdaName, operandSet, origParallelUpperBoundDefOp);
  for (auto opnd : operandSet) {
    if (parallelRegionOperandSet.count(opnd) == 0) {
      (void)operands.emplace_back(opnd);
    }
  }

  return lambdaFunc;
}

static void addCPUParallelLaunchFuncOp(Operation *outliningOp, const func::FuncOp &lambdaFunc,
                                       const ValueRange operands) {
  auto loc = outliningOp->getLoc();
  OpBuilder builder(outliningOp);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  SmallVector<Value, 4> newOperands;
  (void)newOperands.emplace_back(zero);
  (void)newOperands.emplace_back(zero);
  (void)std::copy(operands.begin(), operands.end(), std::back_inserter(newOperands));
  (void)builder.create<func::CallOp>(loc, SymbolRefAttr::get(lambdaFunc), TypeRange{}, newOperands);
  outliningOp->erase();
  return;
}

}  // namespace mlir

namespace {
class AKGFuncOutlining : public impl::AKGFuncOutliningBase<AKGFuncOutlining> {
 public:
  AKGFuncOutlining() {}
  AKGFuncOutlining(bool mindSpore, bool outlining) : isOutlining(outlining), isMindSpore(mindSpore) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<CPU::CPUDialect>();
  }

  void runOnOperation() override {
    // 1.if we run on MLIR, and don't use outlining, we don't need do anything
    if (!isMindSpore && !isOutlining) {
      return;
    }
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> toBeHandleFuncOps;
    context = &getContext();
    getProcessFuncs(module, toBeHandleFuncOps);
    if (isMindSpore && !isOutlining) {
      addCallBackPtrToFunc(toBeHandleFuncOps);
      return;
    }

    SymbolTable symTable(getOperation());
    // try to extract the functions;
    func::FuncOp mainFunc;
    tryToSplitFuncs(toBeHandleFuncOps, mainFunc, &symTable);
    return;
  }

  void getProcessFuncs(ModuleOp &module, SmallVector<func::FuncOp> &funcOps);

  void tryToSplitFuncs(SmallVector<func::FuncOp> &funcOps, func::FuncOp &mainFunc, SymbolTable *symbolTablePtr);
  bool hasParallel = false;
  bool isNestedParallel = false;
  MLIRContext *context = nullptr;
  Operation *origParallelUpperBoundDefOp = nullptr;

  bool isOutlining = false;
  bool isMindSpore = false;

  void handleCalculateFunc(func::FuncOp &func);
  void tryToSplitParallelFunc(func::FuncOp &func, func::FuncOp &mainFunc, SymbolTable *symbolTablePtr);
};
}  // namespace

void AKGFuncOutlining::handleCalculateFunc(func::FuncOp &func) {
  // only if there has parallel,we just add static schdule
  if (hasParallel) {
    SmallVector<scf::ParallelOp> parallelOpVec;
    func->walk([&](const scf::ParallelOp op) {
      if (op->getParentOfType<scf::ParallelOp>()) {
        return;
      }
      (void)parallelOpVec.emplace_back(op);
    });
    // one func we think only one parallel region;
    assert(parallelOpVec.size() == 1);
    scf::ParallelOp parallelOp = parallelOpVec.front();
    auto loc = parallelOp.getLoc();
    OpBuilder builder(parallelOp);
    scf::ParallelOp cloned = cast<scf::ParallelOp>(builder.clone(*parallelOp.getOperation()));

    Value numIdx = func.getArgument(0);
    Value numCores = func.getArgument(1);
    Value upperIdx = parallelOp.getUpperBound()[0];
    Value stepIdx = parallelOp.getStep()[0];

    // record original for upper bound Op
    origParallelUpperBoundDefOp = upperIdx.getDefiningOp();
    auto upperVal = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), upperIdx);
    auto stepVal = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), stepIdx);
    Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

    // ( numCore + (upperBound - 1)) % numCore
    auto upperCores = builder.create<arith::SubIOp>(loc, numCores, one);
    auto numCoreAddBound = builder.create<arith::AddIOp>(loc, upperVal, upperCores);
    auto iterRange = builder.create<arith::DivSIOp>(loc, numCoreAddBound, numCores);
    // real_coreIdx = coreIdx + 1
    auto realCoreIdx = builder.create<arith::AddIOp>(loc, numIdx, one);
    // get the current core's upper bound
    auto curIterUpperBound = builder.create<arith::MulIOp>(loc, iterRange, realCoreIdx);
    auto coreLowerBound = builder.create<arith::MulIOp>(loc, iterRange, numIdx);
    // get aligned upperBound and lowerBound rem
    //  lowerBoundRemVal = (coreLowerBound) % step
    auto lowerBoundRemVal = builder.create<arith::RemSIOp>(loc, coreLowerBound, stepVal);
    auto upperBoundRemVal = builder.create<arith::RemSIOp>(loc, curIterUpperBound, stepVal);

    // calculate real aligned lowerBound/upperBound iteration value
    auto realLowerBoundIterVal = builder.create<arith::SubIOp>(loc, coreLowerBound, lowerBoundRemVal);
    auto realUpperBoundIterVal = builder.create<arith::SubIOp>(loc, curIterUpperBound, upperBoundRemVal);
    // compre(realUpperBoundIterVal, forOpUpperBound)
    auto cmpRes = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, realUpperBoundIterVal, upperVal);
    auto curCoreUpperBound = builder.create<arith::SelectOp>(loc, cmpRes, realUpperBoundIterVal, upperVal);
    auto lowerBoundCmpRes =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, realLowerBoundIterVal, upperVal);
    auto coreLowerBoundIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), realLowerBoundIterVal);
    auto curCoreUpperBoundIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), curCoreUpperBound);

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, lowerBoundCmpRes, false);
    SmallVector<Value> lowerBoundVec;
    SmallVector<Value> upperBoundVec;
    SmallVector<Value> stepVec;
    (void)lowerBoundVec.emplace_back(coreLowerBoundIndex);
    (void)upperBoundVec.emplace_back(curCoreUpperBoundIndex);
    (void)stepVec.emplace_back(stepIdx);
    // set the new cloned parallelOp
    cloned->setOperands(0, lowerBoundVec.size(), lowerBoundVec);
    cloned->setOperands(lowerBoundVec.size(), upperBoundVec.size(), upperBoundVec);
    cloned->setOperands(lowerBoundVec.size() + upperBoundVec.size(), stepVec.size(), stepVec);
    // now, we sink the Parallel down to scf::forOp
    Block *thenBlock = &ifOp.getThenRegion().getBlocks().front();
    builder.setInsertionPointToStart(thenBlock);
    cloned->moveBefore(thenBlock, thenBlock->begin());
    // erase the origin parallelOp
    parallelOp->erase();
  }
  return;
}

void AKGFuncOutlining::tryToSplitParallelFunc(func::FuncOp &funcOps, func::FuncOp &mainFunc,
                                              SymbolTable *symbolTablePtr) {
  // if the original func doesn't have any parallel region, we don't need to outlining the parallel region.
  if (!hasParallel) {
    return;
  }
  // otherwise, we find the Parellel op
  SmallVector<scf::ParallelOp> parallelOps;
  mainFunc.walk([&](scf::ParallelOp op) {
    if (op->getParentOfType<scf::ParallelOp>()) {
      return;
    }
    (void)parallelOps.emplace_back(op);
  });

  // MainFunc real args
  auto args = mainFunc.getArguments();
  SmallVector<Value> mainFuncRealArgs;
  size_t startIdx = isMindSpore ? kMindsporeArgOffset : 0;
  for (size_t i = startIdx; i < args.size(); i++) {
    (void)mainFuncRealArgs.emplace_back(args[i]);
  }
  Block::iterator insertPtr(mainFunc->getNextNode());
  for (const auto &op : llvm::enumerate(parallelOps)) {
    SmallVector<Value> operands = mainFuncRealArgs;
    auto lambdaName = getLambdaName(funcOps.getName());
    Operation *outLiningOp = nullptr;
    auto outLiningParallelFunc =
      tryToRewriteCPUParallelLaunchOp(mainFunc, op.value(), operands, lambdaName, outLiningOp);
    assert(outLiningOp != nullptr);
    // we expect the lambda func only use mainFuncRealArgs as its real args;
    symbolTablePtr->insert(outLiningParallelFunc, insertPtr);
    addCPUParallelLaunchFuncOp(outLiningOp, outLiningParallelFunc, operands);
    handleCalculateFunc(outLiningParallelFunc);
  }

  return;
}

void AKGFuncOutlining::tryToSplitFuncs(SmallVector<func::FuncOp> &funcOps, func::FuncOp &mainFunc,
                                       SymbolTable *symbolTablePtr) {
  assert(funcOps.size() == 1);
  for (auto &func : funcOps) {
    OpBuilder builder(func);
    auto funcName = func.getName();
    auto loc = func.getLoc();
    // try to build the new kernel_name and lambda func
    auto origFuncAttributes = func.getOperation()->getAttrs();
    SmallVector<NamedAttribute> newMainFuncAttrs;
    SmallVector<NamedAttribute> newCalculateFuncAttrs;

    for (auto attr : origFuncAttributes) {
      if (attr.getName().str() == "function_type" || attr.getName().str() == "sym_name") {
        continue;
      }
      (void)newMainFuncAttrs.emplace_back(attr);
      (void)newCalculateFuncAttrs.emplace_back(attr);
    }

    //  build the origin func (the same name but params are different)
    //  (callback func, task_nums, input params)
    SmallVector<Type, 4> mainFuncOperandTypes;
    LLVM::LLVMPointerType llvmPointerType = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    if (isMindSpore) {
      // insert the first arg: callback func
      (void)mainFuncOperandTypes.emplace_back(llvmPointerType);
      // insert the second arg: task_nums
      (void)mainFuncOperandTypes.emplace_back(llvmPointerType);
    }
    std::copy(func.getArgumentTypes().begin(), func.getArgumentTypes().end(), std::back_inserter(mainFuncOperandTypes));
    auto mainFuncTypes = FunctionType::get(context, mainFuncOperandTypes, {});

    mainFunc = builder.create<func::FuncOp>(loc, funcName, mainFuncTypes);
    mainFunc.setPublic();
    for (auto attr : mainFunc->getAttrs()) {
      if (attr.getName().str() == "function_type" || attr.getName().str() == "sym_name") {
        (void)newMainFuncAttrs.emplace_back(attr);
      }
    }
    (void)newMainFuncAttrs.emplace_back(
      NamedAttribute(StringAttr::get(context, kFuncType), StringAttr::get(context, kCpuMainFunc)));
    IRMapping maps;
    auto oldFuncOperands = func.getArguments();

    OpBuilder::InsertionGuard insertGuard(builder);
    Block *FuncBody = mainFunc.addEntryBlock();
    mainFunc->setAttrs(newMainFuncAttrs);
    builder.setInsertionPointToEnd(FuncBody);
    // the first 2 args are callback pointer and task_nums
    size_t argOffset = isMindSpore ? kMindsporeArgOffset : 0;
    if (isMindSpore) {
      assert(mainFuncOperandTypes.size() == oldFuncOperands.size() + argOffset);
    }
    for (auto opnd : llvm::enumerate(oldFuncOperands)) {
      maps.map(opnd.value(), FuncBody->getArgument(opnd.index() + argOffset));
    }

    for (Block &block : func.getBody()) {
      for (Operation &op : block.getOperations()) {
        builder.clone(op, maps);
      }
    }

    // try to find scf.parallel and try to outline them as a lambda func
    tryToSplitParallelFunc(func, mainFunc, symbolTablePtr);
    func.erase();
  }
  return;
}

void AKGFuncOutlining::getProcessFuncs(ModuleOp &module, SmallVector<func::FuncOp> &funcOps) {
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    // try to find the func, which only the one scf.Parallel
    funcOp->walk([&](scf::ParallelOp parallelOp) {
      hasParallel = true;
      if (parallelOp->getParentOfType<scf::ParallelOp>()) {
        isNestedParallel = true;
      }
    });

    (void)funcOps.emplace_back(funcOp);
    if (isNestedParallel) {
      emitWarning(funcOp->getLoc()) << DEBUG_TYPE << " -- Unsupported nested parallel in : " << funcOp->getName()
                                    << ".\n";
    }
  }
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> mlir::createAKGFuncOutliningPass() {
  return std::make_unique<AKGFuncOutlining>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> mlir::createAKGFuncOutliningPass(bool isMindSpore, bool isOutlining) {
  return std::make_unique<AKGFuncOutlining>(isMindSpore, isOutlining);
}
