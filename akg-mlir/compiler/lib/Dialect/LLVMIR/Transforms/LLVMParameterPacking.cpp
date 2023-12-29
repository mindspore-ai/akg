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

#include "akg/Dialect/LLVMIR/Transforms/LLVMParameterPacking.h"
#include "akg/Dialect/CPU/IR/CPUOps.h"
#include "akg/Transforms/AKGFuncOutlining.h"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_PARAMETERPACKING
#define GEN_PASS_DECL_PARAMETERPACKING
#include "akg/Dialect/LLVMIR/Passes.h.inc"
}  // namespace LLVM
}  // namespace mlir

using namespace mlir;
using namespace mlir::CPU;

namespace {
constexpr auto lambdaFuncRealArgIndex = 2;
constexpr auto kIntType8 = 8;
constexpr auto kIntType32 = 32;
constexpr auto kIntType64 = 64;

using CpuFuncPair = SmallVector<std::pair<LLVM::CallOp, LLVM::LLVMFuncOp>>;

void packParamsStaticShape(LLVM::LLVMFuncOp &funcOp, const int argOffset);

void getHandlePairs(ModuleOp &moduleOp, CpuFuncPair *toBeHandled, LLVM::LLVMFuncOp &mainFunc) {
  SmallVector<LLVM::LLVMFuncOp> calculateFuncs;
  for (auto func : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    auto attrs = func.getOperation()->getAttrs();
    for (auto attr : attrs) {
      if (attr.getName().str() == kFuncType) {
        auto val = attr.getValue();
        StringAttr valStr = val.cast<StringAttr>();
        if (valStr.str() == kCpuCalcFunc) {
          calculateFuncs.push_back(func);
          break;
        }
      }
    }
  }

  SmallVector<LLVM::CallOp> callOps;
  mainFunc.walk([&](LLVM::CallOp op) { callOps.push_back(op); });

  for (LLVM::LLVMFuncOp lambdaFunc : calculateFuncs) {
    for (LLVM::CallOp launchOp : callOps) {
      auto funcName = lambdaFunc.getSymNameAttr();
      auto callFuncName = launchOp.getCalleeAttr().getAttr();
      if (funcName == callFuncName) {
        toBeHandled->push_back(std::make_pair(launchOp, lambdaFunc));
      }
    }
  }
  return;
}

LLVM::LLVMPointerType getIntegerType(MLIRContext *context, const unsigned int integerSize) {
  return LLVM::LLVMPointerType::get(IntegerType::get(context, integerSize));
}

LLVM::LLVMPointerType getPtrArrayType(const LLVM::LLVMPointerType ptrTy, const unsigned int arrayLen) {
  return LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(ptrTy, arrayLen));
}

void onlyAddArgToFuncArgs(LLVM::LLVMFuncOp &funcOp, const LLVM::LLVMPointerType llvmType) {
  auto &block = funcOp.getBody().front();
  auto funcType = funcOp.getFunctionType();
  (void)block.addArgument(llvmType, funcOp.getLoc());
  // Reset function type to match changed function args
  llvm::SmallVector<Type> newInTys;
  for (auto value : block.getArguments()) {
    (void)newInTys.emplace_back(value.getType());
  }
  auto newFuncType = LLVM::LLVMFunctionType::get(funcType.getReturnType(), newInTys, false);
  funcOp.setFunctionTypeAttr(TypeAttr::get(newFuncType));
}

// lambdaFunc(%0 core_idx, %1 core_num, params_mainfuncArg0, params_mainfuncArg1, ..., params_externArg0,
// params_externArg1, ...)
// after pack Args=>
// lambdaFunc(%0 core_idx, %1 core_num, params_mainfuncPtr, params_externArgPtr)
void packParamsStaticShapeForLambda(LLVM::LLVMFuncOp &lambdaFuncOp, const uint32_t argOffset,
                                    const size_t realTensorNum) {
  // pack all inputs into an array of ptr
  auto *context = lambdaFuncOp.getContext();

  auto loc = lambdaFuncOp.getLoc();
  auto i8PtrTy = getIntegerType(context, kIntType8);
  // handle the first part;
  auto ptrToArrayTy = getPtrArrayType(i8PtrTy, static_cast<unsigned int>(realTensorNum));

  auto &block = lambdaFuncOp.getBody().front();
  auto builder = OpBuilder::atBlockBegin(&block);
  auto newArg = block.addArgument(ptrToArrayTy, loc);
  auto LoadPtrArrayOp = builder.create<LLVM::LoadOp>(loc, newArg);
  for (size_t i = 0; i < realTensorNum; i++) {
    auto realTensorLoc = i + argOffset;
    auto extractValOp = builder.create<LLVM::ExtractValueOp>(loc, LoadPtrArrayOp, i);
    auto realTensorArg = block.getArgument(static_cast<unsigned int>(realTensorLoc));
    auto dstType = realTensorArg.getType();
    if (dstType != i8PtrTy) {
      auto bitcastOp = builder.create<LLVM::BitcastOp>(loc, dstType, extractValOp);
      realTensorArg.replaceAllUsesWith(bitcastOp);
    } else {
      realTensorArg.replaceAllUsesWith(extractValOp);
    }
  }

  SmallVector<Type, 4> packedTypes;
  SmallVector<Value, 4> packArgs;
  auto args = lambdaFuncOp.getArguments();
  auto startIdx = realTensorNum + argOffset;
  for (size_t i = startIdx; i < args.size(); i++) {
    packedTypes.push_back(args[i].getType());
    packArgs.push_back(args[i]);
  }

  auto ptrToArrayTy1 = getPtrArrayType(i8PtrTy, static_cast<unsigned int>(args.size() - startIdx));
  auto newArg1 = block.addArgument(ptrToArrayTy1, lambdaFuncOp.getLoc());
  auto loadPtrArrayOp1 = builder.create<LLVM::LoadOp>(loc, newArg1);

  for (auto &en : llvm::enumerate(packArgs)) {
    auto extractValOp = builder.create<LLVM::ExtractValueOp>(loc, loadPtrArrayOp1, en.index());
    auto dstType = packedTypes[en.index()];
    unsigned int argIdx = (unsigned int)startIdx + (unsigned int)en.index();
    if (dstType != i8PtrTy) {
      auto bitcastVal = builder.create<LLVM::BitcastOp>(loc, dstType, extractValOp);
      block.getArgument(argIdx).replaceAllUsesWith(bitcastVal);
    } else {
      block.getArgument(argIdx).replaceAllUsesWith(extractValOp);
    }
  }

  // erase unused params;
  for (size_t i = 0; i < packArgs.size() + realTensorNum; i++) {
    block.eraseArgument(argOffset);
  }

  // reset function type;
  SmallVector<Type> newTypes;
  for (auto value : block.getArguments()) {
    newTypes.push_back(value.getType());
  }
  auto newFuncType = LLVM::LLVMFunctionType::get(lambdaFuncOp.getFunctionType().getReturnType(), newTypes, false);
  lambdaFuncOp.setFunctionTypeAttr(TypeAttr::get(newFuncType));
}

void buildNoExternArgsForOutlining(OpBuilder &builder, LLVM::CallOp &launchOp, LLVM::LLVMFuncOp &lambdaFunc,
                                   LLVM::LLVMFuncOp &mainFunc, const bool isMindSpore) {
  auto *context = launchOp.getContext();
  auto loc = launchOp.getLoc();
  auto i8PtrTy = getIntegerType(context, kIntType8);
  packParamsStaticShape(lambdaFunc, lambdaFuncRealArgIndex);
  // try to add  a void* arg to lambdaFunc and update signature;
  onlyAddArgToFuncArgs(lambdaFunc, i8PtrTy);
  // try to update launchOp;
  SmallVector<Value> newOperands;
  size_t startIdx = isMindSpore ? kMindsporeArgOffset : 0;
  auto args = mainFunc.getArguments();
  for (size_t i = startIdx; i < args.size(); i++) {
    newOperands.push_back(args[i]);
  }
  auto nullPtr = builder.create<LLVM::NullOp>(loc, i8PtrTy);
  newOperands.push_back(nullPtr);
  // build a new CPULaunch op;
  (void)builder.create<CPU::ParallelLaunchOp>(loc, TypeRange{}, newOperands, SymbolRefAttr::get(lambdaFunc));
  launchOp->erase();  // erase the launchOp;
}

void buildExternArgsForOutlining(OpBuilder &builder, LLVM::CallOp &launchOp, LLVM::LLVMFuncOp &lambdaFunc,
                                 LLVM::LLVMFuncOp &mainFunc, const bool isMindSpore,
                                 const size_t originalMainFuncArgSize) {
  auto launchOpArgs = launchOp.getArgOperands();
  auto launchOpArgSize = launchOpArgs.size();
  assert(launchOpArgSize > (lambdaFuncRealArgIndex + originalMainFuncArgSize));

  auto *context = launchOp.getContext();
  auto loc = launchOp.getLoc();

  // try to pack the params;
  SmallVector<Type, 4> externOpArgsType;
  SmallVector<Value, 4> externOpArgs;
  // the first two args are the core_idx, core_num, we should skip;
  for (size_t i = originalMainFuncArgSize + lambdaFuncRealArgIndex; i < launchOpArgSize; i++) {
    externOpArgsType.push_back(launchOpArgs[i].getType());
    externOpArgs.push_back(launchOpArgs[i]);
  }

  // create new alloc array for new params;
  // 1) %array = alloca((NumParams) * sizeof(void*))
  // 2) for (i : [0, NumParams)]):
  //       llvm.insertvalue array[i], params[i]
  auto i8PtrTy = getIntegerType(context, kIntType8);
  unsigned int tempArgNum = launchOpArgSize - lambdaFuncRealArgIndex - originalMainFuncArgSize;
  auto arrayType0 = getPtrArrayType(i8PtrTy, tempArgNum);
  Type llvmInt32Type = IntegerType::get(context, kIntType32);
  Value constOneOp = builder.create<LLVM::ConstantOp>(loc, llvmInt32Type, builder.getI32IntegerAttr(1));
  Value arrayPtr = builder.create<LLVM::AllocaOp>(loc, arrayType0, constOneOp, 0);
  Value loadArray = builder.create<LLVM::LoadOp>(loc, arrayPtr);

  //  then we try to store to the array;
  for (const auto &en : llvm::enumerate(externOpArgsType)) {
    auto srcType = externOpArgsType[en.index()];
    if (srcType != i8PtrTy) {
      Value bitcastVal = builder.create<LLVM::BitcastOp>(loc, i8PtrTy, externOpArgs[en.index()]);
      loadArray = builder.create<LLVM::InsertValueOp>(loc, loadArray, bitcastVal, en.index());
    } else {
      loadArray = builder.create<LLVM::InsertValueOp>(loc, loadArray, externOpArgs[en.index()], en.index());
    }
  }
  (void)builder.create<LLVM::StoreOp>(loc, loadArray, arrayPtr);

  // we try to pack lambda func params and unpack these params;
  packParamsStaticShapeForLambda(lambdaFunc, lambdaFuncRealArgIndex, originalMainFuncArgSize);
  //  try to update launchOp;
  SmallVector<Value, 4> newOperands;
  size_t startIdx = isMindSpore ? kMindsporeArgOffset : 0;
  auto mainFuncArgs = mainFunc.getArguments();
  for (size_t i = startIdx; i < mainFuncArgs.size(); i++) {
    newOperands.push_back(mainFuncArgs[i]);
  }
  newOperands.push_back(arrayPtr);
  (void)builder.create<CPU::ParallelLaunchOp>(loc, TypeRange{}, newOperands, SymbolRefAttr::get(lambdaFunc));
  launchOp->erase();  // erase the old launchOp;
}

void packParamsStaticShapeOutlining(LLVM::LLVMFuncOp &mainFunc, const bool isMindSpore) {
  auto moduleOp = mainFunc->getParentOfType<ModuleOp>();
  CpuFuncPair toBeHandled;
  getHandlePairs(moduleOp, &toBeHandled, mainFunc);
  // get origin mainfunc args size;
  auto originalMainFuncArgSize = mainFunc.getArguments().size();
  // when it doesn't cooperate with mindspore, the first arg in mainFunc was a callback ptr, we should skip it
  auto argOffset = isMindSpore ? kMindsporeArgOffset : 0;
  originalMainFuncArgSize -= (unsigned long)argOffset;
  // try to pack mainFunc
  packParamsStaticShape(mainFunc, argOffset);
  // try to pack params for lambda func
  for (auto &[launchOp, lambdaFunc] : toBeHandled) {
    auto launchOpArgSize = launchOp.getArgOperands().size() - lambdaFuncRealArgIndex;
    OpBuilder builder(launchOp);
    if (launchOpArgSize == originalMainFuncArgSize) {
      buildNoExternArgsForOutlining(builder, launchOp, lambdaFunc, mainFunc, isMindSpore);
    } else {
      buildExternArgsForOutlining(builder, launchOp, lambdaFunc, mainFunc, isMindSpore, originalMainFuncArgSize);
    }
  }
  return;
}

void packParamsStaticShape(LLVM::LLVMFuncOp &funcOp, const int argOffset) {
  // pack all inputs into an array of ptr
  auto *context = funcOp.getContext();
  auto loc = funcOp.getLoc();
  auto i8PtrTy = getIntegerType(context, kIntType8);
  // skip non tensor inputs according to arg offset for cpu outlining kernels
  auto funcType = funcOp.getFunctionType();
  auto realTensorNum = funcType.getNumParams() - static_cast<uint32_t>(argOffset);
  auto ptrToArrayTy = getPtrArrayType(i8PtrTy, realTensorNum);

  auto &block = funcOp.getBody().front();
  auto newArg = block.addArgument(ptrToArrayTy, loc);

  auto builder = OpBuilder::atBlockBegin(&block);
  // generate value extraction codes to unpack packed args
  auto LoadPtrArrayOp = builder.create<LLVM::LoadOp>(loc, newArg);
  for (size_t i = 0; i < realTensorNum; ++i) {
    auto realTensorLoc = int(i) + argOffset;
    auto extractVOp = builder.create<LLVM::ExtractValueOp>(loc, LoadPtrArrayOp, i);
    auto realTensorArg = block.getArgument(static_cast<unsigned int>(realTensorLoc));
    auto dstType = realTensorArg.getType();
    if (dstType != i8PtrTy) {
      auto bitcastOp = builder.create<LLVM::BitcastOp>(loc, dstType, extractVOp);
      realTensorArg.replaceAllUsesWith(bitcastOp);
    } else {
      realTensorArg.replaceAllUsesWith(extractVOp);
    }
  }

  // erase all unpacked tensor args, skip int args
  for (size_t i = 0; i < realTensorNum; ++i) {
    block.eraseArgument(static_cast<unsigned int>(argOffset));
  }
  // Reset function type to match changed function args
  llvm::SmallVector<Type> newInTys;
  for (auto value : block.getArguments()) {
    (void)newInTys.emplace_back(value.getType());
  }
  auto newFuncType = LLVM::LLVMFunctionType::get(funcType.getReturnType(), newInTys, false);
  funcOp.setFunctionTypeAttr(TypeAttr::get(newFuncType));
}

void packParamsDynamicShape(LLVM::LLVMFuncOp &funcOp) {
  // deal with dynamic shape kernel
  // the result inputs are an array of ptr and a two dim array of int
  auto funcType = funcOp.getFunctionType();

  constexpr auto kDoubleSize = 2;
  size_t ptrParamsNum =
    llvm::count_if(funcType.getParams(), [](const Type argType) { return isa<LLVM::LLVMPointerType>(argType); });
  if (ptrParamsNum % kDoubleSize != 0) {
    // The number of ptr inputs should be even, which is the double of the
    // number of input tensors. Otherwise we will not deal with this case
    return;
  }

  // For each input tensor, the corresponding int inputs includes
  // an index, an array of length dim for tensor size, and an array of
  // length dim for tensor strides.
  llvm::SmallVector<size_t> paramsDimVector;
  llvm::SmallVector<size_t> paramsIdx = {0};
  size_t maxDim = 0;
  size_t currentDim = 0;

  // start from the first int param, which has the index of 2
  constexpr auto kIntArgOffset = 2;
  size_t idx = kIntArgOffset;
  while (idx < funcType.getParams().size()) {
    if (isa<LLVM::LLVMPointerType>(funcType.getParams()[idx])) {
      if (currentDim > maxDim) {
        maxDim = currentDim;
      }
      paramsDimVector.push_back(currentDim);
      paramsIdx.push_back(idx);
      currentDim = 0;
      idx += kIntArgOffset;
    } else {
      currentDim += 1;
      idx += 1;
    }
  }

  if (currentDim > maxDim) {
    maxDim = currentDim;
  }
  paramsDimVector.push_back(currentDim);

  auto *context = funcOp.getContext();
  llvm::SmallVector<mlir::Type> argTypes;
  auto i8PtrTy = getIntegerType(context, kIntType8);
  size_t paramsNum = ptrParamsNum / 2;
  auto ptrToArrayTy = getPtrArrayType(i8PtrTy, static_cast<unsigned int>(paramsNum));
  argTypes.push_back(ptrToArrayTy);

  // For the packed input as a two dim array of int,
  // the inner dim contains all information for each tensor input
  // while the outer dim is the list of all tensor inputs.
  auto i64ToArrayTy = LLVM::LLVMPointerType::get(
    LLVM::LLVMArrayType::get(IntegerType::get(context, kIntType64), static_cast<unsigned int>(maxDim)));
  auto i64ToDoubleArrayTy = getPtrArrayType(i64ToArrayTy, static_cast<unsigned int>(paramsNum));
  argTypes.push_back(i64ToDoubleArrayTy);
  auto newFuncType = LLVM::LLVMFunctionType::get(funcType.getReturnType(), argTypes, false);

  auto &block = funcOp.getBody().front();
  auto newPtrArg = block.addArgument(ptrToArrayTy, funcOp.getLoc());
  auto newIntArg = block.addArgument(i64ToDoubleArrayTy, funcOp.getLoc());

  auto builder = OpBuilder::atBlockBegin(&block);
  auto LoadPtrArrayOp = builder.create<LLVM::LoadOp>(funcOp.getLoc(), newPtrArg);
  auto LoadIntArrayOp = builder.create<LLVM::LoadOp>(funcOp.getLoc(), newIntArg);
  for (size_t i = 0; i < paramsNum; ++i) {
    // deal with ptr inputs
    // we drop the first ptr for each tensor input as we will not need it in
    // the future and load the second which is the address of the tensor data.
    unsigned int nullArgIdx = (unsigned int)paramsIdx[i];
    auto nullPtrType = block.getArgument(nullArgIdx).getType();
    auto loc = builder.getUnknownLoc();
    auto nullOp = builder.create<mlir::LLVM::NullOp>(loc, nullPtrType);
    block.getArgument(nullArgIdx).replaceAllUsesWith(nullOp);
    auto extractVOp = builder.create<LLVM::ExtractValueOp>(funcOp.getLoc(), LoadPtrArrayOp, i);
    constexpr auto kArrayPtrOffset = 1;
    unsigned int arrayPtrIdx = (unsigned int)paramsIdx[i] + kArrayPtrOffset;
    auto dstType = block.getArgument(arrayPtrIdx).getType();
    if (dstType != i8PtrTy) {
      auto bitcastOp = builder.create<LLVM::BitcastOp>(funcOp.getLoc(), dstType, extractVOp);
      block.getArgument(arrayPtrIdx).replaceAllUsesWith(bitcastOp);
    } else {
      block.getArgument(arrayPtrIdx).replaceAllUsesWith(extractVOp);
    }
    // deal with int inputs
    // for each tensor input, we pack all related int inputs into an array
    auto extractIntArrayOp = builder.create<LLVM::ExtractValueOp>(funcOp.getLoc(), LoadIntArrayOp, i);
    auto loadIntOp = builder.create<LLVM::LoadOp>(funcOp.getLoc(), extractIntArrayOp);
    for (size_t j = 0; j < paramsDimVector[i]; ++j) {
      auto extractIntOp = builder.create<LLVM::ExtractValueOp>(funcOp.getLoc(), loadIntOp, j);
      unsigned int intArgIdx = (unsigned int)paramsIdx[i] + (unsigned int)j + kIntArgOffset;
      block.getArgument(intArgIdx).replaceAllUsesWith(extractIntOp);
    }
  }

  for (size_t i = 0; i < funcType.getNumParams(); ++i) {
    block.eraseArgument(0);
  }

  funcOp.setFunctionTypeAttr(TypeAttr::get(newFuncType));
}

struct ParameterPackingPass : public LLVM::impl::ParameterPackingBase<ParameterPackingPass> {
  ParameterPackingPass() {}
  explicit ParameterPackingPass(bool isMindSpore) : isMindSpore(isMindSpore) {}
  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (funcOp == nullptr || funcOp.getBody().empty() ||
          (op->getAttr("mindspore_kernel") == nullptr && op->getAttr("nvvm.kernel") == nullptr)) {
        return;
      }

      auto funcType = funcOp.getFunctionType();
      if (!llvm::all_of(funcType.getParams(), [](const Type argType) {
            return isa<LLVM::LLVMPointerType>(argType) || isa<IntegerType>(argType);
          })) {
        // Only deal with case that the input type is either pointer or int.
        // The pointer inputs are addresses,
        // while the int inputs are sizes and strides for dynamic shape cases.
        return;
      }

      // For static shape kernels, the inputs are all pointers.
      bool isStatic =
        llvm::all_of(funcType.getParams(), [](const Type argType) { return isa<LLVM::LLVMPointerType>(argType); });

      // Three packing strategies for different FuncType attr:
      // CPU main function, FuncType = "akg_main_kernel_func", arg offset = 1
      // CPU Calc function, , FuncType = "akg_calculate_kernel_func", arg offset = 2
      // Other target of CPU original function, No FuncType attr
      auto FuncTypeAttr = op->getAttr(kFuncType);
      if (FuncTypeAttr == nullptr) {
        if (isStatic) {
          auto argOffset = 0;
          packParamsStaticShape(funcOp, argOffset);
        } else {
          packParamsDynamicShape(funcOp);
        }
        return;
      }

      if (hasHandleOutliningFunc) {
        return;
      }

      auto FuncAttrStr = FuncTypeAttr.dyn_cast<StringAttr>().getValue().str();
      if (FuncAttrStr == kCpuMainFunc && isStatic) {
        packParamsStaticShapeOutlining(funcOp, isMindSpore);
      }
      hasHandleOutliningFunc = true;
    });
  }

  bool isMindSpore = false;
  bool hasHandleOutliningFunc = false;
};
}  // namespace

std::unique_ptr<Pass> LLVM::createParameterPackingPass() { return std::make_unique<ParameterPackingPass>(); }

std::unique_ptr<Pass> LLVM::createParameterPackingPass(bool isMindSpore) {
  return std::make_unique<ParameterPackingPass>(isMindSpore);
}
