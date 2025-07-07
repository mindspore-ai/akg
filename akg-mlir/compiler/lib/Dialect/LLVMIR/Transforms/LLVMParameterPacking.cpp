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

LLVM::LLVMPointerType getIntegerType(MLIRContext *context, const unsigned int integerSize) {
  return LLVM::LLVMPointerType::get(context, integerSize);
}

LLVM::LLVMPointerType getPtrArrayType(const LLVM::LLVMPointerType ptrTy, const unsigned int arrayLen) {
  return LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(ptrTy, arrayLen).getContext());
}

void packParamsStaticShape(LLVM::LLVMFuncOp &funcOp, const int argOffset) {
  // pack all inputs into an array of ptr
  auto *context = funcOp.getContext();
  auto loc = funcOp.getLoc();
  // skip non tensor inputs according to arg offset for cpu outlining kernels
  auto funcType = funcOp.getFunctionType();
  auto realTensorNum = funcType.getNumParams() - static_cast<uint32_t>(argOffset);

  auto i8PtrTy = LLVM::LLVMPointerType::get(context);
  auto &block = funcOp.getBody().front();
  auto newArg = block.addArgument(i8PtrTy, loc);

  auto builder = OpBuilder::atBlockBegin(&block);
  for (size_t i = 0; i < realTensorNum; ++i) {
    auto realTensorLoc = int(i) + argOffset;
    auto gepOp = builder.create<LLVM::GEPOp>(loc, i8PtrTy, i8PtrTy, newArg, ArrayRef<LLVM::GEPArg>{int(i)});
    auto loadOp = builder.create<LLVM::LoadOp>(loc, i8PtrTy, gepOp);
    auto realTensorArg = block.getArgument(static_cast<unsigned int>(realTensorLoc));
    realTensorArg.replaceAllUsesWith(loadOp);
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
    LLVM::LLVMArrayType::get(IntegerType::get(context, kIntType64), static_cast<unsigned int>(maxDim)).getContext());
  auto i64ToDoubleArrayTy = getPtrArrayType(i64ToArrayTy, static_cast<unsigned int>(paramsNum));
  argTypes.push_back(i64ToDoubleArrayTy);
  auto newFuncType = LLVM::LLVMFunctionType::get(funcType.getReturnType(), argTypes, false);

  auto &block = funcOp.getBody().front();
  auto newPtrArg = block.addArgument(ptrToArrayTy, funcOp.getLoc());
  auto newIntArg = block.addArgument(i64ToDoubleArrayTy, funcOp.getLoc());

  auto builder = OpBuilder::atBlockBegin(&block);
  auto LoadPtrArrayOp = builder.create<LLVM::LoadOp>(funcOp.getLoc(), ptrToArrayTy, newPtrArg);
  auto LoadIntArrayOp = builder.create<LLVM::LoadOp>(funcOp.getLoc(), ptrToArrayTy, newIntArg);
  for (size_t i = 0; i < paramsNum; ++i) {
    // deal with ptr inputs
    // we drop the first ptr for each tensor input as we will not need it in
    // the future and load the second which is the address of the tensor data.
    unsigned int nullArgIdx = (unsigned int)paramsIdx[i];
    auto nullPtrType = block.getArgument(nullArgIdx).getType();
    auto loc = builder.getUnknownLoc();
    auto nullOp = builder.create<mlir::LLVM::ZeroOp>(loc, nullPtrType);
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
    auto loadIntOp = builder.create<LLVM::LoadOp>(funcOp.getLoc(), ptrToArrayTy, extractIntArrayOp);
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
