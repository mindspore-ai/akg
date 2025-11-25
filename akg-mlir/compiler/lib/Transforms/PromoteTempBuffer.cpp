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

#include <algorithm>
#include <iterator>
#include "akg/Transforms/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#ifndef GEN_PASS_DECL_PROMOTETEMPBUFFER
#define GEN_PASS_DECL_PROMOTETEMPBUFFER
#ifndef GEN_PASS_DEF_PROMOTETEMPBUFFER
#define GEN_PASS_DEF_PROMOTETEMPBUFFER
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "promote-temp-buffer"

using namespace mlir;
namespace {

// ===----------------------------------------------------------------------===//
// PromoteTempBufferPass
// ===----------------------------------------------------------------------===//

// This pass:
//    1. promotes the temp buffer generated in host func to gpu shared mem;
//    2. move the alloc op of fast memory (i.e. cache level > 1)
//       from gpu kernel func body into workgroup/private gpu func attributes;
//    3. delete the dealloc op of promoted temp buffer.

constexpr auto kSharedCache = 3;
constexpr auto kLocalCache = 5;

struct PromoteTempBufferPass : public PromoteTempBufferBase<PromoteTempBufferPass> {
 public:
  void runOnOperation() override;

 private:
  size_t hasFail = 0;
  bool promoteToPrivateMemory(gpu::GPUFuncOp op, unsigned arg) const {
    if (arg < op.getNumArguments()) {
      Value value = op.getArgument(arg);
      return addToPrivateMemory(op, value);
    } else {
      llvm::errs() << "Exceed limit\n";
      return false;
    }
  }

  bool addToPrivateMemory(gpu::GPUFuncOp op, Value value) const {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || !type.hasStaticShape()) {
      llvm::outs() << "Can only promote static shape memrefs for now.\n";
      return false;
    }
    auto privateAddressSpace = gpu::AddressSpaceAttr::get(op->getContext(), gpu::GPUDialect::getPrivateAddressSpace());
    auto bufferType = MemRefType::get(type.getShape(), type.getElementType(), AffineMap{}, privateAddressSpace);
    Value attribution = op.addPrivateAttribution(bufferType, value.getLoc());
    value.replaceAllUsesWith(attribution);
    return true;
  }

  bool promoteToWorkgroupMemory(gpu::GPUFuncOp op, unsigned arg) {
    if (arg < op.getNumArguments()) {
      Value value = op.getArgument(arg);
      return addToWorkgroupMemory(op, value);
    } else {
      llvm::errs() << "Exceed limit\n";
      return false;
    }
  }

  bool addToWorkgroupMemory(gpu::GPUFuncOp op, Value value) const {
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || !type.hasStaticShape()) {
      llvm::outs() << "Can only promote static shape memrefs for now.\n";
      return false;
    }
    auto workgroupMemoryAddressSpace =
      gpu::AddressSpaceAttr::get(op->getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    auto bufferType = MemRefType::get(type.getShape(), type.getElementType(), AffineMap{}, workgroupMemoryAddressSpace);
    Value attribution = op.addWorkgroupAttribution(bufferType, value.getLoc());
    value.replaceAllUsesWith(attribution);
    return true;
  }

  void findPromoteBufferArgIdx() {
    for (auto it : tempBuffers) {
      auto globalTempBuffer = it.second;
      auto operands = oldLaunchFunc.getKernelOperands();
      for (size_t i = 0; i < operands.size(); i++) {
        auto operand = operands[i];
        for (auto buf : globalTempBuffer) {
          if (operand == buf) {
            promotedArgIdx.push_back(std::make_pair(i, it.first));
            break;
          }
        }
      }
    }
  }

  void findLaunchFunc() {
    getOperation()->walk([&](gpu::LaunchFuncOp launchOp) { oldLaunchFunc = launchOp; });
    if (!oldLaunchFunc) {
      llvm::report_fatal_error(llvm::StringRef("Error during promote temp buffer: no gpu launch func."));
    }
  }

  void EraseDeallocOfTempBuffers() {
    for (auto it : tempBuffers) {
      for (auto buf : it.second) {
        SmallVector<Operation *> toErase;
        for (auto user : buf->getUsers()) {
          if (dyn_cast<memref::DeallocOp>(user)) {
            // Do not directly erase user during getUsers, will get nullptr error for no reason
            toErase.push_back(user);
          }
        }
        for (auto dealloc : toErase) {
          dealloc->erase();
        }
      }
    }
  }

  void EraseAllocOfTempBuffers() {
    for (auto it : tempBuffers) {
      for (auto buf : it.second) {
        if (!buf->use_empty()) {
          llvm::outs() << "Temp buffer remove fail, please check.\n";
          continue;
        }
        buf->erase();
      }
    }
  }

  void createPromotedGpuFunc() {
    getOperation()->walk([&](gpu::GPUFuncOp gpuFunc) {
      for (auto [idx, bufLevel] : promotedArgIdx) {
        if (bufLevel <= kSharedCache) {
          hasFail |= (promoteToWorkgroupMemory(gpuFunc, idx) == false ? 1 : 0);
        } else {
          hasFail |= (promoteToPrivateMemory(gpuFunc, idx) == false ? 1 : 0);
        }
      }

      if (hasFail != 0) {
        return;
      }

      // Do all the promotion first then we can start to erase arguments
      for (auto it = promotedArgIdx.rbegin(); it != promotedArgIdx.rend(); ++it) {
        auto idx = it->first;
        gpuFunc.getBody().front().eraseArgument(idx);
      }

      // After promote main func's temp buffers, we can start to promote gpu func's buffers to the end
      for (auto it : tempBuffers) {
        auto cacheLevel = it.first;
        for (auto buf : it.second) {
          auto value = buf.getResult();
          if (cacheLevel == kSharedCache) {
            hasFail |= (addToWorkgroupMemory(gpuFunc, value) == false ? 1 : 0);
          } else if (cacheLevel == kLocalCache) {
            hasFail |= (addToPrivateMemory(gpuFunc, value) == false ? 1 : 0);
          }
        }
      }

      if (hasFail != 0) {
        return;
      }

      auto functionType = gpuFunc.getFunctionType();
      SmallVector<Type, 4> newInputTypes;
      for (unsigned i = 0, e = functionType.getNumInputs(); i < e; ++i) {
        bool isPromotedArg = false;
        for (auto it : promotedArgIdx) {
          if (it.first == i) {
            isPromotedArg = true;
            break;
          }
        }
        if (isPromotedArg) {
          continue;
        }
        newInputTypes.push_back(functionType.getInput(i));
      }
      auto newFuncType = FunctionType::get(gpuFunc.getContext(), newInputTypes, functionType.getResults());
      gpuFunc.setFunctionType(newFuncType);
    });
  }

  void createPromotedLaunchFunc() {
    if (hasFail != 0) {
      return;
    }
    gpu::GPUFuncOp newGpuFunc;
    getOperation()->walk([&](gpu::GPUFuncOp gpuFunc) { newGpuFunc = gpuFunc; });
    if (!newGpuFunc) {
      llvm::report_fatal_error(llvm::StringRef("Error during promote temp buffer: no gpu func."));
    }
    SetVector<Value> newKernelOperands;
    for (size_t i = 0; i < oldLaunchFunc.getNumKernelOperands(); ++i) {
      auto op = oldLaunchFunc.getKernelOperand(i);
      bool isPromotedArg = false;
      for (auto it : promotedArgIdx) {
        if (it.first == i) {
          isPromotedArg = true;
          break;
        }
      }
      if (isPromotedArg) {
        continue;
      }
      (void)newKernelOperands.insert(op);
    }
    OpBuilder builder(oldLaunchFunc);
    Value asyncToken = oldLaunchFunc.getAsyncToken();
    auto newLaunchFunc = builder.create<gpu::LaunchFuncOp>(
      oldLaunchFunc.getLoc(), newGpuFunc, oldLaunchFunc.getGridSizeOperandValues(),
      oldLaunchFunc.getBlockSizeOperandValues(), oldLaunchFunc.getDynamicSharedMemorySize(),
      newKernelOperands.getArrayRef(), asyncToken ? asyncToken.getType() : nullptr,
      oldLaunchFunc.getAsyncDependencies());
    oldLaunchFunc.replaceAllUsesWith(newLaunchFunc);
    oldLaunchFunc.erase();
  }

  std::map<int, SmallVector<memref::AllocOp>> tempBuffers;
  std::vector<std::pair<size_t, int>> promotedArgIdx;
  gpu::LaunchFuncOp oldLaunchFunc;
};

void PromoteTempBufferPass::runOnOperation() {
  tempBuffers = CommonUtils::findTempBuffer(getOperation());
  findLaunchFunc();
  findPromoteBufferArgIdx();

  // Erase dealloc before promotion so that we can correctly get the users of alloc.
  EraseDeallocOfTempBuffers();
  createPromotedGpuFunc();
  createPromotedLaunchFunc();

  EraseAllocOfTempBuffers();
}
}  // end anonymous namespace

std::unique_ptr<Pass> mlir::createPromoteTempBufferPass() { return std::make_unique<PromoteTempBufferPass>(); }
