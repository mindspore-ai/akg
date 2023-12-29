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

#include "akg/Dialect/GPU/Transforms/GetOrderMapBeforeAfterGpuOutlining.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <fstream>

namespace mlir {
#define GEN_PASS_DECL_GETORDERMAPBEFOREAFTERGPUOUTLINING
#define GEN_PASS_DEF_GETORDERMAPBEFOREAFTERGPUOUTLINING
#include "akg/Dialect/GPU/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace mlir {
namespace gpu {
namespace {

bool IdxIsInVector(size_t funcIdx, SmallVector<int, 8> mapResult) {
  for (auto idx : mapResult) {
    if (idx == static_cast<int>(funcIdx)) {
      return true;
    }
  }
  return false;
}

bool isPermutation(const SmallVector<int, 8> vec) {
  llvm::SmallSet<int, 8> elemSet;
  size_t len = vec.size();
  for (int element : vec) {
    if (element < 0 || element >= static_cast<int>(len)) {
      return false;
    }
    (void)elemSet.insert(element);
  }
  return elemSet.size() == len;
}

Value FindAllocOpForFuncArg(func::FuncOp funcOp, BlockArgument targetArg) {
  memref::CopyOp targetCopyOp = nullptr;
  funcOp.walk([&](memref::CopyOp op) {
    if (op.getTarget() == targetArg) {
      targetCopyOp = op;
    }
  });
  if (!targetCopyOp) {
    (void)funcOp.emitError("Error: can't find memref::CopyOp \n");
    return Value();
  }
  auto prevOp = targetCopyOp.getSource().getDefiningOp();
  if (auto alloc = dyn_cast<memref::AllocOp>(prevOp)) {
    return alloc.getResult();
  } else {
    (void)funcOp.emitError("Error: next Op is not memref::AllocOp \n");
  }
  return Value();
}

struct GetOrderMapBeforeAfterGpuOutlining
    : public impl::GetOrderMapBeforeAfterGpuOutliningBase<GetOrderMapBeforeAfterGpuOutlining> {
  GetOrderMapBeforeAfterGpuOutlining() = default;
  explicit GetOrderMapBeforeAfterGpuOutlining(const std::string path) { this->path = path; }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    auto funcArguments = funcOp.getArguments();
    SmallVector<int, 8> mapResult(funcArguments.size(), -1);

    gpu::LaunchFuncOp launchFuncOp;
    funcOp.walk([&](gpu::LaunchFuncOp op) { launchFuncOp = op; });
    if (!launchFuncOp) {
      return;
    }
    auto operands = launchFuncOp.getKernelOperands();
    if (funcArguments.size() != operands.size()) {
      (void)funcOp.emitError("Error: funcArguments size is not equal to launchFuncOp operands size, plz check it. \n");
      return;
    }
    for (size_t idx = 0; idx < operands.size(); idx++) {
      for (size_t funcIdx = 0; funcIdx < funcArguments.size(); funcIdx++) {
        if (funcArguments[funcIdx] == operands[idx]) {
          mapResult[idx] = funcIdx;
          break;
        }
      }
    }
    for (size_t funcIdx = 0; funcIdx < funcArguments.size(); funcIdx++) {
      if (!IdxIsInVector(funcIdx, mapResult)) {
        auto alloc = FindAllocOpForFuncArg(funcOp, funcArguments[funcIdx]);
        if (!alloc) {
          continue;
        }
        for (size_t idx = 0; idx < operands.size(); idx++) {
          if (alloc == operands[idx]) {
            mapResult[idx] = static_cast<int>(funcIdx);
            break;
          }
        }
      }
    }

    // validity check
    if (!isPermutation(mapResult)) {
      (void)funcOp.emitError("Error: mapResult is not a permutation; plz check the pass. \n");
      return;
    }

    std::ofstream output(this->path);
    for (size_t idx = 0; idx < operands.size(); idx++) {
      output << mapResult[idx] << " ";
    }
    output.close();
  }
};

}  // namespace
}  // namespace gpu
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createGetOrderMapBeforeAfterGpuOutliningPass() {
  return std::make_unique<gpu::GetOrderMapBeforeAfterGpuOutlining>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createGetOrderMapBeforeAfterGpuOutliningPass(
  std::string path) {
  return std::make_unique<gpu::GetOrderMapBeforeAfterGpuOutlining>(path);
}
