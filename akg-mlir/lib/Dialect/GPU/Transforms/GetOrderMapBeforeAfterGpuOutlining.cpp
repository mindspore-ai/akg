/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include <fstream>
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
#include "akg/Utils/Constants.h"

namespace mlir {
#define GEN_PASS_DECL_GETORDERMAPBEFOREAFTERGPUOUTLINING
#define GEN_PASS_DEF_GETORDERMAPBEFOREAFTERGPUOUTLINING
#include "akg/Dialect/GPU/Passes.h.inc"

}  // namespace mlir

namespace mlir {
using mlir::ArrayRef;
using mlir::BlockArgument;
using mlir::dyn_cast;
using mlir::SmallVector;
using mlir::Value;
using mlir::ValueRange;

namespace gpu {
namespace {

bool IdxIsInVector(size_t funcIdx, SmallVector<int, kSmallVectorSizeEight> mapResult) {
  return std::any_of(mapResult.begin(), mapResult.end(),
                     [&funcIdx](int idx) { return idx == static_cast<int>(funcIdx); });
}

void UpdateMapResultForAlloc(Value alloc, ValueRange operands, SmallVector<int, kSmallVectorSizeEight> &mapResult,
                             size_t funcIdx) {
  for (size_t idx = 0; idx < operands.size(); idx++) {
    if (alloc == operands[idx]) {
      mapResult[idx] = static_cast<int>(funcIdx);
      break;
    }
  }
}

bool isPermutation(const SmallVector<int, kSmallVectorSizeEight> vec) {
  llvm::SmallSet<int, kSmallVectorSizeEight> elemSet;
  size_t len = vec.size();
  for (int element : vec) {
    if (element < 0 || element >= static_cast<int>(len)) {
      return false;
    }
    (void)elemSet.insert(element);
  }
  return elemSet.size() == len;
}

Value FindAllocOpForFuncArg(mlir::func::FuncOp funcOp, BlockArgument targetArg) {
  mlir::memref::CopyOp targetCopyOp = nullptr;
  funcOp.walk([&targetArg, &targetCopyOp](mlir::memref::CopyOp op) {
    if (op.getTarget() == targetArg) {
      targetCopyOp = op;
    }
  });
  if (!targetCopyOp) {
    (void)funcOp.emitError("Error: can't find memref::CopyOp \n");
    return {};
  }
  auto prevOp = targetCopyOp.getSource().getDefiningOp();
  if (auto alloc = dyn_cast<mlir::memref::AllocOp>(prevOp)) {
    return alloc.getResult();
  }
  (void)funcOp.emitError("Error: next Op is not memref::AllocOp \n");
  return {};
}

static void matchOperandIndex(Value v, ArrayRef<BlockArgument> funcArguments,
                              SmallVector<int, kSmallVectorSizeEight> &mapResult, size_t idx) {
  for (size_t funcIdx = 0; funcIdx < funcArguments.size(); funcIdx++) {
    if (funcArguments[funcIdx] == v) {
      mapResult[idx] = static_cast<int>(funcIdx);
      return;
    }
  }
}

struct GetOrderMapBeforeAfterGpuOutlining
    : public mlir::impl::GetOrderMapBeforeAfterGpuOutliningBase<GetOrderMapBeforeAfterGpuOutlining> {
  GetOrderMapBeforeAfterGpuOutlining() = default;
  explicit GetOrderMapBeforeAfterGpuOutlining(const std::string &path) { this->path = path; }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();

    auto funcArguments = funcOp.getArguments();
    SmallVector<int, kSmallVectorSizeEight> mapResult(funcArguments.size(), -1);

    mlir::gpu::LaunchFuncOp launchFuncOp = [&funcOp]() {
      mlir::gpu::LaunchFuncOp result;
      funcOp.walk([&result](mlir::gpu::LaunchFuncOp op) { result = op; });
      return result;
    }();
    if (!launchFuncOp) {
      return;
    }
    auto operands = launchFuncOp.getKernelOperands();
    if (funcArguments.size() != operands.size()) {
      (void)funcOp.emitError("Error: funcArguments size is not equal to launchFuncOp operands size, plz check it. \n");
      return;
    }
    for (size_t idx = 0; idx < operands.size(); idx++) {
      matchOperandIndex(operands[idx], funcArguments, mapResult, idx);
    }
    for (size_t funcIdx = 0; funcIdx < funcArguments.size(); funcIdx++) {
      if (!IdxIsInVector(funcIdx, mapResult)) {
        auto alloc = FindAllocOpForFuncArg(funcOp, funcArguments[funcIdx]);
        if (!alloc) {
          continue;
        }
        UpdateMapResultForAlloc(alloc, operands, mapResult, funcIdx);
      }
    }

    // validity check
    if (!isPermutation(mapResult)) {
      (void)funcOp.emitError("Error: mapResult is not a permutation; plz check the pass. \n");
      return;
    }

    if (this->path.empty()) {
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
