/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DECL_COPYRETURNEDBLOCKARGS
#define GEN_PASS_DEF_COPYRETURNEDBLOCKARGS
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

namespace mlir {

namespace {
/// Insert `tensor.empty` + `linalg.copy` / `memref.alloc` + `linalg.copy` before
/// `func.return` when an operand is a function block argument.
Value createCopyForReturnedArg(OpBuilder &b, Location loc, Value src) {
  if (auto rankedTy = dyn_cast<RankedTensorType>(src.getType())) {
    SmallVector<Value> dynamicSizes;
    for (unsigned i = 0; i < rankedTy.getRank(); ++i) {
      if (rankedTy.isDynamicDim(i)) {
        dynamicSizes.push_back(b.create<tensor::DimOp>(loc, src, i));
      }
    }
    Value empty = b.create<tensor::EmptyOp>(loc, rankedTy, dynamicSizes);
    return b.create<linalg::CopyOp>(loc, rankedTy, src, empty, ArrayRef<NamedAttribute>{}).getResult(0);
  }
  if (auto memTy = dyn_cast<MemRefType>(src.getType())) {
    SmallVector<Value> sizes;
    for (unsigned i = 0; i < memTy.getRank(); ++i) {
      if (memTy.isDynamicDim(i)) sizes.push_back(b.create<memref::DimOp>(loc, src, i));
    }
    Value alloc = b.create<memref::AllocOp>(loc, memTy, sizes);
    b.create<linalg::CopyOp>(loc, src, alloc);
    return alloc;
  }
  return nullptr;
}

class CopyReturnedBlockArgs : public impl::CopyReturnedBlockArgsBase<CopyReturnedBlockArgs> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Block &entry = func.getFunctionBody().front();

    SmallVector<func::ReturnOp> returns;
    func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });

    for (func::ReturnOp ret : returns) {
      OpBuilder b(ret);
      llvm::DenseMap<BlockArgument, Value> blockArgToCopy;
      SmallVector<Value> newOperands;
      newOperands.reserve(ret.getNumOperands());
      bool changed = false;

      for (Value v : ret.getOperands()) {
        auto ba = dyn_cast<BlockArgument>(v);
        if (!ba || ba.getOwner() != &entry) {
          newOperands.push_back(v);
          continue;
        }
        auto it = blockArgToCopy.find(ba);
        if (it != blockArgToCopy.end()) {
          newOperands.push_back(it->second);
          changed = true;
          continue;
        }
        Value copy = createCopyForReturnedArg(b, ret.getLoc(), ba);
        if (!copy) {
          newOperands.push_back(v);
          continue;
        }
        blockArgToCopy.insert({ba, copy});
        newOperands.push_back(copy);
        changed = true;
      }

      if (!changed) {
        continue;
      }
      b.create<func::ReturnOp>(ret.getLoc(), newOperands);
      ret.erase();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCopyReturnedBlockArgsPass() {
  return std::make_unique<CopyReturnedBlockArgs>();
}
}  // namespace mlir
