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

#include "akg/Dialect/Linalg/Transforms/LinalgCopyBufferize.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGCOPYBUFFERIZE
#define GEN_PASS_DEF_LINALGCOPYBUFFERIZE
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {
constexpr auto kVectorInitSize4 = 4;

Value castTensorToMemref(OpBuilder &builder, Value value) {
  Type type = value.getType();
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    Type memrefType = MemRefType::get(rankedTensorType.getShape(), rankedTensorType.getElementType());
    return builder.create<bufferization::ToMemrefOp>(value.getLoc(), memrefType, value).getResult();
  }

  if (auto unrankedTensorType = type.dyn_cast<UnrankedTensorType>()) {
    Type memrefType = UnrankedMemRefType::get(unrankedTensorType.getElementType(), {});
    return builder.create<bufferization::ToMemrefOp>(value.getLoc(), memrefType, value).getResult();
  }

  return value;
}

struct LinalgCopyBufferize : public impl::LinalgCopyBufferizeBase<LinalgCopyBufferize> {
 public:
  LinalgCopyBufferize() = default;
  explicit LinalgCopyBufferize(const bool keepOuts) { this->keepOuts = keepOuts; }
  // LinalgCopyBufferize(const LinalgCopyBufferizeOptions &options) : LinalgCopyBufferizeBase(options) {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    // Record linalgCopyOp, Result to be erased and to be kept
    // For GPU, bufferize the linalg copy op, and keep the fake output of InplaceAssign
    // For CPU, remove the fake output from returnOp and funcArguments
    SmallVector<Operation *, kVectorInitSize4> targetCopyOps;
    BitVector erasedResultIndices(funcOp.getFunctionType().getNumResults());
    SmallVector<Value, kVectorInitSize4> keptResultValue;
    funcOp.walk([&](func::ReturnOp op) {
      for (const auto &it : llvm::enumerate(op.getOperands())) {
        auto oprd = it.value();
        auto index = it.index();
        Operation *returnInputOp = oprd.getDefiningOp();
        if (returnInputOp == nullptr) {
          return;
        }
        if (isa<linalg::CopyOp>(returnInputOp)) {
          (void)erasedResultIndices.set(static_cast<unsigned int>(index));
          (void)targetCopyOps.emplace_back(returnInputOp);
          if (this->keepOuts) {
            (void)keptResultValue.emplace_back(dyn_cast<linalg::CopyOp>(returnInputOp).getInputs()[0]);
          }
          continue;
        } else if (isa<bufferization::ToMemrefOp>(returnInputOp)) {
          auto memrefInputOp = returnInputOp->getOperands()[0].getDefiningOp();
          if (isa<linalg::CopyOp>(memrefInputOp)) {
            (void)erasedResultIndices.set(static_cast<unsigned int>(index));
            targetCopyOps.emplace_back(memrefInputOp);
            if (this->keepOuts) {
              (void)keptResultValue.emplace_back(dyn_cast<linalg::CopyOp>(memrefInputOp).getInputs()[0]);
            }
            continue;
          }
        }
        (void)keptResultValue.emplace_back(oprd);
      }
    });

    if (targetCopyOps.size() == 0) {
      return;
    }
    // Cpu only: Erase CopyOp's output from func types/attrs/args
    if (!keepOuts) {
      funcOp.eraseResults(erasedResultIndices);
    }
    // Create new ReturnOp without CopyOp's output
    funcOp.walk([&](func::ReturnOp op) {
      OpBuilder builder(op);
      (void)builder.create<func::ReturnOp>(op.getLoc(), keptResultValue);
      op.erase();
    });
    // Rewrite linalg Copy to Memref Copy
    funcOp.walk([&](linalg::CopyOp op) {
      if (std::find(targetCopyOps.begin(), targetCopyOps.end(), op) == targetCopyOps.end()) {
        return;
      }
      // Cast tensor to memref if CopyOp's input and output are tensors
      OpBuilder builder(op);
      auto src = castTensorToMemref(builder, op.getInputs()[0]);
      auto dst = castTensorToMemref(builder, op.getOutputs()[0]);
      // Create new MemrefCopyOp without result
      (void)builder.create<memref::CopyOp>(op.getLoc(), src, dst);
      op.erase();
    });
  }
};
}  // namespace


std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgCopyBufferizePass() {
  return std::make_unique<LinalgCopyBufferize>();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgCopyBufferizePass(bool keepOuts) {
  return std::make_unique<LinalgCopyBufferize>(keepOuts);
}

