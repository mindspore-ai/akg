/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Transforms/InsertLoadAndStore.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "affine-for-vectorize"

namespace mlir {
#define GEN_PASS_DECL_INSERTLOADANDSTORE
#define GEN_PASS_DEF_INSERTLOADANDSTORE
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kBufferSizeInByteAttr = "buffer_size_in_byte";

int64_t getTensorBufferSize(Type type) {
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    auto elementType = tensorType.getElementType();
    auto shape = tensorType.getShape();
    int64_t size = 1;
    for (auto dimSize : shape) {
      size *= dimSize;
    }
    return size * elementType.getIntOrFloatBitWidth();
  }
  return 0;
}

class InsertLoadAndStore : public impl::InsertLoadAndStoreBase<InsertLoadAndStore> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    mlir::OpBuilder builder(funcOp);

    SmallVector<tensor::ExtractSliceOp> extractSliceOps;
    SmallVector<tensor::InsertSliceOp> insertSliceOps;
    funcOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
      auto emptyOp = extractSliceOp.getSource().getDefiningOp<tensor::EmptyOp>();
      if (!emptyOp) {
        extractSliceOps.push_back(extractSliceOp);
      }
      LLVM_DEBUG(llvm::dbgs() << "match: " << extractSliceOp << "\n");
    });
    funcOp.walk([&](tensor::InsertSliceOp insertSliceOp) {
      auto emptyOp = insertSliceOp.getDest().getDefiningOp<tensor::EmptyOp>();
      if (!emptyOp) {
        insertSliceOps.push_back(insertSliceOp);
      }
      LLVM_DEBUG(llvm::dbgs() << "match: " << insertSliceOp << "\n");
    });

    for (auto sliceOp : extractSliceOps) {
      // Collect all users before creating loadOp to avoid iterator invalidation
      Value sliceResult = sliceOp.getResult();
      SmallVector<Operation *> allUsers;
      for (OpOperand &use : sliceResult.getUses()) {
        allUsers.push_back(use.getOwner());
      }

      builder.setInsertionPointAfter(sliceOp);
      auto loc = sliceOp.getLoc();
      auto emptyTensor = builder.create<tensor::EmptyOp>(loc, sliceOp.getResultType().getShape(),
                                                         sliceOp.getResultType().getElementType());
      auto loadOp = builder.create<hfusion::LoadOp>(loc, ValueRange{sliceResult}, ValueRange{emptyTensor});
      auto markOp = builder.create<annotation::MarkOp>(loc, loadOp.getResult(0));
      markOp->setAttr(kBufferSizeInByteAttr, builder.getI64IntegerAttr(getTensorBufferSize(sliceOp.getResultType())));
      // Get the result of LoadOp
      Value loadResult = loadOp->getResult(0);

      // Replace all uses except the one in loadOp itself
      // This prevents loadOp from using its own result
      for (Operation *user : allUsers) {
        if (user != loadOp) {
          user->replaceUsesOfWith(sliceResult, loadResult);
        }
      }
    }
    for (auto sliceOp : insertSliceOps) {
      // Collect all users before creating storeOp to avoid iterator invalidation
      Value src = sliceOp.getSource();
      SmallVector<Operation *> allUsers;
      for (OpOperand &use : src.getUses()) {
        allUsers.push_back(use.getOwner());
      }
      builder.setInsertionPoint(sliceOp);
      auto loc = sliceOp.getLoc();
      auto markOp = builder.create<annotation::MarkOp>(loc, src);
      markOp->setAttr(kBufferSizeInByteAttr, builder.getI64IntegerAttr(getTensorBufferSize(src.getType())));
      auto extractOp = builder.create<tensor::ExtractSliceOp>(loc, sliceOp.getSourceType(), sliceOp.getDest(),
                                                              sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
                                                              sliceOp.getMixedStrides());
      auto storeOp = builder.create<hfusion::StoreOp>(loc, ValueRange{src}, ValueRange{extractOp});
      // Get the result of storeOp
      Value storeResult = storeOp->getResult(0);

      // Replace all uses except the one in storeOp itself
      // This prevents storeOp from using its own result
      for (Operation *user : allUsers) {
        if (user != storeOp) {
          user->replaceUsesOfWith(src, storeResult);
        }
      }
    }
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::createInsertLoadAndStorePass() { return std::make_unique<InsertLoadAndStore>(); }
