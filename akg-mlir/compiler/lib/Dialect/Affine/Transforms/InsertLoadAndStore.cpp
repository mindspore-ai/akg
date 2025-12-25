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

#include <algorithm>
#include <numeric>
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "insert-load and store"

namespace mlir {
#define GEN_PASS_DECL_INSERTLOADANDSTORE
#define GEN_PASS_DEF_INSERTLOADANDSTORE
#include "akg/Dialect/Affine/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kBufferSizeInByteAttr = "buffer_size_in_byte";

// Helper function to get the initial value if the value is a result from affine.for
static Value getInitValueFromForResult(Value value) {
  if (auto result = dyn_cast<OpResult>(value)) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(result.getOwner())) {
      // Find which result index this is
      auto results = forOp.getResults();
      for (unsigned i = 0, e = results.size(); i < e; ++i) {
        if (results[i] == value) {
          // Get the corresponding iter-arg initial value
          auto inits = forOp.getInits();
          if (i < inits.size()) {
            return inits[i];
          }
          break;
        }
      }
    }
  }
  return nullptr;
}

int64_t getTensorBufferSize(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    auto elementType = tensorType.getElementType();
    auto shape = tensorType.getShape();
    int64_t size = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    return size * elementType.getIntOrFloatBitWidth();
  }
  return 0;
}

bool isFunctionArgument(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *block = blockArg.getOwner();
    Operation *parentOp = block->getParentOp();
    if (dyn_cast<func::FuncOp>(parentOp)) {
      return true;
    }
    if (auto forOp = dyn_cast<affine::AffineForOp>(parentOp)) {
      // The first argument (index 0) is the induction variable
      // Subsequent arguments (index 1, 2, ...) are iteration arguments (iter-args)
      unsigned argNumber = blockArg.getArgNumber();
      if (argNumber == 0) {
        // This is the induction variable, not an iter-arg
        return false;
      }
      // Get iter-arg block arguments from the region
      // getRegionIterArgs() returns block arguments starting from index 1 (skipping induction var)
      auto regionIterArgs = forOp.getRegionIterArgs();
      // Get the iter-arg index (argNumber - 1 because arg 0 is induction variable)
      unsigned iterArgIndex = argNumber - 1;
      // Verify that this blockArg is actually an iter-arg
      if (iterArgIndex >= regionIterArgs.size()) {
        return false;
      }
      // Get the initial value (operand) of the corresponding iter-arg
      // getInits() returns the initial values passed as operands to affine.for
      auto inits = forOp.getInits();
      if (iterArgIndex >= inits.size()) {
        return false;
      }
      // Get the initial value of the corresponding iter-arg
      Value initValue = inits[iterArgIndex];
      // If the initial value is a result from another affine.for, get its init value
      if (Value forInitValue = getInitValueFromForResult(initValue)) {
        // Recursively check the parent for loop's iter-arg initial value
        return isFunctionArgument(forInitValue);
      }
      return isFunctionArgument(initValue);
    }
    return false;
  }
  // If the value is a result from affine.for, continue recursion
  if (Value forInitValue = getInitValueFromForResult(value)) {
    // Recursively check the iter-arg initial value
    return isFunctionArgument(forInitValue);
  }
  return false;
}

class InsertLoadAndStore : public impl::InsertLoadAndStoreBase<InsertLoadAndStore> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    mlir::OpBuilder builder(funcOp);

    SmallVector<tensor::ExtractSliceOp> extractSliceOps;
    SmallVector<tensor::InsertSliceOp> insertSliceOps;
    funcOp.walk([&](tensor::ExtractSliceOp extractSliceOp) {
      if (isFunctionArgument(extractSliceOp.getSource())) {
        extractSliceOps.push_back(extractSliceOp);
      }
      LLVM_DEBUG(llvm::dbgs() << "match: " << extractSliceOp << "\n");
    });
    funcOp.walk([&](tensor::InsertSliceOp insertSliceOp) {
      if (isFunctionArgument(insertSliceOp.getDest())) {
        insertSliceOps.push_back(insertSliceOp);
      }
      LLVM_DEBUG(llvm::dbgs() << "match: " << insertSliceOp << "\n");
    });

    for (auto sliceOp : extractSliceOps) {
      // Collect all users before creating loadOp to avoid iterator invalidation
      Value sliceResult = sliceOp.getResult();
      SmallVector<Operation *> allUsers;
      auto uses = sliceResult.getUses();
      allUsers.reserve(std::distance(uses.begin(), uses.end()));
      std::transform(uses.begin(), uses.end(), std::back_inserter(allUsers),
                     [](OpOperand &use) { return use.getOwner(); });

      builder.setInsertionPointAfter(sliceOp);
      auto loc = sliceOp.getLoc();
      SmallVector<int64_t> staticDims;
      SmallVector<Value> dynDims;
      dispatchIndexOpFoldResults(sliceOp.getMixedSizes(), dynDims, staticDims);
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
      auto uses = src.getUses();
      SmallVector<Operation *> allUsers;
      allUsers.reserve(std::distance(uses.begin(), uses.end()));
      std::transform(uses.begin(), uses.end(), std::back_inserter(allUsers),
                     [](OpOperand &use) { return use.getOwner(); });
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
}  // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createInsertLoadAndStorePass() { return std::make_unique<InsertLoadAndStore>(); }
