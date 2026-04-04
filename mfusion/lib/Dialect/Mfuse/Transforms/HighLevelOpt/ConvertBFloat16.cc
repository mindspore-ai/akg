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

#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTBFLOAT16PASS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

namespace {

// Check if the type is bfloat16 tensor (only tensor, not other types like bool)
bool isBF16Tensor(Type type) {
  // Defensive check: verify type is not null and is valid
  if (!type) {
    return false;
  }

  // Use MLIR's type casting infrastructure which handles edge cases
  // This is equivalent to llvm::dyn_cast but uses MLIR's type system
  if (llvm::isa<RankedTensorType>(type)) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(type);
    return tensorType.getElementType().isBF16();
  }
  return false;
}

// Get element type as f32 if input is bf16, otherwise return original type
Type getF32Type(Type type, MLIRContext *ctx) {
  if (auto rankedTensor = llvm::dyn_cast<RankedTensorType>(type)) {
    // Preserve the original encoding if present
    auto encoding = rankedTensor.getEncoding();
    return RankedTensorType::get(rankedTensor.getShape(), Float32Type::get(ctx), encoding);
  }
  return type;
}

// Convert f32 type to bf16 type with the same shape
Type getBF16Type(Type type, MLIRContext *ctx) {
  if (auto rankedTensor = llvm::dyn_cast<RankedTensorType>(type)) {
    auto shape = rankedTensor.getShape();
    auto elementType = rankedTensor.getElementType();
    auto encoding = rankedTensor.getEncoding();
    // If already bf16, return as is
    if (elementType.isBF16()) {
      return type;
    }
    // If f32, convert to bf16
    if (elementType.isF32()) {
      auto bf16Type = FloatType::getBF16(ctx);
      return RankedTensorType::get(shape, bf16Type, encoding);
    }
  }
  return type;
}

// Operations that pass through bf16 without conversion
bool canKeepBF16Op(Operation *op) { return llvm::isa<ReshapeOp, BroadcastToOp>(op); }

// Get the input indices that need to keep bf16 for this operation
SmallVector<size_t> getKeepBF16InputIndices(Operation *op) {
  return llvm::TypeSwitch<Operation *, SmallVector<size_t>>(op)
    .Case<MatmulOp, BatchMatmulOp, GroupedMatmulOp>([](auto) { return SmallVector<size_t>{0, 1}; })
    .Case<SliceOp, CastOp>([](auto) { return SmallVector<size_t>{0}; })
    .Default([](auto) { return SmallVector<size_t>{}; });
}

// Check if operation needs to keep bf16 (has bf16 inputs that should not be converted)
bool needKeepBF16Op(Operation *op) {
  auto indices = getKeepBF16InputIndices(op);
  return !indices.empty();
}

// Create a Cast operation and return the result
Value createCastOp(OpBuilder &builder, Location loc, Value input, Type dstType) {
  auto castOp = builder.create<CastOp>(loc, dstType, input);
  return castOp.getResult();
}

// Phase 1: Collect keepBF16 nodes (from outputs to inputs)
llvm::MapVector<Value, SmallVector<std::pair<Operation *, size_t>>> collectKeepBF16Nodes(
  Block *block, SmallVector<Operation *> &opsInOrder, Operation *yieldOp) {
  llvm::MapVector<Value, SmallVector<std::pair<Operation *, size_t>>> keepBF16Nodes;

  // Process Yield operation
  if (yieldOp) {
    for (size_t i = 0; i < yieldOp->getNumOperands(); ++i) {
      Value retInput = yieldOp->getOperand(i);
      if (retInput) {
        keepBF16Nodes[retInput].push_back({yieldOp, i});
      }
    }
  }

  // Process NeedKeepBF16 operations in reverse order
  for (auto it = opsInOrder.rbegin(); it != opsInOrder.rend(); ++it) {
    Operation *op = *it;
    if (!needKeepBF16Op(op)) {
      continue;
    }
    auto indices = getKeepBF16InputIndices(op);
    for (size_t idx : indices) {
      if (idx >= op->getNumOperands()) {
        op->emitError("keepBF16 index ") << idx << " out of bounds for operation with " << op->getNumOperands()
                                         << " operands";
      }
      Value input = op->getOperand(idx);
      if (input) {
        keepBF16Nodes[input].push_back({op, idx});
      }
    }
  }
  return keepBF16Nodes;
}

// Convert operation operands from bf16 to f32, create new operation
Operation *convertOpToF32(Operation *op, MLIRContext *ctx) {
  OpBuilder builder(op);
  SmallVector<Value, 4> newOperands;

  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    if (isBF16Tensor(operand.getType())) {
      auto f32Type = getF32Type(operand.getType(), ctx);
      auto castResult = createCastOp(builder, op->getLoc(), operand, f32Type);
      newOperands.push_back(castResult);
    } else {
      newOperands.push_back(operand);
    }
  }

  // Create new operation with converted operands
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(newOperands);
  llvm::for_each(op->getAttrs(), [&](auto &attr) { state.addAttribute(attr.getName(), attr.getValue()); });
  llvm::transform(op->getResults(), std::back_inserter(state.types), [&](auto result) {
    return isBF16Tensor(result.getType()) ? getF32Type(result.getType(), ctx) : result.getType();
  });

  return builder.create(state);
}

// Handle results after conversion: cast back to bf16 where needed
void handleConvertedResults(Operation *op, Operation *newOp,
                            llvm::MapVector<Value, SmallVector<std::pair<Operation *, size_t>>> &keepBF16Nodes,
                            MLIRContext *ctx) {
  OpBuilder builder(op);

  for (size_t i = 0; i < op->getNumResults(); ++i) {
    auto originalResult = op->getResult(i);
    auto newResult = newOp->getResult(i);

    auto iter = keepBF16Nodes.find(originalResult);
    if (iter != keepBF16Nodes.end() && isBF16Tensor(originalResult.getType())) {
      builder.setInsertionPointAfter(newOp);
      auto bf16Type = getBF16Type(originalResult.getType(), ctx);
      auto castBack = createCastOp(builder, op->getLoc(), newResult, bf16Type);
      for (auto &[userOp, idx] : iter->second) {
        if (idx < userOp->getNumOperands()) {
          userOp->setOperand(idx, castBack);
        }
      }
      // Also replace all other uses (not in keepBF16Nodes) to avoid dangling references
      originalResult.replaceAllUsesWith(castBack);
    } else if (isBF16Tensor(originalResult.getType())) {
      builder.setInsertionPointAfter(newOp);
      auto bf16Type = getBF16Type(originalResult.getType(), ctx);
      auto castBack = createCastOp(builder, op->getLoc(), newResult, bf16Type);
      originalResult.replaceAllUsesWith(castBack);
    } else {
      originalResult.replaceAllUsesWith(newResult);
    }
  }
}

// Phase 2: Process each operation (from outputs to inputs)
// Process in reverse order so that when an operation is deleted,
// all operations that use its results have already been processed
void processOpsInOrder(MLIRContext *ctx, const SmallVector<Operation *> &opsInOrder,
                       llvm::MapVector<Value, SmallVector<std::pair<Operation *, size_t>>> &keepBF16Nodes) {
  for (Operation *op : llvm::reverse(opsInOrder)) {
    if (needKeepBF16Op(op)) {
      continue;
    }

    bool hasBF16Operand =
      llvm::any_of(op->getOperands(), [](Value operand) { return isBF16Tensor(operand.getType()); });
    if (!hasBF16Operand) {
      continue;
    }

    bool canKeepBF16 = canKeepBF16Op(op);
    bool needUpdate = false;

    if (canKeepBF16) {
      needUpdate = llvm::any_of(op->getOperands(), [&](Value operand) {
        return isBF16Tensor(operand.getType()) && keepBF16Nodes.count(operand);
      });
    } else {
      needUpdate = true;
      auto *newOp = convertOpToF32(op, ctx);
      handleConvertedResults(op, newOp, keepBF16Nodes, ctx);
      op->erase();
      continue;
    }

    if (!needUpdate) {
      continue;
    }

    for (auto result : op->getResults()) {
      if (isBF16Tensor(result.getType())) {
        result.setType(getF32Type(result.getType(), ctx));
      }
    }
  }
}

// Process a FusedOp region to convert bf16 to f32
void processFusedOp(FusedOp fusedOp) {
  if (fusedOp.getRegion().empty()) {
    return;
  }

  MLIRContext *ctx = fusedOp.getContext();
  Block *block = &fusedOp.getRegion().front();

  SmallVector<Operation *> opsInOrder = llvm::to_vector(llvm::map_range(
    llvm::make_filter_range(block->getOperations(), [](Operation &op) { return !llvm::isa<YieldOp>(op); }),
    [](Operation &op) -> Operation * { return &op; }));

  auto yieldIt = llvm::find_if(block->getOperations(), [](Operation &op) { return llvm::isa<YieldOp>(op); });
  Operation *yieldOp = (yieldIt != block->getOperations().end()) ? &*yieldIt : nullptr;

  auto keepBF16Nodes = collectKeepBF16Nodes(block, opsInOrder, yieldOp);

  processOpsInOrder(ctx, opsInOrder, keepBF16Nodes);
}

// Process a function to convert bf16 to f32
void processFunction(func::FuncOp funcOp) {
  funcOp.walk([](FusedOp fusedOp) { processFusedOp(fusedOp); });
}

}  // namespace

struct ConvertBFloat16Pass : public impl::ConvertBFloat16PassBase<ConvertBFloat16Pass> {
  void runOnOperation() override {
    getOperation().walk([](func::FuncOp funcOp) { processFunction(funcOp); });
  }
};

std::unique_ptr<Pass> createConvertBFloat16Pass() { return std::make_unique<ConvertBFloat16Pass>(); }

}  // namespace mfuse

}  // namespace mlir
