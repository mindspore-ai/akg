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

#include <algorithm>

#include "mfusion/Dialect/Mfuse/Transforms/HighLevelOpt/Passes.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
enum CastType { CAST_UP, CAST_DOWN, CAST_OTHER };

// Check if a Cast operation increases or decreases precision
CastType GetCastType(mfuse::CastOp castOp) {
  auto inputType = castOp.getInput().getType();
  auto outputType = castOp.getType();

  auto inputShapedType = mlir::dyn_cast<ShapedType>(inputType);
  auto outputShapedType = mlir::dyn_cast<ShapedType>(outputType);
  if (!inputShapedType || !outputShapedType) {
    return CAST_OTHER;
  }

  auto inputElementType = mlir::dyn_cast<FloatType>(inputShapedType.getElementType());
  auto outputElementType = mlir::dyn_cast<FloatType>(outputShapedType.getElementType());

  if (!inputElementType || !outputElementType) {
    return CAST_OTHER;
  }

  // Use getWidth() to check float precision
  if (inputElementType.getWidth() < outputElementType.getWidth()) {
    return CAST_UP;
  }
  if (inputElementType.getWidth() > outputElementType.getWidth()) {
    return CAST_DOWN;
  }
  return CAST_OTHER;
}

// Check if all data inputs have the same element type
bool HasConsistentDataInputTypes(Operation *op, const SmallVector<int> &dataIndexes) {
  if (dataIndexes.size() < 2) {
    return true;
  }
  auto firstType = mlir::dyn_cast<ShapedType>(op->getOperand(dataIndexes[0]).getType());
  if (!firstType) {
    return true;
  }
  auto firstElementType = firstType.getElementType();

  for (size_t i = 1; i < dataIndexes.size(); ++i) {
    auto operandType = mlir::dyn_cast<ShapedType>(op->getOperand(dataIndexes[i]).getType());
    if (!operandType) {
      continue;
    }
    if (operandType.getElementType() != firstElementType) {
      return false;
    }
  }
  return true;
}

// Check if all Cast operations cast from the same input type
bool AllCastsHaveSameInputType(Operation *op, const SmallVector<int> &dataIndexes) {
  if (dataIndexes.size() < 2) {
    return true;
  }
  auto firstCastOp = llvm::cast<mfuse::CastOp>(op->getOperand(dataIndexes[0]).getDefiningOp());
  auto firstInputType = mlir::dyn_cast<ShapedType>(firstCastOp.getInput().getType()).getElementType();

  return llvm::all_of(llvm::make_range(dataIndexes.begin() + 1, dataIndexes.end()), [op, firstInputType](int idx) {
    auto castOp = llvm::cast<mfuse::CastOp>(op->getOperand(idx).getDefiningOp());
    auto inputType = mlir::dyn_cast<ShapedType>(castOp.getInput().getType()).getElementType();
    return inputType == firstInputType;
  });
}

// Check if all data inputs have the expected type (larger precision)
bool CheckDataInputsHaveExpectedType(Operation *op, const SmallVector<int> &dataIndexes, Type expectedType) {
  return llvm::all_of(llvm::make_range(dataIndexes.begin(), dataIndexes.end()), [op, expectedType](int idx) {
    auto operandType = mlir::dyn_cast<ShapedType>(op->getOperand(idx).getType());
    return operandType && operandType.getElementType() == expectedType;
  });
}

// Update result types of an operation to use a new element type
// (preserves shape, changes element type)
void UpdateResultElementType(Operation *op, Type newElementType) {
  llvm::for_each(op->getResults(), [&](auto result) {
    auto resultType = mlir::dyn_cast<ShapedType>(result.getType());
    if (resultType) {
      result.setType(RankedTensorType::get(resultType.getShape(), newElementType));
    }
  });
}

Type GetShapedElementType(Value v) {
  auto shapedType = mlir::dyn_cast<ShapedType>(v.getType());
  return shapedType ? shapedType.getElementType() : nullptr;
}

bool IsCastShapeUnchanged(mfuse::CastOp castOp) {
  auto inputType = mlir::dyn_cast<ShapedType>(castOp.getInput().getType());
  auto outputType = mlir::dyn_cast<ShapedType>(castOp.getType());
  return inputType && outputType && inputType.getShape() == outputType.getShape();
}

// Get data input indexes for type-insensitive ops
// Maximum/Minimum: 0, 1
// Select: 1, 2 (condition is index 0)
// Others: 0
SmallVector<int> GetOpDataInputIndexes(Operation *op) {
  return llvm::TypeSwitch<Operation *, SmallVector<int>>(op)
    .Case<mfuse::MaximumOp, mfuse::MinimumOp>([](auto) { return SmallVector<int>{0, 1}; })
    .Case<mfuse::SelectOp>([](auto) { return SmallVector<int>{1, 2}; })
    .Default([](auto) { return SmallVector<int>{0}; });
}

// Type-insensitive ops: mathematical operations that produce same results in different float precisions
// If you add a new type-insensitive op, add it here AND in the TypeSwitch above
bool IsTypeInsensitiveOp(Operation *op) {
  return llvm::isa<mfuse::ReshapeOp, mfuse::PermuteOp, mfuse::BroadcastToOp, mfuse::NegOp, mfuse::ReluOp,
                   mfuse::MaximumOp, mfuse::MinimumOp, mfuse::SelectOp>(op);
}

#define GEN_PASS_DEF_REORDEROPSPASS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

namespace {

/**
 * Pattern: TypeInsensitive -CastDown-> Cast
 * Transform to: CastDown -> TypeInsensitive
 *
 * Reorder Cast (decrease precision) after type-insensitive operations.
 */
struct ReorderCastAfterTypeInsensitiveOp : public OpRewritePattern<mfuse::CastOp> {
  using OpRewritePattern<mfuse::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::CastOp castOp, PatternRewriter &rewriter) const override {
    // Check if the Cast is CAST_DOWN (decrease precision)
    if (GetCastType(castOp) != CAST_DOWN) {
      return failure();
    }

    if (!IsCastShapeUnchanged(castOp)) {
      return failure();
    }

    // Check if the input of the Cast is a type-insensitive operation
    Operation *inputOp = castOp.getInput().getDefiningOp();
    if (!inputOp) return failure();

    // Check if the input operation is type-insensitive
    if (!IsTypeInsensitiveOp(inputOp)) {
      return failure();
    }

    // Check if the type-insensitive operation has only one use (the Cast node)
    if (!inputOp->hasOneUse()) {
      return failure();
    }

    // Check: Cast input shape should match type-insensitive op output shape
    auto inputOpOutputType = mlir::dyn_cast<ShapedType>(inputOp->getResult(0).getType());
    auto castInputType = mlir::dyn_cast<ShapedType>(castOp.getInput().getType());
    if (!inputOpOutputType || !castInputType || inputOpOutputType.getShape() != castInputType.getShape()) {
      return failure();
    }

    // Get data input indexes for the type-insensitive op
    auto dataIndexes = GetOpDataInputIndexes(inputOp);

    // Check that data inputs have consistent types
    if (!HasConsistentDataInputTypes(inputOp, dataIndexes)) {
      return failure();
    }

    // Check that data inputs have the expected type (larger precision)
    auto higherPrecisionType = GetShapedElementType(castOp.getInput());
    for (auto idx : dataIndexes) {
      if (idx >= static_cast<int>(inputOp->getNumOperands())) {
        return failure();
      }
      auto operandType = mlir::dyn_cast<ShapedType>(inputOp->getOperand(idx).getType());
      if (!operandType || operandType.getElementType() != higherPrecisionType) {
        return failure();
      }
    }

    // Create Cast operations for each data input (decrease precision)
    SmallVector<Value> newDataInputs;
    auto castOutputElementType = GetShapedElementType(castOp.getResult());
    for (auto idx : dataIndexes) {
      auto origInput = inputOp->getOperand(idx);
      auto origShapedType = mlir::dyn_cast<ShapedType>(origInput.getType());
      // Create new type with same shape as origInput but element type from cast output
      auto targetType = RankedTensorType::get(origShapedType.getShape(), castOutputElementType);
      auto newCastOp = rewriter.create<mfuse::CastOp>(castOp.getLoc(), targetType, origInput);
      newDataInputs.push_back(newCastOp.getResult());
    }

    // Clone the type-insensitive operation with new inputs
    Operation *newTypeInsensitiveOp = rewriter.clone(*inputOp);
    for (auto [i, idx] : llvm::enumerate(dataIndexes)) {
      newTypeInsensitiveOp->setOperand(idx, newDataInputs[i]);
    }

    // Update the output type to match the cast output element type
    UpdateResultElementType(newTypeInsensitiveOp, castOutputElementType);

    // Replace the original Cast operation with the result of type-insensitive op
    rewriter.replaceOp(castOp, newTypeInsensitiveOp->getResult(0));

    return success();
  }
};

/**
 * Pattern: CastUp -TypeInsensitive-> TypeInsensitive
 * Transform to: TypeInsensitive -CastUp-> Cast
 *
 * Reorder Cast (increase precision) before type-insensitive operations.
 */
struct ReorderCastBeforeTypeInsensitiveOp : public RewritePattern {
  explicit ReorderCastBeforeTypeInsensitiveOp(MLIRContext *context) : RewritePattern(MatchAnyOpTypeTag(), 0, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // Check if the operation is type-insensitive
    if (!IsTypeInsensitiveOp(op)) {
      return failure();
    }

    // Get data input indexes for the type-insensitive op
    auto dataIndexes = GetOpDataInputIndexes(op);

    // Check if all data inputs are Cast operations with CAST_UP type
    // Note: block arguments (function parameters) are not Cast ops, so fail
    for (auto idx : dataIndexes) {
      if (idx >= static_cast<int>(op->getNumOperands())) {
        return failure();
      }
      auto operand = op->getOperand(idx);
      auto *definingOp = operand.getDefiningOp();
      if (!definingOp) {
        return failure();
      }
      auto castOp = llvm::dyn_cast<mfuse::CastOp>(definingOp);
      if (!castOp || GetCastType(castOp) != CAST_UP) {
        return failure();
      }
      if (!IsCastShapeUnchanged(castOp)) {
        return failure();
      }
    }

    // Check that all Cast operations cast from the same input type (smaller precision)
    if (!AllCastsHaveSameInputType(op, dataIndexes)) {
      return failure();
    }

    // Get first Cast op to determine the expected higher precision type
    auto firstCastOp = llvm::cast<mfuse::CastOp>(op->getOperand(dataIndexes[0]).getDefiningOp());
    auto higherPrecisionType = GetShapedElementType(firstCastOp.getResult());
    if (!CheckDataInputsHaveExpectedType(op, dataIndexes, higherPrecisionType)) {
      return failure();
    }

    // Check that the type-insensitive op's outputs have consistent types with inputs
    if (!llvm::all_of(op->getResults(), [higherPrecisionType](auto result) {
          auto resultType = mlir::dyn_cast<ShapedType>(result.getType());
          return resultType && resultType.getElementType() == higherPrecisionType;
        })) {
      return failure();
    }

    // Create a new type-insensitive operation with original types (before Cast)
    SmallVector<Value> originalInputs = llvm::to_vector(llvm::map_range(dataIndexes, [op](int idx) -> Value {
      return llvm::cast<mfuse::CastOp>(op->getOperand(idx).getDefiningOp()).getInput();
    }));

    // Clone the type-insensitive operation with original inputs
    Operation *newTypeInsensitiveOp = rewriter.clone(*op);
    for (auto [i, idx] : llvm::enumerate(dataIndexes)) {
      newTypeInsensitiveOp->setOperand(idx, originalInputs[i]);
    }

    // Update the output type to match the cast input element type (smaller precision)
    auto castInputShapedType = mlir::dyn_cast<ShapedType>(firstCastOp.getInput().getType());
    auto castInputElementType = castInputShapedType.getElementType();
    UpdateResultElementType(newTypeInsensitiveOp, castInputElementType);

    // Create Cast operations after the type-insensitive operation
    SmallVector<Value> newResults =
      llvm::to_vector(llvm::map_range(llvm::seq<uint64_t>(0, op->getNumResults()), [&](uint64_t i) -> Value {
        return rewriter
          .create<mfuse::CastOp>(op->getLoc(), op->getResult(i).getType(), newTypeInsensitiveOp->getResult(i))
          .getResult();
      }));

    // Replace the original operation with the new results
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

}  // namespace

struct ReorderOpsPass : public impl::ReorderOpsPassBase<ReorderOpsPass> {
  void runOnOperation() override {
    getOperation().walk([](func::FuncOp func) {
      RewritePatternSet patterns(func.getContext());
      patterns.add<ReorderCastAfterTypeInsensitiveOp, ReorderCastBeforeTypeInsensitiveOp>(func.getContext());
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        func.emitError("ReorderOpsPass failed");
      }
    });
  }
};

}  // namespace mfuse

namespace mfuse {

std::unique_ptr<Pass> createReorderOpsPass() { return std::make_unique<ReorderOpsPass>(); }

}  // namespace mfuse

}  // namespace mlir
