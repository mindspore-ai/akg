//===----------------------------------------------------------------------===//
//
// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseBatchMatMulToMul.h"

#include <optional>

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Utils/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEBATCHMATMULTOMUL
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

/// K dimension must be 1 for matmul->mul fusion (use OpConstants kDim1 for value 1).
constexpr int64_t kKDimensionSize = static_cast<int64_t>(kDim1);

/// Check if the k dimension is 1 for matmul operations.
/// For matmul A @ B: k is the contracting dimension (last-two-dims semantics).
/// BatchMatmul: (*B, N, C) @ (*B, C, M) -> (*B, N, M); C is at rank-1 for self, rank-2 for mat2.
/// With transpose_a: (*B, C, N); transpose_b: (*B, M, C). So k uses last-two indices.
/// Returns true if k=1 for both input and weight.
static bool isKDimensionOne(RankedTensorType inputType, RankedTensorType weightType, bool transposeA, bool transposeB) {
  // Ensure both types have at least rank 2
  if (inputType.getRank() < kDim2 || weightType.getRank() < kDim2) {
    return false;
  }
  // K is always one of the last two dimensions (batch matmul semantics)
  int64_t inputKDimIdx = transposeA ? (inputType.getRank() - 2) : (inputType.getRank() - 1);
  int64_t inputK = inputType.getShape()[inputKDimIdx];
  int64_t weightKDimIdx = transposeB ? (weightType.getRank() - 1) : (weightType.getRank() - 2);
  int64_t weightK = weightType.getShape()[weightKDimIdx];
  return inputK == kKDimensionSize && weightK == kKDimensionSize;
}

/// Check if the data types are supported for k=1 matmul->mul fusion.
/// Supports: all F32, or all F16/BF16 (mixed F16/BF16 is allowed).
static bool areDataTypesSupported(Type inputType, Type weightType, Type outputType) {
  auto inputRanked = dyn_cast<RankedTensorType>(inputType);
  auto weightRanked = dyn_cast<RankedTensorType>(weightType);
  auto outputRanked = dyn_cast<RankedTensorType>(outputType);

  if (!inputRanked || !weightRanked || !outputRanked) {
    return false;
  }

  auto inputElemType = dyn_cast<FloatType>(inputRanked.getElementType());
  auto weightElemType = dyn_cast<FloatType>(weightRanked.getElementType());
  auto outputElemType = dyn_cast<FloatType>(outputRanked.getElementType());

  if (!inputElemType || !weightElemType || !outputElemType) {
    return false;
  }

  // Check if all are Float32
  if (inputElemType.isF32() && weightElemType.isF32() && outputElemType.isF32()) {
    return true;
  }

  // Check if all are Float16 or BFloat16 (mixed F16/BF16 is allowed)
  bool allF16OrBF16 = (isa<Float16Type>(inputElemType) || isa<BFloat16Type>(inputElemType)) &&
                      (isa<Float16Type>(weightElemType) || isa<BFloat16Type>(weightElemType)) &&
                      (isa<Float16Type>(outputElemType) || isa<BFloat16Type>(outputElemType));

  return allF16OrBF16;
}

/// Compute product of shape dimensions; returns -1 if any dim is non-static or negative.
static int64_t getShapeElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    if (d < 0) return -1;
    count *= d;
  }
  return count;
}

/// Helper to create reshape for transposed input/weight in BatchMatMul.
/// With transpose, the contracting dimension K=1 is one of the last two dims.
/// We need [..., M, 1] for left (transposeA) and [..., 1, N] for right (transposeB)
/// so that mul broadcasts to [..., M, N]. M/N are inferred from the non-K dimension.
/// Returns std::nullopt if input/output element count would differ (invalid reshape).
static std::optional<Value> createReshapeForBatchMatMulTranspose(Value input, RankedTensorType inputType,
                                                                 bool isTransposeA, Location loc,
                                                                 PatternRewriter &rewriter) {
  auto shape = inputType.getShape();
  if (shape.size() < kDim2) {
    return std::make_optional(input);
  }
  const size_t rank = shape.size();
  const int64_t lastDim = shape[rank - 1];
  const int64_t secondLastDim = shape[rank - 2];
  const int64_t nonKDim = (lastDim == kKDimensionSize) ? secondLastDim : lastDim;

  SmallVector<int64_t> newShape;
  for (size_t i = 0; i < rank - kDim2; ++i) {
    newShape.push_back(shape[i]);
  }
  if (isTransposeA) {
    newShape.push_back(nonKDim);
    newShape.push_back(kKDimensionSize);
  } else {
    newShape.push_back(kKDimensionSize);
    newShape.push_back(nonKDim);
  }

  const int64_t inCount = getShapeElementCount(shape);
  const int64_t outCount = getShapeElementCount(newShape);
  if (inCount < 0 || outCount < 0 || inCount != outCount) {
    MLOG(DEBUG) << "FuseBatchMatMulToMul: skip reshape (element count mismatch) in=" << inCount << " out=" << outCount;
    return std::nullopt;
  }

  auto newInputType = RankedTensorType::get(newShape, inputType.getElementType());
  return rewriter.create<ReshapeOp>(loc, newInputType, input);
}

/// Helper to create a 2D reshape (e.g. for MatmulOp transpose case).
/// Returns std::nullopt if input/output element count would differ.
static std::optional<Value> createReshape2D(Value input, RankedTensorType inputType, int64_t dim0, int64_t dim1,
                                            Location loc, PatternRewriter &rewriter) {
  const int64_t inCount = getShapeElementCount(inputType.getShape());
  const int64_t outCount = (dim0 < 0 || dim1 < 0) ? -1 : (dim0 * dim1);
  if (inCount < 0 || outCount < 0 || inCount != outCount) {
    MLOG(DEBUG) << "FuseBatchMatMulToMul: skip 2D reshape (element count mismatch) in=" << inCount
                << " out=" << outCount;
    return std::nullopt;
  }
  SmallVector<int64_t> newShape = {dim0, dim1};
  auto newResultType = RankedTensorType::get(newShape, inputType.getElementType());
  return rewriter.create<ReshapeOp>(loc, newResultType, input);
}

/// Pattern to fuse BatchMatMul with k=1 to Mul operation
class FuseBatchMatMulToMulPattern : public OpRewritePattern<BatchMatmulOp> {
 public:
  using OpRewritePattern<BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchMatmulOp op, PatternRewriter &rewriter) const override {
    // Check if it's a static shape
    auto inputType = dyn_cast<RankedTensorType>(op.getSelf().getType());
    auto weightType = dyn_cast<RankedTensorType>(op.getMat2().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!inputType || !weightType || !outputType) {
      return failure();
    }

    // Check if all shapes are static
    if (!inputType.hasStaticShape() || !weightType.hasStaticShape() || !outputType.hasStaticShape()) {
      return failure();
    }

    // Check if k dimension is 1
    if (!isKDimensionOne(inputType, weightType, op.getTransposeA(), op.getTransposeB())) {
      return failure();
    }

    // Check if data types are supported
    if (!areDataTypesSupported(op.getSelf().getType(), op.getMat2().getType(), op.getResult().getType())) {
      return failure();
    }

    // Get inputs
    Value input = op.getSelf();
    Value weight = op.getMat2();

    if (op.getTransposeA()) {
      auto reshaped = createReshapeForBatchMatMulTranspose(input, inputType, true, op.getLoc(), rewriter);
      if (!reshaped.has_value()) {
        return failure();
      }
      input = *reshaped;
    }
    if (op.getTransposeB()) {
      auto reshaped = createReshapeForBatchMatMulTranspose(weight, weightType, false, op.getLoc(), rewriter);
      if (!reshaped.has_value()) {
        return failure();
      }
      weight = *reshaped;
    }

    Value mulResult = rewriter.create<MulOp>(op.getLoc(), outputType, input, weight);
    rewriter.replaceOp(op, mulResult);
    MLOG(DEBUG) << "FuseBatchMatMulToMul: replaced BatchMatmulOp@" << op.getLoc() << " with MulOp (k=1)";
    return success();
  }
};

/// Pattern to fuse MatmulOp with k=1 to Mul operation
class FuseMatmulToMulPattern : public OpRewritePattern<MatmulOp> {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    // Check if it's a static shape
    auto inputType = dyn_cast<RankedTensorType>(op.getSelf().getType());
    auto weightType = dyn_cast<RankedTensorType>(op.getOther().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!inputType || !weightType || !outputType) {
      return failure();
    }

    // Check if all shapes are static
    if (!inputType.hasStaticShape() || !weightType.hasStaticShape() || !outputType.hasStaticShape()) {
      return failure();
    }

    // Check if k dimension is 1
    if (!isKDimensionOne(inputType, weightType, op.getTransX1(), op.getTransX2())) {
      return failure();
    }

    // Check if data types are supported
    if (!areDataTypesSupported(op.getSelf().getType(), op.getOther().getType(), op.getResult().getType())) {
      return failure();
    }

    // Get inputs
    Value input = op.getSelf();
    Value weight = op.getOther();

    // Handle transpose by inserting reshape if needed.
    // With K=1, the contracting dim is 1; the other last-two dim is M (self) or N (other).
    // Works for both: permuted (1,3)/(4,1) from FuseMatmulTransposeWeight and unpermuted (3,1)/(1,4).
    if (op.getTransX1()) {
      auto inputShape = inputType.getShape();
      if (inputShape.size() < kDim2) {
        return failure();
      }
      int64_t m = (inputShape[0] == kKDimensionSize) ? inputShape[1] : inputShape[0];
      auto reshaped = createReshape2D(input, inputType, m, kKDimensionSize, op.getLoc(), rewriter);
      if (!reshaped.has_value()) {
        return failure();
      }
      input = *reshaped;
    }
    if (op.getTransX2()) {
      auto weightShape = weightType.getShape();
      if (weightShape.size() < kDim2) {
        return failure();
      }
      int64_t n = (weightShape[1] == kKDimensionSize) ? weightShape[0] : weightShape[1];
      auto reshaped = createReshape2D(weight, weightType, kKDimensionSize, n, op.getLoc(), rewriter);
      if (!reshaped.has_value()) {
        return failure();
      }
      weight = *reshaped;
    }

    // Create Mul operation
    Value mulResult = rewriter.create<MulOp>(op.getLoc(), outputType, input, weight);

    // Replace the original MatmulOp
    rewriter.replaceOp(op, mulResult);
    MLOG(DEBUG) << "FuseBatchMatMulToMul: replaced MatmulOp@" << op.getLoc() << " with MulOp (k=1)";
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseBatchMatMulToMul, FuseBatchMatMulToMulPattern, FuseMatmulToMulPattern)

}  // namespace mfuse

}  // namespace mlir
