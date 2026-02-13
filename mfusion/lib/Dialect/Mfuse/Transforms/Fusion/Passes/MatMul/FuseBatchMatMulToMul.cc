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

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"
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

// Constants for matmul-to-mul fusion
constexpr int64_t kMinRankForMatmul = 2;  // Matmul requires at least rank 2
constexpr int64_t kKDimensionSize = 1;    // K dimension must be 1 for matmul->mul fusion

/// Check if the k dimension is 1 for matmul operations.
/// For matmul A @ B: k is the contracting dimension (last-two-dims semantics).
/// BatchMatmul: (*B, N, C) @ (*B, C, M) -> (*B, N, M); C is at rank-1 for self, rank-2 for mat2.
/// With transpose_a: (*B, C, N); transpose_b: (*B, M, C). So k uses last-two indices.
/// Returns true if k=1 for both input and weight.
static bool isKDimensionOne(RankedTensorType inputType, RankedTensorType weightType, bool transposeA, bool transposeB) {
  // Ensure both types have at least rank 2
  if (inputType.getRank() < kMinRankForMatmul || weightType.getRank() < kMinRankForMatmul) {
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

/// Helper to create reshape for transposed input/weight in BatchMatMul.
/// For transposeA: swaps last two dims, sets k=1.
/// For transposeB: swaps last two dims, sets k=1.
static Value createReshapeForBatchMatMulTranspose(
    Value input, RankedTensorType inputType, bool isTransposeA,
    Location loc, PatternRewriter &rewriter) {
  auto shape = inputType.getShape();
  if (shape.size() < kMinRankForMatmul) {
    return input;  // Cannot transpose if rank < kMinRankForMatmul
  }
  
  SmallVector<int64_t> newShape;
  // Keep batch dimensions (all except last 2)
  for (size_t i = 0; i < shape.size() - kMinRankForMatmul; ++i) {
    newShape.push_back(shape[i]);
  }
  
  if (isTransposeA) {
    // Swap: [..., M, K] -> [..., K, M] where K=1
    newShape.push_back(shape[shape.size() - 1]);  // K -> last
    newShape.push_back(kKDimensionSize);           // k dimension = 1
  } else {
    // transposeB: [..., K, N] -> [..., 1, N] where K=1
    newShape.push_back(kKDimensionSize);           // k dimension = 1
    newShape.push_back(shape[shape.size() - 1]);  // N -> last
  }
  
  auto newInputType = RankedTensorType::get(newShape, inputType.getElementType());
  auto shapeType = RankedTensorType::get(
      {static_cast<int64_t>(newShape.size())}, rewriter.getIntegerType(64));
  auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
  auto shapeTensor = rewriter.create<mlir::arith::ConstantOp>(loc, shapeType, shapeAttr);
  return rewriter.create<ReshapeOp>(loc, newInputType, input, shapeTensor);
}

/// Helper to create a 2D reshape (e.g. for MatmulOp transpose case).
static Value createReshape2D(Value input, RankedTensorType inputType,
                             int64_t dim0, int64_t dim1, Location loc,
                             PatternRewriter &rewriter) {
  SmallVector<int64_t> newShape = {dim0, dim1};
  auto newResultType = RankedTensorType::get(newShape, inputType.getElementType());
  auto shapeType = RankedTensorType::get(
      {static_cast<int64_t>(newShape.size())}, rewriter.getIntegerType(64));
  auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
  auto shapeTensor = rewriter.create<mlir::arith::ConstantOp>(loc, shapeType, shapeAttr);
  return rewriter.create<ReshapeOp>(loc, newResultType, input, shapeTensor);
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
    
    // Handle transpose by inserting reshape if needed
    if (op.getTransposeA()) {
      input = createReshapeForBatchMatMulTranspose(
          input, inputType, true, op.getLoc(), rewriter);
    }
    
    if (op.getTransposeB()) {
      weight = createReshapeForBatchMatMulTranspose(
          weight, weightType, false, op.getLoc(), rewriter);
    }
    
    // Create Mul operation
    Value mulResult = rewriter.create<MulOp>(op.getLoc(), outputType, input, weight);
    
    // Replace the original BatchMatmulOp
    rewriter.replaceOp(op, mulResult);
    MLOG(DEBUG) << "FuseBatchMatMulToMul: replaced BatchMatmulOp@" << op.getLoc()
                << " with MulOp (k=1)";
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

    // Handle transpose by inserting reshape if needed
    if (op.getTransX1()) {
      // Insert reshape for transposed input: [M, K] -> [M, 1] where K=1
      auto inputShape = inputType.getShape();
      if (inputShape.size() < kMinRankForMatmul) {
        return failure();  // Safety check
      }
      input = createReshape2D(input, inputType, inputShape[0], kKDimensionSize,
                             op.getLoc(), rewriter);
    }

    if (op.getTransX2()) {
      // Insert reshape for transposed weight: [K, N] -> [1, N] where K=1
      auto weightShape = weightType.getShape();
      if (weightShape.size() < kMinRankForMatmul) {
        return failure();  // Safety check
      }
      weight = createReshape2D(weight, weightType, kKDimensionSize, weightShape[1],
                              op.getLoc(), rewriter);
    }

    // Create Mul operation
    Value mulResult = rewriter.create<MulOp>(op.getLoc(), outputType, input, weight);

    // Replace the original MatmulOp
    rewriter.replaceOp(op, mulResult);
    MLOG(DEBUG) << "FuseBatchMatMulToMul: replaced MatmulOp@" << op.getLoc()
                << " with MulOp (k=1)";
    return success();
  }
};

} // namespace

DEFINE_MFUSE_FUSION_PASS(FuseBatchMatMulToMul, FuseBatchMatMulToMulPattern, FuseMatmulToMulPattern)

} // namespace mfuse

} // namespace mlir
