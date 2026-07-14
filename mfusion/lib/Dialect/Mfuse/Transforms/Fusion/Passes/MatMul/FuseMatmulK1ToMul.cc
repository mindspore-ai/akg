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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatmulK1ToMul.h"

#include <optional>

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEMATMULK1TOMUL
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

constexpr int64_t kKDimensionSize = static_cast<int64_t>(kDim1);

static bool isKDimensionOne(RankedTensorType inputType, RankedTensorType weightType, bool transposeA, bool transposeB) {
  if (static_cast<size_t>(inputType.getRank()) < kDim2 || static_cast<size_t>(weightType.getRank()) < kDim2) {
    return false;
  }
  int64_t inputKDimIdx = transposeA ? (inputType.getRank() - 2) : (inputType.getRank() - 1);
  int64_t inputK = inputType.getShape()[inputKDimIdx];
  int64_t weightKDimIdx = transposeB ? (weightType.getRank() - 1) : (weightType.getRank() - 2);
  int64_t weightK = weightType.getShape()[weightKDimIdx];
  return inputK == kKDimensionSize && weightK == kKDimensionSize;
}

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

  if (inputElemType.isF32() && weightElemType.isF32() && outputElemType.isF32()) {
    return true;
  }

  return (isa<Float16Type>(inputElemType) || isa<BFloat16Type>(inputElemType)) &&
         (isa<Float16Type>(weightElemType) || isa<BFloat16Type>(weightElemType)) &&
         (isa<Float16Type>(outputElemType) || isa<BFloat16Type>(outputElemType));
}

static int64_t getShapeElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    if (d < 0) return -1;
    count *= d;
  }
  return count;
}

static std::optional<Value> createReshapeForMatmulK1Transpose(Value input, RankedTensorType inputType,
                                                              bool isLeftOperand, Location loc,
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
  if (isLeftOperand) {
    newShape.push_back(nonKDim);
    newShape.push_back(kKDimensionSize);
  } else {
    newShape.push_back(kKDimensionSize);
    newShape.push_back(nonKDim);
  }

  const int64_t inCount = getShapeElementCount(shape);
  const int64_t outCount = getShapeElementCount(newShape);
  if (inCount < 0 || outCount < 0 || inCount != outCount) {
    MLOG(DEBUG) << "FuseMatmulK1ToMul: skip reshape (element count mismatch) in=" << inCount << " out=" << outCount;
    return std::nullopt;
  }

  auto newInputType = RankedTensorType::get(newShape, inputType.getElementType());
  return rewriter.create<ReshapeOp>(loc, newInputType, input);
}

static std::optional<Value> createReshape2D(Value input, RankedTensorType inputType, int64_t dim0, int64_t dim1,
                                            Location loc, PatternRewriter &rewriter) {
  const int64_t inCount = getShapeElementCount(inputType.getShape());
  const int64_t outCount = (dim0 < 0 || dim1 < 0) ? -1 : (dim0 * dim1);
  if (inCount < 0 || outCount < 0 || inCount != outCount) {
    MLOG(DEBUG) << "FuseMatmulK1ToMul: skip 2D reshape (element count mismatch) in=" << inCount
                << " out=" << outCount;
    return std::nullopt;
  }
  SmallVector<int64_t> newShape = {dim0, dim1};
  auto newResultType = RankedTensorType::get(newShape, inputType.getElementType());
  return rewriter.create<ReshapeOp>(loc, newResultType, input);
}

class FuseMatmulK1ToMulPattern : public OpRewritePattern<MatmulOp> {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getSelf().getType());
    auto weightType = dyn_cast<RankedTensorType>(op.getOther().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!inputType || !weightType || !outputType) {
      return failure();
    }

    if (!inputType.hasStaticShape() || !weightType.hasStaticShape() || !outputType.hasStaticShape()) {
      return failure();
    }

    if (!isKDimensionOne(inputType, weightType, op.getTransX1(), op.getTransX2())) {
      return failure();
    }

    if (!areDataTypesSupported(op.getSelf().getType(), op.getOther().getType(), op.getResult().getType())) {
      return failure();
    }

    Value input = op.getSelf();
    Value weight = op.getOther();

    if (op.getTransX1()) {
      if (inputType.getRank() > static_cast<int64_t>(kDim2)) {
        auto reshaped = createReshapeForMatmulK1Transpose(input, inputType, true, op.getLoc(), rewriter);
        if (!reshaped.has_value()) {
          return failure();
        }
        input = *reshaped;
      } else {
        auto inputShape = inputType.getShape();
        int64_t m = (inputShape[0] == kKDimensionSize) ? inputShape[1] : inputShape[0];
        auto reshaped = createReshape2D(input, inputType, m, kKDimensionSize, op.getLoc(), rewriter);
        if (!reshaped.has_value()) {
          return failure();
        }
        input = *reshaped;
      }
    }
    if (op.getTransX2()) {
      if (weightType.getRank() > static_cast<int64_t>(kDim2)) {
        auto reshaped = createReshapeForMatmulK1Transpose(weight, weightType, false, op.getLoc(), rewriter);
        if (!reshaped.has_value()) {
          return failure();
        }
        weight = *reshaped;
      } else {
        auto weightShape = weightType.getShape();
        int64_t n = (weightShape[1] == kKDimensionSize) ? weightShape[0] : weightShape[1];
        auto reshaped = createReshape2D(weight, weightType, kKDimensionSize, n, op.getLoc(), rewriter);
        if (!reshaped.has_value()) {
          return failure();
        }
        weight = *reshaped;
      }
    }

    auto loc = op.getLoc();
    Value mulResult = rewriter.create<MulOp>(loc, outputType, input, weight);
    rewriter.replaceOp(op, mulResult);
    MLOG(DEBUG) << "FuseMatmulK1ToMul: replaced MatmulOp@" << loc << " with MulOp (k=1)";
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseMatmulK1ToMul, FuseMatmulK1ToMulPattern)

}  // namespace mfuse

}  // namespace mlir
