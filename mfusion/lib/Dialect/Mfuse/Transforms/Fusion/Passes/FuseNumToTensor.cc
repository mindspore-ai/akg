/*
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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseNumToTensor.h"

#include <optional>
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"

namespace mlir {
#define GEN_PASS_DEF_FUSENUMTOTENSOR
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

// Helper function to convert MLIR type to PyTorch scalar type integer
static std::optional<int64_t> getTorchScalarTypeInt(mlir::Type type) {
  if (mlir::isa<mlir::NoneType>(type)) {
    return std::nullopt;
  }
  if (type.isSignlessInteger() && !type.isSignlessInteger(1)) {
    if (type.isSignlessInteger(8)) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Char);
    if (type.isSignlessInteger(16)) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Short);
    if (type.isSignlessInteger(32)) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Int);
    if (type.isSignlessInteger(64)) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Long);
    return std::nullopt;
  }
  if (type.isUnsignedInteger() && !type.isUnsignedInteger(8)) {
    return std::nullopt;
  }
  // For float types
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.getWidth() == 16) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Half);
    if (floatType.getWidth() == 32) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Float);
    if (floatType.getWidth() == 64) return static_cast<int64_t>(mlir::torch::torch_upstream::ScalarType::Double);
    return std::nullopt;
  }
  return std::nullopt;
}

/// Pattern to fuse NumToTensor followed by Cast into a single Full.
/// This eliminates redundant operations by creating a full tensor directly with the desired type and shape.
class FuseNumToTensorCastPattern : public OpRewritePattern<CastOp> {
 public:
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp castOp, PatternRewriter &rewriter) const override {
    // Check if the input to CastOp is a NumToTensorOp
    auto numToTensorOp = castOp.getInput().getDefiningOp<NumToTensorOp>();
    if (!numToTensorOp) {
      return failure();
    }

    // Get the value from NumToTensorOp
    Value inputValue = numToTensorOp.getValue();
    // Get the result type from CastOp
    Type resultType = castOp.getResult().getType();
    // Get dtype from resultType
    mlir::IntegerAttr dtypeAttr;
    if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(resultType)) {
      if (auto dtypeInt = getTorchScalarTypeInt(shapedType.getElementType())) {
        dtypeAttr = rewriter.getI64IntegerAttr(*dtypeInt);
      }
      // Create a new full tensor with the desired type and shape
      auto fullOp = rewriter.create<FullOp>(castOp.getLoc(), resultType, inputValue,
                                            /*dtype=*/dtypeAttr,
                                            /*layout=*/mlir::IntegerAttr(),
                                            /*device=*/rewriter.getStringAttr("npu"),
                                            /*pin_memory=*/mlir::BoolAttr());

      // Replace the CastOp with the new FullOp
      rewriter.replaceOp(castOp, fullOp.getResult());
      MLOG(DEBUG) << "FuseNumToTensorCast: fused NumToTensorOp + CastOp -> FullOp";

      return success();
    }
    return failure();
  }
};

/// Pattern to convert NumToTensor directly to Full.
/// This eliminates redundant operations by creating a full tensor directly.
class FuseNumToTensorPattern : public OpRewritePattern<NumToTensorOp> {
 public:
  using OpRewritePattern<NumToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NumToTensorOp numToTensorOp, PatternRewriter &rewriter) const override {
    // Get the value from NumToTensorOp
    Value inputValue = numToTensorOp.getValue();
    // Get the result type from NumToTensorOp
    Type resultType = numToTensorOp.getResult().getType();
    // Get dtype from resultType and create a constant tensor for fill_value
    mlir::IntegerAttr dtypeAttr;
    if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(resultType)) {
      if (auto dtypeInt = getTorchScalarTypeInt(shapedType.getElementType())) {
        dtypeAttr = rewriter.getI64IntegerAttr(*dtypeInt);
      }

      // Create a new full tensor with the desired type and shape
      auto fullOp = rewriter.create<FullOp>(numToTensorOp.getLoc(), resultType, inputValue,
                                            /*dtype=*/dtypeAttr,
                                            /*layout=*/mlir::IntegerAttr(),
                                            /*device=*/rewriter.getStringAttr("npu"),
                                            /*pin_memory=*/mlir::BoolAttr());

      // Replace the NumToTensorOp with the new FullOp
      rewriter.replaceOp(numToTensorOp, fullOp.getResult());
      MLOG(DEBUG) << "FuseNumToTensor: converted NumToTensorOp -> FullOp";

      return success();
    }
    return failure();
  }
};

DEFINE_MFUSE_FUSION_PASS(FuseNumToTensor, FuseNumToTensorCastPattern, FuseNumToTensorPattern)

}  // namespace mfuse
}  // namespace mlir
