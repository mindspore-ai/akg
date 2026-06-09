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

#include "mfusion/Dialect/Mfuse/Transforms/HighLevelOpt/Passes.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_RAISEREDUCTIONPRECISIONPASS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

namespace {

/**
 * Pattern to raise precision of ReduceSum from float16 to float32.
 * This pattern matches ReduceSum operations with float16 inputs, converts them to float32,
 * performs the reduction in float32, and converts the result back to float16 if needed.
 */
struct RaiseReduceSumPrecision : public OpRewritePattern<mfuse::ReduceSumOp> {
  using OpRewritePattern<mfuse::ReduceSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mfuse::ReduceSumOp reduceOp, PatternRewriter &rewriter) const override {
    // Check if input is float16
    Type inputType = reduceOp.getInput().getType();
    auto inputTensorType = dyn_cast<RankedTensorType>(inputType);
    if (!inputTensorType) {
      return failure();
    }

    if (!inputTensorType.getElementType().isF16()) {
      return failure();
    }

    MLIRContext *ctx = reduceOp.getContext();
    Float32Type f32Type = Float32Type::get(ctx);

    // Create input type with f32 element type
    RankedTensorType f32InputType =
      RankedTensorType::get(inputTensorType.getShape(), f32Type, inputTensorType.getEncoding());

    // Compute output shape: reduce dimensions based on keepdim
    SmallVector<int64_t> outputShape(inputTensorType.getShape().begin(), inputTensorType.getShape().end());

    // Extract dimensions from ArrayAttr
    auto dimensionsAttr = reduceOp.getDimensions();
    SmallVector<int64_t> dims = llvm::to_vector(llvm::map_range(dimensionsAttr.getValue(), [](auto dimAttr) {
      return mlir::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
    }));

    bool keepdim = reduceOp.getKeepdim();

    if (keepdim) {
      llvm::for_each(dims, [&](auto dim) { outputShape[dim] = 1; });
    } else {
      // Remove reduced dimensions (in reverse order to maintain indices)
      SmallVector<int64_t> dimsSorted(dims);
      llvm::sort(dimsSorted, std::greater<int64_t>());
      for (auto dim : dimsSorted) {
        outputShape.erase(outputShape.begin() + dim);
      }
    }

    RankedTensorType f32OutputType = RankedTensorType::get(outputShape, f32Type, inputTensorType.getEncoding());

    // Create Cast from float16 to float32
    auto castToF32 = rewriter.create<mfuse::CastOp>(reduceOp.getLoc(), f32InputType, reduceOp.getInput());

    // Create new ReduceSum with float32 input and output
    auto newReduceOp = rewriter.create<mfuse::ReduceSumOp>(reduceOp.getLoc(), f32OutputType, castToF32.getResult(),
                                                           dimensionsAttr, BoolAttr::get(ctx, keepdim));

    // Create Cast back to float16 if needed
    Value result;
    Type originalOutputType = reduceOp.getType();
    if (originalOutputType != f32OutputType) {
      auto castBack = rewriter.create<mfuse::CastOp>(reduceOp.getLoc(), originalOutputType, newReduceOp.getResult());
      result = castBack.getResult();
    } else {
      result = newReduceOp.getResult();
    }

    // Replace original ReduceSum with new operations
    rewriter.replaceOp(reduceOp, result);

    return success();
  }
};

}  // namespace

struct RaiseReductionPrecisionPass : public impl::RaiseReductionPrecisionPassBase<RaiseReductionPrecisionPass> {
  void runOnOperation() override {
    getOperation().walk([this](func::FuncOp func) {
      RewritePatternSet patterns(func.getContext());
      patterns.add<RaiseReduceSumPrecision>(func.getContext());
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
      }
    });
  }
};

std::unique_ptr<Pass> createRaiseReductionPrecisionPass() { return std::make_unique<RaiseReductionPrecisionPass>(); }

}  // namespace mfuse

}  // namespace mlir
