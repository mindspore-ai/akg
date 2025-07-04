/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "akg/Conversion/LinalgExtLower/LinalgExtLower.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"

using namespace mlir;
using namespace mlir::linalgExt;

namespace mlir {
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Conversion/Passes.h.inc"
#endif
}  // namespace mlir

constexpr auto kVectorInitSize = 4;

class GatherOpConverter : public OpRewritePattern<linalgExt::GatherOp> {
 public:
  explicit GatherOpConverter(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(linalgExt::GatherOp op, PatternRewriter &rewriter) const override {
    assert(op.hasPureBufferSemantics() && "expected linalg op with buffer semantics");

    Value data = op.getOperands()[0];
    auto dataShape = data.getType().cast<ShapedType>().getShape();
    Value indices = op.getOperands()[1];
    auto indicesShape = indices.getType().cast<ShapedType>().getShape();
    auto axis = op.getAxis();

    SmallVector<int64_t, kVectorInitSize> lowerBounds(indicesShape.size(), 0);
    SmallVector<int64_t, kVectorInitSize> steps(indicesShape.size(), 1);
    SmallVector<int64_t, kVectorInitSize> lowerBounds2(axis, 0);
    SmallVector<int64_t, kVectorInitSize> upperBounds2(dataShape.begin(), dataShape.begin() + axis);
    SmallVector<int64_t, kVectorInitSize> steps2(axis, 1);
    SmallVector<int64_t, kVectorInitSize> lowerBounds3(dataShape.size() - axis - 1, 0);
    SmallVector<int64_t, kVectorInitSize> upperBounds3(dataShape.begin() + axis + 1, dataShape.end());
    SmallVector<int64_t, kVectorInitSize> steps3(dataShape.size() - axis - 1, 1);

    auto loc = op.getLoc();
    Value output = op.getOutput();
    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, indicesShape, steps,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs) {
        Value slice = nestedBuilder.create<memref::LoadOp>(nestedLoc, indices, ivs);
        Value sliceIdx = nestedBuilder.create<arith::IndexCastOp>(nestedLoc, nestedBuilder.getIndexType(), slice);

        affine::buildAffineLoopNest(
          rewriter, loc, lowerBounds2, upperBounds2, steps2,
          [&](OpBuilder &nestedBuilder2, Location nestedLoc2, ValueRange ivs2) {
            affine::buildAffineLoopNest(
              rewriter, loc, lowerBounds3, upperBounds3, steps3,
              [&](OpBuilder &nestedBuilder3, Location nestedLoc3, ValueRange ivs3) {
                SmallVector<Value, kVectorInitSize> dataIndices(ivs2);
                dataIndices.push_back(sliceIdx);
                dataIndices.append(SmallVector<Value, kVectorInitSize>(ivs3));
                SmallVector<Value, kVectorInitSize> outputIndices(ivs2);
                outputIndices.append(SmallVector<Value, kVectorInitSize>(ivs));
                outputIndices.append(SmallVector<Value, kVectorInitSize>(ivs3));
                Value val = nestedBuilder3.create<memref::LoadOp>(nestedLoc3, data, dataIndices);
                (void)nestedBuilder3.create<memref::StoreOp>(nestedLoc3, val, output, outputIndices);
              });
          });
      });
    rewriter.eraseOp(op);
    return success();
  }
};

class UnsortedSegmentSumOpConverter : public OpRewritePattern<linalgExt::UnsortedSegmentSumOp> {
 public:
  explicit UnsortedSegmentSumOpConverter(MLIRContext *context, int64_t vectorSize)
      : OpRewritePattern(context), veclen(vectorSize) {}

  LogicalResult matchAndRewrite(linalgExt::UnsortedSegmentSumOp op, PatternRewriter &rewriter) const override {
    assert(op.hasPureBufferSemantics() && "expected linalgExt op with buffer semantics");
    Value data = op.getOperands()[0];
    Value indices = op.getOperands()[1];
    auto dataShape = data.getType().cast<ShapedType>().getShape();
    auto indicesShape = indices.getType().cast<ShapedType>().getShape();

    SmallVector<int64_t, kVectorInitSize> lowerBounds(indicesShape.size(), 0);
    SmallVector<int64_t, kVectorInitSize> steps(indicesShape.size(), 1);

    auto subViewRank = dataShape.size() - indicesShape.size();
    SmallVector<int64_t, kVectorInitSize> lowerBounds2(subViewRank, 0);
    SmallVector<int64_t, kVectorInitSize> steps2(subViewRank - 1, 1);
    steps2.push_back(veclen);
    SmallVector<int64_t, kVectorInitSize> sliceShape(dataShape.begin() + indicesShape.size(), dataShape.end());

    auto loc = op.getLoc();
    affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, indicesShape, steps,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs) {
        Value slice = nestedBuilder.create<memref::LoadOp>(nestedLoc, indices, ivs);
        Value sliceIdx = nestedBuilder.create<arith::IndexCastOp>(nestedLoc, nestedBuilder.getIndexType(), slice);

        affine::buildAffineLoopNest(
          rewriter, nestedLoc, lowerBounds2, sliceShape, steps2,
          [&](OpBuilder &nestedBuilder2, Location nestedLoc2, ValueRange ivs2) {
            SmallVector<Value, kVectorInitSize> dataIndices(ivs);
            dataIndices.append(ivs2.begin(), ivs2.end());
            SmallVector<Value, kVectorInitSize> outputIndices({sliceIdx});
            outputIndices.append(ivs2.begin(), ivs2.end());

            // create vector mask
            auto minMap = AffineMap::get(2, 1,
                                         {nestedBuilder2.getAffineSymbolExpr(0),
                                          nestedBuilder2.getAffineDimExpr(0) - nestedBuilder2.getAffineDimExpr(1)},
                                         nestedBuilder2.getContext());

            Value step = rewriter.create<arith::ConstantIndexOp>(nestedLoc2, veclen);
            Value upperBound = rewriter.create<arith::ConstantIndexOp>(nestedLoc2, sliceShape.back());
            Value affineMin =
              rewriter.createOrFold<affine::AffineMinOp>(nestedLoc2, minMap, ValueRange{upperBound, ivs2.back(), step});
            auto maskTy = VectorType::get({veclen}, rewriter.getIntegerType(1));
            Value vectorMask = rewriter.create<vector::CreateMaskOp>(nestedLoc2, maskTy, ValueRange{affineMin});

            Type elementType = data.getType().cast<ShapedType>().getElementType();
            Value padding = nestedBuilder2.create<arith::ConstantOp>(nestedLoc2, elementType,
                                                                     nestedBuilder2.getZeroAttr(elementType));
            auto vectorType = VectorType::get({veclen}, elementType);

            auto dataRank = data.getType().cast<ShapedType>().getRank();

            Value output = op.getOutput();
            auto outputRank = output.getType().cast<ShapedType>().getRank();
            auto dataMap =
              AffineMap::get(dataRank, 0, {nestedBuilder2.getAffineDimExpr(dataRank - 1)}, nestedBuilder2.getContext());
            auto outputMap = AffineMap::get(outputRank, 0, {nestedBuilder2.getAffineDimExpr(outputRank - 1)},
                                            nestedBuilder2.getContext());

            Value increment = nestedBuilder2.create<vector::TransferReadOp>(nestedLoc2, vectorType, data, dataIndices,
                                                                            dataMap, padding, vectorMask, ArrayAttr());
            Value base = nestedBuilder2.create<vector::TransferReadOp>(nestedLoc2, vectorType, output, outputIndices,
                                                                       outputMap, padding, vectorMask, ArrayAttr());
            Value result;

            if (elementType.isIntOrIndex()) {
              result = nestedBuilder2.create<arith::AddIOp>(nestedLoc2, base, increment);
            } else {
              result = nestedBuilder2.create<arith::AddFOp>(nestedLoc2, base, increment);
            }

            (void)nestedBuilder2.create<vector::TransferWriteOp>(nestedLoc2, Type(), result, output, outputIndices,
                                                                 outputMap, vectorMask, ArrayAttr());
          });
      });

    rewriter.eraseOp(op);
    return success();
  }

 private:
  int64_t veclen;
};

struct LinalgExtLowerPass : public LinalgExtLowerBase<LinalgExtLowerPass> {
 public:
  LinalgExtLowerPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
        arith::ArithDialect,
        mlir::affine::AffineDialect,
        vector::VectorDialect,
        math::MathDialect,
        shape::ShapeDialect,
        func::FuncDialect,
        tensor::TensorDialect,
        memref::MemRefDialect
    >();
    // clang-format on
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(&getContext());

    FunctionOpInterface func = getOperation();

    (void)patterns.add<UnsortedSegmentSumOpConverter>(ctx, vectorSize);
    (void)patterns.add<GatherOpConverter>(ctx);

    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgExtLowerPass() {
  return std::make_unique<LinalgExtLowerPass>();
}
