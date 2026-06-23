/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include "akg/Conversion/MindSporeFinalizingLower/MindSporeFinalizingLower.h"

#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "akg/Utils/SmallVectorSize.h"

namespace mlir {
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Conversion/Passes.h.inc"

#endif
}  // namespace mlir

namespace {
namespace arith = mlir::arith;
namespace func = mlir::func;
namespace linalg = mlir::linalg;
namespace linalgExt = mlir::linalgExt;
namespace math = mlir::math;
namespace mindspore = mlir::mindspore;
namespace shape = mlir::shape;
namespace tensor = mlir::tensor;
namespace tosa = mlir::tosa;
using mlir::applyFullConversion;
using mlir::ArrayRef;
using mlir::cast;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::DialectRegistry;
using mlir::dyn_cast;
using mlir::failed;
using mlir::FunctionOpInterface;
using mlir::kSmallVectorSizeFour;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MindSporeFinalizingLowerBase;
using mlir::OpConversionPattern;
using mlir::OperationPass;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::ValueRange;
class ConvertMindSporeConstOp : public OpRewritePattern<mindspore::ConstOp> {
 public:
  using OpRewritePattern<mindspore::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::ConstOp op, PatternRewriter &rewriter) const final {
    (void)rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

template <typename SourceOp>
class ConvertMindSporeSliceOp : public OpRewritePattern<SourceOp> {
 public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value input = op.getInput();
    assert(op->getNumOperands() == 1 && "only support static attr: start, strides, sizes");
    // get offsets(starts)
    ArrayRef<int64_t> starts;
    // get strides, default: (1, 1, 1...)
    ArrayRef<int64_t> strides(SmallVector<int64_t>(cast<ShapedType>(op.getType()).getRank(), 1));

    // slice sizes, consists of dynSizes and staticSizes
    // Note: strided_SliceOp only own staticSizes
    SmallVector<int64_t> staticSizes;
    SmallVector<Value, kSmallVectorSizeFour> dynSizes;
    if (mindspore::Strided_SliceOp strided_SliceOp = dyn_cast<mindspore::Strided_SliceOp>(op.getOperation())) {
      starts = *(strided_SliceOp.getStart());
      strides = *(strided_SliceOp.getStrides());
      for (const auto &i : llvm::enumerate(*strided_SliceOp.getEnd())) {
        int64_t end = i.value();
        size_t index = i.index();
        staticSizes.push_back((end - starts[index] + strides[index] - 1) / strides[index]);
      }
    } else if (mindspore::SliceOp sliceOp = dyn_cast<mindspore::SliceOp>(op.getOperation())) {
      starts = sliceOp.getBegin();
      for (const auto &i : llvm::enumerate(sliceOp.getSize())) {
        int64_t size = i.value();
        size_t index = i.index();
        // If size[i] is -1, all remaining elements in dimension i are included in the slice.
        // This is equivalent to setting size[i] = input.shape(i) − start[i].
        staticSizes.push_back(size == -1 ? ShapedType::kDynamic : size);
        if (!ShapedType::isDynamic(staticSizes.back())) {
          continue;
        }
        auto dim = rewriter.create<tensor::DimOp>(loc, input, index);
        auto offset = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(starts[index]));
        dynSizes.push_back(rewriter.create<arith::SubIOp>(loc, dim, offset));
      }
    }
    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
      op.getLoc(), op.getType(), input, ValueRange({}), dynSizes, ValueRange({}), rewriter.getDenseI64ArrayAttr(starts),
      rewriter.getDenseI64ArrayAttr(staticSizes), rewriter.getDenseI64ArrayAttr(strides));

    rewriter.replaceOp(op, newSliceOp.getResult());
    return success();
  }
};

class ConvertMindSporeReshapeOp : public OpConversionPattern<mindspore::ReshapeOp> {
 public:
  using OpConversionPattern<mindspore::ReshapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mindspore::ReshapeOp reshape, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    ShapedType resultTy = cast<ShapedType>(reshape.getType());
    if (adaptor.getNewShapeValue() != nullptr) {
      Value newReshape =
        rewriter.create<tensor::ReshapeOp>(reshape.getLoc(), resultTy, adaptor.getInput(), adaptor.getNewShapeValue());
      rewriter.replaceOp(reshape, newReshape);
      return success();
    }
    if (adaptor.getNewShapeAttr() != nullptr) {
      Value newReshape =
        rewriter.create<tosa::ReshapeOp>(reshape.getLoc(), resultTy, adaptor.getInput(), adaptor.getNewShapeAttr());
      rewriter.replaceOp(reshape, newReshape);
      return success();
    }
    return success();
  }
};
struct MindSporeFinalizingLowerPass : public MindSporeFinalizingLowerBase<MindSporeFinalizingLowerPass> {
 public:
  MindSporeFinalizingLowerPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<shape::ShapeDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    FunctionOpInterface func = getOperation();

    target.addLegalDialect<tosa::TosaDialect, arith::ArithDialect, linalg::LinalgDialect, linalgExt::LinalgExtDialect,
                           tensor::TensorDialect, func::FuncDialect, math::MathDialect, shape::ShapeDialect>();
    target.addIllegalDialect<mindspore::MindSporeDialect>();
    target.addLegalOp<mindspore::AddNOp>();
    // clang-format off
    (void)patterns.add<
      ConvertMindSporeSliceOp<mindspore::SliceOp>,
      ConvertMindSporeSliceOp<mindspore::Strided_SliceOp>
    >(patterns.getContext());
    // clang-format on
    mlir::populateMindSporeLowerPattern(patterns);

    if (failed(applyFullConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

namespace mlir {
void populateMindSporeLowerPattern(RewritePatternSet &patterns) {
  // clang-format off
  (void)patterns.add<
    ConvertMindSporeConstOp,
    ConvertMindSporeReshapeOp
  >(patterns.getContext());
  // clang-format on
  return;
}

std::unique_ptr<OperationPass<func::FuncOp>> createMindSporeFinalizingLowerPass() {
  return std::make_unique<MindSporeFinalizingLowerPass>();
}
}  // namespace mlir
