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

#include "mfusion/Dialect/Muse/Transforms/Fusion/FuseAddRmsNorm.h"

#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEADDRMSNORM
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

namespace muse {
namespace {
// Check if Value is a constant with value 1.0
bool isScalarOne(Value value) {
  auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  auto attr = constOp.getValue();
  if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(attr)) {
    auto elementType = denseAttr.getElementType();
    if (elementType.isa<mlir::FloatType>()) {
      auto floatVal = denseAttr.getSplatValue<mlir::APFloat>();
      return floatVal.isExactlyValue(1.0);
    }
    if (elementType.isa<mlir::IntegerType>()) {
      auto intVal = denseAttr.getSplatValue<mlir::APInt>();
      return intVal.isOne();
    }
    return false;
  }

  return false;
}
}  // namespace

/**
 * @brief Fuse Add and RmsNorm into AddRmsNorm.
 * @example
 * %0 = Add(%x, %y, %alpha)  // %alpha is a constant with value 1.0
 * %1 = RmsNorm(%0, %gamma, %epsilon)
 * return %1
 * --->
 * %0 = AddRmsNorm(%x, %y, %gamma, %epsilon)
 * return %0
 */
class FuseAddRmsNormPattern : public OpRewritePattern<AclnnRmsNormOp> {
 public:
  using OpRewritePattern<AclnnRmsNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AclnnRmsNormOp rmsNormOp, PatternRewriter &rewriter) const override {
    // Get RmsNorm inputs
    Value x = rmsNormOp.getX();
    Value gamma = rmsNormOp.getGamma();
    FloatAttr epsilon = rmsNormOp.getEpsilonAttr();

    // Match only muse.aclnn.add for now
    auto addOp = x.getDefiningOp<AclnnAddOp>();
    if (!addOp || !isScalarOne(addOp.getAlpha())) {
      return failure();
    }

    Value x1 = addOp.getX();
    Value x2 = addOp.getY();
    // Check if shapes are consistent
    auto x1Type = dyn_cast<TensorType>(x1.getType());
    auto x2Type = dyn_cast<TensorType>(x2.getType());
    if (!x1Type || !x2Type || x1Type.getShape() != x2Type.getShape()) {
      return failure();
    }

    // Check if types are consistent
    auto xType = dyn_cast<TensorType>(x.getType());
    auto gammaType = dyn_cast<TensorType>(gamma.getType());
    if (!xType || !gammaType || xType.getElementType() != gammaType.getElementType()) {
      return failure();
    }

    // Create AddRmsNormOp
    SmallVector<Type, 3> resultTypes;
    resultTypes.push_back(rmsNormOp.getYOut().getType());
    resultTypes.push_back(rmsNormOp.getRstdOut().getType());
    resultTypes.push_back(addOp.getResult().getType());

    // Must pass epsilon attribute
    auto addRmsNormOp = rewriter.create<AclnnAddRmsNormOp>(rmsNormOp.getLoc(), resultTypes, x1, x2, gamma, epsilon);

    // Replace results
    rewriter.replaceOp(rmsNormOp, {addRmsNormOp.getYOut(), addRmsNormOp.getRstdOut()});
    rewriter.replaceOp(addOp, addRmsNormOp.getXOut());
    return success();
  }
};

struct FuseAddRmsNormPass : public impl::FuseAddRmsNormBase<FuseAddRmsNormPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseAddRmsNormPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace muse

std::unique_ptr<Pass> createFuseAddRmsNormPass() { return std::make_unique<muse::FuseAddRmsNormPass>(); }

}  // namespace mlir
