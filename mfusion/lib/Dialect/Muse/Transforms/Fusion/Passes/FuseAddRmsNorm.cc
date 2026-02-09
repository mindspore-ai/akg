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

#include "mfusion/Dialect/Muse/Transforms/Fusion/Passes/FuseAddRmsNorm.h"

#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/Utils/OpConstants.h"
#include "mfusion/Dialect/Muse/Transforms/Fusion/FusionPassMacros.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEADDRMSNORM
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

namespace muse {

/**
 * @brief Fuse Add and RmsNorm into AddRmsNorm.
 * @example
 * %0 = Add(%x, %y)
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

    // Match only muse.add
    auto addOp = x.getDefiningOp<AddOp>();
    if (!addOp) {
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

    // Create AddRmsNormOp with 3 results: y_out, rstd_out, x_out
    SmallVector<Type, kOutputSize3> resultTypes;
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

DEFINE_MUSE_FUSION_PASS(FuseAddRmsNorm, FuseAddRmsNormPattern)

}  // namespace muse

}  // namespace mlir
