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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseAddRmsNorm.h"

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEADDRMSNORM
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

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

    MLOG(DEBUG) << "FuseAddRmsNormPattern matched AclnnRmsNormOp, fusing with AddOp";

    // Create AddRmsNormOp with 3 results: y_out, rstd_out, x_out
    SmallVector<Type, kOutputSize3> resultTypes;
    resultTypes.push_back(rmsNormOp.getYOut().getType());
    resultTypes.push_back(rmsNormOp.getRstdOut().getType());
    resultTypes.push_back(addOp.getResult().getType());

    // Must pass epsilon attribute
    auto addRmsNormOp = rewriter.create<AclnnAddRmsNormOp>(rmsNormOp.getLoc(), resultTypes, x1, x2, gamma, epsilon);
    MLOG(DEBUG) << "Created new AclnnAddRmsNormOp";

    // Replace results
    rewriter.replaceOp(rmsNormOp, {addRmsNormOp.getYOut(), addRmsNormOp.getRstdOut()});
    rewriter.replaceOp(addOp, addRmsNormOp.getXOut());
    MLOG(DEBUG) << "Replaced original AclnnRmsNormOp and AddOp with new AclnnAddRmsNormOp";
    return success();
  }
};

DEFINE_MFUSE_FUSION_PASS(FuseAddRmsNorm, FuseAddRmsNormPattern)

}  // namespace mfuse

}  // namespace mlir
