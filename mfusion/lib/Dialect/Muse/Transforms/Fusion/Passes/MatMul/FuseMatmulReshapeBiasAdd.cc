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

#include "mfusion/Dialect/Muse/Transforms/Fusion/Passes/MatMul/FuseMatmulReshapeBiasAdd.h"

#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/Utils/ArithUtils.h"
#include "mfusion/Dialect/Muse/Utils/OpConstants.h"
#include "mfusion/Dialect/Muse/Transforms/Fusion/FusionPassMacros.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEMATMULRESHAPEBIASADD
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

namespace muse {
namespace {

// Constants for dimension and alignment checks
constexpr int kBiasSizeInit = 1;  // Initial value for bias size calculation

/// Pattern to fuse MatMul -> Reshape -> Add into MatMulWithBias -> Reshape.
/// This eliminates the intermediate Add operation by incorporating bias into matmul.
class FuseMatmulReshapeBiasAddPattern : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    Value addLhs = addOp.getX();
    Value addRhs = addOp.getY();
    Value reshapeOut = nullptr;
    Value addBias = nullptr;

    bool lhsIsReshape = addLhs.getDefiningOp<ReshapeOp>() != nullptr;
    bool rhsIsReshape = addRhs.getDefiningOp<ReshapeOp>() != nullptr;

    // Exactly one input must be a Reshape output, and the other must not be a Reshape.
    if (lhsIsReshape && !rhsIsReshape) {
      reshapeOut = addLhs;
      addBias = addRhs;
    } else if (!lhsIsReshape && rhsIsReshape) {
      reshapeOut = addRhs;
      addBias = addLhs;
    } else {
      // Both are Reshape or neither is Reshape - not a valid pattern.
      return failure();
    }

    auto reshapeOp = reshapeOut.getDefiningOp<ReshapeOp>();
    Value matmulOut = reshapeOp.getInput();

    auto matmulType = dyn_cast<RankedTensorType>(matmulOut.getType());
    auto reshapeType = dyn_cast<RankedTensorType>(reshapeOut.getType());
    auto biasType = dyn_cast<RankedTensorType>(addBias.getType());
    if (!matmulType || !reshapeType || !biasType) {
      return failure();
    }

    if (matmulType.getRank() < kDim1 || reshapeType.getRank() < kDim1) {
      return failure();
    }
    int64_t lastMatmul = matmulType.getShape().back();
    SmallVector<int64_t> biasShape(biasType.getShape());
    int64_t biasSize = kBiasSizeInit;
    for (int64_t s : biasShape) {
      biasSize *= s;
    }
    // Bias must have the same size as matmul's last dimension (broadcast along last axis).
    if (biasSize != lastMatmul) {
      return failure();
    }

    Location loc = addOp.getLoc();
    Type newMatmulResultType = matmulType;
    Value fusedBias = addBias;
    Operation *matmulDef = matmulOut.getDefiningOp();

    if (auto matmulOp = dyn_cast<MatmulOp>(matmulDef)) {
      // MatmulOp -> Reshape -> Add => MatmulWithBiasOp -> Reshape
      Value newMatmul =
        rewriter.create<MatmulWithBiasOp>(loc, newMatmulResultType, matmulOp.getSelf(), matmulOp.getOther(), fusedBias,
                                          matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
      Value newReshape = rewriter.create<ReshapeOp>(reshapeOp.getLoc(), reshapeOp.getResult().getType(), newMatmul,
                                                    reshapeOp.getShape());
      rewriter.replaceOp(addOp, newReshape);
      rewriter.eraseOp(reshapeOp);
      rewriter.eraseOp(matmulOp);
      return success();
    }

    if (auto matmulWithBiasOp = dyn_cast<MatmulWithBiasOp>(matmulDef)) {
      // MatmulWithBiasOp -> Reshape -> Add => MatmulWithBiasOp(..., old_bias + add_bias) -> Reshape
      Value oldBias = matmulWithBiasOp.getBias();
      fusedBias = rewriter.create<AddOp>(loc, oldBias.getType(), oldBias, addBias);
      Value newMatmul = rewriter.create<MatmulWithBiasOp>(
        loc, newMatmulResultType, matmulWithBiasOp.getSelf(), matmulWithBiasOp.getOther(), fusedBias,
        matmulWithBiasOp.getTransX1Attr(), matmulWithBiasOp.getTransX2Attr());
      Value newReshape = rewriter.create<ReshapeOp>(reshapeOp.getLoc(), reshapeOp.getResult().getType(), newMatmul,
                                                    reshapeOp.getShape());
      rewriter.replaceOp(addOp, newReshape);
      rewriter.eraseOp(reshapeOp);
      rewriter.eraseOp(matmulWithBiasOp);
      return success();
    }

    return failure();
  }
};

}  // namespace

DEFINE_MUSE_FUSION_PASS(FuseMatmulReshapeBiasAdd, FuseMatmulReshapeBiasAddPattern)

}  // namespace muse

}  // namespace mlir
