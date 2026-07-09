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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatMulBiasAdd.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEMATMULBIASADD
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

/// Shared fusion: Add(matmul_out, bias) -> MatmulWithBias. \p addOp is the Add op.
static LogicalResult tryFuseMatMulBiasAdd(Operation *addOp, Value lhs, Value rhs,
                                          PatternRewriter &rewriter) {
  Value matmulOut;
  Value bias;
  MatmulOp matmulOp;

  if (auto m = lhs.getDefiningOp<MatmulOp>()) {
    matmulOut = lhs;
    bias = rhs;
    matmulOp = m;
  } else if (auto m = rhs.getDefiningOp<MatmulOp>()) {
    matmulOut = rhs;
    bias = lhs;
    matmulOp = m;
  } else {
    return rewriter.notifyMatchFailure(addOp, "add operands are not (matmul, bias)");
  }

  auto matmulOutType = dyn_cast<RankedTensorType>(matmulOut.getType());
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!matmulOutType || !biasType) {
    return rewriter.notifyMatchFailure(addOp, "matmul or bias has no ranked type");
  }
  if (hasDynamicShape(matmulOut.getType()) || hasDynamicShape(bias.getType())) {
    return rewriter.notifyMatchFailure(addOp, "matmul or bias has dynamic shape");
  }

  const int64_t matmulRank = matmulOutType.getRank();
  const int64_t biasRank = biasType.getRank();
  if (matmulRank < static_cast<int64_t>(kDim2)) {
    return rewriter.notifyMatchFailure(addOp, "matmul output rank must be at least 2");
  }
  if (biasRank != static_cast<int64_t>(kDim1)) {
    return rewriter.notifyMatchFailure(addOp, "bias must be rank 1");
  }
  const int64_t lastDimSize = matmulOutType.getShape()[matmulRank - 1];
  const int64_t biasSize = biasType.getShape()[kIndex0];
  if (biasSize != lastDimSize) {
    return rewriter.notifyMatchFailure(addOp, "bias size does not match matmul last dimension");
  }
  if (!matmulOut.hasOneUse()) {
    return rewriter.notifyMatchFailure(addOp, "matmul must have exactly one user (the Add) to fuse");
  }

  MLOG(DEBUG) << "FuseMatMulBiasAdd matched @" << addOp->getLoc() << " (MatMul + bias)";

  Location loc = addOp->getLoc();
  Type resultType = addOp->getResult(0).getType();
  Value newMatmulWithBias = rewriter.create<MatmulWithBiasOp>(
      loc, resultType, matmulOp.getSelf(), matmulOp.getOther(), bias,
      matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
  rewriter.replaceOp(addOp, newMatmulWithBias);
  return success();
}

class FuseMatMulBiasAddPattern : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    return tryFuseMatMulBiasAdd(addOp.getOperation(), addOp.getX(), addOp.getY(), rewriter);
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseMatMulBiasAdd, FuseMatMulBiasAddPattern)

}  // namespace mfuse
}  // namespace mlir
