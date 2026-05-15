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
///
/// Constraints (all enforced below):
/// 1. One of add's two inputs must have rank 1 (the bias).
/// 2. Bias size must equal the last dimension of MatMul/BatchMatMul output.
/// 3. MatMul (2D) output must have rank 2.
/// 4. No broadcast: only rank-1 bias with size = last dim is allowed.
/// 5. Precision: if pre-fusion Add was fp16, fused op may use fp32 (see pass description).
static LogicalResult tryFuseMatMulBiasAdd(Operation *addOp, Value lhs, Value rhs,
                                          PatternRewriter &rewriter) {
  Value matmulOut;
  Value bias;
  MatmulOp matmulOp;
  BatchMatmulOp batchMatmulOp;

  if (auto m = lhs.getDefiningOp<MatmulOp>()) {
    matmulOut = lhs;
    bias = rhs;
    matmulOp = m;
  } else if (auto m = rhs.getDefiningOp<MatmulOp>()) {
    matmulOut = rhs;
    bias = lhs;
    matmulOp = m;
  } else if (auto b = lhs.getDefiningOp<BatchMatmulOp>()) {
    matmulOut = lhs;
    bias = rhs;
    batchMatmulOp = b;
  } else if (auto b = rhs.getDefiningOp<BatchMatmulOp>()) {
    matmulOut = rhs;
    bias = lhs;
    batchMatmulOp = b;
  } else {
    return rewriter.notifyMatchFailure(addOp, "add operands are not (matmul/batch_matmul, bias)");
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
  // Constraint 1 & 4: one add input must be rank 1 (bias); no broadcast (only 1D bias allowed).
  if (biasRank != static_cast<int64_t>(kDim1)) {
    return rewriter.notifyMatchFailure(addOp, "bias must be rank 1");
  }
  // Constraint 2: bias size equals matmul output last dimension.
  const int64_t lastDimSize = matmulOutType.getShape()[matmulRank - 1];
  const int64_t biasSize = biasType.getShape()[kIndex0];
  if (biasSize != lastDimSize) {
    return rewriter.notifyMatchFailure(addOp, "bias size does not match matmul last dimension");
  }
  // Constraint 3: MatMul output must have rank 2.
  if (matmulOp && matmulRank != static_cast<int64_t>(kDim2)) {
    return rewriter.notifyMatchFailure(addOp, "2D MatMul output rank must be 2");
  }
  // Avoid increasing matmul count: fuse only when matmul/batch_matmul has exactly one user (this Add).
  if (!matmulOut.hasOneUse()) {
    return rewriter.notifyMatchFailure(addOp, "matmul/batch_matmul must have exactly one user (the Add) to fuse");
  }

  MLOG(DEBUG) << "FuseMatMulBiasAdd matched @" << addOp->getLoc() << " (MatMul/BatchMatmul + bias)";

  Location loc = addOp->getLoc();
  Type resultType = addOp->getResult(0).getType();
  Value newMatmulWithBias;
  if (matmulOp) {
    newMatmulWithBias = rewriter.create<MatmulWithBiasOp>(
        loc, resultType, matmulOp.getSelf(), matmulOp.getOther(), bias,
        matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
  } else {
    newMatmulWithBias = rewriter.create<MatmulWithBiasOp>(
        loc, resultType, batchMatmulOp.getSelf(), batchMatmulOp.getMat2(), bias,
        rewriter.getBoolAttr(batchMatmulOp.getTransposeA()),
        rewriter.getBoolAttr(batchMatmulOp.getTransposeB()));
  }
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
