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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseMatMulCast.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEMATMULCAST
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

// Aligns with bak/matmul_cast_fusion_pass.cc: matmul must have exactly one output (GetOutDataNodes().size() == 1).
// We enforce this via hasOneUse() on the matmul/matmul_with_bias result before fusing with the Cast.

/// Find any CastOp user of value that casts f16 to f32.
/// When single-use check is required, only that one user is the Cast; otherwise returns first match.
static CastOp getF16ToF32Cast(Value value) {
  auto inType = dyn_cast<RankedTensorType>(value.getType());
  if (!inType || !isa<Float16Type>(inType.getElementType())) {
    return nullptr;
  }
  for (Operation *user : value.getUsers()) {
    auto castOp = dyn_cast<CastOp>(user);
    if (!castOp) {
      continue;
    }
    auto castResultType = dyn_cast<RankedTensorType>(castOp.getResult().getType());
    if (!castResultType) {
      continue;
    }
    auto castDtype = dyn_cast_or_null<FloatType>(castResultType.getElementType());
    if (!castDtype || !castDtype.isF32()) {
      continue;
    }
    return castOp;
  }
  return CastOp();
}

/// Pattern to fuse MatMul followed by f16->f32 cast into MatMul with f32 output.
/// MatMul result must have exactly one user (the Cast); otherwise we do not fuse to avoid increasing matmul count.
/// This eliminates redundant cast operations by computing matmul directly in f32.
class FuseMatMulCastMatmulPattern : public OpRewritePattern<MatmulOp> {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp matmulOp, PatternRewriter &rewriter) const override {
    auto castOp = getF16ToF32Cast(matmulOp.getResult());
    if (!castOp) {
      return failure();
    }
    if (!matmulOp.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(matmulOp, "matmul must have exactly one user (the Cast) to fuse");
    }

    Type outType = castOp.getResult().getType();
    Value newMatmul = rewriter.create<MatmulOp>(matmulOp.getLoc(), outType, matmulOp.getSelf(), matmulOp.getOther(),
                                                matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());

    rewriter.replaceOp(castOp, newMatmul);
    MLOG(DEBUG) << "FuseMatMulCast: fused MatmulOp@" << matmulOp.getLoc() << " + CastOp@" << castOp.getLoc()
                << " (f16->f32) -> MatmulOp with f32 output";

    return success();
  }
};

/// Pattern to fuse MatMulWithBias followed by f16->f32 cast into MatMulWithBias with f32 output.
class FuseMatMulCastMatmulWithBiasPattern : public OpRewritePattern<MatmulWithBiasOp> {
 public:
  using OpRewritePattern<MatmulWithBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulWithBiasOp op, PatternRewriter &rewriter) const override {
    auto castOp = getF16ToF32Cast(op.getResult());
    if (!castOp) {
      return failure();
    }
    if (!op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(op, "matmul_with_bias must have exactly one user (the Cast) to fuse");
    }

    Type outType = castOp.getResult().getType();
    Value newMatmul = rewriter.create<MatmulWithBiasOp>(op.getLoc(), outType, op.getSelf(), op.getOther(), op.getBias(),
                                                        op.getTransX1Attr(), op.getTransX2Attr());

    rewriter.replaceOp(castOp, newMatmul);
    MLOG(DEBUG) << "FuseMatMulCast: fused MatmulWithBiasOp@" << op.getLoc() << " + CastOp@" << castOp.getLoc()
                << " (f16->f32) -> MatmulWithBiasOp with f32 output";

    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseMatMulCast, FuseMatMulCastMatmulPattern, FuseMatMulCastMatmulWithBiasPattern)

}  // namespace mfuse

}  // namespace mlir
