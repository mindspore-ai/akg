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

#include "mfusion/Dialect/Muse/Transforms/Fusion/Passes/MatMul/FuseMatMulCast.h"

#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/Utils/ArithUtils.h"
#include "mfusion/Dialect/Muse/Transforms/Fusion/FusionPassMacros.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEMATMULCAST
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

namespace muse {
namespace {

/// Check if a value has a single CastOp user that casts f16 to f32.
/// Returns the CastOp if found, nullptr otherwise.
static CastOp getF16ToF32Cast(Value value) {
  if (!value.hasOneUse()) {
    return nullptr;
  }
  auto castOp = dyn_cast<CastOp>(*value.user_begin());
  if (!castOp) {
    return nullptr;
  }
  auto castDtype = dyn_cast_or_null<FloatType>(castOp.getDtype());
  if (!castDtype || !castDtype.isF32()) {
    return nullptr;
  }
  auto inType = dyn_cast<RankedTensorType>(value.getType());
  if (!inType || !isa<Float16Type>(inType.getElementType())) {
    return nullptr;
  }
  return castOp;
}

/// Pattern to fuse MatMul followed by f16->f32 cast into MatMul with f32 output.
/// This eliminates redundant cast operations by computing matmul directly in f32.
class FuseMatMulCastMatmulPattern : public OpRewritePattern<MatmulOp> {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp matmulOp, PatternRewriter &rewriter) const override {
    auto castOp = getF16ToF32Cast(matmulOp.getResult());
    if (!castOp) {
      return failure();
    }

    Type outType = castOp.getResult().getType();
    Value newMatmul = rewriter.create<MatmulOp>(matmulOp.getLoc(), outType, matmulOp.getSelf(), matmulOp.getOther(),
                                                matmulOp.getTransX1Attr(), matmulOp.getTransX2Attr());
    rewriter.replaceOp(castOp, newMatmul);
    rewriter.eraseOp(matmulOp);
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

    Type outType = castOp.getResult().getType();
    Value newMatmul = rewriter.create<MatmulWithBiasOp>(op.getLoc(), outType, op.getSelf(), op.getOther(), op.getBias(),
                                                        op.getTransX1Attr(), op.getTransX2Attr());
    rewriter.replaceOp(castOp, newMatmul);
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

DEFINE_MUSE_FUSION_PASS(FuseMatMulCast, FuseMatMulCastMatmulPattern, FuseMatMulCastMatmulWithBiasPattern)

}  // namespace muse

}  // namespace mlir
