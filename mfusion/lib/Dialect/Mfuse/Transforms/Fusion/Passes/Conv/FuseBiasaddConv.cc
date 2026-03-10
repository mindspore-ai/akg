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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/Conv/FuseBiasaddConv.h"

#include <tuple>

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
#define GEN_PASS_DEF_FUSEBIASADDCONV
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

/// Returns (convOut, convInput, convWeight) if \p v is produced by mfuse.conv2d; else (null, null, null).
static std::tuple<Value, Value, Value> getConvOperands(Value v) {
  Value src = v;
  if (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    if (cast.getOperands().size() == 1) src = cast.getOperand(0);
  Operation *def = src.getDefiningOp();
  if (!def) return {Value(), Value(), Value()};
  if (auto c = dyn_cast<Conv2DOp>(def))
    return {src, c.getInput(), c.getWeight()};
  return {Value(), Value(), Value()};
}

/// Shared fusion: Add(conv_out, bias) -> Conv2DWithBias. \p opToReplace is the Add op.
static LogicalResult tryFuseBiasaddConv(Operation *opToReplace, Value lhs, Value rhs,
                                        PatternRewriter &rewriter) {
  auto [convOut, convInput, convWeight] = getConvOperands(lhs);
  Value bias = rhs;
  if (!convInput) {
    auto t = getConvOperands(rhs);
    convOut = std::get<0>(t);
    convInput = std::get<1>(t);
    convWeight = std::get<2>(t);
    bias = lhs;
  }
  if (convInput) {
    auto convOutType = dyn_cast<RankedTensorType>(convOut.getType());
    if (!convOutType || hasDynamicShape(convOut.getType()) || hasDynamicShape(convInput.getType()) ||
        hasDynamicShape(convWeight.getType())) {
      MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc() << " - conv or io has dynamic shape";
      return rewriter.notifyMatchFailure(opToReplace, "conv or its io has dynamic shape");
    }
    if (convOutType.getRank() <= static_cast<int64_t>(kIndex1)) {
      MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc() << " - conv output rank <= 1";
      return rewriter.notifyMatchFailure(opToReplace, "conv output rank <= 1");
    }
    const int64_t outChannels = convOutType.getShape()[kIndex1];

    Value actualBias;
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType) {
      MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc() << " - bias has no ranked type";
      return rewriter.notifyMatchFailure(opToReplace, "bias has no ranked tensor type");
    }
    if (biasType.getRank() == static_cast<int64_t>(kDim1)) {
      if (hasDynamicShape(bias.getType()) || biasType.getShape()[kIndex0] != outChannels) {
        MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc() << " - bias 1D shape mismatch";
        return rewriter.notifyMatchFailure(opToReplace, "bias 1D shape mismatch vs outChannels");
      }
      actualBias = bias;
    } else {
      MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc() << " - bias must be rank 1";
      return rewriter.notifyMatchFailure(opToReplace, "bias must be 1D [C]");
    }

    // Avoid increasing conv count: fuse only when conv has exactly one user (this Add).
    if (!convOut.hasOneUse()) {
      return rewriter.notifyMatchFailure(opToReplace, "conv must have exactly one user (the Add) to fuse");
    }

    Location loc = opToReplace->getLoc();
    Type resultType = opToReplace->getResult(0).getType();
    Value newConv = rewriter.create<Conv2DWithBiasOp>(loc, resultType, convInput, convWeight, actualBias);
    MLOG(DEBUG) << "FuseBiasaddConv: matched @" << loc << " -> Conv2DWithBias";
    rewriter.replaceOp(opToReplace, newConv);
    return success();
  }
  MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc()
              << " - add operands are not (conv2d, bias) or (bias, conv2d)";
  return rewriter.notifyMatchFailure(opToReplace, "add operands are not (conv2d, bias) or (bias, conv2d)");
}

/// Matches mfuse.add (e.g. after decompose lowered aclnn.add to add).
class FuseBiasaddConvPattern : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    MLOG(DEBUG) << "FuseBiasaddConv: trying AddOp@" << addOp.getLoc();
    return tryFuseBiasaddConv(addOp.getOperation(), addOp.getX(), addOp.getY(), rewriter);
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseBiasaddConv, FuseBiasaddConvPattern)

}  // namespace mfuse

}  // namespace mlir
