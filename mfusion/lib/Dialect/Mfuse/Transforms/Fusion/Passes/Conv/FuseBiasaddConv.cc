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

#include <optional>
#include <tuple>

#include "mfusion/Dialect/Mfuse/Analysis/BinaryOpCommonInfer.h"
#include "mfusion/Dialect/Mfuse/Analysis/ConvBiasAddInfer.h"
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

/// Returns (convOut, convInput, convWeight) if \p v is produced by mfuse.aclnn.conv2d
/// (optionally through a single-operand UnrealizedConversionCast); else (null, null, null).
static std::tuple<Value, Value, Value> getConvOperands(Value v) {
  Value src = v;
  if (auto cast = v.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    if (cast.getOperands().size() == 1) src = cast.getOperand(0);
  Operation *def = src.getDefiningOp();
  if (!def) return {Value(), Value(), Value()};
  if (auto c = dyn_cast<AclnnConv2DOp>(def)) {
    return {src, c.getInput(), c.getWeight()};
  }
  return {Value(), Value(), Value()};
}

/// True when Add's conv-side operand exclusively consumes \p convOut (directly or via one cast).
static bool hasExclusiveConvUseChain(Value convOut, Value addConvOperand) {
  if (addConvOperand == convOut) {
    return convOut.hasOneUse();
  }
  auto cast = addConvOperand.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || cast.getOperands().size() != 1 || cast.getOperand(0) != convOut) {
    return false;
  }
  return cast->hasOneUse() && convOut.hasOneUse();
}

// Accept NCHW channel-broadcast reshape targets: [1,C,1,1] and [C,1,1].
static bool isCanonicalChannelBroadcastShape(RankedTensorType outTy, int64_t channels) {
  if (!outTy) {
    return false;
  }
  if (outTy.getRank() == 3) {
    return outTy.getDimSize(0) == channels && outTy.getDimSize(1) == 1 && outTy.getDimSize(2) == 1;
  }
  if (outTy.getRank() == 4) {
    return outTy.getDimSize(0) == 1 && outTy.getDimSize(1) == channels && outTy.getDimSize(2) == 1 &&
           outTy.getDimSize(3) == 1;
  }
  return false;
}

struct ConvBiasInfo {
  Value bias1d;
  Operation *reshapeToErase = nullptr;
};

/// Resolve conv bias to 1D [C]: direct rank-1 bias, or reshape(1D) -> canonical NCHW broadcast shape.
static std::optional<ConvBiasInfo> resolveConvBias(Value bias, int64_t outChannels) {
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || hasDynamicShape(bias.getType())) {
    return std::nullopt;
  }

  if (biasType.getRank() == static_cast<int64_t>(kDim1)) {
    if (biasType.getShape()[kIndex0] != outChannels) {
      return std::nullopt;
    }
    return ConvBiasInfo{bias, nullptr};
  }

  auto reshape = bias.getDefiningOp<ReshapeOp>();
  if (!reshape || !reshape->hasOneUse()) {
    return std::nullopt;
  }
  auto inputTy = dyn_cast<RankedTensorType>(reshape.getInput().getType());
  if (!inputTy || inputTy.getRank() != static_cast<int64_t>(kDim1) ||
      inputTy.getShape()[kIndex0] != outChannels) {
    return std::nullopt;
  }
  if (!isCanonicalChannelBroadcastShape(dyn_cast<RankedTensorType>(reshape.getResult().getType()), outChannels)) {
    return std::nullopt;
  }
  return ConvBiasInfo{reshape.getInput(), reshape.getOperation()};
}

static void eraseDeadReshape(PatternRewriter &rewriter, Operation *reshape) {
  if (reshape && reshape->use_empty()) {
    rewriter.eraseOp(reshape);
  }
}

/// Check Add's consumed value (and peeled static conv out) plus conv io are static ranked.
static bool hasStaticConvIoForFuse(Value addConvOperand, Value convOut, Value convInput, Value convWeight,
                                   RankedTensorType &convOutType) {
  convOutType = dyn_cast<RankedTensorType>(convOut.getType());
  if (!convOutType || hasDynamicShape(addConvOperand.getType()) || hasDynamicShape(convInput.getType()) ||
      hasDynamicShape(convWeight.getType())) {
    return false;
  }
  // When Add consumes a cast, also require the underlying conv result to be static.
  return addConvOperand == convOut || !hasDynamicShape(convOut.getType());
}

/// aten.convolution requires matching element types (no mixed-dtype Add promotion into fuse).
static bool hasMatchingConvBiasDtypes(RankedTensorType inputTy, RankedTensorType weightTy, RankedTensorType biasTy,
                                      RankedTensorType convOutType, RankedTensorType addResultTy) {
  if (!inputTy || !weightTy || !biasTy || !addResultTy) {
    return false;
  }
  Type elem = inputTy.getElementType();
  return weightTy.getElementType() == elem && biasTy.getElementType() == elem &&
         convOutType.getElementType() == elem && addResultTy.getElementType() == elem;
}

/// Shared fusion: Add(conv_out, bias) -> AclnnConv2DWithBias. \p opToReplace is the Add op.
static LogicalResult tryFuseBiasaddConv(Operation *opToReplace, Value lhs, Value rhs,
                                        PatternRewriter &rewriter) {
  auto [convOut, convInput, convWeight] = getConvOperands(lhs);
  Value bias = rhs;
  Value addConvOperand = lhs;
  if (!convInput) {
    auto t = getConvOperands(rhs);
    convOut = std::get<0>(t);
    convInput = std::get<1>(t);
    convWeight = std::get<2>(t);
    if (!convInput) {
      MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc()
                  << " - add operands are not (aclnn.conv2d, bias) or (bias, aclnn.conv2d)";
      return rewriter.notifyMatchFailure(opToReplace,
                                         "add operands are not (aclnn.conv2d, bias) or (bias, aclnn.conv2d)");
    }
    bias = lhs;
    addConvOperand = rhs;
  }

  // getConvOperands already verified the defining op is AclnnConv2DOp.
  auto conv2d = cast<AclnnConv2DOp>(convOut.getDefiningOp());
  RankedTensorType convOutType;
  if (!hasStaticConvIoForFuse(addConvOperand, convOut, convInput, convWeight, convOutType)) {
    MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc() << " - conv or io has dynamic shape";
    return rewriter.notifyMatchFailure(opToReplace, "conv or its io has dynamic shape");
  }
  if (convOutType.getRank() != static_cast<int64_t>(kDim4)) {
    MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc() << " - conv output rank is not 4";
    return rewriter.notifyMatchFailure(opToReplace, "conv output rank must be 4 (NCHW)");
  }
  const int64_t outChannels = convOutType.getShape()[kIndex1];

  std::optional<ConvBiasInfo> biasInfo = resolveConvBias(bias, outChannels);
  if (!biasInfo) {
    MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc()
                << " - unsupported conv bias shape (expected 1D [C] or reshape to [1,C,1,1]/[C,1,1])";
    return rewriter.notifyMatchFailure(
      opToReplace, "unsupported conv bias shape (expected 1D [C] or canonical channel broadcast reshape)");
  }
  Value actualBias = biasInfo->bias1d;

  // Bare [C] is ambiguous when N/H/W == C: trailing broadcast is also legal (often along W).
  // Canonical reshape [1,C,1,1]/[C,1,1] keeps channel semantics explicit and may still fuse.
  if (!biasInfo->reshapeToErase && ConvBiasAddInfer::hasAmbiguousChannelBiasShape(convOutType, outChannels)) {
    MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc()
                << " - ambiguous bare [C] bias when a non-channel dim equals C";
    return rewriter.notifyMatchFailure(
      opToReplace, "ambiguous bare [C] bias (N/H/W equals C); use [1,C,1,1] reshape or keep Add");
  }

  auto inputTy = dyn_cast<RankedTensorType>(convInput.getType());
  auto weightTy = dyn_cast<RankedTensorType>(convWeight.getType());
  auto biasTy = dyn_cast<RankedTensorType>(actualBias.getType());
  auto addResultTy = dyn_cast<RankedTensorType>(opToReplace->getResult(0).getType());
  if (!hasMatchingConvBiasDtypes(inputTy, weightTy, biasTy, convOutType, addResultTy)) {
    MLOG(DEBUG) << "FuseBiasaddConv: no match @" << opToReplace->getLoc()
                << " - bias/result/convOut dtype must match conv input/weight (no mixed-dtype promotion)";
    return rewriter.notifyMatchFailure(
      opToReplace, "bias, conv out, and add result dtype must match conv input/weight element type");
  }

  // Fuse only when this Add (via an optional single-use cast) is the exclusive consumer of conv.
  if (!hasExclusiveConvUseChain(convOut, addConvOperand)) {
    return rewriter.notifyMatchFailure(opToReplace,
                                       "conv must have an exclusive use chain to this Add (optional single cast)");
  }

  Location loc = opToReplace->getLoc();
  Value newConv = rewriter.create<AclnnConv2DWithBiasOp>(
    loc, addResultTy, convInput, convWeight, actualBias, conv2d.getStride(), conv2d.getPadding(),
    conv2d.getDilation(), conv2d.getTransposedAttr(), conv2d.getOutputPadding(), conv2d.getGroupsAttr());
  MLOG(DEBUG) << "FuseBiasaddConv: matched @" << loc << " -> AclnnConv2DWithBias";
  rewriter.replaceOp(opToReplace, newConv);
  eraseDeadReshape(rewriter, biasInfo->reshapeToErase);
  return success();
}

/// When fuse cannot eat a bare-[C] channel Add that is illegal under trailing broadcast
/// (typically from ConvBiasAddInfer decompose), rewrite to reshape([C]->[1,C,1,1]) + add
/// so downstream never sees a non-trailing-legal generic Add.
static LogicalResult tryLegalizeIllegalChannelBiasAdd(Operation *opToReplace, Value lhs, Value rhs,
                                                      PatternRewriter &rewriter) {
  auto [convOut, convInput, unusedWeight] = getConvOperands(lhs);
  (void)convOut;
  (void)unusedWeight;
  Value bias = rhs;
  Value addConvOperand = lhs;
  bool biasOnLeft = false;
  if (!convInput) {
    auto t = getConvOperands(rhs);
    convInput = std::get<1>(t);
    if (!convInput) {
      return rewriter.notifyMatchFailure(opToReplace, "legalize: not conv+bias add");
    }
    bias = lhs;
    addConvOperand = rhs;
    biasOnLeft = true;
  }

  // Already canonical reshape → trailing-legal; nothing to fix.
  if (bias.getDefiningOp<ReshapeOp>()) {
    return rewriter.notifyMatchFailure(opToReplace, "legalize: bias already reshape");
  }

  auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
  auto ndTy = dyn_cast<RankedTensorType>(addConvOperand.getType());
  auto resTy = dyn_cast<RankedTensorType>(opToReplace->getResult(0).getType());
  if (!biasTy || !ndTy || !resTy || biasTy.getRank() != 1 || ndTy.getRank() != 4) {
    return rewriter.notifyMatchFailure(opToReplace, "legalize: expected NCHW + 1D bias");
  }
  if (hasDynamicShape(bias.getType())) {
    return rewriter.notifyMatchFailure(opToReplace, "legalize: bias must be static 1D");
  }

  const int64_t channels = biasTy.getDimSize(0);
  if (channels == ShapedType::kDynamic || ndTy.getDimSize(1) != channels) {
    return rewriter.notifyMatchFailure(opToReplace, "legalize: bias length must equal channel dim");
  }
  if (resTy.getShape() != ndTy.getShape()) {
    return rewriter.notifyMatchFailure(opToReplace, "legalize: result shape must match conv-side operand");
  }

  // Trailing already legal (e.g. W==C) → keep as-is; do not force channel reshape.
  if (BinaryOpCommonInfer::inferTrailingBroadcastShape(ndTy.getShape(), biasTy.getShape())) {
    return rewriter.notifyMatchFailure(opToReplace, "legalize: trailing broadcast already legal");
  }
  if (ConvBiasAddInfer::hasAmbiguousChannelBiasShape(ndTy, channels)) {
    return rewriter.notifyMatchFailure(opToReplace, "legalize: ambiguous channel vs trailing");
  }

  Type elem = biasTy.getElementType();
  auto reshapeTy = RankedTensorType::get({1, channels, 1, 1}, elem);
  Location loc = opToReplace->getLoc();
  Value reshaped = rewriter.create<ReshapeOp>(loc, reshapeTy, bias);
  Value newAdd =
    biasOnLeft ? rewriter.create<AddOp>(loc, resTy, reshaped, addConvOperand)
               : rewriter.create<AddOp>(loc, resTy, addConvOperand, reshaped);
  MLOG(DEBUG) << "FuseBiasaddConv: legalize bare [C] channel add @" << loc
              << " -> reshape([1,C,1,1])+add (fusion missed)";
  rewriter.replaceOp(opToReplace, newAdd);
  return success();
}

/// Matches mfuse.add (e.g. after decompose lowered aclnn.add to add).
class FuseBiasaddConvPattern : public OpRewritePattern<AddOp> {
 public:
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
    MLOG(DEBUG) << "FuseBiasaddConv: trying AddOp@" << addOp.getLoc();
    if (succeeded(tryFuseBiasaddConv(addOp.getOperation(), addOp.getX(), addOp.getY(), rewriter))) {
      return success();
    }
    return tryLegalizeIllegalChannelBiasAdd(addOp.getOperation(), addOp.getX(), addOp.getY(), rewriter);
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseBiasaddConv, FuseBiasaddConvPattern)

}  // namespace mfuse

}  // namespace mlir
