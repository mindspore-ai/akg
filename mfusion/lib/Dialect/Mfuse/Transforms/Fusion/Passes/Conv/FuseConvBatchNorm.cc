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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/Conv/FuseConvBatchNorm.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSECONVBATCHNORM
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

static bool readRankedF32Constant(Value v, SmallVectorImpl<float> &out, RankedTensorType &typeOut) {
  auto cst = v.getDefiningOp<ConstantOp>();
  if (!cst) {
    return false;
  }
  auto type = dyn_cast<RankedTensorType>(cst.getResult().getType());
  if (!type || !type.getElementType().isF32()) {
    return false;
  }
  auto dense = dyn_cast<DenseFPElementsAttr>(cst.getValue());
  if (!dense) {
    return false;
  }
  out.clear();
  out.reserve(dense.getNumElements());
  std::transform(dense.getValues<APFloat>().begin(), dense.getValues<APFloat>().end(), std::back_inserter(out),
                 [](const APFloat &elem) { return elem.convertToFloat(); });
  typeOut = type;
  return true;
}

static std::tuple<Value, Value, Value, Value, Operation *> getConvData(Value bnInput) {
  if (auto conv = bnInput.getDefiningOp<AclnnConv2DOp>()) {
    return {bnInput, conv.getInput(), conv.getWeight(), Value(), conv};
  }
  if (auto conv = bnInput.getDefiningOp<AclnnConv2DWithBiasOp>()) {
    return {bnInput, conv.getInput(), conv.getWeight(), conv.getBias(), conv};
  }
  return {Value(), Value(), Value(), Value(), nullptr};
}

static LogicalResult checkConvBnPreconditions(AclnnBatchNormOp bn, Value convOut, Operation *convTemplateOp,
                                              PatternRewriter &rewriter) {
  if (!convOut || !convTemplateOp) {
    return rewriter.notifyMatchFailure(bn, "aclnn.batch_norm input is not aclnn.conv2d/aclnn.conv2d_with_bias");
  }
  if (bn.getTraining()) {
    return rewriter.notifyMatchFailure(bn, "only inference aclnn.batch_norm is supported for conv+bn folding");
  }
  if (!convOut.hasOneUse()) {
    return rewriter.notifyMatchFailure(bn, "conv output has multiple users");
  }
  if (auto conv2d = dyn_cast<AclnnConv2DOp>(convTemplateOp)) {
    if (conv2d.getTransposed()) {
      return rewriter.notifyMatchFailure(bn, "transposed aclnn.conv2d is not supported in conv+bn folding");
    }
    return success();
  }
  auto convWb = cast<AclnnConv2DWithBiasOp>(convTemplateOp);
  if (convWb.getTransposed()) {
    return rewriter.notifyMatchFailure(bn, "transposed aclnn.conv2d_with_bias is not supported in conv+bn folding");
  }
  return success();
}

static LogicalResult validateBnRank1Shapes(AclnnBatchNormOp bn, RankedTensorType meanTy, RankedTensorType varTy,
                                           RankedTensorType gammaTy, RankedTensorType betaTy,
                                           PatternRewriter &rewriter) {
  if (meanTy.getRank() != 1 || varTy.getRank() != 1 || gammaTy.getRank() != 1 || betaTy.getRank() != 1) {
    return rewriter.notifyMatchFailure(bn, "BN params must be rank-1");
  }
  return success();
}

static LogicalResult validateBnChannelConsistency(AclnnBatchNormOp bn, int64_t channels, RankedTensorType meanTy,
                                                  RankedTensorType varTy, RankedTensorType gammaTy,
                                                  RankedTensorType betaTy, PatternRewriter &rewriter) {
  if (channels <= 0 || meanTy.getShape()[0] != channels || varTy.getShape()[0] != channels ||
      gammaTy.getShape()[0] != channels || betaTy.getShape()[0] != channels) {
    return rewriter.notifyMatchFailure(bn, "channel size mismatch between conv weight and BN params");
  }
  return success();
}

static LogicalResult loadConvBiasValues(AclnnBatchNormOp bn, Value convBias, int64_t channels,
                                        SmallVectorImpl<float> &oldBiasVals, PatternRewriter &rewriter) {
  if (!convBias) {
    oldBiasVals.assign(static_cast<size_t>(channels), 0.0f);
    return success();
  }
  RankedTensorType biasTy;
  if (!readRankedF32Constant(convBias, oldBiasVals, biasTy) || biasTy.getRank() != 1 ||
      biasTy.getShape()[0] != channels) {
    return rewriter.notifyMatchFailure(bn, "conv bias must be rank-1 f32 constant");
  }
  return success();
}

static LogicalResult loadConvBnConstants(AclnnBatchNormOp bn, Value convWeight, Value convBias,
                                         SmallVectorImpl<float> &weightVals, RankedTensorType &weightTy,
                                         SmallVectorImpl<float> &meanVals, SmallVectorImpl<float> &varVals,
                                         SmallVectorImpl<float> &gammaVals, SmallVectorImpl<float> &betaVals,
                                         SmallVectorImpl<float> &oldBiasVals, int64_t &channels,
                                         PatternRewriter &rewriter) {
  RankedTensorType meanTy;
  RankedTensorType varTy;
  RankedTensorType gammaTy;
  RankedTensorType betaTy;
  if (!readRankedF32Constant(convWeight, weightVals, weightTy) || weightTy.getRank() != 4) {
    return rewriter.notifyMatchFailure(bn, "conv weight must be rank-4 f32 constant");
  }
  channels = weightTy.getShape()[0];
  if (!readRankedF32Constant(bn.getRunningMean(), meanVals, meanTy) ||
      !readRankedF32Constant(bn.getRunningVar(), varVals, varTy) ||
      !readRankedF32Constant(bn.getWeight(), gammaVals, gammaTy) ||
      !readRankedF32Constant(bn.getBias(), betaVals, betaTy)) {
    return rewriter.notifyMatchFailure(bn, "BN params must be f32 constants");
  }
  if (failed(validateBnRank1Shapes(bn, meanTy, varTy, gammaTy, betaTy, rewriter))) {
    return failure();
  }
  if (failed(validateBnChannelConsistency(bn, channels, meanTy, varTy, gammaTy, betaTy, rewriter))) {
    return failure();
  }
  if (weightVals.size() % static_cast<size_t>(channels) != 0) {
    return rewriter.notifyMatchFailure(bn, "invalid conv weight element count");
  }
  if (failed(loadConvBiasValues(bn, convBias, channels, oldBiasVals, rewriter))) {
    return failure();
  }
  return success();
}

static void foldConvBnWeightsAndBias(ArrayRef<float> weightVals, ArrayRef<float> meanVals, ArrayRef<float> varVals,
                                     ArrayRef<float> gammaVals, ArrayRef<float> betaVals, ArrayRef<float> oldBiasVals,
                                     int64_t channels, float eps, std::vector<float> &fusedWeight,
                                     std::vector<float> &fusedBias) {
  const size_t perChannelKernel = weightVals.size() / static_cast<size_t>(channels);
  fusedWeight.assign(weightVals.begin(), weightVals.end());
  fusedBias.assign(static_cast<size_t>(channels), 0.0f);
  for (int64_t c = 0; c < channels; ++c) {
    const float invStd = 1.0f / std::sqrt(varVals[static_cast<size_t>(c)] + eps);
    const float scale = gammaVals[static_cast<size_t>(c)] * invStd;
    fusedBias[static_cast<size_t>(c)] =
      betaVals[static_cast<size_t>(c)] +
      (oldBiasVals[static_cast<size_t>(c)] - meanVals[static_cast<size_t>(c)]) * scale;
    for (size_t i = 0; i < perChannelKernel; ++i) {
      const size_t idx = static_cast<size_t>(c) * perChannelKernel + i;
      fusedWeight[idx] = weightVals[idx] * scale;
    }
  }
}

static AclnnConv2DWithBiasOp createFusedConvWithBias(AclnnBatchNormOp bn, Value convInput, Value fusedWeight,
                                                   Value fusedBias, Operation *convTemplateOp,
                                                   PatternRewriter &rewriter) {
  if (auto conv2d = dyn_cast<AclnnConv2DOp>(convTemplateOp)) {
    return rewriter.create<AclnnConv2DWithBiasOp>(
      bn.getLoc(), bn.getOutput().getType(), convInput, fusedWeight, fusedBias, conv2d.getStride(),
      conv2d.getPadding(), conv2d.getDilation(), conv2d.getTransposedAttr(), conv2d.getOutputPadding(),
      conv2d.getGroupsAttr());
  }
  auto convWb = cast<AclnnConv2DWithBiasOp>(convTemplateOp);
  return rewriter.create<AclnnConv2DWithBiasOp>(
    bn.getLoc(), bn.getOutput().getType(), convInput, fusedWeight, fusedBias, convWb.getStride(), convWb.getPadding(),
    convWb.getDilation(), convWb.getTransposedAttr(), convWb.getOutputPadding(), convWb.getGroupsAttr());
}

class FuseConvBatchNormPattern : public OpRewritePattern<AclnnBatchNormOp> {
 public:
  using OpRewritePattern<AclnnBatchNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AclnnBatchNormOp bn, PatternRewriter &rewriter) const override {
    auto [convOut, convInput, convWeight, convBias, convTemplateOp] = getConvData(bn.getInput());
    if (failed(checkConvBnPreconditions(bn, convOut, convTemplateOp, rewriter))) {
      return failure();
    }

    SmallVector<float, 16> weightVals;
    RankedTensorType weightTy;
    int64_t channels = 0;

    SmallVector<float, 16> meanVals;
    SmallVector<float, 16> varVals;
    SmallVector<float, 16> gammaVals;
    SmallVector<float, 16> betaVals;
    SmallVector<float, 16> oldBiasVals;
    if (failed(loadConvBnConstants(bn, convWeight, convBias, weightVals, weightTy, meanVals, varVals, gammaVals,
                                   betaVals, oldBiasVals, channels, rewriter))) {
      return failure();
    }

    const float eps = static_cast<float>(bn.getEpsilonAttr().getValueAsDouble());
    std::vector<float> fusedWeight;
    std::vector<float> fusedBias;
    foldConvBnWeightsAndBias(
      weightVals, meanVals, varVals, gammaVals, betaVals, oldBiasVals, channels, eps, fusedWeight, fusedBias);

    rewriter.setInsertionPoint(bn);
    auto fusedWeightAttr = DenseElementsAttr::get(weightTy, ArrayRef<float>(fusedWeight));
    auto fusedBiasTy = RankedTensorType::get({channels}, rewriter.getF32Type());
    auto fusedBiasAttr = DenseElementsAttr::get(fusedBiasTy, ArrayRef<float>(fusedBias));
    auto fusedWeightCst = rewriter.create<ConstantOp>(bn.getLoc(), weightTy, fusedWeightAttr);
    auto fusedBiasCst = rewriter.create<ConstantOp>(bn.getLoc(), fusedBiasTy, fusedBiasAttr);
    AclnnConv2DWithBiasOp fusedConv =
      createFusedConvWithBias(bn, convInput, fusedWeightCst.getResult(), fusedBiasCst.getResult(), convTemplateOp,
                              rewriter);
    rewriter.replaceOp(bn, fusedConv.getResult());
    if (auto *convOp = convOut.getDefiningOp(); convOp && convOp->use_empty()) {
      rewriter.eraseOp(convOp);
    }
    MLOG(DEBUG) << "FuseConvBatchNorm: folded conv + aclnn.batch_norm";
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseConvBatchNorm, FuseConvBatchNormPattern)

}  // namespace mfuse
}  // namespace mlir
