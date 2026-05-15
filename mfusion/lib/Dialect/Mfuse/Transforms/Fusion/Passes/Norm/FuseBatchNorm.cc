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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/Norm/FuseBatchNorm.h"

#include "llvm/ADT/SmallVector.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_FUSEBATCHNORM
#define GEN_PASS_DEF_FUSEBATCHNORM
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

bool isRank1(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return t && t.getRank() == 1;
}

bool isRank4(mlir::Value v) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return t && t.getRank() == 4;
}

static bool isRank1StaticLen(mlir::Value v, int64_t &len) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t || t.getRank() != 1 || t.isDynamicDim(0)) {
    return false;
  }
  len = t.getDimSize(0);
  return len > 0;
}

// Accept both [C,1,1] and [1,C,1,1] channel-broadcast forms.
static bool isCanonicalChannelBroadcastReshape(ReshapeOp reshape, int64_t channels) {
  auto outTy = mlir::dyn_cast<mlir::RankedTensorType>(reshape.getResult().getType());
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

std::optional<double> readF32SplatConstant(mlir::Value v) {
  auto cst = v.getDefiningOp<ConstantOp>();
  if (!cst) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.getElementType().isF32() || !dense.isSplat()) {
    return std::nullopt;
  }
  return static_cast<double>(dense.getSplatValue<llvm::APFloat>().convertToFloat());
}

bool isF32OneScalar(mlir::Value v) {
  auto o = readF32SplatConstant(v);
  return o.has_value() && *o == 1.0;
}

static bool isF32ScalarConstant(mlir::Value v) {
  return readF32SplatConstant(v).has_value();
}

static bool isReasonableBatchNormEps(double eps) {
  // Keep external-stats matching conservative: BatchNorm epsilon is expected to
  // be a small positive scalar. Rejecting negative/too-large eps helps avoid
  // folding arbitrary affine normalization-like chains into BN.
  return eps > 0.0 && eps <= 1.0e-2;
}

static bool isNchwSpatialReduceDims(mlir::ArrayAttr dimsAttr) {
  if (!dimsAttr || dimsAttr.size() != 3) {
    return false;
  }
  bool has0 = false;
  bool has2 = false;
  bool has3 = false;
  for (auto dimAttr : dimsAttr.getValue()) {
    auto dim = mlir::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
    has0 = has0 || (dim == 0);
    has2 = has2 || (dim == 2);
    has3 = has3 || (dim == 3);
  }
  return has0 && has2 && has3;
}

static bool isReduceSumOverX(mlir::Value v, mlir::Value x) {
  auto rs = v.getDefiningOp<ReduceSumOp>();
  return rs && rs.getInput() == x && !rs.getKeepdim() && isNchwSpatialReduceDims(rs.getDimensions());
}

static bool isScaledReduceSumOverX(mlir::Value v, mlir::Value x) {
  if (isReduceSumOverX(v, x)) {
    return true;
  }
  if (auto mul = v.getDefiningOp<MulOp>()) {
    return (isReduceSumOverX(mul.getLhs(), x) && isF32ScalarConstant(mul.getRhs())) ||
           (isReduceSumOverX(mul.getRhs(), x) && isF32ScalarConstant(mul.getLhs()));
  }
  if (auto div = v.getDefiningOp<DivOp>()) {
    return isReduceSumOverX(div.getSelf(), x) && isF32ScalarConstant(div.getOther());
  }
  if (auto reshape = v.getDefiningOp<ReshapeOp>()) {
    return isScaledReduceSumOverX(reshape.getInput(), x);
  }
  return false;
}

static bool isCenteredByMean(mlir::Value v, mlir::Value x, mlir::Value meanRank1) {
  auto sub = v.getDefiningOp<SubOp>();
  if (!sub) {
    return false;
  }
  mlir::Value lhs = sub.getX();
  mlir::Value rhs = sub.getY();
  if (lhs != x && rhs != x) {
    return false;
  }
  mlir::Value meanSide = (lhs == x) ? rhs : lhs;
  auto reshape = meanSide.getDefiningOp<ReshapeOp>();
  return reshape && reshape.getInput() == meanRank1;
}

static bool isVarianceNumeratorFromX(mlir::Value v, mlir::Value x, mlir::Value meanRank1) {
  auto rs = v.getDefiningOp<ReduceSumOp>();
  if (!rs || rs.getKeepdim() || !isNchwSpatialReduceDims(rs.getDimensions())) {
    return false;
  }
  auto squareMul = rs.getInput().getDefiningOp<MulOp>();
  if (!squareMul) {
    return false;
  }
  return isCenteredByMean(squareMul.getLhs(), x, meanRank1) && isCenteredByMean(squareMul.getRhs(), x, meanRank1);
}

static bool isVarianceOfX(mlir::Value v, mlir::Value x, mlir::Value meanRank1) {
  if (isVarianceNumeratorFromX(v, x, meanRank1)) {
    return true;
  }
  if (auto mul = v.getDefiningOp<MulOp>()) {
    return (isVarianceNumeratorFromX(mul.getLhs(), x, meanRank1) && isF32ScalarConstant(mul.getRhs())) ||
           (isVarianceNumeratorFromX(mul.getRhs(), x, meanRank1) && isF32ScalarConstant(mul.getLhs()));
  }
  if (auto div = v.getDefiningOp<DivOp>()) {
    return isVarianceNumeratorFromX(div.getSelf(), x, meanRank1) && isF32ScalarConstant(div.getOther());
  }
  if (auto reshape = v.getDefiningOp<ReshapeOp>()) {
    return isVarianceOfX(reshape.getInput(), x, meanRank1);
  }
  return false;
}

/// Match `running_var + eps` where `eps` is an f32 scalar constant tensor; return epsilon and `running_var`.
std::optional<double> matchRunningVarPlusEps(AddOp add, mlir::Value &runningVarOut) {
  mlir::Value lhs = add.getX();
  mlir::Value rhs = add.getY();
  auto trySide = [&](mlir::Value a, mlir::Value b) -> std::optional<double> {
    if (!isRank1(a)) {
      return std::nullopt;
    }
    auto eps = readF32SplatConstant(b);
    if (!eps.has_value()) {
      return std::nullopt;
    }
    runningVarOut = a;
    return eps;
  };
  if (auto e = trySide(lhs, rhs)) {
    return e;
  }
  if (auto e = trySide(rhs, lhs)) {
    return e;
  }
  return std::nullopt;
}

void eraseIfDead(mlir::MutableArrayRef<mlir::Operation *> ops, mlir::PatternRewriter &rewriter) {
  bool progress = true;
  while (progress) {
    progress = false;
    for (mlir::Operation *&op : ops) {
      if (!op) {
        continue;
      }
      if (!op->use_empty()) {
        continue;
      }
      rewriter.eraseOp(op);
      op = nullptr;
      progress = true;
    }
  }
}

/// Fuses:
///   inv = reciprocal(sqrt(running_var + eps)) optionally `mul(inv, 1)` (no-op scale)
///   y = (((x - reshape(mean)) * reshape(inv)) * reshape(gamma)) + reshape(beta)
/// into `mfuse.aclnn.batch_norm`.
///
/// Backward-side chains that interleave `reduce_sum`, `select`, and `convolution_backward` are not matched here;
/// extend with additional patterns once a stable subgraph boundary exists.

struct DecomposedBnMatch {
  MulOp mulGamma{};
  ReshapeOp reshapeBeta{};
  MulOp mulInv{};
  ReshapeOp reshapeGamma{};
  SubOp sub{};
  ReshapeOp reshapeInv{};
  ReshapeOp reshapeMean{};
  mlir::Value x{};
  mlir::Value meanRank1{};
  mlir::Value gammaRank1{};
  mlir::Value betaRank1{};
  mlir::Value invRank1{};
  ReciprocalOp recip{};
  SqrtOp sqrt{};
  AddOp varPlusEps{};
  mlir::Value runningVar{};
  double epsVal = 0.0;
  mlir::SmallVector<mlir::Operation *, 16> deadOptional;
};

static mlir::LogicalResult matchMulAndReshapeBeta(AddOp addBeta, mlir::PatternRewriter &rewriter, MulOp &mulGamma,
                                                  ReshapeOp &reshapeBeta) {
  mulGamma = addBeta.getX().getDefiningOp<MulOp>();
  reshapeBeta = addBeta.getY().getDefiningOp<ReshapeOp>();
  if (!mulGamma || !reshapeBeta) {
    mulGamma = addBeta.getY().getDefiningOp<MulOp>();
    reshapeBeta = addBeta.getX().getDefiningOp<ReshapeOp>();
  }
  if (!mulGamma || !reshapeBeta) {
    return rewriter.notifyMatchFailure(addBeta, "not mul + reshape(beta) add");
  }
  return mlir::success();
}

static mlir::LogicalResult matchMulInvAndReshapeGamma(AddOp addBeta, mlir::PatternRewriter &rewriter, MulOp mulGamma,
                                                      MulOp &mulInv, ReshapeOp &reshapeGamma) {
  if (auto m = mulGamma.getLhs().getDefiningOp<MulOp>()) {
    if (mulGamma.getRhs().getDefiningOp<ReshapeOp>()) {
      mulInv = m;
      reshapeGamma = mulGamma.getRhs().getDefiningOp<ReshapeOp>();
    }
  }
  if (!mulInv && mulGamma.getRhs().getDefiningOp<MulOp>()) {
    mulInv = mulGamma.getRhs().getDefiningOp<MulOp>();
    reshapeGamma = mulGamma.getLhs().getDefiningOp<ReshapeOp>();
  }
  if (!mulInv || !reshapeGamma) {
    return rewriter.notifyMatchFailure(addBeta, "gamma mul shape");
  }
  return mlir::success();
}

static mlir::LogicalResult matchSubAndReshapeInv(AddOp addBeta, mlir::PatternRewriter &rewriter, MulOp mulInv,
                                                 SubOp &sub, ReshapeOp &reshapeInv) {
  mlir::Value lhsM = mulInv.getLhs();
  mlir::Value rhsM = mulInv.getRhs();
  if (auto s = lhsM.getDefiningOp<SubOp>()) {
    sub = s;
    reshapeInv = rhsM.getDefiningOp<ReshapeOp>();
  } else if (auto s = rhsM.getDefiningOp<SubOp>()) {
    sub = s;
    reshapeInv = lhsM.getDefiningOp<ReshapeOp>();
  }
  if (!sub || !reshapeInv) {
    return rewriter.notifyMatchFailure(addBeta, "mulInv does not use sub");
  }
  return mlir::success();
}

static void resolveReciprocalInput(mlir::Value invRank1, mlir::SmallVectorImpl<mlir::Operation *> &deadOptional,
                                    mlir::Value &recipValOut) {
  recipValOut = invRank1;
  if (auto scaleMul = invRank1.getDefiningOp<MulOp>()) {
    ReciprocalOp recL = scaleMul.getLhs().getDefiningOp<ReciprocalOp>();
    ReciprocalOp recR = scaleMul.getRhs().getDefiningOp<ReciprocalOp>();
    if (recL && isF32OneScalar(scaleMul.getRhs()) && scaleMul->hasOneUse()) {
      deadOptional.push_back(scaleMul);
      recipValOut = recL.getResult();
    } else if (recR && isF32OneScalar(scaleMul.getLhs()) && scaleMul->hasOneUse()) {
      deadOptional.push_back(scaleMul);
      recipValOut = recR.getResult();
    }
  }
}

static mlir::LogicalResult matchVarPlusEpsChain(AddOp addBeta, mlir::PatternRewriter &rewriter, mlir::Value recipVal,
                                                ReciprocalOp &recip, SqrtOp &sqrt, AddOp &varPlusEps,
                                                mlir::Value &runningVar, double &epsVal) {
  recip = recipVal.getDefiningOp<ReciprocalOp>();
  if (!recip) {
    return rewriter.notifyMatchFailure(addBeta, "inv not reciprocal (after optional mul-by-1)");
  }
  sqrt = recip.getInput().getDefiningOp<SqrtOp>();
  if (!sqrt) {
    return rewriter.notifyMatchFailure(addBeta, "no sqrt in inv");
  }
  varPlusEps = sqrt.getInput().getDefiningOp<AddOp>();
  if (!varPlusEps) {
    return rewriter.notifyMatchFailure(addBeta, "sqrt input is not add");
  }
  auto epsOpt = matchRunningVarPlusEps(varPlusEps, runningVar);
  if (!epsOpt.has_value()) {
    return rewriter.notifyMatchFailure(addBeta, "var+eps add not matched");
  }
  if (runningVar != varPlusEps.getX() && runningVar != varPlusEps.getY()) {
    return rewriter.notifyMatchFailure(addBeta, "running_var not operand of var+eps");
  }
  epsVal = *epsOpt;
  if (!isReasonableBatchNormEps(epsVal)) {
    return rewriter.notifyMatchFailure(addBeta, "eps out of conservative batch-norm range");
  }
  return mlir::success();
}

static mlir::LogicalResult validateBnCoreRanks(AddOp addBeta, mlir::PatternRewriter &rewriter,
                                               const DecomposedBnMatch &m, mlir::RankedTensorType &xTyOut) {
  if (!isRank1(m.meanRank1) || !isRank1(m.gammaRank1) || !isRank1(m.betaRank1) || !isRank4(m.x)) {
    return rewriter.notifyMatchFailure(addBeta, "rank mismatch");
  }
  if (!isRank1(m.invRank1)) {
    return rewriter.notifyMatchFailure(addBeta, "inv rank1 expected before broadcast reshape");
  }
  xTyOut = mlir::dyn_cast<mlir::RankedTensorType>(m.x.getType());
  if (!xTyOut || xTyOut.getRank() != 4) {
    return rewriter.notifyMatchFailure(addBeta, "x must be rank-4 tensor");
  }
  return mlir::success();
}

static mlir::LogicalResult loadBnChannelLengths(AddOp addBeta, mlir::PatternRewriter &rewriter,
                                                const DecomposedBnMatch &m, int64_t &channels, int64_t &varChannels,
                                                int64_t &gammaChannels, int64_t &betaChannels, int64_t &invChannels) {
  if (!isRank1StaticLen(m.meanRank1, channels) || !isRank1StaticLen(m.runningVar, varChannels) ||
      !isRank1StaticLen(m.gammaRank1, gammaChannels) || !isRank1StaticLen(m.betaRank1, betaChannels) ||
      !isRank1StaticLen(m.invRank1, invChannels)) {
    return rewriter.notifyMatchFailure(addBeta, "bn rank-1 inputs must have static channel length");
  }
  return mlir::success();
}

static mlir::LogicalResult validateBnBroadcastReshapes(AddOp addBeta, mlir::PatternRewriter &rewriter,
                                                       const DecomposedBnMatch &m, int64_t channels) {
  if (!isCanonicalChannelBroadcastReshape(m.reshapeInv, channels) ||
      !isCanonicalChannelBroadcastReshape(m.reshapeMean, channels) ||
      !isCanonicalChannelBroadcastReshape(m.reshapeGamma, channels) ||
      !isCanonicalChannelBroadcastReshape(m.reshapeBeta, channels)) {
    return rewriter.notifyMatchFailure(addBeta, "reshape does not match canonical BN channel broadcast");
  }
  return mlir::success();
}

static mlir::LogicalResult validateDecomposedBnRanksAndChannels(AddOp addBeta, mlir::PatternRewriter &rewriter,
                                                                const DecomposedBnMatch &m) {
  mlir::RankedTensorType xTy;
  if (failed(validateBnCoreRanks(addBeta, rewriter, m, xTy))) {
    return mlir::failure();
  }
  int64_t channels = 0;
  int64_t varChannels = 0;
  int64_t gammaChannels = 0;
  int64_t betaChannels = 0;
  int64_t invChannels = 0;
  if (failed(loadBnChannelLengths(
        addBeta, rewriter, m, channels, varChannels, gammaChannels, betaChannels, invChannels))) {
    return mlir::failure();
  }
  if (channels != varChannels || channels != gammaChannels || channels != betaChannels || channels != invChannels) {
    return rewriter.notifyMatchFailure(addBeta, "bn rank-1 inputs channel length mismatch");
  }
  if (!xTy.isDynamicDim(1) && xTy.getDimSize(1) != channels) {
    return rewriter.notifyMatchFailure(addBeta, "x channel dim mismatches bn channel length");
  }
  if (failed(validateBnBroadcastReshapes(addBeta, rewriter, m, channels))) {
    return mlir::failure();
  }
  bool meanMatchesEx = isScaledReduceSumOverX(m.meanRank1, m.x);
  bool varMatchesEx = isVarianceOfX(m.runningVar, m.x, m.meanRank1);
  if (!meanMatchesEx) {
    return rewriter.notifyMatchFailure(addBeta, "mean is not E(x) style reduce over x");
  }
  if (!varMatchesEx) {
    return rewriter.notifyMatchFailure(addBeta, "running_var is not Var(x) style reduce over x");
  }
  return mlir::success();
}

static mlir::LogicalResult validateDecomposedBnSingleUses(AddOp addBeta, mlir::PatternRewriter &rewriter,
                                                        const DecomposedBnMatch &m) {
  if (!m.varPlusEps->hasOneUse() || !m.sqrt->hasOneUse() || !m.recip->hasOneUse()) {
    return rewriter.notifyMatchFailure(addBeta, "inv_std chain shared");
  }
  if (!m.reshapeInv->hasOneUse() || !m.reshapeMean->hasOneUse() || !m.reshapeGamma->hasOneUse() ||
      !m.reshapeBeta->hasOneUse()) {
    return rewriter.notifyMatchFailure(addBeta, "reshape shared");
  }
  if (!m.sub->hasOneUse() || !m.mulInv->hasOneUse() || !m.mulGamma->hasOneUse()) {
    return rewriter.notifyMatchFailure(addBeta, "bn chain shared at binary op");
  }
  return mlir::success();
}

static mlir::LogicalResult tryParseDecomposedBnMatch(AddOp addBeta, mlir::PatternRewriter &rewriter,
                                                     DecomposedBnMatch &m) {
  if (failed(matchMulAndReshapeBeta(addBeta, rewriter, m.mulGamma, m.reshapeBeta))) {
    return mlir::failure();
  }
  if (failed(matchMulInvAndReshapeGamma(addBeta, rewriter, m.mulGamma, m.mulInv, m.reshapeGamma))) {
    return mlir::failure();
  }
  if (failed(matchSubAndReshapeInv(addBeta, rewriter, m.mulInv, m.sub, m.reshapeInv))) {
    return mlir::failure();
  }
  m.reshapeMean = m.sub.getY().getDefiningOp<ReshapeOp>();
  if (!m.reshapeMean) {
    return rewriter.notifyMatchFailure(addBeta, "mean is not reshape");
  }
  m.x = m.sub.getX();
  m.invRank1 = m.reshapeInv.getInput();
  m.meanRank1 = m.reshapeMean.getInput();
  m.gammaRank1 = m.reshapeGamma.getInput();
  m.betaRank1 = m.reshapeBeta.getInput();

  mlir::Value recipVal;
  resolveReciprocalInput(m.invRank1, m.deadOptional, recipVal);
  if (failed(matchVarPlusEpsChain(addBeta, rewriter, recipVal, m.recip, m.sqrt, m.varPlusEps, m.runningVar,
                                  m.epsVal))) {
    return mlir::failure();
  }
  if (failed(validateDecomposedBnRanksAndChannels(addBeta, rewriter, m))) {
    return mlir::failure();
  }
  if (failed(validateDecomposedBnSingleUses(addBeta, rewriter, m))) {
    return mlir::failure();
  }
  return mlir::success();
}

static void replaceAddBetaWithFusedBn(AddOp addBeta, mlir::PatternRewriter &rewriter, const DecomposedBnMatch &m) {
  MLOG(DEBUG) << "FuseBatchNormPattern: epsilon=" << m.epsVal;
  rewriter.setInsertionPoint(addBeta);
  // Training BN and inference BN differ in two key aspects:
  // 1) Training BN uses current batch statistics and updates running stats;
  // 2) Inference BN consumes pre-defined running stats without state updates.
  //
  // This fusion only matches pure forward normalization math and does not preserve
  // any running-stat update side effects. To keep semantics conservative and avoid
  // introducing implicit training behavior, we always materialize the fused op as
  // inference BN (training=false, use_input_stats=false).
  auto fused = rewriter.create<mlir::mfuse::AclnnBatchNormOp>(
    addBeta.getLoc(), addBeta.getType(), m.x, m.gammaRank1, m.betaRank1, m.meanRank1, m.runningVar,
    rewriter.getBoolAttr(false), rewriter.getF64FloatAttr(0.0), rewriter.getF64FloatAttr(m.epsVal),
    rewriter.getBoolAttr(false));
  rewriter.replaceOp(addBeta, fused.getResult());

  llvm::SmallVector<mlir::Operation *, 32> deadMut(m.deadOptional.begin(), m.deadOptional.end());
  deadMut.push_back(m.mulGamma);
  deadMut.push_back(m.mulInv);
  deadMut.push_back(m.sub);
  deadMut.push_back(m.reshapeInv);
  deadMut.push_back(m.reshapeMean);
  deadMut.push_back(m.reshapeGamma);
  deadMut.push_back(m.reshapeBeta);
  deadMut.push_back(m.recip);
  deadMut.push_back(m.sqrt);
  deadMut.push_back(m.varPlusEps);
  eraseIfDead(deadMut, rewriter);
}

}  // namespace

class FuseBatchNormPattern : public mlir::OpRewritePattern<AddOp> {
 public:
  explicit FuseBatchNormPattern(mlir::MLIRContext *context)
      : OpRewritePattern<AddOp>(context, mlir::PatternBenefit(1)) {}

  mlir::LogicalResult matchAndRewrite(AddOp addBeta, mlir::PatternRewriter &rewriter) const override {
    if (!isRank4(addBeta.getResult())) {
      return rewriter.notifyMatchFailure(addBeta, "expect rank-4 BN output");
    }
    DecomposedBnMatch m;
    if (failed(tryParseDecomposedBnMatch(addBeta, rewriter, m))) {
      return mlir::failure();
    }
    replaceAddBetaWithFusedBn(addBeta, rewriter, m);
    return mlir::success();
  }
};

struct FuseBatchNormPass : public impl::FuseBatchNormBase<FuseBatchNormPass> {
  using FuseBatchNormBase::FuseBatchNormBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuseBatchNormPattern>(&getContext());
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createFuseBatchNormPass() {
  return std::make_unique<FuseBatchNormPass>();
}

}  // namespace mfuse
}  // namespace mlir
