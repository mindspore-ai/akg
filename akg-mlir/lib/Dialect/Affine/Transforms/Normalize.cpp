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

#include "akg/Dialect/Affine/Transforms/Normalize.h"

#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

namespace mlir {

#define GEN_PASS_DEF_NORMALIZE
#define GEN_PASS_DECL_NORMALIZE
#include "akg/Dialect/Affine/Passes.h.inc"

static constexpr int kTaylerExpansionNum = 5;
static constexpr int kTaylerOrderSub = 2;
static constexpr double kBaseTwo = 2.0;

#define DEBUG_TYPE "math-normalize"

namespace {

using mlir::hivm::RoundMode;
using mlir::hivm::RoundModeAttr;

static Value buildFloatConst(PatternRewriter &rewriter, Location loc, Type ty, double v) {
  auto fTy = dyn_cast<FloatType>(ty);
  assert(fTy && "expected float type");
  return rewriter.create<arith::ConstantOp>(loc, fTy, rewriter.getFloatAttr(fTy, v));
}

static Value buildIntConst(PatternRewriter &rewriter, Location loc, unsigned bitWidth, int64_t v) {
  auto iTy = rewriter.getIntegerType(bitWidth);
  return rewriter.create<arith::ConstantOp>(loc, iTy, rewriter.getIntegerAttr(iTy, v));
}

static Value buildNaNConst(PatternRewriter &rewriter, Location loc, FloatType fTy) {
  llvm::APFloat nanVal = llvm::APFloat::getNaN(fTy.getFloatSemantics());
  return rewriter.create<arith::ConstantOp>(loc, fTy, rewriter.getFloatAttr(fTy, nanVal));
}

static Value buildInfConst(PatternRewriter &rewriter, Location loc, FloatType fTy) {
  llvm::APFloat val = llvm::APFloat::getInf(fTy.getFloatSemantics(),
                                            /* Negative= */ false);
  return rewriter.create<arith::ConstantOp>(loc, fTy, rewriter.getFloatAttr(fTy, val));
}

static std::optional<llvm::APFloat> getConstantFloat(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto fAttr = dyn_cast<FloatAttr>(cst.getValue())) {
      return fAttr.getValue();
    }
  }
  return std::nullopt;
}

/// round(x)
static Value buildRoundWithMath(PatternRewriter &rewriter, Location loc, Value x) {
  assert(isa<FloatType>(getElementTypeOrSelf(x.getType())) && "expected float type");
  return rewriter.create<math::RoundOp>(loc, x);
}

/// High/low precision norm///   res = x
///   for each p in piApproParams:
///     res = res - p * xRoundHigh
///     [if i==1 and offsetHigh: res = res + offsetHigh]
///     res = res - p * xRoundLow
///   if offsetLow: res = res + offsetLow
/// This avoids the rounding error of `xRound * pi` for large xRound.
static Value buildNormSplit(PatternRewriter &rewriter, Location loc, Value x, Value xRoundHigh, Value xRoundLow,
                            Type elemTy, ArrayRef<double> piApproParams,
                            std::optional<double> offsetHigh = std::nullopt,
                            std::optional<double> offsetLow = std::nullopt) {
  Value res = x;
  for (size_t i = 0; i < piApproParams.size(); ++i) {
    Value piCst = buildFloatConst(rewriter, loc, elemTy, piApproParams[i]);
    Value kpHigh = rewriter.create<arith::MulFOp>(loc, piCst, xRoundHigh);
    res = rewriter.create<arith::SubFOp>(loc, res, kpHigh);

    // For cos: insert offsetHigh between p[1]*high and p[1]*low
    if (i == 1 && offsetHigh.has_value()) {
      Value oh = buildFloatConst(rewriter, loc, elemTy, offsetHigh.value());
      res = rewriter.create<arith::AddFOp>(loc, res, oh);
    }

    Value kpLow = rewriter.create<arith::MulFOp>(loc, piCst, xRoundLow);
    res = rewriter.create<arith::SubFOp>(loc, res, kpLow);
  }
  if (offsetLow.has_value()) {
    Value ol = buildFloatConst(rewriter, loc, elemTy, offsetLow.value());
    res = rewriter.create<arith::AddFOp>(loc, res, ol);
  }
  return res;
}

/// Compute (xRound, xRoundHigh, xRoundLow)///   xRoundHigh = round(inputDivPi / 2048) * 2048
///   xRoundLow  = xRound - xRoundHigh
static std::tuple<Value, Value, Value> buildXRoundSplit(PatternRewriter &rewriter, Location loc, Value inputDivPi,
                                                        Value xRound, Type elemTy) {
  Value invSplit = buildFloatConst(rewriter, loc, elemTy, 1.0 / 2048.0);
  Value split = buildFloatConst(rewriter, loc, elemTy, 2048.0);
  Value scaled = rewriter.create<arith::MulFOp>(loc, inputDivPi, invSplit);

  Value scaledRound = buildRoundWithMath(rewriter, loc, scaled);

  Value xRoundHigh = rewriter.create<arith::MulFOp>(loc, scaledRound, split);
  Value xRoundLow = rewriter.create<arith::SubFOp>(loc, xRound, xRoundHigh);
  return {xRound, xRoundHigh, xRoundLow};
}

enum class TaylerMode { SIN, ATAN };

static SmallVector<double> getTaylerParams(TaylerMode taylerMode, int taylerExpansionNum) {
  SmallVector<double> taylerParams;
  switch (taylerMode) {
    case TaylerMode::SIN: {
      //   tayler(x) ≈ x * (1 + c1*x^2 + c2*x^4 + c3*x^6 + c4*x^8)
      if (taylerExpansionNum == kTaylerExpansionNum) {
        taylerParams.push_back(1.0);             // index 0
        taylerParams.push_back(-0.166666582);    // c1
        taylerParams.push_back(8.333050e-03);    // c2
        taylerParams.push_back(-1.98089445e-4);  // c3
        taylerParams.push_back(2.60492652e-6);   // c4
        return taylerParams;
      }
      // Fallback: standard Taylor expansion.
      taylerParams.push_back(1.0);
      double acc = 1.0;
      for (int i = 1; i < taylerExpansionNum; ++i) {
        acc = acc * (2 * i) * (2 * i + 1) * (-1);
        taylerParams.push_back(1.0 / acc);
      }
      return taylerParams;
    }
    case TaylerMode::ATAN: {
      taylerParams.push_back(1.0);
      for (int i = 1; i < taylerExpansionNum; ++i) {
        const double acc = (i % 2 == 0) ? (2 * i + 1) : (2 * i + 1) * (-1);
        taylerParams.push_back(1.0 / acc);
      }
      return taylerParams;
    }
  }
  llvm_unreachable("Unsupported TaylerMode");
}

static Value createTaylerSeries(PatternRewriter &rewriter, Location loc, Value lastTaylerTerm, Value xPow,
                                int taylerExpansionNum, ArrayRef<double> taylerParams) {
  Value partialRes = lastTaylerTerm;
  auto elemTy = getElementTypeOrSelf(xPow.getType());
  for (int i = 0; i < taylerExpansionNum - kTaylerOrderSub; ++i) {
    double coef = taylerParams[taylerExpansionNum - i - 2];
    Value coefCst = buildFloatConst(rewriter, loc, elemTy, coef);
    Value curTerm = rewriter.create<arith::AddFOp>(loc, partialRes, coefCst);
    partialRes = rewriter.create<arith::MulFOp>(loc, curTerm, xPow);
  }
  return partialRes;
}

/// tayler(x) = x * (1 + \sum_{i>=1} a_i x^{2i})
static Value buildTayler(PatternRewriter &rewriter, Location loc, Value x, int taylerExpansionNum, TaylerMode mode) {
  SmallVector<double> taylerParams = getTaylerParams(mode, taylerExpansionNum);

  auto elemTy = getElementTypeOrSelf(x.getType());
  Value xPow = rewriter.create<arith::MulFOp>(loc, x, x);  // x^2

  // lastTerm = a_(n-1) * x^2
  double lastCoef = taylerParams[taylerExpansionNum - 1];
  Value lastCoefCst = buildFloatConst(rewriter, loc, elemTy, lastCoef);
  Value lastTerm = rewriter.create<arith::MulFOp>(loc, xPow, lastCoefCst);

  Value partialRes = createTaylerSeries(rewriter, loc, lastTerm, xPow, taylerExpansionNum, taylerParams);

  Value one = buildFloatConst(rewriter, loc, elemTy, 1.0);
  Value poly1 = rewriter.create<arith::AddFOp>(loc, partialRes, one);

  // tayler(x) = poly1 * x
  Value res = rewriter.create<arith::MulFOp>(loc, poly1, x);
  return res;
}

/// sign for sin: floor(x/2)*4 - x*2 + 1
static Value buildSinSign(PatternRewriter &rewriter, Location loc, Value x) {
  auto elemTy = getElementTypeOrSelf(x.getType());

  Value half = buildFloatConst(rewriter, loc, elemTy, 0.5);
  Value kHalf = rewriter.create<arith::MulFOp>(loc, x, half);
  Value kHalfFloor = rewriter.create<math::FloorOp>(loc, kHalf);

  Value four = buildFloatConst(rewriter, loc, elemTy, 4.0);
  Value kHalfFloor4 = rewriter.create<arith::MulFOp>(loc, kHalfFloor, four);

  Value minusTwo = buildFloatConst(rewriter, loc, elemTy, -2.0);
  Value k2 = rewriter.create<arith::MulFOp>(loc, x, minusTwo);

  Value sign = rewriter.create<arith::AddFOp>(loc, kHalfFloor4, k2);
  Value one = buildFloatConst(rewriter, loc, elemTy, 1.0);
  Value res = rewriter.create<arith::AddFOp>(loc, sign, one);
  return res;
}

static double getFPMAX(FloatType fType) {
  if (fType.isF32()) {
    return std::pow(kBaseTwo, fType.getWidth() + 30);
  }
  return std::pow(kBaseTwo, fType.getWidth() - 1);
}

static double getFPMIN(FloatType fType) {
  if (fType.isF32()) {
    return std::pow(kBaseTwo, -static_cast<int>(fType.getWidth() + 30));
  }
  return std::pow(kBaseTwo, -static_cast<int>(fType.getWidth() - 1));
}

/// sign(x) = FP_MAX * x /(FP_MIN + FP_MAX *|x|)
static Value buildAtanSign(PatternRewriter &rewriter, Location loc, Value x) {
  auto elemTy = dyn_cast<FloatType>(getElementTypeOrSelf(x.getType()));
  assert(elemTy && "Only support floatType");

  double fpMax = getFPMAX(elemTy);
  double fpMin = getFPMIN(elemTy);

  Value fpMaxCst = buildFloatConst(rewriter, loc, elemTy, fpMax);
  Value fpMinCst = buildFloatConst(rewriter, loc, elemTy, fpMin);

  Value mul = rewriter.create<arith::MulFOp>(loc, x, fpMaxCst);
  Value absVal = rewriter.create<math::AbsFOp>(loc, mul);
  Value denom = rewriter.create<arith::AddFOp>(loc, absVal, fpMinCst);
  Value res = rewriter.create<arith::DivFOp>(loc, mul, denom);
  return res;
}

static Value buildSign(PatternRewriter &rewriter, Location loc, Value x, TaylerMode mode) {
  switch (mode) {
    case TaylerMode::SIN:
      return buildSinSign(rewriter, loc, x);
    case TaylerMode::ATAN:
      return buildAtanSign(rewriter, loc, x);
  }
  llvm_unreachable("unsupported TaylerMode");
}

static Value clipInput(PatternRewriter &rewriter, Location loc, Value input, double maxVal, double minVal) {
  auto elemTy = getElementTypeOrSelf(input.getType());
  Value maxCst = buildFloatConst(rewriter, loc, elemTy, maxVal);
  Value minCst = buildFloatConst(rewriter, loc, elemTy, minVal);
  Value tmp = rewriter.create<arith::MinimumFOp>(loc, input, maxCst);
  return rewriter.create<arith::MaximumFOp>(loc, tmp, minCst);
}

// === piApproParams that approximate pi via 4 high-precision splits ===
//   3.14160156 + (-8.9071691E-6) + (-1.74122761E-9) + (1.24467439E-13) ≈ pi
static const double kPiApproParams[] = {3.14160156, -8.9071691E-6, -1.74122761E-9, 1.24467439E-13};
// pi/2 split into high + low for cos offset.
static constexpr double kPiOver2High = 1.57079637;
static constexpr double kPiOver2Low = -4.37113883E-8;

struct NormalizeSinOp : public OpRewritePattern<math::SinOp> {
  using OpRewritePattern<math::SinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::SinOp op, PatternRewriter &rewriter) const override {
    Value input = op.getOperand();
    auto inTy = getElementTypeOrSelf(input.getType());
    auto fTy = dyn_cast<FloatType>(inTy);
    if (!fTy || (!fTy.isF16() && !fTy.isF32())) {
      return failure();
    }

    Location loc = op.getLoc();

    bool needCastBack = fTy.isF16();
    if (needCastBack) {
      auto f32Ty = rewriter.getF32Type();
      auto ext = rewriter.create<arith::ExtFOp>(loc, f32Ty, input);

      auto roundAttr = RoundModeAttr::get(rewriter.getContext(), RoundMode::ROUND);
      ext->setAttr(RoundModeAttr::getMnemonic(), roundAttr);

      input = ext;
      fTy = f32Ty;
    }

    // inputDivPi = input * (1/pi)
    Value piRec = buildFloatConst(rewriter, loc, fTy, 1.0 / static_cast<double>(M_PI));
    Value inputDivPi = rewriter.create<arith::MulFOp>(loc, input, piRec);

    // xRound = round(inputDivPi)
    Value xRound = buildRoundWithMath(rewriter, loc, inputDivPi);

    // xRound = xRoundHigh + xRoundLow, with xRoundHigh = round(inputDivPi / 2048) * 2048
    auto [xRoundOrig, xRoundHigh, xRoundLow] = buildXRoundSplit(rewriter, loc, inputDivPi, xRound, fTy);
    (void)xRoundOrig;

    ArrayRef<double> piApproParams(kPiApproParams, sizeof(kPiApproParams) / sizeof(double));

    // norm_x = input - sum_p (p * xRoundHigh + p * xRoundLow)
    Value normInput = buildNormSplit(rewriter, loc, input, xRoundHigh, xRoundLow, fTy, piApproParams,
                                     /* offsetHigh= */ std::nullopt, /* offsetLow= */ std::nullopt);

    // sinTayler(norm_x), 5 terms
    Value sinTaylerNorm = buildTayler(rewriter, loc, normInput, /* taylerExpansionNum= */ 5, TaylerMode::SIN);

    // sign(xRound) = floor(xRound/2)*4 - xRound*2 + 1
    Value signX = buildSign(rewriter, loc, xRound, TaylerMode::SIN);

    Value res = rewriter.create<arith::MulFOp>(loc, sinTaylerNorm, signX);

    if (needCastBack) {
      auto f16Ty = rewriter.getF16Type();
      auto trunc = rewriter.create<arith::TruncFOp>(loc, f16Ty, res);

      auto roundAttr = RoundModeAttr::get(rewriter.getContext(), RoundMode::ROUND);
      trunc->setAttr(RoundModeAttr::getMnemonic(), roundAttr);

      res = trunc;
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct NormalizeCosOp : public OpRewritePattern<math::CosOp> {
  using OpRewritePattern<math::CosOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::CosOp op, PatternRewriter &rewriter) const override {
    Value input = op.getOperand();
    auto inTy = getElementTypeOrSelf(input.getType());
    auto fTy = dyn_cast<FloatType>(inTy);
    if (!fTy || (!fTy.isF16() && !fTy.isF32())) {
      return failure();
    }

    Location loc = op.getLoc();
    bool needCastBack = fTy.isF16();
    if (needCastBack) {
      auto f32Ty = rewriter.getF32Type();
      auto ext = rewriter.create<arith::ExtFOp>(loc, f32Ty, input);

      auto roundAttr = RoundModeAttr::get(rewriter.getContext(), RoundMode::ROUND);
      ext->setAttr(RoundModeAttr::getMnemonic(), roundAttr);

      input = ext;
      fTy = f32Ty;
    }

    // inputDivPi = input * (1/pi)
    Value piRec = buildFloatConst(rewriter, loc, fTy, 1.0 / static_cast<double>(M_PI));
    Value inputDivPi = rewriter.create<arith::MulFOp>(loc, input, piRec);

    // xRound = round(inputDivPi + 0.5)
    Value half = buildFloatConst(rewriter, loc, fTy, 0.5);
    Value xPlusHalf = rewriter.create<arith::AddFOp>(loc, inputDivPi, half);
    Value xRound = buildRoundWithMath(rewriter, loc, xPlusHalf);

    // High/low split of xRound; note: split base is inputDivPi (no +0.5), per IR.
    auto [xRoundOrig, xRoundHigh, xRoundLow] = buildXRoundSplit(rewriter, loc, inputDivPi, xRound, fTy);
    (void)xRoundOrig;

    ArrayRef<double> piApproParams(kPiApproParams, sizeof(kPiApproParams) / sizeof(double));

    // norm with split offset: pi/2_high inserted between p[1]*high and p[1]*low,
    // pi/2_low added at the end.
    Value normInput = buildNormSplit(rewriter, loc, input, xRoundHigh, xRoundLow, fTy, piApproParams,
                                     /* offsetHigh= */ kPiOver2High, /* offsetLow= */ kPiOver2Low);

    // sinTayler is reused for cos via the offset shift.
    Value cosTayler = buildTayler(rewriter, loc, normInput, /* taylerExpansionNum= */ 5, TaylerMode::SIN);

    Value signX = buildSign(rewriter, loc, xRound, TaylerMode::SIN);

    Value res = rewriter.create<arith::MulFOp>(loc, cosTayler, signX);

    // Clip result to [-1, 1] (cos's mathematical range).
    res = clipInput(rewriter, loc, res, /* maxVal= */ 1.0, /* minVal= */ -1.0);

    if (needCastBack) {
      auto f16Ty = rewriter.getF16Type();
      auto trunc = rewriter.create<arith::TruncFOp>(loc, f16Ty, res);

      auto roundAttr = RoundModeAttr::get(rewriter.getContext(), RoundMode::ROUND);
      trunc->setAttr(RoundModeAttr::getMnemonic(), roundAttr);

      res = trunc;
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct NormalizeTanhOp : public OpRewritePattern<math::TanhOp> {
  using OpRewritePattern<math::TanhOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::TanhOp op, PatternRewriter &rewriter) const override {
    Value input = op.getOperand();
    auto inTy = getElementTypeOrSelf(input.getType());
    auto fTy = dyn_cast<FloatType>(inTy);
    if (!fTy || (!fTy.isF16() && !fTy.isF32())) {
      return failure();
    }

    Location loc = op.getLoc();
    bool needCastBack = fTy.isF16();
    if (needCastBack) {
      auto f32Ty = rewriter.getF32Type();
      auto ext = rewriter.create<arith::ExtFOp>(loc, f32Ty, input);

      auto roundAttr = RoundModeAttr::get(rewriter.getContext(), RoundMode::ROUND);
      ext->setAttr(RoundModeAttr::getMnemonic(), roundAttr);

      input = ext;
      fTy = f32Ty;
    }

    // step 1: clip to [-8.8, 8.8] to avoid exp(2x) overflow; epsilon ~ 1e-8.
    Value clippedInput = clipInput(rewriter, loc, input, 8.8, -8.8);

    // step 2: y = exp(2x)
    Value two = buildFloatConst(rewriter, loc, fTy, 2.0);
    Value mul2x = rewriter.create<arith::MulFOp>(loc, clippedInput, two);
    Value exp2x = rewriter.create<math::ExpOp>(loc, mul2x);

    // step 3: number = exp(2x) - 1
    Value negOne = buildFloatConst(rewriter, loc, fTy, -1.0);
    Value number = rewriter.create<arith::AddFOp>(loc, exp2x, negOne);

    // step 4: denom = exp(2x) + 1
    Value posOne = buildFloatConst(rewriter, loc, fTy, 1.0);
    Value denom = rewriter.create<arith::AddFOp>(loc, exp2x, posOne);

    // step 5: tanh(x) = number / denom
    Value res = rewriter.create<arith::DivFOp>(loc, number, denom);

    if (needCastBack) {
      auto f16Ty = rewriter.getF16Type();
      auto trunc = rewriter.create<arith::TruncFOp>(loc, f16Ty, res);

      auto roundAttr = RoundModeAttr::get(rewriter.getContext(), RoundMode::ROUND);
      trunc->setAttr(RoundModeAttr::getMnemonic(), roundAttr);

      res = trunc;
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct NormalizePowfOp : public OpRewritePattern<math::PowFOp> {
  using OpRewritePattern<math::PowFOp>::OpRewritePattern;

  Value buildAbs(PatternRewriter &rewriter, Location loc, Value x) const {
    return rewriter.create<math::AbsFOp>(loc, x);
  }

  Value buildFloor(PatternRewriter &rewriter, Location loc, Value x) const {
    return rewriter.create<math::FloorOp>(loc, x);
  }

  Value buildLog(PatternRewriter &rewriter, Location loc, Value x) const {
    return rewriter.create<math::LogOp>(loc, x);
  }

  Value buildExp(PatternRewriter &rewriter, Location loc, Value x) const {
    return rewriter.create<math::ExpOp>(loc, x);
  }

  Value buildSqrt(PatternRewriter &rewriter, Location loc, Value x) const {
    return rewriter.create<math::SqrtOp>(loc, x);
  }

  Value buildIsFinite(PatternRewriter &rewriter, Location loc, Value x) const {
    auto fTy = cast<FloatType>(x.getType());
    Value absX = buildAbs(rewriter, loc, x);
    Value inf = buildInfConst(rewriter, loc, fTy);

    Value isInfinite = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, absX, inf);

    auto i1Ty = isInfinite.getType();
    auto one = rewriter.create<arith::ConstantIntOp>(loc, 1, i1Ty);
    return rewriter.create<arith::XOrIOp>(loc, isInfinite, one);
  }

  Value buildIsInteger(PatternRewriter &rewriter, Location loc, Value y) const {
    Value floorY = buildFloor(rewriter, loc, y);
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, y, floorY);
  }

  Value getSignbitOfBase(PatternRewriter &rewriter, Location loc, Value base) const {
    auto fTy = dyn_cast<FloatType>(base.getType());
    assert(fTy && "expected float type");
    unsigned bitWidth = fTy.getWidth();
    auto iTy = rewriter.getIntegerType(bitWidth);

    Value bitcast = rewriter.create<arith::BitcastOp>(loc, iTy, base);

    Value shiftAmount = buildIntConst(rewriter, loc, bitWidth, bitWidth - 1);
    // ShRSIOp: arithmetic shift right, like HFusion shrsi
    Value shr = rewriter.create<arith::ShRSIOp>(loc, bitcast, shiftAmount);

    // const -1 of same integer type
    Value negOne = buildIntConst(rewriter, loc, bitWidth, -1);
    auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, shr, negOne);
    return cmp;
  }

  Value buildIsNegCondition(PatternRewriter &rewriter, Location loc, Value base, Value exponent) const {
    Value isNeg = getSignbitOfBase(rewriter, loc, base);
    Value isInteger = buildIsInteger(rewriter, loc, exponent);
    return rewriter.create<arith::AndIOp>(loc, isNeg, isInteger);
  }

  Value buildNegOnePowY(PatternRewriter &rewriter, Location loc, Value y) const {
    auto fTy = cast<FloatType>(y.getType());

    Value one = buildFloatConst(rewriter, loc, fTy, 1.0);
    Value two = buildFloatConst(rewriter, loc, fTy, 2.0);
    Value negTwo = buildFloatConst(rewriter, loc, fTy, -2.0);

    Value absY = buildAbs(rewriter, loc, y);
    Value mod = rewriter.create<arith::RemFOp>(loc, absY, two);
    Value mul = rewriter.create<arith::MulFOp>(loc, mod, negTwo);
    Value add = rewriter.create<arith::AddFOp>(loc, mul, one);
    return add;
  }

  Value buildNegativeCompute(PatternRewriter &rewriter, Location loc, Value base, Value exponent) const {
    Value absX = buildAbs(rewriter, loc, base);
    Value logAbsX = buildLog(rewriter, loc, absX);
    Value mul = rewriter.create<arith::MulFOp>(loc, exponent, logAbsX);
    Value expVal = buildExp(rewriter, loc, mul);
    Value coef = buildNegOnePowY(rewriter, loc, exponent);
    return rewriter.create<arith::MulFOp>(loc, expVal, coef);
  }

  Value buildPositiveCompute(PatternRewriter &rewriter, Location loc, Value base, Value exponent) const {
    Value absX = buildAbs(rewriter, loc, base);
    Value logAbsX = buildLog(rewriter, loc, absX);
    Value mul = rewriter.create<arith::MulFOp>(loc, exponent, logAbsX);
    return buildExp(rewriter, loc, mul);
  }

  Value buildBoundaryCondForOne(PatternRewriter &rewriter, Location loc, Value base, Value exponent) const {
    auto fTy = cast<FloatType>(base.getType());

    Value absX = buildAbs(rewriter, loc, base);
    Value one = buildFloatConst(rewriter, loc, fTy, 1.0);
    Value mask0 = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, absX, one);

    Value absY = buildAbs(rewriter, loc, exponent);
    Value inf = buildInfConst(rewriter, loc, fTy);
    Value mask1 = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, absY, inf);

    return rewriter.create<arith::AndIOp>(loc, mask0, mask1);
  }

  Value buildIsPowNanResult(PatternRewriter &rewriter, Location loc, Value base, Value exponent) const {
    auto fTy = cast<FloatType>(base.getType());
    Value zero = buildFloatConst(rewriter, loc, fTy, 0.0);

    Value isNeg = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, base, zero);
    Value isXFinite = buildIsFinite(rewriter, loc, base);
    Value mask1 = rewriter.create<arith::AndIOp>(loc, isNeg, isXFinite);

    Value isYFinite = buildIsFinite(rewriter, loc, exponent);
    Value isYInteger = buildIsInteger(rewriter, loc, exponent);

    auto i1Ty = isYInteger.getType();
    auto one = rewriter.create<arith::ConstantIntOp>(loc, 1, i1Ty);
    Value notYInteger = rewriter.create<arith::XOrIOp>(loc, isYInteger, one);

    Value mask2 = rewriter.create<arith::AndIOp>(loc, isYFinite, notYInteger);

    return rewriter.create<arith::AndIOp>(loc, mask1, mask2);
  }

  Value buildIsZeroPowZeroCond(PatternRewriter &rewriter, Location loc, Value exponent) const {
    auto fTy = cast<FloatType>(exponent.getType());
    Value zero = buildFloatConst(rewriter, loc, fTy, 0.0);
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, exponent, zero);
  }

  Value buildPowerByMul(PatternRewriter &rewriter, Location loc, Value base, int exponent) const {
    assert(exponent >= 1 && "exponent must be >= 1");
    if (exponent == 1) {
      return base;
    }
    Value result = base;
    for (int i = 1; i < exponent; ++i) {
      result = rewriter.create<arith::MulFOp>(loc, result, base);
    }
    return result;
  }

  LogicalResult tryNormalizeCstExponent(PatternRewriter &rewriter, Location loc, math::PowFOp op, Value base,
                                        Value exponent) const {
    auto cstOpt = getConstantFloat(exponent);
    if (!cstOpt.has_value()) {
      return failure();
    }

    llvm::APFloat val = cstOpt.value();
    auto fTy = cast<FloatType>(base.getType());

    if (val.isZero()) {
      Value one = buildFloatConst(rewriter, loc, fTy, 1.0);
      rewriter.replaceOp(op, one);
      return success();
    }

    llvm::APFloat half(val.getSemantics(), "5e-1");
    if (val.compare(half) == llvm::APFloat::cmpEqual) {
      Value res = buildSqrt(rewriter, loc, base);
      rewriter.replaceOp(op, res);
      return success();
    }

    double asDouble = val.convertToDouble();
    double asRound = std::round(asDouble);
    const int upperLimit = 3;
    if (val.isInteger()) {
      auto iVal = static_cast<int64_t>(asRound);
      if (iVal >= 1 && iVal <= upperLimit) {
        Value res = buildPowerByMul(rewriter, loc, base, static_cast<int>(iVal));
        rewriter.replaceOp(op, res);
        return success();
      }
    }

    return failure();
  }

  LogicalResult normalizePowf(PatternRewriter &rewriter, math::PowFOp op) const {
    Value base = op.getLhs();
    Value exponent = op.getRhs();
    Location loc = op.getLoc();

    auto fTy = dyn_cast<FloatType>(base.getType());
    if (!fTy) {
      return failure();
    }

    if (succeeded(tryNormalizeCstExponent(rewriter, loc, op, base, exponent))) {
      return success();
    }

    Value negCond = buildIsNegCondition(rewriter, loc, base, exponent);
    Value negRes = buildNegativeCompute(rewriter, loc, base, exponent);
    Value posRes = buildPositiveCompute(rewriter, loc, base, exponent);

    Value partial0 = rewriter.create<arith::SelectOp>(loc, negCond, negRes, posRes);

    Value boundaryCond = buildBoundaryCondForOne(rewriter, loc, base, exponent);
    Value one = buildFloatConst(rewriter, loc, fTy, 1.0);
    Value partial1 = rewriter.create<arith::SelectOp>(loc, boundaryCond, one, partial0);

    Value constNaN = buildNaNConst(rewriter, loc, fTy);
    Value isNanCond = buildIsPowNanResult(rewriter, loc, base, exponent);
    Value partial2 = rewriter.create<arith::SelectOp>(loc, isNanCond, constNaN, partial1);

    Value isZeroPowZeroCond = buildIsZeroPowZeroCond(rewriter, loc, exponent);
    Value partial3 = rewriter.create<arith::SelectOp>(loc, isZeroPowZeroCond, one, partial2);

    rewriter.replaceOp(op, partial3);
    return success();
  }

  LogicalResult matchAndRewrite(math::PowFOp op, PatternRewriter &rewriter) const override {
    auto fTy = dyn_cast<FloatType>(op.getType());
    if (!fTy) {
      return failure();
    }
    return normalizePowf(rewriter, op);
  }
};

// rsqrt(x) = 1 / sqrt(x)
struct NormalizeRSqrtOp : public OpRewritePattern<math::RsqrtOp> {
  using OpRewritePattern<math::RsqrtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::RsqrtOp op, PatternRewriter &rewriter) const override {
    Value input = op.getOperand();
    auto inTy = getElementTypeOrSelf(input.getType());
    auto fTy = dyn_cast<FloatType>(inTy);
    if (!fTy || (!fTy.isF16() && !fTy.isF32())) {
      return failure();
    }

    Location loc = op.getLoc();

    Value sqrtVal = rewriter.create<math::SqrtOp>(loc, input);
    Value one = buildFloatConst(rewriter, loc, fTy, 1.0);
    Value res = rewriter.create<arith::DivFOp>(loc, one, sqrtVal);

    rewriter.replaceOp(op, res);
    return success();
  }
};

}  // namespace

void populateNormalizeMathPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizePowfOp>(patterns.getContext());
  patterns.add<NormalizeSinOp>(patterns.getContext());
  patterns.add<NormalizeCosOp>(patterns.getContext());
  patterns.add<NormalizeTanhOp>(patterns.getContext());
  patterns.add<NormalizeRSqrtOp>(patterns.getContext());
}

namespace {

struct NormalizePass : public impl::NormalizeBase<NormalizePass> {
 public:
  void runOnOperation() final {
    getContext().getOrLoadDialect<mlir::hivm::HIVMDialect>();

    RewritePatternSet patterns(&getContext());
    populateNormalizeMathPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createNormalizePass() { return std::make_unique<NormalizePass>(); }

}  // namespace mlir
