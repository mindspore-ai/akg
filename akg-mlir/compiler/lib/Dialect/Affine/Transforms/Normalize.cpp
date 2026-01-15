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

namespace mlir {

#define GEN_PASS_DEF_NORMALIZE
#define GEN_PASS_DECL_NORMALIZE
#include "akg/Dialect/Affine/Passes.h.inc"

#define DEBUG_TYPE "math-normalize"

namespace {

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
                                            /*Negative=*/false);
  return rewriter.create<arith::ConstantOp>(loc, fTy, rewriter.getFloatAttr(fTy, val));
}

static std::optional<llvm::APFloat> getConstantFloat(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto fAttr = dyn_cast<FloatAttr>(cst.getValue())) return fAttr.getValue();
  }
  return std::nullopt;
}

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
    if (exponent == 1) return base;
    Value result = base;
    for (int i = 1; i < exponent; ++i) result = rewriter.create<arith::MulFOp>(loc, result, base);
    return result;
  }

  LogicalResult tryNormalizeCstExponent(PatternRewriter &rewriter, Location loc, math::PowFOp op, Value base,
                                        Value exponent) const {
    auto cstOpt = getConstantFloat(exponent);
    if (!cstOpt.has_value()) return failure();

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
      int64_t iVal = static_cast<int64_t>(asRound);
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
    if (!fTy) return failure();

    if (succeeded(tryNormalizeCstExponent(rewriter, loc, op, base, exponent))) return success();

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
    if (!fTy) return failure();
    return normalizePowf(rewriter, op);
  }
};

}  // namespace

void populateNormalizeMathPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizePowfOp>(patterns.getContext());
}

namespace {

struct NormalizePass : public impl::NormalizeBase<NormalizePass> {
 public:
  void runOnOperation() final {
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
