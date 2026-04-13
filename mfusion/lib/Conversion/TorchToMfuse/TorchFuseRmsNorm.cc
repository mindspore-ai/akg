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

#include <cassert>
#include <cmath>
#include <optional>
#include "mfusion/Conversion/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace mlir {

namespace TorchD = torch::Torch;

namespace {

// -----------------------------------------------------------------------------
// torch-fuse-rms-norm — decomposed RMSNorm shapes this pass fuses into
//   torch.operator "torch.npu.npu_rms_norm"
//
// Root: aten.mul.Tensor( gamma, norm )  or  aten.mul.Tensor( norm, gamma ),  where
//   norm = aten.mul.Tensor(x, scale).
//
// Dataflow:  scale (rstd)  ->  norm = mul(x, scale)  ->  out = mul(norm, gamma).
//   scale comes from exactly one of the inverse-scale cases in [3].
//
// [1] Variance (one of):
//     - mean.dim(pow.Tensor_Scalar(x, 2), dim, keepdim=*, ...)  exponent 2, pow self == x
//       (after stripCasts on the norm-mul x path).
//     - mean.dim(mul.Tensor(x, x), ...)  with both mul operands == x (after stripCasts).
//
// [2] Stabilization:
//     add.Scalar(mean, eps, alpha)  with  alpha == 1  and  eps  constant float;
//     add.self is the mean; add result feeds the inverse-scale op (casts allowed).
//
// [3] Inverse scale (one of):
//     - aten.rsqrt(add)
//     - reciprocal(aten.sqrt(add))
//     - pow.Tensor_Scalar(add, -0.5)  with constant exponent -0.5
//     - div.Tensor(numerator, sqrt(add)): denominator is sqrt(add); numerator is
//       ones_like(sqrt_out)  or  full_like(sqrt_out, 1)  tied to that sqrt result.
//
// [4] Optional: torch.npu._npu_dtype_cast  on variance input and/or single-use cast
//     on norm output; matching strips casts and records them for fused operand wiring.
// -----------------------------------------------------------------------------

static bool isNpuDtypeCast(TorchD::OperatorOp op) {
  if (!op || !op.getNameAttr()) {
    return false;
  }
  return op.getNameAttr().getValue() == "torch.npu._npu_dtype_cast";
}

static Value stripCasts(Value v) {
  while (true) {
    auto op = v.getDefiningOp<TorchD::OperatorOp>();
    // Dtype casts are common in mixed-precision RmsNorm; allow stripping even
    // when the cast result has multiple uses (we only use this for matching).
    if (isNpuDtypeCast(op) && op.getNumOperands() >= 1) {
      v = op.getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

static std::optional<int64_t> getConstInt(Value v) {
  v = stripCasts(v);
  if (auto cst = v.getDefiningOp<TorchD::ConstantIntOp>()) {
    return cst.getValueAttr().getInt();
  }
  return std::nullopt;
}

static std::optional<double> getConstFloat(Value v) {
  v = stripCasts(v);
  if (auto cst = v.getDefiningOp<TorchD::ConstantFloatOp>()) {
    return cst.getValue().convertToDouble();
  }
  return std::nullopt;
}

static std::optional<bool> getConstBool(Value v) {
  v = stripCasts(v);
  if (auto cst = v.getDefiningOp<TorchD::ConstantBoolOp>()) {
    return cst.getValue();
  }
  return std::nullopt;
}

/// Exponent for aten.pow.Tensor_Scalar (int or float constant).
static std::optional<double> getPowTensorScalarExponent(Value exp) {
  exp = stripCasts(exp);
  if (auto cst = exp.getDefiningOp<TorchD::ConstantFloatOp>()) {
    return cst.getValue().convertToDouble();
  }
  if (auto cst = exp.getDefiningOp<TorchD::ConstantIntOp>()) {
    return static_cast<double>(cst.getValueAttr().getInt());
  }
  return std::nullopt;
}

/// True if exponent is -0.5 (pow(x,-0.5) == rsqrt(x) for positive x).
static bool isNegHalfExponent(double e) { return std::abs(e + 0.5) < 1e-12; }

static bool isOnePointOScalar(Value v) {
  auto f = getConstFloat(v);
  if (f && std::abs(*f - 1.0) < 1e-9) {
    return true;
  }
  auto i = getConstInt(v);
  return i && *i == 1;
}

/// Populated when matchRmsNorm succeeds; fields reflect which branch of the catalog applied.
struct MatchState {
  TorchD::AtenMulTensorOp outMul;
  TorchD::AtenMulTensorOp normMul;
  /// Exactly one inverse-scale group is non-null: rsqrt XOR (reciprocal+sqrt) XOR invPow XOR div(ones,sqrt).
  TorchD::AtenRsqrtOp rsqrt;
  TorchD::AtenReciprocalOp reciprocal;
  TorchD::AtenSqrtOp sqrtForRstd;
  TorchD::AtenPowTensorScalarOp invPow;
  TorchD::AtenDivTensorOp divOneOverSqrt;
  TorchD::AtenOnesLikeOp onesLikeNumerator;
  TorchD::AtenFullLikeOp fullLikeNumerator;
  TorchD::AtenAddScalarOp add;
  TorchD::AtenMeanDimOp mean;
  /// Variance: pow(x,2) XOR mul(x,x)
  TorchD::AtenPowTensorScalarOp pow;
  TorchD::AtenMulTensorOp varMul;
  TorchD::OperatorOp dtypeCastBefore;
  TorchD::OperatorOp dtypeCastAfter;
  Value x;
  Value gamma;
  Value eps;
  /// Dim list for mean.dim (full path only); used to erase prim.list after fusion.
  Value dims;
};

/// Full path: mean.dim(pow(x,2)) or mean.dim(mul(x,x))
static LogicalResult matchVarianceSource(TorchD::AtenMulTensorOp outMul, TorchD::AtenAddScalarOp add, Value normX,
                                         PatternRewriter &rewriter, TorchD::AtenMeanDimOp &meanOut,
                                         TorchD::AtenPowTensorScalarOp &powOut, TorchD::AtenMulTensorOp &varMulOut,
                                         Value &dimsForErase) {
  meanOut = stripCasts(add.getSelf()).getDefiningOp<TorchD::AtenMeanDimOp>();
  powOut = nullptr;
  varMulOut = nullptr;
  dimsForErase = Value();

  if (!meanOut) {
    return rewriter.notifyMatchFailure(outMul, "add input is not mean.dim");
  }

  Value meanInput = stripCasts(meanOut.getSelf());

  if (auto p = meanInput.getDefiningOp<TorchD::AtenPowTensorScalarOp>()) {
    auto power = getConstInt(p.getExponent());
    if (!power || *power != 2) {
      return rewriter.notifyMatchFailure(outMul, "pow exponent must be 2");
    }
    if (stripCasts(p.getSelf()) != stripCasts(normX)) {
      return rewriter.notifyMatchFailure(outMul, "pow input does not match norm x");
    }
    powOut = p;
  } else if (auto mm = meanInput.getDefiningOp<TorchD::AtenMulTensorOp>()) {
    if (stripCasts(mm.getSelf()) != stripCasts(normX) || stripCasts(mm.getOther()) != stripCasts(normX)) {
      return rewriter.notifyMatchFailure(outMul, "mean input is not mul(x,x) with same x");
    }
    varMulOut = mm;
  } else {
    return rewriter.notifyMatchFailure(outMul, "mean input is not pow.Tensor_Scalar^2 or mul(x,x)");
  }

  if (!getConstBool(meanOut.getKeepdim()).has_value()) {
    return rewriter.notifyMatchFailure(outMul, "keepdim must be constant");
  }
  dimsForErase = meanOut.getDim();
  return success();
}

static void resolveDtypeCastsFromVarianceInput(Value varianceInput, Value normX,
                                               TorchD::AtenMulTensorOp normMul, TorchD::OperatorOp &dtypeCastBefore,
                                               TorchD::OperatorOp &dtypeCastAfter) {
  dtypeCastBefore = nullptr;
  dtypeCastAfter = nullptr;

  if (auto castOp = varianceInput.getDefiningOp<TorchD::OperatorOp>()) {
    if (isNpuDtypeCast(castOp)) {
      dtypeCastBefore = castOp;
    }
  } else if (auto castOp = normX.getDefiningOp<TorchD::OperatorOp>()) {
    if (isNpuDtypeCast(castOp)) {
      dtypeCastBefore = castOp;
    }
  }

  if (normMul.getResult().hasOneUse()) {
    Operation *onlyUser = *normMul.getResult().getUsers().begin();
    if (auto castOp = dyn_cast<TorchD::OperatorOp>(onlyUser)) {
      if (isNpuDtypeCast(castOp)) {
        dtypeCastAfter = castOp;
      }
    }
  }
}

/// Value produced as the RMS inverse-scale (rstd) fed into mul with x; same semantic as rsqrt output.
/// Takes non-const `MatchState` because MLIR OpState accessors (`getResult`, `getOperation`, …) are
/// not const-qualified and fail to compile when the op is reached through `const MatchState &`.
static Value getRstdScaleValue(MatchState &st) {
  if (st.rsqrt) {
    return st.rsqrt.getResult();
  }
  if (st.reciprocal) {
    return st.reciprocal.getResult();
  }
  if (st.invPow) {
    return st.invPow.getResult();
  }
  if (st.divOneOverSqrt) {
    return st.divOneOverSqrt.getResult();
  }
  return {};
}

/// Holds one matched inverse-scale branch (rsqrt, reciprocal∘sqrt, pow(,-0.5), or div(ones,sqrt)).
struct InverseScaleCapture {
  TorchD::AtenRsqrtOp rsqrt = nullptr;
  TorchD::AtenReciprocalOp reciprocal = nullptr;
  TorchD::AtenSqrtOp sqrtForRstd = nullptr;
  TorchD::AtenPowTensorScalarOp invPow = nullptr;
  TorchD::AtenDivTensorOp divOneOverSqrt = nullptr;
  TorchD::AtenOnesLikeOp onesLikeNumerator = nullptr;
  TorchD::AtenFullLikeOp fullLikeNumerator = nullptr;
  Value x;
};

static void copyInverseScaleToMatchState(MatchState &st, const InverseScaleCapture &cap) {
  st.rsqrt = cap.rsqrt;
  st.reciprocal = cap.reciprocal;
  st.sqrtForRstd = cap.sqrtForRstd;
  st.invPow = cap.invPow;
  st.divOneOverSqrt = cap.divOneOverSqrt;
  st.onesLikeNumerator = cap.onesLikeNumerator;
  st.fullLikeNumerator = cap.fullLikeNumerator;
  st.x = cap.x;
}

static LogicalResult resolveNormMulGamma(TorchD::AtenMulTensorOp outMul, PatternRewriter &rewriter,
                                         TorchD::AtenMulTensorOp &normMul, Value &gamma) {
  Value a = outMul.getSelf();
  Value b = outMul.getOther();
  Value aBase = stripCasts(a);
  Value bBase = stripCasts(b);
  auto aMul = aBase.getDefiningOp<TorchD::AtenMulTensorOp>();
  auto bMul = bBase.getDefiningOp<TorchD::AtenMulTensorOp>();

  if (aMul && !bMul) {
    normMul = aMul;
    gamma = b;
    return success();
  }
  if (bMul && !aMul) {
    normMul = bMul;
    gamma = a;
    return success();
  }
  if (aMul && bMul) {
    return rewriter.notifyMatchFailure(outMul, "ambiguous: both operands are mul.Tensor");
  }
  return rewriter.notifyMatchFailure(outMul, "no mul.Tensor feeding output mul");
}

/// Non-const `InverseScaleCapture`: generated Torch ops' `getSelf()` are not const-qualified.
static TorchD::AtenAddScalarOp getAddFromInverseScaleCapture(InverseScaleCapture &c) {
  if (c.rsqrt) {
    return stripCasts(c.rsqrt.getSelf()).getDefiningOp<TorchD::AtenAddScalarOp>();
  }
  if (c.invPow) {
    return stripCasts(c.invPow.getSelf()).getDefiningOp<TorchD::AtenAddScalarOp>();
  }
  if (c.sqrtForRstd) {
    return stripCasts(c.sqrtForRstd.getSelf()).getDefiningOp<TorchD::AtenAddScalarOp>();
  }
  return nullptr;
}

static bool tryMatchRsqrt(Value scaleSide, Value xSide, InverseScaleCapture &cap) {
  if (auto r = scaleSide.getDefiningOp<TorchD::AtenRsqrtOp>()) {
    cap.rsqrt = r;
    cap.x = xSide;
    return true;
  }
  return false;
}

static bool tryMatchReciprocalSqrt(Value scaleSide, Value xSide, InverseScaleCapture &cap) {
  auto rec = scaleSide.getDefiningOp<TorchD::AtenReciprocalOp>();
  if (!rec) {
    return false;
  }
  auto sqrtOp = stripCasts(rec.getSelf()).getDefiningOp<TorchD::AtenSqrtOp>();
  if (!sqrtOp) {
    return false;
  }
  cap.reciprocal = rec;
  cap.sqrtForRstd = sqrtOp;
  cap.x = xSide;
  return true;
}

static bool tryMatchPowNegHalf(Value scaleSide, Value xSide, InverseScaleCapture &cap) {
  auto p = scaleSide.getDefiningOp<TorchD::AtenPowTensorScalarOp>();
  if (!p) {
    return false;
  }
  auto exp = getPowTensorScalarExponent(p.getExponent());
  if (!exp || !isNegHalfExponent(*exp)) {
    return false;
  }
  cap.invPow = p;
  cap.x = xSide;
  return true;
}

static bool divNumeratorIsOnesOrUnitFullLike(Value lhs, Value sqrtOut, InverseScaleCapture &cap,
                                               TorchD::AtenDivTensorOp divOp, TorchD::AtenSqrtOp sqrtOp) {
  lhs = stripCasts(lhs);
  if (auto ol = lhs.getDefiningOp<TorchD::AtenOnesLikeOp>()) {
    if (stripCasts(ol.getSelf()) != sqrtOut) {
      return false;
    }
    cap.onesLikeNumerator = ol;
    cap.fullLikeNumerator = nullptr;
  } else if (auto fl = lhs.getDefiningOp<TorchD::AtenFullLikeOp>()) {
    if (stripCasts(fl.getSelf()) != sqrtOut || !isOnePointOScalar(fl.getFillValue())) {
      return false;
    }
    cap.fullLikeNumerator = fl;
    cap.onesLikeNumerator = nullptr;
  } else {
    return false;
  }
  cap.divOneOverSqrt = divOp;
  cap.sqrtForRstd = sqrtOp;
  return true;
}

static bool tryMatchDivOnesOverSqrt(Value scaleSide, Value xSide, InverseScaleCapture &cap) {
  auto divOp = scaleSide.getDefiningOp<TorchD::AtenDivTensorOp>();
  if (!divOp) {
    return false;
  }
  Value rhs = stripCasts(divOp.getOther());
  auto sqrtOp = rhs.getDefiningOp<TorchD::AtenSqrtOp>();
  if (!sqrtOp) {
    return false;
  }
  Value lhs = stripCasts(divOp.getSelf());
  Value sqrtOut = sqrtOp.getResult();
  if (!divNumeratorIsOnesOrUnitFullLike(lhs, sqrtOut, cap, divOp, sqrtOp)) {
    return false;
  }
  cap.x = xSide;
  return true;
}

static bool matchInverseScaleOnSides(Value scaleSide, Value xSide, InverseScaleCapture &cap) {
  return tryMatchRsqrt(scaleSide, xSide, cap) || tryMatchReciprocalSqrt(scaleSide, xSide, cap) ||
         tryMatchPowNegHalf(scaleSide, xSide, cap) || tryMatchDivOnesOverSqrt(scaleSide, xSide, cap);
}

static LogicalResult matchInverseScaleToAdd(TorchD::AtenMulTensorOp outMul, Value nABase, Value nBBase, Value nA,
                                            Value nB, InverseScaleCapture &cap, TorchD::AtenAddScalarOp &add,
                                            PatternRewriter &rewriter) {
  if (!matchInverseScaleOnSides(nABase, nB, cap) && !matchInverseScaleOnSides(nBBase, nA, cap)) {
    return rewriter.notifyMatchFailure(
        outMul, "norm mul has no rsqrt / reciprocal(sqrt) / pow(,-0.5) / div(ones,sqrt) operand");
  }
  add = getAddFromInverseScaleCapture(cap);
  if (!add) {
    return rewriter.notifyMatchFailure(outMul, "inverse-scale input is not add.Scalar");
  }
  return success();
}

static LogicalResult matchRmsNorm(TorchD::AtenMulTensorOp outMul, MatchState &st, PatternRewriter &rewriter) {
  // Implements the pattern catalog in the file-level "torch-fuse-rms-norm" comment block.

  TorchD::AtenMulTensorOp normMul;
  Value gamma;
  if (failed(resolveNormMulGamma(outMul, rewriter, normMul, gamma))) {
    return failure();
  }

  Value nA = normMul.getSelf();
  Value nB = normMul.getOther();
  Value nABase = stripCasts(nA);
  Value nBBase = stripCasts(nB);

  InverseScaleCapture inv;
  TorchD::AtenAddScalarOp add;
  if (failed(matchInverseScaleToAdd(outMul, nABase, nBBase, nA, nB, inv, add, rewriter))) {
    return failure();
  }

  auto alpha = getConstInt(add.getAlpha());
  if (!alpha || *alpha != 1) {
    return rewriter.notifyMatchFailure(outMul, "add.Scalar alpha must be 1");
  }
  auto eps = add.getOther();
  if (!getConstFloat(eps).has_value()) {
    return rewriter.notifyMatchFailure(outMul, "eps must be torch.constant.float");
  }

  TorchD::AtenMeanDimOp mean;
  TorchD::AtenPowTensorScalarOp pow;
  TorchD::AtenMulTensorOp varMul;
  Value dimsForErase;
  if (failed(matchVarianceSource(outMul, add, inv.x, rewriter, mean, pow, varMul, dimsForErase))) {
    return failure();
  }

  if (!normMul.getResult().hasOneUse()) {
    return rewriter.notifyMatchFailure(outMul, "normalized value has multiple uses");
  }

  TorchD::OperatorOp dtypeCastBefore;
  TorchD::OperatorOp dtypeCastAfter;
  Value varianceInput = pow ? pow.getSelf() : varMul.getSelf();
  resolveDtypeCastsFromVarianceInput(varianceInput, inv.x, normMul, dtypeCastBefore, dtypeCastAfter);

  st.outMul = outMul;
  st.normMul = normMul;
  copyInverseScaleToMatchState(st, inv);
  st.add = add;
  st.mean = mean;
  st.pow = pow;
  st.varMul = varMul;
  st.gamma = stripCasts(gamma);
  st.eps = eps;
  st.dims = dimsForErase;
  st.dtypeCastBefore = dtypeCastBefore;
  st.dtypeCastAfter = dtypeCastAfter;

  return success();
}

// Explicitly erase the decomposed chain here instead of relying on canonicalize DCE:
// - The full Torch pipeline runs canonicalize after torch-fusion (see inductor.py:
//   torch-fusion,canonicalize), but many torch.aten.* ops in torch-mlir carry
//   RecursiveSideEffects or are not modeled as side-effect-free, so
//   wouldOpBeTriviallyDead / canonicalizer DCE often keeps pow/mean/add/rsqrt|inv-scale/mul
//   even when they have no users.
// - Lit tests that only run --torch-fuse-rms-norm do not append canonicalize; without
//   erasing here, dead nodes remain in the dumped IR.
// Erasing in this pass keeps the IR clean and stable regardless of torch-mlir effect
// metadata.
static void eraseDecomposedChain(const MatchState &st, PatternRewriter &rewriter) {
  auto eraseIfDead = [&](Operation *defOp) {
    if (defOp && defOp->use_empty()) {
      rewriter.eraseOp(defOp);
    }
  };

  if (st.dtypeCastAfter) {
    eraseIfDead(st.dtypeCastAfter);
  }
  eraseIfDead(st.normMul);
  eraseIfDead(st.rsqrt);
  eraseIfDead(st.reciprocal);
  eraseIfDead(st.invPow);
  eraseIfDead(st.divOneOverSqrt);
  eraseIfDead(st.onesLikeNumerator);
  eraseIfDead(st.fullLikeNumerator);
  eraseIfDead(st.sqrtForRstd);
  eraseIfDead(st.add);
  if (st.mean) {
    eraseIfDead(st.mean);
  }
  if (st.pow) {
    eraseIfDead(st.pow);
  }
  if (st.varMul) {
    eraseIfDead(st.varMul);
  }
  if (st.dtypeCastBefore) {
    eraseIfDead(st.dtypeCastBefore);
  }
  if (st.dims) {
    if (auto *listOp = st.dims.getDefiningOp()) {
      eraseIfDead(listOp);
    }
  }
}

class TorchFuseRmsNormPattern : public OpRewritePattern<TorchD::AtenMulTensorOp> {
 public:
  using OpRewritePattern<TorchD::AtenMulTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TorchD::AtenMulTensorOp op, PatternRewriter &rewriter) const override {
    MatchState st;
    if (failed(matchRmsNorm(op, st, rewriter))) {
      return failure();
    }

    Value scaleVal = getRstdScaleValue(st);
    assert(scaleVal && "torch-fuse-rms-norm: inverse-scale op must be set");
    SmallVector<Type> resultTypes = {st.outMul.getResult().getType(), scaleVal.getType()};
    Value fusedX = st.dtypeCastBefore ? st.dtypeCastBefore.getOperand(0) : st.x;
    SmallVector<Value> operands = {fusedX, st.gamma, st.eps};

    rewriter.setInsertionPoint(st.outMul);
    auto fused = rewriter.create<TorchD::OperatorOp>(st.outMul.getLoc(), resultTypes,
                                                     rewriter.getStringAttr("torch.npu.npu_rms_norm"), operands,
                                                     /*numRegions=*/0);

    rewriter.replaceOp(st.outMul, fused.getResult(0));

    if (!scaleVal.hasOneUse()) {
      scaleVal.replaceAllUsesWith(fused.getResult(1));
    }

    eraseDecomposedChain(st, rewriter);
    return success();
  }
};

struct TorchFuseRmsNormPass : public PassWrapper<TorchFuseRmsNormPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "torch-fuse-rms-norm"; }
  StringRef getDescription() const final {
    return "Fuse decomposed RmsNorm into torch.npu.npu_rms_norm on Torch dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<TorchD::TorchDialect>(); }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TorchFuseRmsNormPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createTorchFuseRmsNormPass() { return std::make_unique<TorchFuseRmsNormPass>(); }

}  // namespace mlir
