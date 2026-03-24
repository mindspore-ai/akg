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

#include "mfusion/Conversion/Passes.h"

#include <optional>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace mlir {

namespace TorchD = torch::Torch;

namespace {

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

struct MatchState {
  TorchD::AtenMulTensorOp outMul;
  TorchD::AtenMulTensorOp normMul;
  TorchD::AtenRsqrtOp rsqrt;
  TorchD::AtenAddScalarOp add;
  TorchD::AtenMeanDimOp mean;
  TorchD::AtenPowTensorScalarOp pow;
  TorchD::OperatorOp dtypeCastBefore;
  TorchD::OperatorOp dtypeCastAfter;
  Value x;
  Value gamma;
  Value eps;
  /// Dim list for mean.dim (full path only); used to erase prim.list after fusion.
  Value dims;
};

/// Full path: mean.dim(pow(x,2)). Simplified: add(variance,eps) with variance not from mean.dim.
static LogicalResult matchVarianceSource(TorchD::AtenMulTensorOp outMul, TorchD::AtenAddScalarOp add,
                                         Value normX, PatternRewriter &rewriter, TorchD::AtenMeanDimOp &meanOut,
                                         TorchD::AtenPowTensorScalarOp &powOut, Value &dimsForErase) {
  meanOut = stripCasts(add.getSelf()).getDefiningOp<TorchD::AtenMeanDimOp>();
  powOut = nullptr;
  dimsForErase = Value();

  if (meanOut) {
    powOut = stripCasts(meanOut.getSelf()).getDefiningOp<TorchD::AtenPowTensorScalarOp>();
    if (!powOut) {
      return rewriter.notifyMatchFailure(outMul, "mean input is not pow.Tensor_Scalar");
    }
    auto power = getConstInt(powOut.getExponent());
    if (!power || *power != 2) {
      return rewriter.notifyMatchFailure(outMul, "pow exponent must be 2");
    }
    if (stripCasts(powOut.getSelf()) != stripCasts(normX)) {
      return rewriter.notifyMatchFailure(outMul, "pow input does not match norm x");
    }
    if (!getConstBool(meanOut.getKeepdim()).has_value()) {
      return rewriter.notifyMatchFailure(outMul, "keepdim must be constant");
    }
    dimsForErase = meanOut.getDim();
    return success();
  }

  // Backward-compat: variance not defined by mean.dim in this region (e.g. block args).
  // npu_rms_norm recomputes from x; fusion is only sound when variance matches x's second moment.
  Value variance = stripCasts(add.getSelf());
  auto vType = dyn_cast<TorchD::ValueTensorType>(variance.getType());
  auto xType = dyn_cast<TorchD::ValueTensorType>(stripCasts(normX).getType());
  if (!vType || !xType || vType.getDtype() != xType.getDtype()) {
    return rewriter.notifyMatchFailure(
        outMul, "simplified pattern: variance and x must be vtensor types with same dtype");
  }
  if (!variance.hasOneUse()) {
    return rewriter.notifyMatchFailure(outMul,
                                        "simplified pattern: add.Scalar self must have a single use (the add)");
  }
  return success();
}

static void resolveDtypeCasts(TorchD::AtenPowTensorScalarOp pow, Value normX, TorchD::AtenMulTensorOp normMul,
                              TorchD::OperatorOp &dtypeCastBefore, TorchD::OperatorOp &dtypeCastAfter) {
  dtypeCastBefore = nullptr;
  dtypeCastAfter = nullptr;

  if (pow) {
    if (auto castOp = pow.getSelf().getDefiningOp<TorchD::OperatorOp>()) {
      if (isNpuDtypeCast(castOp)) {
        dtypeCastBefore = castOp;
      }
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

static LogicalResult matchRmsNorm(TorchD::AtenMulTensorOp outMul, MatchState &st, PatternRewriter &rewriter) {
  // Match: pow(x,2)->mean->add->rsqrt->mul(x,rsqrt)->mul(gamma), optional npu_dtype_casts,
  // or simplified add(variance,eps)->rsqrt->mul->mul.

  Value a = outMul.getSelf();
  Value b = outMul.getOther();
  Value aBase = stripCasts(a);
  Value bBase = stripCasts(b);

  auto aMul = aBase.getDefiningOp<TorchD::AtenMulTensorOp>();
  auto bMul = bBase.getDefiningOp<TorchD::AtenMulTensorOp>();
  TorchD::AtenMulTensorOp normMul;
  Value gamma;

  if (aMul && !bMul) {
    normMul = aMul;
    gamma = b;
  } else if (bMul && !aMul) {
    normMul = bMul;
    gamma = a;
  } else if (aMul && bMul) {
    return rewriter.notifyMatchFailure(outMul, "ambiguous: both operands are mul.Tensor");
  } else {
    return rewriter.notifyMatchFailure(outMul, "no mul.Tensor feeding output mul");
  }

  Value nA = normMul.getSelf();
  Value nB = normMul.getOther();
  Value nABase = stripCasts(nA);
  Value nBBase = stripCasts(nB);

  auto rsA = nABase.getDefiningOp<TorchD::AtenRsqrtOp>();
  auto rsB = nBBase.getDefiningOp<TorchD::AtenRsqrtOp>();
  TorchD::AtenRsqrtOp rsqrt;
  Value x;

  if (rsA && !rsB) {
    rsqrt = rsA;
    x = nB;
  } else if (rsB && !rsA) {
    rsqrt = rsB;
    x = nA;
  } else if (rsA && rsB) {
    return rewriter.notifyMatchFailure(outMul, "ambiguous: both operands are rsqrt");
  } else {
    return rewriter.notifyMatchFailure(outMul, "no rsqrt feeding norm mul");
  }

  auto add = stripCasts(rsqrt.getSelf()).getDefiningOp<TorchD::AtenAddScalarOp>();
  if (!add) {
    return rewriter.notifyMatchFailure(outMul, "rsqrt input is not add.Scalar");
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
  Value dimsForErase;
  if (failed(matchVarianceSource(outMul, add, x, rewriter, mean, pow, dimsForErase))) {
    return failure();
  }

  if (!normMul.getResult().hasOneUse()) {
    return rewriter.notifyMatchFailure(outMul, "normalized value has multiple uses");
  }

  TorchD::OperatorOp dtypeCastBefore;
  TorchD::OperatorOp dtypeCastAfter;
  resolveDtypeCasts(pow, x, normMul, dtypeCastBefore, dtypeCastAfter);

  st.outMul = outMul;
  st.normMul = normMul;
  st.rsqrt = rsqrt;
  st.add = add;
  st.mean = mean;
  st.pow = pow;
  st.x = x;
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
//   wouldOpBeTriviallyDead / canonicalizer DCE often keeps pow/mean/add/rsqrt/mul
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
  eraseIfDead(st.add);
  if (st.mean) {
    eraseIfDead(st.mean);
  }
  if (st.pow) {
    eraseIfDead(st.pow);
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

    SmallVector<Type> resultTypes = {st.outMul.getResult().getType(), st.rsqrt.getResult().getType()};
    Value fusedX = st.dtypeCastBefore ? st.dtypeCastBefore.getOperand(0) : st.x;
    SmallVector<Value> operands = {fusedX, st.gamma, st.eps};

    rewriter.setInsertionPoint(st.rsqrt);
    auto fused = rewriter.create<TorchD::OperatorOp>(st.rsqrt.getLoc(), resultTypes,
                                                       rewriter.getStringAttr("torch.npu.npu_rms_norm"), operands,
                                                       /*numRegions=*/0);

    rewriter.replaceOp(st.outMul, fused.getResult(0));

    if (!st.rsqrt.getResult().hasOneUse()) {
      st.rsqrt.getResult().replaceAllUsesWith(fused.getResult(1));
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
