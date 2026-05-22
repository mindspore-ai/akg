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

#include <optional>

#include "mfusion/Conversion/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace mlir {

namespace TorchD = torch::Torch;

namespace {

static std::optional<int64_t> getConstInt(Value v) {
  if (!v) {
    return std::nullopt;
  }
  if (auto cst = v.getDefiningOp<TorchD::ConstantIntOp>()) {
    return cst.getValueAttr().getInt();
  }
  return std::nullopt;
}

static std::optional<int64_t> getKnownLastDim(Value v) {
  auto tensorTy = mlir::dyn_cast<TorchD::BaseTensorType>(v.getType());
  if (!tensorTy || !tensorTy.hasSizes()) {
    return std::nullopt;
  }
  auto sizes = tensorTy.getSizes();
  if (sizes.empty()) {
    return std::nullopt;
  }
  int64_t last = sizes.back();
  if (last == TorchD::kUnknownSize) {
    return std::nullopt;
  }
  return last;
}

static bool isLastDimIndex(Value v, int64_t dim) {
  auto tensorTy = mlir::dyn_cast<TorchD::BaseTensorType>(v.getType());
  if (!tensorTy || !tensorTy.hasSizes()) {
    return dim == -1;
  }
  int64_t rank = static_cast<int64_t>(tensorTy.getSizes().size());
  if (rank <= 0) {
    return dim == -1;
  }
  return dim == -1 || dim == (rank - 1);
}

struct RotateHalfCatOperands {
  Value negVal;
  Value leftVal;
};

struct RotateHalfSlices {
  Value base;
  TorchD::AtenSliceTensorOp left;
  TorchD::AtenSliceTensorOp right;
};

static std::optional<RotateHalfCatOperands> matchRotateHalfCatOperands(Value v) {
  auto catOp = v.getDefiningOp<TorchD::AtenCatOp>();
  if (!catOp) {
    return std::nullopt;
  }
  auto catDim = getConstInt(catOp.getDim());
  if (!catDim || *catDim != -1) {
    return std::nullopt;
  }

  auto listOp = catOp.getTensors().getDefiningOp<TorchD::PrimListConstructOp>();
  if (!listOp || listOp->getNumOperands() != 2) {
    return std::nullopt;
  }

  return RotateHalfCatOperands{listOp->getOperand(0), listOp->getOperand(1)};
}

static std::optional<RotateHalfSlices> matchRotateHalfSlices(const RotateHalfCatOperands &operands) {
  auto negOp = operands.negVal.getDefiningOp<TorchD::AtenNegOp>();
  if (!negOp) {
    return std::nullopt;
  }

  Value rightVal = negOp.getSelf();
  auto sliceRight = rightVal.getDefiningOp<TorchD::AtenSliceTensorOp>();
  auto sliceLeft = operands.leftVal.getDefiningOp<TorchD::AtenSliceTensorOp>();
  if (!sliceRight || !sliceLeft) {
    return std::nullopt;
  }
  Value xRightBase = sliceRight.getSelf();
  Value xLeftBase = sliceLeft.getSelf();
  if (xRightBase != xLeftBase) {
    return std::nullopt;
  }

  return RotateHalfSlices{xLeftBase, sliceLeft, sliceRight};
}

static bool isLastDimSlice(Value base, TorchD::AtenSliceTensorOp sliceOp) {
  auto dim = getConstInt(sliceOp.getDim());
  return dim && isLastDimIndex(base, *dim);
}

static bool hasUnitStep(TorchD::AtenSliceTensorOp sliceOp) {
  auto step = getConstInt(sliceOp.getStep());
  return step && *step == 1;
}

static bool hasConstRange(TorchD::AtenSliceTensorOp sliceOp, int64_t expectedStart, int64_t expectedEnd) {
  auto start = getConstInt(sliceOp.getStart());
  auto end = getConstInt(sliceOp.getEnd());
  return start && end && *start == expectedStart && *end == expectedEnd;
}

static bool hasRotateHalfSliceParams(const RotateHalfSlices &slices) {
  if (!isLastDimSlice(slices.base, slices.left) || !isLastDimSlice(slices.base, slices.right)) {
    return false;
  }
  if (!hasUnitStep(slices.left) || !hasUnitStep(slices.right)) {
    return false;
  }

  auto lastDim = getKnownLastDim(slices.base);
  if (!lastDim || (*lastDim % 2) != 0) {
    return false;
  }

  int64_t half = *lastDim / 2;
  // Use the canonical torch constant for max end.
  constexpr int64_t kEndMax = 9223372036854775807LL;
  return hasConstRange(slices.left, 0, half) && hasConstRange(slices.right, half, kEndMax);
}

// Match rotate_half(x) in the common cat/list form:
//   x_left  = slice(x, 0:D/2)
//   x_right = slice(x, D/2:...)
//   rot = cat([neg(x_right), x_left], dim=-1)
static Value matchRotateHalfCat(Value v) {
  auto operands = matchRotateHalfCatOperands(v);
  if (!operands) {
    return Value();
  }

  auto slices = matchRotateHalfSlices(*operands);
  if (!slices || !hasRotateHalfSliceParams(*slices)) {
    return Value();
  }

  return slices->base;
}

struct RoPEMatchState {
  TorchD::AtenAddTensorOp addOp;
  TorchD::AtenMulTensorOp cosMulOp;
  TorchD::AtenMulTensorOp sinMulOp;
  Value x;
  Value cos;
  Value sin;
};

static LogicalResult matchRoPE(TorchD::AtenAddTensorOp addOp, RoPEMatchState &state, PatternRewriter &rewriter) {
  auto alpha = getConstInt(addOp.getAlpha());
  if (!alpha || *alpha != 1) {
    return rewriter.notifyMatchFailure(addOp, "RoPE requires add alpha == 1");
  }

  auto lhsMul = addOp.getSelf().getDefiningOp<TorchD::AtenMulTensorOp>();
  auto rhsMul = addOp.getOther().getDefiningOp<TorchD::AtenMulTensorOp>();
  if (!lhsMul || !rhsMul) {
    return rewriter.notifyMatchFailure(addOp, "add inputs must come from torch.aten.mul.Tensor");
  }

  auto tryAssign = [&](TorchD::AtenMulTensorOp cosMul, TorchD::AtenMulTensorOp sinMul) -> LogicalResult {
    Value rotInput = sinMul.getSelf();
    Value sinCandidate = sinMul.getOther();
    Value xFromRot = matchRotateHalfCat(rotInput);
    if (!xFromRot) {
      rotInput = sinMul.getOther();
      sinCandidate = sinMul.getSelf();
      xFromRot = matchRotateHalfCat(rotInput);
    }
    if (!xFromRot) {
      return rewriter.notifyMatchFailure(addOp, "rotate_half(x) not matched");
    }

    Value a = cosMul.getSelf();
    Value b = cosMul.getOther();
    Value aBase = a;
    Value bBase = b;
    Value xBase = xFromRot;

    Value xVal;
    Value cosVal;
    if (aBase == xBase) {
      xVal = a;
      cosVal = b;
    } else if (bBase == xBase) {
      xVal = b;
      cosVal = a;
    } else {
      return rewriter.notifyMatchFailure(addOp, "x in cos mul must match rotate_half input");
    }

    state.addOp = addOp;
    state.cosMulOp = cosMul;
    state.sinMulOp = sinMul;
    state.x = xVal;
    state.cos = cosVal;
    state.sin = sinCandidate;
    return success();
  };

  if (succeeded(tryAssign(lhsMul, rhsMul)) || succeeded(tryAssign(rhsMul, lhsMul))) {
    return success();
  }
  return failure();
}

class TorchFuseRoPEPattern : public OpRewritePattern<TorchD::AtenAddTensorOp> {
 public:
  using OpRewritePattern<TorchD::AtenAddTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TorchD::AtenAddTensorOp op, PatternRewriter &rewriter) const override {
    RoPEMatchState state;
    if (failed(matchRoPE(op, state, rewriter))) {
      return failure();
    }

    // Skip fusion when RoPE input x is bf16 (ACL npu_rotary_mul numeric drift vs eager).
    // Decision uses x's element type only; cos/sin are assumed aligned in practice.
    auto xTensorTy = mlir::dyn_cast<TorchD::BaseTensorType>(state.x.getType());
    if (xTensorTy && xTensorTy.getOptionalDtype()) {
      auto dtype = xTensorTy.getOptionalDtype();
      if (dtype.isBF16()) {
        return rewriter.notifyMatchFailure(op, "bf16: skip fusion to avoid ACL numeric drift");
      }
    }

    rewriter.replaceOpWithNewOp<TorchD::OperatorOp>(
      op, op.getResult().getType(), rewriter.getStringAttr("torch.npu.npu_rotary_mul"),
      SmallVector<Value>{state.x, state.cos, state.sin}, /*numResults=*/0);
    return success();
  }
};

struct TorchFuseRoPEPass : public PassWrapper<TorchFuseRoPEPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "torch-fuse-rope"; }
  StringRef getDescription() const final {
    return "Fuse decomposed RoPE into torch.npu.npu_rotary_mul (skipped when x is bf16)";
  }

  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<TorchD::TorchDialect>(); }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TorchFuseRoPEPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createTorchFuseRoPEPass() { return std::make_unique<TorchFuseRoPEPass>(); }

}  // namespace mlir
