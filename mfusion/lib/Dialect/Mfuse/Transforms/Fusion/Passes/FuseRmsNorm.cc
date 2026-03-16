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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseRmsNorm.h"

#include <optional>

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSERMSNORM
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {

namespace {

constexpr double kRmsNormPowExponentTolerance = 1e-9;

/// Check if an operation is a type-system cast (builtin.unrealized_conversion_cast).
bool isCastOp(Operation *def) {
  if (!def) return false;
  return def->getName().getStringRef() == "builtin.unrealized_conversion_cast";
}

/// Check if an operation is a dtype cast via torch.operator "torch.npu._npu_dtype_cast".
/// In MLIR IR the registered op name is "torch.operator" and the actual operator name
/// is stored in the "name" string attribute.
bool isDtypeCastOp(Operation *def) {
  if (!def) return false;
  if (def->getName().getStringRef() == "torch.operator") {
    if (auto nameAttr = def->getAttrOfType<StringAttr>("name"))
      return nameAttr.getValue() == "torch.npu._npu_dtype_cast";
  }
  return false;
}

/// Check if an operation is any kind of cast we might need to look through,
/// including both type-system casts and dtype casts.
bool isAnyCastOp(Operation *def) {
  return isCastOp(def) || isDtypeCastOp(def);
}

/// Look through type-system cast operations to get the source value.
/// Only follows builtin.unrealized_conversion_cast with exactly one use.
Value getSourceThroughCast(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || !isCastOp(def) || !v.hasOneUse()) {
    return v;
  }
  return def->getOperand(0);
}

/// Look through type-system casts (unrealized_conversion_cast) recursively.
/// Does NOT follow through dtype casts (npu_dtype_cast).
Value getSourceThroughAllCasts(Value v) {
  while (true) {
    Operation *def = v.getDefiningOp();
    if (!def || !isCastOp(def)) {
      break;
    }
    v = def->getOperand(0);
  }
  return v;
}

/// Look through ALL cast operations recursively, including both type-system casts
/// and dtype casts (torch.operator "torch.npu._npu_dtype_cast").
/// Used to trace from gamma mul back to first mul across precision boundaries.
Value getSourceThroughAllCastsIncludingDtype(Value v) {
  while (true) {
    Operation *def = v.getDefiningOp();
    if (!def || !isAnyCastOp(def)) {
      break;
    }
    v = def->getOperand(0);
  }
  return v;
}

/// Try to get a constant float (epsilon) from a Value. Supports arith.constant
/// and torch.constant.float (by op name and "value" attribute).
std::optional<double> getConstantFloat(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    Attribute attr = cst.getValue();
    if (auto fa = dyn_cast<FloatAttr>(attr)) {
      return fa.getValueAsDouble();
    }
    if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
      if (dense.isSplat() &&
          (dense.getElementType().isF32() || dense.getElementType().isF64())) {
        return dense.getSplatValue<APFloat>().convertToDouble();
      }
    }
  }
  Operation *def = v.getDefiningOp();
  if (def && def->getName().getStringRef() == "torch.constant.float") {
    Attribute a = def->getAttr("value");
    if (FloatAttr fa = dyn_cast_or_null<FloatAttr>(a)) {
      return fa.getValueAsDouble();
    }
  }
  return std::nullopt;
}

/// Match torch.aten.add.Scalar(mean_result, eps, dim): operands 0=mean, 1=scalar, 2=dim.
/// Returns eps value if matched.
std::optional<double> matchTorchAddScalar(Value addOutput) {
  Value addInput = getSourceThroughCast(addOutput);
  Operation *addOp = addInput.getDefiningOp();
  if (!addOp || addOp->getName().getStringRef() != "torch.aten.add.Scalar") {
    return std::nullopt;
  }
  if (addOp->getNumOperands() < 2) {
    return std::nullopt;
  }
  return getConstantFloat(addOp->getOperand(1));
}

/// Match torch.aten.mean.dim. Returns the input (pow result) only when keepdim=true,
/// which is required for correct rstd shape (e.g., [1,512,1] for input [1,512,2560]).
Value matchTorchMeanDim(Operation *addOp) {
  if (!addOp || addOp->getNumOperands() < 4) {
    return Value();
  }
  Value meanInput = addOp->getOperand(0);
  Operation *meanOp = meanInput.getDefiningOp();
  if (!meanOp || meanOp->getName().getStringRef() != "torch.aten.mean.dim") {
    return Value();
  }
  if (meanOp->getNumOperands() < 3) {
    return Value();
  }
  Operation *keepdimDef = meanOp->getOperand(2).getDefiningOp();
  if (!keepdimDef || keepdimDef->getName().getStringRef() != "torch.constant.bool") {
    return Value();
  }
  auto keepdimAttr = keepdimDef->getAttrOfType<BoolAttr>("value");
  if (!keepdimAttr || !keepdimAttr.getValue()) {
    return Value();
  }
  return meanOp->getOperand(0);
}

/// Match torch.aten.pow.Tensor_Scalar. Returns the base (x) if exponent is 2.
Value matchTorchPowTensorScalar(Value powOutput) {
  Operation *powOp = powOutput.getDefiningOp();
  if (!powOp || powOp->getName().getStringRef() != "torch.aten.pow.Tensor_Scalar") {
    return Value();
  }
  if (powOp->getNumOperands() < 2) {
    return Value();
  }
  std::optional<double> expVal = getConstantFloat(powOp->getOperand(1));
  if (!expVal || std::abs(*expVal - 2.0) > kRmsNormPowExponentTolerance) {
    return Value();
  }
  return powOp->getOperand(0);
}

/// Result of matching the add op in the RmsNorm chain: epsilon and the "mean" operand.
struct EpsilonAndMean {
  double epsilon;
  Value meanOutput;
};

/// Match add op (torch.aten.add.Scalar, mfuse.add, or mfuse.aclnn.add) and extract constant epsilon + mean operand.
/// Returns nullopt if add is not supported or epsilon is not a constant.
std::optional<EpsilonAndMean> getEpsilonAndMeanFromAdd(Operation *addOp, Value addOutput) {
  if (addOp->getName().getStringRef() == "torch.aten.add.Scalar") {
    std::optional<double> epsOpt = matchTorchAddScalar(addOutput);
    if (!epsOpt) {
      return std::nullopt;
    }
    return EpsilonAndMean{*epsOpt, addOp->getOperand(0)};
  }
  StringRef opName = addOp->getName().getStringRef();
  if (opName == "mfuse.add") {
    if (auto addMfuse = dyn_cast<AddOp>(addOp)) {
      std::optional<double> epsOpt = getConstantFloat(addMfuse.getX());
      if (epsOpt) {
        return EpsilonAndMean{*epsOpt, addMfuse.getY()};
      }
      epsOpt = getConstantFloat(addMfuse.getY());
      if (!epsOpt) {
        return std::nullopt;
      }
      return EpsilonAndMean{*epsOpt, addMfuse.getX()};
    }
  }
  if (opName == "mfuse.aclnn.add") {
    std::optional<double> epsOpt = getConstantFloat(addOp->getOperand(0));
    if (epsOpt) {
      return EpsilonAndMean{*epsOpt, addOp->getOperand(1)};
    }
    epsOpt = getConstantFloat(addOp->getOperand(1));
    if (!epsOpt) {
      return std::nullopt;
    }
    return EpsilonAndMean{*epsOpt, addOp->getOperand(0)};
  }
  return std::nullopt;
}

/// Intermediate state for RmsNorm pattern matching across helper functions.
struct RmsNormMatchState {
  MulOp firstMulOp;
  RsqrtOp rsqrtOp;
  Operation *addOp = nullptr;
  Value xVal;
  Value gammaVal;
  Value normalizedVal;
  Value normalizedValueToCheck;
  double epsilon = 0.0;
};

/// Identify normalized value and gamma from the gamma-mul operands.
/// Resolves cast chains (incl. npu_dtype_cast) to find the first MulOp.
LogicalResult matchGammaMulOperands(MulOp gammaMulOp, RmsNormMatchState &state,
                                    PatternRewriter &rewriter) {
  Value lhs = gammaMulOp.getLhs();
  Value rhs = gammaMulOp.getRhs();

  state.normalizedVal = getSourceThroughAllCastsIncludingDtype(lhs);
  state.gammaVal = rhs;
  if (!state.normalizedVal.getDefiningOp<MulOp>()) {
    state.normalizedVal = getSourceThroughAllCastsIncludingDtype(rhs);
    state.gammaVal = lhs;
  }

  state.firstMulOp = state.normalizedVal.getDefiningOp<MulOp>();
  if (!state.firstMulOp) {
    return rewriter.notifyMatchFailure(
        gammaMulOp, "gamma*mul: normalized value must come from a MulOp");
  }

  bool isLhsNormalized =
      (state.normalizedVal == getSourceThroughAllCastsIncludingDtype(lhs));
  // normalizedValueToCheck is the value flowing from firstMulOp to gammaMulOp,
  // including any dtype / unrealized_conversion casts in between. We require
  // this chain to have a single use so that eraseDeadRmsNormChainOps can
  // safely clean up the intermediate casts after fusion without breaking
  // other users.
  state.normalizedValueToCheck = isLhsNormalized ? lhs : rhs;
  if (!state.normalizedValueToCheck.hasOneUse()) {
    return rewriter.notifyMatchFailure(
        gammaMulOp, "normalized value (or its cast chain) must have single use");
  }

  return success();
}

/// Verify optional pow→mean chain and x consistency.
LogicalResult verifyPowMeanChain(MulOp gammaMulOp, RmsNormMatchState &state,
                                 Value meanOutput, PatternRewriter &rewriter) {
  Value powOutput = meanOutput;
  if (state.addOp->getName().getStringRef() == "torch.aten.add.Scalar") {
    Value meanDimOutput = matchTorchMeanDim(state.addOp);
    if (meanDimOutput) {
      powOutput = meanDimOutput;
    }
  }

  Value xTorch = matchTorchPowTensorScalar(powOutput);
  if (xTorch) {
    Value xTorchFromMul = getSourceThroughAllCasts(state.xVal);
    if (xTorchFromMul != xTorch) {
      return rewriter.notifyMatchFailure(
          gammaMulOp, "x in first mul must match pow input (through cast)");
    }
  }
  return success();
}

/// Verify add result has single use through cast chain to rsqrt.
LogicalResult verifyAddSingleUse(MulOp gammaMulOp, RmsNormMatchState &state,
                                 PatternRewriter &rewriter) {
  // Walk from rsqrt input backwards through any unrealized_conversion_cast
  // ops and require that the originating add result has a single use. This
  // guarantees that erasing the add / rsqrt chain after fusion will not
  // invalidate other users in the graph.
  Value check = state.rsqrtOp.getInput();
  Operation *cur = check.getDefiningOp();
  while (cur && isCastOp(cur)) {
    check = cur->getOperand(0);
    cur = check.getDefiningOp();
  }
  if (!check.hasOneUse()) {
    return rewriter.notifyMatchFailure(
        gammaMulOp, "add result must have single use in the chain");
  }
  return success();
}

/// Match the normalization chain: firstMul(x, rsqrt(add(mean, eps))).
/// Validates rsqrt, add, optional pow/mean, and extracts epsilon.
LogicalResult matchNormalizationChain(MulOp gammaMulOp, RmsNormMatchState &state,
                                      PatternRewriter &rewriter) {
  state.xVal = state.firstMulOp.getLhs();
  Value rsqrtVal = state.firstMulOp.getRhs();
  if (!rsqrtVal.getDefiningOp<RsqrtOp>()) {
    rsqrtVal = state.firstMulOp.getLhs();
    state.xVal = state.firstMulOp.getRhs();
  }
  if (!rsqrtVal.getDefiningOp<RsqrtOp>()) {
    return rewriter.notifyMatchFailure(
        gammaMulOp, "first mul must be (x, rsqrt) or (rsqrt, x)");
  }

  state.rsqrtOp = cast<RsqrtOp>(rsqrtVal.getDefiningOp());
  // rsqrt may have extra uses (e.g. unrealized_conversion_cast returned for
  // backward pass in training graphs). We allow this and replace all rsqrt
  // uses with rstdOut after fusion.

  Value addOutput = getSourceThroughAllCasts(state.rsqrtOp.getInput());
  state.addOp = addOutput.getDefiningOp();
  if (!state.addOp) {
    return rewriter.notifyMatchFailure(
        gammaMulOp, "rsqrt input must have defining op (add)");
  }

  auto epsAndMean = getEpsilonAndMeanFromAdd(state.addOp, addOutput);
  if (!epsAndMean) {
    return rewriter.notifyMatchFailure(
        gammaMulOp,
        "add must be torch.aten.add.Scalar or mfuse.add with constant epsilon");
  }
  state.epsilon = epsAndMean->epsilon;

  if (failed(verifyPowMeanChain(gammaMulOp, state, epsAndMean->meanOutput,
                                rewriter))) {
    return failure();
  }

  return verifyAddSingleUse(gammaMulOp, state, rewriter);
}

/// Create fused aclnn.rms_norm, handle type mismatches, and replace gamma mul.
LogicalResult createFusedRmsNorm(MulOp gammaMulOp, RmsNormMatchState &state,
                                 PatternRewriter &rewriter) {
  auto xType = dyn_cast<RankedTensorType>(state.xVal.getType());
  auto gammaType = dyn_cast<RankedTensorType>(state.gammaVal.getType());
  if (!xType || !gammaType) {
    return rewriter.notifyMatchFailure(
        gammaMulOp, "x and gamma must be ranked tensor types");
  }

  Location loc = gammaMulOp.getLoc();
  auto epsilonAttr = rewriter.getF64FloatAttr(state.epsilon);

  // Insert the fused op at the rsqrt position so that rstdOut dominates all
  // original rsqrt users (e.g. unrealized_conversion_cast for backward pass
  // that appears before gammaMulOp in the IR).
  // Defensive check: xVal and gammaVal must already dominate rsqrtOp
  // (block arguments have no definingOp and dominate everything).
  auto *rsqrtBlock = state.rsqrtOp->getBlock();
  if (auto *xDef = state.xVal.getDefiningOp()) {
    if (xDef->getBlock() == rsqrtBlock &&
        !xDef->isBeforeInBlock(state.rsqrtOp)) {
      return rewriter.notifyMatchFailure(
          gammaMulOp, "x must be defined before rsqrt for correct SSA dominance");
    }
  }
  if (auto *gammaDef = state.gammaVal.getDefiningOp()) {
    if (gammaDef->getBlock() == rsqrtBlock &&
        !gammaDef->isBeforeInBlock(state.rsqrtOp)) {
      return rewriter.notifyMatchFailure(
          gammaMulOp, "gamma must be defined before rsqrt for correct SSA dominance");
    }
  }
  rewriter.setInsertionPoint(state.rsqrtOp);

  // NPU aclnnRmsNorm requires x and gamma to have the same element type.
  Value gammaForOp = state.gammaVal;
  if (xType.getElementType() != gammaType.getElementType()) {
    auto castedType = RankedTensorType::get(
        gammaType.getShape(), xType.getElementType());
    gammaForOp = rewriter.create<CastOp>(loc, castedType, state.gammaVal);
  }

  // rstdOut uses the reduced shape from rsqrt input (e.g., [1,512,1]).
  auto rstdType = dyn_cast<RankedTensorType>(
      state.rsqrtOp.getInput().getType());
  if (!rstdType) {
    rstdType = xType;
  }
  SmallVector<Type, 2> resultTypes = {xType, rstdType};
  auto rmsNormOp = rewriter.create<AclnnRmsNormOp>(
      loc, resultTypes, state.xVal, gammaForOp, epsilonAttr);

  // Insert output cast if fused op type differs from original result type.
  Value yOut = rmsNormOp.getYOut();
  Type origResultType = gammaMulOp.getResult().getType();
  if (yOut.getType() != origResultType) {
    yOut = rewriter.create<CastOp>(loc, origResultType, yOut);
  }

  // Replace rsqrt uses (e.g. unrealized_conversion_cast for backward pass)
  // with rstdOut, which is semantically equivalent.
  Value rstdOut = rmsNormOp.getRstdOut();
  if (rstdOut.getType() != state.rsqrtOp.getResult().getType()) {
    rstdOut = rewriter.create<CastOp>(
        loc, state.rsqrtOp.getResult().getType(), rstdOut);
  }
  rewriter.replaceAllUsesWith(state.rsqrtOp.getResult(), rstdOut);

  rewriter.replaceAllUsesWith(gammaMulOp.getOperation()->getResult(0), yOut);
  rewriter.eraseOp(gammaMulOp);
  return success();
}

/// Erase dead ops in the matched RmsNorm chain after fusion.
void eraseDeadRmsNormChainOps(RmsNormMatchState &state,
                              PatternRewriter &rewriter) {
  // Clean up cast chain between gamma mul and first mul (mixed-precision path).
  Value cur = state.normalizedValueToCheck;
  while (cur != state.normalizedVal) {
    Operation *def = cur.getDefiningOp();
    if (!def) break;
    cur = def->getOperand(0);
    if (def->use_empty()) rewriter.eraseOp(def);
  }

  if (state.firstMulOp->use_empty()) rewriter.eraseOp(state.firstMulOp);
  if (state.rsqrtOp->use_empty()) rewriter.eraseOp(state.rsqrtOp);
  if (state.addOp && state.addOp->use_empty()) rewriter.eraseOp(state.addOp);
}

}  // namespace

/**
 * Fuse decomposed RmsNorm: pow(x,2) -> mean.dim -> add.Scalar(eps) -> rsqrt -> mul(x, rsqrt)
 * -> mul(gamma)  into  aclnn.rms_norm(x, gamma, epsilon).
 * Matches from the last op (second Mul = gamma * normalized).
 */
class FuseRmsNormPattern : public OpRewritePattern<MulOp> {
 public:
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp gammaMulOp,
                                PatternRewriter &rewriter) const override {
    RmsNormMatchState state;

    if (failed(matchGammaMulOperands(gammaMulOp, state, rewriter)) ||
        failed(matchNormalizationChain(gammaMulOp, state, rewriter))) {
      return failure();
    }

    MLOG(DEBUG) << "FuseRmsNormPattern matched, fusing to "
                   "aclnn.rms_norm";

    if (failed(createFusedRmsNorm(gammaMulOp, state, rewriter))) {
      return failure();
    }
    eraseDeadRmsNormChainOps(state, rewriter);
    return success();
  }
};

// Register the fusion pass and expose createFuseRmsNormPass().
DEFINE_MFUSE_FUSION_PASS(FuseRmsNorm, FuseRmsNormPattern)

}  // namespace mfuse
}  // namespace mlir

