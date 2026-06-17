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

#include "mfusion/Analysis/Split/LayerNormDvmUtils.h"

#include <algorithm>
#include <cmath>

#include "mfusion/Analysis/Split/FusionRegionTag.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Support/VarianceUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace mfuse {
namespace layernorm_dvm {
namespace {

constexpr double kDefaultLayerNormEps = 1e-5;
constexpr double kEpsTolerance = 1e-6;

bool getSingleElementFloatValue(Value v, double &out) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) {
    return false;
  }
  auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!denseAttr || !denseAttr.isSplat()) {
    return false;
  }
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType || !isScalarOrSingleElement(tensorType)) {
    return false;
  }
  if (!isa<FloatType>(denseAttr.getElementType())) {
    return false;
  }
  out = denseAttr.getSplatValue<APFloat>().convertToDouble();
  return true;
}

bool getSingleElementNumericValue(Value v, double &out) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) {
    return false;
  }
  auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!denseAttr || !denseAttr.isSplat()) {
    return false;
  }
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType || !isScalarOrSingleElement(tensorType)) {
    return false;
  }
  Type elemType = denseAttr.getElementType();
  if (isa<FloatType>(elemType)) {
    out = denseAttr.getSplatValue<APFloat>().convertToDouble();
    return true;
  }
  if (isa<IntegerType>(elemType)) {
    out = static_cast<double>(denseAttr.getSplatValue<APInt>().getSExtValue());
    return true;
  }
  return false;
}

bool isPositiveEpsScalar(Value v, double epsTol = kEpsTolerance) {
  double val = 0.0;
  if (!getSingleElementFloatValue(v, val)) {
    return false;
  }
  if (std::abs(val - kDefaultLayerNormEps) <= epsTol) {
    return true;
  }
  return val > 0.0 && val < 1e-2;
}

bool isPositiveDimScalar(Value v) {
  double val = 0.0;
  return getSingleElementNumericValue(v, val) && val > 0.0;
}

bool matchSqrtRstdPrologue(Value rstdVal, SqrtOp &sqrtOp, AddOp &addEpsOp) {
  rstdVal = peelBroadcast(rstdVal);
  auto addOp = rstdVal.getDefiningOp<AddOp>();
  if (!addOp) {
    return false;
  }
  Value lhs = addOp.getX();
  Value rhs = addOp.getY();
  Value sqrtVal;
  if (lhs.getDefiningOp<SqrtOp>() && isPositiveEpsScalar(rhs)) {
    sqrtVal = lhs;
  } else if (rhs.getDefiningOp<SqrtOp>() && isPositiveEpsScalar(lhs)) {
    sqrtVal = rhs;
  } else {
    return false;
  }
  auto sqrt = sqrtVal.getDefiningOp<SqrtOp>();
  if (!sqrt) {
    return false;
  }
  sqrtOp = sqrt;
  addEpsOp = addOp;
  return true;
}

bool isReduceSumKeepdim(ReduceSumOp reduce) {
  if (!reduce || !reduce.getKeepdim()) {
    return false;
  }
  auto dims = reduce.getDimensions();
  return dims && !dims.empty();
}

bool isReduceMeanKeepdim(ReduceMeanOp reduce) {
  if (!reduce || !reduce.getKeepdim()) {
    return false;
  }
  auto dims = reduce.getDimensions();
  return dims && !dims.empty();
}

bool isMeanFromReduceMean(Value meanVal, ReduceMeanOp &reduceMean) {
  meanVal = peelBroadcast(meanVal);
  auto reduce = meanVal.getDefiningOp<ReduceMeanOp>();
  if (!isReduceMeanKeepdim(reduce)) {
    return false;
  }
  reduceMean = reduce;
  return true;
}

bool isMeanFromReduceSum(Value meanVal, ReduceSumOp &reduceSum, DivOp &meanDiv) {
  meanVal = peelBroadcast(meanVal);
  auto divOp = meanVal.getDefiningOp<DivOp>();
  if (!divOp) {
    return false;
  }
  Value sumVal = peelBroadcast(divOp.getSelf());
  auto reduce = sumVal.getDefiningOp<ReduceSumOp>();
  if (!isReduceSumKeepdim(reduce)) {
    return false;
  }
  if (!isPositiveDimScalar(divOp.getOther())) {
    return false;
  }
  reduceSum = reduce;
  meanDiv = divOp;
  return true;
}

bool collectDecomposedVarOps(SqrtOp sqrt, LayerNormDvmMatch &result) {
  DecomposedVarianceChain chain;
  if (!matchDecomposedVarianceChain(sqrt.getInput(), result.x, chain)) {
    return false;
  }
  result.varDiv = chain.varDiv;
  result.varReduceSum = chain.varReduceSum;
  result.varSquareMul = chain.varSquareMul;
  return true;
}

bool collectUniqueOps(LayerNormDvmMatch &result) {
  llvm::SmallVector<Operation *, 16> ordered = {
    result.betaAdd,      result.normDiv,      result.gammaMul,   result.centerSub, result.meanDiv,
    result.reduceSum,    result.reduceMean,   result.rstdAdd,    result.sqrtOp,    result.varDiv,
    result.varReduceSum, result.varSquareMul,
  };
  llvm::DenseSet<Operation *> seen;
  result.ops.clear();
  for (Operation *op : ordered) {
    if (op && seen.insert(op).second) {
      result.ops.push_back(op);
    }
  }
  return !result.ops.empty();
}

void appendBroadcastOps(Value v, llvm::SmallVectorImpl<Operation *> &ops, llvm::DenseSet<Operation *> &seen) {
  while (auto bcast = v.getDefiningOp<BroadcastToOp>()) {
    if (seen.insert(bcast).second) {
      ops.push_back(bcast);
    }
    v = bcast.getInput();
  }
}

bool hasAtMostUses(Value v, unsigned maxUses) {
  unsigned count = 0;
  for (auto &use : v.getUses()) {
    (void)use;
    if (++count > maxUses) {
      return false;
    }
  }
  return true;
}

Value getFirstResult(Operation *op) { return (op && op->getNumResults() > 0) ? op->getResult(0) : Value{}; }

bool verifySingleUseExceptX(LayerNormDvmMatch &result) {
  auto check = [&](Operation *op, unsigned maxUses) {
    Value v = getFirstResult(op);
    if (!v || v == result.x) {
      return true;
    }
    return hasAtMostUses(v, maxUses);
  };
  return check(result.betaAdd.getOperation(), 1) && check(result.normDiv.getOperation(), 1) &&
         check(result.gammaMul.getOperation(), 1) && check(result.centerSub.getOperation(), 2) &&
         check(result.meanDiv.getOperation(), 2) && check(result.reduceSum.getOperation(), 1) &&
         check(result.reduceMean.getOperation(), 2) && check(result.rstdAdd.getOperation(), 1) &&
         check(result.sqrtOp.getOperation(), 2) && check(result.varDiv.getOperation(), 1) &&
         check(result.varReduceSum.getOperation(), 1) && check(result.varSquareMul.getOperation(), 1);
}

bool matchNormDivFromBetaAdd(AddOp addOp, LayerNormDvmMatch &result) {
  result.betaAdd = addOp;
  Value normVal;
  if (addOp.getX().getDefiningOp<DivOp>()) {
    normVal = addOp.getX();
    result.beta = addOp.getY();
  } else if (addOp.getY().getDefiningOp<DivOp>()) {
    normVal = addOp.getY();
    result.beta = addOp.getX();
  } else {
    return false;
  }

  result.normDiv = normVal.getDefiningOp<DivOp>();
  if (!result.normDiv) {
    return false;
  }

  Value scaledVal = result.normDiv.getSelf();
  Value otherVal = result.normDiv.getOther();
  if (!scaledVal.getDefiningOp<MulOp>()) {
    if (otherVal.getDefiningOp<MulOp>()) {
      scaledVal = otherVal;
    } else {
      return false;
    }
  }

  result.gammaMul = scaledVal.getDefiningOp<MulOp>();
  return result.gammaMul != nullptr;
}

bool matchGammaCenterFromNormDiv(LayerNormDvmMatch &result) {
  Value centeredVal;
  if (result.gammaMul.getLhs().getDefiningOp<SubOp>()) {
    centeredVal = result.gammaMul.getLhs();
    result.gamma = result.gammaMul.getRhs();
  } else if (result.gammaMul.getRhs().getDefiningOp<SubOp>()) {
    centeredVal = result.gammaMul.getRhs();
    result.gamma = result.gammaMul.getLhs();
  } else {
    return false;
  }

  result.centerSub = centeredVal.getDefiningOp<SubOp>();
  if (!result.centerSub) {
    return false;
  }
  result.x = result.centerSub.getX();
  return true;
}

bool matchMeanFromCenter(LayerNormDvmMatch &result) {
  Value meanVal = result.centerSub.getY();
  if (isMeanFromReduceMean(meanVal, result.reduceMean)) {
    return result.reduceMean.getInput() == result.x;
  }
  if (!isMeanFromReduceSum(meanVal, result.reduceSum, result.meanDiv)) {
    return false;
  }
  return result.reduceSum.getInput() == result.x;
}

bool reduceSumDimsEqual(ReduceSumOp reduce, llvm::ArrayRef<int64_t> expected) {
  if (!reduce || !reduce.getDimensions()) {
    return false;
  }
  auto dims = reduce.getDimensions().getValue();
  if (dims.size() != expected.size()) {
    return false;
  }
  llvm::SmallVector<int64_t> actual;
  actual.reserve(dims.size());
  std::transform(dims.begin(), dims.end(), std::back_inserter(actual), [](Attribute dimAttr) {
    return cast<IntegerAttr>(dimAttr).getValue().getSExtValue();
  });
  llvm::SmallVector<int64_t> normalized = actual;
  auto inputType = dyn_cast<RankedTensorType>(reduce.getInput().getType());
  if (inputType) {
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [&](int64_t d) {
      return d < 0 ? d + inputType.getRank() : d;
    });
  }
  llvm::sort(normalized.begin(), normalized.end());
  llvm::SmallVector<int64_t> expectedNorm(expected.begin(), expected.end());
  llvm::sort(expectedNorm.begin(), expectedNorm.end());
  return normalized == expectedNorm;
}

bool reduceSumOnAllButLastDim(ReduceSumOp reduce) {
  auto inputType = dyn_cast<RankedTensorType>(reduce.getInput().getType());
  if (!inputType || inputType.getRank() < 2) {
    return false;
  }
  const int64_t rank = inputType.getRank();
  llvm::SmallVector<int64_t> expected;
  for (int64_t i = 0; i < rank - 1; ++i) {
    expected.push_back(i);
  }
  return reduceSumDimsEqual(reduce, expected);
}

bool reduceSumOnLastDimOnly(ReduceSumOp reduce) {
  auto inputType = dyn_cast<RankedTensorType>(reduce.getInput().getType());
  if (!inputType || inputType.getRank() < 1) {
    return false;
  }
  const int64_t last = inputType.getRank() - 1;
  return reduceSumDimsEqual(reduce, {last});
}

bool matchRstdPlusEpsAdd(Value v, AddOp &addEps) {
  v = peelBroadcast(v);
  auto add = v.getDefiningOp<AddOp>();
  if (!add) {
    return false;
  }
  if (isPositiveEpsScalar(add.getX()) || isPositiveEpsScalar(add.getY())) {
    addEps = add;
    return true;
  }
  return false;
}

void appendUniqueOp(Operation *op, llvm::SmallVectorImpl<Operation *> &ops, llvm::DenseSet<Operation *> &seen) {
  if (op && seen.insert(op).second) {
    ops.push_back(op);
  }
}

bool collectUniqueBwdOps(LayerNormDvmBwdMatch &result) {
  llvm::SmallVector<Operation *, 16> ordered = {
    result.gradDiv, result.centerSub, result.meanDiv, result.reduceSum, result.reduceMean,
  };
  llvm::DenseSet<Operation *> seen;
  result.ops.clear();
  for (Operation *op : ordered) {
    if (op && seen.insert(op).second) {
      result.ops.push_back(op);
    }
  }
  return !result.ops.empty();
}

}  // namespace

bool hasLayerNormDvmAttr(Operation *op) { return fusion_region::hasRegionMember(op); }

bool hasLayerNormDvmAffinityAttr(Operation *op) { return fusion_region::hasRegionAffinity(op); }

void tagLayerNormDvmOp(Operation *op, llvm::StringRef groupId) {
  if (!op || hasLayerNormDvmAttr(op)) {
    return;
  }
  std::string owned;
  if (groupId.empty()) {
    owned = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
    groupId = owned;
  }
  fusion_region::tagMember(op, groupId, fusion_region::kLayerNormFuseKind);
}

void tagLayerNormDvmAffinityOp(Operation *op, llvm::StringRef groupId) {
  if (!op || fusion_region::isTagged(op)) {
    return;
  }
  std::string owned;
  if (groupId.empty()) {
    owned = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
    groupId = owned;
  }
  fusion_region::tagAffinity(op, groupId, fusion_region::kLayerNormFuseKind);
}

namespace {

bool isSharedRstdAffinityOp(Operation *op) {
  if (!op) {
    return false;
  }
  if (isa<SqrtOp>(op)) {
    return true;
  }
  if (auto add = dyn_cast<AddOp>(op)) {
    return isPositiveEpsScalar(add.getX()) || isPositiveEpsScalar(add.getY());
  }
  return false;
}

void tagLayerNormDvmAffinityBroadcasts(Value v, llvm::StringRef groupId, llvm::DenseSet<Operation *> &seen) {
  while (auto bcast = v.getDefiningOp<BroadcastToOp>()) {
    if (seen.insert(bcast).second) {
      tagLayerNormDvmAffinityOp(bcast, groupId);
    }
    v = bcast.getInput();
  }
}

}  // namespace

namespace {

bool isAllowedBwdTagUser(Operation *user, const llvm::DenseSet<Operation *> &closure) {
  if (!user) {
    return false;
  }
  if (closure.contains(user)) {
    return true;
  }
  return isa<func::ReturnOp>(user);
}

bool opOnlyUsedByOpsInSet(Operation *op, const llvm::DenseSet<Operation *> &closure,
                          llvm::DenseSet<Operation *> *visited = nullptr) {
  if (!op) {
    return false;
  }
  llvm::DenseSet<Operation *> localVisited;
  if (!visited) {
    visited = &localVisited;
  }
  if (!visited->insert(op).second) {
    return true;
  }
  for (Value result : op->getResults()) {
    for (auto &use : result.getUses()) {
      Operation *user = use.getOwner();
      if (!isAllowedBwdTagUser(user, closure)) {
        return false;
      }
      if (closure.contains(user) && !opOnlyUsedByOpsInSet(user, closure, visited)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

void tagLayerNormDvmForwardOps(ArrayRef<Operation *> ops) {
  std::string groupId = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
  fusion_region::tagMembers(ops, groupId, fusion_region::kLayerNormFuseKind);
}

unsigned tagLayerNormDvmBackwardOpsExclusive(ArrayRef<Operation *> ops, llvm::StringRef groupId) {
  std::string ownedGroup;
  if (groupId.empty()) {
    ownedGroup = fusion_region::allocateGroupId(fusion_region::kLayerNormFuseKind);
    groupId = ownedGroup;
  }
  llvm::DenseSet<Operation *> closure(ops.begin(), ops.end());
  unsigned tagged = 0;
  for (Operation *op : ops) {
    if (!op || hasLayerNormDvmAttr(op)) {
      continue;
    }
    if (!opOnlyUsedByOpsInSet(op, closure)) {
      if (isSharedRstdAffinityOp(op)) {
        tagLayerNormDvmAffinityOp(op, groupId);
        llvm::DenseSet<Operation *> seen;
        tagLayerNormDvmAffinityBroadcasts(op->getResult(0), groupId, seen);
      }
      continue;
    }
    fusion_region::tagMember(op, groupId, fusion_region::kLayerNormFuseKind);
    tagged++;
  }
  return tagged;
}

LayerNormDvmMatch matchLayerNormDvmFromBetaAdd(AddOp addOp) {
  LayerNormDvmMatch result;
  if (!matchNormDivFromBetaAdd(addOp, result) || !matchGammaCenterFromNormDiv(result) ||
      !matchMeanFromCenter(result)) {
    return result;
  }

  Value self = result.normDiv.getSelf();
  Value other = result.normDiv.getOther();
  Value rstdVal = (self.getDefiningOp<MulOp>() == result.gammaMul.getOperation()) ? other : self;

  if (!matchSqrtRstdPrologue(rstdVal, result.sqrtOp, result.rstdAdd)) {
    return result;
  }

  // Optional: tag decomposed var ops when rstd is computed inline (saved-rstd graphs skip this).
  (void)collectDecomposedVarOps(result.sqrtOp, result);

  if (!verifySingleUseExceptX(result) || !collectUniqueOps(result)) {
    return result;
  }

  llvm::DenseSet<Operation *> seen(result.ops.begin(), result.ops.end());
  appendBroadcastOps(result.normDiv.getOther(), result.ops, seen);
  appendBroadcastOps(result.centerSub.getY(), result.ops, seen);
  appendBroadcastOps(result.beta, result.ops, seen);

  result.matched = true;
  return result;
}

LayerNormDvmBwdMatch matchLayerNormDvmBackwardFromGradDiv(DivOp gradDiv) {
  LayerNormDvmBwdMatch result;
  result.gradDiv = gradDiv;

  Value centeredVal = gradDiv.getSelf();
  Value rstdVal = gradDiv.getOther();
  if (!centeredVal.getDefiningOp<SubOp>()) {
    return result;
  }

  result.centerSub = centeredVal.getDefiningOp<SubOp>();
  result.x = result.centerSub.getX();
  Value meanVal = result.centerSub.getY();
  if (!isMeanFromReduceSum(meanVal, result.reduceSum, result.meanDiv)) {
    ReduceMeanOp reduceMean;
    if (!isMeanFromReduceMean(meanVal, reduceMean)) {
      return result;
    }
    result.reduceMean = reduceMean;
  }

  result.rstd = peelBroadcast(rstdVal);
  if (!result.rstd) {
    return result;
  }

  if (!collectUniqueBwdOps(result)) {
    return result;
  }

  llvm::DenseSet<Operation *> seen(result.ops.begin(), result.ops.end());
  appendBroadcastOps(result.centerSub.getY(), result.ops, seen);
  appendBroadcastOps(rstdVal, result.ops, seen);

  result.matched = true;
  return result;
}

LayerNormDvmBwdMatch matchLayerNormDvmBackwardFromCqxnSum(ReduceSumOp sumOp) {
  LayerNormDvmBwdMatch result;
  if (!isReduceSumKeepdim(sumOp) || !reduceSumOnAllButLastDim(sumOp)) {
    return result;
  }

  auto mulPair = sumOp.getInput().getDefiningOp<MulOp>();
  if (!mulPair) {
    return result;
  }

  DivOp divOp = nullptr;
  if (mulPair.getLhs().getDefiningOp<DivOp>()) {
    divOp = mulPair.getLhs().getDefiningOp<DivOp>();
  } else if (mulPair.getRhs().getDefiningOp<DivOp>()) {
    divOp = mulPair.getRhs().getDefiningOp<DivOp>();
  }
  if (!divOp) {
    return result;
  }

  AddOp rstdAdd;
  Value rstdVal;
  Value gradVal;
  if (matchRstdPlusEpsAdd(divOp.getOther(), rstdAdd)) {
    rstdVal = divOp.getOther();
    gradVal = divOp.getSelf();
  } else if (matchRstdPlusEpsAdd(divOp.getSelf(), rstdAdd)) {
    rstdVal = divOp.getSelf();
    gradVal = divOp.getOther();
  } else {
    return result;
  }

  result.x = gradVal;
  result.rstd = peelBroadcast(rstdVal);

  llvm::DenseSet<Operation *> seen;
  appendUniqueOp(sumOp, result.ops, seen);
  appendUniqueOp(mulPair, result.ops, seen);
  appendUniqueOp(divOp, result.ops, seen);
  appendUniqueOp(rstdAdd, result.ops, seen);
  appendBroadcastOps(divOp.getOther(), result.ops, seen);
  appendBroadcastOps(divOp.getSelf(), result.ops, seen);

  for (auto &use : divOp.getResult().getUses()) {
    if (auto gammaMul = dyn_cast<MulOp>(use.getOwner())) {
      appendUniqueOp(gammaMul, result.ops, seen);
      appendBroadcastOps(gammaMul.getLhs(), result.ops, seen);
      appendBroadcastOps(gammaMul.getRhs(), result.ops, seen);
    }
  }

  result.matched = !result.ops.empty();
  return result;
}

LayerNormDvmBwdMatch matchLayerNormDvmBackwardFromCuahirSum(ReduceSumOp sumOp) {
  LayerNormDvmBwdMatch result;
  if (!isReduceSumKeepdim(sumOp) || !reduceSumOnLastDimOnly(sumOp)) {
    return result;
  }

  auto finalMul = sumOp.getInput().getDefiningOp<MulOp>();
  if (!finalMul) {
    return result;
  }

  NegOp negOp = nullptr;
  DivOp innerDiv = nullptr;
  if (finalMul.getLhs().getDefiningOp<NegOp>()) {
    negOp = finalMul.getLhs().getDefiningOp<NegOp>();
    innerDiv = finalMul.getRhs().getDefiningOp<DivOp>();
  } else if (finalMul.getRhs().getDefiningOp<NegOp>()) {
    negOp = finalMul.getRhs().getDefiningOp<NegOp>();
    innerDiv = finalMul.getLhs().getDefiningOp<DivOp>();
  }
  if (!negOp || !innerDiv) {
    return result;
  }

  auto outerDiv = innerDiv.getSelf().getDefiningOp<DivOp>();
  if (!outerDiv) {
    return result;
  }

  AddOp rstdAdd;
  Value rstdSide = peelBroadcast(innerDiv.getOther());
  if (!matchRstdPlusEpsAdd(rstdSide, rstdAdd)) {
    return result;
  }
  AddOp outerRstdAdd;
  Value outerRstd = peelBroadcast(outerDiv.getOther());
  if (!matchRstdPlusEpsAdd(outerRstd, outerRstdAdd) || outerRstdAdd.getOperation() != rstdAdd.getOperation()) {
    return result;
  }

  auto gammaMul = outerDiv.getSelf().getDefiningOp<MulOp>();
  if (!gammaMul) {
    return result;
  }

  result.x = negOp.getInput();
  result.rstd = isPositiveEpsScalar(rstdAdd.getX()) ? rstdAdd.getY() : rstdAdd.getX();

  llvm::DenseSet<Operation *> seen;
  appendUniqueOp(sumOp, result.ops, seen);
  appendUniqueOp(finalMul, result.ops, seen);
  appendUniqueOp(negOp, result.ops, seen);
  appendUniqueOp(innerDiv, result.ops, seen);
  appendUniqueOp(outerDiv, result.ops, seen);
  appendUniqueOp(rstdAdd, result.ops, seen);
  appendUniqueOp(gammaMul, result.ops, seen);
  appendBroadcastOps(innerDiv.getOther(), result.ops, seen);
  appendBroadcastOps(outerDiv.getOther(), result.ops, seen);
  appendBroadcastOps(gammaMul.getLhs(), result.ops, seen);
  appendBroadcastOps(gammaMul.getRhs(), result.ops, seen);

  result.matched = !result.ops.empty();
  return result;
}

}  // namespace layernorm_dvm
}  // namespace mfuse
}  // namespace mlir
