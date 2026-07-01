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

#include "mfusion/Analysis/DvmFusion/SafeSoftmax/SafeSoftmaxDvmUtils.h"

#include <algorithm>
#include <iterator>

#include "mfusion/Analysis/DvmFusion/SafeSoftmax/BroadcastCondSelectUtils.h"
#include "mfusion/Analysis/FusionRegion/FusionRegionTag.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace mfuse {
namespace safe_softmax_dvm {
namespace {

bool isLastDimReduce(ReduceSumOp sumOp, Value expValue) {
  auto outType = dyn_cast<RankedTensorType>(sumOp.getResult().getType());
  auto expType = dyn_cast<RankedTensorType>(expValue.getType());
  if (!outType || !expType || outType.getRank() != expType.getRank()) {
    return false;
  }
  if (!sumOp.getKeepdim()) {
    return false;
  }
  auto dimsAttr = sumOp.getDimensions();
  if (!dimsAttr || dimsAttr.size() != 1) {
    return false;
  }
  int64_t dim = cast<IntegerAttr>(dimsAttr[0]).getInt();
  int64_t lastDim = expType.getRank() - 1;
  return dim == lastDim || dim == -1;
}

bool isZeroFull(FullOp fullOp) {
  auto constVal = fullOp.getFillValue().getDefiningOp<ConstantOp>();
  if (!constVal) {
    return false;
  }
  if (auto attr = dyn_cast<DenseFPElementsAttr>(constVal.getValueAttr())) {
    return attr.isSplat() && attr.getSplatValue<APFloat>().isZero();
  }
  if (auto attr = dyn_cast<DenseIntElementsAttr>(constVal.getValueAttr())) {
    return attr.isSplat() && attr.getSplatValue<APInt>().isZero();
  }
  return false;
}

bool matchSoftmaxDiv(DivOp divOp, SafeSoftmaxDvmMatch &result) {
  auto tryOrder = [&](Value num, Value den) -> bool {
    auto expOp = num.getDefiningOp<ExpOp>();
    if (!expOp) {
      return false;
    }
    auto sumOp = den.getDefiningOp<ReduceSumOp>();
    if (!sumOp || sumOp.getInput() != expOp.getResult()) {
      return false;
    }
    if (!isLastDimReduce(sumOp, expOp.getResult())) {
      return false;
    }
    auto subOp = expOp.getInput().getDefiningOp<SubOp>();
    if (!subOp) {
      return false;
    }
    result.softmaxDiv = divOp;
    result.exp = expOp;
    result.reduceSum = sumOp;
    result.centerSub = subOp;
    return true;
  };
  return tryOrder(divOp.getSelf(), divOp.getOther()) || tryOrder(divOp.getOther(), divOp.getSelf());
}

ReduceMaxOp findReduceMaxForCenterSub(SubOp subOp) {
  for (Value operand : subOp->getOperands()) {
    if (auto reduceMax = operand.getDefiningOp<ReduceMaxOp>()) {
      return reduceMax;
    }
  }
  return {};
}

constexpr llvm::StringLiteral kAtenSoftmaxOpName = "torch.aten._softmax";

bool isSafeSoftmaxZeroOutput(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return false;
  }
  auto fused = dyn_cast<FusedOp>(result.getDefiningOp());
  if (!fused) {
    return false;
  }
  auto kind = fused->getAttrOfType<StringAttr>(mfusion_attrs::kDvmFuseKind);
  if (!kind || kind.getValue() != fusion_region::kSafeSoftmaxFuseKind) {
    return false;
  }
  return result.getResultNumber() == 0;
}

struct ZeroSoftmaxSide {
  Value zeroBranch;
  FullOp fullOp;
  Value softmaxSide;
  bool matched = false;
};

ZeroSoftmaxSide resolveZeroAndSoftmaxSide(SelectOp selectOp, Value softmaxSideCandidate) {
  ZeroSoftmaxSide result;
  Value onTrue = selectOp.getOnTrue();
  Value onFalse = selectOp.getOnFalse();

  auto tryZeroOnTrue = [&]() -> bool {
    if (auto full = onTrue.getDefiningOp<FullOp>()) {
      if (isZeroFull(full)) {
        result.zeroBranch = onTrue;
        result.fullOp = full;
        result.softmaxSide = onFalse;
        return true;
      }
    }
    if (isSafeSoftmaxZeroOutput(onTrue)) {
      result.zeroBranch = onTrue;
      result.softmaxSide = onFalse;
      return true;
    }
    return false;
  };

  auto tryZeroOnFalse = [&]() -> bool {
    if (auto full = onFalse.getDefiningOp<FullOp>()) {
      if (isZeroFull(full)) {
        result.zeroBranch = onFalse;
        result.fullOp = full;
        result.softmaxSide = onTrue;
        return true;
      }
    }
    if (isSafeSoftmaxZeroOutput(onFalse)) {
      result.zeroBranch = onFalse;
      result.softmaxSide = onTrue;
      return true;
    }
    return false;
  };

  if (tryZeroOnTrue() || tryZeroOnFalse()) {
    if (softmaxSideCandidate && result.softmaxSide != softmaxSideCandidate) {
      return {};
    }
    result.matched = true;
  }
  return result;
}

bool isAtenSoftmaxOpInternal(Operation *op) {
  return op && op->getName().getStringRef() == kAtenSoftmaxOpName;
}

Operation *resolveSoftmaxProducer(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def) {
    return nullptr;
  }
  if (isa<SoftmaxOp>(def) || isAtenSoftmaxOpInternal(def)) {
    return def;
  }
  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
    if (cast.getNumOperands() == 1) {
      if (Operation *prod = cast.getOperand(0).getDefiningOp()) {
        if (isa<SoftmaxOp>(prod) || isAtenSoftmaxOpInternal(prod)) {
          return prod;
        }
      }
    }
  }
  return nullptr;
}

FusedSafeSoftmaxCandidate matchFusedSoftmaxCandidateFromSelectImpl(SelectOp selectOp) {
  FusedSafeSoftmaxCandidate result;
  if (!broadcast_cond_select::isBroadcastConditionalSelect(selectOp)) {
    return result;
  }

  ZeroSoftmaxSide branches = resolveZeroAndSoftmaxSide(selectOp, Value{});
  if (!branches.matched) {
    return result;
  }

  Operation *softmaxOp = resolveSoftmaxProducer(branches.softmaxSide);
  if (!softmaxOp) {
    return result;
  }

  result.matched = true;
  result.select = selectOp;
  result.full = branches.fullOp;
  result.zeroBranch = branches.zeroBranch;
  result.softmaxOp = softmaxOp;
  result.softmaxOutput = branches.softmaxSide;
  return result;
}

bool valueFeedsSafeSoftmaxSelect(Value value, Operation *softmaxOp) {
  llvm::SmallVector<Value> worklist = {value};
  llvm::DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!seen.insert(current).second) {
      continue;
    }
    for (Operation *user : current.getUsers()) {
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
        std::copy(cast.result_begin(), cast.result_end(), std::back_inserter(worklist));
        continue;
      }
      if (auto selectOp = dyn_cast<SelectOp>(user)) {
        FusedSafeSoftmaxCandidate candidate = matchFusedSoftmaxCandidateFromSelectImpl(selectOp);
        if (candidate.matched && candidate.softmaxOp == softmaxOp) {
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace

FusedSafeSoftmaxCandidate matchFusedSoftmaxCandidateFromSelect(SelectOp selectOp) {
  return matchFusedSoftmaxCandidateFromSelectImpl(selectOp);
}

SafeSoftmaxDvmMatch matchSafeSoftmaxFromSelect(SelectOp selectOp) {
  SafeSoftmaxDvmMatch result;
  if (!broadcast_cond_select::isBroadcastConditionalSelect(selectOp)) {
    return result;
  }

  Value onTrue = selectOp.getOnTrue();
  Value onFalse = selectOp.getOnFalse();
  DivOp divOp;
  if (auto div = onTrue.getDefiningOp<DivOp>()) {
    divOp = div;
  } else if (auto div = onFalse.getDefiningOp<DivOp>()) {
    divOp = div;
  } else {
    return result;
  }

  ZeroSoftmaxSide branches = resolveZeroAndSoftmaxSide(selectOp, divOp.getResult());
  if (!branches.matched || !divOp) {
    return result;
  }
  if (!matchSoftmaxDiv(divOp, result)) {
    return result;
  }

  result.matched = true;
  result.select = selectOp;
  result.full = branches.fullOp;
  result.zeroBranch = branches.zeroBranch;
  result.reduceMax = findReduceMaxForCenterSub(result.centerSub);
  return result;
}

FusedSafeSoftmaxCandidate matchFusedSafeSoftmaxFromSelect(SelectOp selectOp) {
  if (matchSafeSoftmaxFromSelect(selectOp).matched) {
    return {};
  }
  return matchFusedSoftmaxCandidateFromSelect(selectOp);
}

bool isSafeSoftmaxBroadcastSelect(SelectOp selectOp) {
  if (matchSafeSoftmaxFromSelect(selectOp).matched) {
    return true;
  }
  // Call matchFusedSoftmaxCandidateFromSelect directly to avoid re-running
  // matchSafeSoftmaxFromSelect inside matchFusedSafeSoftmaxFromSelect.
  return matchFusedSoftmaxCandidateFromSelect(selectOp).matched;
}

bool isMfuseSoftmaxOp(Operation *op) { return op && isa<SoftmaxOp>(op); }

bool isAtenSoftmaxOp(Operation *op) { return isAtenSoftmaxOpInternal(op); }

bool isSoftmaxProducerOp(Operation *op) { return isMfuseSoftmaxOp(op) || isAtenSoftmaxOp(op); }

bool isSafeSoftmaxSoftmaxProducer(Operation *softmaxOp) {
  if (!isSoftmaxProducerOp(softmaxOp)) {
    return false;
  }
  return std::any_of(softmaxOp->result_begin(), softmaxOp->result_end(), [&](Value result) {
    return valueFeedsSafeSoftmaxSelect(result, softmaxOp);
  });
}

bool hasSafeSoftmaxCandidate(Operation *root) {
  if (!root) {
    return false;
  }
  bool found = false;
  root->walk([&](SelectOp selectOp) {
    if (found) {
      return;
    }
    if (isSafeSoftmaxBroadcastSelect(selectOp)) {
      found = true;
    }
  });
  return found;
}

void markSafeSoftmaxPipelineActive(ModuleOp module) {
  if (!module || module->hasAttr(mfusion_attrs::kSafeSoftmaxPipelineActive)) {
    return;
  }
  module->setAttr(mfusion_attrs::kSafeSoftmaxPipelineActive, UnitAttr::get(module.getContext()));
}

bool isSafeSoftmaxTagged(Operation *op) {
  if (!op) {
    return false;
  }
  if (auto kind = op->getAttrOfType<StringAttr>(mfusion_attrs::kDvmFuseKind)) {
    return kind.getValue() == fusion_region::kSafeSoftmaxFuseKind;
  }
  return false;
}

void collectSafeSoftmaxMemberOps(SafeSoftmaxDvmMatch &match) {
  match.memberOps.clear();
  if (!match.matched) {
    return;
  }
  llvm::DenseSet<Operation *> memberSet;
  auto addMember = [&](Operation *op) {
    if (op && memberSet.insert(op).second) {
      match.memberOps.push_back(op);
    }
  };

  if (match.reduceMax) {
    addMember(match.reduceMax.getOperation());
  }
  addMember(match.centerSub.getOperation());
  addMember(match.exp.getOperation());
  addMember(match.reduceSum.getOperation());
  addMember(match.softmaxDiv.getOperation());
  // Shared zero buffers stay external when an earlier op still consumes them; otherwise
  // materialize full(0) inside the cluster so the zero tensor can be yielded for reuse.
  if (match.full && match.zeroBranch) {
    bool earlierSharedUser = false;
    for (Operation *user : match.zeroBranch.getUsers()) {
      if (user == match.select.getOperation()) {
        continue;
      }
      if (user->getBlock() == match.select->getBlock() && user->isBeforeInBlock(match.select)) {
        earlierSharedUser = true;
        break;
      }
    }
    if (!earlierSharedUser) {
      addMember(match.full.getOperation());
    }
  }
  addMember(match.select.getOperation());

  Value cond = match.select.getCondition();
  if (fusion_region::isSingleUse(cond)) {
    if (auto notOp = cond.getDefiningOp<LogicalNotOp>()) {
      // Broadcast mask from func arg (layer-1 style): not(mask) can live in the fused body.
      if (!notOp.getInput().getDefiningOp()) {
        addMember(notOp.getOperation());
      }
    } else if (auto castOp = cond.getDefiningOp<CastOp>()) {
      if (!castOp.getInput().getDefiningOp()) {
        addMember(castOp.getOperation());
      }
    }
  }

  if (ReshapeOp reshapeOp = dyn_cast_or_null<ReshapeOp>(fusion_region::getSingleUserOp(match.select.getResult()))) {
    addMember(reshapeOp.getOperation());
  }
}

void appendAlbertMaskChainMemberOps(SafeSoftmaxDvmMatch &match, const AlbertMaskChainOps &chain) {
  if (!match.matched) {
    return;
  }
  llvm::DenseSet<Operation *> memberSet(match.memberOps.begin(), match.memberOps.end());
  auto addMember = [&](Operation *op) {
    if (op && memberSet.insert(op).second) {
      match.memberOps.push_back(op);
    }
  };
  addMember(chain.eq);
  addMember(chain.eqNot);
  addMember(chain.ne);
  addMember(chain.reduceAny);
  addMember(chain.condNot);
}

void appendMemberOp(SafeSoftmaxDvmMatch &match, Operation *op) {
  if (!match.matched || !op) {
    return;
  }
  llvm::DenseSet<Operation *> memberSet(match.memberOps.begin(), match.memberOps.end());
  if (memberSet.insert(op).second) {
    match.memberOps.push_back(op);
  }
}

}  // namespace safe_softmax_dvm
}  // namespace mfuse
}  // namespace mlir
