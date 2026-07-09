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
#include "mfusion/Analysis/FusionRegion/FusionRegionTag.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/IR/MfuseAttributes.h"
#include "mfusion/Dialect/Mfuse/Support/FusedOpUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Decompose/ComputeOpBuilder.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace mfuse {
namespace {

constexpr llvm::StringLiteral kTorchConstantIntOpName = "torch.constant.int";
constexpr llvm::StringLiteral kTorchConstantBoolOpName = "torch.constant.bool";
constexpr llvm::StringLiteral kAtenAnyDimOpName = "torch.aten.any.dim";

static bool hasScalarMarker(RankedTensorType type) {
  auto dictAttr = dyn_cast_or_null<DictionaryAttr>(type.getEncoding());
  return dictAttr && dictAttr.contains(kScalarMarkerAttr);
}

/// Materialize a comparison op's scalar operand as f32 inside the cluster for dvm.binary_scalar.
/// Works for both EqOp and NeOp (Albert-style attention masks).
template <typename CmpOp>
static ConstantOp materializeAlbertCmpScalarForDvm(CmpOp cmpOp, PatternRewriter &rewriter) {
  if (!cmpOp) {
    return nullptr;
  }

  RankedTensorType tensorType;
  Value scalarVal;
  if (auto lhsType = dyn_cast<RankedTensorType>(cmpOp.getOperand(0).getType())) {
    tensorType = lhsType;
    scalarVal = cmpOp.getOperand(1);
  } else if (auto rhsType = dyn_cast<RankedTensorType>(cmpOp.getOperand(1).getType())) {
    tensorType = rhsType;
    scalarVal = cmpOp.getOperand(0);
  } else {
    return nullptr;
  }

  auto constOp = scalarVal.getDefiningOp<ConstantOp>();
  if (!constOp) {
    return nullptr;
  }

  auto scalarType = dyn_cast<RankedTensorType>(constOp.getResult().getType());
  if (!scalarType || !hasScalarMarker(scalarType)) {
    return nullptr;
  }

  auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValueAttr());
  if (!denseAttr || denseAttr.getNumElements() != 1) {
    return nullptr;
  }

  Type targetElem = tensorType.getElementType();
  auto replaceScalarOperand = [&](Value newScalar) {
    rewriter.modifyOpInPlace(cmpOp, [&]() {
      if (cmpOp.getOperand(1) == scalarVal) {
        cmpOp.setOperand(1, newScalar);
      } else {
        cmpOp.setOperand(0, newScalar);
      }
    });
  };

  if (scalarType.getElementType() == targetElem) {
    return constOp;
  }

  auto floatType = dyn_cast<FloatType>(targetElem);
  if (!floatType) {
    return nullptr;
  }

  APFloat apf = *denseAttr.getValues<APFloat>().begin();
  bool ignored;
  apf.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &ignored);
  auto newAttr = DenseElementsAttr::get(RankedTensorType::get({}, floatType), apf);
  auto newType = RankedTensorType::get({}, floatType, scalarType.getEncoding());
  // Insert before cmp so fused-body clone maps the scalar before cloning cmp.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(cmpOp.getOperation());
  auto newConst = rewriter.create<ConstantOp>(cmpOp.getLoc(), newType, newAttr);
  replaceScalarOperand(newConst.getResult());
  return newConst;
}

static ConstantOp materializeAlbertEqScalarForDvm(EqOp eqOp, PatternRewriter &rewriter) {
  return materializeAlbertCmpScalarForDvm(eqOp, rewriter);
}

static ConstantOp materializeAlbertNeScalarForDvm(NeOp neOp, PatternRewriter &rewriter) {
  return materializeAlbertCmpScalarForDvm(neOp, rewriter);
}

FailureOr<int64_t> getTorchConstantInt(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getName().getStringRef() != kTorchConstantIntOpName) {
    return failure();
  }
  if (auto attr = def->getAttrOfType<IntegerAttr>("value")) {
    return attr.getInt();
  }
  for (NamedAttribute namedAttr : def->getAttrs()) {
    if (auto attr = dyn_cast<IntegerAttr>(namedAttr.getValue())) {
      return attr.getInt();
    }
  }
  return failure();
}

FailureOr<bool> getTorchConstantBool(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getName().getStringRef() != kTorchConstantBoolOpName) {
    return failure();
  }
  if (auto attr = def->getAttrOfType<BoolAttr>("value")) {
    return attr.getValue();
  }
  for (NamedAttribute namedAttr : def->getAttrs()) {
    if (auto attr = dyn_cast<BoolAttr>(namedAttr.getValue())) {
      return attr.getValue();
    }
  }
  return failure();
}

FailureOr<int64_t> normalizeDim(int64_t dim, int64_t rank) {
  if (rank <= 0) {
    return failure();
  }
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    return failure();
  }
  return dim;
}

bool isAtenAnyDimOp(Operation *op) { return op && op->getName().getStringRef() == kAtenAnyDimOpName; }

static bool isInteger1RankedTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getElementType().isInteger(1);
}

static safe_softmax_dvm::AlbertMaskChainOps buildAlbertMaskChainFromEq(LogicalNotOp condNot, EqOp eqOp,
                                                                          Operation *eqNotOp,
                                                                          Operation *reduceAnyOp) {
  safe_softmax_dvm::AlbertMaskChainOps chain;
  chain.eq = eqOp.getOperation();
  chain.eqNot = eqNotOp;
  chain.reduceAny = reduceAnyOp;
  chain.condNot = condNot.getOperation();
  return chain;
}

static safe_softmax_dvm::AlbertMaskChainOps buildAlbertMaskChainFromNe(LogicalNotOp condNot, NeOp neOp,
                                                                          Operation *reduceAnyOp) {
  safe_softmax_dvm::AlbertMaskChainOps chain;
  chain.ne = neOp.getOperation();
  chain.reduceAny = reduceAnyOp;
  chain.condNot = condNot.getOperation();
  return chain;
}

static FailureOr<safe_softmax_dvm::AlbertMaskChainOps> tryAlbertMaskChainFromExistingReduce(
  LogicalNotOp condNot, ReduceMaxOp existingReduce) {
  Value fullMask = existingReduce.getInput();
  auto preAnyNot = fullMask.getDefiningOp<LogicalNotOp>();
  if (preAnyNot && isInteger1RankedTensor(fullMask)) {
    // Pattern eq -> logical_not -> reduce_max
    auto eqOp = preAnyNot.getInput().getDefiningOp<EqOp>();
    if (eqOp) {
      return buildAlbertMaskChainFromEq(condNot, eqOp, preAnyNot.getOperation(), existingReduce.getOperation());
    }
    // Pattern ne -> logical_not -> reduce_max  (rare: double negation ne(x,-inf)=true → not → false)
    auto neOpBehindNot = preAnyNot.getInput().getDefiningOp<NeOp>();
    if (neOpBehindNot) {
      return buildAlbertMaskChainFromNe(condNot, neOpBehindNot, existingReduce.getOperation());
    }
  }
  // Pattern ne -> reduce_max (when there is no outer logical_not, ne feeds reduce_max directly)
  auto neOpDirect = fullMask.getDefiningOp<NeOp>();
  if (neOpDirect && isInteger1RankedTensor(fullMask)) {
    return buildAlbertMaskChainFromNe(condNot, neOpDirect, existingReduce.getOperation());
  }
  return failure();
}

struct AlbertAnyDimMaskParseResult {
  EqOp eqOp;
  NeOp neOp;
  Operation *preAnyNot = nullptr;
  Value fullMask;
  Operation *postAnyCast;
  Operation *anyOp;
  Operation *preAnyCast;
};

static FailureOr<AlbertAnyDimMaskParseResult> parseAlbertAnyDimMaskChain(Value reducedMask) {
  AlbertAnyDimMaskParseResult result;
  result.postAnyCast = reducedMask.getDefiningOp();
  if (!result.postAnyCast || !isa<UnrealizedConversionCastOp>(result.postAnyCast)) {
    return failure();
  }

  Value anyTorchResult = result.postAnyCast->getOperand(0);
  result.anyOp = anyTorchResult.getDefiningOp();
  if (!isAtenAnyDimOp(result.anyOp) || result.anyOp->getNumOperands() < 3) {
    return failure();
  }

  Value preAnyTorch = result.anyOp->getOperand(0);
  result.preAnyCast = preAnyTorch.getDefiningOp();
  if (!result.preAnyCast || !isa<UnrealizedConversionCastOp>(result.preAnyCast)) {
    return failure();
  }

  result.fullMask = result.preAnyCast->getOperand(0);

  // Pattern A: eq -> logical_not -> any.dim  (original pattern).
  // If the logical_not does not feed an EqOp, intentionally fall through
  // to Pattern B (ne -> any.dim) below — the negated value cannot match
  // NeOp anyway (it is i1-from-not, not i1-from-ne), so Pattern B will
  // simply return failure(), making the overall parse result "no match".
  auto preAnyNotOp = result.fullMask.getDefiningOp<LogicalNotOp>();
  if (preAnyNotOp && isInteger1RankedTensor(result.fullMask)) {
    result.eqOp = preAnyNotOp.getInput().getDefiningOp<EqOp>();
    if (result.eqOp) {
      result.preAnyNot = preAnyNotOp.getOperation();
      return result;
    }
  }

  // Pattern B: ne -> any.dim  (ne absorbs the eq+not, so no preAnyNot).
  // When fullMask is produced by LogicalNotOp but not fed by EqOp, this
  // branch also fails because getDefiningOp<NeOp>() on a LogicalNotOp
  // result returns null — which is correct, since such a chain doesn't
  // match either pattern.
  result.neOp = result.fullMask.getDefiningOp<NeOp>();
  if (result.neOp && isInteger1RankedTensor(result.fullMask)) {
    result.preAnyNot = nullptr;
    return result;
  }

  return failure();
}

static FailureOr<ReduceMaxOp> replaceAlbertAtenAnyDimWithReduceMax(Value reducedMask,
                                                                   const AlbertAnyDimMaskParseResult &parsed,
                                                                   PatternRewriter &rewriter) {
  auto fullMaskType = cast<RankedTensorType>(parsed.fullMask.getType());
  auto dimOr = getTorchConstantInt(parsed.anyOp->getOperand(1));
  if (failed(dimOr)) {
    return failure();
  }
  auto normalizedDimOr = normalizeDim(*dimOr, fullMaskType.getRank());
  if (failed(normalizedDimOr)) {
    return failure();
  }

  bool keepdim = true;
  if (auto keepdimOr = getTorchConstantBool(parsed.anyOp->getOperand(2)); succeeded(keepdimOr)) {
    keepdim = *keepdimOr;
  }

  auto dimsAttr = rewriter.getI64ArrayAttr({*normalizedDimOr});
  auto keepdimAttr = rewriter.getBoolAttr(keepdim);
  auto reducedType =
    ReduceMaxOp::inferResultType(parsed.fullMask, dimsAttr, keepdimAttr, fullMaskType.getElementType());
  if (!reducedType) {
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  // Determine insertion point so the new ReduceMaxOp dominates its users:
  //   - Pattern A (eq-based): insert after the logical_not that wraps eq,
  //     because fullMask (which feeds reduce_max) depends on that not.
  //   - Pattern B (ne-based): insert after the ne op itself, since
  //     fullMask is the ne result and there is no preAnyNot.
  //   - If neither condition applies (should not happen after a successful
  //     parse), bail out.
  Operation *insertAfter = parsed.preAnyNot ? parsed.preAnyNot : parsed.neOp ? &*parsed.neOp : nullptr;
  if (!insertAfter) {
    return failure();
  }
  rewriter.setInsertionPointAfter(insertAfter);
  Location loc = insertAfter->getLoc();
  auto reduceAny =
    rewriter.create<ReduceMaxOp>(loc, reducedType, parsed.fullMask, dimsAttr, keepdimAttr);
  rewriter.replaceAllUsesWith(reducedMask, reduceAny.getResult());
  if (parsed.postAnyCast->use_empty()) {
    rewriter.eraseOp(parsed.postAnyCast);
  }
  if (parsed.anyOp->use_empty()) {
    rewriter.eraseOp(parsed.anyOp);
  }
  if (parsed.preAnyCast->use_empty()) {
    rewriter.eraseOp(parsed.preAnyCast);
  }
  return reduceAny;
}

/// Albert-style mask canonicalization.
/// Pattern A: eq -> logical_not -> any.dim -> logical_not -> select
/// Pattern B: ne -> any.dim -> logical_not -> select  (ne absorbs eq+not)
/// Replace torch.aten.any.dim with mfuse.reduce_max on i1 so the chain can fuse into DVM.
FailureOr<safe_softmax_dvm::AlbertMaskChainOps> canonicalizeAlbertMaskChain(SelectOp selectOp,
                                                                              PatternRewriter &rewriter) {
  auto condNot = selectOp.getCondition().getDefiningOp<LogicalNotOp>();
  if (!condNot) {
    return failure();
  }

  Value reducedMask = condNot.getInput();
  if (auto existingReduce = dyn_cast_or_null<ReduceMaxOp>(reducedMask.getDefiningOp())) {
    return tryAlbertMaskChainFromExistingReduce(condNot, existingReduce);
  }

  auto parsedOr = parseAlbertAnyDimMaskChain(reducedMask);
  if (failed(parsedOr)) {
    return failure();
  }

  auto reduceAnyOr = replaceAlbertAtenAnyDimWithReduceMax(reducedMask, *parsedOr, rewriter);
  if (failed(reduceAnyOr)) {
    return failure();
  }

  if (parsedOr->eqOp) {
    return buildAlbertMaskChainFromEq(condNot, parsedOr->eqOp, parsedOr->preAnyNot,
                              reduceAnyOr->getOperation());
  }
  return buildAlbertMaskChainFromNe(condNot, parsedOr->neOp, reduceAnyOr->getOperation());
}

FailureOr<Value> getSoftmaxInput(Operation *softmaxOp) {
  if (!softmaxOp || softmaxOp->getNumOperands() < 1) {
    return failure();
  }
  Value input = softmaxOp->getOperand(0);
  if (isa<RankedTensorType>(input.getType())) {
    return input;
  }
  if (auto cast = input.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() == 1 && isa<RankedTensorType>(cast.getOperand(0).getType())) {
      return cast.getOperand(0);
    }
  }
  return failure();
}

FailureOr<int64_t> getSoftmaxDim(Operation *softmaxOp, RankedTensorType inputType) {
  int64_t dim = inputType.getRank() - 1;
  if (auto softmax = dyn_cast<SoftmaxOp>(softmaxOp)) {
    dim = softmax.getDim();
  } else if (softmaxOp->getNumOperands() > 1) {
    if (auto dimOr = getTorchConstantInt(softmaxOp->getOperand(1)); succeeded(dimOr)) {
      dim = *dimOr;
    }
  }
  return normalizeDim(dim, inputType.getRank());
}

struct DecomposedSoftmaxChain {
  ReduceMaxOp reduceMax;
  SubOp centerSub;
  ExpOp exp;
  ReduceSumOp reduceSum;
  DivOp div;
};

FailureOr<DecomposedSoftmaxChain> decomposeSoftmaxProducer(Operation *softmaxOp, PatternRewriter &rewriter) {
  if (!safe_softmax_dvm::isSoftmaxProducerOp(softmaxOp)) {
    return failure();
  }

  auto inputOr = getSoftmaxInput(softmaxOp);
  if (failed(inputOr)) {
    return failure();
  }
  Value input = *inputOr;
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return failure();
  }

  auto dimOr = getSoftmaxDim(softmaxOp, inputType);
  if (failed(dimOr)) {
    return failure();
  }

  Location loc = softmaxOp->getLoc();
  rewriter.setInsertionPoint(softmaxOp);
  auto dimsAttr = rewriter.getI64ArrayAttr({*dimOr});
  auto keepdimAttr = rewriter.getBoolAttr(true);

  auto maxType = ReduceMaxOp::inferResultType(input, dimsAttr, keepdimAttr, inputType.getElementType());
  auto sumType = ReduceSumOp::inferResultType(input, dimsAttr, keepdimAttr, inputType.getElementType());
  if (!maxType || !sumType) {
    return failure();
  }

  DecomposedSoftmaxChain chain;
  ComputeOpBuilder builder(rewriter, loc);
  chain.reduceMax = rewriter.create<ReduceMaxOp>(loc, maxType, input, dimsAttr, keepdimAttr);
  Value centered = builder.sub(input, chain.reduceMax.getResult());
  chain.centerSub = centered.getDefiningOp<SubOp>();
  Value unnormExp = builder.exp(centered);
  chain.exp = unnormExp.getDefiningOp<ExpOp>();
  chain.reduceSum = rewriter.create<ReduceSumOp>(loc, sumType, unnormExp, dimsAttr, keepdimAttr);
  Value divResult = builder.div(unnormExp, chain.reduceSum.getResult());
  chain.div = divResult.getDefiningOp<DivOp>();
  if (!chain.centerSub || !chain.exp || !chain.div) {
    return failure();
  }
  return chain;
}

void eraseSoftmaxProducer(Operation *softmaxOp, PatternRewriter &rewriter) {
  // Only erase a cast user once its own result is unused: a softmax result may feed
  // several cast chains (e.g. one into this safe-softmax select, another into an
  // unrelated consumer), and blind erasure of all casts would drop a still-live op.
  for (OpResult result : softmaxOp->getResults()) {
    for (Operation *user : llvm::make_early_inc_range(result.getUsers())) {
      if (isa<UnrealizedConversionCastOp>(user) && user->use_empty()) {
        rewriter.eraseOp(user);
      }
    }
  }
  if (softmaxOp->use_empty()) {
    rewriter.eraseOp(softmaxOp);
  }
}

void tagSafeSoftmaxFusedOp(FusedOp fusedOp, llvm::StringRef groupId) {
  fusion_region::tagMember(fusedOp.getOperation(), groupId, fusion_region::kSafeSoftmaxFuseKind);
}

LogicalResult decomposeFusedSoftmaxCandidate(SelectOp selectOp,
                                             safe_softmax_dvm::FusedSafeSoftmaxCandidate &fusedCandidate,
                                             safe_softmax_dvm::SafeSoftmaxDvmMatch &matchResult,
                                             PatternRewriter &rewriter) {
  auto chainOr = decomposeSoftmaxProducer(fusedCandidate.softmaxOp, rewriter);
  if (failed(chainOr)) {
    return failure();
  }

  DecomposedSoftmaxChain chain = *chainOr;
  Value divResult = chain.div.getResult();
  if (fusedCandidate.softmaxOutput != divResult) {
    rewriter.replaceAllUsesWith(fusedCandidate.softmaxOutput, divResult);
  }
  eraseSoftmaxProducer(fusedCandidate.softmaxOp, rewriter);

  matchResult.matched = true;
  matchResult.select = selectOp;
  matchResult.full = fusedCandidate.full;
  matchResult.zeroBranch = fusedCandidate.zeroBranch;
  matchResult.reduceMax = chain.reduceMax;
  matchResult.centerSub = chain.centerSub;
  matchResult.exp = chain.exp;
  matchResult.reduceSum = chain.reduceSum;
  matchResult.softmaxDiv = chain.div;
  return success();
}

class FuseSafeSoftmaxDvmPattern : public OpRewritePattern<SelectOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp selectOp, PatternRewriter &rewriter) const override {
    if (selectOp->getParentOfType<FusedOp>()) {
      return failure();
    }
    if (safe_softmax_dvm::isSafeSoftmaxTagged(selectOp.getOperation())) {
      return failure();
    }

    safe_softmax_dvm::SafeSoftmaxDvmMatch matchResult =
      safe_softmax_dvm::matchSafeSoftmaxFromSelect(selectOp);
    if (!matchResult.matched) {
      safe_softmax_dvm::FusedSafeSoftmaxCandidate fusedCandidate =
        safe_softmax_dvm::matchFusedSoftmaxCandidateFromSelect(selectOp);
      if (!fusedCandidate.matched) {
        return rewriter.notifyMatchFailure(selectOp, "not a safe-softmax broadcast select");
      }
      if (failed(decomposeFusedSoftmaxCandidate(selectOp, fusedCandidate, matchResult, rewriter))) {
        return rewriter.notifyMatchFailure(selectOp, "failed to decompose softmax for safe-softmax fusion");
      }
    }

    auto maskChain = canonicalizeAlbertMaskChain(selectOp, rewriter);
    safe_softmax_dvm::collectSafeSoftmaxMemberOps(matchResult);
    if (succeeded(maskChain)) {
      safe_softmax_dvm::appendAlbertMaskChainMemberOps(matchResult, *maskChain);
      if (auto eqOp = dyn_cast_or_null<EqOp>((*maskChain).eq)) {
        if (ConstantOp scalarConst = materializeAlbertEqScalarForDvm(eqOp, rewriter)) {
          safe_softmax_dvm::appendMemberOp(matchResult, scalarConst.getOperation());
        }
      } else if (auto neOp = dyn_cast_or_null<NeOp>((*maskChain).ne)) {
        if (ConstantOp scalarConst = materializeAlbertNeScalarForDvm(neOp, rewriter)) {
          safe_softmax_dvm::appendMemberOp(matchResult, scalarConst.getOperation());
        }
      }
    }

    // Pre-flight DVM legality gate before materializeFusedOpFromBuildInfo: buildCluster
    // only validates SSA/cluster structure. Reuse cluster whitelist +
    // checkDvmOpConstraints (scalar constants bypass the whitelist).
    ClusterBuildInfo buildInfo;
    if (!allMemberOpsDvmSupported(matchResult.memberOps) ||
        !buildCluster(matchResult.memberOps, buildInfo)) {
      return rewriter.notifyMatchFailure(selectOp, "failed to build safe-softmax cluster");
    }

    FusedOp fusedOp;
    if (!materializeFusedOpFromBuildInfo(rewriter, buildInfo, "dvm", fusedOp)) {
      return rewriter.notifyMatchFailure(selectOp, "failed to materialize safe-softmax mfuse.fused");
    }

    std::string groupId = fusion_region::allocateGroupId(fusion_region::kSafeSoftmaxFuseKind);
    tagSafeSoftmaxFusedOp(fusedOp, groupId);

    MLOG(DEBUG) << "FuseSafeSoftmaxDvmPattern: fused safe-softmax region into mfuse.fused " << groupId;
    return success();
  }
};

}  // namespace

}  // namespace mfuse
}  // namespace mlir

namespace mlir {
namespace mfuse {

void registerFuseSafeSoftmaxDvmPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseSafeSoftmaxDvmPattern>(patterns.getContext());
}

}  // namespace mfuse
}  // namespace mlir
