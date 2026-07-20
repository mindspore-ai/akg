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

#include "mfusion/Analysis/Split/Area.h"
#include "mfusion/Analysis/Split/OpRegister.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/HighLevelOpt/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DECL_FOLDBNSDFLATTENSUMPASS
#define GEN_PASS_DEF_FOLDBNSDFLATTENSUMPASS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

struct PeeledValue {
  Value value;
  Operation *castOp = nullptr;
};

static PeeledValue peelSingleUnrealizedCast(Value value) {
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp->getNumOperands() == 1) {
      return {castOp->getOperand(0), castOp.getOperation()};
    }
  }
  return {value, nullptr};
}

static void eraseDeadOpsInOrder(PatternRewriter &rewriter, ArrayRef<Operation *> ops) {
  llvm::SmallPtrSet<Operation *, 8> seen;
  for (Operation *op : ops) {
    if (op && seen.insert(op).second && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }
}

static std::optional<int64_t> getTorchConstantInt(Value value) {
  Operation *op = value.getDefiningOp();
  if (!op || op->getName().getStringRef() != "torch.constant.int") {
    return std::nullopt;
  }
  auto valueAttr = op->getAttrOfType<IntegerAttr>("value");
  if (!valueAttr) {
    return std::nullopt;
  }
  return valueAttr.getInt();
}

static bool isContiguousClone(Operation *op) {
  if (!op || op->getName().getStringRef() != "torch.aten.clone" || op->getNumOperands() < 2 ||
      op->getNumResults() != 1) {
    return false;
  }
  auto memoryFormat = getTorchConstantInt(op->getOperand(1));
  return memoryFormat && *memoryFormat == 0;
}

static bool readI64Array(ArrayAttr attr, SmallVectorImpl<int64_t> &values) {
  if (!attr) {
    return false;
  }
  values.clear();
  values.reserve(attr.size());
  for (Attribute item : attr) {
    auto intAttr = dyn_cast<IntegerAttr>(item);
    if (!intAttr) {
      return false;
    }
    values.push_back(intAttr.getInt());
  }
  return true;
}

static bool dimsEqual(ArrayAttr attr, ArrayRef<int64_t> expected) {
  SmallVector<int64_t> values;
  return readI64Array(attr, values) && values == expected;
}

static bool isInsideMfuseFusedOp(Operation *op) { return op->getParentOfType<FusedOp>() != nullptr; }

static RankedTensorType getStaticRankedType(Type type, int64_t rank) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (!rankedType || !rankedType.hasStaticShape() || rankedType.getRank() != rank) {
    return {};
  }
  return rankedType;
}

// Guard A: DVM clusterable producer. The rewrite is only profitable when the
// new `producer -> reduce_sum` edge can be consumed by the DVM Cluster/Split
// pass as a pointwise-to-reduce fusion. We approximate "DVM clusterable
// pointwise producer" with the Split OpRegistry classification
// `NodePattern::ELEMWISE`, plus a side-effect-free and same-block check. This is
// a conservative profitability approximation, not a complete DVM legality oracle.
static bool hasDvmElemwiseReduceFusionOpportunity(Value sourceValue, Operation *reduceOp) {
  Operation *producer = sourceValue.getDefiningOp();
  if (!producer) {
    return false;  // Block argument / opaque input: no fusible producer.
  }
  if (producer->getBlock() != reduceOp->getBlock()) {
    return false;
  }
  if (!producer->getName().getStringRef().starts_with("mfuse.")) {
    return false;
  }
  if (!isMemoryEffectFree(producer)) {
    return false;
  }
  auto pattern = split::OpRegistry::Instance().GetPattern(producer->getName().getStringRef().str());
  return pattern == split::NodePattern::ELEMWISE;
}

struct BnsdFlattenSumMatch {
  ReshapeOp flatReshape;
  Operation *flatInputCast = nullptr;
  Operation *cloneOp = nullptr;
  Operation *cloneInputCast = nullptr;
  PermuteOp permuteOp;
  RankedTensorType reduceType;
  int64_t group = 0;
  int64_t inner = 0;
};

static bool matchFlattenClonePermuteChain(ReduceSumOp reduceOp, BnsdFlattenSumMatch &match) {
  if (!dimsEqual(reduceOp.getDimensions(), {0})) {
    return false;
  }

  match.flatReshape = reduceOp.getInput().getDefiningOp<ReshapeOp>();
  if (!match.flatReshape) {
    return false;
  }

  auto flatType = getStaticRankedType(match.flatReshape.getResult().getType(), 2);
  match.reduceType = dyn_cast<RankedTensorType>(reduceOp.getResult().getType());
  if (!flatType || !match.reduceType || !match.reduceType.hasStaticShape()) {
    return false;
  }

  PeeledValue flatInput = peelSingleUnrealizedCast(match.flatReshape.getInput());
  match.flatInputCast = flatInput.castOp;
  match.cloneOp = flatInput.value.getDefiningOp();
  if (!isContiguousClone(match.cloneOp)) {
    return false;
  }

  PeeledValue cloneInput = peelSingleUnrealizedCast(match.cloneOp->getOperand(0));
  match.cloneInputCast = cloneInput.castOp;
  match.permuteOp = cloneInput.value.getDefiningOp<PermuteOp>();
  return match.permuteOp && dimsEqual(match.permuteOp.getPermAttr(), {0, 2, 1, 3});
}

static bool matchBnsdShapeContract(ReduceSumOp reduceOp, BnsdFlattenSumMatch &match) {
  auto sourceType = getStaticRankedType(match.permuteOp.getInput().getType(), 4);
  auto permutedType = getStaticRankedType(match.permuteOp.getResult().getType(), 4);
  auto flatType = getStaticRankedType(match.flatReshape.getResult().getType(), 2);
  if (!sourceType || !permutedType || !flatType) {
    return false;
  }

  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  ArrayRef<int64_t> permutedShape = permutedType.getShape();
  if (permutedShape[0] != sourceShape[0] || permutedShape[1] != sourceShape[2] ||
      permutedShape[2] != sourceShape[1] || permutedShape[3] != sourceShape[3]) {
    return false;
  }

  int64_t outer = sourceShape[0];
  match.group = sourceShape[1];
  int64_t reduce = sourceShape[2];
  match.inner = sourceShape[3];
  if (flatType.getShape()[0] != outer * reduce || flatType.getShape()[1] != match.group * match.inner) {
    return false;
  }

  if (reduceOp.getKeepdim()) {
    return match.reduceType.getRank() == 2 && match.reduceType.getShape()[0] == 1 &&
           match.reduceType.getShape()[1] == match.group * match.inner;
  }

  return match.reduceType.getRank() == 1 && match.reduceType.getShape()[0] == match.group * match.inner;
}

struct FoldBnsdFlattenSumPattern : public OpRewritePattern<ReduceSumOp> {
  FoldBnsdFlattenSumPattern(MLIRContext *ctx, bool guardEnabled)
      : OpRewritePattern<ReduceSumOp>(ctx), guardEnabled(guardEnabled) {}

  LogicalResult matchAndRewrite(ReduceSumOp reduceOp, PatternRewriter &rewriter) const override {
    if (isInsideMfuseFusedOp(reduceOp)) {
      return failure();
    }

    BnsdFlattenSumMatch match;
    if (!matchFlattenClonePermuteChain(reduceOp, match) || !matchBnsdShapeContract(reduceOp, match)) {
      return failure();
    }

    if (guardEnabled) {
      Value sourceValue = match.permuteOp.getInput();
      if (!hasDvmElemwiseReduceFusionOpportunity(sourceValue, reduceOp)) {
        return failure();
      }
    }

    auto directReduceType = RankedTensorType::get({match.group, match.inner}, match.reduceType.getElementType(),
                                                  match.reduceType.getEncoding());
    auto directReduce = rewriter.create<ReduceSumOp>(reduceOp.getLoc(), directReduceType,
                                                     match.permuteOp.getInput(), rewriter.getI64ArrayAttr({0, 2}),
                                                     rewriter.getBoolAttr(false));
    Operation *memoryFormatConst = match.cloneOp->getOperand(1).getDefiningOp();
    rewriter.replaceOpWithNewOp<ReshapeOp>(reduceOp, match.reduceType, directReduce.getResult());
    eraseDeadOpsInOrder(rewriter, {match.flatReshape.getOperation(), match.flatInputCast, match.cloneOp,
                                   match.cloneInputCast, match.permuteOp.getOperation(), memoryFormatConst});
    return success();
  }

  bool guardEnabled;
};

struct FoldBnsdFlattenSumPass : public impl::FoldBnsdFlattenSumPassBase<FoldBnsdFlattenSumPass> {
  using impl::FoldBnsdFlattenSumPassBase<FoldBnsdFlattenSumPass>::FoldBnsdFlattenSumPassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<FoldBnsdFlattenSumPattern>(funcOp.getContext(), this->guard.getValue());
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createFoldBnsdFlattenSumPass() {
  return std::make_unique<FoldBnsdFlattenSumPass>();
}

}  // namespace mfuse
}  // namespace mlir
