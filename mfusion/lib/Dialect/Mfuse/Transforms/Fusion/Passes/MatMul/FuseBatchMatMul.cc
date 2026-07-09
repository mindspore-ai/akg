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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/MatMul/FuseBatchMatMul.h"

#include <optional>

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/SymbolAttrUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Dialect/Mfuse/Support/OpConstants.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEBATCHMATMUL
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

/// Rank required for swapping the last two dimensions.
constexpr int64_t kRank2D = static_cast<int64_t>(kDim2);

/// Returns true if the permute only swaps the last two dimensions:
/// input (..., a, b) -> output (..., b, a); batch dims unchanged.
static bool isPermuteSwapLastTwoDims(PermuteOp permuteOp) {
  auto inputType = dyn_cast<RankedTensorType>(permuteOp.getInput().getType());
  if (!inputType) {
    return false;
  }
  int64_t rank = inputType.getRank();
  if (rank < kRank2D) {
    return false;
  }
  auto permAttr = permuteOp.getPermAttr();
  if (!permAttr) {
    return false;
  }
  auto permValues = permAttr.getValue();
  if (permValues.size() != static_cast<size_t>(rank)) {
    return false;
  }
  int64_t lastIdx = rank - 1;
  int64_t secondLastIdx = rank - 2;
  for (int64_t i = 0; i < rank; ++i) {
    auto intAttr = dyn_cast<IntegerAttr>(permValues[i]);
    if (!intAttr) {
      return false;
    }
    int64_t p = intAttr.getInt();
    if (i < secondLastIdx) {
      if (p != i) {
        return false;
      }
    } else if (i == secondLastIdx) {
      if (p != lastIdx) {
        return false;
      }
    } else {
      if (p != secondLastIdx) {
        return false;
      }
    }
  }
  return true;
}

/// Matmul input dtypes only support float16, float32, bfloat16.
static bool isSupportedMatmulDtype(Type type) {
  auto elemType = dyn_cast<FloatType>(type);
  if (!elemType) return false;
  return isa<Float16Type>(elemType) || elemType.isF32() || isa<BFloat16Type>(elemType);
}

struct BroadcastedPermuteFoldPlan {
  Value operand;
  Value replacementInput;
  RankedTensorType replacementType;
  std::optional<Location> replacementLoc;
  bool needsBroadcastRewrite{false};
  bool toggleTranspose{false};
};

static bool hasOnlyMatmulLikeUsers(Value value) {
  return llvm::all_of(value.getUsers(), [](Operation *user) { return isa<MatmulOp, MatmulWithBiasOp>(user); });
}

// Match matmul operands of the form:
//   broadcast_to(permute(x))
// and prepare the equivalent operand:
//   broadcast_to(x) + toggled matmul transpose flag.
// The rewrite is only considered when all users of the broadcast are matmul-like.
// If a non-matmul user exists, keeping the original broadcast/permute chain is
// more conservative: the matmul-only form would otherwise split one shared value
// into old and new representations for different consumers.
static FailureOr<BroadcastedPermuteFoldPlan> analyzeBroadcastedPermuteOperand(Value operand, OpBuilder &builder) {
  BroadcastedPermuteFoldPlan plan;
  plan.operand = operand;

  auto broadcastOp = operand.getDefiningOp<BroadcastToOp>();
  if (!broadcastOp) {
    return plan;
  }

  auto permuteOp = broadcastOp.getInput().getDefiningOp<PermuteOp>();
  if (!permuteOp || !isPermuteSwapLastTwoDims(permuteOp)) {
    return plan;
  }
  if (!hasOnlyMatmulLikeUsers(broadcastOp.getResult())) {
    return plan;
  }

  auto rootType = dyn_cast<RankedTensorType>(permuteOp.getInput().getType());
  auto permType = dyn_cast<RankedTensorType>(permuteOp.getResult().getType());
  auto outType = dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
  if (!rootType || !permType || !outType) {
    return failure();
  }

  int64_t rootRank = rootType.getRank();
  int64_t outRank = outType.getRank();
  if (rootRank < kRank2D || permType.getRank() != rootRank || outRank < rootRank) {
    return failure();
  }

  int64_t leading = outRank - rootRank;
  auto outShape = outType.getShape();
  llvm::SmallVector<int64_t, 8> newShape(outShape.begin(), outShape.end());
  std::swap(newShape[static_cast<size_t>(leading + rootRank - 2)],
            newShape[static_cast<size_t>(leading + rootRank - 1)]);

  auto newType = RankedTensorType::get(newShape, outType.getElementType(), outType.getEncoding());
  if (SymbolAttrUtils::hasSymbolicShapeEncoding(outType)) {
    auto maybeExprs = SymbolAttrUtils::getSymbolicShapeExprs(outType);
    if (failed(maybeExprs)) {
      return failure();
    }
    llvm::SmallVector<SymbolAttrUtils::SymExpr> newExprs(maybeExprs->begin(), maybeExprs->end());
    std::swap(newExprs[static_cast<size_t>(leading + rootRank - 2)],
              newExprs[static_cast<size_t>(leading + rootRank - 1)]);
    newType = SymbolAttrUtils::withSymbolicAttr(newType, builder, newExprs);
  }

  plan.replacementInput = permuteOp.getInput();
  plan.replacementType = newType;
  plan.replacementLoc = broadcastOp.getLoc();
  plan.needsBroadcastRewrite = true;
  plan.toggleTranspose = true;
  return plan;
}

// Shared broadcast_to(permute(x)) values may feed multiple matmul-like ops. We do a local reuse.
static Value findReusableBroadcastForPlan(const BroadcastedPermuteFoldPlan &plan, Operation *anchorOp) {
  if (!plan.replacementInput || !anchorOp) {
    return {};
  }
  for (Operation *user : plan.replacementInput.getUsers()) {
    auto broadcastOp = dyn_cast<BroadcastToOp>(user);
    if (!broadcastOp) {
      continue;
    }
    if (broadcastOp.getInput() != plan.replacementInput || broadcastOp.getResult().getType() != plan.replacementType) {
      continue;
    }
    if (broadcastOp->getBlock() != anchorOp->getBlock() || !broadcastOp->isBeforeInBlock(anchorOp)) {
      continue;
    }
    return broadcastOp.getResult();
  }
  return {};
}

static Value materializeOperandRewrite(const BroadcastedPermuteFoldPlan &plan, Operation *anchorOp, Location loc,
                                       PatternRewriter &rewriter) {
  if (!plan.needsBroadcastRewrite) {
    return plan.operand;
  }
  if (Value reusable = findReusableBroadcastForPlan(plan, anchorOp)) {
    return reusable;
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(anchorOp);
  Location broadcastLoc = plan.replacementLoc.has_value() ? *plan.replacementLoc : loc;
  return rewriter.create<BroadcastToOp>(broadcastLoc, plan.replacementType, plan.replacementInput);
}

struct MatmulLikeRewriteState {
  Value lhs;
  Value rhs;
  bool transLhs{false};
  bool transRhs{false};
  bool changed{false};
};

template <typename OpTy>
struct MatmulLikeTransposeEliminationTraits;

template <>
struct MatmulLikeTransposeEliminationTraits<MatmulOp> {
  static Value getLhs(MatmulOp op) { return op.getSelf(); }
  static Value getRhs(MatmulOp op) { return op.getOther(); }
  static bool getTransLhs(MatmulOp op) { return op.getTransX1(); }
  static bool getTransRhs(MatmulOp op) { return op.getTransX2(); }
  static constexpr llvm::StringLiteral kPatternName = "FuseBatchMatMulTransposeMatmulPattern";

  static Operation *create(PatternRewriter &rewriter, MatmulOp op, Value lhs, Value rhs, bool transLhs, bool transRhs) {
    return rewriter
      .create<MatmulOp>(op.getLoc(), op.getResult().getType(), lhs, rhs, rewriter.getBoolAttr(transLhs),
                        rewriter.getBoolAttr(transRhs))
      .getOperation();
  }
};

template <>
struct MatmulLikeTransposeEliminationTraits<MatmulWithBiasOp> {
  static Value getLhs(MatmulWithBiasOp op) { return op.getSelf(); }
  static Value getRhs(MatmulWithBiasOp op) { return op.getOther(); }
  static bool getTransLhs(MatmulWithBiasOp op) { return op.getTransX1(); }
  static bool getTransRhs(MatmulWithBiasOp op) { return op.getTransX2(); }
  static constexpr llvm::StringLiteral kPatternName = "FuseBatchMatMulTransposeMatmulWithBiasPattern";

  static Operation *create(PatternRewriter &rewriter, MatmulWithBiasOp op, Value lhs, Value rhs, bool transLhs,
                           bool transRhs) {
    return rewriter
      .create<MatmulWithBiasOp>(op.getLoc(), op.getResult().getType(), lhs, rhs, op.getBias(),
                                rewriter.getBoolAttr(transLhs), rewriter.getBoolAttr(transRhs))
      .getOperation();
  }
};

template <typename OpTy>
static FailureOr<MatmulLikeRewriteState> computeMatmulLikeRewrite(OpTy op, PatternRewriter &rewriter,
                                                                  Value forcedOldOperand = {},
                                                                  Value forcedNewOperand = {}, bool dryRun = false,
                                                                  const BroadcastedPermuteFoldPlan *lhsPlan = nullptr,
                                                                  const BroadcastedPermuteFoldPlan *rhsPlan = nullptr) {
  using Traits = MatmulLikeTransposeEliminationTraits<OpTy>;

  MatmulLikeRewriteState state{Traits::getLhs(op), Traits::getRhs(op), Traits::getTransLhs(op),
                               Traits::getTransRhs(op), false};
  auto lhsType = dyn_cast<RankedTensorType>(state.lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(state.rhs.getType());
  if (!lhsType || !rhsType) {
    return failure();
  }
  if (!isSupportedMatmulDtype(lhsType.getElementType()) || !isSupportedMatmulDtype(rhsType.getElementType())) {
    return failure();
  }

  auto rewriteOperand = [&](Value &operand, bool &transpose,
                            const BroadcastedPermuteFoldPlan *precomputedPlan) -> LogicalResult {
    if (forcedOldOperand && operand == forcedOldOperand) {
      operand = forcedNewOperand;
      transpose ^= true;
      state.changed = true;
      return success();
    }

    std::optional<BroadcastedPermuteFoldPlan> analyzedPlan;
    const BroadcastedPermuteFoldPlan *folded = precomputedPlan;
    if (!folded) {
      auto maybeFolded = analyzeBroadcastedPermuteOperand(operand, rewriter);
      if (failed(maybeFolded)) {
        return failure();
      }
      analyzedPlan = *maybeFolded;
      folded = &*analyzedPlan;
    }
    if (folded->needsBroadcastRewrite) {
      if (!dryRun) {
        operand = materializeOperandRewrite(*folded, op, op.getLoc(), rewriter);
      }
      transpose ^= folded->toggleTranspose;
      state.changed = true;
      return success();
    }

    auto permuteOp = operand.getDefiningOp<PermuteOp>();
    if (permuteOp && isPermuteSwapLastTwoDims(permuteOp)) {
      operand = permuteOp.getInput();
      transpose ^= true;
      state.changed = true;
    }
    return success();
  };

  if (failed(rewriteOperand(state.lhs, state.transLhs, lhsPlan)) ||
      failed(rewriteOperand(state.rhs, state.transRhs, rhsPlan))) {
    return failure();
  }

  return state;
}

template <typename UserOpTy>
static LogicalResult checkMatmulLikeOpWithSharedBroadcast(UserOpTy userOp, PatternRewriter &rewriter,
                                                          Value oldOperand) {
  auto rewriteState = computeMatmulLikeRewrite(userOp, rewriter, oldOperand, oldOperand, /*dryRun=*/true);
  return success(succeeded(rewriteState) && rewriteState->changed);
}

template <typename UserOpTy>
static LogicalResult rewriteMatmulLikeOpWithSharedBroadcast(UserOpTy userOp, PatternRewriter &rewriter,
                                                            Value oldOperand, Value newOperand) {
  using Traits = MatmulLikeTransposeEliminationTraits<UserOpTy>;
  auto rewriteState = computeMatmulLikeRewrite(userOp, rewriter, oldOperand, newOperand);
  if (failed(rewriteState) || !rewriteState->changed) {
    return failure();
  }

  auto oldOpName = userOp->getName().getStringRef();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(userOp);
  Operation *newOp = Traits::create(rewriter, userOp, rewriteState->lhs, rewriteState->rhs, rewriteState->transLhs,
                                    rewriteState->transRhs);
  MLOG(DEBUG) << "FuseBatchMatMul: created " << newOp->getName().getStringRef() << "@" << newOp->getLoc()
              << " lhs_trans=" << rewriteState->transLhs << " rhs_trans=" << rewriteState->transRhs;
  rewriter.replaceOp(userOp, newOp->getResults());
  MLOG(DEBUG) << "FuseBatchMatMul: replaced " << oldOpName << " with transpose-eliminated op";
  return success();
}

static LogicalResult rewriteSharedBroadcastedPermuteUsers(const BroadcastedPermuteFoldPlan &plan,
                                                          PatternRewriter &rewriter) {
  if (!plan.needsBroadcastRewrite || !plan.operand || !plan.replacementLoc.has_value()) {
    return failure();
  }

  SmallVector<Operation *> users(plan.operand.getUsers().begin(), plan.operand.getUsers().end());
  if (users.empty()) {
    return failure();
  }
  auto sourceOp = plan.operand.getDefiningOp();
  if (!sourceOp) {
    return failure();
  }
  if (!llvm::all_of(users, [&](Operation *user) { return user->getBlock() == sourceOp->getBlock(); })) {
    return failure();
  }
  std::sort(users.begin(), users.end(), [](Operation *lhs, Operation *rhs) { return lhs->isBeforeInBlock(rhs); });

  // Validate every user before mutating IR. Pattern rewrites should not replace
  // a prefix of users and then fail on a later user, because the greedy driver
  // cannot roll those edits back for us.
  for (Operation *user : users) {
    if (auto matmulOp = dyn_cast<MatmulOp>(user)) {
      if (failed(checkMatmulLikeOpWithSharedBroadcast(matmulOp, rewriter, plan.operand))) {
        return failure();
      }
      continue;
    }
    if (auto matmulWithBiasOp = dyn_cast<MatmulWithBiasOp>(user)) {
      if (failed(checkMatmulLikeOpWithSharedBroadcast(matmulWithBiasOp, rewriter, plan.operand))) {
        return failure();
      }
      continue;
    }
    return failure();
  }

  Value newBroadcast = materializeOperandRewrite(plan, users.front(), *plan.replacementLoc, rewriter);

  for (Operation *user : users) {
    if (auto matmulOp = dyn_cast<MatmulOp>(user)) {
      if (failed(rewriteMatmulLikeOpWithSharedBroadcast(matmulOp, rewriter, plan.operand, newBroadcast))) {
        return failure();
      }
      continue;
    }
    if (auto matmulWithBiasOp = dyn_cast<MatmulWithBiasOp>(user)) {
      if (failed(rewriteMatmulLikeOpWithSharedBroadcast(matmulWithBiasOp, rewriter, plan.operand, newBroadcast))) {
        return failure();
      }
      continue;
    }
    return failure();
  }

  return success();
}

static const BroadcastedPermuteFoldPlan *selectSharedBroadcastedPermutePlan(const BroadcastedPermuteFoldPlan &lhsPlan,
                                                                            const BroadcastedPermuteFoldPlan &rhsPlan) {
  llvm::SmallVector<const BroadcastedPermuteFoldPlan *, 2> candidates;
  if (lhsPlan.needsBroadcastRewrite && lhsPlan.operand) {
    candidates.push_back(&lhsPlan);
  }
  if (rhsPlan.needsBroadcastRewrite && rhsPlan.operand) {
    candidates.push_back(&rhsPlan);
  }
  if (candidates.empty()) {
    return nullptr;
  }
  if (candidates.size() == 1) {
    return candidates.front();
  }

  Operation *lhsSource = candidates[0]->operand.getDefiningOp();
  Operation *rhsSource = candidates[1]->operand.getDefiningOp();
  if (!lhsSource || !rhsSource || lhsSource == rhsSource || lhsSource->getBlock() != rhsSource->getBlock()) {
    return candidates.front();
  }
  return lhsSource->isBeforeInBlock(rhsSource) ? candidates[0] : candidates[1];
}

template <typename OpTy>
static LogicalResult rewriteMatmulLikeTransposeElimination(OpTy op, PatternRewriter &rewriter) {
  using Traits = MatmulLikeTransposeEliminationTraits<OpTy>;

  Value lhs = Traits::getLhs(op);
  Value rhs = Traits::getRhs(op);
  auto lhsFolded = analyzeBroadcastedPermuteOperand(lhs, rewriter);
  if (failed(lhsFolded)) {
    return failure();
  }
  auto rhsFolded = analyzeBroadcastedPermuteOperand(rhs, rewriter);
  if (failed(rhsFolded)) {
    return failure();
  }

  if (const BroadcastedPermuteFoldPlan *sharedPlan = selectSharedBroadcastedPermutePlan(*lhsFolded, *rhsFolded)) {
    return rewriteSharedBroadcastedPermuteUsers(*sharedPlan, rewriter);
  }

  const BroadcastedPermuteFoldPlan &lhsPlan = *lhsFolded;
  const BroadcastedPermuteFoldPlan &rhsPlan = *rhsFolded;
  auto rewriteState = computeMatmulLikeRewrite(op, rewriter, Value(), Value(), /*dryRun=*/false, &lhsPlan, &rhsPlan);
  if (failed(rewriteState) || !rewriteState->changed) {
    return failure();
  }

  MLOG(DEBUG) << Traits::kPatternName << " matched " << op->getName().getStringRef() << "@" << op.getLoc()
              << " (transpose elimination: lhs_trans=" << rewriteState->transLhs
              << ", rhs_trans=" << rewriteState->transRhs << ")";

  auto oldOpName = op->getName().getStringRef();
  Operation *newOp =
    Traits::create(rewriter, op, rewriteState->lhs, rewriteState->rhs, rewriteState->transLhs, rewriteState->transRhs);
  MLOG(DEBUG) << "FuseBatchMatMul: created " << newOp->getName().getStringRef() << "@" << newOp->getLoc()
              << " lhs_trans=" << rewriteState->transLhs << " rhs_trans=" << rewriteState->transRhs;
  rewriter.replaceOp(op, newOp->getResults());
  MLOG(DEBUG) << "FuseBatchMatMul: replaced " << oldOpName << " with transpose-eliminated op";
  return success();
}

/// Mode 1: Eliminate Permute (swap last two dims) into MatmulOp by using permute
/// input and flipping trans_x1/trans_x2.
class FuseBatchMatMulTransposeMatmulPattern : public OpRewritePattern<MatmulOp> {
 public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op, PatternRewriter &rewriter) const override {
    return rewriteMatmulLikeTransposeElimination(op, rewriter);
  }
};

/// Mode 1: Eliminate Permute (swap last two dims) into MatmulWithBiasOp by
/// using permute input and flipping trans_x1/trans_x2.
class FuseBatchMatMulTransposeMatmulWithBiasPattern : public OpRewritePattern<MatmulWithBiasOp> {
 public:
  using OpRewritePattern<MatmulWithBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulWithBiasOp op, PatternRewriter &rewriter) const override {
    return rewriteMatmulLikeTransposeElimination(op, rewriter);
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseBatchMatMul, FuseBatchMatMulTransposeMatmulPattern,
                         FuseBatchMatMulTransposeMatmulWithBiasPattern)

}  // namespace mfuse
}  // namespace mlir
