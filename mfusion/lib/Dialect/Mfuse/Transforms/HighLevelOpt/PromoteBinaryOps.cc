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

#include "mfusion/Dialect/Mfuse/Transforms/HighLevelOpt/Passes.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Analysis/BinaryOpCommonInfer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

#define GEN_PASS_DEF_PROMOTEBINARYOPSPASS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

struct BinaryOpInfo {
  Value lhs, rhs;
  RankedTensorType lhsType, rhsType, resultType;
  Type lhsElem, rhsElem, resultElem;
  bool lhsIsScalar;
  bool rhsIsScalar;
  bool isArithmetic;
  bool isComparison;

  static std::optional<BinaryOpInfo> fromOp(Operation *op) {
    bool isArith = llvm::isa<mfuse::AddOp, mfuse::DivOp, mfuse::MaximumOp, mfuse::MinimumOp, mfuse::MulOp, mfuse::PowOp,
                             mfuse::RealDivOp, mfuse::SubOp>(op);
    bool isComp = llvm::isa<mfuse::EqOp, mfuse::GeOp, mfuse::GtOp, mfuse::LeOp, mfuse::LogicalAndOp, mfuse::LogicalOrOp,
                            mfuse::LtOp, mfuse::NeOp>(op);
    if (!isArith && !isComp) return std::nullopt;

    if (op->getNumOperands() < 2 || op->getNumResults() < 1) return std::nullopt;

    auto curLhsType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto curRhsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto curResultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!curLhsType || !curRhsType || !curResultType) return std::nullopt;

    return BinaryOpInfo{op->getOperand(0),
                        op->getOperand(1),
                        curLhsType,
                        curRhsType,
                        curResultType,
                        curLhsType.getElementType(),
                        curRhsType.getElementType(),
                        curResultType.getElementType(),
                        isScalarType(curLhsType),
                        isScalarType(curRhsType),
                        isArith,
                        isComp};
  }

  bool needsPromotion() const {
    Type targetElem = getTargetElemType();
    if (!targetElem) return false;
    bool exactlyOneScalar = lhsIsScalar != rhsIsScalar;
    if (exactlyOneScalar) {
      if (isComparison) return false;
      Type tensorElem = lhsIsScalar ? rhsElem : lhsElem;
      return tensorElem != targetElem;
    }
    if (lhsIsScalar && rhsIsScalar) return false;
    if (isArithmetic) {
      bool bothMatch = (lhsElem == resultElem) && (rhsElem == resultElem);
      return !bothMatch;
    }
    if (isComparison) {
      return lhsElem != rhsElem;
    }
    return false;
  }

  Type getTargetElemType() const {
    if (isArithmetic) return resultElem;
    if (isComparison) return inferBinaryOpResultType(lhsElem, rhsElem);
    return nullptr;
  }
};

struct PromoteBinaryOpPattern : public RewritePattern {
  explicit PromoteBinaryOpPattern(MLIRContext *context) : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto info = BinaryOpInfo::fromOp(op);
    if (!info || !info->needsPromotion()) return mlir::failure();
    Type targetElem = info->getTargetElemType();
    if (!targetElem) return mlir::failure();
    // Create new operands with target element type
    Value newLhs = info->lhs;
    Value newRhs = info->rhs;
    if (info->lhsElem != targetElem && !info->lhsIsScalar) {
      auto newType = RankedTensorType::get(info->lhsType.getShape(), targetElem, info->lhsType.getEncoding());
      newLhs = rewriter.create<CastOp>(op->getLoc(), newType, info->lhs);
    }
    if (info->rhsElem != targetElem && !info->rhsIsScalar) {
      auto newType = RankedTensorType::get(info->rhsType.getShape(), targetElem, info->rhsType.getEncoding());
      newRhs = rewriter.create<CastOp>(op->getLoc(), newType, info->rhs);
    }
    OperationState newState(op->getLoc(), op->getName());
    newState.addOperands({newLhs, newRhs});
    llvm::for_each(op->getAttrs(), [&](auto &attr) { newState.addAttribute(attr.getName(), attr.getValue()); });
    llvm::for_each(op->getResults(), [&](auto result) { newState.types.push_back(result.getType()); });
    rewriter.replaceOp(op, rewriter.create(newState)->getResults());
    return mlir::success();
  }
};

}  // namespace

struct PromoteBinaryOpsPass : public impl::PromoteBinaryOpsPassBase<PromoteBinaryOpsPass> {
  void runOnOperation() override {
    getOperation().walk([this](func::FuncOp func) {
      func.walk([this](FusedOp fusedOp) {
        RewritePatternSet patterns(fusedOp.getContext());
        patterns.add<PromoteBinaryOpPattern>(fusedOp.getContext());
        if (failed(applyPatternsAndFoldGreedily(fusedOp.getRegion(), std::move(patterns)))) {
          signalPassFailure();
        }
      });
    });
  }
};

std::unique_ptr<Pass> createPromoteBinaryOpsPass() { return std::make_unique<PromoteBinaryOpsPass>(); }

}  // namespace mfuse
}  // namespace mlir
