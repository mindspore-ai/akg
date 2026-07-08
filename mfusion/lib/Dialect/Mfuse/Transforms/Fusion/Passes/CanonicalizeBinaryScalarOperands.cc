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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/CanonicalizeBinaryScalarOperands.h"

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mfusion/Dialect/Mfuse/Analysis/BinaryOpCommonInfer.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CANONICALIZEBINARYSCALAROPERANDS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

bool isSupportedBinary(Operation *op) {
  return llvm::isa<AddOp, MulOp, SubOp, DivOp>(op);
}

bool isCommutativeSupportedBinary(Operation *op) {
  return llvm::isa<AddOp, MulOp>(op);
}

bool isScalarConstant(Value value) {
  auto constantOp = value.getDefiningOp<ConstantOp>();
  if (!constantOp) {
    return false;
  }
  return isScalarType(value.getType());
}

struct ScalarOperandInfo {
  Value scalar;
  Operation *numToTensor = nullptr;
};

std::optional<ScalarOperandInfo> getRecoverableScalar(Value value) {
  if (isScalarConstant(value)) {
    return ScalarOperandInfo{value};
  }

  auto numToTensorOp = value.getDefiningOp<NumToTensorOp>();
  if (!numToTensorOp) {
    return std::nullopt;
  }

  Value input = numToTensorOp.getValue();
  if (!isScalarConstant(input)) {
    return std::nullopt;
  }
  return ScalarOperandInfo{input, numToTensorOp.getOperation()};
}

void eraseDeadNumToTensor(PatternRewriter &rewriter, const std::optional<ScalarOperandInfo> &info) {
  if (info && info->numToTensor && info->numToTensor->use_empty()) {
    rewriter.eraseOp(info->numToTensor);
  }
}

// This pass must run before FuseNumToTensor.
//
// FxImporter can represent scalar-lhs Tensor overloads such as `1 - x` as:
//   constant -> num_to_tensor -> mfuse.sub
// FuseNumToTensor would otherwise materialize num_to_tensor as mfuse.full,
// turning a scalar binary operand into a real tensor producer. That blocks DVM
// cluster/type checks for cases like i64 full feeding f32 sub. This pass
// recovers the scalar constant operand first, then leaves unrelated
// num_to_tensor users for FuseNumToTensor.
//
// Before:
//   %c1 = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
//   %n = mfuse.num_to_tensor %c1
//     : tensor<i64, {is_scalar = ""}> -> tensor<si64>
//   %r = mfuse.sub %n, %x
//     : (tensor<si64>, tensor<4x4xf32>) -> tensor<4x4xf32>
//
// After:
//   %c1 = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
//   %r = mfuse.sub %c1, %x
//     : (tensor<i64, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xf32>
//
// For commutative ops, scalar lhs is additionally normalized to rhs:
//   mfuse.add %scalar, %x  ->  mfuse.add %x, %scalar
class CanonicalizeBinaryScalarOperandsPattern : public RewritePattern {
 public:
  explicit CanonicalizeBinaryScalarOperandsPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (!isSupportedBinary(op) || op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return failure();
    }

    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    std::optional<ScalarOperandInfo> lhsScalar = getRecoverableScalar(lhs);
    std::optional<ScalarOperandInfo> rhsScalar = getRecoverableScalar(rhs);
    if (!lhsScalar && !rhsScalar) {
      return failure();
    }

    // Binary ops with two scalar operands are outside this pass' contract. Do not
    // rewrite them into a form that later DVM lowering explicitly rejects.
    if (lhsScalar && rhsScalar) {
      return failure();
    }

    SmallVector<Value, 2> newOperands;
    bool changed = false;
    if (isCommutativeSupportedBinary(op) && lhsScalar) {
      newOperands.push_back(rhs);
      newOperands.push_back(lhsScalar->scalar);
      changed = true;
    } else {
      Value newLhs = lhsScalar ? lhsScalar->scalar : lhs;
      Value newRhs = rhsScalar ? rhsScalar->scalar : rhs;
      newOperands.push_back(newLhs);
      newOperands.push_back(newRhs);
      changed = (newLhs != lhs || newRhs != rhs);
    }

    if (!changed) {
      return failure();
    }

    OperationState newState(op->getLoc(), op->getName());
    newState.addOperands(newOperands);
    llvm::for_each(op->getAttrs(),
                   [&](NamedAttribute attr) { newState.addAttribute(attr.getName(), attr.getValue()); });
    llvm::for_each(op->getResultTypes(), [&](Type type) { newState.addTypes(type); });
    std::string opName = op->getName().getStringRef().str();
    Operation *newOp = rewriter.create(newState);
    rewriter.replaceOp(op, newOp->getResults());
    eraseDeadNumToTensor(rewriter, lhsScalar);
    eraseDeadNumToTensor(rewriter, rhsScalar);
    MLOG(DEBUG) << "Canonicalized binary scalar operands for " << opName;
    return success();
  }
};

}  // namespace

struct CanonicalizeBinaryScalarOperandsPass
    : public impl::CanonicalizeBinaryScalarOperandsBase<CanonicalizeBinaryScalarOperandsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CanonicalizeBinaryScalarOperandsPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createCanonicalizeBinaryScalarOperandsPass() {
  return std::make_unique<CanonicalizeBinaryScalarOperandsPass>();
}

}  // namespace mfuse
}  // namespace mlir
