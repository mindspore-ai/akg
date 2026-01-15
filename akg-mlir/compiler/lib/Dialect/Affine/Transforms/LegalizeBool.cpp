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

#include <algorithm>
#include <iterator>

#include "akg/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DECL_LEGALIZEBOOL
#define GEN_PASS_DEF_LEGALIZEBOOL
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace affine {
namespace {

static bool isI1ElemType(Type type) {
  return getElementTypeOrSelf(type).isInteger(1);
}

static Type convertI1ElemToI8(Type type) {
  if (!isI1ElemType(type)) {
    return type;
  }
  MLIRContext *ctx = type.getContext();
  Type i8Type = IntegerType::get(ctx, 8);
  if (type.isInteger(1)) {
    return i8Type;
  }
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::get(memRefType.getShape(), i8Type, memRefType.getLayout(),
                           memRefType.getMemorySpace());
  }
  return type;
}

static Value createI1ToI8ScalarCast(Location loc, Value input, OpBuilder &builder) {
  if (!input.getType().isInteger(1)) {
    return input;
  }
  Value inputF16 = builder.create<arith::UIToFPOp>(loc, builder.getF16Type(), input);
  return builder.create<arith::FPToUIOp>(loc, builder.getI8Type(), inputF16);
}

static Value createI8ToI1ScalarCast(Location loc, Value input, OpBuilder &builder) {
  if (!input.getType().isInteger(8)) {
    return input;
  }
  Value zeroF16 = builder.create<arith::ConstantOp>(loc, FloatAttr::get(builder.getF16Type(), 0.0));
  Value inputF16 = builder.create<arith::UIToFPOp>(loc, builder.getF16Type(), input);
  return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, inputF16, zeroF16);
}

struct LegalizeBoolIntBinaryOpMatcher {
  static bool match(Operation *op) {
    return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp, arith::DivUIOp, arith::RemSIOp,
               arith::RemUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp, arith::ShLIOp,
               arith::ShRSIOp, arith::ShRUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
  }
};

static Value createI1ToI8MemRefCast(Location loc, Value input, MemRefType expectedType,
                                   OpBuilder &builder) {
  if (input.getType() == expectedType) {
    return input;
  }
  auto inputType = dyn_cast<MemRefType>(input.getType());
  if (!inputType) {
    return {};
  }
  if (!inputType.getElementType().isInteger(1) || !expectedType.getElementType().isInteger(8)) {
    return {};
  }
  if (inputType.getShape() != expectedType.getShape() || inputType.getLayout() != expectedType.getLayout() ||
      inputType.getMemorySpace() != expectedType.getMemorySpace()) {
    return {};
  }
  auto castOp = builder.create<UnrealizedConversionCastOp>(loc, TypeRange{expectedType},
                                                          ValueRange{input});
  return castOp.getResult(0);
}

static bool isI1OrI8IntType(Type type) {
  return type.isInteger(1) || type.isInteger(8);
}

static void legalizeIntBinaryOperandsAndResult(Operation *op, OpBuilder &builder) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
    return;
  }

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Type lhsType = lhs.getType();
  Type rhsType = rhs.getType();
  if (!isI1OrI8IntType(lhsType) || !isI1OrI8IntType(rhsType)) {
    return;
  }

  Type desiredType;
  if (lhsType == rhsType) {
    desiredType = lhsType;
  } else if (lhsType.isInteger(8) && rhsType.isInteger(1)) {
    desiredType = lhsType;
    rhs = createI1ToI8ScalarCast(op->getLoc(), rhs, builder);
    op->setOperand(1, rhs);
  } else if (lhsType.isInteger(1) && rhsType.isInteger(8)) {
    desiredType = rhsType;
    lhs = createI1ToI8ScalarCast(op->getLoc(), lhs, builder);
    op->setOperand(0, lhs);
  } else {
    return;
  }

  if (op->getResult(0).getType() != desiredType) {
    op->getResult(0).setType(desiredType);
  }
}

static void legalizeCmpIOperands(Operation *op, OpBuilder &builder) {
  auto cmpOp = dyn_cast<arith::CmpIOp>(op);
  if (!cmpOp || cmpOp.getNumOperands() != 2) {
    return;
  }
  Value lhs = cmpOp.getLhs();
  Value rhs = cmpOp.getRhs();
  Type lhsType = lhs.getType();
  Type rhsType = rhs.getType();
  if (lhsType == rhsType) {
    return;
  }
  if (lhsType.isInteger(8) && rhsType.isInteger(1)) {
    rhs = createI1ToI8ScalarCast(op->getLoc(), rhs, builder);
    cmpOp->setOperand(1, rhs);
    return;
  }
  if (lhsType.isInteger(1) && rhsType.isInteger(8)) {
    lhs = createI1ToI8ScalarCast(op->getLoc(), lhs, builder);
    cmpOp->setOperand(0, lhs);
    return;
  }
}

static void updateFunctionType(func::FuncOp func) {
  OpBuilder builder(func);

  FunctionType oldType = func.getFunctionType();
  SmallVector<Type, 4> newInputs;
  SmallVector<Type, 4> newResults;
  newInputs.reserve(oldType.getNumInputs());
  newResults.reserve(oldType.getNumResults());

  std::transform(oldType.getInputs().begin(), oldType.getInputs().end(), std::back_inserter(newInputs),
                 convertI1ElemToI8);
  std::transform(oldType.getResults().begin(), oldType.getResults().end(), std::back_inserter(newResults),
                 convertI1ElemToI8);

  FunctionType newType = builder.getFunctionType(newInputs, newResults);
  if (newType == oldType) {
    return;
  }

  func.setType(newType);
  if (func.empty()) {
    return;
  }

  Block &entry = func.getBody().front();
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    entry.getArgument(i).setType(newInputs[i]);
  }
}

static WalkResult legalizeMemrefLoadOp(memref::LoadOp loadOp) {
  auto memrefType = dyn_cast<MemRefType>(loadOp.getMemRef().getType());
  if (!memrefType) {
    return WalkResult::advance();
  }
  Type expectedType = memrefType.getElementType();
  if (loadOp.getType() != expectedType) {
    loadOp.getResult().setType(expectedType);
  }
  return WalkResult::advance();
}

static WalkResult legalizeMemrefStoreOp(memref::StoreOp storeOp) {
  auto memrefType = dyn_cast<MemRefType>(storeOp.getMemRef().getType());
  if (!memrefType) {
    return WalkResult::advance();
  }
  Type expectedType = memrefType.getElementType();
  Value value = storeOp.getValueToStore();
  if (value.getType() == expectedType) {
    return WalkResult::advance();
  }
  if (value.getType().isInteger(1) && expectedType.isInteger(8)) {
    OpBuilder builder(storeOp);
    builder.setInsertionPoint(storeOp);
    Value casted = createI1ToI8ScalarCast(storeOp.getLoc(), value, builder);
    storeOp->setOperand(0, casted);
  } else if (value.getType().isInteger(8) && expectedType.isInteger(1)) {
    OpBuilder builder(storeOp);
    builder.setInsertionPoint(storeOp);
    Value casted = createI8ToI1ScalarCast(storeOp.getLoc(), value, builder);
    storeOp->setOperand(0, casted);
  }
  return WalkResult::advance();
}

static WalkResult legalizeMemrefExpandShapeOp(memref::ExpandShapeOp expandOp) {
  auto srcType = dyn_cast<MemRefType>(expandOp.getSrc().getType());
  auto dstType = dyn_cast<MemRefType>(expandOp.getResult().getType());
  if (!srcType || !dstType) {
    return WalkResult::advance();
  }
  Type expectedElemType = srcType.getElementType();
  if (dstType.getElementType() != expectedElemType) {
    expandOp.getResult().setType(
        MemRefType::get(dstType.getShape(), expectedElemType, dstType.getLayout(), dstType.getMemorySpace()));
  }
  return WalkResult::advance();
}

static WalkResult legalizeMemrefCollapseShapeOp(memref::CollapseShapeOp collapseOp) {
  auto srcType = dyn_cast<MemRefType>(collapseOp.getSrc().getType());
  auto dstType = dyn_cast<MemRefType>(collapseOp.getResult().getType());
  if (!srcType || !dstType) {
    return WalkResult::advance();
  }
  Type expectedElemType = srcType.getElementType();
  if (dstType.getElementType() != expectedElemType) {
    collapseOp.getResult().setType(
        MemRefType::get(dstType.getShape(), expectedElemType, dstType.getLayout(), dstType.getMemorySpace()));
  }
  return WalkResult::advance();
}

static WalkResult legalizeArithSelectOp(arith::SelectOp selectOp) {
  Value cond = selectOp.getCondition();
  if (cond.getType().isInteger(8)) {
    OpBuilder builder(selectOp);
    builder.setInsertionPoint(selectOp);
    Value i1Cond = createI8ToI1ScalarCast(selectOp.getLoc(), cond, builder);
    selectOp->setOperand(0, i1Cond);
  }
  return WalkResult::advance();
}

static WalkResult legalizeArithIntBinaryOp(Operation *op) {
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  legalizeIntBinaryOperandsAndResult(op, builder);
  return WalkResult::advance();
}

static WalkResult legalizeArithCmpIOp(arith::CmpIOp cmpOp) {
  OpBuilder builder(cmpOp);
  builder.setInsertionPoint(cmpOp);
  legalizeCmpIOperands(cmpOp, builder);
  return WalkResult::advance();
}

static WalkResult legalizeFuncReturnOp(func::ReturnOp returnOp, func::FuncOp func) {
  OpBuilder builder(returnOp);
  builder.setInsertionPoint(returnOp);
  for (auto [idx, operand] : llvm::enumerate(returnOp->getOpOperands())) {
    Type expectedType = func.getFunctionType().getResult(idx);
    Value cur = operand.get();
    if (cur.getType() == expectedType) {
      continue;
    }

    Value converted;
    if (cur.getType().isInteger(1) && expectedType.isInteger(8)) {
      converted = createI1ToI8ScalarCast(returnOp.getLoc(), cur, builder);
    } else if (cur.getType().isInteger(8) && expectedType.isInteger(1)) {
      converted = createI8ToI1ScalarCast(returnOp.getLoc(), cur, builder);
    } else if (auto expectedMemref = dyn_cast<MemRefType>(expectedType)) {
      converted = createI1ToI8MemRefCast(returnOp.getLoc(), cur, expectedMemref, builder);
    }

    if (!converted) {
      return returnOp.emitError("unsupported i1 to i8 legalization for return operand"), WalkResult::interrupt();
    }
    operand.set(converted);
  }
  return WalkResult::advance();
}

static LogicalResult legalizeOpsWithSingleWalk(func::FuncOp func) {
  WalkResult result = func.walk([&](Operation *op) -> WalkResult {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      return legalizeMemrefLoadOp(loadOp);
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      return legalizeMemrefStoreOp(storeOp);
    }

    if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op)) {
      return legalizeMemrefExpandShapeOp(expandOp);
    }

    if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(op)) {
      return legalizeMemrefCollapseShapeOp(collapseOp);
    }

    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      return legalizeArithSelectOp(selectOp);
    }

    if (LegalizeBoolIntBinaryOpMatcher::match(op)) {
      return legalizeArithIntBinaryOp(op);
    }

    if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
      return legalizeArithCmpIOp(cmpOp);
    }

    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      return legalizeFuncReturnOp(returnOp, func);
    }

    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

struct LegalizeBoolPass : public impl::LegalizeBoolBase<LegalizeBoolPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    updateFunctionType(func);

    if (func.empty()) {
      return;
    }
    if (failed(legalizeOpsWithSingleWalk(func))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace affine
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createLegalizeBoolPass() {
  return std::make_unique<affine::LegalizeBoolPass>();
}
