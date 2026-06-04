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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

Type convertI1ElemToI8(Type type) {
  if (!getElementTypeOrSelf(type).isInteger(1)) return type;
  MLIRContext *ctx = type.getContext();
  Type i8 = IntegerType::get(ctx, 8);
  if (type.isInteger(1)) return i8;
  if (auto m = dyn_cast<MemRefType>(type)) return MemRefType::get(m.getShape(), i8, m.getLayout(), m.getMemorySpace());
  return type;
}

Value createI1ToI8ScalarCast(Location loc, Value v, OpBuilder &b) {
  if (!v.getType().isInteger(1)) return v;
  Value f = b.create<arith::UIToFPOp>(loc, b.getF16Type(), v);
  return b.create<arith::FPToUIOp>(loc, b.getI8Type(), f);
}

Value createI8ToI1ScalarCast(Location loc, Value v, OpBuilder &b) {
  if (!v.getType().isInteger(8)) return v;
  Value z = b.create<arith::ConstantOp>(loc, FloatAttr::get(b.getF16Type(), 0.0));
  Value f = b.create<arith::UIToFPOp>(loc, b.getF16Type(), v);
  return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, f, z);
}

Value castBitBinaryOperandToResultTy(Location loc, Type resTy, Value v, OpBuilder &b) {
  Type t = v.getType();
  if (t == resTy) return v;
  if (resTy.isInteger(8) && t.isInteger(1)) return b.create<arith::ExtUIOp>(loc, resTy, v).getResult();
  if (resTy.isInteger(1) && t.isInteger(8)) return createI8ToI1ScalarCast(loc, v, b);
  return v;
}

void updateFunctionType(func::FuncOp func) {
  OpBuilder b(func);
  FunctionType old = func.getFunctionType();
  SmallVector<Type, 4> in, out;
  std::transform(old.getInputs().begin(), old.getInputs().end(), std::back_inserter(in), convertI1ElemToI8);
  std::transform(old.getResults().begin(), old.getResults().end(), std::back_inserter(out), convertI1ElemToI8);
  FunctionType neu = b.getFunctionType(in, out);
  if (neu == old) return;
  func.setType(neu);
  if (func.empty()) return;
  Block &e = func.getBody().front();
  for (unsigned i = 0; i < func.getNumArguments(); ++i) e.getArgument(i).setType(in[i]);
}

struct MemRefViewSrc {
  static Value src(memref::ExpandShapeOp op) { return op.getSrc(); }
  static Value src(memref::CollapseShapeOp op) { return op.getSrc(); }
  static Value src(memref::ReshapeOp op) { return op.getSource(); }
  static Value src(memref::SubViewOp op) { return op.getSource(); }
};

template <typename OpTy>
struct MemRefReshapePattern final : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &) const override {
    Value src = MemRefViewSrc::src(op);
    Value res = op.getResult();
    auto st = dyn_cast<MemRefType>(src.getType()), dt = dyn_cast<MemRefType>(res.getType());
    if (!st || !dt || dt.getElementType() == st.getElementType()) return failure();
    res.setType(MemRefType::get(dt.getShape(), st.getElementType(), dt.getLayout(), dt.getMemorySpace()));
    return success();
  }
};

/// arith.select: when the condition is i8, turn it into i1 to match scalar i1/i8 semantics.
template <typename OpTy>
struct ArithSelectPattern final : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &b) const override {
    Value c = op.getCondition();
    if (!c.getType().isInteger(8)) return failure();
    b.setInsertionPoint(op);
    op->setOperand(0, createI8ToI1ScalarCast(op.getLoc(), c, b));
    return success();
  }
};

template <typename IntOp>
struct IntToFloatMap;
template <>
struct IntToFloatMap<arith::AddIOp> {
  using type = arith::AddFOp;
};
template <>
struct IntToFloatMap<arith::SubIOp> {
  using type = arith::SubFOp;
};
template <>
struct IntToFloatMap<arith::MulIOp> {
  using type = arith::MulFOp;
};
template <>
struct IntToFloatMap<arith::DivSIOp> {
  using type = arith::DivFOp;
};
template <>
struct IntToFloatMap<arith::DivUIOp> {
  using type = arith::DivFOp;
};
template <>
struct IntToFloatMap<arith::RemSIOp> {
  using type = arith::RemFOp;
};
template <>
struct IntToFloatMap<arith::RemUIOp> {
  using type = arith::RemFOp;
};
template <>
struct IntToFloatMap<arith::MaxSIOp> {
  using type = arith::MaximumFOp;
};
template <>
struct IntToFloatMap<arith::MaxUIOp> {
  using type = arith::MaximumFOp;
};
template <>
struct IntToFloatMap<arith::MinSIOp> {
  using type = arith::MinimumFOp;
};
template <>
struct IntToFloatMap<arith::MinUIOp> {
  using type = arith::MinimumFOp;
};

/// arith int ops that use the float (uitofp / fop / fptoui) path.
template <typename OpTy>
struct ArithIntBinaryPattern final : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &r) const override {
    Operation *o = op.getOperation();
    if (o->getNumOperands() != 2 || o->getNumResults() != 1) return failure();
    Value lhs = o->getOperand(0), rhs = o->getOperand(1);
    Type lt = lhs.getType(), rt = rhs.getType();
    if (!(lt.isInteger(1) || lt.isInteger(8) || rt.isInteger(1) || rt.isInteger(8))) return failure();
    r.setInsertionPoint(o);
    Location loc = o->getLoc();
    Type i8ty = r.getI8Type();
    Type f16 = r.getF16Type();
    auto toF16 = [&](Value v) { return r.create<arith::UIToFPOp>(loc, f16, v).getResult(); };
    using FloatOp = typename IntToFloatMap<OpTy>::type;
    Value w = r.create<FloatOp>(loc, toF16(lhs), toF16(rhs)).getResult();
    r.replaceOp(o, {r.create<arith::FPToUIOp>(loc, i8ty, w).getResult()});
    return success();
  }
};

/// arith int ops that stay integer (and / or / xor / shifts): operand alignment for i1/i8.
template <typename OpTy>
struct ArithBitBinaryPattern final : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &r) const override {
    Operation *o = op.getOperation();
    if (o->getNumOperands() != 2 || o->getNumResults() != 1) return failure();
    Value lhs = o->getOperand(0), rhs = o->getOperand(1);
    Type lt = lhs.getType(), rt = rhs.getType();
    Type resTy = o->getResult(0).getType();
    if (!(lt.isInteger(1) || lt.isInteger(8) || rt.isInteger(1) || rt.isInteger(8))) return failure();
    if (lt == rt && rt == resTy) return failure();
    r.setInsertionPoint(o);
    Location loc = o->getLoc();
    Value l8 = castBitBinaryOperandToResultTy(loc, resTy, lhs, r);
    Value r8 = castBitBinaryOperandToResultTy(loc, resTy, rhs, r);
    if (isa<OpTy>(o)) {
      r.replaceOpWithNewOp<OpTy>(cast<OpTy>(o), l8, r8);
      return success();
    }
    return failure();
  }
};

/// arith.cmpi: if lhs and rhs mix i1 and i8, cast the i1 side to i8 to align operand types.
template <typename OpTy>
struct ArithCmplPattern final : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &b) const override {
    b.setInsertionPoint(op);
    if (op->getNumOperands() != 2) return failure();
    Value lhs = op.getLhs(), rhs = op.getRhs();
    Type lt = lhs.getType(), rt = rhs.getType();
    if (lt == rt) return failure();
    Location loc = op.getLoc();
    if (lt.isInteger(8) && rt.isInteger(1)) {
      op->setOperand(1, createI1ToI8ScalarCast(loc, rhs, b));
      return success();
    }
    if (lt.isInteger(1) && rt.isInteger(8)) {
      op->setOperand(0, createI1ToI8ScalarCast(loc, lhs, b));
      return success();
    }
    return failure();
  }
};

struct MemRefLoadPattern final : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::LoadOp op, PatternRewriter &) const override {
    auto m = dyn_cast<MemRefType>(op.getMemRef().getType());
    if (!m || op.getType() == m.getElementType()) return failure();
    op.getResult().setType(m.getElementType());
    return success();
  }
};

struct MemRefStorePattern final : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::StoreOp op, PatternRewriter &b) const override {
    auto m = dyn_cast<MemRefType>(op.getMemRef().getType());
    if (!m) return failure();
    Type want = m.getElementType(), vt = op.getValueToStore().getType();
    if (vt == want) return failure();
    b.setInsertionPoint(op);
    if (vt.isInteger(1) && want.isInteger(8)) {
      op->setOperand(0, createI1ToI8ScalarCast(op.getLoc(), op.getValueToStore(), b));
      return success();
    }
    if (vt.isInteger(8) && want.isInteger(1)) {
      op->setOperand(0, createI8ToI1ScalarCast(op.getLoc(), op.getValueToStore(), b));
      return success();
    }
    return failure();
  }
};

struct ReturnPattern final : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op, PatternRewriter &b) const override {
    func::FuncOp callee = op->getParentOfType<func::FuncOp>();
    if (!callee) return failure();
    b.setInsertionPoint(op);
    bool any = false;
    for (auto [i, use] : llvm::enumerate(op->getOpOperands())) {
      Type want = callee.getFunctionType().getResult(i);
      Value cur = use.get();
      if (cur.getType() == want) continue;
      Value v;
      if (cur.getType().isInteger(1) && want.isInteger(8)) {
        v = createI1ToI8ScalarCast(op.getLoc(), cur, b);
      } else if (cur.getType().isInteger(8) && want.isInteger(1)) {
        v = createI8ToI1ScalarCast(op.getLoc(), cur, b);
      } else {
        op.emitOpError("unsupported i1 to i8 legalization for return operand");
        return failure();
      }
      use.set(v);
      any = true;
    }
    return any ? success() : failure();
  }
};

struct LegalizeBoolPass : public impl::LegalizeBoolBase<LegalizeBoolPass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    updateFunctionType(fn);
    if (fn.empty()) return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet p(ctx);
    p.add<MemRefLoadPattern, MemRefStorePattern, ReturnPattern, MemRefReshapePattern<memref::ExpandShapeOp>,
          MemRefReshapePattern<memref::CollapseShapeOp>, MemRefReshapePattern<memref::ReshapeOp>,
          MemRefReshapePattern<memref::SubViewOp>, ArithSelectPattern<arith::SelectOp>, ArithCmplPattern<arith::CmpIOp>,
          ArithIntBinaryPattern<arith::AddIOp>, ArithIntBinaryPattern<arith::SubIOp>,
          ArithIntBinaryPattern<arith::MulIOp>, ArithIntBinaryPattern<arith::DivSIOp>,
          ArithIntBinaryPattern<arith::DivUIOp>, ArithIntBinaryPattern<arith::RemSIOp>,
          ArithIntBinaryPattern<arith::RemUIOp>, ArithIntBinaryPattern<arith::MaxSIOp>,
          ArithIntBinaryPattern<arith::MaxUIOp>, ArithIntBinaryPattern<arith::MinSIOp>,
          ArithIntBinaryPattern<arith::MinUIOp>, ArithBitBinaryPattern<arith::ShLIOp>,
          ArithBitBinaryPattern<arith::ShRSIOp>, ArithBitBinaryPattern<arith::ShRUIOp>,
          ArithBitBinaryPattern<arith::AndIOp>, ArithBitBinaryPattern<arith::OrIOp>,
          ArithBitBinaryPattern<arith::XOrIOp>>(ctx);

    GreedyRewriteConfig cfg;
    cfg.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(fn, std::move(p), cfg))) signalPassFailure();
  }
};

}  // namespace
}  // namespace affine
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createLegalizeBoolPass() {
  return std::make_unique<affine::LegalizeBoolPass>();
}
