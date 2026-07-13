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

#include "akg/Dialect/Linalg/Transforms/LegalizeIntWidth.h"

#include <optional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_LEGALIZEINTWIDTH
#define GEN_PASS_DEF_LEGALIZEINTWIDTH
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "legalize-int-width"

namespace mlir {
namespace {

bool isEqualityPredicate(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::eq || p == arith::CmpIPredicate::ne;
}

bool isSignedPredicate(arith::CmpIPredicate p) {
  switch (p) {
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::sge:
      return true;
    default:
      return false;
  }
}

/// If `v` is an integer extension, return its source and set `isSigned`.
Value getExtensionSource(Value v, bool &isSigned) {
  if (auto e = v.getDefiningOp<arith::ExtSIOp>()) {
    isSigned = true;
    return e.getIn();
  }
  if (auto e = v.getDefiningOp<arith::ExtUIOp>()) {
    isSigned = false;
    return e.getIn();
  }
  return nullptr;
}

/// Re-extends `v` to `wideTy` using the extension kind selected by `isSigned`.
Value extendTo(PatternRewriter &rewriter, Location loc, Value v, Type wideTy, bool isSigned) {
  if (isSigned) {
    return rewriter.create<arith::ExtSIOp>(loc, wideTy, v);
  }
  return rewriter.create<arith::ExtUIOp>(loc, wideTy, v);
}

/// Narrow operands with a common narrow type and extension kind for value-exact rewrites.
struct NarrowedOperands {
  IntegerType narrowTy;
  bool isSigned = false;
  SmallVector<Value> values;
};

/// Narrows `operands` to a common narrow integer type. Succeeds when:
///   - at least one operand is extsi/extui (anchors narrow type and signedness);
///   - all extensions share the same narrow type and signedness;
///   - other operands are constants that round-trip through the narrow type.
FailureOr<NarrowedOperands> narrowOperands(ArrayRef<Value> operands, PatternRewriter &rewriter, Location loc) {
  IntegerType narrowTy;
  std::optional<bool> isSigned;

  // First pass: derive common narrow type and signedness from extensions.
  for (Value v : operands) {
    bool s = false;
    Value src = getExtensionSource(v, s);
    if (!src) {
      continue;
    }
    auto srcTy = dyn_cast<IntegerType>(src.getType());
    if (!srcTy) {
      return failure();
    }
    if (!narrowTy) {
      narrowTy = srcTy;
      isSigned = s;
    } else if (narrowTy != srcTy || *isSigned != s) {
      return failure();
    }
  }
  if (!narrowTy) {
    return failure();
  }

  // Second pass: build narrow operand list.
  for (Value v : operands) {
    bool s = false;
    if (getExtensionSource(v, s)) continue;
    auto wideTy = dyn_cast<IntegerType>(v.getType());
    IntegerAttr cstAttr;
    if (!wideTy || wideTy.getWidth() <= narrowTy.getWidth() ||
        !matchPattern(v, m_Constant(&cstAttr))) {
      return failure();
    }
    APInt wide = cstAttr.getValue();
    APInt narrow = wide.trunc(narrowTy.getWidth());
    APInt roundTrip = *isSigned ? narrow.sext(wideTy.getWidth())
                                : narrow.zext(wideTy.getWidth());
    if (roundTrip != wide) return failure();
  }

  // Third pass: all checks passed.
  NarrowedOperands result;
  result.narrowTy = narrowTy;
  result.isSigned = *isSigned;
  result.values.reserve(operands.size());
  for (Value v : operands) {
    bool s = false;
    if (Value src = getExtensionSource(v, s)) {
      result.values.push_back(src);
      continue;
    }
    IntegerAttr cstAttr;
    bool matched = matchPattern(v, m_Constant(&cstAttr));
    (void)matched;
    assert(matched && "validated in pass 2");
    APInt narrow = cstAttr.getValue().trunc(narrowTy.getWidth());
    result.values.push_back(rewriter.create<arith::ConstantOp>(
        loc, narrowTy, rewriter.getIntegerAttr(narrowTy, narrow)));
  }
  return result;
}

// Rewrites `cmpi pred, ext(x:iN):iM, y:iM` -> `cmpi pred, x, y':iN`.
// Result is i1, so no re-extension needed.
struct NarrowCmpPattern : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp, PatternRewriter &rewriter) const override {
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    // Peek signedness from any extension operand WITHOUT touching IR,
    // and reject mismatched predicates early so narrowOperands is never
    // invoked for a guaranteed-to-fail rewrite.
    if (!isEqualityPredicate(pred)) {
      bool peekedSigned = false;
      bool found = false;
      for (Value v : {cmpOp.getLhs(), cmpOp.getRhs()}) {
        bool s = false;
        if (getExtensionSource(v, s)) {
          peekedSigned = s;
          found = true;
          break;
        }
      }
      if (!found) {
        return failure();
      }
      if (isSignedPredicate(pred) != peekedSigned) {
        return failure();
      }
    }

    FailureOr<NarrowedOperands> narrowed = narrowOperands({cmpOp.getLhs(), cmpOp.getRhs()}, rewriter, cmpOp.getLoc());
    if (failed(narrowed)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmpOp, pred, narrowed->values[0], narrowed->values[1]);
    return success();
  }
};

// Rewrites `select c, ext(x):iM, ext(y):iM` -> `ext (select c, x, y:iN):iM`.
struct NarrowSelectPattern : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selOp, PatternRewriter &rewriter) const override {
    if (!isa<IntegerType>(selOp.getType())) {
      return failure();
    }
    FailureOr<NarrowedOperands> narrowed =
      narrowOperands({selOp.getTrueValue(), selOp.getFalseValue()}, rewriter, selOp.getLoc());
    if (failed(narrowed)) {
      return failure();
    }
    Value narrowSel =
      rewriter.create<arith::SelectOp>(selOp.getLoc(), selOp.getCondition(), narrowed->values[0], narrowed->values[1]);
    rewriter.replaceOp(selOp, extendTo(rewriter, selOp.getLoc(), narrowSel, selOp.getType(), narrowed->isSigned));
    return success();
  }
};

// Rewrites bitwise `and/or/xor ext(x):iM, ext(y):iM` -> `ext (op x, y:iN):iM`.
template <typename OpTy>
struct NarrowBitwisePattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    FailureOr<NarrowedOperands> narrowed = narrowOperands({op.getLhs(), op.getRhs()}, rewriter, op.getLoc());
    if (failed(narrowed)) {
      return failure();
    }
    Value narrowOp = rewriter.create<OpTy>(op.getLoc(), narrowed->values[0], narrowed->values[1]);
    rewriter.replaceOp(op, extendTo(rewriter, op.getLoc(), narrowOp, op.getType(), narrowed->isSigned));
    return success();
  }
};

/// Returns true if `operand` feeds `user` in a narrowable position (integer
/// operand of cmpi/select/bitwise, not the i1 condition of select).
bool isNarrowableConsumer(Operation *user, Value operand) {
  if (isa<arith::CmpIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(user)) {
    return true;
  }
  if (auto sel = dyn_cast<arith::SelectOp>(user)) {
    return operand != sel.getCondition();
  }
  return false;
}

/// Returns true if `g` is a 1-in/1-out elementwise generic whose body is a
/// pure integer extension (e.g. `extui i1 to i32`).
bool isExtensionOnlyGeneric(linalg::GenericOp g) {
  if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1) {
    return false;
  }
  Block &body = g.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1) {
    return false;
  }
  Operation *def = yieldOp.getOperand(0).getDefiningOp();
  if (!def || !isa<arith::ExtUIOp, arith::ExtSIOp>(def)) {
    return false;
  }
  // Extension must consume the input block argument directly.
  auto src = dyn_cast<BlockArgument>(def->getOperand(0));
  return src && src.getOwner() == &body && src.getArgNumber() == 0;
}

// Fuses an extension-only generic into its consumer so narrowing patterns can fire.
struct FuseExtensionIntoConsumerGenericPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer, PatternRewriter &rewriter) const override {
    Block &body = consumer.getRegion().front();
    for (OpOperand *opOperand : consumer.getDpsInputOperands()) {
      auto producer = opOperand->get().getDefiningOp<linalg::GenericOp>();
      if (!producer || !isExtensionOnlyGeneric(producer)) {
        continue;
      }

      // Only fuse when the extended value feeds a narrowable op.
      BlockArgument bbArg = body.getArgument(opOperand->getOperandNumber());
      if (llvm::none_of(bbArg.getUsers(), [&bbArg](Operation *u) { return isNarrowableConsumer(u, bbArg); })) {
        continue;
      }

      if (!linalg::areElementwiseOpsFusable(opOperand)) {
        continue;
      }

      FailureOr<linalg::ElementwiseOpFusionResult> fused = linalg::fuseElementwiseOps(rewriter, opOperand);
      if (failed(fused)) {
        continue;
      }
      rewriter.replaceOp(consumer, fused->fusedOp->getResults().take_back(consumer.getNumResults()));
      return success();
    }
    return failure();
  }
};

// Folds an int-to-float cast generic with a splat constant input into a float
// constant. E.g.:
//   %c = arith.constant dense<0> : tensor<i64>
//   %r = linalg.generic ... sitofp ... -> tensor<f32>
// becomes %r = arith.constant dense<0.0> : tensor<f32>
struct FoldConstantIntToFpGenericPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp g, PatternRewriter &rewriter) const override {
    if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1) {
      return failure();
    }
    Block &body = g.getRegion().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
      return failure();
    }
    Operation *def = yieldOp.getOperand(0).getDefiningOp();
    if (!def) {
      return failure();
    }
    bool isSigned = false;
    if (isa<arith::SIToFPOp>(def)) {
      isSigned = true;
    } else if (isa<arith::UIToFPOp>(def)) {
      isSigned = false;
    } else {
      return failure();
    }
    // Cast must consume the input block argument directly.
    auto src = dyn_cast<BlockArgument>(def->getOperand(0));
    if (!src || src.getOwner() != &body || src.getArgNumber() != 0) {
      return failure();
    }

    // Input must be a splat integer constant.
    DenseElementsAttr inAttr;
    if (!matchPattern(g.getDpsInputOperand(0)->get(), m_Constant(&inAttr)) || !inAttr.isSplat() ||
        !isa<IntegerType>(inAttr.getElementType())) {
      return failure();
    }
    auto resTy = dyn_cast<RankedTensorType>(g.getResult(0).getType());
    if (!resTy) {
      return failure();
    }
    auto fpTy = dyn_cast<FloatType>(resTy.getElementType());
    if (!fpTy) {
      return failure();
    }

    APFloat folded(fpTy.getFloatSemantics());
    folded.convertFromAPInt(inAttr.getSplatValue<APInt>(), isSigned, APFloat::rmNearestTiesToEven);
    auto splat = DenseElementsAttr::get(resTy, rewriter.getFloatAttr(fpTy, folded));
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(g, resTy, splat);
    return success();
  }
};

struct LegalizeIntWidth : public impl::LegalizeIntWidthBase<LegalizeIntWidth> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<NarrowCmpPattern, NarrowSelectPattern, NarrowBitwisePattern<arith::AndIOp>,
                 NarrowBitwisePattern<arith::OrIOp>, NarrowBitwisePattern<arith::XOrIOp>,
                 FuseExtensionIntoConsumerGenericPattern, FoldConstantIntToFpGenericPattern>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeIntWidthPass() {
  return std::make_unique<LegalizeIntWidth>();
}
}  // namespace mlir
