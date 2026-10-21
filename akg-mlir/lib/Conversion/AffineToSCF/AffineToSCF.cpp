/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "akg/Conversion/AffineToSCF/AffineToSCF.h"
#include <algorithm>
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "akg/Utils/AnalysisCommon.hpp"

namespace mlir {
#define GEN_PASS_DEF_CONVERTAFFINETOSCF
#include "akg/Conversion/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace {

/// Compute bound value from affine map using affine.max (lower) or affine.min
/// (upper); replaces arith min/max with affine dialect ops.
static Value lowerBoundWithAffineMinMax(affine::AffineForOp op, bool isLowerBound, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  AffineMap map = isLowerBound ? op.getLowerBoundMap() : op.getUpperBoundMap();
  ValueRange operands = isLowerBound ? op.getLowerBoundOperands() : op.getUpperBoundOperands();
  if (map.getNumResults() == 0) {
    return nullptr;
  }
  if (isLowerBound) {
    return rewriter.create<affine::AffineMaxOp>(loc, map, operands).getResult();
  }
  return rewriter.create<affine::AffineMinOp>(loc, map, operands).getResult();
}

/// Convert affine.for to scf.for, preserving all attributes
class AffineForToSCFPattern : public OpRewritePattern<affine::AffineForOp> {
 public:
  explicit AffineForToSCFPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<affine::AffineForOp>(context, benefit) {}
  LogicalResult matchAndRewrite(affine::AffineForOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Compute lower and upper bounds using affine.max / affine.min
    // isLowerBound= true
    Value lowerBound = lowerBoundWithAffineMinMax(op, true, rewriter);
    // isLowerBound= false
    Value upperBound = lowerBoundWithAffineMinMax(op, false, rewriter);
    if (!lowerBound || !upperBound) {
      return failure();
    }
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStepAsInt());

    // Create scf.for operation
    auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, op.getInits());

    // Copy MapForToForall attributes from affine.for to scf.for
    if (op->getAttr(kMapForToForallAttr)) {
      scfForOp->setAttr(kMapForToForallAttr, rewriter.getUnitAttr());
    }

    // Copy reduction attribute from affine.for to scf.for
    if (op->getAttr(kReductionLoopAttr)) {
      scfForOp->setAttr(kReductionLoopAttr, rewriter.getUnitAttr());
    }

    // Copy broadcast attribute from affine.for to scf.for
    if (op->getAttr(kBroadcastLoopAttr)) {
      scfForOp->setAttr(kBroadcastLoopAttr, rewriter.getUnitAttr());
    }

    // Move the body from affine.for to scf.for
    rewriter.eraseBlock(scfForOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), scfForOp.getRegion(), scfForOp.getRegion().end());

    // Replace affine.for with scf.for results
    rewriter.replaceOp(op, scfForOp.getResults());

    return success();
  }
};

/// Convert affine.if to scf.if: use affine.apply to compute each constraint
/// expression, then cmp with 0 and AND to get cond.
class AffineIfToSCFPattern : public OpRewritePattern<affine::AffineIfOp> {
 public:
  explicit AffineIfToSCFPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<affine::AffineIfOp>(context, benefit) {}

  LogicalResult matchAndRewrite(affine::AffineIfOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto integerSet = op.getIntegerSet();
    auto skipAttr = rewriter.getUnitAttr();
    Value zeroConstant = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    ValueRange operands = op.getOperands();
    auto numDims = integerSet.getNumDims();

    // For each constraint: affine.apply(expr)(operands) -> index, then cmp with 0.
    Value cond = nullptr;
    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);
      // Build single-result map for this constraint: (dims)[syms] -> (expr)
      AffineMap map = AffineMap::get(numDims, integerSet.getNumSymbols(), constraintExpr, op.getContext());
      Value affResult = rewriter.create<affine::AffineApplyOp>(loc, map, operands).getResult();
      auto pred = isEquality ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
      auto cmpVal = rewriter.create<arith::CmpIOp>(loc, pred, affResult, zeroConstant);
      cmpVal->setAttr(kSkipVectorizeAttr, skipAttr);
      cond = cond ? [&rewriter, &loc, &cond, &cmpVal, &skipAttr]() {
        auto andOp = rewriter.create<arith::AndIOp>(loc, cond, cmpVal);
        andOp->setAttr(kSkipVectorizeAttr, skipAttr);
        return andOp.getResult();
      }()
                  : cmpVal;
    }
    // value= 1, width= 1
    cond = cond ? cond : rewriter.create<arith::ConstantIntOp>(loc, 1, 1);

    bool hasElseRegion = !op.getElseRegion().empty();
    auto scfIfOp = rewriter.create<scf::IfOp>(loc, op.getResultTypes(), cond, hasElseRegion);
    rewriter.inlineRegionBefore(op.getThenRegion(), &scfIfOp.getThenRegion().back());
    rewriter.eraseBlock(&scfIfOp.getThenRegion().back());
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(op.getElseRegion(), &scfIfOp.getElseRegion().back());
      rewriter.eraseBlock(&scfIfOp.getElseRegion().back());
    }

    rewriter.replaceOp(op, scfIfOp.getResults());
    return success();
  }
};

/// Lower affine memory accesses while keeping their index expressions in the
/// affine dialect. The generic affine-to-standard patterns materialize these
/// expressions with arith operations, which loses the structure consumed by
/// later vectorization passes.
static SmallVector<Value> materializeAffineIndices(AffineMap map, ValueRange operands, Location loc,
                                                   PatternRewriter &rewriter) {
  SmallVector<Value> canonicalOperands(operands.begin(), operands.end());
  affine::canonicalizeMapAndOperands(&map, &canonicalOperands);
  // After the enclosing affine loop is converted to scf, its induction
  // variables are dimensions rather than affine symbols. Flatten the map so
  // every operand is represented as a dimension in the generated affine.apply.
  SmallVector<AffineExpr> dimReplacements;
  dimReplacements.reserve(map.getNumDims());
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    dimReplacements.push_back(getAffineDimExpr(i, rewriter.getContext()));
  }
  SmallVector<AffineExpr> symbolReplacements;
  symbolReplacements.reserve(map.getNumSymbols());
  for (unsigned i = 0; i < map.getNumSymbols(); ++i) {
    symbolReplacements.push_back(
        getAffineDimExpr(map.getNumDims() + i, rewriter.getContext()));
  }
  map = map.replaceDimsAndSymbols(dimReplacements, symbolReplacements, map.getNumInputs(), 0);
  SmallVector<Value> indices;
  indices.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineMap resultMap = AffineMap::get(map.getNumDims(), 0, expr, rewriter.getContext());
    indices.push_back(rewriter.create<affine::AffineApplyOp>(loc, resultMap, canonicalOperands).getResult());
  }
  return indices;
}

class AffineLoadToMemRefPattern : public OpRewritePattern<affine::AffineLoadOp> {
 public:
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLoadOp op, PatternRewriter &rewriter) const override {
    SmallVector<Value> indices =
        materializeAffineIndices(op.getAffineMap(), op.getMapOperands(), op.getLoc(), rewriter);
    auto load = rewriter.create<memref::LoadOp>(op.getLoc(), op.getMemRef(), indices);
    if (Attribute nontemporal = op->getAttr("nontemporal")) {
      load->setAttr("nontemporal", nontemporal);
    }
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class AffineStoreToMemRefPattern : public OpRewritePattern<affine::AffineStoreOp> {
 public:
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineStoreOp op, PatternRewriter &rewriter) const override {
    SmallVector<Value> indices =
        materializeAffineIndices(op.getAffineMap(), op.getMapOperands(), op.getLoc(), rewriter);
    auto store = rewriter.create<memref::StoreOp>(op.getLoc(), op.getValueToStore(), op.getMemRef(), indices);
    if (Attribute nontemporal = op->getAttr("nontemporal")) {
      store->setAttr("nontemporal", nontemporal);
    }
    rewriter.replaceOp(op, store.getOperation());
    return success();
  }
};

class ConvertAffineToSCF : public impl::ConvertAffineToSCFBase<ConvertAffineToSCF> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Mark scf, arith, memref as legal; affine.apply/min/max remain (erased
    // from patterns) and are legal for this pass.
    target.addLegalDialect<scf::SCFDialect, arith::ArithDialect, memref::MemRefDialect>();
    target.addLegalOp<affine::AffineApplyOp, affine::AffineMinOp, affine::AffineMaxOp>();

    // Lower memory accesses while they are still nested in affine loops. Once
    // an affine loop is converted to scf, its induction variables are no
    // longer valid affine symbols, so delaying this rewrite makes affine
    // memory operations unverifiable.
    RewritePatternSet memoryPatterns(context);
    memoryPatterns.add<AffineLoadToMemRefPattern, AffineStoreToMemRefPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(memoryPatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    patterns.add<AffineForToSCFPattern, AffineIfToSCFPattern, AffineLoadToMemRefPattern, AffineStoreToMemRefPattern>(
        context);
    populateAffineToStdConversionPatterns(patterns);
    // Erase AffineApplyLowering, AffineMinLowering, AffineMaxLowering from
    // patterns (caller keeps affine.apply/min/max).
    OperationName applyName("affine.apply", context);
    OperationName minName("affine.min", context);
    OperationName maxName("affine.max", context);
    OperationName loadName("affine.load", context);
    OperationName storeName("affine.store", context);
    auto &nativePatterns = patterns.getNativePatterns();
    nativePatterns.erase(std::remove_if(nativePatterns.begin(), nativePatterns.end(),
                                        [&applyName, &minName, &maxName, &loadName, &storeName](
                                            const std::unique_ptr<RewritePattern> &p) {
                                          std::optional<OperationName> root = p->getRootKind();
                                          if (!root) {
                                            return false;
                                          }
                                          return *root == applyName || *root == minName || *root == maxName ||
                                                 *root == loadName || *root == storeName;
                                        }),
                         nativePatterns.end());
    populateAffineToVectorConversionPatterns(patterns);
    affine::populateAffineExpandIndexOpsPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createConvertAffineToSCFPass() { return std::make_unique<ConvertAffineToSCF>(); }

}  // namespace mlir
