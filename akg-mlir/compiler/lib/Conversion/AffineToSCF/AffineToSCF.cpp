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
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "akg/Utils/AnalysisCommon.hpp"

namespace mlir {
#define GEN_PASS_DEF_CONVERTAFFINETOSCF
#include "akg/Conversion/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace {
/// Convert affine.load to memref.load
class AffineLoadToMemRefPattern : public OpRewritePattern<affine::AffineLoadOp> {
 public:
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLoadOp op, PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands = affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands) return failure();

    // Build memref.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(), *resultOperands);
    return success();
  }
};

/// Convert affine.store to memref.store
class AffineStoreToMemRefPattern : public OpRewritePattern<affine::AffineStoreOp> {
 public:
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineStoreOp op, PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands = affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands) return failure();

    // Build memref.store value, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, op.getValue(), op.getMemRef(), *resultOperands);
    return success();
  }
};

/// Convert affine.yield to scf.yield
class AffineYieldToSCFPattern : public OpRewritePattern<affine::AffineYieldOp> {
 public:
  using OpRewritePattern<affine::AffineYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineYieldOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
    return success();
  }
};

/// Convert affine.for to scf.for, preserving all attributes
class AffineForToSCFPattern : public OpRewritePattern<affine::AffineForOp> {
 public:
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Compute lower and upper bounds
    Value lowerBound = lowerAffineLowerBound(op, rewriter);
    Value upperBound = lowerAffineUpperBound(op, rewriter);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStepAsInt());

    // Create scf.for operation
    auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, op.getInits());

    // Copy MapForToForall attributes from affine.for to scf.for
    if (op->getAttr(kMapForToForallAttr)) {
      scfForOp->setAttr(kMapForToForallAttr, rewriter.getUnitAttr());
    }

    // Move the body from affine.for to scf.for
    rewriter.eraseBlock(scfForOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), scfForOp.getRegion(), scfForOp.getRegion().end());

    // Replace affine.for with scf.for results
    rewriter.replaceOp(op, scfForOp.getResults());

    return success();
  }
};

class ConvertAffineToSCF : public impl::ConvertAffineToSCFBase<ConvertAffineToSCF> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Mark scf and memref dialects as legal
    target.addLegalDialect<scf::SCFDialect, arith::ArithDialect, memref::MemRefDialect>();

    // Mark affine.for, affine.yield, affine.load, and affine.store as illegal (needs conversion)
    target.addIllegalOp<affine::AffineForOp, affine::AffineYieldOp, affine::AffineLoadOp, affine::AffineStoreOp>();

    // Mark other affine ops as legal (they will be handled by other passes)
    target.addLegalDialect<affine::AffineDialect>();

    RewritePatternSet patterns(context);
    patterns.add<AffineForToSCFPattern, AffineYieldToSCFPattern, AffineLoadToMemRefPattern, AffineStoreToMemRefPattern>(
      patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createConvertAffineToSCFPass() { return std::make_unique<ConvertAffineToSCF>(); }

}  // namespace mlir
