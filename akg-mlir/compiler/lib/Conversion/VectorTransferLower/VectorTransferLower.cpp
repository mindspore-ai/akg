/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Conversion/VectorTransferLower/VectorTransferLower.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"

namespace mlir {
#define GEN_PASS_DEF_VECTORTRANSFERLOWER
#define GEN_PASS_DECL_VECTORTRANSFERLOWER
#define GEN_PASS_CLASSES
#include "akg/Conversion/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::vector;

namespace {
class VectorTransferLowerPass : public impl::VectorTransferLowerBase<VectorTransferLowerPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, memref::MemRefDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

class AKGTransferReadToVectorLoadLowering : public OpRewritePattern<vector::TransferReadOp> {
 public:
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op, PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    if (isa<BoolAttr>(cast<arith::ConstantOp>(op.getPadding().getDefiningOp()).getValueAttr())) {
      return failure();
    }
    auto loadOp = rewriter.create<vector::LoadOp>(op.getLoc(), op.getVectorType(), op.getSource(), op.getIndices());
    rewriter.replaceOp(op, loadOp->getResult(0));
    rewriter.eraseOp(op.getPadding().getDefiningOp());
    return success();
  }
};

class AKGTransferWriteToVectorStoreLowering : public OpRewritePattern<vector::TransferWriteOp> {
 public:
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op, PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    if (cast<VectorType>(op.getVector().getType()).getElementType().isInteger(1)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<vector::StoreOp>(op, op.getVector(), op.getSource(), op.getIndices());
    return success();
  }
};

void mlir::populateVectorTransferLowerPatterns(RewritePatternSet &patterns) {
  (void)patterns.add<AKGTransferReadToVectorLoadLowering, AKGTransferWriteToVectorStoreLowering>(patterns.getContext());
}

void VectorTransferLowerPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  target.addLegalOp<vector::LoadOp>();
  target.addLegalOp<vector::StoreOp>();
  target.addLegalOp<vector::BroadcastOp>();
  target.addLegalOp<memref::LoadOp>();

  RewritePatternSet mlirPatterns(context);
  vector::populateVectorTransferLoweringPatterns(mlirPatterns);
  vector::populateVectorTransferPermutationMapLoweringPatterns(mlirPatterns);
  if (failed(applyPartialConversion(getOperation(), target, std::move(mlirPatterns)))) {
    signalPassFailure();
  }

  RewritePatternSet patterns(context);
  populateVectorTransferLowerPatterns(patterns);
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>> mlir::createVectorTransferLowerPass() {
  return std::make_unique<VectorTransferLowerPass>();
}
