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

#include "akg/Conversion/FusionToMemVec/FusionToMemVec.h"

#include "akg/Dialect/Fusion/IR/Fusion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace {
class FusionLoadLowering : public OpRewritePattern<fusion::LoadOp> {
 public:
  using OpRewritePattern<fusion::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(fusion::LoadOp op, PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    if (!isa<VectorType>(resultType)) {
      auto memLoad = rewriter.create<memref::LoadOp>(op.getLoc(), op.getMemRef(), op.getIndices());
      rewriter.replaceOp(op, memLoad->getResults());
      return success();
    }

    auto resultVecType = cast<VectorType>(resultType);
    auto padding = op.getPadding();
    if (padding.getImpl() == nullptr) {
      auto elemType = resultVecType.getElementType();
      padding = rewriter.create<arith::ConstantOp>(op->getLoc(), elemType, rewriter.getZeroAttr(elemType));
    }

    auto vectorTransferRead = rewriter.create<vector::TransferReadOp>(
      op->getLoc(), resultVecType, op.getMemRef(), op.getIndices(),
      rewriter.getMultiDimIdentityMap(resultVecType.getRank()), padding, Value(), op.getInBoundsAttr());
    rewriter.replaceOp(op, vectorTransferRead->getResults());
    return success();
  }
};

class FusionMultiLoadLowering : public OpRewritePattern<fusion::MultiLoadOp> {
 public:
  using OpRewritePattern<fusion::MultiLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(fusion::MultiLoadOp op, PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class FusionInsertLowering : public OpRewritePattern<fusion::InsertOp> {
 public:
  using OpRewritePattern<fusion::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(fusion::InsertOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getData());
    return success();
  }
};

class FusionSubViewOpLowering : public OpRewritePattern<fusion::SubViewOp> {
 public:
  using OpRewritePattern<fusion::SubViewOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(fusion::SubViewOp op, PatternRewriter &rewriter) const override {
    auto memSubView = rewriter.create<memref::SubViewOp>(op.getLoc(), op.getSource(), ValueRange{op.getOffsets()},
                                                         ValueRange{op.getSizes()}, ValueRange{op.getStrides()});
    rewriter.replaceOp(op, memSubView->getResults());
    return success();
  }
};

class FusionBroadcastLowering : public OpRewritePattern<fusion::BroadcastOp> {
 public:
  using OpRewritePattern<fusion::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(fusion::BroadcastOp op, PatternRewriter &rewriter) const override {
    auto vecBroadcast = rewriter.create<vector::BroadcastOp>(op.getLoc(), op.getVectorType(), op.getSource());
    rewriter.replaceOp(op, vecBroadcast->getResults());
    return success();
  }
};

class FusionTransposeLowering : public OpRewritePattern<fusion::TransposeOp> {
 public:
  using OpRewritePattern<fusion::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(fusion::TransposeOp op, PatternRewriter &rewriter) const override {
    return success();
  }
};

class FusionStoreOpLowering : public OpRewritePattern<fusion::StoreOp> {
 public:
  using OpRewritePattern<fusion::StoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(fusion::StoreOp op, PatternRewriter &rewriter) const override {
    if (!isa<VectorType>(op.getValueToStore().getType())) {
      auto memStore =
        rewriter.create<memref::StoreOp>(op.getLoc(), op.getValueToStore(), op.getMemRef(), op.getIndices());
      assert(memStore->getResults().size() == 0 && "memRef.store should no need results");

      rewriter.eraseOp(op);
      return success();
    }

    auto vectorType = dyn_cast<VectorType>(op.getValueToStore().getType());
    (void)rewriter.create<vector::TransferWriteOp>(
      op->getLoc(), op.getValueToStore(), op.getMemRef(), op.getIndices(),
      AffineMapAttr::get(rewriter.getMultiDimIdentityMap(vectorType.getRank())), op.getInBoundsAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void mlir::populateFusionPatterns(RewritePatternSet &patterns) {
  (void)patterns.add<FusionLoadLowering, FusionMultiLoadLowering, FusionInsertLowering, FusionSubViewOpLowering,
                     FusionBroadcastLowering, FusionTransposeLowering, FusionStoreOpLowering>(patterns.getContext());
}
