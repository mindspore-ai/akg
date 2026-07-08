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

#include "mfusion/Dialect/Mfuse/Transforms/Cluster/DVMCluster.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

#define GEN_PASS_DEF_DECOMPOSEMATMULWITHBIASFORDVMCLUSTER
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace {

struct DecomposeMatmulWithBiasForDvmClusterPass
    : public impl::DecomposeMatmulWithBiasForDvmClusterBase<DecomposeMatmulWithBiasForDvmClusterPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<mfuse::MatmulWithBiasOp> ops;
    func.walk([&](mfuse::MatmulWithBiasOp op) { ops.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (mfuse::MatmulWithBiasOp op : ops) {
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      Type resultType = op.getResult().getType();
      Value matmul = rewriter.create<mfuse::MatmulOp>(loc, resultType, op.getSelf(), op.getOther(),
                                                      op.getTransX1Attr(), op.getTransX2Attr());
      Value add = rewriter.create<mfuse::AddOp>(loc, resultType, matmul, op.getBias());
      rewriter.replaceOp(op, add);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createDecomposeMatmulWithBiasForDvmClusterPass() {
  return std::make_unique<DecomposeMatmulWithBiasForDvmClusterPass>();
}

}  // namespace mlir

