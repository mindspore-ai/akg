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

#include "akg/Dialect/Affine/Transforms/AKGLoopUnroll.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#define GEN_PASS_DEF_AKGAFFINELOOPUNROLL
#define GEN_PASS_DECL_AKGAFFINELOOPUNROLL
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "akg-affine-loop-unroll"

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Loop unrolling pass. Unrolls all innermost loops unless full unrolling and a
/// full unroll threshold was specified, in which case, fully unrolls all loops
/// with trip count less than the specified threshold. The latter is for testing
/// purposes, especially for testing outer loop unrolling.
struct AKGLoopUnroll : public impl::AKGAffineLoopUnrollBase<AKGLoopUnroll> {
  AKGLoopUnroll() {}

  void runOnOperation() override;
};
}  // namespace

/// Returns true if no other affine.for ops are nested within `op`.
static bool isInnermostAffineForOp(affine::AffineForOp op) {
  return !op.getBody()
            ->walk([&](affine::AffineForOp nestedForOp) {
              // Add original result expressions from lower/upper bound map.
              AffineMap lbMap = nestedForOp.getLowerBound().getMap();
              AffineMap ubMap = nestedForOp.getUpperBound().getMap();
              SmallVector<AffineExpr, 1> origLbExprs(lbMap.getResults().begin(), lbMap.getResults().end());
              SmallVector<AffineExpr, 2> origUbExprs(ubMap.getResults().begin(), ubMap.getResults().end());

              // Insert all combinations of upper/lower bound results.
              int64_t origLoopStep = nestedForOp.getStepAsInt();
              for (unsigned i = 0; i < origUbExprs.size(); ++i) {
                AffineExpr newUb = (origUbExprs[i] - origLbExprs[0]).ceilDiv(origLoopStep);
                if (newUb != 1 || newUb.floorDiv(65) == 0) {
                  return WalkResult::interrupt();
                }
              }

              return WalkResult::advance();
            })
            .wasInterrupted();
}

/// Returns false if no vector::TransferReadOp or vector::TransferWriteOp ops
/// are nested within `op`.
static bool isVectorizedAffineForOp(affine::AffineForOp op) {
  if (!isInnermostAffineForOp(op)) {
    return false;
  }

  return op.getBody()
    ->walk([&](Operation *op) {
      if (isa<vector::TransferReadOp, vector::TransferWriteOp, vector::StoreOp, vector::LoadOp>(op)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    })
    .wasInterrupted();
}

/// Gathers loops that have no affine.for's nested within.
static void gatherInnermostLoops(func::FuncOp f, SmallVectorImpl<affine::AffineForOp> &loops) {
  f.walk([&](affine::AffineForOp forOp) {
    if (isVectorizedAffineForOp(forOp)) {
      loops.push_back(forOp);
    }
  });
}

void AKGLoopUnroll::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.isExternal()) {
    return;
  }

  SmallVector<affine::AffineForOp, 4> loops;
  gatherInnermostLoops(funcOp, loops);
  if (loops.empty()) {
    return;
  }
  bool unrolled = false;
  for (auto forOp : loops) {
    LogicalResult isUnrolled = loopUnrollFull(forOp);
    unrolled |= succeeded(isUnrolled);
  }
  if (!unrolled) {
    llvm::outs() << "No Affinefor loop is unrolled\n";
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopUnrollPass() {
  return std::make_unique<AKGLoopUnroll>();
}
