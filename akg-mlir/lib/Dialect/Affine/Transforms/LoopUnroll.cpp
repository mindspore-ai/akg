/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Transforms/LoopUnroll.h"

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
#include "akg/Utils/SmallVectorSize.h"

namespace mlir {
#define GEN_PASS_DEF_AKGAFFINELOOPUNROLL
#define GEN_PASS_DECL_AKGAFFINELOOPUNROLL
#include "akg/Dialect/Affine/Passes.h.inc"

}  // namespace mlir

#define DEBUG_TYPE "akg-affine-loop-unroll"

namespace {
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::dyn_cast;
using mlir::isa;
using mlir::kSmallVectorSizeFour;
using mlir::kSmallVectorSizeOne;
using mlir::kSmallVectorSizeTwo;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::OperationPass;
using mlir::SmallVector;
using mlir::SmallVectorImpl;
using mlir::succeeded;
using mlir::WalkResult;
using mlir::func::FuncOp;

/// Loop unrolling pass. Unrolls all innermost loops unless full unrolling and a
/// full unroll threshold was specified, in which case, fully unrolls all loops
/// with trip count less than the specified threshold. The latter is for testing
/// purposes, especially for testing outer loop unrolling.
struct AKGLoopUnroll : public mlir::impl::AKGAffineLoopUnrollBase<AKGLoopUnroll> {
  AKGLoopUnroll() {}

  void runOnOperation() override;
};
}  // namespace

/// Returns true if no other affine.for ops are nested within `op`.
static bool isInnermostAffineForOp(mlir::affine::AffineForOp op) {
  return !op.getBody()
            ->walk([](mlir::affine::AffineForOp nestedForOp) {
              // Add original result expressions from lower/upper bound map.
              AffineMap lbMap = nestedForOp.getLowerBound().getMap();
              AffineMap ubMap = nestedForOp.getUpperBound().getMap();
              SmallVector<AffineExpr, kSmallVectorSizeOne> origLbExprs(lbMap.getResults().begin(),
                                                                       lbMap.getResults().end());
              SmallVector<AffineExpr, kSmallVectorSizeTwo> origUbExprs(ubMap.getResults().begin(),
                                                                       ubMap.getResults().end());

              // Insert all combinations of upper/lower bound results.
              int64_t origLoopStep = nestedForOp.getStepAsInt();
              for (auto origUbExpr : origUbExprs) {
                AffineExpr newUb = (origUbExpr - origLbExprs[0]).ceilDiv(origLoopStep);
                if (newUb != 1 || newUb.floorDiv(65) == 0) {
                  return WalkResult::interrupt();
                }
              }

              return WalkResult::advance();
            })
            .wasInterrupted();
}

/// Returns false if no mlir::vector::TransferReadOp or mlir::vector::TransferWriteOp ops
/// are nested within `op`.
static bool isVectorizedAffineForOp(mlir::affine::AffineForOp op) {
  if (!isInnermostAffineForOp(op)) {
    return false;
  }

  return op.getBody()
    ->walk([](Operation *op) {
      if (isa<mlir::vector::TransferReadOp, mlir::vector::TransferWriteOp, mlir::vector::StoreOp, mlir::vector::LoadOp>(
            op)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    })
    .wasInterrupted();
}

/// Gathers loops that have no affine.for's nested within.
static void gatherInnermostLoops(mlir::func::FuncOp f, SmallVectorImpl<mlir::affine::AffineForOp> &loops) {
  f.walk([&loops](mlir::affine::AffineForOp forOp) {
    if (isVectorizedAffineForOp(forOp)) {
      loops.push_back(forOp);
    }
  });
}

void AKGLoopUnroll::runOnOperation() {
  mlir::func::FuncOp funcOp = getOperation();
  if (funcOp.isExternal()) {
    return;
  }

  SmallVector<mlir::affine::AffineForOp, kSmallVectorSizeFour> loops;
  gatherInnermostLoops(funcOp, loops);
  if (loops.empty()) {
    return;
  }
  bool unrolled = false;
  for (auto forOp : loops) {
    LogicalResult isUnrolled = mlir::affine::loopUnrollFull(forOp);
    unrolled |= succeeded(isUnrolled);
  }
  if (!unrolled) {
    llvm::outs() << "No Affinefor loop is unrolled\n";
  }
}

namespace mlir {
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAKGLoopUnrollPass() {
  return std::make_unique<AKGLoopUnroll>();
}
}  // namespace mlir
