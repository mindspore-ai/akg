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

#include "akg/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#ifndef GEN_PASS_DECL_SYMBOLREMOVAL
#define GEN_PASS_DECL_SYMBOLREMOVAL
#ifndef GEN_PASS_DEF_SYMBOLREMOVAL
#define GEN_PASS_DEF_SYMBOLREMOVAL
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "symbol-removal"

using namespace mlir;
using namespace MemoryEffects;

namespace {

// ===----------------------------------------------------------------------===//
// SymbolRemovalPass
// ===----------------------------------------------------------------------===//

struct SymbolRemovalPass : public SymbolRemovalBase<SymbolRemovalPass> {
 public:
  void runOnOperation() override;
};

// Dispatch affine expression construction based on kind.
static AffineExpr getBinaryOpExpr(AffineExprKind kind, AffineExpr lhs, AffineExpr rhs) {
  if (kind == AffineExprKind::Add) {
    return lhs + rhs;
  }
  if (kind == AffineExprKind::Mul) {
    return lhs * rhs;
  }
  if (kind == AffineExprKind::FloorDiv) {
    return lhs.floorDiv(rhs);
  }
  if (kind == AffineExprKind::CeilDiv) {
    return lhs.ceilDiv(rhs);
  }
  if (kind == AffineExprKind::Mod) {
    return lhs % rhs;
  }

  llvm_unreachable("unknown binary operation on affine expressions");
}

AffineExpr replaceSymbolExpr(AffineExpr expr, OpBuilder b, unsigned numDims) {
  switch (expr.getKind()) {
    case AffineExprKind::Constant:
      return expr;
    case AffineExprKind::DimId:
      return expr;
    case AffineExprKind::SymbolId: {
      unsigned pos = expr.cast<AffineSymbolExpr>().getPosition();
      unsigned idx = pos + numDims;
      auto newExpr = b.getAffineDimExpr(idx);
      return newExpr;
    }
    case AffineExprKind::Add:
    case AffineExprKind::Mul:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
    case AffineExprKind::Mod:
      auto binOp = expr.cast<AffineBinaryOpExpr>();
      auto lhs = binOp.getLHS(), rhs = binOp.getRHS();
      auto newLHS = replaceSymbolExpr(lhs, b, numDims);
      auto newRHS = replaceSymbolExpr(rhs, b, numDims);
      if (newLHS == lhs && newRHS == rhs) {
        return expr;
      }
      return getBinaryOpExpr(expr.getKind(), newLHS, newRHS);
  }
  llvm_unreachable("Unknown AffineExpr");
}

void SymbolRemovalPass::runOnOperation() {
  OpBuilder b(getOperation());
  SmallVector<AffineLoadOp, 8> loadOps;
  SmallVector<AffineStoreOp, 8> storeOps;
  getOperation()->walk([&](Operation *op) {
    if (AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(op)) {
      loadOps.push_back(loadOp);
    }
    if (AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(op)) {
      storeOps.push_back(storeOp);
    }
  });

  for (auto loadOp : loadOps) {
    AffineMap map = loadOp.getAffineMapAttr().getValue();
    SmallVector<AffineExpr> exprs;
    for (auto expr : map.getResults()) {
      AffineExpr newExpr = replaceSymbolExpr(expr, b, map.getNumDims());
      exprs.push_back(newExpr);
    }
    auto newMap = AffineMap::get(/*dimCount=*/loadOp.getIndices().size(),
                                 /*symbolCount=*/0, exprs, map.getContext());
    b.setInsertionPoint(loadOp);
    auto newLoadOp = b.create<AffineLoadOp>(loadOp.getLoc(), loadOp.getMemRef(), newMap, loadOp.getIndices());
    loadOp.getOperation()->replaceAllUsesWith(newLoadOp.getOperation());
    loadOp.erase();
  }

  for (auto storeOp : storeOps) {
    AffineMap map = storeOp.getAffineMapAttr().getValue();
    SmallVector<AffineExpr> exprs;
    for (auto expr : map.getResults()) {
      AffineExpr newExpr = replaceSymbolExpr(expr, b, map.getNumDims());
      exprs.push_back(newExpr);
    }
    auto newMap = AffineMap::get(/*dimCount=*/storeOp.getIndices().size(),
                                 /*symbolCount=*/0, exprs, map.getContext());
    b.setInsertionPoint(storeOp);
    b.create<AffineStoreOp>(storeOp.getLoc(), storeOp.getValue(), storeOp.getMemRef(), newMap, storeOp.getIndices());
    storeOp.erase();
  }
}
}  // end anonymous namespace

std::unique_ptr<Pass> mlir::createSymbolRemovalPass() { return std::make_unique<SymbolRemovalPass>(); }
