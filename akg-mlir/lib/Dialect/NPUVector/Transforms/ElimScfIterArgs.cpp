/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <memory>
#include <utility>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "akg/Dialect/NPUVector/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"

namespace mlir {
namespace npuvector {
#define GEN_PASS_DECL_ELIMSCFITERARGS
#define GEN_PASS_DEF_ELIMSCFITERARGS
#include "akg/Dialect/NPUVector/Passes.h.inc"

namespace {

// =============================================================================
// Pass: ElimScfIterArgs
//
// This pass rewrites every scf.for whose iter_args contain values of
// NPUVectorType. The rewrite removes those iter_args and replaces them with
// in-memory traffic through memref buffers:
//   * Before the loop, each initial value is written into a freshly allocated
//     buffer.
//   * Inside the new (iter-arg-free) loop body, the latest buffer contents are
//     loaded back into NPUVectors so that the original computation can be
//     cloned unchanged. Each yielded NPUVector is written back to its buffer.
//   * After the loop, the final buffer contents are loaded and used as the
//     replacements for the original loop results.
//
// =============================================================================
class ElimScfIterArgs : public mlir::npuvector::impl::ElimScfIterArgsBase<ElimScfIterArgs> {
 public:
  ElimScfIterArgs() = default;
  ElimScfIterArgs(const ElimScfIterArgs &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, npuvector::NPUVectorDialect, memref::MemRefDialect, func::FuncDialect,
                    arith::ArithDialect>();
  }

  static bool isNpuVectorType(Type t) { return t && llvm::isa<mlir::npuvector::NPUVectorType>(t); }

 private:
  static void collectIndexOperands(Operation *op, SmallVectorImpl<Value> &out) {
    for (Value v : op->getOperands()) {
      if (v.getType().isIndex()) out.push_back(v);
    }
  }

  static bool extractFromTransferRead(mlir::npuvector::TransferReadOp rd, int rank, SmallVectorImpl<Value> &sizes,
                                      SmallVectorImpl<Value> &maxSizes) {
    const auto &maxSizesRef = rd.getMaxSizes();
    std::copy(maxSizesRef.begin(), maxSizesRef.end(), std::back_inserter(maxSizes));

    SmallVector<Value> idxOpnds;
    collectIndexOperands(rd, idxOpnds);
    int n = static_cast<int>(idxOpnds.size());

    if (n >= 2 * rank) {
      sizes.clear();
      for (int i = 0; i < rank; ++i) {
        sizes.push_back(idxOpnds[n - 2 * rank + i]);
      }
      if (maxSizes.empty()) {
        for (int i = 0; i < rank; ++i) {
          maxSizes.push_back(idxOpnds[n - rank + i]);
        }
      }
    }
    return !sizes.empty() || !maxSizes.empty();
  }

  static bool extractFromGenericNpuOp(Operation *defOp, int rank, SmallVectorImpl<Value> &sizes,
                                      SmallVectorImpl<Value> &maxSizes) {
    SmallVector<Value> idxOpnds;
    collectIndexOperands(defOp, idxOpnds);
    int n = static_cast<int>(idxOpnds.size());
    if (n < 2 * rank) return false;
    for (int i = 0; i < rank; ++i) {
      sizes.push_back(idxOpnds[n - 2 * rank + i]);
    }
    for (int i = 0; i < rank; ++i) {
      maxSizes.push_back(idxOpnds[n - rank + i]);
    }
    return true;
  }

  static bool tryExtractVecSizes(Value v, SmallVectorImpl<Value> &sizes, SmallVectorImpl<Value> &maxSizes) {
    sizes.clear();
    maxSizes.clear();

    llvm::SmallPtrSet<Operation *, 8> visited;
    Value cur = v;

    while (cur) {
      auto vt = llvm::dyn_cast<npuvector::NPUVectorType>(cur.getType());
      if (!vt) return false;

      int rank = static_cast<int>(vt.getShape().size());
      if (rank == 0) return true;

      Operation *defOp = cur.getDefiningOp();
      if (!defOp || !visited.insert(defOp).second) return false;

      if (auto rd = llvm::dyn_cast<npuvector::TransferReadOp>(defOp)) {
        if (extractFromTransferRead(rd, rank, sizes, maxSizes)) return true;
      }
      if (defOp->getDialect() && defOp->getDialect()->getNamespace() == "npuvector") {
        if (extractFromGenericNpuOp(defOp, rank, sizes, maxSizes)) return true;
        sizes.clear();
        maxSizes.clear();
      }

      Value next;
      for (Value opnd : defOp->getOperands()) {
        if (isNpuVectorType(opnd.getType())) {
          next = opnd;
          break;
        }
      }
      cur = next;
    }
    return false;
  }

  static MemRefType buildBufferType(mlir::npuvector::NPUVectorType vt, ArrayRef<Value> sizes, ArrayRef<Value> maxSizes,
                                    SmallVectorImpl<Value> &dynOperands) {
    SmallVector<int64_t> bufShape;
    for (size_t d = 0; d < vt.getShape().size(); ++d) {
      int64_t dim = vt.getShape()[d];
      if (!ShapedType::isDynamic(dim)) {
        bufShape.push_back(dim);
        continue;
      }
      Value chosen;
      if (d < maxSizes.size()) {
        chosen = maxSizes[d];
      } else if (d < sizes.size()) {
        chosen = sizes[d];
      }
      if (chosen) {
        if (auto cst = chosen.getDefiningOp<arith::ConstantIndexOp>()) {
          bufShape.push_back(cst.value());
          continue;
        }
        bufShape.push_back(ShapedType::kDynamic);
        dynOperands.push_back(chosen);
        continue;
      }
      bufShape.push_back(ShapedType::kDynamic);
    }
    return MemRefType::get(bufShape, vt.getElementType());
  }

  static Operation *findAllocAnchor(scf::ForOp forOp) {
    Operation *earliest = nullptr;
    const Block *parent = forOp->getBlock();
    for (Value initVal : forOp.getInitArgs()) {
      Operation *defOp = initVal.getDefiningOp();
      if (!defOp) continue;
      if (defOp->getBlock() != parent) continue;
      if (!earliest || defOp->isBeforeInBlock(earliest)) {
        earliest = defOp;
      }
    }
    return earliest;
  }

  static Value allocOneBuffer(OpBuilder &allocBuilder, Location loc, Value initVal, SmallVector<Value> &outSizes,
                              SmallVector<Value> &outMaxSizes) {
    auto npuVec = llvm::cast<mlir::npuvector::NPUVectorType>(initVal.getType());
    // Best-effort recovery; if it fails the output vectors are empty, which
    // is the same fallback behavior as the original `value_or({})` path.
    (void)tryExtractVecSizes(initVal, outSizes, outMaxSizes);

    SmallVector<Value> dynOperands;
    MemRefType memrefType = buildBufferType(npuVec, outSizes, outMaxSizes, dynOperands);
    if (dynOperands.empty()) {
      return allocBuilder.create<memref::AllocOp>(loc, memrefType);
    }
    return allocBuilder.create<memref::AllocOp>(loc, memrefType, dynOperands);
  }

  // Allocate buffers for every NPUVector iter_arg of `forOp`. The recovered
  // size info per init arg is appended to `allSizes` and `allMaxSizes`.
  void allocateAllBuffers(scf::ForOp forOp, OpBuilder &allocBuilder, Location loc, SmallVectorImpl<Value> &buffers,
                          SmallVector<SmallVector<Value>> &allSizes, SmallVector<SmallVector<Value>> &allMaxSizes) {
    for (Value initVal : forOp.getInitArgs()) {
      SmallVector<Value> sizes;
      SmallVector<Value> maxSizes;
      Value buf = allocOneBuffer(allocBuilder, loc, initVal, sizes, maxSizes);
      buffers.push_back(buf);
      allSizes.push_back(std::move(sizes));
      allMaxSizes.push_back(std::move(maxSizes));
    }
  }

  static void padSizesForShape(OpBuilder &b, Location loc, ArrayRef<int64_t> shape, SmallVectorImpl<Value> &sizes,
                               SmallVectorImpl<Value> &maxSizes) {
    for (size_t d = 0; d < shape.size(); ++d) {
      int64_t fallback = ShapedType::isDynamic(shape[d]) ? 0 : shape[d];
      if (d >= sizes.size()) {
        sizes.push_back(b.create<arith::ConstantIndexOp>(loc, fallback));
      }
      if (d >= maxSizes.size()) {
        maxSizes.push_back(b.create<arith::ConstantIndexOp>(loc, fallback));
      }
    }
  }

  static Value readBufferAsVector(OpBuilder &b, Location loc, mlir::npuvector::NPUVectorType vt, Value buffer,
                                  Value zeroIdx, ArrayRef<Value> partialSizes, ArrayRef<Value> partialMaxSizes) {
    ArrayRef<int64_t> shape = vt.getShape();
    SmallVector<Value> sizes(partialSizes.begin(), partialSizes.end());
    SmallVector<Value> maxSizes(partialMaxSizes.begin(), partialMaxSizes.end());
    padSizesForShape(b, loc, shape, sizes, maxSizes);

    SmallVector<Value> idx(shape.size(), zeroIdx);
    Value padding = b.create<arith::ConstantOp>(loc, vt.getElementType(), b.getZeroAttr(vt.getElementType()));
    return b.create<mlir::npuvector::TransferReadOp>(loc, vt, buffer, idx, padding, /*mask=*/Value(), sizes, maxSizes);
  }

  static void writeVectorToBuffer(OpBuilder &b, Location loc, Value vec, Value buffer, Value zeroIdx) {
    auto npuVec = llvm::cast<mlir::npuvector::NPUVectorType>(vec.getType());
    SmallVector<Value> idx(npuVec.getShape().size(), zeroIdx);
    b.create<mlir::npuvector::TransferWriteOp>(loc, TypeRange{}, vec, buffer, idx, /*mask=*/Value());
  }

  void writeSeedValues(OpBuilder &builder, Location loc, scf::ForOp forOp, ArrayRef<Value> buffers, Value zeroIdx) {
    for (size_t i = 0; i < forOp.getInitArgs().size(); ++i) {
      writeVectorToBuffer(builder, loc, forOp.getInitArgs()[i], buffers[i], zeroIdx);
    }
  }

  void buildNewLoopBody(scf::ForOp oldFor, scf::ForOp newFor, ArrayRef<Value> buffers,
                        ArrayRef<SmallVector<Value>> allSizes, ArrayRef<SmallVector<Value>> allMaxSizes) {
    Block *oldBody = oldFor.getBody();
    Block *newBody = newFor.getBody();
    Location loc = oldFor.getLoc();
    OpBuilder bb(newBody, newBody->begin());
    Value c0 = bb.create<arith::ConstantIndexOp>(loc, 0);

    IRMapping map;
    // Remap the induction variable.
    map.map(oldBody->getArgument(0), newBody->getArgument(0));

    for (size_t i = 0; i < oldFor.getInitArgs().size(); ++i) {
      auto vt = llvm::cast<mlir::npuvector::NPUVectorType>(oldFor.getInitArgs()[i].getType());
      Value loaded = readBufferAsVector(bb, loc, vt, buffers[i], c0, allSizes[i], allMaxSizes[i]);
      map.map(oldBody->getArgument(i + 1), loaded);
    }

    // Clone all original ops except the terminator (the yield).
    for (Operation &op : oldBody->without_terminator()) {
      bb.clone(op, map);
    }

    // Translate the yield into TransferWriteOps that update the buffers.
    auto yield = cast<scf::YieldOp>(oldBody->getTerminator());
    for (size_t i = 0; i < yield.getNumOperands(); ++i) {
      Value v = map.lookup(yield.getOperand(i));
      writeVectorToBuffer(bb, loc, v, buffers[i], c0);
    }
  }

  void replaceForOpResults(OpBuilder &builder, Location loc, scf::ForOp forOp, ArrayRef<Value> buffers,
                           ArrayRef<SmallVector<Value>> allSizes, ArrayRef<SmallVector<Value>> allMaxSizes,
                           Value zeroIdx) {
    SmallVector<Value> results;
    for (size_t i = 0; i < forOp.getNumResults(); ++i) {
      auto vt = llvm::cast<mlir::npuvector::NPUVectorType>(forOp.getResult(i).getType());
      Value loaded = readBufferAsVector(builder, loc, vt, buffers[i], zeroIdx, allSizes[i], allMaxSizes[i]);
      results.push_back(loaded);
    }
    forOp.replaceAllUsesWith(results);
    forOp.erase();
  }

 public:
  void eliminateIterArgs(func::FuncOp funcOp) {
    SmallVector<scf::ForOp> targets;
    funcOp.walk([&](scf::ForOp forOp) {
      if (forOp.getInitArgs().empty()) return;
      for (Value v : forOp.getInitArgs()) {
        if (isNpuVectorType(v.getType())) {
          targets.push_back(forOp);
          break;
        }
      }
    });
    for (scf::ForOp forOp : targets) {
      transformForOp(forOp);
    }
  }

  // Rewrite a single scf.for so that its NPUVector iter_args are routed
  // through memref buffers instead of being passed as block arguments.
  // The high-level structure is:
  //   1. Allocate a buffer per iter_arg (using a stable anchor point).
  //   2. Seed each buffer with its initial value just before the for op.
  //   3. Create a new for op without iter_args.
  //   4. Populate the new body to read/clone/write through the buffers.
  //   5. Read the final values out of the buffers and replace the old
  //      results, then erase the old for op.
  void transformForOp(scf::ForOp forOp) {
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value c0Outer = builder.create<arith::ConstantIndexOp>(loc, 0);

    // Anchor allocations at the earliest same-block init value producer.
    Operation *anchor = findAllocAnchor(forOp);
    OpBuilder allocBuilder = anchor ? OpBuilder(anchor) : OpBuilder(forOp);

    SmallVector<Value> buffers;
    SmallVector<SmallVector<Value>> allSizes;
    SmallVector<SmallVector<Value>> allMaxSizes;
    allSizes.reserve(forOp.getInitArgs().size());
    allMaxSizes.reserve(forOp.getInitArgs().size());

    // Pass 1: allocate all backing buffers, retaining recovered size info.
    allocateAllBuffers(forOp, allocBuilder, loc, buffers, allSizes, allMaxSizes);

    // Pass 2: seed-write the initial values immediately before the for op.
    writeSeedValues(builder, loc, forOp, buffers, c0Outer);

    // Build the replacement loop with no iter_args.
    auto newFor = builder.create<scf::ForOp>(loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep());

    // Translate the original body into the new loop, routed via buffers.
    buildNewLoopBody(forOp, newFor, buffers, allSizes, allMaxSizes);

    // After the new loop, read the final buffer contents and use them as the
    // replacements for the original for op's results.
    builder.setInsertionPointAfter(newFor);
    replaceForOpResults(builder, loc, forOp, buffers, allSizes, allMaxSizes, c0Outer);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> targetFuncs;
    module.walk([&](func::FuncOp f) { targetFuncs.push_back(f); });
    for (auto f : targetFuncs) eliminateIterArgs(f);
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createElimScfIterArgsPass() { return std::make_unique<ElimScfIterArgs>(); }

}  // namespace npuvector
}  // namespace mlir
