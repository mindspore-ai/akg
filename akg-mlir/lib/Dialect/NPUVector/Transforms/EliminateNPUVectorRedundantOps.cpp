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

#include "akg/Dialect/NPUVector/Transforms/EliminateNPUVectorRedundantOps.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Dialect/NPUVector/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace npuvector {
#define GEN_PASS_DECL_ELIMINATENPUVECTORREDUNDANTOPS
#define GEN_PASS_DEF_ELIMINATENPUVECTORREDUNDANTOPS
#include "akg/Dialect/NPUVector/Passes.h.inc"

namespace {

constexpr int kMaxMemrefRootTraceSteps = 32;

// This pass performs local redundant memory-access elimination for NPUVector IR.
//
// Supported cases:
// 1. Equivalent transfer_read ops in the same block, with no intervening memory-writing side-effect op.
// 2. A transfer_read from the same tile after a preceding transfer_write, forwarded to the written SSA value.
// 3. Local memref.alloc buffers that become write-only after forwarding.
//
// Unsupported cases:
// 1. Forwarding across unknown memory-writing side effects, such as memref.store or unknown side-effect ops.
// 2. Available-value propagation across blocks. Nested regions are processed recursively, but outer block state is
//    not passed into or out of nested blocks.
// 3. Complex alias analysis. This pass only traces roots through subview/cast/expand_shape/collapse_shape.
// 4. Aggressive dynamic-bound reasoning when write/read matching needs more than source/index/type equivalence.
// 5. Non-NPUVector load/store elimination.
struct TileAccessKey {
  Value source;
  Value root;
  Type resultType;
  SmallVector<Value> indices;
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> maxSizes;
};

struct AvailableRead {
  TileAccessKey key;
  Value value;
};

struct AvailableWrite {
  TileAccessKey key;
  Value value;
};

static Value traceMemrefRoot(Value value) {
  Value current = value;
  for (int step = 0; step < kMaxMemrefRootTraceSteps; ++step) {
    Operation *defOp = current.getDefiningOp();
    if (defOp == nullptr) {
      break;
    }

    if (auto subview = dyn_cast<memref::SubViewOp>(defOp)) {
      current = subview.getSource();
    } else if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      current = cast.getSource();
    } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(defOp)) {
      current = expand.getSrc();
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(defOp)) {
      current = collapse.getSrc();
    } else {
      break;
    }
  }
  return current;
}

static bool sameValues(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    if (lhsValue != rhsValue) {
      return false;
    }
  }
  return true;
}

static bool sameKey(const TileAccessKey &lhs, const TileAccessKey &rhs) {
  return lhs.source == rhs.source && lhs.resultType == rhs.resultType && sameValues(lhs.indices, rhs.indices) &&
         sameValues(lhs.dynamicSizes, rhs.dynamicSizes) && sameValues(lhs.maxSizes, rhs.maxSizes);
}

static bool sameMemoryTile(const TileAccessKey &lhs, const TileAccessKey &rhs) {
  return lhs.source == rhs.source && lhs.resultType == rhs.resultType && sameValues(lhs.indices, rhs.indices);
}

static TileAccessKey getReadKey(npuvector::TransferReadOp op) {
  TileAccessKey key;
  key.source = op.getSource();
  key.root = traceMemrefRoot(key.source);
  key.resultType = op.getResult().getType();
  key.indices.assign(op.getIndices().begin(), op.getIndices().end());
  key.dynamicSizes.assign(op.getDynamicSizes().begin(), op.getDynamicSizes().end());
  key.maxSizes.assign(op.getMaxSizes().begin(), op.getMaxSizes().end());
  return key;
}

static TileAccessKey getWriteKey(npuvector::TransferWriteOp op) {
  TileAccessKey key;
  key.source = op.getSource();
  key.root = traceMemrefRoot(key.source);
  key.resultType = op.getVector().getType();
  key.indices.assign(op.getIndices().begin(), op.getIndices().end());
  return key;
}

template <typename EntryT>
static Value lookupValue(const SmallVectorImpl<EntryT> &entries, const TileAccessKey &key) {
  auto entry =
    std::find_if(entries.rbegin(), entries.rend(), [&key](const EntryT &entry) { return sameKey(entry.key, key); });
  return entry == entries.rend() ? Value() : entry->value;
}

static Value lookupLastWrite(ArrayRef<AvailableWrite> entries, const TileAccessKey &key) {
  auto entry = std::find_if(entries.rbegin(), entries.rend(),
                            [&key](const AvailableWrite &entry) { return sameMemoryTile(entry.key, key); });
  return entry == entries.rend() ? Value() : entry->value;
}

template <typename EntryT>
static void eraseEntriesForRoot(SmallVectorImpl<EntryT> &entries, Value root) {
  llvm::erase_if(entries, [root](const EntryT &entry) { return entry.key.root == root; });
}

template <typename EntryT>
static void eraseEntriesForKey(SmallVectorImpl<EntryT> &entries, const TileAccessKey &key) {
  llvm::erase_if(entries, [&key](const EntryT &entry) { return sameKey(entry.key, key); });
}

static bool mayWriteMemory(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    iface.getEffects(effects);
    return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
      return isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect());
    });
  }
  return !isMemoryEffectFree(op);
}

static void processBlock(Block &block, SmallVectorImpl<Operation *> &toErase) {
  SmallVector<AvailableRead> availableReads;
  SmallVector<AvailableWrite> availableWrites;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (auto readOp = dyn_cast<npuvector::TransferReadOp>(&op)) {
      TileAccessKey key = getReadKey(readOp);
      if (Value forwarded = lookupLastWrite(availableWrites, key)) {
        readOp.getResult().replaceAllUsesWith(forwarded);
        toErase.push_back(readOp);
        continue;
      }
      if (Value existing = lookupValue(availableReads, key)) {
        readOp.getResult().replaceAllUsesWith(existing);
        toErase.push_back(readOp);
        continue;
      }
      availableReads.push_back({std::move(key), readOp.getResult()});
      continue;
    }

    if (auto writeOp = dyn_cast<npuvector::TransferWriteOp>(&op)) {
      TileAccessKey key = getWriteKey(writeOp);
      eraseEntriesForRoot(availableReads, key.root);
      eraseEntriesForKey(availableWrites, key);
      availableWrites.push_back({std::move(key), writeOp.getVector()});
      continue;
    }

    for (Region &region : op.getRegions()) {
      for (Block &nestedBlock : region) {
        processBlock(nestedBlock, toErase);
      }
    }

    if (mayWriteMemory(&op)) {
      availableReads.clear();
      availableWrites.clear();
    }
  }
}

static void eraseWriteOnlyLocalAllocs(func::FuncOp funcOp) {
  SmallVector<memref::AllocOp> allocsToErase;
  SmallVector<Operation *> writesToErase;

  funcOp.walk([&allocsToErase, &writesToErase](memref::AllocOp allocOp) {
    if (allocOp->use_empty()) {
      allocsToErase.push_back(allocOp);
      return;
    }

    SmallVector<Operation *> users(allocOp->getUsers().begin(), allocOp->getUsers().end());
    if (llvm::all_of(users, [](Operation *user) { return isa<npuvector::TransferWriteOp>(user); })) {
      writesToErase.append(users.begin(), users.end());
      allocsToErase.push_back(allocOp);
    }
  });

  for (Operation *write : writesToErase) {
    write->erase();
  }
  for (memref::AllocOp allocOp : allocsToErase) {
    if (allocOp->use_empty()) {
      allocOp.erase();
    }
  }
}

class EliminateNPUVectorRedundantOps
    : public npuvector::impl::EliminateNPUVectorRedundantOpsBase<EliminateNPUVectorRedundantOps> {
 public:
  EliminateNPUVectorRedundantOps() = default;
  EliminateNPUVectorRedundantOps(const EliminateNPUVectorRedundantOps &) = default;
  EliminateNPUVectorRedundantOps &operator=(const EliminateNPUVectorRedundantOps &) = default;

  void runOnOperation() override {
    if (getOperation().isDeclaration()) {
      return;
    }

    SmallVector<Operation *> toErase;
    processBlock(getOperation().getBody().front(), toErase);

    for (Operation *op : llvm::reverse(toErase)) {
      if (op->use_empty()) {
        op->erase();
      }
    }
    eraseWriteOnlyLocalAllocs(getOperation());
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createEliminateNPUVectorRedundantOpsPass() {
  return std::make_unique<EliminateNPUVectorRedundantOps>();
}

}  // namespace npuvector
}  // namespace mlir
