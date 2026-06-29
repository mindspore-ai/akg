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

#include "akg/Dialect/Affine/Transforms/HoistLoopIndependentOps.h"

#include <algorithm>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "akg/Utils/Constants.h"

namespace mlir {
#define GEN_PASS_DEF_HOISTLOOPINDEPENDENTOPS
#define GEN_PASS_DECL_HOISTLOOPINDEPENDENTOPS
#include "akg/Dialect/Affine/Passes.h.inc"

}  // namespace mlir

#define DEBUG_TYPE "hoist-loop-independent-ops"

namespace mlir {
namespace {

struct HoistLoopIndependentOps : public impl::HoistLoopIndependentOpsBase<HoistLoopIndependentOps> {
 public:
  HoistLoopIndependentOps() {}

  void runOnOperation() override {
    auto funcOp = getOperation();

    hoistLoopInvariantLoad(funcOp);

    moveIndependentOpsBeforeFirstAffineFor(funcOp);
  }

 private:
  // Recursively checks whether a Value depends on seenAffineForOps,
  // i.e., computations inside all previously encountered affine.for ops.
  static bool valueDependsOnSeenAffineFors(Value v, const llvm::DenseSet<Operation *> &seenAffineForOps,
                                           llvm::DenseMap<Value, bool> &valueDepCache,
                                           llvm::DenseSet<Operation *> &visitingOps) {
    auto cacheIt = valueDepCache.find(v);
    if (cacheIt != valueDepCache.end()) {
      return cacheIt->second;
    }

    bool depends = false;

    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      Block *ownerBlock = blockArg.getOwner();
      Operation *parentOp = (ownerBlock != nullptr) ? ownerBlock->getParentOp() : nullptr;
      depends = (parentOp != nullptr && seenAffineForOps.contains(parentOp));
    } else if (auto opResult = dyn_cast<OpResult>(v)) {
      Operation *defOp = opResult.getOwner();

      if (seenAffineForOps.contains(defOp)) {
        depends = true;
      } else {
        if (!visitingOps.contains(defOp)) {
          visitingOps.insert(defOp);
          depends =
            std::any_of(defOp->getOperands().begin(), defOp->getOperands().end(),
                        [&seenAffineForOps, &valueDepCache, &visitingOps](Value operand) {
                          return valueDependsOnSeenAffineFors(operand, seenAffineForOps, valueDepCache, visitingOps);
                        });
          visitingOps.erase(defOp);
        } else {
          // Cycles are generally not expected in SSA; handle conservatively.
          depends = false;
        }
      }
    } else {
      depends = false;
    }

    valueDepCache[v] = depends;
    return depends;
  }

  // Checks whether any operand of op depends on all previously seen affine.for ops.
  static bool opDependsOnSeenAffineFors(Operation *op, const llvm::DenseSet<Operation *> &seenAffineForOps,
                                        llvm::DenseMap<Value, bool> &valueDepCache) {
    llvm::DenseSet<Operation *> visitingOps;
    return std::any_of(op->getOperands().begin(), op->getOperands().end(),
                       [&seenAffineForOps, &valueDepCache, &visitingOps](Value operand) {
                         return valueDependsOnSeenAffineFors(operand, seenAffineForOps, valueDepCache, visitingOps);
                       });
  }

  static void collectNestedOps(Operation *root, llvm::DenseSet<Operation *> &set) {
    root->walk([&set](Operation *nested) { set.insert(nested); });
  }

  // Collect the set of memrefs that are read/written by op (including its
  // nested ops). These are used to detect memory dependencies that are not
  // visible through SSA value chains.
  static void collectMemRefEffects(Operation *root, llvm::DenseSet<Value> &readMemRefs,
                                   llvm::DenseSet<Value> &writtenMemRefs) {
    root->walk([&readMemRefs, &writtenMemRefs](Operation *op) {
      if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(op)) {
        readMemRefs.insert(affineLoad.getMemRef());
      } else if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) {
        writtenMemRefs.insert(affineStore.getMemRef());
      } else if (auto memLoad = dyn_cast<memref::LoadOp>(op)) {
        readMemRefs.insert(memLoad.getMemRef());
      } else if (auto memStore = dyn_cast<memref::StoreOp>(op)) {
        writtenMemRefs.insert(memStore.getMemRef());
      }
    });
  }

  // Checks whether op has a memory dependency (RAW/WAR/WAW) with the memory
  // footprint of the ops that must stay after the first affine.for.
  // Conservatively compares memref SSA values by identity.
  static bool opHasMemoryDepWithStaySet(Operation *op, const llvm::DenseSet<Value> &stayReadMemRefs,
                                        const llvm::DenseSet<Value> &stayWrittenMemRefs) {
    llvm::DenseSet<Value> opReadMemRefs;
    llvm::DenseSet<Value> opWrittenMemRefs;
    collectMemRefEffects(op, opReadMemRefs, opWrittenMemRefs);

    // Read-after-write: op reads a memref written inside the stay set.
    for (Value readMemRef : opReadMemRefs) {
      if (stayWrittenMemRefs.contains(readMemRef)) {
        return true;
      }
    }

    // Write-after-write / write-after-read: op writes a memref that the stay
    // set reads or writes.
    for (Value writtenMemRef : opWrittenMemRefs) {
      if (stayWrittenMemRefs.contains(writtenMemRef) || stayReadMemRefs.contains(writtenMemRef)) {
        return true;
      }
    }

    return false;
  }

  void moveIndependentOpsBeforeFirstAffineFor(func::FuncOp funcOp) {
    Block &block = funcOp.getBody().front();

    Operation *firstAffineFor = nullptr;
    for (Operation &op : block) {
      if (isa<affine::AffineForOp>(op)) {
        firstAffineFor = &op;
        break;
      }
    }

    if (firstAffineFor == nullptr) {
      llvm::errs() << "[AKG] HoistLoopIndependentOps: no top-level affine.for found in func " << funcOp.getName()
                   << "\n";
      return;
    }

    // stayOps: all ops that must remain after firstAffineFor because they
    // (transitively) depend on the loop computations. This includes the
    // top-level affine.for ops encountered so far (with their nested ops) as
    // well as any non-loop op that is found to depend on them through SSA
    // values or through memory. Accumulating non-loop ops here is required so
    // that their downstream consumers are also kept in place.
    llvm::DenseSet<Operation *> stayOps;
    collectNestedOps(firstAffineFor, stayOps);

    // Memory footprint of the stay set, used to detect memory dependencies
    // (load/store on the same memref) that SSA dependency analysis misses.
    llvm::DenseSet<Value> stayReadMemRefs;
    llvm::DenseSet<Value> stayWrittenMemRefs;
    collectMemRefEffects(firstAffineFor, stayReadMemRefs, stayWrittenMemRefs);

    SmallVector<Operation *, kSmallVectorSizeEight> toMove;
    llvm::DenseMap<Value, bool> valueDepCache;

    for (auto it = std::next(firstAffineFor->getIterator()), e = block.end(); it != e; ++it) {
      Operation *op = &*it;

      if (op == block.getTerminator()) {
        continue;
      }

      // Another top-level affine.for must stay; fold it into the stay set.
      if (isa<affine::AffineForOp>(op)) {
        collectNestedOps(op, stayOps);
        collectMemRefEffects(op, stayReadMemRefs, stayWrittenMemRefs);
        // The dependency cache is relative to the stay set, which just grew.
        valueDepCache.clear();
        continue;
      }

      // Keep the op in place if it depends on the stay set either through SSA
      // values or through memory; otherwise it is safe to hoist.
      bool mustStay = opDependsOnSeenAffineFors(op, stayOps, valueDepCache) ||
                      opHasMemoryDepWithStaySet(op, stayReadMemRefs, stayWrittenMemRefs);

      if (mustStay) {
        collectNestedOps(op, stayOps);
        collectMemRefEffects(op, stayReadMemRefs, stayWrittenMemRefs);
        valueDepCache.clear();
        continue;
      }

      toMove.push_back(op);
    }

    if (toMove.empty()) {
      llvm::errs() << "[AKG] HoistLoopIndependentOps: no independent ops to move "
                   << "after first top-level affine.for in func " << funcOp.getName() << "\n";
      return;
    }

    // Preserve original relative order while moving.
    for (Operation *op : toMove) {
      op->moveBefore(firstAffineFor);
    }
  }

  // Helper function to check if a value depends on a loop IV
  bool valueDependsOnIV(Value value, Value iv) {
    if (value == iv) {
      return true;
    }

    Operation *defOp = value.getDefiningOp();
    if (defOp == nullptr) {
      return false;
    }

    bool depends = false;
    defOp->walk([&depends, &iv](Operation *op) {
      if (std::any_of(op->getOperands().begin(), op->getOperands().end(),
                      [&iv](Value operand) { return operand == iv; })) {
        depends = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    return depends;
  }

  // Helper function to check if a memref is invariant to all loops
  bool isMemrefInvariantToAllLoops(affine::AffineLoadOp loadOp,
                                   const SmallVector<affine::AffineForOp> &allEnclosingLoops) {
    Value memref = loadOp.getMemRef();

    for (auto enclosingFor : allEnclosingLoops) {
      Value iv = enclosingFor.getInductionVar();
      // Check if memref depends on loop IV
      if (valueDependsOnIV(memref, iv)) {
        return false;
      }

      // Check if indices depend on loop IV
      for (Value operand : loadOp.getIndices()) {
        if (operand == iv) {
          return false;
        }
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getOwner() == enclosingFor.getBody()) {
            return false;
          }
        }
      }
    }

    return true;
  }

  // Helper function to check if there are any stores to the memref in the loop subtree
  bool hasStoreToMemrefInLoopTree(affine::AffineForOp forOp, Value memref) {
    bool hasStore = false;
    forOp.walk([&hasStore, &memref](Operation *op) {
      if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (storeOp.getMemRef() == memref) {
          hasStore = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return hasStore;
  }

  // Collect loop-invariant loads that are candidates for hoisting
  SmallVector<Operation *> collectLoopInvariantLoads(affine::AffineForOp outermostLoop) {
    SmallVector<affine::AffineForOp> nestedLoops;
    outermostLoop.walk([&nestedLoops](affine::AffineForOp forOp) { nestedLoops.push_back(forOp); });

    SmallVector<Operation *> toHoist;
    for (auto forOp : nestedLoops) {
      for (Operation &op : forOp.getBody()->getOperations()) {
        auto loadOp = dyn_cast<affine::AffineLoadOp>(&op);
        if (!loadOp) {
          continue;
        }

        Value memref = loadOp.getMemRef();
        Operation *memrefDefOp = memref.getDefiningOp();
        if ((memrefDefOp == nullptr) || !isa<memref::AllocOp>(memrefDefOp)) {
          continue;
        }

        SmallVector<affine::AffineForOp> allEnclosingLoops;
        affine::getAffineForIVs(*loadOp, &allEnclosingLoops);
        allEnclosingLoops.push_back(forOp);

        if (!isMemrefInvariantToAllLoops(loadOp, allEnclosingLoops)) {
          continue;
        }

        if (hasStoreToMemrefInLoopTree(forOp, memref)) {
          continue;
        }

        toHoist.push_back(loadOp);
      }
    }
    return toHoist;
  }

  // Try to find a scalar store value that can be forwarded to replace a load
  std::pair<Value, affine::AffineStoreOp> tryForwardStoreToLoad(Value memref, affine::AffineForOp outermostLoop) {
    Block *block = outermostLoop->getBlock();
    for (auto it = Block::iterator(outermostLoop); it != block->begin(); --it) {
      auto storeOp = dyn_cast<affine::AffineStoreOp>(&*it);
      if (!storeOp || storeOp.getMemRef() != memref) {
        continue;
      }

      auto memrefType = cast<MemRefType>(memref.getType());
      if (memrefType.getRank() == 0) {
        return {storeOp.getValueToStore(), storeOp};
      }
      break;
    }
    return {Value(), nullptr};
  }

  // Eliminate a load by forwarding a stored value, or fall back to hoisting
  void eliminateOrHoistLoad(Operation *op, affine::AffineForOp outermostLoop) {
    auto loadOp = cast<affine::AffineLoadOp>(op);
    Value memref = loadOp.getMemRef();

    auto [forwardedValue, matchedStore] = tryForwardStoreToLoad(memref, outermostLoop);
    if (!forwardedValue) {
      op->moveBefore(outermostLoop);
      return;
    }

    loadOp.getResult().replaceAllUsesWith(forwardedValue);
    SmallVector<Operation *, kSmallVectorSizeFour> toErase;
    toErase.push_back(op);

    if (matchedStore) {
      auto users = memref.getUsers();
      bool memrefStillUsed = std::any_of(users.begin(), users.end(), [&op, &matchedStore](Operation *user) {
        return user != op && user != matchedStore.getOperation();
      });
      if (!memrefStillUsed) {
        toErase.push_back(matchedStore.getOperation());
        Operation *memrefDefOp = memref.getDefiningOp();
        if (memrefDefOp != nullptr) {
          toErase.push_back(memrefDefOp);
        }
      }
    }

    for (Operation *eraseOp : toErase) {
      eraseOp->erase();
    }
  }

  void hoistLoopInvariantLoad(func::FuncOp funcOp) {
    SmallVector<affine::AffineForOp> outermostLoops;
    funcOp.walk([&outermostLoops](affine::AffineForOp forOp) {
      if (!forOp->getParentOfType<affine::AffineForOp>()) {
        outermostLoops.push_back(forOp);
      }
    });

    for (auto outermostLoop : outermostLoops) {
      SmallVector<Operation *> toHoist = collectLoopInvariantLoads(outermostLoop);
      for (Operation *op : toHoist) {
        eliminateOrHoistLoad(op, outermostLoop);
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createHoistLoopIndependentOpsPass() {
  return std::make_unique<HoistLoopIndependentOps>();
}
}  // namespace mlir
