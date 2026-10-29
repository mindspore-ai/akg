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

#include "akg/Transforms/HoistLoopIndependentOps.h"

#include <algorithm>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Utils/Constants.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
#define GEN_PASS_DEF_HOISTLOOPINDEPENDENTOPS
#define GEN_PASS_DECL_HOISTLOOPINDEPENDENTOPS
#include "akg/Transforms/Passes.h.inc"

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
      } else if (auto transferRead = dyn_cast<npuvector::TransferReadOp>(op)) {
        readMemRefs.insert(transferRead.getSource());
      } else if (auto transferWrite = dyn_cast<npuvector::TransferWriteOp>(op)) {
        writtenMemRefs.insert(transferWrite.getSource());
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

    // No top-level affine.for (e.g. post-vectorize scf-only IR): skip phase B.
    if (firstAffineFor == nullptr) {
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
      return;
    }

    // Preserve original relative order while moving.
    for (Operation *op : toMove) {
      op->moveBefore(firstAffineFor);
    }
  }

  // ----- Phase A: hoist loop-invariant loads / transfer_read / pure ops -----

  // affine.for, scf.for and scf.forall
  static bool isForLoop(Operation *op) { return isa<affine::AffineForOp, scf::ForOp, scf::ForallOp>(op); }

  static Block *getForBody(Operation *loopOp) {
    if (auto affineFor = dyn_cast<affine::AffineForOp>(loopOp)) {
      return affineFor.getBody();
    }
    if (auto scfFor = dyn_cast<scf::ForOp>(loopOp)) {
      return scfFor.getBody();
    }
    if (auto forall = dyn_cast<scf::ForallOp>(loopOp)) {
      return forall.getBody();
    }
    return nullptr;
  }

  // Memref of a read op.
  static Value getLoadedMemRef(Operation *op) {
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      return loadOp.getMemRef();
    }
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      return loadOp.getMemRef();
    }
    if (auto transferRead = dyn_cast<npuvector::TransferReadOp>(op)) {
      return transferRead.getSource();
    }
    return Value();
  }

  // Memref of a write op.
  static Value getStoredMemRef(Operation *op) {
    if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      return storeOp.getMemRef();
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      return storeOp.getMemRef();
    }
    if (auto transferWrite = dyn_cast<npuvector::TransferWriteOp>(op)) {
      return transferWrite.getSource();
    }
    return Value();
  }

  // True iff `value` is defined outside all enclosing fors (IV/iter_args/in-loop SSA).
  static bool isDefinedOutsideLoops(Value value, ArrayRef<Operation *> enclosingLoops) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Block *owner = blockArg.getOwner();
      return !llvm::any_of(enclosingLoops,
                           [owner](Operation *loopOp) { return owner == getForBody(loopOp); });
    }

    Operation *defOp = value.getDefiningOp();
    if (defOp == nullptr) {
      return true;
    }
    return !llvm::any_of(enclosingLoops,
                         [defOp](Operation *loopOp) { return loopOp->isAncestor(defOp); });
  }

  // Safe to hoist only when every operand already dominates the loop nest.
  static bool isOpInvariantToLoops(Operation *op, ArrayRef<Operation *> enclosingLoops) {
    return llvm::all_of(op->getOperands(),
                        [&](Value operand) { return isDefinedOutsideLoops(operand, enclosingLoops); });
  }

  // Helper function to check if there are any stores to the memref in the loop subtree
  bool hasStoreToMemrefInLoopTree(Operation *loopOp, Value memref) {
    bool hasStore = false;
    loopOp->walk([&hasStore, memref](Operation *op) {
      Value written = getStoredMemRef(op);
      if (written && written == memref) {
        hasStore = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return hasStore;
  }

  // Collect enclosing fors (inner to outer).
  static void collectEnclosingFors(Operation *op, SmallVectorImpl<Operation *> &loops) {
    for (Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
      if (isForLoop(parent)) {
        loops.push_back(parent);
      }
    }
  }

  // Collect invariant reads (incl. transfer_read) and pure followers (e.g. extf).
  SmallVector<Operation *> collectLoopInvariantOps(Operation *outermostLoop) {
    SmallVector<Operation *> nestedLoops;
    outermostLoop->walk([&nestedLoops](Operation *op) {
      if (isForLoop(op)) {
        nestedLoops.push_back(op);
      }
    });

    SmallVector<Operation *> toHoist;
    for (Operation *forOp : nestedLoops) {
      Block *body = getForBody(forOp);
      if (body == nullptr) {
        continue;
      }
      for (Operation &opRef : body->getOperations()) {
        Operation *op = &opRef;
        // Skip terminators, nested fors, and ops with regions.
        if (op->hasTrait<OpTrait::IsTerminator>() || isForLoop(op) || op->getNumRegions() != 0u) {
          continue;
        }

        Value loadedMemRef = getLoadedMemRef(op);
        bool isRead = static_cast<bool>(loadedMemRef);
        // Candidates: reads, or memory-effect-free (Pure) ops such as extf.
        if (!isRead && !isMemoryEffectFree(op)) {
          continue;
        }

        SmallVector<Operation *, kSmallVectorSizeEight> enclosingLoops;
        collectEnclosingFors(op, enclosingLoops);
        if (!isOpInvariantToLoops(op, enclosingLoops)) {
          continue;
        }

        // Reject reads if the outermost nest may write the same memref.
        if (isRead && hasStoreToMemrefInLoopTree(outermostLoop, loadedMemRef)) {
          continue;
        }

        toHoist.push_back(op);
      }
    }
    return toHoist;
  }

  // Try to find a scalar store value that can be forwarded to replace a load
  std::pair<Value, affine::AffineStoreOp> tryForwardStoreToLoad(Value memref, Operation *outermostLoop) {
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
  void eliminateOrHoistOp(Operation *op, Operation *outermostLoop) {
    // Keep store-to-load forwarding for affine.load; otherwise just hoist.
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      Value memref = loadOp.getMemRef();
      auto [forwardedValue, matchedStore] = tryForwardStoreToLoad(memref, outermostLoop);
      if (forwardedValue) {
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
        return;
      }
    }

    op->moveBefore(outermostLoop);
  }

  void hoistLoopInvariantLoad(func::FuncOp funcOp) {
    SmallVector<Operation *> outermostLoops;
    funcOp.walk([&outermostLoops](Operation *op) {
      if (!isForLoop(op)) {
        return;
      }
      // Outermost: no parent affine.for / scf.for.
      bool hasParentFor = false;
      for (Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
        if (isForLoop(parent)) {
          hasParentFor = true;
          break;
        }
      }
      if (!hasParentFor) {
        outermostLoops.push_back(op);
      }
    });

    for (Operation *outermostLoop : outermostLoops) {
      // Fixpoint: hoist read first, then Pure users (e.g. extf) once operands dominate.
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<Operation *> toHoist = collectLoopInvariantOps(outermostLoop);
        for (Operation *op : toHoist) {
          eliminateOrHoistOp(op, outermostLoop);
          changed = true;
        }
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createHoistLoopIndependentOpsPass() {
  return std::make_unique<HoistLoopIndependentOps>();
}
}  // namespace mlir
