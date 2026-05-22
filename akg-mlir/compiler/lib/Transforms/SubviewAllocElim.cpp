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

#include <numeric>

#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "akg/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#ifndef GEN_PASS_DECL_SUBVIEWALLOCELIM
#define GEN_PASS_DECL_SUBVIEWALLOCELIM
#ifndef GEN_PASS_DEF_SUBVIEWALLOCELIM
#define GEN_PASS_DEF_SUBVIEWALLOCELIM
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "subview-alloc-elim"

namespace mlir {
namespace {

// Describes one subview partition of an alloc along a single axis.
struct Partition {
  memref::SubViewOp subviewOp;
  int64_t offset;  // offset on the partition axis
  int64_t size;    // size on the partition axis

  // The store operations that write into this partition (through the view chain).
  SmallVector<affine::AffineStoreOp> stores;

  // The affine.if guard enclosing the stores (may be null if unconditional).
  affine::AffineIfOp guardIfOp;
};

static bool isLoadOp(Operation *op) { return isa<affine::AffineLoadOp, memref::LoadOp>(op); }

static bool isStoreOp(Operation *op) { return isa<affine::AffineStoreOp, memref::StoreOp>(op); }

static Value getLoadResult(Operation *op) {
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) return loadOp.getResult();
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) return loadOp.getResult();
  return {};
}

static SmallVector<Value> getLoadIndices(Operation *op) {
  if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(op))
    return SmallVector<Value>(affineLoad.getIndices().begin(), affineLoad.getIndices().end());
  if (auto memrefLoad = dyn_cast<memref::LoadOp>(op))
    return SmallVector<Value>(memrefLoad.getIndices().begin(), memrefLoad.getIndices().end());
  return {};
}

// Check whether a value is used as a return operand in the function.
static bool isReturnedValue(Value val) {
  for (auto *user : val.getUsers()) {
    if (isa<func::ReturnOp>(user)) return true;
  }
  return false;
}

// Try to get the "next value" in a view chain from a user op.
// Returns the result Value if the user is a supported view-chain op, or null otherwise.
static Value getViewChainNext(Operation *user) {
  if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) return collapseOp.getResult();
  if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(user)) return expandOp.getResult();
  if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(user)) return castOp.getResult();
  if (auto castOp = dyn_cast<memref::CastOp>(user)) return castOp.getResult();
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) return subviewOp.getResult();
  return {};
}

// Recursively check if a value derived from an alloc (through view chain) is returned.
static bool isAllocReturned(memref::AllocOp allocOp) {
  SmallVector<Value> worklist;
  worklist.push_back(allocOp.getResult());
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (!visited.insert(val).second) continue;
    if (isReturnedValue(val)) return true;
    for (auto *user : val.getUsers()) {
      Value next = getViewChainNext(user);
      if (next) worklist.push_back(next);
    }
  }
  return false;
}

// Collect store ops reachable from a subview result through view chains.
static bool collectStoresFromSubview(Value memref, SmallVectorImpl<Operation *> &storeOps) {
  for (auto *user : memref.getUsers()) {
    if (isStoreOp(user)) {
      storeOps.push_back(user);
      continue;
    }
    if (isLoadOp(user) || isa<func::ReturnOp>(user)) continue;

    Value next = getViewChainNext(user);
    if (!next) return false;
    if (!collectStoresFromSubview(next, storeOps)) return false;
  }
  return true;
}

// Given a store to a subview partition, trace the stored value back to its source.
// If the value comes from a simple affine.load, return that load's memref and indices.
// If the value comes from a computation, return the operations that need to be cloned.
struct SourceInfo {
  // For simple load forwarding: the source load operation.
  affine::AffineLoadOp sourceLoad;
  // For complex computation: the operations to clone (in order), and the final value.
  SmallVector<Operation *> computeOps;
  Value finalValue;
};

static SourceInfo traceStoreSource(affine::AffineStoreOp storeOp) {
  SourceInfo info;
  Value storeVal = storeOp.getValueToStore();

  // Case 1: stored value is directly from an affine.load.
  if (auto loadOp = storeVal.getDefiningOp<affine::AffineLoadOp>()) {
    info.sourceLoad = loadOp;
    info.finalValue = storeVal;
    return info;
  }

  // Case 2: stored value is from a computation chain inside an affine.if.
  // Collect all ops in the if-block that contribute to the stored value.
  auto *parentBlock = storeOp->getBlock();
  for (auto &op : *parentBlock) {
    if (&op == storeOp.getOperation()) break;
    if (op.isBeforeInBlock(storeOp)) {
      info.computeOps.push_back(&op);
    }
  }
  info.finalValue = storeVal;
  return info;
}

// Check if a Value dominates an Operation (handles both op results and block arguments).
static bool valueDominates(Value val, Operation *op) {
  if (auto *defOp = val.getDefiningOp()) {
    // Walk up from op to find defOp's block in the ancestor chain.
    Operation *cur = op;
    while (cur) {
      if (cur->getBlock() == defOp->getBlock()) return defOp->isBeforeInBlock(cur);
      cur = cur->getParentOp();
    }
    return false;
  }
  // Block argument: dominates everything nested inside its parent op.
  Block *argBlock = cast<BlockArgument>(val).getOwner();
  Operation *parentOp = argBlock->getParentOp();
  if (!parentOp) return true;  // Function argument, always dominates.
  return parentOp->isProperAncestor(op);
}

// Check if all external operands of a set of operations dominate an insertion point.
// "External" means not defined by any op in the set itself.
static bool allExternalOperandsDominate(ArrayRef<Operation *> ops, Operation *insertionPt) {
  DenseSet<Operation *> opSet(ops.begin(), ops.end());
  for (auto *op : ops) {
    for (Value operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (opSet.count(defOp)) continue;
      }
      if (!valueDominates(operand, insertionPt)) return false;
    }
  }
  return true;
}

// Detect if stores to an alloc are transposed copies from a source memref.
// Returns the index permutation: storePerm[j] = k means alloc dim j was filled from source dim k.
struct TransposeInfo {
  SmallVector<unsigned> storePerm;    // alloc dim j ← source dim storePerm[j]
  SmallVector<unsigned> inversePerm;  // source dim k ← alloc dim inversePerm[k]
  Value sourceMemref;
};

// Find initialization stores (e.g., store 0 to the entire alloc)
struct InitInfo {
  bool hasInit = false;
  Value initValue;  // The constant value used to initialize (e.g., 0.0)
};

struct SubviewAllocElimPass : public SubviewAllocElimBase<SubviewAllocElimPass> {
 public:
  void runOnOperation() override;

 private:
  SmallVector<memref::SubViewOp> subviewOps;
  SmallVector<Operation *> directLoads;
  SmallVector<Operation *> directStores;
  SmallVector<Operation *> viewOps;
  int partitionAxis = -1;
  SmallVector<Partition> partitions;
  InitInfo initInfo;

  // Collection
  bool collectUsersRecursive(Value memref);

  // Analysis
  bool processAlloc(memref::AllocOp allocOp);
  bool processSimpleAlloc(memref::AllocOp allocOp);
  bool processTransposeAlloc(memref::AllocOp allocOp);
  bool analyzeAxisPartition(memref::AllocOp allocOp);
  std::optional<TransposeInfo> detectTransposePattern(ArrayRef<Operation *> stores, Value allocResult);
  AffineMap buildViewChainToAllocMap(Value loadMemref, Value allocResult);
  bool tracePartitionStores(Partition &partition);
  InitInfo findAllocInitialization(memref::AllocOp allocOp);

  // Transformation
  bool replaceLoads(memref::AllocOp allocOp);
  void eraseDeadOps(memref::AllocOp allocOp);
  Value cloneComputationIntoBlock(OpBuilder &builder, Location loc, ArrayRef<Operation *> computeOps, Value finalValue,
                                  IRMapping &mapping);
  Value buildPartitionValue(OpBuilder &builder, Location loc, const Partition &partition, Operation *loadOp,
                            IRMapping &mapping);
  SmallVector<Value> buildPartitionValues(OpBuilder &b, Location loc, const Partition &part,
                                          ArrayRef<Operation *> loads);
  SmallVector<Value> buildNestedIf(OpBuilder &b, Location loc, unsigned partIdx, ArrayRef<Operation *> loads,
                                   ArrayRef<Type> resultTypes, Value partAxisIdx, bool isFullCoverage);
};

// Collect all SubViewOps that are direct or indirect users of a memref value
// (through reshape/cast chains). Also collect all terminal access ops.
// Results are stored in member variables: subviewOps, directLoads, directStores, viewOps.
bool SubviewAllocElimPass::collectUsersRecursive(Value memref) {
  for (auto *user : memref.getUsers()) {
    if (isLoadOp(user)) {
      directLoads.push_back(user);
      continue;
    }
    if (isStoreOp(user)) {
      directStores.push_back(user);
      continue;
    }

    if (isa<func::ReturnOp>(user)) continue;

    if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) subviewOps.push_back(subviewOp);
    Value nextValue = getViewChainNext(user);
    if (!nextValue) return false;

    viewOps.push_back(user);
    if (!collectUsersRecursive(nextValue)) return false;
  }
  return true;
}

// Validate that a single subview is suitable for axis-partitioning:
// all strides must be 1, the partition-axis offset/size must be static, and
// non-partition dimensions must span the full alloc extent.
static bool isValidSubviewForPartition(memref::SubViewOp subviewOp, MemRefType allocType, unsigned rank,
                                       int partitionAxis) {
  auto offsets = subviewOp.getStaticOffsets();
  auto sizes = subviewOp.getStaticSizes();
  auto strides = subviewOp.getStaticStrides();

  if (!llvm::all_of(strides, [](int64_t s) { return s == 1; })) {
    return false;
  }
  if (ShapedType::isDynamic(offsets[partitionAxis]) || ShapedType::isDynamic(sizes[partitionAxis])) return false;
  for (unsigned d = 0; d < rank; d++) {
    if (static_cast<int>(d) == partitionAxis) continue;
    if (offsets[d] != 0) return false;
    if (sizes[d] != allocType.getDimSize(d) && !ShapedType::isDynamic(sizes[d])) return false;
  }
  return true;
}

// Given a set of SubViewOps on an alloc, check if they form a complete partition
// along exactly one axis. Results are stored in member variables: partitionAxis, partitions.
bool SubviewAllocElimPass::analyzeAxisPartition(memref::AllocOp allocOp) {
  auto allocType = allocOp.getType();
  unsigned rank = allocType.getRank();
  if (subviewOps.empty() || rank == 0) return false;

  // Find the partition axis: the one dimension where subviews differ in offset/size.
  partitionAxis = -1;
  for (unsigned d = 0; d < rank; d++) {
    bool differs = false;
    for (auto subviewOp : subviewOps) {
      auto offsets = subviewOp.getStaticOffsets();
      auto sizes = subviewOp.getStaticSizes();
      if (offsets.size() != rank || sizes.size() != rank) return false;
      if (offsets[d] != 0 || sizes[d] != allocType.getDimSize(d)) {
        differs = true;
      }
    }
    if (differs) {
      // multiple axes differ: not a simple partition
      if (partitionAxis != -1) return false;
      partitionAxis = static_cast<int>(d);
    }
  }

  if (partitionAxis < 0) return false;
  int64_t axisDim = allocType.getDimSize(partitionAxis);
  if (ShapedType::isDynamic(axisDim)) return false;

  // Collect partitions and verify they cover the axis completely.
  partitions.clear();
  for (auto subviewOp : subviewOps) {
    if (!isValidSubviewForPartition(subviewOp, allocType, rank, partitionAxis)) return false;

    auto offsets = subviewOp.getStaticOffsets();
    auto sizes = subviewOp.getStaticSizes();
    Partition p;
    p.subviewOp = subviewOp;
    p.offset = offsets[partitionAxis];
    p.size = sizes[partitionAxis];
    p.guardIfOp = nullptr;
    partitions.push_back(p);
  }

  // Sort partitions by offset.
  llvm::sort(partitions, [](const Partition &a, const Partition &b) { return a.offset < b.offset; });

  // TODO(hujiahui): discontinuous scenes
  // Verify contiguous coverage: offsets must be [0, s0, s0+s1, ...] summing to axisDim.
  int64_t expectedOffset = 0;
  for (auto &p : partitions) {
    if (p.offset != expectedOffset) return false;
    expectedOffset += p.size;
  }
  return true;
}

// Detect if stores to an alloc are transposed copies from a source memref.
std::optional<TransposeInfo> SubviewAllocElimPass::detectTransposePattern(ArrayRef<Operation *> stores,
                                                                          Value allocResult) {
  if (stores.empty()) return std::nullopt;

  TransposeInfo info;
  for (auto *op : stores) {
    auto storeOp = dyn_cast<affine::AffineStoreOp>(op);
    if (!storeOp || storeOp.getMemRef() != allocResult) return std::nullopt;
    if (!storeOp.getAffineMap().isIdentity()) return std::nullopt;

    auto sourceLoad = storeOp.getValueToStore().getDefiningOp<affine::AffineLoadOp>();
    if (!sourceLoad || !sourceLoad.getAffineMap().isIdentity()) return std::nullopt;

    auto storeIndices = storeOp.getIndices();
    auto loadIndices = sourceLoad.getIndices();
    if (storeIndices.size() != loadIndices.size()) return std::nullopt;

    unsigned rank = storeIndices.size();
    SmallVector<unsigned> perm(rank);
    DenseSet<unsigned> seen;
    for (unsigned j = 0; j < rank; j++) {
      bool found = false;
      for (unsigned k = 0; k < rank; k++) {
        if (storeIndices[j] == loadIndices[k]) {
          perm[j] = k;
          found = true;
          break;
        }
      }
      if (!found || !seen.insert(perm[j]).second) return std::nullopt;
    }

    if (info.sourceMemref) {
      if (info.sourceMemref != sourceLoad.getMemRef() || info.storePerm != perm) return std::nullopt;
    } else {
      info.sourceMemref = sourceLoad.getMemRef();
      info.storePerm = perm;
    }
  }

  unsigned rank = info.storePerm.size();
  info.inversePerm.resize(rank);
  for (unsigned j = 0; j < rank; j++) info.inversePerm[info.storePerm[j]] = j;
  return info;
}

// Apply a CollapseShapeOp to the current expression list, delinearizing each
// collapsed group. Returns false if any dimension is dynamic.
static bool applyCollapseShapeToExprs(memref::CollapseShapeOp cs, SmallVectorImpl<AffineExpr> &exprs) {
  auto reassoc = cs.getReassociationIndices();
  auto srcType = cs.getSrcType();
  SmallVector<AffineExpr> newExprs(srcType.getRank());
  unsigned resultDim = 0;
  for (const auto &group : reassoc) {
    AffineExpr remaining = exprs[resultDim++];
    for (int i = group.size() - 1; i >= 0; i--) {
      int64_t dimSize = srcType.getDimSize(group[i]);
      if (ShapedType::isDynamic(dimSize)) return false;
      if (i == 0) {
        newExprs[group[i]] = remaining;
      } else {
        newExprs[group[i]] = remaining % dimSize;
        remaining = remaining.floorDiv(dimSize);
      }
    }
  }
  exprs = newExprs;
  return true;
}

// Apply an ExpandShapeOp to the current expression list, linearizing each
// expanded group. Returns false if any dimension is dynamic.
static bool applyExpandShapeToExprs(memref::ExpandShapeOp es, SmallVectorImpl<AffineExpr> &exprs, MLIRContext *ctx) {
  auto reassoc = es.getReassociationIndices();
  auto resultType = es.getResultType();
  unsigned srcRank = es.getSrcType().getRank();
  SmallVector<AffineExpr> newExprs(srcRank);
  for (unsigned k = 0; k < reassoc.size(); k++) {
    const auto &group = reassoc[k];
    AffineExpr combined = getAffineConstantExpr(0, ctx);
    for (unsigned i = 0; i < group.size(); i++) {
      int64_t stride = 1;
      for (unsigned j = i + 1; j < group.size(); j++) {
        int64_t dimSize = resultType.getDimSize(group[j]);
        if (ShapedType::isDynamic(dimSize)) return false;
        stride *= dimSize;
      }
      combined = combined + exprs[group[i]] * stride;
    }
    newExprs[k] = combined;
  }
  exprs = newExprs;
  return true;
}

// Trace from a load's memref back to allocResult through the view chain,
// building an AffineMap that maps (load indices) → (alloc indices).
AffineMap SubviewAllocElimPass::buildViewChainToAllocMap(Value loadMemref, Value allocResult) {
  auto *ctx = &getContext();
  if (affine::getSourceMemRef(loadMemref) != allocResult) return AffineMap();

  SmallVector<Operation *> chain;
  for (Value cur = loadMemref; cur != allocResult;) {
    auto *defOp = cur.getDefiningOp();
    chain.push_back(defOp);
    cur = cast<ViewLikeOpInterface>(defOp).getViewSource();
  }

  unsigned loadRank = cast<MemRefType>(loadMemref.getType()).getRank();
  SmallVector<AffineExpr> exprs;
  for (unsigned i = 0; i < loadRank; i++) exprs.push_back(getAffineDimExpr(i, ctx));

  for (auto *op : chain) {
    if (isa<memref::MemorySpaceCastOp>(op)) continue;

    if (auto sv = dyn_cast<memref::SubViewOp>(op)) {
      auto offsets = sv.getStaticOffsets();
      auto strides = sv.getStaticStrides();
      if (offsets.size() != exprs.size()) return AffineMap();
      for (unsigned i = 0; i < exprs.size(); i++) {
        if (strides[i] != 1) return AffineMap();
        if (offsets[i] != 0) exprs[i] = exprs[i] + offsets[i];
      }
      continue;
    }

    if (auto cs = dyn_cast<memref::CollapseShapeOp>(op)) {
      if (!applyCollapseShapeToExprs(cs, exprs)) return AffineMap();
      continue;
    }

    if (auto es = dyn_cast<memref::ExpandShapeOp>(op)) {
      if (!applyExpandShapeToExprs(es, exprs, ctx)) return AffineMap();
      continue;
    }

    return AffineMap();
  }

  return AffineMap::get(loadRank, 0, exprs, ctx);
}

// Store Tracing
// For each partition, find the store(s) that write to it and their enclosing affine.if.
// Returns false if the pattern is not recognized.
bool SubviewAllocElimPass::tracePartitionStores(Partition &partition) {
  SmallVector<Operation *> storeOps;
  if (!collectStoresFromSubview(partition.subviewOp.getResult(), storeOps)) return false;

  for (auto *op : storeOps) {
    auto affineStore = dyn_cast<affine::AffineStoreOp>(op);
    if (!affineStore) return false;
    partition.stores.push_back(affineStore);

    // Check if the store is inside an affine.if.
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(affineStore->getParentOp())) {
      if (partition.guardIfOp && partition.guardIfOp != ifOp) return false;  // conflicting guards
      partition.guardIfOp = ifOp;
    }
  }
  return !partition.stores.empty();
}

// Find initialization stores (e.g., store 0 to the entire alloc)
InitInfo SubviewAllocElimPass::findAllocInitialization(memref::AllocOp allocOp) {
  InitInfo info;
  for (auto *op : directStores) {
    auto affineStore = dyn_cast<affine::AffineStoreOp>(op);
    if (!affineStore) continue;
    if (affineStore.getMemRef() != allocOp.getResult()) continue;
    Value val = affineStore.getValueToStore();
    if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
      info.hasInit = true;
      info.initValue = val;
      return info;
    }
  }
  return info;
}

Value SubviewAllocElimPass::cloneComputationIntoBlock(OpBuilder &builder, Location loc,
                                                      ArrayRef<Operation *> computeOps, Value finalValue,
                                                      IRMapping &mapping) {
  for (auto *op : computeOps) {
    builder.clone(*op, mapping);
  }
  return mapping.lookupOrDefault(finalValue);
}

Value SubviewAllocElimPass::buildPartitionValue(OpBuilder &builder, Location loc, const Partition &partition,
                                                Operation *loadOp, IRMapping &mapping) {
  if (partition.stores.empty()) return Value();

  // When the partition has a non-zero offset, the original store was executed with
  // the loop IV in [0, partition.size). But in the replacement context, the IV is
  // in [partition.offset, partition.offset + partition.size). We must remap the
  // partition-axis IV to (IV - offset) so cloned ops index correctly.
  if (partition.offset != 0) {
    SmallVector<Value> loadIndices = getLoadIndices(loadOp);
    if (partitionAxis < static_cast<int>(loadIndices.size())) {
      Value origIV = loadIndices[partitionAxis];
      // Create: adjustedIV = origIV - partition.offset
      auto adjustMap = AffineMap::get(1, 0, getAffineDimExpr(0, builder.getContext()) - partition.offset);
      Value adjustedIV = builder.create<affine::AffineApplyOp>(loc, adjustMap, ValueRange{origIV});
      mapping.map(origIV, adjustedIV);
    }
  }

  // Use the first store as the source template.
  auto storeOp = partition.stores[0];
  SourceInfo srcInfo = traceStoreSource(storeOp);

  // Verify external operands dominate the load site before cloning.
  SmallVector<Operation *> opsToCheck;
  if (srcInfo.sourceLoad)
    opsToCheck.push_back(srcInfo.sourceLoad.getOperation());
  else
    opsToCheck.assign(srcInfo.computeOps.begin(), srcInfo.computeOps.end());
  if (!allExternalOperandsDominate(opsToCheck, loadOp)) return Value();

  if (srcInfo.sourceLoad) {
    auto newLoad = builder.clone(*srcInfo.sourceLoad.getOperation(), mapping);
    return newLoad->getResult(0);
  }

  // Complex case: clone the entire computation DAG.
  if (!srcInfo.computeOps.empty()) {
    return cloneComputationIntoBlock(builder, loc, srcInfo.computeOps, srcInfo.finalValue, mapping);
  }

  return Value();
}

// Build partition values for all loads in a group from a single partition.
SmallVector<Value> SubviewAllocElimPass::buildPartitionValues(OpBuilder &b, Location loc, const Partition &part,
                                                              ArrayRef<Operation *> loads) {
  SmallVector<Value> vals;
  for (auto *loadOp : loads) {
    IRMapping mapping;
    Value val = buildPartitionValue(b, loc, part, loadOp, mapping);
    if (!val) return {};
    vals.push_back(val);
  }
  return vals;
}

// Recursively build a nested affine.if/else chain that selects the correct partition
// value(s) based on the partition-axis index. Works for any number of loads (1 or N).
SmallVector<Value> SubviewAllocElimPass::buildNestedIf(OpBuilder &b, Location loc, unsigned partIdx,
                                                       ArrayRef<Operation *> loads, ArrayRef<Type> resultTypes,
                                                       Value partAxisIdx, bool isFullCoverage) {
  unsigned numLoads = loads.size();

  // Base case: past all partitions — return init values or fail.
  if (partIdx >= partitions.size()) {
    if (initInfo.hasInit) return SmallVector<Value>(numLoads, initInfo.initValue);
    return {};
  }

  const Partition &part = partitions[partIdx];
  bool isLastPartition = (partIdx == partitions.size() - 1);

  // Fast path: last partition + full coverage → no if needed, directly emit values.
  if (isLastPartition && isFullCoverage) return buildPartitionValues(b, loc, part, loads);

  // Build condition: partAxisIdx <= partition.offset + partition.size - 1
  int64_t boundary = part.offset + part.size - 1;
  auto d0 = getAffineDimExpr(0, b.getContext());
  IntegerSet condSet = IntegerSet::get(1, 0, {-d0 + boundary}, {false});
  bool hasElse = (!isLastPartition || !isFullCoverage);
  auto ifOp = b.create<affine::AffineIfOp>(loc, resultTypes, condSet, ValueRange{partAxisIdx}, hasElse);

  // Then block: values from the current partition.
  OpBuilder thenBuilder = OpBuilder::atBlockBegin(ifOp.getThenBlock());
  SmallVector<Value> thenVals = buildPartitionValues(thenBuilder, loc, part, loads);
  if (thenVals.empty()) return {};
  thenBuilder.create<affine::AffineYieldOp>(loc, thenVals);

  // Else block: next partition or init value.
  if (hasElse) {
    OpBuilder elseBuilder = OpBuilder::atBlockBegin(ifOp.getElseBlock());
    SmallVector<Value> elseVals;
    if (isLastPartition && !isFullCoverage) {
      if (!initInfo.hasInit) return {};
      elseVals.assign(numLoads, initInfo.initValue);
    } else {
      elseVals = buildNestedIf(elseBuilder, loc, partIdx + 1, loads, resultTypes, partAxisIdx, isFullCoverage);
      if (elseVals.empty()) return {};
    }
    elseBuilder.create<affine::AffineYieldOp>(loc, elseVals);
  }

  SmallVector<Value> results;
  for (unsigned i = 0; i < numLoads; i++) results.push_back(ifOp.getResult(i));
  return results;
}

bool SubviewAllocElimPass::replaceLoads(memref::AllocOp allocOp) {
  if (directLoads.empty()) return false;

  // Compute whether partitions fully cover the partition axis.
  auto allocType = allocOp.getType();
  int64_t axisDim = allocType.getDimSize(partitionAxis);
  int64_t covered = std::accumulate(partitions.begin(), partitions.end(), int64_t{0},
                                    [](int64_t sum, const auto &p) { return sum + p.size; });
  bool isFullCoverage = (covered == axisDim);

  // Group loads by (partition-axis index, block) so loads sharing the same
  // index in the same block can be combined into a single multi-result affine.if.
  struct LoadGroup {
    Value partAxisIdx;
    Block *block;
    SmallVector<Operation *> loads;
  };

  SmallVector<LoadGroup> groups;
  for (auto *loadOp : directLoads) {
    SmallVector<Value> indices = getLoadIndices(loadOp);
    if (static_cast<int>(indices.size()) <= partitionAxis) continue;
    Value partIdx = indices[partitionAxis];
    Block *block = loadOp->getBlock();

    auto it = llvm::find_if(groups, [&](const LoadGroup &g) { return g.partAxisIdx == partIdx && g.block == block; });
    if (it != groups.end()) {
      it->loads.push_back(loadOp);
    } else {
      groups.push_back({partIdx, block, {loadOp}});
    }
  }

  // Replace each group's loads with a nested affine.if/else chain.
  SmallVector<Operation *> toErase;
  for (auto &group : groups) {
    Operation *firstLoad = group.loads[0];
    for (auto *l : group.loads)
      if (l->isBeforeInBlock(firstLoad)) firstLoad = l;

    OpBuilder builder(firstLoad);
    auto loc = firstLoad->getLoc();

    SmallVector<Type> resultTypes;
    llvm::transform(group.loads, std::back_inserter(resultTypes),
                    [this](Operation *l) { return getLoadResult(l).getType(); });

    SmallVector<Value> replacements =
      buildNestedIf(builder, loc, 0, group.loads, resultTypes, group.partAxisIdx, isFullCoverage);
    if (replacements.size() != group.loads.size()) continue;

    for (unsigned i = 0; i < group.loads.size(); i++) {
      getLoadResult(group.loads[i]).replaceAllUsesWith(replacements[i]);
      toErase.push_back(group.loads[i]);
    }
  }

  for (auto *op : toErase) {
    if (op->use_empty()) op->erase();
  }
  return !toErase.empty();
}

// Forcibly erase store ops and clean up their stored values' defining ops if dead.
static void eraseStoresWithDefs(ArrayRef<Operation *> allStores, DenseSet<Operation *> &erased) {
  for (auto *op : allStores) {
    if (erased.count(op)) continue;
    Value storedVal;
    if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) storedVal = affineStore.getValueToStore();
    erased.insert(op);
    op->erase();
    if (storedVal) {
      if (auto *defOp = storedVal.getDefiningOp()) {
        if (!erased.count(defOp) && isOpTriviallyDead(defOp)) {
          erased.insert(defOp);
          defOp->erase();
        }
      }
    }
  }
}

// Clean up guard affine.if blocks after their stores have been removed.
static void cleanupGuardIfs(const SetVector<affine::AffineIfOp> &guardIfs, DenseSet<Operation *> &erased) {
  for (auto ifOp : guardIfs) {
    if (erased.count(ifOp.getOperation())) continue;
    for (auto &region : ifOp->getRegions()) {
      if (region.empty()) continue;
      auto &block = region.front();
      SmallVector<Operation *> deadOps;
      for (auto it = block.rbegin(); it != block.rend(); ++it) {
        if (isa<affine::AffineYieldOp>(&*it)) continue;
        if (isOpTriviallyDead(&*it) && !erased.count(&*it)) deadOps.push_back(&*it);
      }
      for (auto *op : deadOps) {
        erased.insert(op);
        op->erase();
      }
    }
    if (isOpTriviallyDead(ifOp)) {
      ifOp->walk([&](Operation *inner) { erased.insert(inner); });
      erased.insert(ifOp.getOperation());
      ifOp->erase();
    }
  }
}

// Remove dead stores, view chain ops, and the alloc after load replacement.
// Strategy: first forcibly erase all stores (which have Write side-effects and
// cannot be detected by isOpTriviallyDead), then use isOpTriviallyDead to
// clean up the remaining side-effect-free ops (view chain, computations, alloc).
void SubviewAllocElimPass::eraseDeadOps(memref::AllocOp allocOp) {
  DenseSet<Operation *> erased;

  // Step 1: Collect all stores and their enclosing guard ifs.
  SmallVector<Operation *> allStores(directStores);
  SetVector<affine::AffineIfOp> guardIfs;
  for (auto &part : partitions) {
    llvm::transform(part.stores, std::back_inserter(allStores), [](auto storeOp) { return storeOp.getOperation(); });
    if (part.guardIfOp) guardIfs.insert(part.guardIfOp);
  }
  for (auto *op : directStores) {
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(op->getParentOp())) guardIfs.insert(ifOp);
  }

  // Step 2: Forcibly erase all stores (Write effect prevents isOpTriviallyDead).
  eraseStoresWithDefs(allStores, erased);

  // Step 3: Clean up guard affine.if blocks (stores removed, remaining ops are pure).
  cleanupGuardIfs(guardIfs, erased);

  // Step 4: Remove view chain ops in reverse order, then subviews.
  for (auto it = viewOps.rbegin(); it != viewOps.rend(); ++it) {
    if (!erased.count(*it) && isOpTriviallyDead(*it)) {
      erased.insert(*it);
      (*it)->erase();
    }
  }
  for (auto svOp : subviewOps) {
    if (!erased.count(svOp.getOperation()) && isOpTriviallyDead(svOp)) {
      erased.insert(svOp.getOperation());
      svOp->erase();
    }
  }

  // Step 5: Remove the alloc itself.
  if (isOpTriviallyDead(allocOp)) allocOp->erase();
}

// Handle allocs with no subview partitions: forward store values directly to loads.
// Pattern: alloc is written by stores inside affine.if guards, and read by loads
// elsewhere. Replace each load by cloning the computation from the store source.
bool SubviewAllocElimPass::processSimpleAlloc(memref::AllocOp allocOp) {
  if (!subviewOps.empty() || directStores.empty() || directLoads.empty()) return false;

  // Skip trivial store-load pairs where all stores and loads are in the same block.
  // These are simple local temporaries (e.g., reduction results stored then immediately
  // loaded). Cloning their source computation (which may include entire loops) is
  // wasteful and produces redundant code. Let mem2reg handle these instead.
  bool allSameBlock = llvm::all_of(directStores, [&](Operation *store) {
    return llvm::all_of(directLoads, [&](Operation *load) { return store->getBlock() == load->getBlock(); });
  });
  if (allSameBlock) return false;

  // All stores must be affine.store with traceable sources.
  SmallVector<affine::AffineStoreOp> stores;
  for (auto *op : directStores) {
    auto affineStore = dyn_cast<affine::AffineStoreOp>(op);
    if (!affineStore) return false;
    stores.push_back(affineStore);
  }

  // All stores must write the same SSA value; otherwise folding would drop other stores' computation.
  Value templateVal = stores[0].getValueToStore();
  auto storeDiffersFromTemplate = [&](affine::AffineStoreOp s) { return s.getValueToStore() != templateVal; };
  if (llvm::any_of(stores, storeDiffersFromTemplate)) return false;

  // Use the first store as source template (all stores write the same computation).
  SourceInfo srcInfo = traceStoreSource(stores[0]);
  if (!srcInfo.sourceLoad && srcInfo.computeOps.empty()) return false;

  // Verify all external operands of the source computation dominate every load site.
  // If any load is unreachable, we cannot safely erase the stores, so bail out entirely.
  SmallVector<Operation *> opsToCheck;
  if (srcInfo.sourceLoad)
    opsToCheck.push_back(srcInfo.sourceLoad.getOperation());
  else
    opsToCheck = srcInfo.computeOps;
  if (llvm::any_of(directLoads, [&](Operation *loadOp) { return !allExternalOperandsDominate(opsToCheck, loadOp); })) {
    return false;
  }

  // Replace each load with a clone of the store source computation.
  SmallVector<Operation *> toErase;
  for (auto *loadOp : directLoads) {
    Value loadResult = getLoadResult(loadOp);
    if (!loadResult) continue;

    OpBuilder builder(loadOp);
    IRMapping mapping;
    Value replacement;
    if (srcInfo.sourceLoad) {
      auto *newLoad = builder.clone(*srcInfo.sourceLoad.getOperation(), mapping);
      replacement = newLoad->getResult(0);
    } else {
      replacement =
        cloneComputationIntoBlock(builder, loadOp->getLoc(), srcInfo.computeOps, srcInfo.finalValue, mapping);
    }
    if (!replacement) continue;

    loadResult.replaceAllUsesWith(replacement);
    toErase.push_back(loadOp);
  }

  for (auto *op : toErase) {
    if (op->use_empty()) op->erase();
  }
  if (toErase.empty()) return false;

  eraseDeadOps(allocOp);
  return true;
}

// Validate that all stores write directly to the alloc and all loads go
// through the view chain with identity maps.
static bool validateTransposeStoresAndLoads(ArrayRef<Operation *> stores, ArrayRef<Operation *> loads,
                                            Value allocResult) {
  for (auto *op : stores) {
    auto storeOp = dyn_cast<affine::AffineStoreOp>(op);
    if (!storeOp || storeOp.getMemRef() != allocResult) return false;
  }
  for (auto *op : loads) {
    auto loadOp = dyn_cast<affine::AffineLoadOp>(op);
    if (!loadOp || loadOp.getMemRef() == allocResult) return false;
    if (!loadOp.getAffineMap().isIdentity()) return false;
  }
  return true;
}

// Handle allocs that serve as transpose buffers: written with transposed indices,
// read only through a view chain (collapse/expand/subview). Replace view-chain loads
// with direct loads from the source memref using a composed AffineMap.
bool SubviewAllocElimPass::processTransposeAlloc(memref::AllocOp allocOp) {
  if (subviewOps.empty() || directStores.empty() || directLoads.empty()) return false;
  if (!validateTransposeStoresAndLoads(directStores, directLoads, allocOp.getResult())) return false;

  auto transposeInfo = detectTransposePattern(directStores, allocOp.getResult());
  if (!transposeInfo) return false;

  unsigned allocRank = allocOp.getType().getRank();
  auto *ctx = &getContext();

  // Replace each view-chain load with a direct load from the source.
  SmallVector<Operation *> toErase;
  for (auto *op : directLoads) {
    auto affineLoad = cast<affine::AffineLoadOp>(op);

    AffineMap viewMap = buildViewChainToAllocMap(affineLoad.getMemRef(), allocOp.getResult());
    if (!viewMap || viewMap.getNumResults() != allocRank) return false;

    // Compose: source[k] = alloc[inversePerm[k]]
    SmallVector<AffineExpr> sourceExprs(allocRank);
    for (unsigned k = 0; k < allocRank; k++) sourceExprs[k] = viewMap.getResult(transposeInfo->inversePerm[k]);
    AffineMap sourceMap = AffineMap::get(viewMap.getNumDims(), 0, sourceExprs, ctx);

    if (!valueDominates(transposeInfo->sourceMemref, op)) return false;

    OpBuilder builder(op);
    auto newLoad = builder.create<affine::AffineLoadOp>(op->getLoc(), transposeInfo->sourceMemref, sourceMap,
                                                        affineLoad.getIndices());
    affineLoad.getResult().replaceAllUsesWith(newLoad.getResult());
    toErase.push_back(op);
  }

  if (toErase.empty()) return false;
  for (auto *op : toErase)
    if (op->use_empty()) op->erase();

  // Reuse eraseDeadOps: it erases directStores, view chain ops, subviews, and the alloc.
  eraseDeadOps(allocOp);
  return true;
}

bool SubviewAllocElimPass::processAlloc(memref::AllocOp allocOp) {
  // Skip if the alloc (or any alias) is returned.
  if (isAllocReturned(allocOp)) return false;

  // Collect all users of the alloc through view chains.
  subviewOps.clear();
  directLoads.clear();
  directStores.clear();
  viewOps.clear();
  if (!collectUsersRecursive(allocOp.getResult())) return false;

  // Analyze subview partitions.
  partitionAxis = -1;
  partitions.clear();
  if (!analyzeAxisPartition(allocOp)) {
    // No subview partitions — try transpose buffer elimination, then simple forwarding.
    if (processTransposeAlloc(allocOp)) return true;
    return processSimpleAlloc(allocOp);
  }

  // Trace store sources for each partition.
  for (auto &part : partitions) {
    if (!tracePartitionStores(part)) return false;
  }

  // Find initialization info (e.g., store 0 to alloc).
  initInfo = findAllocInitialization(allocOp);

  // Replace all direct loads from the alloc.
  if (!replaceLoads(allocOp)) return false;

  // Clean up dead ops left behind after load replacement.
  eraseDeadOps(allocOp);
  return true;
}

void SubviewAllocElimPass::runOnOperation() {
  // Early exit if no subview ops exist in the operation.
  auto walkResult = getOperation()->walk([](memref::SubViewOp) { return WalkResult::interrupt(); });
  if (!walkResult.wasInterrupted()) return;

  // Only collect non-returned allocs that have subview users (potential candidates).
  // This avoids repeatedly processing returned allocs and trivial temporaries.
  SmallVector<memref::AllocOp> allocOps;
  getOperation()->walk([&](memref::AllocOp op) {
    if (!isAllocReturned(op)) allocOps.push_back(op);
  });
  if (allocOps.empty()) return;

  for (auto allocOp : allocOps) processAlloc(allocOp);
}

}  // end anonymous namespace

std::unique_ptr<Pass> createSubviewAllocElimPass() { return std::make_unique<SubviewAllocElimPass>(); }
}  // namespace mlir
