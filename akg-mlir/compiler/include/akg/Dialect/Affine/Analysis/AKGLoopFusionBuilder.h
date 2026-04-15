/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONBUILDER_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONBUILDER_H_

#include <climits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Dialect/Affine/Analysis/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"

namespace mlir {
namespace akg {

struct LoopNestStateCollector {
  void collect(Operation *opToWalk);

  llvm::SmallVector<affine::AffineForOp, 4> forOps;
  llvm::SmallVector<Operation *, 4> loadOpInsts;
  llvm::SmallVector<Operation *, 4> storeOpInsts;
  llvm::SmallVector<Operation *, 4> otherInsts;
  bool hasNonAffineRegionOp = false;
};

struct MemRefDependenceGraphForFusion : public MemRefDependenceGraph {
 public:
  explicit MemRefDependenceGraphForFusion(Block *block) : MemRefDependenceGraph(block, false) {}

  // Group management
  GroupPtr getGroup(unsigned groupId);
  GroupPtr getGroupByNode(unsigned nodeId);

  // Graph initialization
  bool init();
  void createInitNode(llvm::DenseMap<Value, llvm::SetVector<unsigned>> &memrefAccesses);
  bool createEdges(const llvm::DenseMap<Value, llvm::SetVector<unsigned>> &memrefAccesses);

  // Group type analysis
  OperatorTemplate getGroupType(const std::vector<unsigned> &nodes);
  int getMemrefSourceOfNode(unsigned id);

  // Dependency analysis
  std::unordered_set<unsigned> getDependentGroups(unsigned groupId);

  // Precomputation for dependency analysis
  void precomputeDependentGroups();

  // Debug and print
  void print(llvm::raw_ostream &os) const override;
  void dump() const override { print(llvm::errs()); }

  std::unordered_map<unsigned, GroupPtr> groups;
  std::unordered_map<unsigned, GroupPtr> nodeToGroup;
  unsigned nextGroupId = 0;

  // Cache for dependent groups (precomputed once, used many times)
  std::unordered_map<unsigned, std::unordered_set<unsigned>> dependentGroupsCache;

  OperatorTemplate funcOperatorType = OperatorTemplate::Default;

 private:
  bool areNonOverlappingSubviewStores(unsigned nodeIdA, unsigned nodeIdB);
  void collectLoadNodeIdsAndNonForNodes(Value memref, const llvm::SetVector<unsigned> &nodeIds,
                                        llvm::SmallVector<unsigned> &loadNodeIds,
                                        llvm::SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore,
                                        llvm::SmallVector<std::pair<unsigned, bool>, 8> &forNodesWithStore);
  void addAliasedStoreEdges(Value memref, const llvm::SetVector<unsigned> &nodeIds,
                            const llvm::SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore,
                            const llvm::SmallVector<std::pair<unsigned, bool>, 8> &forNodesWithStore,
                            bool hasLoadsForBaseMemref);
  void addMultipleLoadEdges(Value memref, const llvm::SetVector<unsigned> &nodeIds,
                            llvm::SmallVector<unsigned> &loadNodeIds);
};

struct FusionLoopNestInfo {
  affine::AffineForOp root;
  llvm::SmallVector<affine::AffineForOp, 4> loops;
  unsigned perfectDepth = 0;
  unsigned loopDepth = UINT_MAX;
  bool isPerfect = false;

  void collect(affine::AffineForOp rootOp);
};

// Holds collected load/store accesses and memref flags for both sides of a fusion.
// Always stores data in true src/dst order; callers handle reversal when needed.
struct FusionAccessInfo {
  DenseMap<Value, bool> srcMemFlags;
  llvm::SmallVector<Operation *, 4> srcAccesses;
  DenseMap<Value, bool> dstMemFlags;
  llvm::SmallVector<Operation *, 4> dstAccesses;
};

// Guard condition applied to cloned operations during subview fusion.
struct FusionGuard {
  // ExtraDimEqLB: secondary has fewer loops;
  // cloned ops execute only when the extra dimension's IV equals its lower bound (boundValue - IV >= 0).

  // SmallerUB: same loop count, one dimension has a smaller upper bound;
  // cloned ops execute only when IV < boundValue (boundValue-1-IV >= 0).
  enum Kind { None, ExtraDimEqLB, SmallerUB };
  Kind kind = None;
  // Which primary loop dimension to guard on.
  unsigned dimPos = 0;
  // LB for ExtraDimEqLB, UB for SmallerUB.
  int64_t boundValue = 0;

  bool isNeeded() const { return kind != None; }

  // Builds an IntegerSet encoding the guard condition over a single dimension.
  IntegerSet buildCondSet(MLIRContext *ctx) const;
};

struct SubviewFusionPlan {
  FusionLoopNestInfo *srcInfo;
  FusionLoopNestInfo *dstInfo;
  // true when the src side provides the loop structure (kept)
  bool srcIsPrimary;
  // dimMap[i] maps secondary loop i → primary loop dimMap[i] for IV alignment
  llvm::SmallVector<int, 4> dimMap;
  // optional condition wrapping the cloned operations
  FusionGuard guard;

  FusionLoopNestInfo *primaryInfo() const { return srcIsPrimary ? srcInfo : dstInfo; }
  FusionLoopNestInfo *secondaryInfo() const { return srcIsPrimary ? dstInfo : srcInfo; }
};

using CloneStage = llvm::SmallVector<Operation *, 8>;
struct SubviewFusionHelper {
  // Attempts subview fusion between src and dst loop nests.
  // srcRank/dstRank: memref ranks of the dependency accesses (-1 → falls back to loop depth).
  // Returns the fusion plan if fusion was performed, nullopt otherwise.
  std::optional<SubviewFusionPlan> tryFuse(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo, int srcRank,
                                           int dstRank);

 private:
  // Builds a fusion plan into this->plan. Returns true on success.
  // Dispatches to buildExtraDimPlan or buildSameRankPlan.
  bool buildSubviewFusionPlan(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo, int srcRank, int dstRank);

  // Handles ranks differing by exactly one (one nest has an extra loop dimension).
  bool buildExtraDimPlan(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo, int srcRank, int dstRank);

  // Handles same-rank case: all bounds match (trivial) or exactly one UB differs.
  bool buildSameRankPlan(FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo);

  // Collects clone stages from the secondary side's loop body.
  // Perfect nest → 1 stage; imperfect nest → one stage per level from perfectDepth-1.
  llvm::SmallVector<CloneStage> collectCloneStages(const FusionLoopNestInfo &info);

  // Emits all clone stages into the primary side's innermost body with IV mapping and optional guard.
  void emitCloneStages(const llvm::SmallVector<CloneStage> &stages, IRMapping &mapper);

  SubviewFusionPlan plan;
};

struct FusionCodeGenHelper {
 public:
  explicit FusionCodeGenHelper(MemRefDependenceGraphForFusion &mdg) : mdg(mdg) {}

  // Alias management
  unsigned getAliasId(unsigned srcId);

  // Fusion operation: perform different types of loop fusion
  void doVFuse(unsigned srcGroupId, unsigned dstGroupId, affine::AffineForOp srcAffineForOp,
               affine::AffineForOp dstAffineForOp, const FusionPlan &plan);
  void doHFuse(unsigned srcGroupId, unsigned dstGroupId, affine::AffineForOp srcAffineForOp,
               affine::AffineForOp dstAffineForOp, const FusionPlan &plan);
  void doIFuse(unsigned srcGroupId, unsigned dstGroupId, FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo,
               const FusionPlan &plan);

 private:
  // Finds the maximum legal fusion depth for fusing src loop nest into dst loop nest.
  // accessInfo: pre-collected accesses in true src/dst order; srcDstReversed swaps references internally.
  unsigned findMaxLegalFusionDepth(const FusionLoopNestInfo &srcInfo, const FusionLoopNestInfo &dstInfo,
                                   const affine::FusionStrategy &strategy,
                                   llvm::SmallVector<affine::ComputationSliceState, 8> &depthSliceUnions,
                                   const FusionPlan &plan, FusionAccessInfo &accessInfo, bool srcDstReversed = false);

  void buildStrategyOpsA(const affine::FusionStrategy &strategy, llvm::ArrayRef<Operation *> loadAndStoreOpsA,
                         llvm::SmallVector<Operation *, 4> &strategyOpsA);

  // Records alias, erases the fused-away loop, and updates plan.depInfo (refreshes
  // alias target node's loads/stores and replaces erased node refs in depInfo).
  void eraseLoopAndCleanupNode(unsigned erasedNodeId, unsigned aliasTargetId, affine::AffineForOp loopToErase);

  MemRefDependenceGraphForFusion &mdg;
  std::unordered_map<unsigned, unsigned> nodeAlias;
  SubviewFusionHelper subviewHelper;
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONBUILDER_H_
