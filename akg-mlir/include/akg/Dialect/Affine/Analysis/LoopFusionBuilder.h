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

#ifndef AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONBUILDER_H_
#define AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONBUILDER_H_

#include <climits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Dialect/Affine/Analysis/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "akg/Utils/Constants.h"

namespace mlir {
namespace akg {

struct LoopNestStateCollector {
  void collect(Operation *opToWalk);

  llvm::SmallVector<affine::AffineForOp, kSmallVectorSizeFour> forOps;
  llvm::SmallVector<Operation *, kSmallVectorSizeFour> loadOpInsts;
  llvm::SmallVector<Operation *, kSmallVectorSizeFour> storeOpInsts;
  llvm::SmallVector<Operation *, kSmallVectorSizeFour> otherInsts;
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
  void collectLoadNodeIdsAndNonForNodes(
    Value memref, const llvm::SetVector<unsigned> &nodeIds, llvm::SmallVector<unsigned> &loadNodeIds,
    llvm::SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeSixteen> &nonForNodesWithStore,
    llvm::SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeEight> &forNodesWithStore);
  void addAliasedStoreEdges(
    Value memref, const llvm::SetVector<unsigned> &nodeIds,
    const llvm::SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeSixteen> &nonForNodesWithStore,
    const llvm::SmallVector<std::pair<unsigned, bool>, kSmallVectorSizeEight> &forNodesWithStore,
    bool hasLoadsForBaseMemref);
  void addMultipleLoadEdges(Value memref, const llvm::SetVector<unsigned> &nodeIds,
                            llvm::SmallVector<unsigned> &loadNodeIds);
  void collectForNodeLoadIds(Operation *loadOpInst, Value memref, llvm::SmallVector<unsigned> &loadNodeIds);
  void collectDepGroupsForNode(unsigned nodeId, unsigned groupId, std::unordered_set<unsigned> &depGroups);
};

struct FusionLoopNestInfo {
  affine::AffineForOp root;
  llvm::SmallVector<affine::AffineForOp, kSmallVectorSizeFour> loops;
  unsigned perfectDepth = 0;
  unsigned loopDepth = UINT_MAX;
  bool isPerfect = false;

  void collect(affine::AffineForOp rootOp);
};

// Holds collected load/store accesses and memref flags for both sides of a fusion.
// Always stores data in true src/dst order; callers handle reversal when needed.
struct FusionAccessInfo {
  DenseMap<Value, bool> srcMemFlags;
  llvm::SmallVector<Operation *, kSmallVectorSizeFour> srcAccesses;
  DenseMap<Value, bool> dstMemFlags;
  llvm::SmallVector<Operation *, kSmallVectorSizeFour> dstAccesses;
};

struct FusionCodeGenHelper {
 public:
  explicit FusionCodeGenHelper(MemRefDependenceGraphForFusion &mdg) : mdg(mdg) {}

  // Alias management
  unsigned getAliasId(unsigned srcId);

  // Fusion operation: perform different types of loop fusion
  void doFuse(unsigned srcGroupId, unsigned dstGroupId, affine::AffineForOp srcAffineForOp,
              affine::AffineForOp dstAffineForOp, const FusionPlan &plan);
  void doIFuse(unsigned srcGroupId, unsigned dstGroupId, FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo,
               const FusionPlan &plan);

 private:
  // Finds the maximum legal fusion depth for fusing src loop nest into dst loop nest.
  // accessInfo: pre-collected accesses in true src/dst order; srcDstReversed swaps references internally.
  unsigned findMaxLegalFusionDepth(
    const FusionLoopNestInfo &srcInfo, const FusionLoopNestInfo &dstInfo, const affine::FusionStrategy &strategy,
    llvm::SmallVector<affine::ComputationSliceState, kSmallVectorSizeEight> &depthSliceUnions,
    const FusionPlan &plan, const DependenceInfo &depInfo, FusionAccessInfo &accessInfo,
    bool srcDstReversed = false);

  // Selects the most appropriate DependenceInfo from plan.depInfo for the fusion:
  // the one whose memref appears latest (last access) in srcAffineForOp, so the fused
  // slice is placed closest to the producer of the dependency.
  const DependenceInfo *selectDepInfo(const FusionPlan &plan,
                                       affine::AffineForOp srcAffineForOp) const;

  void buildStrategyOpsA(const affine::FusionStrategy &strategy, llvm::ArrayRef<Operation *> loadAndStoreOpsA,
                         llvm::SmallVector<Operation *, kSmallVectorSizeFour> &strategyOpsA);

  void applySliceFusion(affine::AffineForOp srcAffineForOp, affine::AffineForOp dstAffineForOp,
                        FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo, bool keepSrcDst,
                        affine::ComputationSliceState &bestSlice, const FusionAccessInfo &accessInfo,
                        unsigned srcGroupId, unsigned dstGroupId);

  // Records alias, erases the fused-away loop, and updates plan.depInfo (refreshes
  // alias target node's loads/stores and replaces erased node refs in depInfo).
  void eraseLoopAndCleanupNode(unsigned erasedNodeId, unsigned aliasTargetId, affine::AffineForOp loopToErase);

  MemRefDependenceGraphForFusion &mdg;
  std::unordered_map<unsigned, unsigned> nodeAlias;
};

}  // namespace akg
}  // namespace mlir

#endif  // AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONBUILDER_H_
