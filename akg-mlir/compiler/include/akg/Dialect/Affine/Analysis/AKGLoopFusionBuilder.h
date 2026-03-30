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
  bool isDependencyInGraph(unsigned fromGroupId, unsigned toGroupId);
  std::vector<unsigned> getDependentGroups(unsigned groupId);

  // Precomputation for dependency analysis
  void precomputeDependentGroups();

  // Debug and print
  void print(llvm::raw_ostream &os) const override;
  void dump() const override { print(llvm::errs()); }

  std::unordered_map<unsigned, GroupPtr> groups;
  std::unordered_map<unsigned, GroupPtr> nodeToGroup;
  unsigned nextGroupId = 0;

  // Cache for dependent groups (precomputed once, used many times)
  std::unordered_map<unsigned, std::vector<unsigned>> dependentGroupsCache;

  OperatorTemplate funcOperatorType = OperatorTemplate::Default;

 private:
  void collectLoadNodeIdsAndNonForNodes(Value memref, const llvm::SetVector<unsigned> &nodeIds,
                                        llvm::SmallVector<unsigned> &loadNodeIds,
                                        llvm::SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore);
  void addAliasedStoreEdges(Value memref, const llvm::SetVector<unsigned> &nodeIds,
                            const llvm::SmallVector<std::pair<unsigned, bool>, 16> &nonForNodesWithStore);
  void addMultipleLoadEdges(Value memref, const llvm::SetVector<unsigned> &nodeIds,
                            llvm::SmallVector<unsigned> &loadNodeIds);
};

/// Information about an AffineForOp loop nest: root loop, full nest from
/// collectLoopNest, perfectly-nested band depth, and whether the nest is perfectly nested.
struct FusionLoopNestInfo {
  affine::AffineForOp root;
  llvm::SmallVector<affine::AffineForOp, 4> loops;
  unsigned perfectDepth = 0;
  unsigned loopDepth = UINT_MAX;
  bool isPerfect = false;

  void collect(affine::AffineForOp rootOp);
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
  void doIFuse(unsigned srcGroupId, unsigned dstGroupId, FusionLoopNestInfo &srcInfo, FusionLoopNestInfo &dstInfo);

 private:
  // Finds the maximum legal fusion depth for fusing src loop nest into dst loop nest.
  unsigned findMaxLegalFusionDepth(const FusionLoopNestInfo &srcInfo, const FusionLoopNestInfo &dstInfo,
                                   const affine::FusionStrategy &strategy,
                                   llvm::SmallVector<affine::ComputationSliceState, 8> &depthSliceUnions,
                                   const FusionPlan &plan, bool srcDstReversed = false);

  void buildStrategyOpsA(const affine::FusionStrategy &strategy, llvm::ArrayRef<Operation *> loadAndStoreOpsA,
                         llvm::SmallVector<Operation *, 4> &strategyOpsA);

  // Records alias, erases the fused-away loop, and updates plan.depInfo (refreshes
  // alias target node's loads/stores and replaces erased node refs in depInfo).
  void eraseLoopAndCleanupNode(unsigned erasedNodeId, unsigned aliasTargetId, affine::AffineForOp loopToErase);

  MemRefDependenceGraphForFusion &mdg;
  std::unordered_map<unsigned, unsigned> nodeAlias;
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONBUILDER_H_
