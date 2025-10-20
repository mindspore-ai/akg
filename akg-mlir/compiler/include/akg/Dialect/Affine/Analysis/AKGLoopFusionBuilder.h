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

#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace mlir {
namespace akg {

// Forward declarations
enum LoopTransform {
  Replicate,
  Permute,
  StripMine,
  Collapse,
  BackTracking
};

// Group of nodes that can be fused together
struct Group {
public:
  Group(unsigned groupId, unsigned rootId, affine::AffineForOp root) 
  : groupId(groupId), rootId(rootId), root(root) {}

  unsigned groupId;
  unsigned rootId;
  affine::AffineForOp root;
  std::vector<unsigned> nodesId;

  OperatorTemplate groupTemplate{OperatorTemplate::Default};
  bool isGlobalOut{false};
  std::vector<unsigned> fusedGroupId;
  std::unordered_map<unsigned, std::vector<LoopTransform>> nodeTransformRecords;

  affine::AffineForOp getLeadingFor() const { return root; }
  void addNode(unsigned nodeId) { nodesId.push_back(nodeId); }
  void setIsGlobalOut(bool isOut) { isGlobalOut = isOut; }
  void dump() const { print(llvm::errs()); }
  void print(llvm::raw_ostream &os) const;
  std::string getGroupTemplateString() const;

private:
  std::unordered_map<int, std::string> loopTransformToStr{
    {static_cast<int>(LoopTransform::Replicate), "Replicate"},
    {static_cast<int>(LoopTransform::Permute), "Permute"},
    {static_cast<int>(LoopTransform::StripMine), "StripMine"},
    {static_cast<int>(LoopTransform::Collapse), "Collapse"},
    {static_cast<int>(LoopTransform::BackTracking), "BackTracking"}
  };
};
using GroupPtr = std::shared_ptr<Group>;

// Loop nest state collector for analyzing loop structures
struct LoopNestStateCollector {
  llvm::SmallVector<affine::AffineForOp, 4> forOps;
  llvm::SmallVector<Operation *, 4> loadOpInsts;
  llvm::SmallVector<Operation *, 4> storeOpInsts;
  llvm::SmallVector<Operation *, 4> otherInsts;
  bool hasNonAffineRegionOp = false;

  void collect(Operation *opToWalk);
};

// Extended MemRef dependence graph for fusion analysis
// Inherits from MemRefDependenceGraph and adds fusion-specific functionality
struct MemRefDependenceGraphForFusion : public MemRefDependenceGraph {
public:
  explicit MemRefDependenceGraphForFusion(Block *block) : MemRefDependenceGraph(block, false) {}

  GroupPtr getGroup(unsigned groupId);
  GroupPtr getGroupByNode(unsigned nodeId);
  std::unordered_set<GroupPtr> getGroupsByNode(llvm::DenseSet<unsigned> nodeIds);
  bool init();
  void print(llvm::raw_ostream &os) const override;
  void dump() const override { print(llvm::errs()); }
  void createInitNode(llvm::DenseMap<Value, llvm::SetVector<unsigned>> &memrefAccesses);
  OperatorTemplate getGroupType(const std::vector<unsigned> &nodes);
  bool elementwiseMatch(Operation *op);
  int getMemrefSourceOfNode(unsigned id);
  bool isGlobalMemref(unsigned id);


  std::unordered_map<unsigned, GroupPtr> groups;
  std::unordered_map<unsigned, GroupPtr> nodeToGroup;
  unsigned nextGroupId = 0;
};

// Helper class for fusion code generation
struct FusionCodeGenHelper {
public:
  explicit FusionCodeGenHelper(MemRefDependenceGraphForFusion &mdg) : mdg(mdg) {}

  unsigned getAliasId(unsigned srcId);
  // Perform vertical fusion (V-fusion)
  void doVFuse(unsigned srcId, unsigned dstId, affine::AffineForOp sibAffineForOp,
               affine::AffineForOp dstAffineForOp, unsigned maxLegalFusionDepth, unsigned dstLoopDepthTest);

  // Perform horizontal fusion (H-fusion)
  void doHFuse(unsigned srcId, unsigned dstId, affine::AffineForOp srcAffineForOp,
               affine::AffineForOp dstAffineForOp);

  MemRefDependenceGraphForFusion &mdg;
  std::unordered_map<unsigned, unsigned> nodeAlias;
};

}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AKGLOOPFUSIONBUILDER_H_
