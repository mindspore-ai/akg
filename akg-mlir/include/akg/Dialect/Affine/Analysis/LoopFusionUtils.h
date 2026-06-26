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

#ifndef AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONUTILS_H_
#define AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONUTILS_H_

#include <climits>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
namespace akg {

// Loop transformation types for fusion planning.
enum class LoopTransform { Merge, Replicate, ReplicateIf, Permute, StripMine, Collapse, BackTracking };
inline const char *loopTransformToString(LoopTransform type) {
  switch (type) {
    case LoopTransform::Merge:
      return "Merge";
    case LoopTransform::Replicate:
      return "Replicate";
    case LoopTransform::ReplicateIf:
      return "ReplicateIf";
    case LoopTransform::Permute:
      return "Permute";
    case LoopTransform::StripMine:
      return "StripMine";
    case LoopTransform::Collapse:
      return "Collapse";
    case LoopTransform::BackTracking:
      return "BackTracking";
  }
  return "UNKNOWN";
}

// Dependency type between two memory access operations.
enum class DepType { RAW, WAR, WAW, RAR, OTHER };
inline const char *depTypeToString(DepType type) {
  switch (type) {
    case DepType::RAW:
      return "RAW";
    case DepType::WAR:
      return "WAR";
    case DepType::WAW:
      return "WAW";
    case DepType::RAR:
      return "RAR";
    case DepType::OTHER:
      return "OTHER";
  }
  return "UNKNOWN";
}

// Memory reference kind classification.
enum class MemrefKind { Normal, Input, Subview };
inline const char *memrefKindToString(MemrefKind kind) {
  switch (kind) {
    case MemrefKind::Normal:
      return "Normal";
    case MemrefKind::Input:
      return "Input";
    case MemrefKind::Subview:
      return "Subview";
  }
  return "UNKNOWN";
}

struct Group {
 public:
  Group(unsigned groupId, unsigned rootId, affine::AffineForOp root) : groupId(groupId), rootId(rootId), root(root) {}

  // Group management
  affine::AffineForOp getLeadingFor() const { return root; }
  void addNode(unsigned nodeId) { nodesId.push_back(nodeId); }
  void setIsGlobalOut(bool isOut) { isGlobalOut = isOut; }

  // Debug and print
  void dump() const { print(llvm::errs()); }
  void print(llvm::raw_ostream &os) const;
  std::string getGroupTemplateString() const;

  unsigned groupId;
  unsigned rootId;
  affine::AffineForOp root;
  std::vector<unsigned> nodesId;
  OperatorTemplate groupTemplate{OperatorTemplate::Default};
  bool isGlobalOut{false};
  std::vector<unsigned> fusedGroupId;
};
using GroupPtr = std::shared_ptr<Group>;

struct FuseEdge {
  FuseEdge(unsigned from, unsigned to) : from(from), to(to) {}
  unsigned from;
  unsigned to;
};

struct DependenceInfo {
  unsigned predNodeId{UINT_MAX};
  unsigned targetNodeId{UINT_MAX};
  DepType depType{DepType::OTHER};
  Value memref;
  unsigned loopDepth{UINT_MAX};
  MemrefKind memrefKind{MemrefKind::Normal};
};

struct FusionPlan {
  FuseEdge fusedGroup{FuseEdge(0, 0)};
  FuseEdge fusedBand{FuseEdge(0, 0)};

  DependenceInfo depInfo;
  std::string fusionType{"H"};
  LoopTransform loopTransform{LoopTransform::Merge};
};

}  // namespace akg
}  // namespace mlir

#endif  // AKG_DIALECT_AFFINE_ANALYSIS_LOOPFUSIONUTILS_H_
