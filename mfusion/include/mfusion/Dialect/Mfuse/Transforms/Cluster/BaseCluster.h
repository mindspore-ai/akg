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

#ifndef MFUSION_INCLUDE_DIALECT_MFUSE_TRANSFORMS_CLUSTER_BASECLUSTER_H_
#define MFUSION_INCLUDE_DIALECT_MFUSE_TRANSFORMS_CLUSTER_BASECLUSTER_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "mfusion/Analysis/Cluster/Graph.h"

namespace mlir {
namespace mfuse {

//===----------------------------------------------------------------------===//
// BaseCluster - Base class for clustering passes
//===----------------------------------------------------------------------===//

/// Base class for operation clustering passes.
/// Provides common infrastructure for Mfuse clustering.
class BaseCluster {
 public:
  BaseCluster() = default;
  virtual ~BaseCluster() = default;

  /// Run the clustering algorithm on a function.
  virtual bool run(func::FuncOp funcOp);

 protected:
  /// Get the list of operation names that can be clustered.
  virtual llvm::DenseSet<llvm::StringRef> getClusterableOpList() = 0;

  /// Check if an operation can be clustered.
  virtual bool isClusterableOp(Operation *op) = 0;

  /// Get the fusion type string for this cluster ("dvm" or "akg").
  virtual std::string getFusionType() = 0;

  /// Initialize the pass with the clusterable operation list.
  void init();

  /// Main clustering process.
  bool process(func::FuncOp funcOp);

  /// Build graph and merge clusters.
  void graphMerge(Block *block, bool aggressiveCut);

  /// Find candidate operations that can be merged with base node.
  std::vector<size_t> findCandidates(size_t baseNodeId);

  /// Create a mfuse.fused operation for a cluster.
  void createFusedOp(func::FuncOp funcOp, const std::vector<size_t> &nodeIds);

  /// Clean up internal state.
  void clean() {
    ops_.clear();
    graph_.reset();
  }

  std::unique_ptr<Graph> graph_;
  std::vector<Operation *> ops_;
  llvm::DenseSet<llvm::StringRef> opList_;
};

/// DVMCluster - Clusters operations checked by DVM backend.
class DVMCluster : public BaseCluster {
 public:
  DVMCluster() = default;
  ~DVMCluster() override = default;

  static llvm::DenseSet<llvm::StringRef> getClusterableOps();

  static bool canClusterableOp(const llvm::DenseSet<llvm::StringRef> &opList, Operation *op);

 protected:
  llvm::DenseSet<llvm::StringRef> getClusterableOpList() override;

  bool isClusterableOp(Operation *op) override;

  std::string getFusionType() override;
};

/// AKGCluster - Clusters operations checked by AKG backend.
class AKGCluster : public BaseCluster {
 public:
  AKGCluster() = default;
  ~AKGCluster() override = default;

  static llvm::DenseSet<llvm::StringRef> getClusterableOps();

  static bool canClusterableOp(const llvm::DenseSet<llvm::StringRef> &opList, Operation *op);

 protected:
  llvm::DenseSet<llvm::StringRef> getClusterableOpList() override;

  bool isClusterableOp(Operation *op) override;

  std::string getFusionType() override;
};

/// BishengCluster - Clusters operations checked by Bisheng backend.
class BishengCluster : public BaseCluster {
 public:
  BishengCluster() = default;
  ~BishengCluster() override = default;

  static llvm::DenseSet<llvm::StringRef> getClusterableOps();

  static bool canClusterableOp(const llvm::DenseSet<llvm::StringRef> &opList, Operation *op);

 protected:
  llvm::DenseSet<llvm::StringRef> getClusterableOpList() override;

  bool isClusterableOp(Operation *op) override;

  std::string getFusionType() override;
};

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_INCLUDE_DIALECT_MFUSE_TRANSFORMS_CLUSTER_BASECLUSTER_H_
