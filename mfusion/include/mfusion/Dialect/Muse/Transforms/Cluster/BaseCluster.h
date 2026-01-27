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

#ifndef MFUSION_INCLUDE_DIALECT_MUSE_TRANSFORMS_CLUSTER_BASECLUSTER_H_
#define MFUSION_INCLUDE_DIALECT_MUSE_TRANSFORMS_CLUSTER_BASECLUSTER_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "mfusion/Analysis/Graph.h"

namespace mlir {
namespace muse {

//===----------------------------------------------------------------------===//
// BaseCluster - Base class for clustering passes
//===----------------------------------------------------------------------===//

/// Base class for operation clustering passes.
/// Provides common infrastructure for DVM and AKG clustering.
class BaseCluster {
 public:
  BaseCluster() = default;
  virtual ~BaseCluster() = default;

  /// Run the clustering algorithm on a function.
  virtual bool Run(func::FuncOp func_op);

 protected:
  /// Get the list of operation names that can be clustered.
  virtual llvm::DenseSet<llvm::StringRef> GetClusterableOpList() = 0;

  /// Check if an operation can be clustered.
  virtual bool IsClusterableOp(Operation *op) = 0;

  /// Get the fusion type string for this cluster ("dvm" or "akg").
  virtual std::string GetFusionType() = 0;

  /// Initialize the pass with the clusterable operation list.
  void Init();

  /// Main clustering process.
  bool Process(func::FuncOp func_op);

  /// Build graph and merge clusters.
  void GraphMerge(Block *block, bool aggressive_cut);

  /// Find candidate operations that can be merged with basenode.
  std::vector<size_t> FindCandidates(size_t basenode_id);

  /// Create a muse.fused operation for a cluster.
  void CreateFusedOp(func::FuncOp func_op, const std::vector<size_t> &nodes_id);

  /// Clean up internal state.
  void Clean() {
    ops_.clear();
    graph_.reset();
  }

  std::unique_ptr<Graph> graph_;
  std::vector<Operation *> ops_;
  llvm::DenseSet<llvm::StringRef> op_list_;
};

/// DVMCluster - Clusters operations checked by DVM backend.
class DVMCluster : public BaseCluster {
 public:
  DVMCluster() = default;
  ~DVMCluster() override = default;

  static llvm::DenseSet<llvm::StringRef> GetClusterableOps();

  static bool CanClusterableOp(const llvm::DenseSet<llvm::StringRef> &op_list, Operation *op);

 protected:
  llvm::DenseSet<llvm::StringRef> GetClusterableOpList() override;

  bool IsClusterableOp(Operation *op) override;

  std::string GetFusionType() override;
};

}  // namespace muse
}  // namespace mlir

#endif  // MFUSION_INCLUDE_DIALECT_MUSE_TRANSFORMS_CLUSTER_BASECLUSTER_H_
