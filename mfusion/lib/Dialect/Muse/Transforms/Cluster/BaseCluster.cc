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

#include "mfusion/Dialect/Muse/Transforms/Cluster/BaseCluster.h"

#include <algorithm>
#include <utility>
#include <unordered_set>

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "mfusion/Analysis/Graph.h"
#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/Transforms/Cluster/Utils.h"

#define DEBUG_TYPE "graph-kernel-cluster"

namespace mlir {
namespace muse {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {
/// Check if constant value is finite based on its type
bool isFiniteValue(DenseElementsAttr dense_attr) {
  // Handle dense attributes (tensor constants)
  // Get element type and check accordingly
  Type element_type = dense_attr.getElementType();
  if (auto float_type = dyn_cast<FloatType>(element_type)) {
    // For all floating point types, use APFloat to check if finite
    auto apfloat_values = dense_attr.getValues<APFloat>();
    if (!apfloat_values.empty()) {
      APFloat value = *apfloat_values.begin();
      return !value.isNaN() && !value.isInfinity();
    }
  }
  // For non-floating point types or if we can't check, consider finite
  return true;
}

/// Collect constant operations that should be included in the cluster.
/// This function analyzes operands of cluster operations and determines which
/// constant operations should be fused into the cluster (vs extracted as inputs).
llvm::DenseSet<Operation *> CollectConstantsToCluster(llvm::ArrayRef<Operation *> cluster_ops) {
  // Build a set of cluster operations for fast lookup
  llvm::DenseSet<Operation *> cluster_op_set(cluster_ops.begin(), cluster_ops.end());

  llvm::DenseSet<Operation *> constants_to_cluster;
  const auto &op_index_info = getConstInputIndexInfo();

  for (Operation *op : cluster_ops) {
    std::string op_name = op->getName().getStringRef().str();

    // Get the const input indices for this operation type
    const std::unordered_set<size_t> *const_indices = nullptr;
    auto iter = op_index_info.find(op_name);
    if (iter != op_index_info.end()) {
      const_indices = &iter->second;
    }

    for (size_t idx = 0; idx < op->getNumOperands(); ++idx) {
      Value operand = op->getOperand(idx);
      Operation *def_op = operand.getDefiningOp();
      if (def_op == nullptr || cluster_op_set.contains(def_op) || constants_to_cluster.contains(def_op)) {
        continue;
      }

      // Check if this is a constant operation
      auto arith_const_op = dyn_cast<arith::ConstantOp>(def_op);
      if (!arith_const_op) {
        continue;
      }
      Type result_type = arith_const_op.getResult().getType();

      // If constant is not a tensor type (e.g., scalar), include it in cluster
      if (!isa<TensorType>(result_type)) {
        LLVM_DEBUG(llvm::dbgs() << "Constant at index " << idx << " for " << op_name
                                << " is not a tensor type, keeping in cluster\n");
        constants_to_cluster.insert(def_op);
        continue;
      }

      // Check if this constant is used at a value-dependent index
      if (const_indices != nullptr && const_indices->count(idx) != 0) {
        LLVM_DEBUG(llvm::dbgs() << "Constant at index " << idx << " for " << op_name
                                << " is value-dependent, keeping in cluster\n");
        constants_to_cluster.insert(def_op);
        continue;
      }

      // Get the constant attribute
      Attribute value_attr = arith_const_op.getValueAttr();
      if (!value_attr || !result_type) {
        continue;
      }

      // Check if it's a single-element finite tensor constant
      auto dense_attr = dyn_cast<DenseElementsAttr>(value_attr);
      if (!dense_attr || dense_attr.getNumElements() != 1 || !isFiniteValue(dense_attr)) {
        continue;
      }

      // Special case: bool type or grouped_matmul's int64 group_list parameter
      Type element_type = dense_attr.getElementType();
      if (auto int_type = dyn_cast<IntegerType>(element_type)) {
        const size_t group_list_index = 7;
        if (int_type.getWidth() == 1 ||
            (op_name == "muse.grouped_matmul" && idx == group_list_index && int_type.getWidth() == 64)) {
          continue;
        }
      }

      // Include it in the cluster
      LLVM_DEBUG(llvm::dbgs() << "Constant at index " << idx << " for " << op_name
                              << " is single-element finite value, keeping in cluster\n");
      constants_to_cluster.insert(def_op);
    }
  }

  return constants_to_cluster;
}
}  // namespace

//===----------------------------------------------------------------------===//
// BaseCluster Implementation
//===----------------------------------------------------------------------===//

void BaseCluster::Init() { op_list_ = GetClusterableOpList(); }

bool BaseCluster::Run(func::FuncOp func_op) {
  Init();
  bool changed = Process(func_op);
  Clean();
  return changed;
}

bool BaseCluster::Process(func::FuncOp func_op) {
  Block &block = func_op.getBody().front();
  GraphMerge(&block, true);

  if (graph_->HasCircle()) {
    LLVM_DEBUG(llvm::dbgs() << "Graph has circle, trying again with conservative strategy\n");
    GraphMerge(&block, false);
    if (graph_->HasCircle()) {
      LLVM_DEBUG(llvm::dbgs() << "Graph still has circle!\n");
    }
  }

  // Rebuild the IR with fused operations
  bool changed = false;
  auto clusters = graph_->CollectClusters();

  for (size_t i = 0; i < clusters.size(); ++i) {
    size_t node_count = clusters[i].size();
    if (node_count == 0 || node_count == 1) {
      continue;
    }
    CreateFusedOp(func_op, clusters[i]);
    changed = true;
  }

  return changed;
}

void BaseCluster::GraphMerge(Block *block, bool aggressive_cut) {
  llvm::DenseMap<Operation *, size_t> op_idx_map;
  graph_ = Graph::Build(block, &ops_, &op_idx_map, aggressive_cut);

  // Process nodes in reverse order (from outputs to inputs)
  for (int i = static_cast<int>(ops_.size()) - 1; i >= 0; --i) {
    // Skip if already part of a multi-node cluster
    if (graph_->GetSize(static_cast<size_t>(i)) > 1) {
      continue;
    }

    auto candidates = FindCandidates(static_cast<size_t>(i));
    CircleChecker circle_checker(graph_.get());
    circle_checker.RemoveCircle(&candidates);

    if (candidates.size() <= 1) {
      continue;
    }

    // Merge candidates into one cluster
    graph_->Merge(candidates);
  }
}

std::vector<size_t> BaseCluster::FindCandidates(size_t basenode_id) {
  std::vector<size_t> candidates;
  Operation *base_op = ops_[basenode_id];
  Block *block = base_op->getBlock();

  auto include = [this, &candidates, block](size_t cluster_id) {
    Operation *op = this->ops_[cluster_id];
    // Must be in the same block
    if (op->getBlock() != block) {
      return VisitResult::kExclude;
    }
    // Must be clusterable
    if (!IsClusterableOp(op)) {
      return VisitResult::kExclude;
    }
    candidates.push_back(cluster_id);
    // Do not search from already clustered node
    if (this->graph_->GetSize(cluster_id) > 1) {
      return VisitResult::kNoFollow;
    }
    return VisitResult::kFollow;
  };

  graph_->Dfs(basenode_id, include);
  std::reverse(candidates.begin(), candidates.end());
  return candidates;
}

void BaseCluster::CreateFusedOp(func::FuncOp func_op, const std::vector<size_t> &nodes_id) {
  if (nodes_id.empty()) {
    return;
  }

  // Collect operations in the cluster
  llvm::SmallVector<Operation *> cluster_ops;
  for (size_t id : nodes_id) {
    cluster_ops.push_back(ops_[id]);
  }

  // Collect constant operations that should be included in the cluster
  llvm::DenseSet<Operation *> constants_to_cluster = CollectConstantsToCluster(cluster_ops);

  // Add constant operations to cluster_ops
  for (Operation *const_op : constants_to_cluster) {
    cluster_ops.push_back(const_op);
  }

  // Sort operations after adding constants
  std::sort(cluster_ops.begin(), cluster_ops.end(), [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  // Find external inputs (values defined outside the cluster)
  llvm::SetVector<Value> external_inputs;
  llvm::DenseSet<Operation *> cluster_op_set(cluster_ops.begin(), cluster_ops.end());

  for (Operation *op : cluster_ops) {
    for (Value operand : op->getOperands()) {
      Operation *def_op = operand.getDefiningOp();
      // External if defined by block argument or by an op outside the cluster
      if (def_op == nullptr || !cluster_op_set.contains(def_op)) {
        external_inputs.insert(operand);
      }
    }
  }

  // Find external outputs (values used outside the cluster)
  llvm::SetVector<Value> external_outputs;
  for (Operation *op : cluster_ops) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!cluster_op_set.contains(user)) {
          external_outputs.insert(result);
          break;
        }
      }
    }
  }

  if (external_outputs.empty()) {
    // No external outputs means this cluster's results are not used
    // This shouldn't normally happen, but handle it gracefully
    return;
  }

  // Create the fused operation
  OpBuilder builder(func_op.getContext());
  builder.setInsertionPointAfter(cluster_ops.front());

  // Collect result types
  llvm::SmallVector<Type> result_types;
  for (Value output : external_outputs) {
    result_types.push_back(output.getType());
  }

  // Create FusedOp
  auto fused_op = builder.create<FusedOp>(cluster_ops.front()->getLoc(), result_types, external_inputs.getArrayRef(),
                                          builder.getStringAttr(GetFusionType()), /*kernel_name=*/nullptr);

  // Create the body block with arguments for each external input
  Block *body = new Block();
  fused_op.getBody().push_back(body);

  llvm::SmallVector<Location> arg_locs;
  for (Value input : external_inputs) {
    arg_locs.push_back(input.getLoc());
  }
  body->addArguments(TypeRange(external_inputs.getArrayRef()), arg_locs);

  // Create mapping from external inputs to block arguments
  IRMapping mapping;
  for (auto [input, arg] : llvm::zip(external_inputs, body->getArguments())) {
    mapping.map(input, arg);
  }

  // Clone operations into the fused body
  builder.setInsertionPointToStart(body);
  for (Operation *op : cluster_ops) {
    builder.clone(*op, mapping);
  }

  // Create yield operation with the outputs
  llvm::SmallVector<Value> yield_values;
  for (Value output : external_outputs) {
    yield_values.push_back(mapping.lookup(output));
  }
  builder.create<YieldOp>(fused_op.getLoc(), yield_values);

  // Replace uses of cluster outputs with fused op results
  for (auto [old_output, new_result] : llvm::zip(external_outputs, fused_op.getResults())) {
    Value output = old_output;  // Create non-const copy for replaceAllUsesWith
    output.replaceAllUsesWith(new_result);
  }

  // Erase original operations in reverse order
  for (auto it = cluster_ops.rbegin(); it != cluster_ops.rend(); ++it) {
    (*it)->erase();
  }
}

}  // namespace muse
}  // namespace mlir
