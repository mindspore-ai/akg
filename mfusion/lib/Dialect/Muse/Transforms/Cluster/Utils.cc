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

#include "mfusion/Dialect/Muse/Transforms/Cluster/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace muse {

/// Check if operation has zero-shaped tensors
bool hasZeroShape(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  auto isZeroShape = [](Type type) {
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.hasRank() && llvm::any_of(shapedType.getShape(), [](int64_t dim) { return dim == 0; });
    }
    return false;
  };

  return llvm::any_of(op->getResults(), [&](Value v) { return isZeroShape(v.getType()); });
}

/// Check if operation has dynamic shape tensor
bool isDynamicShapeNode(Operation *op) {
  if (op == nullptr) {
    return false;
  }

  auto hasDynamicShape = [](Type type) -> bool {
    auto shapedType = dyn_cast<ShapedType>(type);
    // Only check ranked tensors with unknown dimensions
    return shapedType && shapedType.hasRank() && !shapedType.hasStaticShape();
  };

  auto checkOperandsOrResults = [&](ValueRange values) -> bool {
    return llvm::any_of(values, [&](Value v) { return hasDynamicShape(v.getType()); });
  };

  return checkOperandsOrResults(op->getOperands()) || checkOperandsOrResults(op->getResults());
}

/// Get const input index info for value-dependent operations
const std::unordered_map<std::string, std::unordered_set<size_t>> &getConstInputIndexInfo() {
  static const std::unordered_map<std::string, std::unordered_set<size_t>> op_idx_info = {
    {"muse.reshape", {1}},
    {"muse.reduce_max", {1}},
    {"muse.expand_dims", {1}},
    {"muse.reduce_min", {1}},
    {"muse.reduce_sum", {1}},
    {"muse.permute", {1}},
    {"muse.tile", {1}},
    {"muse.broadcast_to", {1}},
    {"muse.reduce_mean", {1}},
    {"muse.slice", {1, 2}},
    {"muse.strided_slice", {1, 2, 3}},
    {"muse.one_hot", {1}},
    {"muse.reduce_fusion", {1}},
    {"muse.constant_of_shape", {0}},
    {"muse.gather", {2}},
    {"muse.unsorted_segment_sum", {2}},
    {"muse.cumsum", {1}},
  };
  return op_idx_info;
}

}  // namespace muse
}  // namespace mlir
