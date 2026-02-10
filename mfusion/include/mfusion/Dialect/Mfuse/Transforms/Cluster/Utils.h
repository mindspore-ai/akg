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

#ifndef MFUSION_INCLUDE_DIALECT_MFUSE_TRANSFORMS_CLUSTER_UTILS_H_
#define MFUSION_INCLUDE_DIALECT_MFUSE_TRANSFORMS_CLUSTER_UTILS_H_

#include <unordered_set>
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace mfuse {

/// Check if operation has zero-shaped tensors
bool hasZeroShape(Operation *op);

/// Check if operation has dynamic shape tensor
bool isDynamicShapeNode(Operation *op);

/// Get const input index info for value-dependent operations
const std::unordered_map<std::string, std::unordered_set<size_t>> &getConstInputIndexInfo();
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_INCLUDE_DIALECT_MFUSE_TRANSFORMS_CLUSTER_UTILS_H_