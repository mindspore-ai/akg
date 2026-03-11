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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_SCHEMER_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_SCHEMER_H

#include <vector>
#include <memory>
#include <string>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace mfuse {
namespace split {

//===----------------------------------------------------------------------===//
// SplitSchemer
//===----------------------------------------------------------------------===

/// SplitSchemer defines the splitting scheme for graph kernels
class SplitSchemer {
 public:
  SplitSchemer() = default;
  virtual ~SplitSchemer() = default;

  /// Split the function according to the scheme
  virtual bool split(Block *block) = 0;

  /// Check if the group needs to be inlined
  virtual bool needInline(size_t groupId) const;

  /// Get the split plan
  const std::vector<SmallVector<Operation *>> &getSplitPlan() const { return split_plan_; }

  /// Get the name of the schemer
  std::string name() const;

 protected:
  std::vector<SmallVector<Operation *>> split_plan_;
  std::vector<int> need_inline_;
};

using SplitSchemerPtr = std::shared_ptr<SplitSchemer>;

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_SCHEMER_H
