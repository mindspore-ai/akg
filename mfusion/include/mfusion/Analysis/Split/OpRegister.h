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

#ifndef MFUSION_ANALYSIS_SPLIT_OPREGISTER_H
#define MFUSION_ANALYSIS_SPLIT_OPREGISTER_H

#include <string>
#include "mlir/IR/Operation.h"

namespace mlir {
namespace mfuse {
namespace split {

// OpRegistry is used to register operation patterns
class OpRegistry {
 public:
  static OpRegistry &Instance() {
    static OpRegistry instance;
    return instance;
  }

  void Register(const std::string &op_name, NodePattern pattern) { (void)patterns_.emplace(op_name, pattern); }

  NodePattern GetPattern(const std::string &op) {
    // Default to OPAQUE if not found
    return patterns_.find(op) == patterns_.end() ? NodePattern::OPAQUE : patterns_[op];
  }

  int GetComputeType(const std::string &op) { return static_cast<int>(GetPattern(op)); }

 private:
  OpRegistry() = default;
  ~OpRegistry() = default;
  OpRegistry(const OpRegistry &) = delete;
  OpRegistry(const OpRegistry &&) = delete;
  OpRegistry &operator=(const OpRegistry &) = delete;
  OpRegistry &operator=(const OpRegistry &&) = delete;

  std::map<std::string, NodePattern> patterns_;
};

class OpRegister {
 public:
  OpRegister(const std::string &name, NodePattern pattern) : name_(name) {
    OpRegistry::Instance().Register(name, pattern);
  }
  ~OpRegister() = default;

 private:
  std::string name_;
};

#define JOIN(x, y) x##y
#define UNIQUE_NAME(prefix, cnt) JOIN(prefix, cnt)

// OP_REGISTER for enum values
#define OP_REGISTER(name, pattern) static const OpRegister UNIQUE_NAME(g_mfuse_op, __COUNTER__)(name, pattern)

// Register operations
// Only support dvm now
OP_REGISTER("mfuse.abs", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.add", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.broadcast_to", NodePattern::BROADCAST);
OP_REGISTER("mfuse.cast", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.exp", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.log", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.relu", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.maximum", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.minimum", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.mul", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.neg", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.pow", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.div", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.real_div", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.reciprocal", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.rsqrt", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.sqrt", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.sub", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.eq", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.ne", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.gt", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.ge", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.lt", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.le", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.logical_and", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.logical_or", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.logical_not", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.select", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.assign", NodePattern::VIRTUAL);
OP_REGISTER("mfuse.reduce_sum", NodePattern::REDUCE);
OP_REGISTER("mfuse.is_finite", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.reshape", NodePattern::RESHAPE);
OP_REGISTER("mfuse.permute", NodePattern::OPAQUE);  // Transpose
OP_REGISTER("mfuse.floor", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.ceil", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.trunc", NodePattern::ELEMWISE);
OP_REGISTER("mfuse.matmul", NodePattern::OPAQUE);
OP_REGISTER("mfuse.batch_matmul", NodePattern::OPAQUE);
OP_REGISTER("mfuse.grouped_matmul", NodePattern::OPAQUE);

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_OPREGISTER_H
