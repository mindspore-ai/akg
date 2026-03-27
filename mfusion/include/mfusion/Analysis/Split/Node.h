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

#ifndef MFUSION_ANALYSIS_SPLIT_NODE_H
#define MFUSION_ANALYSIS_SPLIT_NODE_H

#include <memory>
#include <vector>
#include <set>
#include <string>
#include <unordered_map>

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "symengine/basic.h"

namespace mlir {
namespace mfuse {
namespace split {

typedef std::vector<int64_t> DShape;
typedef std::unordered_map<std::string, Attribute> DAttrs;
typedef SymEngine::RCP<const SymEngine::Basic> SymExpr;

struct NodeBase {
  DShape shape;
  std::vector<SymExpr> sym_shape;
};

// Node base class
class Node : public NodeBase {
 public:
  Node() = default;
  explicit Node(Operation *op) : op_(op) {}
  virtual ~Node() {}

  virtual std::string toString() const;
  void addInput(Node *new_input);
  void setInput(size_t i, Node *new_input);
  void setInputs(const std::vector<Node *> &inputs);
  void clearInputs() noexcept;
  void replaceWith(Node *other_node);
  void setAttrs(const DAttrs &attrs) { attrs_ = attrs; }
  void setAttr(const std::string &key, const Attribute &value) { attrs_[key] = value; }
  void setDebugName(const std::string &debug_name) { debug_name_ = debug_name; }
  Operation *op() const { return op_; }

  template <typename T>
  T *as() {
    return static_cast<T *>(this);
  }

  const std::string &debugName() const { return debug_name_; }
  const DAttrs &attrs() const { return attrs_; }
  const Node *input(size_t i) const { return inputs_[i]; }
  const std::vector<Node *> &inputs() const { return inputs_; }
  const std::unordered_map<Node *, std::set<size_t>> &users() const { return users_; }
  size_t inputNum() const { return inputs_.size(); }
  size_t userNum() const { return users_.size(); }

 protected:
  mutable std::string debug_name_;
  DAttrs attrs_;
  std::vector<Node *> inputs_;
  std::unordered_map<Node *, std::set<size_t>> users_;
  Operation *op_;

 private:
  void addUser(Node *user, size_t index) { users_[user].insert(index); }
  void removeUser(Node *user, size_t index);
};

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_NODE_H
