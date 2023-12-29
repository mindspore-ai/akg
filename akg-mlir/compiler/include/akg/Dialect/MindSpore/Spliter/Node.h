/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_NODE_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_NODE_H_
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/Spliter/Utils.h"

using SymEngine::Expression;
namespace mlir::spliter {
enum class NType {
  Base,
  Primitive,
  Parameter,
  Value,
  Output,
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const NType &x);

using DFormat = std::string;
using DShape = ShapeVector;
using DAttrs = mlir::DictionaryAttr;

struct NodeBase {
  DShape shape;
  mlir::Type type;
  DFormat format;
  std::vector<std::string> symbolicShape;
};

using NodeBaseList = std::vector<NodeBase>;

class Node;
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
class Node : public NodeBase, public std::enable_shared_from_this<Node> {
 public:
  explicit Node(const NodeBase &baseinfo) : NodeBase(baseinfo) {}
  virtual ~Node() { clearInputs(); }  // remove this node from the previous nodes' user.

  virtual NType nodeType() { return NType::Base; }
  virtual std::string toString() const;

  virtual void setBaseInfo(const NodeBaseList &baseinfo);
  void addInput(const NodePtr &newInput);
  void setInput(size_t i, const NodePtr &newInput);
  void setInputs(const NodePtrList &inputs);
  void clearInputs() noexcept;
  void replaceWith(const NodePtr &otherNode);
  void setAttrs(const DAttrs &attr) { attrs = attr; }
  void setDebugName(const std::string &name) { debugName = name; }

  template <typename T>
  std::shared_ptr<T> as() {
    return std::static_pointer_cast<T>(shared_from_this());
  }

  const std::string &getDebugName() const { return debugName; }
  const DAttrs &getAttrs() const { return attrs; }
  const NodePtr &getInput(size_t i) const { return inputs[i]; }
  const NodePtrList &getInputs() const { return inputs; }
  const HashMap<Node *, std::set<size_t>> &getUsers() const { return users; }
  bool hasSymbolicShape() const { return !symbolicShape.empty(); }
  size_t tensorSize(bool inBytes = false) const;
  SymEngine::Expression dynTensorSize(bool inBytes = false) const;
  bool hasSameTensorSize(const NodePtr &other) const;
  bool hasSameShape(const NodePtr &other) const;
  const NodeBaseList &getOutputs() const { return outputs; }

 protected:
  // only used in Dump function
  mutable std::string debugName;
  DAttrs attrs;
  NodePtrList inputs;
  // {user_node: {input edge index set}}
  HashMap<Node *, std::set<size_t>> users;
  // save output tensor info when the node is a multi-output operator.
  // it should keep empty when the node is single-output.
  NodeBaseList outputs;

 private:
  // the nodes' users are only maintained by AddInput/setInput.
  void addUser(Node *const user, const size_t index) { (void)users[user].insert(index); }
  void removeUser(Node *const user, size_t index);
};

class ConstTensorNode : public Node {
 public:
  explicit ConstTensorNode(const NodeBase &baseinfo) : Node(baseinfo) {}
  ~ConstTensorNode() = default;

  NType nodeType() override { return NType::Value; }
  std::string toString() const override { return "const value"; }
};

class ParamNode : public Node {
 public:
  explicit ParamNode(const NodeBase &baseinfo) : Node(baseinfo) {}
  ~ParamNode() = default;

  NType nodeType() override { return NType::Parameter; }
};

// the OutputNode's inputs are the real outputs of graph, like the `make_tuple`
// in FuncGraph.
class OutputNode : public Node {
 public:
  OutputNode() : Node({{1}, nullptr, kOpFormat_DEFAULT}) {}
  ~OutputNode() = default;

  NType nodeType() override { return NType::Output; }
};
}  // namespace mlir::spliter
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_NODE_H_
