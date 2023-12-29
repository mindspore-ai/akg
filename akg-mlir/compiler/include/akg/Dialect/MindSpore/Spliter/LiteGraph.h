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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_LITEGRAPH_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_LITEGRAPH_H_

#include <memory>
#include <string>
#include "akg/Dialect/MindSpore/Spliter/Node.h"

namespace mlir::spliter {
class LiteGraph {
 public:
  class GraphBuilderBase;
  explicit LiteGraph(const std::string &graphName = "") : name(graphName), output(new OutputNode()) {}
  const NodePtrList &getOrderedNodes();
  std::string toString(bool resetNodeName = false) const;
  const std::string &getName() const { return name; }
  const NodePtrList &getOps() const { return ops; }
  const NodePtrList &getInputs() const { return inputs; }
  const NodePtr &getOutput(size_t i) const { return output->getInput(i); }
  const NodePtrList &getOutputs() const { return output->getInputs(); }

  void setOutput(size_t i, const NodePtr &node) { output->setInput(i, node); }
  void setOutputs(const NodePtrList &nodes) { output->setInputs(nodes); }

 protected:
  std::string name;
  NodePtrList ops;  // save all operators in topo order
  NodePtrList inputs;
  NodePtr output;

 private:
  std::string paramName() const { return "input_" + std::to_string(paramId++); }
  std::string nodeName() const { return "output_" + std::to_string(nodeId++); }
  mutable int paramId{0};
  mutable int nodeId{0};
};

using LiteGraphPtr = std::shared_ptr<LiteGraph>;
class LiteGraph::GraphBuilderBase {
 public:
  explicit GraphBuilderBase(const std::string &name = "") { graph = std::make_shared<LiteGraph>(name); }
  ~GraphBuilderBase() = default;

  // Create a parameter of graph
  NodePtr parameter(const NodeBase &baseInfo) const {
    auto para = std::make_shared<ParamNode>(baseInfo);
    para->setDebugName(graph->paramName());
    graph->inputs.push_back(para);
    return para;
  }

  // Create a const value node
  NodePtr value(const NodeBase &baseInfo) const { return std::make_shared<ConstTensorNode>(baseInfo); }

  void setOutputs(const NodePtrList &nodes) const { graph->output->setInputs(nodes); }

  // Create op node with given baseInfo.
  NodePtr getOp(const std::string &op, const NodeBase &baseInfo, const NodePtrList &inputs,
                const DAttrs &attrs = {}) const;
  LiteGraphPtr get() const { return graph; }

 private:
  LiteGraphPtr graph;
};

class GraphBuilder : public LiteGraph::GraphBuilderBase {
 public:
  explicit GraphBuilder(const std::string &name = "") : GraphBuilderBase(name) {}
  ~GraphBuilder() = default;
};

LiteGraphPtr mindsporeToLiteGraph(const func::FuncOp func, HashMap<NodePtr, Operation *> *opNodeMap);

}  // namespace mlir::spliter
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_LITEGRAPH_H_
