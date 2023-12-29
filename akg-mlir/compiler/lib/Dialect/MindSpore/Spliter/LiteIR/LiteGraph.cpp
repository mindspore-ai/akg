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
#include "akg/Dialect/MindSpore/Spliter/LiteGraph.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/Spliter/Node.h"
#include "akg/Dialect/MindSpore/Spliter/OpNode.h"
#include "akg/Dialect/MindSpore/Spliter/OpRegister.h"

namespace mlir::spliter {
std::string LiteGraph::toString(bool resetNodeName) const {
  if (resetNodeName) {
    paramId = nodeId = 0;
    for (auto &inp : inputs) {
      inp->setDebugName(paramName());
    }
    for (auto &node : ops) {
      node->setDebugName(nodeName());
    }
  }
  std::ostringstream os;
  os << name << "(";
  for (size_t i = 0; i < inputs.size(); i++) {
    os << inputs[i]->getDebugName();
    if (i != inputs.size() - 1) {
      os << ", ";
    }
  }
  os << ") -> ";
  auto &outputs = getOutputs();
  for (size_t i = 0; i < outputs.size(); i++) {
    os << outputs[i]->getDebugName();
    if (i != outputs.size() - 1) {
      os << ", ";
    }
  }
  os << " {\n";
  for (const NodePtr &op : ops) {
    os << "  " << op->toString() << "\n";
  }
  os << "}";
  return os.str();
}

const NodePtrList &LiteGraph::getOrderedNodes() {
  HashMap<NodePtr, size_t> outdegrees;
  std::function<void(NodePtr)> dfs;
  std::set<NodePtr> visited;
  // record the out degree of each nodes by Dfs.
  dfs = [&dfs, &outdegrees, &visited](const NodePtr &node) {
    (void)visited.insert(node);
    for (auto &input : node->getInputs()) {
      if (input->nodeType() == NType::Primitive) {
        ++outdegrees[input];
        if (visited.count(input) == 0) {
          dfs(input);
        }
      }
    }
  };
  dfs(output);
  NodePtrList res;
  NodePtrList stack;

  // toposort algorithm with out degree
  stack.push_back(output);
  while (!stack.empty()) {
    auto cur = stack.back();
    stack.pop_back();
    res.push_back(cur);
    for (auto &input : cur->getInputs()) {
      if (input->nodeType() != NType::Primitive) {
        continue;
      }
      --outdegrees[input];
      if (outdegrees[input] == 0) {
        stack.push_back(input);
        (void)outdegrees.erase(input);
      }
    }
  }
  if (!outdegrees.empty()) {
    llvm::errs() << "Circle was found:";
    for (auto &node : outdegrees) {
      llvm::errs() << "  " << node.first->getDebugName();
    }
    llvm::errs() << "Circle size: " << outdegrees.size();
  }
  std::reverse(res.begin(), res.end());
  // remove the "OutputNode"
  res.pop_back();
  ops = std::move(res);
  return ops;
}

PrimOpPtr CreateOp(const std::string &op, const std::string &debugName) {
  auto node = OpRegistry::getInstance().newOp(op);
  node->setDebugName(debugName);
  return node;
}

NodePtr LiteGraph::GraphBuilderBase::getOp(const std::string &op, const NodeBase &baseInfo, const NodePtrList &inputs,
                                           const DAttrs &attrs) const {
  PrimOpPtr opPtr = CreateOp(op, graph->nodeName());
  opPtr->setInputs(inputs);
  opPtr->setAttrs(attrs);
  opPtr->setBaseInfo({baseInfo});
  (void)graph->ops.emplace_back(opPtr);
  return opPtr;
}

LiteGraphPtr mindsporeToLiteGraph(const func::FuncOp func, HashMap<NodePtr, Operation *> *opNodeMap) {
  std::string name = "Default";
  GraphBuilder gb(name);
  DenseMap<Value, NodePtr> nodeMap;

  auto extractBuildInfo = [](const Value &output) {
    auto json_desc = JsonOpBuilder().getTensorJson(output);
    Type type = output.getType();
    assert(type.isa<RankedTensorType>());
    auto symbolicShape = SymbolicShapeAnalysis::getInstance().getSymbolicShape(type);
    if (symbolicShape.has_value() && !SymbolicShapeAnalysis::getInstance().isRankedTensorStaticShape(type)) {
      auto sym_shape = symbolicShape.value();
      return NodeBase({json_desc[kJsonKeyShape], type.cast<RankedTensorType>().getElementType(),
                       json_desc[kJsonKeyFormat], std::vector<std::string>(sym_shape.begin(), sym_shape.end())});
    } else {
      return NodeBase(
        {json_desc[kJsonKeyShape], type.cast<RankedTensorType>().getElementType(), json_desc[kJsonKeyFormat]});
    }
  };

  auto genParamNodes = [&](func::FuncOp func) {
    Block &entryBlock = func.getBody().front();
    for (Value opnd : entryBlock.getArguments()) {
      nodeMap[opnd] = gb.parameter(extractBuildInfo(opnd));
    }
  };

  auto genInputNodes = [&](Operation *op) {
    auto inputs = op->getOperands();
    NodePtrList nodeInputs;
    (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(nodeInputs), [&nodeMap](const Value &input) {
      assert(nodeMap.find(input) != nodeMap.end());
      return nodeMap[input];
    });
    return nodeInputs;
  };

  auto genOpNodes = [&](func::FuncOp func) {
    Block &entryBlock = func.getBody().front();
    for (auto &opRef : entryBlock.getOperations()) {
      auto op = &opRef;
      if (!isGraphKernelOp(op)) {
        continue;
      }
      auto outputs = op->getResults();
      assert(outputs.size() > 0);

      if (isa<tosa::ConstOp>(op) || isa<mindspore::ConstOp>(op)) {
        nodeMap[outputs[0]] = gb.value(extractBuildInfo(outputs[0]));
      } else {
        auto opName = JsonFuncBuilder().opBuilderFactory(op)->getOpName();
        auto inputs = genInputNodes(op);
        auto opNode = gb.getOp(opName, extractBuildInfo(outputs[0]), inputs, op->getAttrDictionary());
        if (opNodeMap != nullptr) {
          (*opNodeMap)[opNode] = op;
        }
        for (auto output = outputs.begin(); output != outputs.end(); output++) {
          nodeMap[*output] = opNode;
        }
        // mutil-output op may adapted here
      }
    }
  };

  auto genOutputNodes = [&](func::FuncOp func) {
    NodePtrList nodeOutputs;
    func.walk([&](func::ReturnOp op) {
      for (mlir::Value opnd : op.getOperation()->getOperands()) {
        nodeOutputs.push_back(nodeMap[opnd]);
      }
    });
    gb.setOutputs(nodeOutputs);
  };
  // set inputs
  genParamNodes(func);
  // set ops
  genOpNodes(func);
  // set outputs
  genOutputNodes(func);
  return gb.get();
}
}  // namespace mlir::spliter
