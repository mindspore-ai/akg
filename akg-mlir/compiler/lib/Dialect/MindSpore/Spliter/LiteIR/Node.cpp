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
#include "akg/Dialect/MindSpore/Spliter/Node.h"

#include <algorithm>
#include <sstream>
#include <utility>
#include "akg/Analysis/SymbolicShapeAnalysis.h"

namespace mlir::spliter {
void Node::setBaseInfo(const NodeBaseList &baseInfo) {
  this->shape = baseInfo[0].shape;
  this->type = baseInfo[0].type;
  this->format = baseInfo[0].format;
  this->symbolicShape = baseInfo[0].symbolicShape;
  if (baseInfo.size() > 1) {
    outputs = baseInfo;
  }
}

std::string Node::toString() const {
  std::ostringstream oss;
  oss << getDebugName() << "[";
  if (hasSymbolicShape()) {
    for (size_t i = 0; i < symbolicShape.size(); i++) {
      oss << symbolicShape[i];
      if (i + 1 < symbolicShape.size()) {
        oss << ",";
      }
    }
  } else {
    for (size_t i = 0; i < shape.size(); i++) {
      oss << shape[i];
      if (i + 1 < shape.size()) {
        oss << ",";
      }
    }
  }
  oss << "]{" << typeIdToString(type) << "x" << format << "}";
  return oss.str();
}

void Node::addInput(const NodePtr &newInput) {
  assert(newInput != nullptr);
  newInput->addUser(this, inputs.size());
  (void)inputs.emplace_back(newInput);
}

void Node::setInput(size_t i, const NodePtr &newInput) {
  assert(newInput != nullptr);
  if (i >= inputs.size()) {
    llvm::errs() << "The index " << i << " is out of the inputs range [0, " << inputs.size() << ")";
  }
  auto &oldInput = inputs[i];
  oldInput->removeUser(this, i);
  newInput->addUser(this, i);
  inputs[i] = newInput;
}

void Node::setInputs(const NodePtrList &newInputs) {
  clearInputs();
  inputs.reserve(newInputs.size());
  for (const auto &inp : newInputs) {
    addInput(inp);
  }
}

void Node::clearInputs() noexcept {
  if (!inputs.empty()) {
    // remove the original inputs
    for (size_t i = 0; i < inputs.size(); i++) {
      inputs[i]->removeUser(this, i);
    }
    inputs.clear();
  }
}

void Node::replaceWith(const NodePtr &otherNode) {
  if (this->users.empty()) {
    return;
  }
  // the users will be changed, so we copy the users before traversal
  auto newUsers = this->users;
  for (auto &user : newUsers) {
    for (const auto &idx : user.second) {
      user.first->setInput(idx, otherNode);
    }
  }
}

void Node::removeUser(Node *const user, size_t index) {
  if (auto iter = users.find(user); iter != users.end()) {
    (void)iter->second.erase(index);
    if (iter->second.empty()) {
      (void)users.erase(iter);
    }
  }
}

size_t Node::tensorSize(bool inBytes) const {
  size_t size = longToSize(shapeSize(this->shape));
  return inBytes ? typeIdSize(this->type) * size : size;
}

SymEngine::Expression Node::dynTensorSize(bool inBytes) const {
  auto symExprs = mlir::SymbolicShapeAnalysis::getInstance().getSymbolicExprsFromStrs(this->symbolicShape);
  SymEngine::Expression size(1);
  (void)std::accumulate(symExprs.begin(), symExprs.end(), size, std::multiplies<SymEngine::Expression>());
  return inBytes ? typeIdSize(this->type) * size : size;
}

bool Node::hasSameShape(const NodePtr &other) const {
  if (this->hasSymbolicShape() && other->hasSymbolicShape()) {
    auto thisExprs = mlir::SymbolicShapeAnalysis::getInstance().getSymbolicExprsFromStrs(this->symbolicShape);
    auto otherExprs = mlir::SymbolicShapeAnalysis::getInstance().getSymbolicExprsFromStrs(other->symbolicShape);
    return std::equal(thisExprs.begin(), thisExprs.end(), otherExprs.begin(), otherExprs.end());
  }
  return this->shape == other->shape;
}

bool Node::hasSameTensorSize(const NodePtr &other) const {
  if (this->hasSymbolicShape() && other->hasSymbolicShape()) {
    return this->dynTensorSize() == other->dynTensorSize();
  }
  return this->tensorSize() == other->tensorSize();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const NType &x) { return os << static_cast<int>(x); }

}  // namespace mlir::spliter
