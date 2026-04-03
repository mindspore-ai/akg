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

#include "mfusion/Analysis/Split/Node.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace mfuse {
namespace split {

// Node implementation
std::string Node::toString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  op_->print(os, mlir::OpPrintingFlags().assumeVerified().useLocalScope());
  return result;
}

void Node::addInput(Node *new_input) {
  if (!new_input) {
    return;
  }
  new_input->addUser(this, inputs_.size());
  inputs_.push_back(new_input);
}

void Node::setInput(size_t i, Node *new_input) {
  if (!new_input) {
    return;
  }
  if (i >= inputs_.size()) {
    std::string err_msg =
      "The index " + std::to_string(i) + " is out of the inputs range [0, " + std::to_string(inputs_.size()) + ")";
    llvm::report_fatal_error(llvm::StringRef(err_msg));
  }
  auto &old_input = inputs_[i];
  old_input->removeUser(this, i);
  new_input->addUser(this, i);
  inputs_[i] = new_input;
}

void Node::setInputs(const std::vector<Node *> &inputs) {
  clearInputs();
  inputs_.reserve(inputs.size());
  for (const auto &inp : inputs) {
    addInput(inp);
  }
}

void Node::clearInputs() noexcept {
  if (!inputs_.empty()) {
    for (size_t i = 0; i < inputs_.size(); i++) {
      inputs_[i]->removeUser(this, i);
    }
    inputs_.clear();
  }
}

void Node::replaceWith(Node *other_node) {
  if (!other_node || users_.empty()) {
    return;
  }
  auto users_copy = users_;
  for (const auto &[user, indices] : users_copy) {
    for (size_t idx : indices) {
      user->setInput(idx, other_node);
    }
  }
}

void Node::removeUser(Node *user, size_t index) {
  if (auto iter = users_.find(user); iter != users_.end()) {
    iter->second.erase(index);
    if (iter->second.empty()) {
      users_.erase(iter);
    }
  }
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
