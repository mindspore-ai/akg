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

#include "mfusion/Analysis/Split/Area.h"

#include <algorithm>
#include <numeric>
#include <unordered_set>
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Analysis/Split/OpRegister.h"
#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"
#include "mfusion/Dialect/Dvm/IR/Dvm.h"
#include "mfusion/Analysis/SymbolicShape/SymExprBuilder.h"
#include "mfusion/Analysis/SymbolicShape/SymEngineAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace mfuse {
namespace split {

namespace {

bool isDynamic(const Type &type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  // Only check ranked tensors with unknown dimensions
  return shapedType && shapedType.hasRank() && !shapedType.hasStaticShape();
}

bool symExprEqual(const SymExpr &a, const SymExpr &b) {
  static mfusion::SymEngineAnalysis analysis;
  return analysis.isStructurallyEqual(a, b);
}

bool shapeEqual(Node *a, Node *b, bool skip_leading_one = true) {
  if (!a || !b) {
    return false;
  }
  auto l = a->shape.size() < b->shape.size() ? b : a;
  auto s = a->shape.size() < b->shape.size() ? a : b;
  const auto &l_shape = l->shape;
  const auto &s_shape = s->shape;
  const auto &l_sym_shape = l->sym_shape;
  const auto &s_sym_shape = s->sym_shape;
  auto diff = l_shape.size() - s_shape.size();
  bool has_symshape = (l_sym_shape.size() == l_shape.size() && s_sym_shape.size() == s_shape.size());
  if (diff != 0 && !skip_leading_one) {
    // shapes with different rank
    return false;
  }
  // check leading one
  for (size_t i = 0; i < diff; ++i) {
    if (l_shape[i] != 1) {
      return false;
    }
  }
  // check other dimensions
  for (size_t i = 0; i < s_shape.size(); ++i) {
    auto il = i + diff;
    if (l_shape[il] < 0 || s_shape[i] < 0) {
      if (!has_symshape || !symExprEqual(l_sym_shape[il], s_sym_shape[i])) {
        return false;
      }
    } else if (l_shape[il] != s_shape[i]) {
      return false;
    }
  }
  return true;
}

EdgeRelation getRelation(Node *node, Node *input) {
  if (!node || !input || !node->op() || !input->op()) {
    llvm::report_fatal_error("op or input is nullptr");
  }
  auto op = node->op();
  // Get operation name and compute type
  std::string op_name = op->getName().getStringRef().str();
  NodePattern op_pattern = OpRegistry::Instance().GetPattern(op_name);

  if (op_pattern != NodePattern::ELEMWISE) {
    return op_pattern == NodePattern::BROADCAST ? EdgeRelation::BROADCAST : EdgeRelation::INJECTIVE;
  }

  if (op->getNumOperands() == 1) {
    return EdgeRelation::INJECTIVE;
  }

  auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(input->op()->getResult(0).getType());
  if (input_type && isDynamic(input_type)) {
    if (std::all_of(op->getOperands().begin(), op->getOperands().end(),
                    [input](Value operand) { return operand.getDefiningOp() == input->op(); })) {
      return EdgeRelation::INJECTIVE;
    }
  }

  // naively set the edge relation to "broadcast" if the result shape is not equal to the input shape.
  return shapeEqual(node, input) ? EdgeRelation::INJECTIVE : EdgeRelation::BROADCAST;
}

bool sameArea(const AreaWithRelation &a, const AreaWithRelation &b) { return a.first == b.first; }

bool areaWithRelationCmp(const AreaWithRelation &a, const AreaWithRelation &b) {
  return sameArea(a, b) ? (a.second > b.second) : (a.first->id() < b.first->id());
}

}  // namespace

Area::Area(size_t id, Node *node, bool is_output, const std::unordered_map<Node *, AreaPtr> &node_area_map)
    : unique_id_(id), is_output_(is_output), nodes_(1, node) {
  std::string op_name = node->op()->getName().getStringRef().str();
  NodePattern pattern = OpRegistry::Instance().GetPattern(op_name);
  // Initialize NodeHandle
  hd_ = std::make_shared<NodeHandle>(this, pattern);
  // link inputs of the handle node
  auto init_pattern = pattern;
  for (auto input : node->inputs()) {
    auto input_relation = getRelation(node, input);
    if (init_pattern == NodePattern::ELEMWISE && input_relation == EdgeRelation::BROADCAST) {
      hd_->setPattern(NodePattern::BROADCAST);
    }
    if (auto inp_area_iter = node_area_map.find(input); inp_area_iter != node_area_map.end()) {
      inputs_with_relation_.emplace_back(std::make_pair(inp_area_iter->second, input_relation));
    }
  }
  // ELEMWISE if op has one variable input, other inputs are const input with shape [1]
  // e.g. Cast(out_0, 43)
  //      Add(param0, const)
  if (hd_->pattern() == NodePattern::BROADCAST && init_pattern == NodePattern::ELEMWISE) {
    size_t scalar_input_num = 0;
    size_t input_num = node->inputNum();
    for (auto input : node->inputs()) {
      auto defType = mlir::dyn_cast<RankedTensorType>(input->op()->getResult(0).getType());
      if (defType && defType.getNumElements() == 1) {
        scalar_input_num++;
      }
    }
    if (scalar_input_num + 1 == input_num) {
      hd_->setPattern(NodePattern::ELEMWISE);
      if (!inputs_with_relation_.empty()) {
        inputs_with_relation_[0].second = EdgeRelation::INJECTIVE;
      }
    }
  }
  makeUniqueAndSyncInputs();
}

std::vector<AreaPtr> Area::inputs() const {
  std::vector<AreaPtr> result;
  result.reserve(inputs_with_relation_.size());
  std::transform(inputs_with_relation_.begin(), inputs_with_relation_.end(), std::back_inserter(result),
                 [](const auto &pair) { return pair.first; });
  return result;
}

std::vector<AreaPtr> Area::users() const {
  std::vector<AreaPtr> result;
  (void)std::transform(hd_->users().begin(), hd_->users().end(), std::back_inserter(result), [](const auto &user) {
    Node *node = user.first;
    return node->as<NodeHandle>()->area();
  });
  return result;
}

std::vector<AreaWithRelation> Area::usersWithRelation() const {
  std::vector<AreaWithRelation> result;
  (void)std::transform(hd_->users().begin(), hd_->users().end(), std::back_inserter(result), [](const auto &u) {
    Node *node = u.first;
    auto area = node->as<NodeHandle>()->area();
    // the input edge of area is unique
    const auto relation = area->inputRelation(*(u.second.begin()));
    return std::make_pair(area, relation);
  });
  return result;
}

int64_t Area::computeSize() const {
  auto op = dom()->op();
  auto op_type = op->getResult(0).getType();
  if (isDynamic(op_type)) {
    return 0;
  }
  auto op_shape = mlir::dyn_cast<RankedTensorType>(op_type).getShape();
  return std::accumulate(op_shape.begin(), op_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
}

bool Area::computeSizeEqual(const AreaPtr &other) const {
  if (!other || nodes_.empty() || other->nodes_.empty()) {
    return false;
  }
  auto op = dom();
  auto other_op = other->dom();
  auto op_type = op->op()->getResult(0).getType();
  auto other_op_type = other_op->op()->getResult(0).getType();
  if (op_type && other_op_type && !isDynamic(op_type) && !isDynamic(other_op_type)) {
    return computeSize() == other->computeSize();
  }
  return shapeEqual(op, other_op);
}

std::string Area::toString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "Area " << id() << " (" << nodes_.size() << " nodes):";
  for (auto node : nodes_) {
    os << "\n  " << node->op()->getName().getStringRef().data();
  }
  return os.str();
}

void Area::fuseInput(const AreaPtr &input_area) {
  auto iter = std::find_if(inputs_with_relation_.begin(), inputs_with_relation_.end(),
                           [&input_area](const AreaWithRelation &a) { return a.first == input_area; });
  if (iter == inputs_with_relation_.end()) {
    std::string err_msg =
      "The area " + input_area->toString() + " should be the input of area " + this->toString() + "\n";
    llvm::report_fatal_error(llvm::StringRef(err_msg));
  }
  auto input_idx = iter - inputs_with_relation_.begin();

  if (input_area->is_output_) {
    is_output_ = true;
  }

  // Update ops, and discard the input_area's ops.
  // The dominant node is ops[0], keep the dominant with greater pattern.
  if (pattern() < input_area->pattern()) {
    nodes_.swap(input_area->nodes_);
  }
  (void)nodes_.insert(nodes_.cend(), input_area->nodes_.cbegin(), input_area->nodes_.cend());

  // update area pattern
  NodePattern new_pattern = std::max(pattern(), input_area->pattern());
  if ((new_pattern == NodePattern::ELEMWISE) && (inputRelation(input_idx) == EdgeRelation::BROADCAST)) {
    new_pattern = NodePattern::BROADCAST;
  }
  hd_->setPattern(new_pattern);

  inputs_with_relation_.erase(iter);
  inputs_with_relation_.insert(inputs_with_relation_.cend(), input_area->inputs_with_relation_.cbegin(),
                               input_area->inputs_with_relation_.cend());
  makeUniqueAndSyncInputs();
  updateUsersRelation(input_area);

  // clear the input_area.
  input_area->nodes_.clear();
  input_area->inputs_with_relation_.clear();
  input_area->hd_->clearInputs();
}

void Area::makeUniqueAndSyncInputs() {
  // remove the repeated inputs, keep the area with greater EdgeRelation.
  std::sort(inputs_with_relation_.begin(), inputs_with_relation_.end(), areaWithRelationCmp);
  auto last = std::unique(inputs_with_relation_.begin(), inputs_with_relation_.end(), sameArea);
  inputs_with_relation_.erase(last, inputs_with_relation_.cend());
  this->hd_->clearInputs();
  std::for_each(inputs_with_relation_.begin(), inputs_with_relation_.end(),
                [this](const AreaWithRelation &pair) { this->hd_->addInput(pair.first->hd_.get()); });
}

void Area::updateUsersRelation(const AreaPtr &input_area) {
  auto &user_node_with_index = input_area->hd_->users();
  std::vector<AreaPtr> user_areas;
  for (auto &[user_hd, index] : user_node_with_index) {
    user_areas.emplace_back(user_hd->as<NodeHandle>()->area());
    const auto idx = *(index.begin());
    user_areas.back()->inputs_with_relation_[idx].first = this->shared_from_this();
  }
  // the inputs should be updated outside the above for-loop,
  // since the users cannot be updated while traversing.
  for (auto user : user_areas) {
    user->makeUniqueAndSyncInputs();
  }
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
